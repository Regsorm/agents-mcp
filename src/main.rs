//! agents-mcp — MCP-сервер платформы специализированных LLM-агентов.
//!
//! Phase 1 — каркас + storage + registry + runtime с mock-провайдером.
//! Реальные провайдеры (OpenRouter, Anthropic, claude-cli) — Дни 3-4.

use std::future::Future;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::Parser;
use tracing::{error, info, warn};

mod cache;
mod config;
mod errors;
mod health;
mod log_layer;
mod store;
mod pid_lock;
mod proc_tree;
mod providers;
mod registry;
mod reload;
mod skills;
mod runtime;
mod server;
mod storage;
mod watcher;

use crate::store::Store;

#[derive(Parser, Debug)]
#[command(
    name = "agents-mcp",
    version,
    about = "MCP-сервер платформы специализированных LLM-агентов"
)]
struct Cli {
    /// Путь к agents-mcp.toml. Если не указан, ищется в env AGENTS_MCP_CONFIG;
    /// при отсутствии — используются дефолты.
    #[arg(short = 'c', long = "config", value_name = "PATH")]
    config: Option<PathBuf>,

    /// Способ связи с клиентом:
    ///   http  — общая служба на порту из конфига (по умолчанию, так её
    ///           запускает супервизор); подключаются все клиенты сразу;
    ///   stdio — клиент сам запускает этот процесс и говорит с ним через его
    ///           стандартный ввод/вывод; порт не занимается, экземпляр свой
    ///           у каждого клиента.
    #[arg(short = 't', long = "transport", value_name = "http|stdio", default_value = "http")]
    transport: String,
}

/// Причина в поле `error` осиротевших вызовов и в их файлах-итогах.
const ORPHAN_REASON: &str = "служба перезапущена во время вызова";
/// Причина штатного закрытия фоновых вызовов и срок их финализации.
const SHUTDOWN_REASON: &str = "служба остановлена";
const SHUTDOWN_FINALIZE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
const LOG_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
/// Срок хранения строк `agent_calls` и соответствующих им файлов-итогов.
const AGENT_CALLS_RETENTION_DAYS: i64 = 90;

/// Выполнить стартовое действие только после успешного bind. Отдельная функция
/// закрепляет порядок: занятый порт не должен менять строки живой службы.
async fn bind_then<F, Fut>(
    addr: SocketAddr,
    after_bind: F,
) -> std::io::Result<tokio::net::TcpListener>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = ()>,
{
    let listener = tokio::net::TcpListener::bind(addr).await?;
    after_bind().await;
    Ok(listener)
}

/// Закрыть оставшиеся от прежнего процесса running-вызовы этого экземпляра.
async fn mark_orphaned_at_start(store: &Arc<dyn Store>, runs_dir: &Path, instance: &str) {
    let service_started_at = chrono::Utc::now().timestamp();
    match store
        .mark_orphaned_running(service_started_at, ORPHAN_REASON, instance)
        .await
    {
        Ok(orphans) if !orphans.is_empty() => {
            let agents = store::summarize_orphans(&orphans);
            tracing::warn!(
                count = orphans.len(),
                "осиротевшие вызовы помечены ошибкой: {} — {}",
                orphans.len(),
                agents
            );
            // Файл-итог нужен только свежим вызовам: у начатых раньше часа
            // ждущего давно нет, а каталог засорять незачем.
            let fresh_cutoff = service_started_at - 3_600;
            for call in orphans.iter().filter(|c| c.created_at >= fresh_cutoff) {
                let path = runtime::orphan_result_path(runs_dir, call);
                let envelope = runtime::envelope_error(call.id, &call.agent_name, ORPHAN_REASON);
                if let Err(e) = runtime::write_result_file(&path, &envelope) {
                    tracing::warn!(
                        call_id = call.id,
                        path = %path.display(),
                        error = %e,
                        "файл-итог осиротевшего вызова не записан"
                    );
                }
            }
            health::set_orphaned_at_start(health::OrphanedAtStart {
                count: orphans.len(),
                agents,
            });
        }
        Ok(_) => {}
        Err(e) => tracing::warn!(error = %e, "осиротевшие вызовы пометить не удалось"),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Загрузить .env ДО чтения конфига и регистрации провайдеров: секреты
    // (AGENTS_MCP_TASK_STORE_DSN, *_API_KEY) приходят отсюда. Причина — Task
    // Scheduler не передаёт супервизору User-env, добавленные после логина, и
    // унаследовать их дочерний agents-mcp не может. .env рядом с cwd — обходит это.
    // Нет файла — не ошибка (env может прийти и штатным путём). Ошибки пока
    // сохраняем: журнал будет готов только ниже.
    let (provider_env, startup_dotenv_errors) = reload::load_startup_dotenv();
    let cli = Cli::parse();
    let stdio_mode = match cli.transport.as_str() {
        "http" => false,
        "stdio" => true,
        other => {
            eprintln!("ERROR: транспорт '{other}' не поддерживается, нужен 'http' или 'stdio'");
            std::process::exit(2);
        }
    };
    let cfg = config::Config::load_or_default(cli.config.as_deref())
        .map_err(|e| anyhow::Error::msg(errors::safe_config_error(&e)))?;
    // Экземпляр службы: [server] instance, а если он не задан — «имя машины:порт».
    // Пишется в строки вызовов и по нему же при старте закрываются только свои
    // осиротевшие вызовы — у двух служб на одной базе он обязан различаться.
    let instance = cfg.server.instance_name(stdio_mode);

    // Singleton до bind: второй экземпляр не должен запускать пул, registry и т.п.
    // Сообщение ошибки уходит в stderr через eprintln (tracing ещё не настроен).
    //
    // На транспорте stdio замок не берём: экземпляр там свой у каждого клиента
    // и живёт ровно столько, сколько открыт его стандартный ввод. Порт он не
    // занимает, поэтому делить с общей службой ему нечего — а общий замок
    // запретил бы запуск, пока служба работает.
    let _pid_lock = if stdio_mode {
        None
    } else {
        match pid_lock::PidLock::acquire(&cfg.storage.log_dir) {
            Ok(lock) => Some(lock),
            Err(e) => {
                eprintln!("ERROR: {e}");
                return Err(e);
            }
        }
    };

    // Единое хранилище: по умолчанию встроенный SQLite (файл базы создаётся сам
    // при первом запуске), PostgreSQL — по адресу из [storage].task_store_dsn.
    // Типаж-объект отдаётся всему, что ходит в БД и не знает про её реализацию
    // (retention, стартовая пометка осиротевших вызовов); описание
    // хранилища возвращается наружу — записать его в журнал можно только ниже,
    // после init_tracing.
    let (store, storage_desc): (Arc<dyn Store>, String) = store::connect(&cfg.storage)?;
    // Журнал службы — отдельный файл SQLite рядом с PID-замком: не зависит от
    // основного хранилища, поэтому отказ основной базы записи журнала не теряет.
    let log_db_path = cfg.storage.log_dir.join("agents-mcp.logs.db");
    let log_store: Arc<dyn Store> = Arc::new(store::SqliteStore::open(&log_db_path)?);

    // Tracing: stdout (docker logs / stderr.log обвязки) + LogLayer
    // (журнал службы в agents-mcp.logs.db). На транспорте stdio вывод уходит в
    // stderr: stdout там занят самим протоколом, и любая строка журнала в нём
    // ломает разбор сообщений у клиента.
    let (log_layer, log_writer_handle) = log_layer::LogLayer::install(log_store.clone());
    init_tracing(log_layer, stdio_mode);
    cfg.warn_unknown_keys();

    // Отдельный блок П19в: `.env` читается до создания журнала, поэтому
    // безопасные сообщения об ошибках записываем сразу после init_tracing.
    for dotenv_error in startup_dotenv_errors {
        error!("файл .env: {dotenv_error}");
    }

    info!(
        version = env!("CARGO_PKG_VERSION"),
        host = %cfg.server.host,
        port = cfg.server.port,
        agents_dir = %cfg.agents.agents_dir.display(),
        log_dir = %cfg.storage.log_dir.display(),
        log_db = %log_db_path.display(),
        storage = %storage_desc,
        instance = %instance,
        "agents-mcp starting"
    );

    // HTTP сначала занимает порт и только потом меняет строки в общей базе:
    // второй процесс с занятым портом не имеет права гасить живые вызовы.
    // На stdio пометки при старте нет: имя экземпляра «машина:stdio» общее у
    // всех одновременно запущенных stdio-процессов, и новый погасил бы живые
    // вызовы соседнего. Свои фоновые вызовы stdio закрывает при штатной остановке.
    let addr: SocketAddr = SocketAddr::new(cfg.server.host, cfg.server.port);
    let listener = if stdio_mode {
        None
    } else {
        Some(
            bind_then(addr, || {
                mark_orphaned_at_start(&store, &cfg.storage.runs_dir, &instance)
            })
            .await?,
        )
    };

    // Реестр агентов.
    let registry = Arc::new(registry::Registry::load(cfg.agents.agents_dir.clone())?);

    // Провайдеры собирает reload.rs: тот же код нужен перечитке главного конфига
    // (смена [providers.*] без перезапуска службы).
    let (provider_set, _provider_changes) =
        reload::build_providers(&cfg.providers, None, &provider_env).await;
    let skills_client = skills::SkillsClient::new(cfg.skills.rag_query_url.clone());

    // Тест-override модели за RwLock — Runtime читает, watcher главного конфига
    // перечитывает на лету (смена force_provider/force_model без рестарта).
    let force_override: runtime::SharedOverride =
        Arc::new(std::sync::RwLock::new(runtime::ModelOverride {
            provider: cfg.agents.force_provider.clone(),
            model: cfg.agents.force_model.clone(),
        }));

    // Путь главного конфига — тот же fallback, что в Config::load_or_default
    // (--config → env). Нужен и перечитке конфига, и наблюдателю файла.
    let config_path = cli
        .config
        .clone()
        .or_else(|| std::env::var_os("AGENTS_MCP_CONFIG").map(PathBuf::from));

    let runtime_inner = Arc::new(runtime::Runtime::new(
        store.clone(),
        registry.clone(),
        provider_set.providers.clone(),
        skills_client,
        force_override.clone(),
        cfg.storage.runs_dir.clone(),
        instance.clone(),
        cfg.agents.default_timeout_sec,
    ));
    runtime_inner.set_provider_env(provider_env.clone());

    // Перечитка главного конфига без перезапуска: применяет изменившееся и по
    // инструменту config_reload, и по сохранению файла конфига. Наблюдатель
    // каталога агентов живёт внутри неё (его надо держать живым на весь срок
    // службы — потому и заводится здесь, а не в main).
    let reloader = reload::ConfigReloader::new(
        config_path.clone(),
        cfg.clone(),
        provider_set,
        provider_env,
        registry.clone(),
        runtime_inner.clone(),
        force_override.clone(),
    );
    reloader.start_agents_watcher(cfg.agents.agents_dir.clone(), cfg.agents.hot_reload);

    // Итоги и строки вызовов живут один срок: иначе каталог runs_dir снова
    // вырастет бесконечно, даже если agent_calls уже очищен.
    let removed_result_files = runtime::cleanup_result_files(
        &cfg.storage.runs_dir,
        std::time::Duration::from_secs(AGENT_CALLS_RETENTION_DAYS as u64 * 24 * 3600),
    );
    if removed_result_files > 0 {
        info!(removed_result_files, "старые файлы-итоги очищены");
    }

    // Retention в фоне: events 30 дней, agent_calls 90 дней.
    let _retention_handle = tokio::spawn(storage::retention_loop(
        store.clone(),
        AGENT_CALLS_RETENTION_DAYS,
        30,
    ));
    // Журнал службы живёт в отдельном файле, поэтому и чистится своим циклом:
    // журнал хранится 30 дней.
    let _log_retention_handle = tokio::spawn(storage::retention_loop(log_store.clone(), 90, 30));

    // Watcher главного конфига: при сохранении файла запускает перечитку —
    // журнал и применение целиком её дело. Путь конфига — тот же fallback, что
    // и у перечитки (--config → env).
    let _config_watcher = match (&config_path, cfg.agents.hot_reload) {
        (Some(path), true) => match watcher::spawn_config(path.clone(), reloader.clone()) {
            Ok(w) => Some(w),
            Err(e) => {
                tracing::warn!(error = %e, "не удалось запустить watcher главного конфига");
                None
            }
        },
        _ => None,
    };

    let started_at = chrono::Utc::now();

    // Транспорт стандартного ввода/вывода: клиент сам запускает этот процесс и
    // говорит с ним через его stdin/stdout. HTTP не поднимается вовсе — порт
    // не занимается, и общая служба на том же порту работе не мешает.
    // Набор инструментов тот же, что по HTTP.
    if stdio_mode {
        use rmcp::ServiceExt;
        let mcp = server::build_mcp_server(
            started_at,
            registry,
            runtime_inner.clone(),
            reloader.clone(),
        );
        info!("транспорт stdio: жду хендшейк на стандартном вводе");
        let service = mcp
            .serve(rmcp::transport::io::stdio())
            .await
            .map_err(|e| anyhow::anyhow!("MCP serve (stdio): {e}"))?;
        let waited = service.waiting().await;
        runtime_inner
            .finalize_background_calls(SHUTDOWN_REASON, SHUTDOWN_FINALIZE_TIMEOUT)
            .await;
        let result = waited
            .map(|_| ())
            .map_err(|e| anyhow::anyhow!("MCP wait (stdio): {e}"));
        match &result {
            Ok(()) => info!("stdio-сессия завершена"),
            Err(e) => error!(error = %e, "stdio-сессия завершена с ошибкой"),
        }
        if !log_writer_handle.shutdown(LOG_SHUTDOWN_TIMEOUT).await {
            eprintln!("event-writer: журнал stdio-сессии слит не полностью");
        }
        return result;
    }

    let app = server::build_router(
        cfg.clone(),
        started_at,
        registry,
        runtime_inner.clone(),
        reloader,
    );
    let listener = listener.expect("HTTP listener создан до инициализации runtime");

    // Windows: снять флаг наследования со слушающего сокета. Дочерние claude.exe
    // спавнятся с bInheritHandles=TRUE (ради piped stdio) и иначе наследуют этот
    // сокет; после смерти родителя осиротевший ребёнок продолжает держать порт,
    // и новый экземпляр не может забиндиться → краш-петля 10048 при рестарте
    // во время активного прогона (инцидент 2026-06-18).
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawSocket;
        const HANDLE_FLAG_INHERIT: u32 = 0x1;
        extern "system" {
            fn SetHandleInformation(h: *mut core::ffi::c_void, mask: u32, flags: u32) -> i32;
        }
        let h = listener.as_raw_socket() as usize as *mut core::ffi::c_void;
        let ok = unsafe { SetHandleInformation(h, HANDLE_FLAG_INHERIT, 0) };
        if ok == 0 {
            tracing::warn!("не удалось снять HANDLE_FLAG_INHERIT со слушающего сокета");
        }
    }

    info!(%addr, "listening");

    let serve_result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await;
    runtime_inner
        .finalize_background_calls(SHUTDOWN_REASON, SHUTDOWN_FINALIZE_TIMEOUT)
        .await;

    drop(_pid_lock); // подскажет stale-cleanup даже при ошибке axum
    let result = match serve_result {
        Ok(_) => {
            info!("graceful shutdown complete");
            Ok(())
        }
        Err(e) => {
            error!(error = %e, "server stopped with error");
            Err(e.into())
        }
    };
    if !log_writer_handle.shutdown(LOG_SHUTDOWN_TIMEOUT).await {
        eprintln!("event-writer: журнал HTTP-сессии слит не полностью");
    }
    result
}

/// `to_stderr = true` — вывод журнала в stderr вместо stdout. Обязателен для
/// транспорта stdio: stdout там принадлежит протоколу MCP целиком.
fn init_tracing(log_layer: log_layer::LogLayer, to_stderr: bool) {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    // RUST_LOG задаёт общий уровень; шумные библиотеки при этом остаются на
    // warn, если RUST_LOG не называет их явно.
    let rust_log = std::env::var("RUST_LOG").ok();
    let env_filter = match EnvFilter::try_new(filter_spec(rust_log.as_deref())) {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "warning: не удалось разобрать RUST_LOG ({e}), беру фильтр по умолчанию"
            );
            EnvFilter::new(filter_spec(None))
        }
    };

    let started = if to_stderr {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().with_writer(std::io::stderr))
            .with(log_layer)
            .try_init()
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().with_writer(std::io::stdout))
            .with(log_layer)
            .try_init()
    };

    if started.is_err() {
        eprintln!("warning: tracing already initialized");
    }
}

/// Шумные библиотеки: на info они пишут служебные строки на каждый запрос.
const QUIET_TARGETS: &[&str] = &["hyper", "h2", "axum", "tower_http", "tower", "rmcp", "tokio_util"];

/// Выражение фильтра журнала. RUST_LOG задаёт общий уровень, но шумные библиотеки
/// остаются на warn, если RUST_LOG не называет их явно (например `rmcp=debug`
/// для отладки). Без этого RUST_LOG=info от супервизора целиком заменял фильтр.
fn filter_spec(rust_log: Option<&str>) -> String {
    let base = rust_log
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("info");
    let mut spec = base.to_string();
    for target in QUIET_TARGETS {
        if !base.contains(&format!("{target}=")) {
            spec.push_str(&format!(",{target}=warn"));
        }
    }
    spec
}

/// Graceful shutdown по сигналам платформы.
///
/// На Windows под Scheduled Task `schtasks /End` шлёт `CTRL_CLOSE_EVENT`
/// (закрытие консоли), а не `Ctrl+C`. Чтобы корректно подхватывать остановку
/// из планировщика, обрабатываем все четыре варианта:
///   - Ctrl+C (интерактивный stop через консоль)
///   - Ctrl+Break (некоторые supervisor'ы шлют именно его)
///   - Ctrl+Close (closing the console window — это шлёт schtasks /End)
///   - Ctrl+Shutdown (выключение Windows)
/// На Unix — Ctrl+C и SIGTERM.
async fn shutdown_signal() {
    #[cfg(windows)]
    {
        use tokio::signal::windows::{ctrl_break, ctrl_c, ctrl_close, ctrl_shutdown};
        let c = async {
            match ctrl_c() {
                Ok(mut signal) => { signal.recv().await; }
                Err(e) => {
                    warn!(error = %e, "не удалось установить обработчик Ctrl+C");
                    std::future::pending::<()>().await;
                }
            }
        };
        let b = async {
            match ctrl_break() {
                Ok(mut signal) => { signal.recv().await; }
                Err(e) => {
                    warn!(error = %e, "не удалось установить обработчик Ctrl+Break");
                    std::future::pending::<()>().await;
                }
            }
        };
        let cl = async {
            match ctrl_close() {
                Ok(mut signal) => { signal.recv().await; }
                Err(e) => {
                    warn!(error = %e, "не удалось установить обработчик Ctrl+Close");
                    std::future::pending::<()>().await;
                }
            }
        };
        let sh = async {
            match ctrl_shutdown() {
                Ok(mut signal) => { signal.recv().await; }
                Err(e) => {
                    warn!(error = %e, "не удалось установить обработчик Ctrl+Shutdown");
                    std::future::pending::<()>().await;
                }
            }
        };
        tokio::select! {
            _ = c => info!("получен Ctrl+C"),
            _ = b => info!("получен Ctrl+Break"),
            _ = cl => info!("получен Ctrl+Close (закрытие консоли / schtasks /End)"),
            _ = sh => info!("получен Ctrl+Shutdown"),
        }
    }
    #[cfg(unix)]
    {
        let ctrl_c = async {
            if let Err(e) = tokio::signal::ctrl_c().await {
                warn!(error = %e, "не удалось установить обработчик Ctrl+C");
                std::future::pending::<()>().await;
            }
        };
        let terminate = async {
            use tokio::signal::unix::{signal, SignalKind};
            match signal(SignalKind::terminate()) {
                Ok(mut signal) => { signal.recv().await; }
                Err(e) => {
                    warn!(error = %e, "не удалось установить обработчик SIGTERM");
                    std::future::pending::<()>().await;
                }
            }
        };
        tokio::select! {
            _ = ctrl_c => info!("получен Ctrl+C"),
            _ = terminate => info!("получен SIGTERM"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{bind_then, filter_spec};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use tracing_subscriber::EnvFilter;

    const DEFAULT_SPEC: &str =
        "info,hyper=warn,h2=warn,axum=warn,tower_http=warn,tower=warn,rmcp=warn,tokio_util=warn";

    #[test]
    fn filter_spec_quiet_by_default() {
        for spec in [
            filter_spec(None),
            filter_spec(Some("info")),
            filter_spec(Some("  ")),
        ] {
            assert_eq!(spec, DEFAULT_SPEC);
            assert!(EnvFilter::try_new(spec.as_str()).is_ok(), "фильтр не разобран: {spec}");
        }
    }

    #[test]
    fn filter_spec_keeps_explicitly_named_targets() {
        let spec = filter_spec(Some("debug,rmcp=debug"));
        assert!(spec.starts_with("debug,rmcp=debug"), "получился {spec}");
        assert!(spec.contains("hyper=warn"), "получился {spec}");
        assert!(!spec.contains("rmcp=warn"), "получился {spec}");
        assert!(EnvFilter::try_new(spec.as_str()).is_ok(), "фильтр не разобран: {spec}");
    }

    #[tokio::test]
    async fn failed_bind_does_not_run_startup_action() {
        let occupied = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("занять тестовый порт");
        let addr = occupied.local_addr().expect("адрес занятого порта");
        let called = Arc::new(AtomicBool::new(false));
        let called_in_action = called.clone();

        let result = bind_then(addr, move || async move {
            called_in_action.store(true, Ordering::SeqCst);
        })
        .await;

        assert!(result.is_err(), "повторный bind занятого порта обязан упасть");
        assert!(
            !called.load(Ordering::SeqCst),
            "стартовая пометка сирот не должна запускаться до успешного bind"
        );
    }
}
