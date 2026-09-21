//! Единый вход в хранилище службы: типаж [`Store`] и общие для реализаций типы.
//!
//! Весь SQL службы (доска задач, журнал вызовов, ходы прогона, кеш ответов,
//! лог событий) живёт за этим типажом: рантайм, кеш и слой журнала обращаются
//! к БД только именованными методами и не знают ни диалекта, ни драйвера.
//! Реализаций две: PostgreSQL ([`pg::PgStore`]) — для тех, кто указал адрес
//! внешней базы, и встроенный SQLite ([`sqlite::SqliteStore`]) — по умолчанию,
//! без внешней базы: файл создаётся сам при первом запуске.

mod pg;
mod sqlite;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

pub use pg::PgStore;
pub use sqlite::SqliteStore;

/// Параметры создания корневой задачи.
#[derive(Debug, Default)]
pub struct NewTask {
    pub root_call_id: Option<i64>,
    pub external_task_id: Option<String>,
    pub task_kind: Option<String>,
    pub goal: Option<String>,
    pub target_base: Option<String>,
    pub working_dir: Option<String>,
    pub sandbox_path: Option<String>,
}

/// Артефакт доски задачи (вход/выход шага, срез контекста).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Artifact {
    pub id: i64,
    pub kind: String,
    pub key: String,
    pub content: Option<String>,
    pub summary: Option<String>,
    pub producer_agent: Option<String>,
    pub producer_call_id: Option<i64>,
    pub depends_on: Vec<i64>,
    pub status: String,
}

/// Осиротевший вызов: строка `agent_calls` со статусом `running`, начатая
/// раньше текущего запуска службы. Супервизор гасит службу жёстко
/// (TerminateProcess), поэтому такой вызов идти уже не может — владевший им
/// процесс мёртв, а статус никогда не сменится сам.
#[derive(Debug, Clone)]
pub struct OrphanedCall {
    pub id: i64,
    pub agent_name: String,
    pub created_at: i64,
    /// Сохранённый путь файла-итога `agent_run`; None у вызовов старой сборки.
    pub result_path: Option<String>,
}

/// Строка истории вызовов (`agent_calls`) для инструмента `agent_history`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HistoryEntry {
    pub id: i64,
    pub agent_name: String,
    pub variant: String,
    pub model_used: String,
    pub provider: String,
    pub tokens_in: Option<u32>,
    pub tokens_out: Option<u32>,
    pub cost_usd: Option<f64>,
    pub latency_ms: Option<u64>,
    pub cached: bool,
    pub created_at: i64,
    pub error: Option<String>,
    /// ID родительского вызова — для построения дерева оркестрации.
    pub parent_call_id: Option<i64>,
    /// Экземпляр службы, создавший вызов (имя машины и порт). None — вызов
    /// старой сборки, которая колонку ещё не писала.
    pub instance: Option<String>,
}

/// Запись кеша ответов агента: сам ответ + metadata вызова (model, tokens, cost).
#[derive(Debug, Clone)]
pub struct CachedEntry {
    pub output_json: String,
    pub metadata_json: String,
}

/// Строка `agent_calls` со всем, что нужно для сборки итога вызова
/// (long-poll `poll_call` и мгновенная проверка): статус, результат, метрики.
/// `status` — 'running' | 'done' | 'incomplete' | 'error' | 'cancelled' |
/// 'persistence_failed'. Провал остаётся 'error': по нему ждут итог внешние
/// клиенты и проверочные скрипты.
#[derive(Debug, Clone)]
pub struct CallRow {
    pub status: String,
    pub output_json: Option<String>,
    pub error: Option<String>,
    pub agent_name: String,
    pub variant: String,
    pub model_used: String,
    pub provider: String,
    pub tokens_in: Option<i64>,
    pub tokens_out: Option<i64>,
    pub cost_usd: Option<f64>,
    pub latency_ms: Option<i64>,
    pub cached: bool,
    pub parent_call_id: Option<i64>,
}

/// Финальный статус вызова. `running` сюда намеренно не входит: этот тип
/// используется только при завершении уже созданной строки `agent_calls`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallStatus {
    Done,
    Incomplete,
    Failed,
    Cancelled,
    PersistenceFailed,
}

impl CallStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Done => "done",
            Self::Incomplete => "incomplete",
            Self::Failed => "error",
            Self::Cancelled => "cancelled",
            Self::PersistenceFailed => "persistence_failed",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("NotFound: задача {0} не найдена")]
    TaskNotFound(i64),
    #[error("недопустимый статус задачи '{0}'")]
    InvalidTaskStatus(String),
    #[error("недопустимый переход статуса задачи: {from} -> {to}")]
    InvalidTaskStatusTransition { from: String, to: String },
}

pub(crate) fn validate_task_status(status: &str) -> Result<()> {
    match status {
        "running" | "needs_input" | "completed" | "failed" | "cancelled" => Ok(()),
        other => Err(StoreError::InvalidTaskStatus(other.to_string()).into()),
    }
}

pub(crate) fn validate_task_status_transition(current: &str, next: &str) -> Result<()> {
    validate_task_status(next)?;
    // `cancelled` — закрытый статус: цепочку остановили вручную (chain_cancel),
    // и обратно в `running` её уже не пускаем.
    let allowed = current == next
        || matches!(
            (current, next),
            ("queued", "running")
                | ("queued", "failed")
                | ("queued", "cancelled")
                | ("running", "needs_input")
                | ("running", "completed")
                | ("running", "failed")
                | ("running", "cancelled")
                | ("needs_input", "running")
                | ("needs_input", "completed")
                | ("needs_input", "failed")
                | ("needs_input", "cancelled")
                | ("completed", "running")
                | ("failed", "running")
        );
    if allowed {
        Ok(())
    } else {
        Err(StoreError::InvalidTaskStatusTransition {
            from: current.to_string(),
            to: next.to_string(),
        }
        .into())
    }
}

/// Событие журнала службы, готовое к записи (`agents_mcp.events`). Поля
/// события уже сериализованы в JSON и сжаты zstd — этим занимается слой
/// журнала, хранилищу остаётся только INSERT.
pub struct LogEvent {
    pub ts_millis: i64,
    pub level: i16,
    pub target: String,
    pub message: String,
    /// JSON полей события, сжатый zstd; None — полей не было.
    pub context_zstd: Option<Vec<u8>>,
}

/// Разбивка осиротевших вызовов по агентам для записи в журнал:
/// «<имя>×<число>» через ", ", по убыванию количества, при равенстве — по
/// алфавиту. Пустой список → пустая строка.
pub fn summarize_orphans(calls: &[OrphanedCall]) -> String {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for call in calls {
        *counts.entry(call.agent_name.as_str()).or_insert(0) += 1;
    }
    let mut items: Vec<(&str, usize)> = counts.into_iter().collect();
    items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    items
        .into_iter()
        .map(|(name, n)| format!("{name}×{n}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Адрес хранилища из конфига — какая реализация им выбрана.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Backend {
    /// Встроенный SQLite: файл базы (создаётся сам при необходимости).
    Sqlite(PathBuf),
    /// PostgreSQL: адрес подключения (DSN).
    Postgres(String),
}

impl Backend {
    /// Описание хранилища для журнала — без адреса и секретов: пароль из DSN
    /// в лог не попадает.
    pub fn describe(&self) -> String {
        match self {
            Backend::Sqlite(path) => format!("SQLite {}", path.display()),
            Backend::Postgres(dsn) => {
                format!("PostgreSQL sslmode={}", postgres_sslmode(dsn))
            }
        }
    }
}

fn postgres_sslmode(dsn: &str) -> &'static str {
    let Ok(config) = dsn.parse::<tokio_postgres::Config>() else {
        return "unknown";
    };
    match config.get_ssl_mode() {
        tokio_postgres::config::SslMode::Disable => "disable",
        tokio_postgres::config::SslMode::Prefer => "prefer",
        tokio_postgres::config::SslMode::Require => "require",
        _ => "unknown",
    }
}

/// Разобрать адрес хранилища из `[storage]`:
///   — не задан или пуст → SQLite по `sqlite_path`;
///   — `sqlite://путь` → SQLite по этому пути;
///   — `postgres://…` / `postgresql://…` → PostgreSQL;
///   — без `://`, но со знаком `=` (`host=… user=…`) → PostgreSQL;
///   — любая другая схема → ошибка (в тексте только схема, без пароля).
pub fn parse_backend(task_store_dsn: Option<&str>, sqlite_path: &Path) -> Result<Backend> {
    let Some(addr) = task_store_dsn.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(Backend::Sqlite(sqlite_path.to_path_buf()));
    };

    if let Some(path) = addr.strip_prefix("sqlite://") {
        return Ok(Backend::Sqlite(PathBuf::from(path)));
    }
    if addr.starts_with("postgres://") || addr.starts_with("postgresql://") {
        return Ok(Backend::Postgres(addr.to_string()));
    }
    if !addr.contains("://") && addr.contains('=') {
        return Ok(Backend::Postgres(addr.to_string()));
    }

    let scheme = scheme_of(addr);
    Err(anyhow::anyhow!(
        "хранилище '{scheme}://' не поддерживается: доступны встроенный SQLite \
         (адрес не задан или sqlite://путь) и PostgreSQL (postgres://…); \
         как добавить свою базу — раздел «Хранилище» в README"
    ))
}

/// Схема адреса для текста ошибки: часть до «://», иначе — до первого «:».
/// Нужна ровно затем, чтобы адрес с паролем (напр. `mysql://user:secret@host/db`)
/// в текст ошибки целиком не попал — только схема.
fn scheme_of(addr: &str) -> &str {
    match addr.split_once("://") {
        Some((scheme, _)) => scheme,
        None => addr.split(':').next().unwrap_or(addr),
    }
}

/// Выбор реализации хранилища: встроенный SQLite (адрес не задан или
/// `sqlite://путь`) либо PostgreSQL (адрес `postgres://…`/`postgresql://…`/
/// `host=…`). Возвращает готовое хранилище и описание для журнала — без
/// секретов и адреса: писать в журнал здесь нельзя, его слой поднимается
/// позже, уже на этом хранилище (см. порядок в main).
///
/// Чтобы добавить свою базу: завести файл реализации типажа [`Store`]
/// (`src/store/<имя>.rs`), добавить ветку разбора в [`parse_backend`] и разбор
/// адреса здесь — правок рантайма не требуется, он видит только типаж.
pub fn connect(cfg: &crate::config::StorageConfig) -> Result<(Arc<dyn Store>, String)> {
    let backend = parse_backend(cfg.task_store_dsn.as_deref(), &cfg.sqlite_path)?;
    let store: Arc<dyn Store> = match &backend {
        Backend::Sqlite(path) => Arc::new(SqliteStore::open(path)?),
        Backend::Postgres(dsn) => Arc::new(PgStore::connect(dsn, cfg.task_store_pool)?),
    };
    Ok((store, backend.describe()))
}

/// Единый вход в хранилище: методы — по смыслу операции, а не «выполни SQL».
/// Реализация обязана быть потокобезопасной: рантайм держит её в `Arc`.
#[async_trait]
pub trait Store: Send + Sync {
    /// Проверка живости соединения настоящим запросом (healthcheck).
    async fn health(&self) -> Result<()>;

    /// Создать корневую задачу (статус running), вернуть её id.
    async fn create_task(&self, t: &NewTask) -> Result<i64>;

    /// Сменить статус задачи (running/needs_input/completed/failed/cancelled).
    async fn set_task_status(&self, task_id: i64, status: &str) -> Result<()>;

    /// Текущий статус задачи; None — задачи с таким id нет (нужен `chain_cancel`
    /// и tool `task_get`, чтобы решить, закрыта задача или ещё идёт).
    async fn get_task_status(&self, task_id: i64) -> Result<Option<String>>;

    /// Записать/обновить артефакт доски задачи (upsert по (task_id,key)).
    #[allow(clippy::too_many_arguments)]
    async fn write_artifact(
        &self,
        task_id: i64,
        kind: &str,
        key: &str,
        content: Option<&str>,
        summary: Option<&str>,
        producer_agent: Option<&str>,
        producer_call_id: Option<i64>,
        depends_on: &[i64],
    ) -> Result<i64>;

    /// Прочитать артефакты задачи; опционально только указанных `kind`.
    async fn read_artifacts(&self, task_id: i64, kinds: Option<&[String]>)
        -> Result<Vec<Artifact>>;

    /// Добавить событие в журнал задачи (seq = max+1).
    async fn append_task_event(
        &self,
        task_id: i64,
        event_type: &str,
        agent: Option<&str>,
        call_id: Option<i64>,
        payload_json: Option<&str>,
    ) -> Result<()>;

    /// Записать ход прогона в `agent_turns` (потоковая запись ходов вызова).
    async fn append_turn(
        &self,
        call_id: i64,
        seq: i32,
        event: &str,
        record_json: &str,
        ts_millis: i64,
    ) -> Result<()>;

    /// Пометить ошибкой осиротевшие вызовы ЭТОГО экземпляра службы: строки,
    /// начатые раньше запуска службы и созданные тем же `instance`. Вызовы
    /// другого экземпляра на общей базе и строки без экземпляра (старые
    /// сборки) не трогаются — иначе две службы на одной базе гасили бы вызовы
    /// друг друга.
    async fn mark_orphaned_running(
        &self,
        service_started_at: i64,
        reason: &str,
        instance: &str,
    ) -> Result<Vec<OrphanedCall>>;

    /// Retention: чистка старых вызовов, событий и истёкшего кеша.
    async fn retain(&self, calls_days: i64, events_days: i64) -> Result<(u64, u64, u64)>;

    /// Последние вызовы из `agent_calls` (инструмент `agent_history`).
    async fn list_calls(
        &self,
        agent: Option<&str>,
        since: i64,
        limit: u32,
    ) -> Result<Vec<HistoryEntry>>;

    /// `created_at` строки вызова (unixepoch); None — строки нет.
    async fn get_call_created_at(&self, call_id: i64) -> Result<Option<i64>>;

    /// Создать заготовку строки вызова (`agent_calls`) ДО старта провайдера —
    /// резерв `call_id` под shared-session. Записывается только известное до
    /// вызова; метрики — нули, output/error/session_id — NULL. Возвращает id,
    /// который оркестратор прокидывает дочерним вызовам как `parent_call_id`.
    /// `instance` — экземпляр службы, создавший вызов.
    #[allow(clippy::too_many_arguments)]
    async fn insert_call_stub(
        &self,
        agent: &str,
        variant: &str,
        input_hash: &str,
        model_used: &str,
        provider: &str,
        parent_call_id: Option<i64>,
        task_id: Option<i64>,
        instance: &str,
    ) -> Result<i64>;

    /// Сохранить путь файла-итога фонового `agent_run` до старта задачи.
    async fn set_call_result_path(&self, call_id: i64, result_path: &Path) -> Result<()>;

    /// Пометить строку вызова как обслуженную из кеша.
    async fn mark_call_cached(&self, call_id: i64) -> Result<()>;

    /// Дописать заготовку вызова финальными данными провайдера (UPDATE по id).
    /// Статус задаётся явно: наличие ответа и наличие диагностической ошибки
    /// независимы для `incomplete` и `persistence_failed`.
    #[allow(clippy::too_many_arguments)]
    async fn update_call(
        &self,
        call_id: i64,
        status: CallStatus,
        output_json: Option<&str>,
        error: Option<&str>,
        tokens_in: u32,
        tokens_out: u32,
        cost_usd: Option<f64>,
        latency_ms: u64,
        session_id: Option<&str>,
        raw_input_in: u32,
        cache_creation_in: u32,
        cache_read_in: u32,
        reasoning: Option<&str>,
    ) -> Result<()>;

    /// Ранний UPDATE: записать `session_id` в заготовку вызова ДО окончания
    /// provider.complete() — чтобы дочерние invoke, пришедшие внутри
    /// subprocess parent'а, успели прочитать его и зайти в ту же сессию.
    async fn set_call_session_id(&self, call_id: i64, session_id: &str) -> Result<()>;

    /// `session_id` вызова по его id; None — NULL или строки нет.
    async fn get_call_session_id(&self, call_id: i64) -> Result<Option<String>>;

    /// Строка вызова для сборки итога; None — строки нет.
    async fn get_call_row(&self, call_id: i64) -> Result<Option<CallRow>>;

    /// Найти в кеше живую запись по ключу. None — нет или истекла.
    async fn cache_lookup(&self, key: &str) -> Result<Option<CachedEntry>>;

    /// Записать ответ в кеш с TTL (upsert по cache_key).
    async fn cache_store(
        &self,
        key: &str,
        output_json: &str,
        metadata_json: &str,
        ttl_sec: u64,
    ) -> Result<()>;

    /// Записать пачку событий журнала одной транзакцией.
    async fn write_events(&self, batch: &[LogEvent]) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orphan(id: i64, agent: &str) -> OrphanedCall {
        OrphanedCall {
            id,
            agent_name: agent.to_string(),
            created_at: 0,
            result_path: None,
        }
    }

    #[test]
    fn summarize_orphans_empty_is_empty_string() {
        assert_eq!(summarize_orphans(&[]), "");
    }

    #[test]
    fn summarize_orphans_orders_by_count_desc_then_alphabet() {
        let calls = vec![
            orphan(1, "skill-writer"),
            orphan(2, "gen-agent"),
            orphan(3, "gen-agent"),
            orphan(4, "b-agent"),
            orphan(5, "a-agent"),
            orphan(6, "gen-agent"),
        ];
        // Одинаковое количество (1) — по алфавиту: a-agent, b-agent, skill-writer.
        assert_eq!(
            summarize_orphans(&calls),
            "gen-agent×3, a-agent×1, b-agent×1, skill-writer×1"
        );
    }

    #[test]
    fn summarize_orphans_single_agent() {
        let calls = vec![orphan(1, "gen-agent"), orphan(2, "gen-agent")];
        assert_eq!(summarize_orphans(&calls), "gen-agent×2");
    }

    fn def_sqlite_path() -> std::path::PathBuf {
        std::path::PathBuf::from("data/agents-mcp.sqlite")
    }

    #[test]
    fn backend_defaults_to_sqlite() {
        let def = def_sqlite_path();
        assert_eq!(
            parse_backend(None, &def).expect("разбор"),
            Backend::Sqlite(def.clone())
        );
        assert_eq!(
            parse_backend(Some(""), &def).expect("разбор"),
            Backend::Sqlite(def.clone())
        );
        assert_eq!(
            parse_backend(Some("   "), &def).expect("разбор"),
            Backend::Sqlite(def)
        );
    }

    #[test]
    fn backend_sqlite_scheme_takes_path() {
        let backend =
            parse_backend(Some("sqlite://data/agents.sqlite"), &def_sqlite_path()).expect("разбор");
        assert_eq!(
            backend,
            Backend::Sqlite(std::path::PathBuf::from("data/agents.sqlite"))
        );
        assert_eq!(backend.describe(), "SQLite data/agents.sqlite");
    }

    #[test]
    fn backend_postgres_schemes_are_postgres() {
        let def = def_sqlite_path();
        for addr in [
            "postgres://user:secret@db-host:5432/agents",
            "postgresql://user:secret@db-host/agents",
        ] {
            let backend = parse_backend(Some(addr), &def).expect("разбор");
            assert_eq!(backend, Backend::Postgres(addr.to_string()));
            // Описание для журнала адрес и пароль не несёт.
            assert_eq!(backend.describe(), "PostgreSQL sslmode=prefer");
        }
    }

    #[test]
    fn backend_keyword_dsn_is_postgres() {
        let addr = "host=db-host user=agents password=secret dbname=agents";
        let backend = parse_backend(Some(addr), &def_sqlite_path()).expect("разбор");
        assert_eq!(backend, Backend::Postgres(addr.to_string()));
        assert_eq!(backend.describe(), "PostgreSQL sslmode=prefer");
    }

    #[test]
    fn backend_description_includes_sslmode_without_dsn() {
        let addr = "postgres://user:secret@db-host/agents?sslmode=require";
        let backend = parse_backend(Some(addr), &def_sqlite_path()).expect("разбор");
        assert_eq!(backend.describe(), "PostgreSQL sslmode=require");
        assert!(!backend.describe().contains("secret"));
    }

    #[test]
    fn backend_unknown_scheme_errors_without_secrets() {
        let err = parse_backend(
            Some("mysql://user:secret@db-host/agents"),
            &def_sqlite_path(),
        )
        .expect_err("схема не поддерживается")
        .to_string();
        assert!(err.contains("mysql://"), "в ошибке нужна схема: {err}");
        assert!(
            !err.contains("secret"),
            "пароль в текст ошибки не попадает: {err}"
        );
        assert!(
            !err.contains("db-host"),
            "адрес в текст ошибки не попадает: {err}"
        );
        assert!(
            err.contains("«Хранилище» в README"),
            "нужна подсказка, как добавить свою базу: {err}"
        );
    }
}
