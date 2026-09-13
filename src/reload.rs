//! Перечитка главного конфига без перезапуска службы.
//!
//! Одна функция — [`ConfigReloader::reload`] — и два входа в неё: MCP-инструмент
//! `config_reload` и наблюдатель файла конфига ([`crate::watcher::spawn_config`]).
//! Применяется только изменившееся, отчёт — три списка: applied (применено),
//! restart_required (требует перезапуска службы), errors (ошибки).
//!
//! На лету НЕ применяются: весь `[server]` (host/port/allowed_hosts) и
//! `[storage] log_dir, sqlite_path, task_store_dsn, task_store_pool` — о них
//! отчёт сообщает в restart_required. Значение task_store_dsn (там пароль) не
//! выводится никогда — только имя поля.
//!
//! Все перечитки идут под одним tokio-мьютексом: он же хранит последний
//! применённый конфиг и набор провайдеров, так что две перечитки одновременно не
//! пойдут, а решения об изменениях принимаются от одного состояния.

use std::collections::{HashMap, HashSet};
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use tracing::{error, info, warn};

use crate::config::{Config, ProviderApi, ProvidersConfig};
use crate::errors::safe_config_error;
use crate::health::ProviderStatus;
use crate::providers::{
    anthropic::AnthropicProvider, claude_cli::ClaudeCliProvider, codex_cli::CodexCliProvider,
    mock::MockProvider, openrouter::OpenRouterProvider, LlmProvider,
};
use crate::registry::Registry;
use crate::runtime::{Runtime, SharedOverride};
use crate::watcher;

fn safe_dotenv_error(error: &dotenvy::Error, line: Option<usize>) -> String {
    match error {
        dotenvy::Error::LineParse(_, _) => format!(
            "ошибка разбора .env в строке {} (текст строки скрыт)",
            line.unwrap_or(1)
        ),
        dotenvy::Error::Io(error) => format!("ошибка чтения .env: {:?}", error.kind()),
        dotenvy::Error::EnvVar(_) => "ошибка переменной окружения .env".to_string(),
        _ => "ошибка загрузки .env (подробности скрыты)".to_string(),
    }
}

#[derive(Clone, Default)]
pub struct ProviderEnv {
    process: HashMap<OsString, OsString>,
    dotenv: HashMap<String, String>,
}

impl ProviderEnv {
    pub(crate) fn capture(dotenv: HashMap<String, String>) -> Self {
        Self {
            process: std::env::vars_os().collect(),
            dotenv,
        }
    }

    fn with_dotenv(&self, dotenv: HashMap<String, String>) -> Self {
        Self {
            process: self.process.clone(),
            dotenv,
        }
    }

    pub(crate) fn var(&self, name: &str) -> Option<String> {
        match self.process.get(OsStr::new(name)) {
            Some(value) => value.to_str().map(str::to_owned),
            None => self.dotenv.get(name).cloned(),
        }
    }
}

struct DotenvValues {
    values: HashMap<String, String>,
    errors: Vec<String>,
}

/// Единый поиск ключей провайдеров: исходное окружение процесса имеет
/// приоритет над картой из первого найденного `.env`.
fn provider_env_var(env: &ProviderEnv, name: &str) -> Option<String> {
    env.var(name)
}

fn expand_provider_proxy(
    proxy: &Option<String>,
    env: &ProviderEnv,
) -> Result<Option<String>, String> {
    proxy
        .as_deref()
        .map(|value| crate::providers::mcp_client::expand_vars(value, |name| env.var(name)))
        .transpose()
}

fn first_dotenv_path() -> Result<Option<PathBuf>, dotenvy::Error> {
    let cwd = std::env::current_dir().map_err(dotenvy::Error::Io)?;
    for dir in cwd.ancestors() {
        let candidate = dir.join(".env");
        match std::fs::metadata(&candidate) {
            Ok(metadata) if metadata.is_file() => return Ok(Some(candidate)),
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(dotenvy::Error::Io(error)),
        }
    }
    Ok(None)
}

fn line_number(source: &str, error: &dotenvy::Error, search_from: &mut usize) -> Option<usize> {
    let dotenvy::Error::LineParse(text, _) = error else {
        return None;
    };
    let relative = source.get(*search_from..)?.find(text)?;
    let offset = *search_from + relative;
    *search_from = offset + text.len();
    Some(source[..offset].bytes().filter(|byte| *byte == b'\n').count() + 1)
}

fn load_dotenv_path(path: &Path) -> DotenvValues {
    let source = match std::fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) => {
            let error = dotenvy::Error::Io(error);
            return DotenvValues {
                values: HashMap::new(),
                errors: vec![safe_dotenv_error(&error, None)],
            };
        }
    };
    let mut values = HashMap::new();
    let mut errors = Vec::new();
    let mut search_from = 0;
    for item in dotenvy::from_read_iter(source.as_bytes()) {
        match item {
            Ok((key, value)) => {
                values.insert(key, value);
            }
            Err(error) => {
                let line = line_number(&source, &error, &mut search_from);
                errors.push(safe_dotenv_error(&error, line));
            }
        }
    }
    DotenvValues { values, errors }
}

fn load_dotenv_values() -> DotenvValues {
    match first_dotenv_path() {
        Ok(Some(path)) => load_dotenv_path(&path),
        Ok(None) => DotenvValues {
            values: HashMap::new(),
            errors: Vec::new(),
        },
        Err(error) => DotenvValues {
            values: HashMap::new(),
            errors: vec![safe_dotenv_error(&error, None)],
        },
    }
}

/// Загрузить `.env` до старта потоков. В окружение добавляются только
/// отсутствующие ключи; отдельный снимок исходного окружения нужен перечитке.
pub fn load_startup_dotenv() -> (ProviderEnv, Vec<String>) {
    let loaded = load_dotenv_values();
    let env = ProviderEnv::capture(loaded.values);
    for (key, value) in &env.dotenv {
        if !env.process.contains_key(OsStr::new(key)) {
            std::env::set_var(key, value);
        }
    }
    (env, loaded.errors)
}

/// Набор собранных провайдеров вместе с отпечатками их настроек.
#[derive(Clone)]
pub struct ProviderSet {
    pub providers: HashMap<String, Arc<dyn LlmProvider>>,
    /// имя → регистрация либо итог последней выполненной CLI-проверки
    pub statuses: HashMap<String, ProviderStatus>,
    /// имя → отпечаток настроек и ключа (ключ в открытом виде не хранить)
    fingerprints: HashMap<String, String>,
    /// Провайдеры, пропущенные из-за пустой переменной ключа: предупреждение о
    /// них пишем один раз, а не при каждой перечитке.
    skipped: HashSet<String>,
}

/// Предупреждать ли о пропуске провайдера: при старте — всегда, при перечитке —
/// только если в прошлой сборке он пропущен не был.
fn warn_skip(prev: Option<&ProviderSet>, name: &str) -> bool {
    prev.map_or(true, |p| !p.skipped.contains(name))
}

/// Что изменилось в наборе провайдеров за одну сборку (имена, по алфавиту).
#[derive(Debug, Default)]
pub struct ProviderChanges {
    pub added: Vec<String>,
    pub changed: Vec<String>,
    pub removed: Vec<String>,
}

impl ProviderChanges {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.changed.is_empty() && self.removed.is_empty()
    }
}

/// Собрать набор провайдеров по секции `[providers]`. Логика и сообщения журнала —
/// те же, что раньше стояли в main.rs. Отличие одно: провайдер, настройки
/// которого (и значение ключа) не изменились с прошлой сборки, берётся ИЗ `prev`
/// тем же `Arc` — так у него сохраняются семафор и соединения, а doctor self-test
/// не гоняется заново.
pub async fn build_providers(
    cfg: &ProvidersConfig,
    prev: Option<&ProviderSet>,
    env: &ProviderEnv,
) -> (ProviderSet, ProviderChanges) {
    let mut providers: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
    let mut statuses: HashMap<String, ProviderStatus> = HashMap::new();
    let mut fingerprints: HashMap<String, String> = HashMap::new();
    let mut skipped: HashSet<String> = HashSet::new();

    // mock — всегда включён для тестов, отпечаток постоянный.
    providers.insert("mock".to_string(), Arc::new(MockProvider::new()));
    statuses.insert("mock".to_string(), ProviderStatus::registered());
    fingerprints.insert("mock".to_string(), "mock".to_string());

    if let Some(entry) = &cfg.openrouter {
        if entry.api == Some(ProviderApi::Anthropic) {
            warn!(
                api = "anthropic",
                "секция providers.openrouter задаёт противоречащий вид API — использую openai"
            );
        }
        let proxy = match expand_provider_proxy(&entry.proxy, env) {
            Ok(proxy) => proxy,
            Err(error) => {
                if warn_skip(prev, "openrouter") {
                    tracing::warn!(
                        provider = "openrouter",
                        error = %error,
                        "proxy провайдера не раскрыт — пропускаю"
                    );
                }
                skipped.insert("openrouter".into());
                None
            }
        };
        if !skipped.contains("openrouter") {
            match provider_env_var(env, &entry.api_key_env) {
                Some(key) if !key.is_empty() => {
                    let fp = fingerprint_with_key_and_proxy(&format!("{:?}", entry), &key, proxy.as_deref());
                    match reuse_prev(prev, "openrouter", &fp) {
                        Some(provider) => {
                            providers.insert("openrouter".into(), provider);
                        }
                        None => {
                            let provider = OpenRouterProvider::new(
                                "openrouter",
                                key,
                                entry.base_url.clone(),
                                entry.default_referer.clone(),
                                proxy,
                                entry.proxy_bypass.clone(),
                                entry.max_concurrent,
                                entry.prices.clone(),
                            );
                            providers.insert("openrouter".into(), Arc::new(provider));
                            info!(env = %entry.api_key_env, "провайдер openrouter подключён");
                        }
                    }
                    fingerprints.insert("openrouter".into(), fp);
                    statuses.insert("openrouter".into(), ProviderStatus::registered());
                }
                _ => {
                    if warn_skip(prev, "openrouter") {
                        tracing::warn!(
                            env = %entry.api_key_env,
                            "провайдер openrouter настроен в конфиге, но env-переменная пуста — пропускаю"
                        );
                    }
                    skipped.insert("openrouter".into());
                }
            }
        }
    }
    if let Some(entry) = &cfg.anthropic {
        if entry.api == Some(ProviderApi::Openai) {
            warn!(
                api = "openai",
                "секция providers.anthropic задаёт противоречащий вид API — использую anthropic"
            );
        }
        let proxy = match expand_provider_proxy(&entry.proxy, env) {
            Ok(proxy) => proxy,
            Err(error) => {
                if warn_skip(prev, "anthropic") {
                    tracing::warn!(
                        provider = "anthropic",
                        error = %error,
                        "proxy провайдера не раскрыт — пропускаю"
                    );
                }
                skipped.insert("anthropic".into());
                None
            }
        };
        if !skipped.contains("anthropic") {
            match provider_env_var(env, &entry.api_key_env) {
                Some(key) if !key.is_empty() => {
                    let fp = fingerprint_with_key_and_proxy(&format!("{:?}", entry), &key, proxy.as_deref());
                    match reuse_prev(prev, "anthropic", &fp) {
                        Some(provider) => {
                            providers.insert("anthropic".into(), provider);
                        }
                        None => {
                            let provider = AnthropicProvider::new(
                                "anthropic".into(),
                                key,
                                entry.base_url.clone(),
                                proxy,
                                entry.proxy_bypass.clone(),
                                entry.max_concurrent,
                                entry.prompt_cache,
                                entry.prices.clone(),
                            );
                            providers.insert("anthropic".into(), Arc::new(provider));
                            info!(env = %entry.api_key_env, "провайдер anthropic подключён");
                        }
                    }
                    fingerprints.insert("anthropic".into(), fp);
                    statuses.insert("anthropic".into(), ProviderStatus::registered());
                }
                _ => {
                    if warn_skip(prev, "anthropic") {
                        tracing::warn!(
                            env = %entry.api_key_env,
                            "провайдер anthropic настроен в конфиге, но env-переменная пуста — пропускаю"
                        );
                    }
                    skipped.insert("anthropic".into());
                }
            }
        }
    }
    // Прямые OpenAI-совместимые и Anthropic Messages endpoint'ы.
    // Один универсальный механизм на любое число подключений: каждая секция
    // [providers.direct.<name>] поднимает отдельный провайдер под своим именем.
    // Добавление нового вендора = только конфиг, без правок кода.
    for (name, entry) in &cfg.direct {
        if matches!(
            name.as_str(),
            "mock" | "openrouter" | "anthropic" | "claude-cli" | "codex-cli"
        ) {
            tracing::warn!(
                provider = %name,
                "имя прямого провайдера конфликтует с зарезервированным — пропускаю"
            );
            continue;
        }
        let proxy = match expand_provider_proxy(&entry.proxy, env) {
            Ok(proxy) => proxy,
            Err(error) => {
                if warn_skip(prev, name) {
                    tracing::warn!(
                        provider = %name,
                        error = %error,
                        "proxy прямого провайдера не раскрыт — пропускаю"
                    );
                }
                skipped.insert(name.clone());
                continue;
            }
        };
        match provider_env_var(env, &entry.api_key_env) {
            Some(key) if !key.is_empty() => {
                let fp = fingerprint_with_key_and_proxy(&format!("{:?}", entry), &key, proxy.as_deref());
                match reuse_prev(prev, name, &fp) {
                    Some(provider) => {
                        providers.insert(name.clone(), provider);
                    }
                    None => {
                        let provider: Arc<dyn LlmProvider> = match entry
                            .api
                            .unwrap_or(ProviderApi::Openai)
                        {
                            ProviderApi::Openai => Arc::new(OpenRouterProvider::new(
                                name.clone(),
                                key,
                                entry.base_url.clone(),
                                entry.default_referer.clone(),
                                proxy.clone(),
                                entry.proxy_bypass.clone(),
                                entry.max_concurrent,
                                entry.prices.clone(),
                            )),
                            ProviderApi::Anthropic => Arc::new(AnthropicProvider::new(
                                name.clone(),
                                key,
                                entry.base_url.clone(),
                                proxy.clone(),
                                entry.proxy_bypass.clone(),
                                entry.max_concurrent,
                                entry.prompt_cache,
                                entry.prices.clone(),
                            )),
                        };
                        providers.insert(name.clone(), provider);
                        info!(provider = %name, env = %entry.api_key_env, "прямой провайдер подключён");
                    }
                }
                fingerprints.insert(name.clone(), fp);
                statuses.insert(name.clone(), ProviderStatus::registered());
            }
            _ => {
                if warn_skip(prev, name) {
                    tracing::warn!(
                        provider = %name,
                        env = %entry.api_key_env,
                        "прямой провайдер настроен в конфиге, но env-переменная пуста — пропускаю"
                    );
                }
                skipped.insert(name.clone());
            }
        }
    }

    if let Some(cli_cfg) = &cfg.claude_cli {
        let fp = fingerprint_section(&format!("{:?}", cli_cfg));
        match reuse_prev(prev, "claude-cli", &fp) {
            Some(provider) => {
                providers.insert("claude-cli".into(), provider);
                statuses.insert(
                    "claude-cli".into(),
                    reuse_status(prev, "claude-cli").unwrap_or_else(ProviderStatus::registered),
                );
            }
            None => {
                let exe_path = std::path::PathBuf::from(&cli_cfg.executable);
                // Doctor self-test: вызов `claude --version`. Если CLI недоступен —
                // регистрируем провайдер всё равно, но в логе warn (агенты упадут
                // на complete() с понятной ошибкой). Так health видно сразу.
                let status = match ClaudeCliProvider::doctor(&exe_path).await {
                    Ok(version) => {
                        info!(executable = %cli_cfg.executable, %version, "провайдер claude-cli подключён");
                        ProviderStatus::ok()
                    }
                    Err(e) => {
                        tracing::warn!(
                            executable = %cli_cfg.executable,
                            error = %e,
                            "doctor self-test claude --version упал — провайдер всё равно регистрируется, но invoke упадёт"
                        );
                        ProviderStatus::down(e)
                    }
                };
                let provider = ClaudeCliProvider::new(
                    exe_path,
                    cli_cfg.max_concurrent,
                    cli_cfg.default_max_turns,
                    cli_cfg.config_dir.clone().map(std::path::PathBuf::from),
                );
                providers.insert("claude-cli".into(), Arc::new(provider));
                statuses.insert("claude-cli".into(), status);
            }
        }
        fingerprints.insert("claude-cli".into(), fp);
    }

    if let Some(codex_cfg) = &cfg.codex_cli {
        let fp = fingerprint_section(&format!("{:?}", codex_cfg));
        match reuse_prev(prev, "codex-cli", &fp) {
            Some(provider) => {
                providers.insert("codex-cli".into(), provider);
                statuses.insert(
                    "codex-cli".into(),
                    reuse_status(prev, "codex-cli").unwrap_or_else(ProviderStatus::registered),
                );
            }
            None => {
                let exe_path = std::path::PathBuf::from(&codex_cfg.executable);
                // Doctor self-test: вызов `codex --version`. Если CLI недоступен —
                // регистрируем провайдер всё равно, но в логе warn (агенты упадут
                // на complete() с понятной ошибкой). Так health видно сразу.
                let status = match CodexCliProvider::doctor(&exe_path).await {
                    Ok(version) => {
                        info!(executable = %codex_cfg.executable, %version, "провайдер codex-cli подключён");
                        ProviderStatus::ok()
                    }
                    Err(e) => {
                        tracing::warn!(
                            executable = %codex_cfg.executable,
                            error = %e,
                            "doctor self-test codex --version упал — провайдер всё равно регистрируется, но invoke упадёт"
                        );
                        ProviderStatus::down(e)
                    }
                };
                let provider = CodexCliProvider::new(
                    exe_path,
                    std::path::PathBuf::from(&codex_cfg.codex_home),
                    codex_cfg.max_concurrent,
                    codex_cfg.proxy.clone(),
                    codex_cfg.proxy_bypass.clone(),
                );
                providers.insert("codex-cli".into(), Arc::new(provider));
                statuses.insert("codex-cli".into(), status);
            }
        }
        fingerprints.insert("codex-cli".into(), fp);
    }

    let changes = provider_changes(prev, &fingerprints);
    (
        ProviderSet {
            providers,
            statuses,
            fingerprints,
            skipped,
        },
        changes,
    )
}

/// Провайдер с тем же именем и тем же отпечатком берём из прошлого набора: не
/// создаём заново и не гоняем doctor.
fn reuse_prev(prev: Option<&ProviderSet>, name: &str, fp: &str) -> Option<Arc<dyn LlmProvider>> {
    let prev = prev?;
    match prev.fingerprints.get(name) {
        Some(old) if old == fp => prev.providers.get(name).cloned(),
        _ => None,
    }
}

fn reuse_status(prev: Option<&ProviderSet>, name: &str) -> Option<ProviderStatus> {
    prev?.statuses.get(name).cloned()
}

fn sha256_hex(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Отпечаток настройки провайдера вместе со значением ключа (в открытом виде
/// ключ не хранится — только хеш).
fn fingerprint_with_key_and_proxy(
    entry_debug: &str,
    key: &str,
    proxy: Option<&str>,
) -> String {
    sha256_hex(&format!("{entry_debug}\u{0}{key}\u{0}{}", proxy.unwrap_or_default()))
}

/// Отпечаток секции без ключа — у claude-cli/codex-cli ключа нет.
fn fingerprint_section(section_debug: &str) -> String {
    sha256_hex(section_debug)
}

/// Что изменилось: сравнение имён и отпечатков с прошлой сборкой. При prev=None
/// всё, кроме mock, считается добавленным.
fn provider_changes(
    prev: Option<&ProviderSet>,
    fingerprints: &HashMap<String, String>,
) -> ProviderChanges {
    let mut changes = ProviderChanges::default();
    for (name, fp) in fingerprints {
        // mock есть всегда и собирается заново — изменением не считается.
        if name == "mock" {
            continue;
        }
        match prev.and_then(|p| p.fingerprints.get(name)) {
            None => changes.added.push(name.clone()),
            Some(old) if old != fp => changes.changed.push(name.clone()),
            Some(_) => {}
        }
    }
    if let Some(prev) = prev {
        for name in prev.providers.keys() {
            if name != "mock" && !fingerprints.contains_key(name) {
                changes.removed.push(name.clone());
            }
        }
    }
    changes.added.sort();
    changes.changed.sort();
    changes.removed.sort();
    changes
}

/// Последний применённый конфиг и набор провайдеров.
struct ReloadState {
    cfg: Config,
    providers: ProviderSet,
    provider_env: ProviderEnv,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ServicePath {
    pub(crate) name: &'static str,
    pub(crate) path: PathBuf,
    pub(crate) write_only: bool,
}

/// Пути самой службы, которые не должны становиться доступными агентам даже
/// при слишком широком `[fs] allowed_roots`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ServicePaths {
    pub(crate) entries: Vec<ServicePath>,
}

impl ServicePaths {
    fn from_config(config_path: Option<&std::path::Path>, cfg: &Config) -> Self {
        let mut entries = Vec::new();
        if let Some(config_path) = config_path {
            let config_path = absolute_path(config_path);
            if let Some(config_dir) = config_path.parent() {
                entries.push(ServicePath {
                    name: "каталог главного конфига",
                    path: config_dir.to_path_buf(),
                    write_only: false,
                });
                entries.push(ServicePath {
                    name: ".env рядом с главным конфигом",
                    path: config_dir.join(".env"),
                    write_only: false,
                });
            }
        }
        // dotenvy (старт и перечитка) берёт первый .env от текущего каталога
        // вверх по родителям — защищаем каждого кандидата: и существующий файл,
        // и место, где новый .env перекрыл бы его.
        if let Ok(cwd) = std::env::current_dir() {
            for dir in cwd.ancestors() {
                entries.push(ServicePath {
                    name: ".env службы",
                    path: dir.join(".env"),
                    write_only: false,
                });
            }
        }
        entries.extend([
            ServicePath {
                name: "agents_dir",
                path: absolute_path(&cfg.agents.agents_dir),
                write_only: false,
            },
            ServicePath {
                name: "log_dir",
                path: absolute_path(&cfg.storage.log_dir),
                write_only: false,
            },
            ServicePath {
                name: "runs_dir",
                path: absolute_path(&cfg.storage.runs_dir),
                write_only: true,
            },
        ]);
        Self { entries }
    }

    fn extend_missing(&mut self, other: Self) {
        for entry in other.entries {
            if !self.entries.iter().any(|current| current == &entry) {
                self.entries.push(entry);
            }
        }
    }
}

fn absolute_path(path: &std::path::Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
}

/// Тексты предупреждений вынесены в чистую функцию, чтобы проверять их без
/// перехвата журнала.
pub(crate) fn service_path_overlap_warnings(
    roots: &[PathBuf],
    service_paths: &ServicePaths,
) -> Vec<String> {
    let canonical_roots: Vec<PathBuf> = roots
        .iter()
        .filter_map(|root| std::fs::canonicalize(root).ok())
        .collect();
    let mut warnings = Vec::new();
    for entry in &service_paths.entries {
        let Ok(path) = crate::server::canonicalize_with_missing(&entry.path) else {
            continue;
        };
        for root in &canonical_roots {
            if path.starts_with(root) {
                warnings.push(format!(
                    "allowed_roots '{}' накрывает служебный путь {} '{}'; доступ через fs_* всё равно запрещён",
                    root.display(),
                    entry.name,
                    entry.path.display()
                ));
            }
        }
    }
    warnings
}

/// Перечитка главного конфига: применяет изменившееся и рассказывает, что без
/// перезапуска применить нельзя.
pub struct ConfigReloader {
    config_path: Option<PathBuf>,
    /// Последний применённый конфиг и набор провайдеров. Замок защищает только
    /// снимок и фазу применения; медленная самопроверка CLI идёт без него.
    state: tokio::sync::Mutex<ReloadState>,
    generation: std::sync::atomic::AtomicU64,
    registry: Arc<Registry>,
    runtime: Arc<Runtime>,
    force_override: SharedOverride,
    fs_roots: Arc<std::sync::RwLock<Vec<PathBuf>>>,
    service_paths: Arc<std::sync::RwLock<ServicePaths>>,
    agents_watcher: std::sync::Mutex<Option<notify::RecommendedWatcher>>,
}

/// Отчёт перечитки: что применено, что требует перезапуска и что упало.
#[derive(Debug, Default, serde::Serialize)]
pub struct ReloadReport {
    pub applied: Vec<String>,
    pub restart_required: Vec<String>,
    pub errors: Vec<String>,
}

impl ConfigReloader {
    pub fn new(
        config_path: Option<PathBuf>,
        cfg: Config,
        providers: ProviderSet,
        provider_env: ProviderEnv,
        registry: Arc<Registry>,
        runtime: Arc<Runtime>,
        force_override: SharedOverride,
    ) -> Arc<Self> {
        runtime.set_provider_set(providers.providers.clone(), providers.statuses.clone());
        let fs_roots = Arc::new(std::sync::RwLock::new(cfg.fs.allowed_roots.clone()));
        let service_paths = Arc::new(std::sync::RwLock::new(ServicePaths::from_config(
            config_path.as_deref(),
            &cfg,
        )));
        let reloader = Arc::new(Self {
            config_path,
            state: tokio::sync::Mutex::new(ReloadState {
                cfg,
                providers,
                provider_env,
            }),
            generation: std::sync::atomic::AtomicU64::new(0),
            registry,
            runtime,
            force_override,
            fs_roots,
            service_paths,
            agents_watcher: std::sync::Mutex::new(None),
        });
        reloader.warn_service_path_overlaps();
        reloader
    }

    /// Общий с файловыми инструментами список разрешённых корней: перечитка
    /// `[fs] allowed_roots` подменяет его содержимое на лету.
    pub fn fs_roots(&self) -> Arc<std::sync::RwLock<Vec<PathBuf>>> {
        self.fs_roots.clone()
    }

    /// Общий с файловыми инструментами запретный список служебных путей.
    pub(crate) fn service_paths(&self) -> Arc<std::sync::RwLock<ServicePaths>> {
        self.service_paths.clone()
    }

    fn warn_service_path_overlaps(&self) {
        let roots = self.fs_roots.read().unwrap_or_else(|e| e.into_inner());
        let service_paths = self
            .service_paths
            .read()
            .unwrap_or_else(|e| e.into_inner());
        for warning in service_path_overlap_warnings(&roots, &service_paths) {
            warn!("{warning}");
        }
    }

    /// Наблюдатель каталога `agents/` при старте службы: включён hot_reload —
    /// запускаем, выключен — так и пишем в журнал.
    pub fn start_agents_watcher(&self, agents_dir: PathBuf, hot_reload: bool) {
        if hot_reload {
            let _ = self.restart_agents_watcher(agents_dir);
        } else {
            info!("hot-reload отключён в конфиге");
        }
    }

    /// Пересоздать наблюдатель `agents/` на указанном каталоге: прежний
    /// RecommendedWatcher кладётся поверх и вместе с ним останавливается.
    fn restart_agents_watcher(&self, agents_dir: PathBuf) -> Result<(), String> {
        let spawned = match watcher::spawn(agents_dir, self.registry.clone()) {
            Ok(w) => Some(w),
            Err(e) => {
                warn!(error = %e, "не удалось запустить hot-reload watcher");
                let mut slot = self
                    .agents_watcher
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                *slot = None;
                return Err(e.to_string());
            }
        };
        let mut slot = self
            .agents_watcher
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *slot = spawned;
        Ok(())
    }

    /// Остановить наблюдатель `agents/` — hot_reload выключили в конфиге.
    fn stop_agents_watcher(&self) {
        let mut slot = self
            .agents_watcher
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *slot = None;
    }

    /// Перечитать главный конфиг и применить изменившееся. Звать можно
    /// конкурентно: вторая перечитка дождётся первой.
    pub async fn reload(&self) -> ReloadReport {
        #[cfg(not(test))]
        let dotenv = load_dotenv_values();
        // Юнит-тесты не должны читать настоящий `.env` рабочего каталога.
        #[cfg(test)]
        let dotenv = DotenvValues {
            values: HashMap::new(),
            errors: Vec::new(),
        };
        self.reload_with_dotenv(dotenv).await
    }

    async fn reload_with_dotenv(&self, dotenv: DotenvValues) -> ReloadReport {
        let mut report = ReloadReport::default();
        // Двухфазная перечитка: снимок состояния берём под замком, но doctor
        // запускаем без него. Более новая перечитка увеличит поколение, и этот
        // результат будет отброшен до применения каких-либо изменений.
        let generation = self
            .generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        let (previous_providers, previous_env) = {
            let st = self.state.lock().await;
            (st.providers.clone(), st.provider_env.clone())
        };

        let Some(path) = self.config_path.clone() else {
            report.errors.push(
                "служба запущена без файла конфига (--config или AGENTS_MCP_CONFIG) — перечитывать нечего"
                    .to_string(),
            );
            self.warn_service_path_overlaps();
            log_report(&report);
            return report;
        };

        // Перечитка обновляет только собственную карту: окружение живого
        // многопоточного процесса не меняем. Файла нет — пустая карта.
        report
            .errors
            .extend(dotenv.errors.iter().map(|error| format!("файл .env: {error}")));
        let provider_env = previous_env.with_dotenv(dotenv.values);

        let new = match Config::load_or_default(Some(path.as_path())) {
            Ok(cfg) => cfg,
            Err(e) => {
                report.errors.push(format!(
                    "конфиг не прочитан, ничего не применено: {}",
                    safe_config_error(&e)
                ));
                self.warn_service_path_overlaps();
                log_report(&report);
                return report;
            }
        };

        let (set, changes) =
            build_providers(&new.providers, Some(&previous_providers), &provider_env).await;
        let mut st = self.state.lock().await;
        if self.generation.load(std::sync::atomic::Ordering::SeqCst) != generation {
            report
                .errors
                .push("результат перечитки устарел и не применён".to_string());
            log_report(&report);
            return report;
        }

        // До применения изменений защищаем и прежние, и новые пути. Так смена
        // agents_dir/runs_dir не создаёт короткого окна между двумя RwLock.
        {
            let mut paths = self
                .service_paths
                .write()
                .unwrap_or_else(|e| e.into_inner());
            paths.extend_missing(ServicePaths::from_config(
                self.config_path.as_deref(),
                &new,
            ));
        }

        // ── [storage] runs_dir — каталог файлов-итогов ──────────────────────
        if new.storage.runs_dir != st.cfg.storage.runs_dir {
            self.runtime.set_runs_dir(new.storage.runs_dir.clone());
            report.applied.push(format!(
                "[storage] runs_dir: {}",
                new.storage.runs_dir.display()
            ));
        }

        // ── [agents] force_provider/force_model — тест-override модели ──────
        if new.agents.force_provider != st.cfg.agents.force_provider
            || new.agents.force_model != st.cfg.agents.force_model
        {
            {
                let mut ov = self
                    .force_override
                    .write()
                    .unwrap_or_else(|e| e.into_inner());
                ov.provider = new.agents.force_provider.clone();
                ov.model = new.agents.force_model.clone();
            }
            report.applied.push(format!(
                "[agents] force_provider/force_model: {:?} / {:?}",
                new.agents.force_provider, new.agents.force_model
            ));
        }
        if new.agents.default_timeout_sec != st.cfg.agents.default_timeout_sec {
            self.runtime
                .set_default_timeout_sec(new.agents.default_timeout_sec);
            report.applied.push(format!(
                "[agents] default_timeout_sec: {}",
                new.agents.default_timeout_sec
            ));
        }

        // ── [agents] agents_dir — подмена каталога реестра ──────────────────
        let dir_changed = new.agents.agents_dir != st.cfg.agents.agents_dir;
        let mut dir_applied = false;
        if dir_changed {
            match self.registry.reload_from(new.agents.agents_dir.clone()) {
                Ok(()) => {
                    dir_applied = true;
                    report.applied.push(format!(
                        "[agents] agents_dir: {}",
                        new.agents.agents_dir.display()
                    ));
                }
                // Каталог не прочитался: прежние каталог и реестр остаются.
                Err(e) => report.errors.push(format!(
                    "[agents] agents_dir {}: {e} — прежние каталог и реестр остаются",
                    new.agents.agents_dir.display()
                )),
            }
        }

        // ── [agents] hot_reload — наблюдатель agents/ ───────────────────────
        // Наблюдатель главного конфига не трогаем: он решается при старте службы.
        // Каталог, который реально в работе: новый не прочитался — прежний.
        let active_dir = if dir_changed && !dir_applied {
            st.cfg.agents.agents_dir.clone()
        } else {
            new.agents.agents_dir.clone()
        };
        let mut watcher_applied = true;
        if new.agents.hot_reload != st.cfg.agents.hot_reload {
            if new.agents.hot_reload {
                match self.restart_agents_watcher(active_dir) {
                    Ok(()) => report.applied.push(
                        "[agents] hot_reload: включён — наблюдатель agents/ запущен (наблюдатель главного конфига не трогаем: он решён при старте службы)"
                            .to_string(),
                    ),
                    Err(error) => {
                        watcher_applied = false;
                        report.errors.push(format!(
                            "[agents] hot_reload: наблюдатель agents/ не запущен: {error}"
                        ));
                    }
                }
            } else {
                self.stop_agents_watcher();
                report.applied.push(
                    "[agents] hot_reload: выключен — наблюдатель agents/ остановлен (наблюдатель главного конфига не трогаем: он решён при старте службы)"
                        .to_string(),
                );
            }
        } else if dir_applied && new.agents.hot_reload {
            // Каталог подменён — переводим на него и наблюдатель.
            if let Err(error) = self.restart_agents_watcher(new.agents.agents_dir.clone()) {
                watcher_applied = false;
                report.errors.push(format!(
                    "[agents] hot_reload: наблюдатель agents/ не запущен: {error}"
                ));
            }
        }

        // ── [providers.*] — пересборка набора ──────────────────────────────
        if !changes.is_empty() {
            self.runtime
                .set_provider_set(set.providers.clone(), set.statuses.clone());
            report.applied.push(format!(
                "[providers] добавлены: {}; изменены: {}; удалены: {}",
                join_names(&changes.added),
                join_names(&changes.changed),
                join_names(&changes.removed)
            ));
        }
        // Набор запоминаем всегда: тот же Arc остаётся и для следующей сборки.
        st.providers = set;
        self.runtime.set_provider_env(provider_env.clone());
        st.provider_env = provider_env;

        // ── [skills] rag_query_url ──────────────────────────────────────────
        if new.skills != st.cfg.skills {
            self.runtime
                .set_skills(crate::skills::SkillsClient::new(new.skills.rag_query_url.clone()));
            report.applied.push(format!(
                "[skills] rag_query_url: {}",
                new.skills
                    .rag_query_url
                    .clone()
                    .unwrap_or_else(|| "не задан (инъекция навыков выключена)".to_string())
            ));
        }

        // ── [fs] allowed_roots ──────────────────────────────────────────────
        if new.fs != st.cfg.fs {
            {
                let mut roots = self.fs_roots.write().unwrap_or_else(|e| e.into_inner());
                *roots = new.fs.allowed_roots.clone();
            }
            report.applied.push(format!(
                "[fs] allowed_roots: {}",
                join_paths(&new.fs.allowed_roots)
            ));
        }

        // ── Что применяется только перезапуском службы ──────────────────────
        if new.server.host != st.cfg.server.host {
            report
                .restart_required
                .push("[server] host: применяется только перезапуском службы".to_string());
        }
        if new.server.port != st.cfg.server.port {
            report
                .restart_required
                .push("[server] port: применяется только перезапуском службы".to_string());
        }
        if new.server.instance != st.cfg.server.instance {
            report.restart_required.push(
                "[server] instance: применяется только перезапуском службы".to_string(),
            );
        }
        if new.server.allowed_hosts != st.cfg.server.allowed_hosts {
            report.restart_required.push(
                "[server] allowed_hosts: применяется только перезапуском службы".to_string(),
            );
        }
        if new.storage.log_dir != st.cfg.storage.log_dir {
            report
                .restart_required
                .push("[storage] log_dir: применяется только перезапуском службы".to_string());
        }
        if new.storage.sqlite_path != st.cfg.storage.sqlite_path {
            report.restart_required.push(
                "[storage] sqlite_path: применяется только перезапуском службы".to_string(),
            );
        }
        // Значение DSN (там пароль) не выводим никогда — только имя поля.
        if new.storage.task_store_dsn != st.cfg.storage.task_store_dsn {
            report.restart_required.push(
                "[storage] task_store_dsn: значение не выводится, применяется только перезапуском службы"
                    .to_string(),
            );
        }
        if new.storage.task_store_pool != st.cfg.storage.task_store_pool {
            report.restart_required.push(
                "[storage] task_store_pool: применяется только перезапуском службы".to_string(),
            );
        }
        // Применённое состояние: в next остаётся ровно то, что служба реально
        // взяла в работу. Неприменённое возвращаем из прежнего конфига — тогда
        // каждая следующая перечитка до перезапуска снова о нём сообщит.
        let mut next = new;
        next.server = st.cfg.server.clone();
        next.storage.log_dir = st.cfg.storage.log_dir.clone();
        next.storage.sqlite_path = st.cfg.storage.sqlite_path.clone();
        next.storage.task_store_dsn = st.cfg.storage.task_store_dsn.clone();
        next.storage.task_store_pool = st.cfg.storage.task_store_pool.clone();
        if dir_changed && !dir_applied {
            next.agents.agents_dir = st.cfg.agents.agents_dir.clone();
        }
        if !watcher_applied {
            // Фактически watcher выключен; следующая перечитка повторит запуск.
            next.agents.hot_reload = false;
        }
        {
            let mut paths = self
                .service_paths
                .write()
                .unwrap_or_else(|e| e.into_inner());
            *paths = ServicePaths::from_config(self.config_path.as_deref(), &next);
        }
        st.cfg = next;

        self.warn_service_path_overlaps();
        log_report(&report);
        report
    }
}

/// Журнал отчёта: по записи на строку — applied через info!, restart_required
/// через warn!, errors через error!. Пусто во всех трёх — одна строка.
fn log_report(report: &ReloadReport) {
    if report.applied.is_empty() && report.restart_required.is_empty() && report.errors.is_empty() {
        info!("главный конфиг перечитан: изменений нет");
        return;
    }
    for line in &report.applied {
        info!("{}", line);
    }
    for line in &report.restart_required {
        warn!("{}", line);
    }
    for line in &report.errors {
        error!("{}", line);
    }
}

fn join_names(names: &[String]) -> String {
    if names.is_empty() {
        "—".to_string()
    } else {
        names.join(", ")
    }
}

fn join_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    #[test]
    fn dotenv_parse_error_hides_secret() {
        let dir = temp_dir("dotenv-error");
        let path = dir.join(".env");
        std::fs::write(
            &path,
            "BEFORE=\"one\ntwo\"\nAGENTS_MCP_TASK_STORE_DSN=postgres://u:hunter2@h/db bad\nAFTER=three\n",
        )
        .expect(".env записан");
        let loaded = super::load_dotenv_path(&path);
        assert_eq!(loaded.values.get("BEFORE").map(String::as_str), Some("one\ntwo"));
        assert_eq!(loaded.values.get("AFTER").map(String::as_str), Some("three"));
        assert_eq!(loaded.errors.len(), 1);
        assert!(loaded.errors[0].contains("строке 3"));
        assert!(!loaded.errors[0].contains("hunter2"));
        let error = dotenvy::Error::EnvVar(std::env::VarError::NotUnicode("hunter2".into()));
        assert!(!super::safe_dotenv_error(&error, None).contains("hunter2"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn process_environment_has_priority_over_dotenv() {
        let name = OsString::from("AGENTS_MCP_ENV_PRIORITY_TEST");
        let mut process = HashMap::new();
        process.insert(name, OsString::from("process"));
        let mut first_dotenv = HashMap::new();
        first_dotenv.insert("AGENTS_MCP_ENV_PRIORITY_TEST".to_string(), "startup-dotenv".to_string());
        let startup = ProviderEnv {
            process,
            dotenv: first_dotenv,
        };
        assert_eq!(
            provider_env_var(&startup, "AGENTS_MCP_ENV_PRIORITY_TEST").as_deref(),
            Some("process")
        );

        let mut reloaded_dotenv = HashMap::new();
        reloaded_dotenv.insert("AGENTS_MCP_ENV_PRIORITY_TEST".to_string(), "reload-dotenv".to_string());
        let reloaded = startup.with_dotenv(reloaded_dotenv);
        assert_eq!(
            provider_env_var(&reloaded, "AGENTS_MCP_ENV_PRIORITY_TEST").as_deref(),
            Some("process")
        );
    }

    use super::*;
    use crate::runtime::ModelOverride;
    use crate::store::SqliteStore;

    /// Временный каталог для одного теста: в имени — метка и наносекунды, чтобы
    /// параллельные тесты друг с другом не пересекались.
    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("время")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("agents-mcp-reload-{tag}-{nanos}"));
        std::fs::create_dir_all(&dir).expect("временный каталог");
        dir
    }

    fn write_config(path: &std::path::Path, text: &str) {
        std::fs::write(path, text).expect("конфиг записан");
    }

    async fn test_reloader(
        config_path: PathBuf,
        cfg: Config,
        provider_env: ProviderEnv,
    ) -> Arc<ConfigReloader> {
        let (providers, _) = build_providers(&cfg.providers, None, &provider_env).await;
        let registry = Arc::new(
            Registry::load(cfg.agents.agents_dir.clone()).expect("реестр для перечитки"),
        );
        let store: Arc<dyn crate::store::Store> = Arc::new(
            SqliteStore::open(std::path::Path::new(":memory:")).expect("sqlite в памяти"),
        );
        let force_override: SharedOverride =
            Arc::new(std::sync::RwLock::new(ModelOverride::default()));
        let runtime = Arc::new(Runtime::new(
            store,
            registry.clone(),
            HashMap::new(),
            crate::skills::SkillsClient::new(None),
            force_override.clone(),
            cfg.storage.runs_dir.clone(),
            "test:1".to_string(),
            cfg.agents.default_timeout_sec,
        ));
        ConfigReloader::new(
            Some(config_path),
            cfg,
            providers,
            provider_env,
            registry,
            runtime,
            force_override,
        )
    }

    #[tokio::test]
    async fn reload_uses_changed_dotenv_without_mutating_process() {
        let dir = temp_dir("dotenv-reload");
        let config_path = dir.join("agents-mcp.toml");
        let dotenv_path = dir.join(".env");
        let agents_dir = dir.join("agents");
        std::fs::create_dir_all(&agents_dir).expect("каталог агентов");
        let key_var = format!("AGENTS_MCP_RELOAD_DOTENV_KEY_{}", std::process::id());
        assert!(std::env::var_os(&key_var).is_none());
        let config_text = format!(
            "[agents]\nagents_dir = \"{}\"\nhot_reload = false\n\n[providers.direct.reload-test]\napi_key_env = \"{key_var}\"\nbase_url = \"https://example.test/v1\"\n",
            agents_dir.display().to_string().replace('\\', "/")
        );
        write_config(&config_path, &config_text);
        std::fs::write(&dotenv_path, format!("{key_var}=old-key\n")).expect("первый .env");
        let initial = load_dotenv_path(&dotenv_path);
        let provider_env = ProviderEnv {
            process: HashMap::new(),
            dotenv: initial.values,
        };
        let cfg = Config::load_or_default(Some(&config_path)).expect("конфиг");
        let reloader = test_reloader(config_path, cfg, provider_env).await;

        std::fs::write(
            &dotenv_path,
            format!("BEFORE=ok\nBROKEN hunter2\n{key_var}=new-key\n"),
        )
        .expect("новый .env");
        let report = reloader
            .reload_with_dotenv(load_dotenv_path(&dotenv_path))
            .await;
        assert_eq!(report.errors.len(), 1, "ожидалась одна ошибка: {report:?}");
        assert!(report.errors[0].contains("строке 2"));
        assert!(!report.errors[0].contains("hunter2"));
        assert!(
            report
                .applied
                .iter()
                .any(|line| line.contains("[providers]") && line.contains("reload-test")),
            "смена ключа должна пересобрать провайдера: {report:?}"
        );
        assert!(std::env::var_os(&key_var).is_none());
        let state = reloader.state.lock().await;
        assert_eq!(
            provider_env_var(&state.provider_env, &key_var).as_deref(),
            Some("new-key")
        );
        drop(state);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn watcher_start_failure_is_reported() {
        let dir = temp_dir("watcher-error");
        let config_path = dir.join("agents-mcp.toml");
        let missing_dir = dir.join("missing-agents");
        let config_text = |hot_reload: bool| {
            format!(
                "[agents]\nagents_dir = \"{}\"\nhot_reload = {hot_reload}\n",
                missing_dir.display().to_string().replace('\\', "/")
            )
        };
        write_config(&config_path, &config_text(false));
        let cfg = Config::load_or_default(Some(&config_path)).expect("конфиг");
        let provider_env = ProviderEnv {
            process: HashMap::new(),
            dotenv: HashMap::new(),
        };
        let reloader = test_reloader(config_path.clone(), cfg, provider_env).await;

        write_config(&config_path, &config_text(true));
        let report = reloader.reload().await;
        assert!(
            !report.applied.iter().any(|line| line.contains("наблюдатель agents/ запущен")),
            "неуспешный запуск не должен быть applied: {report:?}"
        );
        assert!(
            report.errors.iter().any(|line| line.contains("наблюдатель agents/ не запущен")),
            "ошибка запуска должна попасть в отчёт: {report:?}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn warning_generated_when_allowed_root_covers_service_path() {
        let dir = temp_dir("service-warning");
        let service_dir = dir.join("service");
        std::fs::create_dir(&service_dir).expect("служебный каталог");
        let paths = ServicePaths {
            entries: vec![ServicePath {
                name: "agents_dir",
                path: service_dir,
                write_only: false,
            }],
        };

        let warnings = service_path_overlap_warnings(&[dir.clone()], &paths);
        assert_eq!(warnings.len(), 1, "ожидалось предупреждение: {warnings:?}");
        assert!(warnings[0].contains("allowed_roots"));
        assert!(warnings[0].contains("agents_dir"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn service_paths_cover_dotenv_candidates_from_cwd_up() {
        let paths = ServicePaths::from_config(None, &Config::default());
        let cwd = std::env::current_dir().expect("текущий каталог");
        for dir in cwd.ancestors() {
            assert!(
                paths.entries.iter().any(|e| e.path == dir.join(".env") && !e.write_only),
                "не защищён {}",
                dir.join(".env").display()
            );
        }
    }

    #[tokio::test]
    async fn direct_provider_changes_track_fingerprints() {
        let name = "reload-test-vendor";
        let key_var = "AGENTS_MCP_RELOAD_TEST_KEY";
        std::env::set_var(key_var, "key-1");
        let env = ProviderEnv::capture(HashMap::new());
        let one = format!(
            "[direct.{name}]\napi_key_env = \"{key_var}\"\nbase_url = \"https://one.example/v1\"\n\n[direct.{name}.prices.\"test-model\"]\ninput = 0.3\noutput = 1.2\n"
        );
        let cfg: ProvidersConfig = toml::from_str(&one).expect("конфиг разобран");

        let (set1, ch1) = build_providers(&cfg, None, &env).await;
        assert!(set1.providers.contains_key("mock"), "mock есть всегда");
        assert!(set1.providers.contains_key(name), "прямой провайдер собран");
        assert_eq!(set1.statuses["mock"].status, "registered");
        assert_eq!(set1.statuses[name].status, "registered");
        assert_eq!(ch1.added, vec![name.to_string()]);

        // Ничего не изменилось — тот же Arc (семафор и соединения сохраняются).
        let (set2, ch2) = build_providers(&cfg, Some(&set1), &env).await;
        assert!(ch2.is_empty(), "изменений нет: {ch2:?}");
        assert!(Arc::ptr_eq(&set1.providers[name], &set2.providers[name]));

        // Изменена цена — провайдер в changed и это новый Arc.
        let two = one.replace("input = 0.3", "input = 0.4");
        let cfg2: ProvidersConfig = toml::from_str(&two).expect("конфиг разобран");
        let (set3, ch3) = build_providers(&cfg2, Some(&set2), &env).await;
        assert_eq!(ch3.changed, vec![name.to_string()]);
        assert!(!Arc::ptr_eq(&set2.providers[name], &set3.providers[name]));

        // Изменён base_url — провайдер в changed и это новый Arc.
        let base_changed = two.replace("https://one.example/v1", "https://two.example/v1");
        let cfg_base: ProvidersConfig = toml::from_str(&base_changed).expect("конфиг разобран");
        let (set_base, ch_base) = build_providers(&cfg_base, Some(&set3), &env).await;
        assert_eq!(ch_base.changed, vec![name.to_string()]);
        assert!(!Arc::ptr_eq(&set3.providers[name], &set_base.providers[name]));

        // Изменено значение ключа — тоже changed.
        std::env::set_var(key_var, "key-2");
        let changed_env = ProviderEnv::capture(HashMap::new());
        let (set4, ch4) = build_providers(&cfg2, Some(&set3), &changed_env).await;
        assert_eq!(ch4.changed, vec![name.to_string()]);

        // Секция удалена — removed.
        let empty: ProvidersConfig = toml::from_str("").expect("пустой конфиг разобран");
        let (set5, ch5) = build_providers(&empty, Some(&set4), &changed_env).await;
        assert_eq!(ch5.removed, vec![name.to_string()]);
        assert!(!set5.providers.contains_key(name));
        assert!(set5.providers.contains_key("mock"));
    }

    #[tokio::test]
    async fn direct_proxy_expands_from_dotenv_and_changes_fingerprint() {
        let name = "proxy-env-test";
        let raw = format!(
            "[direct.{name}]\napi_key_env = \"API_KEY\"\nbase_url = \"https://api.example/v1\"\nproxy = \"http://${{PROXY_HOST}}:3128\"\n"
        );
        let cfg: ProvidersConfig = toml::from_str(&raw).expect("конфиг разобран");
        let env_one = ProviderEnv {
            process: HashMap::new(),
            dotenv: HashMap::from([
                ("API_KEY".to_string(), "key".to_string()),
                ("PROXY_HOST".to_string(), "one".to_string()),
            ]),
        };
        assert_eq!(
            expand_provider_proxy(&cfg.direct[name].proxy, &env_one).unwrap().as_deref(),
            Some("http://one:3128")
        );
        let (first, _) = build_providers(&cfg, None, &env_one).await;
        assert!(first.providers.contains_key(name));

        let missing = ProviderEnv {
            process: HashMap::new(),
            dotenv: HashMap::from([("API_KEY".to_string(), "key".to_string())]),
        };
        let (skipped, _) = build_providers(&cfg, None, &missing).await;
        assert!(!skipped.providers.contains_key(name));
        assert!(!skipped.statuses.contains_key(name));
        assert!(skipped.skipped.contains(name));

        let env_two = ProviderEnv {
            process: HashMap::new(),
            dotenv: HashMap::from([
                ("API_KEY".to_string(), "key".to_string()),
                ("PROXY_HOST".to_string(), "two".to_string()),
            ]),
        };
        let (second, changes) = build_providers(&cfg, Some(&first), &env_two).await;
        assert_eq!(changes.changed, vec![name.to_string()]);
        assert!(!Arc::ptr_eq(&first.providers[name], &second.providers[name]));
    }

    #[tokio::test]
    async fn named_http_providers_expand_proxy_and_skip_when_variable_is_missing() {
        let raw = "[openrouter]\napi_key_env = \"OR_KEY\"\nproxy = \"http://${PROXY_HOST}:3128\"\n\
                   [anthropic]\napi_key_env = \"ANTHROPIC_KEY\"\nproxy = \"http://${PROXY_HOST}:3128\"\n";
        let cfg: ProvidersConfig = toml::from_str(raw).expect("конфиг разобран");
        let env_one = ProviderEnv {
            process: HashMap::new(),
            dotenv: HashMap::from([
                ("OR_KEY".to_string(), "or-key".to_string()),
                ("ANTHROPIC_KEY".to_string(), "anthropic-key".to_string()),
                ("PROXY_HOST".to_string(), "one".to_string()),
            ]),
        };
        let (first, _) = build_providers(&cfg, None, &env_one).await;
        assert!(first.providers.contains_key("openrouter"));
        assert!(first.providers.contains_key("anthropic"));

        let missing = ProviderEnv {
            process: HashMap::new(),
            dotenv: HashMap::from([
                ("OR_KEY".to_string(), "or-key".to_string()),
                ("ANTHROPIC_KEY".to_string(), "anthropic-key".to_string()),
            ]),
        };
        let (skipped, _) = build_providers(&cfg, None, &missing).await;
        for name in ["openrouter", "anthropic"] {
            assert!(!skipped.providers.contains_key(name));
            assert!(!skipped.statuses.contains_key(name));
        }

        let mut env_two = env_one;
        env_two.dotenv.insert("PROXY_HOST".to_string(), "two".to_string());
        let (_, changes) = build_providers(&cfg, Some(&first), &env_two).await;
        assert_eq!(changes.changed, vec!["anthropic".to_string(), "openrouter".to_string()]);
    }

    #[tokio::test]
    async fn direct_anthropic_provider_tracks_prompt_cache_fingerprint() {
        let name = "reload-test-anthropic";
        let key_var = "AGENTS_MCP_RELOAD_TEST_ANTHROPIC_KEY";
        std::env::set_var(key_var, "key");
        let env = ProviderEnv::capture(HashMap::new());
        let raw = format!(
            "[direct.{name}]\napi = \"anthropic\"\napi_key_env = \"{key_var}\"\nbase_url = \"https://api.example.com/anthropic\"\nprompt_cache = false\n"
        );
        let cfg: ProvidersConfig = toml::from_str(&raw).expect("конфиг разобран");
        let (first, changes) = build_providers(&cfg, None, &env).await;
        assert!(first.providers.contains_key(name));
        assert_eq!(first.statuses[name].status, "registered");
        assert_eq!(changes.added, vec![name.to_string()]);

        let cached = raw.replace("prompt_cache = false", "prompt_cache = true");
        let cfg: ProvidersConfig = toml::from_str(&cached).expect("конфиг разобран");
        let (second, changes) = build_providers(&cfg, Some(&first), &env).await;
        assert_eq!(changes.changed, vec![name.to_string()]);
        assert!(!Arc::ptr_eq(&first.providers[name], &second.providers[name]));
        std::env::remove_var(key_var);
    }

    #[tokio::test]
    async fn failed_cli_doctor_is_saved_as_down_but_provider_stays_registered() {
        let dir = temp_dir("missing_cli");
        let missing = dir.join("definitely-missing-cli");
        let cfg: ProvidersConfig = toml::from_str(&format!(
            "[claude_cli]\nexecutable = '{}'\n",
            missing.display()
        ))
        .expect("конфиг claude-cli");

        let env = ProviderEnv::capture(HashMap::new());
        let (set, _) = build_providers(&cfg, None, &env).await;
        assert!(set.providers.contains_key("claude-cli"));
        assert_eq!(set.statuses["claude-cli"].status, "down");
        assert!(set.statuses["claude-cli"].message.as_deref().is_some_and(|m| !m.is_empty()));

        let (reused, changes) = build_providers(&cfg, Some(&set), &env).await;
        assert!(changes.is_empty());
        assert_eq!(reused.statuses["claude-cli"], set.statuses["claude-cli"]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn successful_cli_doctor_is_saved_as_ok() {
        let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
        let cfg: ProvidersConfig = toml::from_str(&format!(
            "[claude_cli]\nexecutable = '{}'\n",
            rustc
        ))
        .expect("конфиг claude-cli");

        let env = ProviderEnv::capture(HashMap::new());
        let (set, _) = build_providers(&cfg, None, &env).await;
        assert!(set.providers.contains_key("claude-cli"));
        assert_eq!(set.statuses["claude-cli"].status, "ok");
        assert!(set.statuses["claude-cli"].message.is_none());
    }

    #[tokio::test]
    async fn reload_reports_and_applies_changes() {
        let dir = temp_dir("reloader");
        let config_path = dir.join("agents-mcp.toml");
        let agents_dir = dir.join("agents");
        std::fs::create_dir_all(&agents_dir).expect("каталог агентов");

        write_config(
            &config_path,
            "[server]\nport = 18025\n\n[fs]\nallowed_roots = [\"first-root\"]\n",
        );
        let cfg = Config::load_or_default(Some(config_path.as_path())).expect("конфиг");
        let prev_roots = cfg.fs.allowed_roots.clone();
        let provider_env = ProviderEnv::capture(HashMap::new());
        let (providers, _) = build_providers(&cfg.providers, None, &provider_env).await;

        let registry = Arc::new(Registry::load(agents_dir).expect("реестр"));
        let store: Arc<dyn crate::store::Store> =
            Arc::new(SqliteStore::open(std::path::Path::new(":memory:")).expect("sqlite в памяти"));
        let force_override: SharedOverride =
            Arc::new(std::sync::RwLock::new(ModelOverride::default()));
        let runtime = Arc::new(Runtime::new(
            store,
            registry.clone(),
            HashMap::new(),
            crate::skills::SkillsClient::new(None),
            force_override.clone(),
            dir.join("runs"),
            "test:1".to_string(),
            cfg.agents.default_timeout_sec,
        ));
        let reloader = ConfigReloader::new(
            Some(config_path.clone()),
            cfg,
            providers,
            provider_env,
            registry,
            runtime,
            force_override,
        );

        // (а) Файл с синтаксической ошибкой — не применено ничего.
        write_config(&config_path, "[server]\nport = \"не-число\"\n");
        let rep = reloader.reload().await;
        assert!(!rep.errors.is_empty(), "ожидалась ошибка разбора: {rep:?}");
        assert!(rep.applied.is_empty(), "применять нечего: {rep:?}");
        let roots_now = reloader
            .fs_roots()
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        assert_eq!(roots_now, prev_roots, "корни прежние");

        // (б) Смена [server] port — только «требует перезапуска».
        write_config(
            &config_path,
            "[server]\nport = 18026\n\n[fs]\nallowed_roots = [\"first-root\"]\n",
        );
        let rep = reloader.reload().await;
        assert!(rep.applied.is_empty(), "применять нечего: {rep:?}");
        assert!(
            rep.restart_required.iter().any(|l| l.contains("[server]")),
            "ожидалась строка [server]: {rep:?}"
        );
        assert!(rep.errors.is_empty(), "ошибок быть не должно: {rep:?}");

        // (в) Смена [fs] allowed_roots — применено на лету.
        write_config(
            &config_path,
            "[server]\nport = 18026\n\n[fs]\nallowed_roots = [\"first-root\", \"second-root\"]\n",
        );
        let rep = reloader.reload().await;
        assert!(!rep.applied.is_empty(), "ожидалось применение: {rep:?}");
        assert!(rep.errors.is_empty(), "ошибок быть не должно: {rep:?}");
        let roots_now = reloader
            .fs_roots()
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        assert_eq!(
            roots_now,
            vec![dir.join("first-root"), dir.join("second-root")]
        );

        // (г) Смена runs_dir обновляет запретный список для записи fs_*.
        write_config(
            &config_path,
            "[server]\nport = 18026\n\n[storage]\nruns_dir = \"new-runs\"\n\n[fs]\nallowed_roots = [\"first-root\", \"second-root\"]\n",
        );
        let rep = reloader.reload().await;
        assert!(rep.errors.is_empty(), "ошибок быть не должно: {rep:?}");
        let paths = reloader
            .service_paths()
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        assert!(paths.entries.iter().any(|entry| {
            entry.name == "runs_dir" && entry.path.ends_with("new-runs") && entry.write_only
        }));

        // (д) Повторная перечитка без изменений файла — тишина.
        let rep = reloader.reload().await;
        assert!(rep.applied.is_empty(), "применять нечего: {rep:?}");
        assert!(rep.errors.is_empty(), "ошибок быть не должно: {rep:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
