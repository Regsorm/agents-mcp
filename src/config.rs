//! Парсинг конфига `configs/agents-mcp.toml`.
//!
//! Все секции опциональны: если файл отсутствует или поле не задано,
//! используются значения по умолчанию для текущей платформы.

use std::collections::{BTreeMap, HashMap};
use std::net::IpAddr;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::errors::{AgentsMcpError, Result};

#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub agents: AgentsConfig,
    #[serde(default)]
    pub providers: ProvidersConfig,
    #[serde(default)]
    pub skills: SkillsConfig,
    #[serde(default)]
    pub fs: FsConfig,
    /// Ключи, которые TOML-разбор успешно пропустил как неизвестные.
    /// Поле служебное и обратно в формат конфига не входит.
    #[serde(skip)]
    pub(crate) unknown_keys: Vec<String>,
}

/// Секция [fs] — разрешённые корни для файловых MCP-инструментов (fs_write_file,
/// fs_read_file, fs_mkdir, fs_list_dir). Операции допускаются ТОЛЬКО внутри
/// этих каталогов (защита от записи в системные пути). Эти инструменты дают
/// файловые операции через MCP, поэтому работают на любом провайдере (deepseek/
/// mimo), а не только на claude-cli со встроенными Write/Bash.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FsConfig {
    #[serde(default = "default_fs_roots")]
    pub allowed_roots: Vec<PathBuf>,
}

impl Default for FsConfig {
    fn default() -> Self {
        Self {
            allowed_roots: default_fs_roots(),
        }
    }
}

#[cfg(windows)]
fn default_fs_roots() -> Vec<PathBuf> {
    vec![PathBuf::from("C:/Temp")]
}

#[cfg(not(windows))]
fn default_fs_roots() -> Vec<PathBuf> {
    vec![PathBuf::from("/tmp")]
}

/// Секция [skills] — источник навыков для push-инъекции в промпты.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct SkillsConfig {
    /// URL внешнего MCP-сервиса библиотеки навыков: из него читается таблица
    /// skills (POST tools/call). None — инъекция навыков выключена.
    #[serde(default)]
    pub rag_query_url: Option<String>,
}

/// Секция [providers.*] в TOML. Mock — всегда включён в коде.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct ProvidersConfig {
    #[serde(default)]
    pub openrouter: Option<ProviderEntry>,
    #[serde(default)]
    pub anthropic: Option<ProviderEntry>,
    #[serde(default)]
    pub claude_cli: Option<ClaudeCliConfig>,
    #[serde(default)]
    pub codex_cli: Option<CodexCliConfig>,
    /// Пул «прямых» HTTP-подключений. Ключ map = имя провайдера, под которым
    /// агент его выбирает (`provider = "mimo"`). Подключения различаются
    /// `base_url` и `api_key_env`, а вид API задаётся полем `api`: OpenAI-
    /// совместимый клиент `OpenRouterProvider` или `AnthropicProvider`. Это один
    /// универсальный механизм на любое число подключений: добавить нового
    /// поставщика = только секция `[providers.direct.<имя>]` в TOML, без правок
    /// кода.
    #[serde(default)]
    pub direct: HashMap<String, ProviderEntry>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ProviderEntry {
    /// Вид HTTP API. Для `[providers.direct.*]` отсутствие ключа означает
    /// OpenAI-совместимый API; именованные секции выбирают вид по своему имени.
    #[serde(default)]
    pub api: Option<ProviderApi>,
    /// Имя env-переменной с API-ключом. Если переменная не задана —
    /// провайдер не регистрируется (warn при старте), агенты с этим
    /// провайдером упадут с UnknownProvider.
    pub api_key_env: String,
    /// Базовый URL API. Если не задан — берётся дефолт.
    #[serde(default)]
    pub base_url: Option<String>,
    /// HTTP-Referer для OpenRouter analytics. Игнорируется Anthropic'ом.
    #[serde(default)]
    pub default_referer: Option<String>,
    /// Адрес сетевого посредника для ЭТОГО провайдера, например
    /// "http://10.0.0.1:3128". Не задан или пуст — как раньше.
    /// Допустимы только схемы http и https: возможность socks в сборку не
    /// включена. Своя проверка нужна потому, что библиотека socks-адрес
    /// принимает молча и потом обращается к этому узлу как к обычному
    /// HTTP-посреднику — то есть говорит с socks-портом на чужом языке, и
    /// получается либо ошибка соединения, либо ожидание до предела времени.
    #[serde(default)]
    pub proxy: Option<String>,
    /// Дополнительные адреса, куда ходить МИМО посредника, через запятую.
    /// Свои сети (петлевые адреса и частные диапазоны) исключаются ВСЕГДА, снять
    /// их этой настройкой нельзя: клиент провайдера ходит не только к модели, но
    /// и к серверам инструментов внутри сети. Не задано или пусто — действует
    /// только обязательное ядро (PROXY_BYPASS_CORE).
    /// ВНИМАНИЕ: библиотека разбирает этот список лениво и ошибок не возвращает —
    /// опечатка в нём не будет замечена, запись просто не сработает.
    #[serde(default)]
    pub proxy_bypass: Option<String>,
    /// Предел одновременно идущих вызовов этого провайдера. `None` — без
    /// ограничения; ожидание места входит в общий срок прогона.
    #[serde(default)]
    pub max_concurrent: Option<u32>,
    /// Пометить системный промпт и последний инструмент для prompt caching.
    /// Действует только для Anthropic Messages API.
    #[serde(default)]
    pub prompt_cache: bool,
    /// Цены моделей в долларах за миллион токенов. Ключ должен точно
    /// совпадать с именем модели из конфига агента (`req.model`).
    #[serde(default)]
    pub prices: BTreeMap<String, ModelPrice>,
}

/// Цена модели в долларах за миллион токенов.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ModelPrice {
    pub input: f64,
    pub output: f64,
    #[serde(default)]
    pub cache_read: Option<f64>,
    #[serde(default)]
    pub cache_write: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderApi {
    Openai,
    Anthropic,
}

/// Конфиг провайдера claude-cli (subprocess через `claude -p`).
///
/// API-ключ не нужен — авторизация через OAuth Claude Code на машине.
/// При старте сервиса делается doctor self-test (`claude --version`); если
/// CLI не найден, провайдер регистрируется со статусом "down" (агенты с
/// provider=claude-cli получат UnknownProvider при invoke).
///
/// Бюджет-чекер по решению пользователя не подключаем (биллинг через
/// подписку Claude Max, контроля изнутри сервиса не делаем). Поле
/// `cost_usd` всё равно сохраняется в `agent_calls` для ретроспективной
/// аналитики.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ClaudeCliConfig {
    /// Путь к бинарнику claude. По умолчанию — "claude" (берётся из PATH).
    #[serde(default = "default_claude_executable")]
    pub executable: String,

    /// Максимум параллельных subprocess-вызовов claude. 1 — sequential
    /// (минимум нагрузки на аккаунт); 2 — компромисс при умеренном трафике.
    #[serde(default = "default_cli_max_concurrent")]
    pub max_concurrent: u32,

    /// Дефолтное значение `--max-turns` если агент не задал своё. Защита от
    /// бесконечных tool-loop'ов.
    #[serde(default = "default_max_turns")]
    pub default_max_turns: u32,

    /// Каталог fake-home для под-агентов (CLAUDE_CONFIG_DIR). Подменяет
    /// user-scope claude-cli: пустой CLAUDE.md, без rules/skills/глобального
    /// .mcp.json, но с .credentials.json для авторизации. Срезает ~350k токенов
    /// на cold-старте под-агента. Путь живёт ЗДЕСЬ (а не в env родителя)
    /// намеренно: конфиг version-controlled и идёт через AGENTS_MCP_CONFIG,
    /// поэтому переживает перенос под supervisor. None — fallback на env
    /// CLAUDE_CONFIG_DIR.
    #[serde(default)]
    pub config_dir: Option<String>,
}

fn default_claude_executable() -> String {
    "claude".into()
}

fn default_cli_max_concurrent() -> u32 {
    2
}

/// Конфиг провайдера codex-cli (subprocess через `codex.exe exec`).
///
/// Авторизация — через профиль в CODEX_HOME (учётные данные ChatGPT/Codex),
/// путь берётся из конфига службы, а не из окружения родителя: тот же приём,
/// что и у claude-cli `config_dir`, — конфиг version-controlled и переживает
/// перенос под supervisor. codex_home должен указывать на собственный профиль
/// службы, а не на профиль чужого процесса.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CodexCliConfig {
    /// Путь к бинарнику codex. По умолчанию — "codex" (берётся из PATH).
    #[serde(default = "default_codex_executable")]
    pub executable: String,

    /// CODEX_HOME — каталог профиля с учётными данными codex.
    pub codex_home: String,

    /// Максимум параллельных subprocess-вызовов codex.
    #[serde(default = "default_cli_max_concurrent")]
    pub max_concurrent: u32,

    /// Прокси для запускаемого процесса codex (HTTP_PROXY/HTTPS_PROXY).
    /// Прямой маршрут до OpenAI бывает закрыт или медленным — тогда процессу
    /// нужен посредник. Служба
    /// поднимается супервизором без прокси в окружении, а задавать его всей
    /// службе нельзя: у неё есть обращения к локальным адресам.
    /// None — процесс идёт напрямую.
    #[serde(default)]
    pub proxy: Option<String>,

    /// Адреса в обход прокси (NO_PROXY) для того же процесса.
    #[serde(default)]
    pub proxy_bypass: Option<String>,
}

fn default_codex_executable() -> String {
    "codex".into()
}

fn default_max_turns() -> u32 {
    8
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: IpAddr,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Список разрешённых Host для /mcp (защита rmcp от DNS-rebinding).
    #[serde(default = "default_allowed_hosts")]
    pub allowed_hosts: Vec<String>,
    /// Имя экземпляра службы в строках вызовов (`agent_calls.instance`).
    /// Не задано — `<имя машины>:<порт>`; по этому имени при старте
    /// закрываются только свои осиротевшие вызовы, поэтому у двух служб на
    /// одной базе оно обязано различаться.
    #[serde(default)]
    pub instance: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            allowed_hosts: default_allowed_hosts(),
            instance: None,
        }
    }
}

impl ServerConfig {
    /// Имя экземпляра службы, как оно пишется в `agent_calls.instance`:
    /// `[server] instance`, а если он не задан (или пуст после trim) —
    /// автоматическое `<имя машины>:<порт>`. На транспорте stdio вместо порта
    /// идёт `stdio`: там экземпляр свой у каждого клиента.
    pub fn instance_name(&self, stdio: bool) -> String {
        if let Some(name) = self
            .instance
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            return if stdio {
                format!("{name}:stdio")
            } else {
                name.to_string()
            };
        }
        let host = sysinfo::System::host_name()
            .filter(|h| !h.trim().is_empty())
            .unwrap_or_else(|| "unknown-host".to_string());
        let tail = if stdio {
            "stdio".to_string()
        } else {
            self.port.to_string()
        };
        format!("{host}:{tail}")
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct StorageConfig {
    /// Каталог для PID-файла и журнала службы (`agents-mcp.logs.db`).
    /// Создаётся программно при старте, если его ещё нет.
    #[serde(default = "default_log_dir")]
    pub log_dir: PathBuf,
    /// Каталог файлов-итогов фоновых вызовов (`agent_run`): на каждый вызов
    /// кладётся конверт `<call_id>-<агент>.json`. Нужен, чтобы забрать итог не
    /// обращаясь к MCP — обычным чтением файла — и чтобы клиент мог дождаться
    /// появления этого файла своими средствами, не держа чат.
    #[serde(default = "default_runs_dir")]
    pub runs_dir: PathBuf,
    /// Файл встроенного SQLite: используется, когда `task_store_dsn` не задан
    /// (служба создаёт его сама при первом запуске). Относительный путь
    /// резолвится от каталога конфига — как log_dir и runs_dir.
    #[serde(default = "default_sqlite_path")]
    pub sqlite_path: PathBuf,
    /// DSN PG task-store (схема agents_mcp), напр. postgres://user:pass@host:port/db.
    /// Не задан — служба работает на встроенном SQLite (`sqlite_path`).
    #[serde(default)]
    pub task_store_dsn: Option<String>,
    /// Размер пула соединений task-store.
    #[serde(default = "default_task_store_pool")]
    pub task_store_pool: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            log_dir: default_log_dir(),
            runs_dir: default_runs_dir(),
            sqlite_path: default_sqlite_path(),
            task_store_dsn: None,
            task_store_pool: default_task_store_pool(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AgentsConfig {
    /// Каталог реестра агентов (по умолчанию относительный `agents/`).
    #[serde(default = "default_agents_dir")]
    pub agents_dir: PathBuf,
    /// Hot-reload папки agents/ через notify watcher.
    #[serde(default = "default_true")]
    pub hot_reload: bool,
    #[serde(default = "default_timeout_sec")]
    pub default_timeout_sec: u64,
    /// Тест-override модели на ВСЕХ агентах. Если заданы оба
    /// (`force_provider` + `force_model`) — каждый invoke идёт через этого
    /// провайдера с этой моделью, игнорируя per-agent `[model]`. None (дефолт) —
    /// каждый агент по своему config.toml. Нужен, чтобы прогнать всю платформу
    /// на одной модели (deepseek/mimo/…), а не вперемешку Max/deepseek/mimo.
    /// Заданный только один из двух игнорируется (нужны оба).
    #[serde(default)]
    pub force_provider: Option<String>,
    #[serde(default)]
    pub force_model: Option<String>,
    /// Адреса, разрешённые в перекрытии `mcp.<сервер>.url`. Пусто — умолчание:
    /// схема `http`, хост `127.0.0.1`, путь `/mcp`, порт задан.
    #[serde(default)]
    pub allowed_mcp_urls: Vec<String>,
}

impl Default for AgentsConfig {
    fn default() -> Self {
        Self {
            agents_dir: default_agents_dir(),
            hot_reload: default_true(),
            default_timeout_sec: default_timeout_sec(),
            force_provider: None,
            force_model: None,
            allowed_mcp_urls: Vec::new(),
        }
    }
}

impl Config {
    /// Загрузить конфиг из файла, либо вернуть дефолты если путь не указан.
    /// Применяется fallback: `--config` → env `AGENTS_MCP_CONFIG` → дефолт.
    pub fn load_or_default(explicit: Option<&Path>) -> Result<Self> {
        let path = explicit
            .map(|p| p.to_path_buf())
            .or_else(|| std::env::var_os("AGENTS_MCP_CONFIG").map(PathBuf::from));

        // Сначала файл либо значения по умолчанию, затем единообразно env.
        let mut cfg = if let Some(path) = path.as_deref() {
            let raw = std::fs::read_to_string(path).map_err(AgentsMcpError::ConfigRead)?;
            let mut cfg: Self = toml::from_str(&raw).map_err(AgentsMcpError::ConfigParse)?;
            cfg.unknown_keys =
                unknown_main_config_keys(&raw).map_err(AgentsMcpError::ConfigParse)?;
            cfg
        } else {
            Self::default()
        };

        // DSN доски задач — из env, чтобы не держать пароль PG в
        // version-controlled конфиге (как api_key_env у провайдеров). Если
        // переменная задана и непуста — переопределяет значение из toml.
        if let Ok(dsn) = std::env::var("AGENTS_MCP_TASK_STORE_DSN") {
            if !dsn.is_empty() {
                cfg.storage.task_store_dsn = Some(dsn);
            }
        }

        // Относительные пути считаются от каталога конфига; без файла — от cwd.
        let base_dir = config_base_dir(path.as_deref())?;
        resolve_relative_paths(&mut cfg, &base_dir);
        resolve_relative_sqlite_dsn(&mut cfg.storage.task_store_dsn, &base_dir);

        // При старте tracing ещё не установлен, поэтому main повторит сохранённые
        // предупреждения сразу после init_tracing. При перечитке они видны здесь.
        cfg.warn_unknown_keys();

        Ok(cfg)
    }

    pub(crate) fn warn_unknown_keys(&self) {
        for key in &self.unknown_keys {
            tracing::warn!(
                config_key = %key,
                "неизвестный ключ в agents-mcp.toml — значение проигнорировано"
            );
        }
    }
}

fn unknown_keys_in_table(
    table: &toml::map::Map<String, toml::Value>,
    prefix: &str,
    known: &[&str],
    out: &mut Vec<String>,
) {
    for key in table.keys() {
        if !known.contains(&key.as_str()) {
            out.push(format!("{prefix}{key}"));
        }
    }
}

/// Неизвестные поля не мешают загрузке главного конфига, но сохраняются для
/// предупреждения в журнале с полным путём ключа.
fn unknown_main_config_keys(raw: &str) -> std::result::Result<Vec<String>, toml::de::Error> {
    let value: toml::Value = toml::from_str(raw)?;
    let Some(root) = value.as_table() else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    unknown_keys_in_table(
        root,
        "",
        &["server", "storage", "agents", "providers", "skills", "fs"],
        &mut out,
    );
    for (section, known) in [
        ("server", &["host", "port", "allowed_hosts", "instance"][..]),
        (
            "storage",
            &[
                "log_dir",
                "runs_dir",
                "sqlite_path",
                "task_store_dsn",
                "task_store_pool",
            ],
        ),
        (
            "agents",
            &[
                "agents_dir",
                "hot_reload",
                "default_timeout_sec",
                "force_provider",
                "force_model",
                "allowed_mcp_urls",
            ],
        ),
        (
            "providers",
            &[
                "openrouter",
                "anthropic",
                "claude_cli",
                "codex_cli",
                "direct",
            ],
        ),
        ("skills", &["rag_query_url"]),
        ("fs", &["allowed_roots"]),
    ] {
        if let Some(table) = root.get(section).and_then(toml::Value::as_table) {
            unknown_keys_in_table(table, &format!("{section}."), known, &mut out);
        }
    }

    let provider_fields = &[
        "api",
        "api_key_env",
        "base_url",
        "default_referer",
        "proxy",
        "proxy_bypass",
        "max_concurrent",
        "prompt_cache",
        "prices",
    ];
    if let Some(providers) = root.get("providers").and_then(toml::Value::as_table) {
        for name in ["openrouter", "anthropic"] {
            if let Some(table) = providers.get(name).and_then(toml::Value::as_table) {
                unknown_keys_in_table(
                    table,
                    &format!("providers.{name}."),
                    provider_fields,
                    &mut out,
                );
                if let Some(prices) = table.get("prices").and_then(toml::Value::as_table) {
                    for (model, value) in prices {
                        if let Some(price) = value.as_table() {
                            unknown_keys_in_table(
                                price,
                                &format!("providers.{name}.prices.{model}."),
                                &["input", "output", "cache_read", "cache_write"],
                                &mut out,
                            );
                        }
                    }
                }
            }
        }
        if let Some(table) = providers.get("claude_cli").and_then(toml::Value::as_table) {
            unknown_keys_in_table(
                table,
                "providers.claude_cli.",
                &[
                    "executable",
                    "max_concurrent",
                    "default_max_turns",
                    "config_dir",
                ],
                &mut out,
            );
        }
        if let Some(table) = providers.get("codex_cli").and_then(toml::Value::as_table) {
            unknown_keys_in_table(
                table,
                "providers.codex_cli.",
                &[
                    "executable",
                    "codex_home",
                    "max_concurrent",
                    "proxy",
                    "proxy_bypass",
                ],
                &mut out,
            );
        }
        if let Some(direct) = providers.get("direct").and_then(toml::Value::as_table) {
            for (name, value) in direct {
                if let Some(table) = value.as_table() {
                    unknown_keys_in_table(
                        table,
                        &format!("providers.direct.{name}."),
                        provider_fields,
                        &mut out,
                    );
                    if let Some(prices) = table.get("prices").and_then(toml::Value::as_table) {
                        for (model, value) in prices {
                            if let Some(price) = value.as_table() {
                                unknown_keys_in_table(
                                    price,
                                    &format!("providers.direct.{name}.prices.{model}."),
                                    &["input", "output", "cache_read", "cache_write"],
                                    &mut out,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    out.sort();
    Ok(out)
}

fn config_base_dir(path: Option<&Path>) -> Result<PathBuf> {
    let parent = path.and_then(Path::parent).unwrap_or_else(|| Path::new(""));
    if parent.is_absolute() {
        Ok(parent.to_path_buf())
    } else {
        Ok(std::env::current_dir()
            .map_err(AgentsMcpError::ConfigRead)?
            .join(parent))
    }
}

fn resolve_relative_paths(cfg: &mut Config, base_dir: &Path) {
    for path in [
        &mut cfg.storage.log_dir,
        &mut cfg.storage.runs_dir,
        &mut cfg.storage.sqlite_path,
        &mut cfg.agents.agents_dir,
    ] {
        if path.is_relative() {
            *path = base_dir.join(&*path);
        }
    }
    for root in &mut cfg.fs.allowed_roots {
        if root.is_relative() {
            *root = base_dir.join(&*root);
        }
    }
}

fn resolve_relative_sqlite_dsn(dsn: &mut Option<String>, base_dir: &Path) {
    let Some(value) = dsn.as_mut() else {
        return;
    };
    let Some(path) = value.strip_prefix("sqlite://") else {
        return;
    };
    let path = Path::new(path);
    if path.is_relative() {
        *value = format!("sqlite://{}", base_dir.join(path).display());
    }
}

fn default_host() -> IpAddr {
    "127.0.0.1".parse().unwrap()
}

fn default_port() -> u16 {
    8025
}

fn default_allowed_hosts() -> Vec<String> {
    vec!["localhost".into(), "127.0.0.1".into(), "::1".into()]
}

#[cfg(windows)]
fn default_log_dir() -> PathBuf {
    PathBuf::from("C:/agents-mcp/logs")
}

#[cfg(not(windows))]
fn default_log_dir() -> PathBuf {
    PathBuf::from("logs")
}

#[cfg(windows)]
fn default_runs_dir() -> PathBuf {
    PathBuf::from("C:/agents-mcp/runs")
}

#[cfg(not(windows))]
fn default_runs_dir() -> PathBuf {
    PathBuf::from("runs")
}

#[cfg(windows)]
fn default_sqlite_path() -> PathBuf {
    PathBuf::from("C:/agents-mcp/data/agents-mcp.sqlite")
}

#[cfg(not(windows))]
fn default_sqlite_path() -> PathBuf {
    PathBuf::from("agents-mcp.db")
}

fn default_task_store_pool() -> usize {
    8
}

fn default_agents_dir() -> PathBuf {
    PathBuf::from("agents")
}

fn default_true() -> bool {
    true
}

fn default_timeout_sec() -> u64 {
    120
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        name: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(name: &'static str, value: Option<&str>) -> Self {
            let previous = std::env::var_os(name);
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
            Self { name, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(value) => std::env::set_var(self.name, value),
                None => std::env::remove_var(self.name),
            }
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("системное время")
            .as_nanos();
        std::env::temp_dir().join(format!("agents-mcp-config-{tag}-{nanos}"))
    }

    #[test]
    fn proxy_fields_are_read_from_config() {
        let raw = r#"
[providers.openrouter]
api_key_env = "OPENROUTER_API_KEY"
proxy = "http://10.0.0.1:3128"
proxy_bypass = "corp.local"
"#;
        let cfg: Config = toml::from_str(raw).expect("конфиг разбирается");
        let entry = cfg.providers.openrouter.expect("секция openrouter");
        assert_eq!(entry.proxy.as_deref(), Some("http://10.0.0.1:3128"));
        assert_eq!(entry.proxy_bypass.as_deref(), Some("corp.local"));
    }

    #[test]
    fn provider_api_and_prompt_cache_are_read_from_config() {
        let raw = r#"
[providers.direct.vendor]
api = "anthropic"
api_key_env = "VENDOR_KEY"
prompt_cache = true
"#;
        let cfg: Config = toml::from_str(raw).expect("конфиг разбирается");
        let entry = &cfg.providers.direct["vendor"];
        assert_eq!(entry.api, Some(ProviderApi::Anthropic));
        assert!(entry.prompt_cache);
        assert!(unknown_main_config_keys(raw).unwrap().is_empty());
    }

    #[test]
    fn provider_prices_are_read_from_config() {
        let raw = r#"
[providers.direct.deepseek]
api_key_env = "DEEPSEEK_API_KEY"

[providers.direct.deepseek.prices."deepseek-flash"]
input = 0.3
output = 1.2
cache_read = 0.006

[providers.direct.deepseek.prices."local-model"]
input = 2
output = 4
"#;
        let cfg: Config = toml::from_str(raw).expect("цены разбираются");
        let prices = &cfg.providers.direct["deepseek"].prices;
        assert_eq!(prices["deepseek-flash"].input, 0.3);
        assert_eq!(prices["deepseek-flash"].output, 1.2);
        assert_eq!(prices["deepseek-flash"].cache_read, Some(0.006));
        assert_eq!(prices["deepseek-flash"].cache_write, None);
        assert_eq!(prices["local-model"].input, 2.0);
        assert_eq!(prices["local-model"].output, 4.0);
        assert_eq!(prices["local-model"].cache_read, None);
        assert_eq!(prices["local-model"].cache_write, None);
        assert!(unknown_main_config_keys(raw).unwrap().is_empty());
    }

    #[test]
    fn unknown_price_key_is_reported_with_full_path() {
        let raw = r#"
[providers.direct.deepseek]
api_key_env = "DEEPSEEK_API_KEY"

[providers.direct.deepseek.prices."deepseek-flash"]
input = 0.3
output = 1.2
cache_hit = 0.006
"#;
        assert_eq!(
            unknown_main_config_keys(raw).unwrap(),
            vec!["providers.direct.deepseek.prices.deepseek-flash.cache_hit"]
        );
    }

    #[test]
    fn direct_provider_defaults_to_openai_without_prompt_cache() {
        let cfg: ProvidersConfig =
            toml::from_str("[direct.vendor]\napi_key_env = \"VENDOR_KEY\"\n")
                .expect("конфиг разбирается");
        let entry = &cfg.direct["vendor"];
        assert_eq!(
            entry.api.unwrap_or(ProviderApi::Openai),
            ProviderApi::Openai
        );
        assert!(!entry.prompt_cache);
    }

    #[test]
    fn unknown_provider_api_is_rejected() {
        let error = toml::from_str::<ProvidersConfig>(
            "[direct.vendor]\napi = \"messages\"\napi_key_env = \"VENDOR_KEY\"\n",
        )
        .expect_err("неизвестный вид API обязан отклонить конфиг");
        assert!(error.to_string().contains("messages"));
    }

    #[test]
    fn provider_max_concurrent_is_optional() {
        let cfg: Config =
            toml::from_str("[providers.openrouter]\napi_key_env = \"KEY\"\nmax_concurrent = 1\n")
                .unwrap();
        assert_eq!(cfg.providers.openrouter.unwrap().max_concurrent, Some(1));

        let cfg: Config =
            toml::from_str("[providers.openrouter]\napi_key_env = \"KEY\"\n").unwrap();
        assert_eq!(cfg.providers.openrouter.unwrap().max_concurrent, None);
    }

    #[test]
    fn runs_dir_has_default_and_is_read_from_config() {
        // Поле новое: прежние конфиги без него обязаны читаться (дефолт), а
        // заданное значение — доезжать до StorageConfig.
        let cfg: Config = toml::from_str("[storage]\nlog_dir = \"C:/x/logs\"\n")
            .expect("конфиг без runs_dir разбирается");
        assert_eq!(cfg.storage.runs_dir, default_runs_dir());

        let cfg: Config = toml::from_str("[storage]\nruns_dir = \"C:/x/runs\"\n")
            .expect("конфиг с runs_dir разбирается");
        assert_eq!(cfg.storage.runs_dir, PathBuf::from("C:/x/runs"));
    }

    #[test]
    fn sqlite_path_has_default_and_is_read_from_config() {
        // Поле новое: прежние конфиги без него обязаны читаться (дефолт), а
        // заданное относительное значение — доезжать до StorageConfig и
        // резолвиться от каталога конфига (как log_dir и runs_dir).
        let cfg: Config = toml::from_str("[storage]\nlog_dir = \"C:/x/logs\"\n")
            .expect("конфиг без sqlite_path разбирается");
        assert_eq!(cfg.storage.sqlite_path, default_sqlite_path());

        let cfg: Config = toml::from_str("[storage]\nsqlite_path = \"data/agents.sqlite\"\n")
            .expect("конфиг с sqlite_path разбирается");
        assert_eq!(cfg.storage.sqlite_path, PathBuf::from("data/agents.sqlite"));
        assert!(cfg.storage.sqlite_path.is_relative());
        assert_eq!(
            PathBuf::from("C:/cfg").join(&cfg.storage.sqlite_path),
            PathBuf::from("C:/cfg/data/agents.sqlite")
        );
    }

    #[test]
    fn config_without_proxy_fields_still_reads() {
        // Прежние конфиги обязаны читаться по-старому: поля необязательные.
        let raw = r#"
[providers.openrouter]
api_key_env = "OPENROUTER_API_KEY"
"#;
        let cfg: Config = toml::from_str(raw).expect("конфиг разбирается");
        let entry = cfg.providers.openrouter.expect("секция openrouter");
        assert!(entry.proxy.is_none());
        assert!(entry.proxy_bypass.is_none());
    }

    #[test]
    fn dsn_environment_applies_without_config_file() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _config = EnvVarGuard::set("AGENTS_MCP_CONFIG", None);
        let _dsn = EnvVarGuard::set(
            "AGENTS_MCP_TASK_STORE_DSN",
            Some("postgres://example.invalid/agents"),
        );

        let cfg = Config::load_or_default(None).expect("значения по умолчанию");
        assert_eq!(
            cfg.storage.task_store_dsn.as_deref(),
            Some("postgres://example.invalid/agents")
        );
    }

    #[test]
    fn unknown_main_keys_are_reported_without_rejecting_config() {
        let raw = r#"
[fs]
allowed_root = ["work"]

[providers.direct.vendor]
api_key_env = "VENDOR_KEY"
unexpected = true
"#;
        let dir = temp_dir("unknown-keys");
        std::fs::create_dir_all(&dir).expect("временный каталог");
        let config_path = dir.join("agents-mcp.toml");
        std::fs::write(&config_path, raw).expect("запись конфига");

        let cfg = Config::load_or_default(Some(&config_path)).expect("конфиг разбирается");
        assert_eq!(cfg.fs.allowed_roots, default_fs_roots());
        assert_eq!(
            cfg.unknown_keys,
            vec![
                "fs.allowed_root".to_string(),
                "providers.direct.vendor.unexpected".to_string(),
            ]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn relative_sqlite_dsn_and_allowed_roots_use_config_directory() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _dsn = EnvVarGuard::set("AGENTS_MCP_TASK_STORE_DSN", None);
        let dir = temp_dir("relative-paths");
        std::fs::create_dir_all(&dir).expect("временный каталог");
        let config_path = dir.join("agents-mcp.toml");
        std::fs::write(
            &config_path,
            "[storage]\ntask_store_dsn = \"sqlite://data/x.db\"\n\n[fs]\nallowed_roots = [\"work\"]\n",
        )
        .expect("запись конфига");

        let cfg = Config::load_or_default(Some(&config_path)).expect("загрузка конфига");
        assert_eq!(
            cfg.storage.task_store_dsn,
            Some(format!("sqlite://{}", dir.join("data/x.db").display()))
        );
        assert_eq!(cfg.fs.allowed_roots, vec![dir.join("work")]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(not(windows))]
    #[test]
    fn unix_defaults_are_resolved_from_current_directory() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _config = EnvVarGuard::set("AGENTS_MCP_CONFIG", None);
        let _dsn = EnvVarGuard::set("AGENTS_MCP_TASK_STORE_DSN", None);
        let cwd = std::env::current_dir().expect("текущий каталог");

        let cfg = Config::load_or_default(None).expect("значения по умолчанию");
        assert_eq!(cfg.storage.log_dir, cwd.join("logs"));
        assert_eq!(cfg.storage.runs_dir, cwd.join("runs"));
        assert_eq!(cfg.storage.sqlite_path, cwd.join("agents-mcp.db"));
        assert_eq!(cfg.fs.allowed_roots, vec![PathBuf::from("/tmp")]);
    }

    #[cfg(windows)]
    #[test]
    fn windows_path_defaults_are_unchanged() {
        let cfg = Config::default();
        assert_eq!(cfg.storage.log_dir, PathBuf::from("C:/agents-mcp/logs"));
        assert_eq!(cfg.storage.runs_dir, PathBuf::from("C:/agents-mcp/runs"));
        assert_eq!(
            cfg.storage.sqlite_path,
            PathBuf::from("C:/agents-mcp/data/agents-mcp.sqlite")
        );
        assert_eq!(cfg.fs.allowed_roots, vec![PathBuf::from("C:/Temp")]);
    }

    #[test]
    fn instance_name_defaults_to_host_and_port() {
        // [server] instance не задан: имя собирается как «<имя машины>:<порт>»
        // (на stdio — «<имя машины>:stdio»).
        let cfg = ServerConfig::default();
        let name = cfg.instance_name(false);
        assert!(name.ends_with(":8025"), "не порт в хвосте: {name}");
        assert!(name.len() > ":8025".len(), "нет имени машины: {name}");

        let stdio_name = cfg.instance_name(true);
        assert!(
            stdio_name.ends_with(":stdio"),
            "не stdio в хвосте: {stdio_name}"
        );
    }

    #[test]
    fn instance_name_from_config_is_trimmed() {
        let cfg: Config = toml::from_str("[server]\ninstance = \" work-pc \"\n")
            .expect("конфиг с instance разбирается");
        assert_eq!(cfg.server.instance_name(false), "work-pc");
        assert_eq!(cfg.server.instance_name(true), "work-pc:stdio");
    }

    #[test]
    fn empty_instance_falls_back_to_default() {
        let cfg: Config = toml::from_str("[server]\ninstance = \"\"\n")
            .expect("конфиг с пустым instance разбирается");
        assert_eq!(
            cfg.server.instance_name(false),
            ServerConfig::default().instance_name(false)
        );
    }
}
