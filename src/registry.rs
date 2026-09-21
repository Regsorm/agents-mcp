//! Реестр агентов — чтение `agents/<name>/{prompt.md, config.toml}` в память.
//!
//! Холодное чтение при старте и атомарная перечитка через notify watcher.
//! Состояние хранится внутри `RwLock<RegistryInner>`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

/// Загруженный агент: конфиг + варианты prompt.md.
#[derive(Debug, Clone, Serialize)]
pub struct AgentDefinition {
    pub name: String,
    pub config: AgentConfig,
    /// Варианты промптов: "default" → prompt.md, "v2" → prompt.v2.md и т.д.
    #[serde(skip)]
    pub prompts: HashMap<String, String>,
    /// Mtime самого свежего из файлов агента — для кеш-инвалидации и
    /// поля `last_modified` в get_agent_info.
    pub last_modified: chrono::DateTime<chrono::Utc>,
    /// Загруженная JSON-схема ответа (из `response.schema_file`), если задана.
    /// Используется для лёгкой валидации JSON-ответа агента в runtime.
    #[serde(skip)]
    pub schema: Option<serde_json::Value>,
}

/// Архитектурный класс агента.
///
/// - `AgentLoop` — настоящий tool-use loop (`max_turns > 1`, есть `allowed_tools`).
///   Пример: `bsl-reviewer` итеративно читает код → валидирует → возвращается.
/// - `PromptTemplate` — one-shot LLM call без tools (`max_turns = 1` или
///   provider не CLI). Пример: `brief-analyst`, `doc-writer`.
/// - `Orchestrator` — agent-loop с рекурсивным `invoke_agent` (через inline
///   `mcp_config` обратно к agents-mcp). Decision-making на лету, диспетчирует
///   узких под-агентов. Защищён `max_orchestration_depth`.
///
/// Поле — метаданные для клиентов: по `kind` они решают,
/// какие лимиты ставить и можно ли вызывать рекурсивно.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum AgentKind {
    AgentLoop,
    #[default]
    PromptTemplate,
    Orchestrator,
}

/// Содержимое per-agent config.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Архитектурный класс агента. Если в config.toml не задан — `prompt-template`.
    #[serde(default)]
    pub kind: AgentKind,

    pub model: ModelConfig,

    #[serde(default)]
    pub response: ResponseConfig,
    #[serde(default)]
    pub input: InputConfig,
    #[serde(default)]
    pub limits: LimitsConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    /// Секция [execution] — как исполняется вызов агента: какие инструменты
    /// выданы, в каком рабочем каталоге, сколько ходов. Действует у ВСЕХ
    /// провайдеров, не только у claude-cli. Прежнее имя секции `[claude_cli]`
    /// принимается как псевдоним, чтобы старые config.toml продолжали читаться.
    #[serde(default, alias = "claude_cli")]
    pub execution: Option<ExecutionConfig>,
    /// Скоуп навыков для push-инъекции в {{ skills_index }}: список project-slug'ов.
    /// Глобальные навыки добавляются всегда. Пусто — инъекция выключена для агента.
    #[serde(default)]
    pub skill_scope: Vec<String>,
    /// Показывать агенту весь домен 1С: навыки с тегом '1c' в {{ skills_index }},
    /// независимо от project-scope. Для 1С-агентов (кодер, ревьюер, query/data).
    #[serde(default)]
    pub skill_include_1c: bool,
    /// Сколько САМЫХ релевантных навыков подгрузить целиком в {{ skills_bodies }}.
    /// 0 (по умолчанию) — только оглавление в {{ skills_index }}, тело агент
    /// тянет сам через skill_load. >0 — тела кладём в промпт заранее: слабая
    /// локальная модель до правил сама не доходит, а оглавление без доступа к
    /// телу бесполезно. Держать 1-2: системный промпт уходит провайдеру заново
    /// на КАЖДОМ ходу агентного цикла, и каждое тело умножается на число ходов.
    #[serde(default)]
    pub skill_bodies_top: usize,
}

/// Условия исполнения вызова агента: разрешённые и запрещённые инструменты,
/// режим разрешений, встроенный mcp-config, шаблон рабочего каталога, предел
/// ходов. Передаются в LlmRequest.cli_hints через runtime → provider и
/// действуют у всех провайдеров, а не только у claude-cli.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutionConfig {
    /// Список инструментов в allow (через запятую → `--allowed-tools`).
    #[serde(default)]
    pub allowed_tools: Vec<String>,
    /// Список инструментов в deny (`--disallowed-tools`).
    #[serde(default)]
    pub disallowed_tools: Vec<String>,
    /// "default" | "acceptEdits" | "bypassPermissions" | "plan".
    /// Если не задан — используется "default".
    #[serde(default)]
    pub permission_mode: Option<String>,
    /// Tera-шаблон рабочей директории, подставляются поля из input
    /// (`{{sandbox_path}}/{{task_dir}}` → "/sandbox/abc/xyz/").
    /// Если не задан — используется текущая рабочая директория сервиса.
    #[serde(default)]
    pub cwd_template: Option<String>,
    /// Корни файлового доступа fs_* инструментов этого агента. Задан —
    /// действует ВМЕСТО общего [fs].allowed_roots службы, а рабочий каталог
    /// вызова обязан лежать внутри одного из корней. Не задан — действует
    /// общий список службы.
    #[serde(default)]
    pub allowed_roots: Option<Vec<PathBuf>>,
    /// Inline MCP-config JSON для `--mcp-config`. Если задан — добавляется
    /// флаг `--strict-mcp-config` (иначе claude подтянет глобальный ~/.claude/.mcp.json).
    #[serde(default)]
    pub mcp_config: Option<String>,
    /// Per-agent override `--max-turns`. None → берётся
    /// ClaudeCliConfig::default_max_turns.
    #[serde(default)]
    pub max_turns: Option<u32>,
    /// Дополнительные argv для `claude -p`. Используется для редких флагов
    /// типа `--strict-mcp-config`.
    #[serde(default)]
    pub extra_args: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// "openrouter" | "anthropic" | "claude-cli" | "codex-cli" | "mock"
    pub provider: String,
    pub name: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Произвольные поля, которые уходят в тело chat/completions как есть.
    /// Задаются таблицей `[model.extra_body]` в config.toml агента. Нужны для
    /// расширений конкретного сервера, которых нет в общей схеме запроса:
    /// llama-server отключает рассуждения Qwen только через
    /// `chat_template_kwargs = {enable_thinking = false}`.
    #[serde(default)]
    pub extra_body: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseConfig {
    #[serde(default = "default_format")]
    pub format: ResponseFormat,
    #[serde(default)]
    pub schema_file: Option<String>,
    /// Жёсткая проверка ответа против `schema_file`: несоответствие делает
    /// вызов неполным (`incomplete`) и сохраняет сырой ответ рядом с итогом,
    /// а не только пишет предупреждение в журнал.
    #[serde(default)]
    pub schema_strict: bool,
}

impl Default for ResponseConfig {
    fn default() -> Self {
        Self {
            format: ResponseFormat::Text,
            schema_file: None,
            schema_strict: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ResponseFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InputConfig {
    #[serde(default)]
    pub required: Vec<String>,
    #[serde(default)]
    pub optional: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LimitsConfig {
    #[serde(default)]
    pub max_input_tokens: Option<u32>,
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
    /// `None` означает, что применяется `[agents] default_timeout_sec`.
    #[serde(default)]
    pub timeout_sec: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_cache_ttl")]
    pub ttl_sec: u64,
    /// Поля из input, участвующие в построении cache_key. Пустой массив —
    /// используется весь input.
    #[serde(default)]
    pub key_fields: Vec<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ttl_sec: default_cache_ttl(),
            key_fields: vec![],
        }
    }
}

fn default_version() -> String {
    "0.0.1".into()
}

fn default_format() -> ResponseFormat {
    ResponseFormat::Text
}

fn default_cache_ttl() -> u64 {
    86_400
}

/// Реестр загруженных агентов. Потокобезопасный.
pub struct Registry {
    inner: RwLock<RegistryInner>,
}

struct RegistryInner {
    agents_dir: PathBuf,
    agents: HashMap<String, Arc<AgentDefinition>>,
    agents_by_dir: HashMap<PathBuf, Arc<AgentDefinition>>,
    generation: u64,
}

struct LoadedAgents {
    agents: HashMap<String, Arc<AgentDefinition>>,
    agents_by_dir: HashMap<PathBuf, Arc<AgentDefinition>>,
}

impl Registry {
    /// Прочитать `agents/` и собрать реестр. Не считается фатальной ошибкой,
    /// если каталог пуст или отсутствует — лог warn и возврат пустого реестра.
    pub fn load(agents_dir: PathBuf) -> Result<Self> {
        let agents_dir = absolute_agents_dir(&agents_dir)?;
        let loaded = load_all_agents(&agents_dir, None)?;
        info!(
            count = loaded.agents.len(),
            dir = %agents_dir.display(),
            "реестр агентов загружен"
        );
        Ok(Self {
            inner: RwLock::new(RegistryInner {
                agents_dir,
                agents: loaded.agents,
                agents_by_dir: loaded.agents_by_dir,
                generation: 0,
            }),
        })
    }

    /// Количество загруженных агентов.
    pub fn len(&self) -> usize {
        self.inner
            .read()
            .expect("registry RwLock poisoned")
            .agents
            .len()
    }

    /// Список всех агентов (отсортирован по имени).
    pub fn list(&self) -> Vec<Arc<AgentDefinition>> {
        let inner = self.inner.read().expect("registry RwLock poisoned");
        let mut out: Vec<Arc<AgentDefinition>> = inner.agents.values().cloned().collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// Найти агента по имени.
    pub fn get(&self, name: &str) -> Option<Arc<AgentDefinition>> {
        let inner = self.inner.read().expect("registry RwLock poisoned");
        inner.agents.get(name).cloned()
    }

    /// Перечитать каталог `agents/` и атомарно подменить состояние.
    pub fn reload(&self) -> Result<()> {
        let (generation, agents_dir, previous) = self.begin_reload();
        ensure_reloadable_dir(&agents_dir)?;
        let fresh = load_all_agents(&agents_dir, Some(&previous))?;
        let count = self.apply_loaded(generation, agents_dir, fresh)?;
        info!(count, "реестр агентов перезагружен");
        Ok(())
    }

    /// Перечитать ДРУГОЙ каталог агентов и атомарно подменить и его, и набор
    /// агентов — перечитка `[agents] agents_dir` в главном конфиге. При ошибке
    /// состояние не трогаем: прежние каталог и реестр остаются рабочими.
    pub fn reload_from(&self, agents_dir: PathBuf) -> Result<()> {
        let (generation, _, previous) = self.begin_reload();
        let agents_dir = absolute_agents_dir(&agents_dir)?;
        ensure_reloadable_dir(&agents_dir)?;
        let fresh = load_all_agents(&agents_dir, Some(&previous))?;
        let applied_dir = agents_dir.clone();
        let count = self.apply_loaded(generation, agents_dir, fresh)?;
        info!(
            count,
            dir = %applied_dir.display(),
            "реестр агентов перечитан из нового каталога"
        );
        Ok(())
    }

    fn begin_reload(&self) -> (u64, PathBuf, HashMap<PathBuf, Arc<AgentDefinition>>) {
        let mut inner = self.inner.write().expect("registry RwLock poisoned");
        inner.generation = inner.generation.wrapping_add(1);
        (
            inner.generation,
            inner.agents_dir.clone(),
            inner.agents_by_dir.clone(),
        )
    }

    fn apply_loaded(
        &self,
        generation: u64,
        agents_dir: PathBuf,
        fresh: LoadedAgents,
    ) -> Result<usize> {
        let mut inner = self.inner.write().expect("registry RwLock poisoned");
        if inner.generation != generation {
            return Err(anyhow!("результат перечитки агентов устарел и не применён"));
        }
        inner.agents_dir = agents_dir;
        inner.agents = fresh.agents;
        inner.agents_by_dir = fresh.agents_by_dir;
        Ok(inner.agents.len())
    }
}

pub(crate) fn absolute_agents_dir(agents_dir: &Path) -> Result<PathBuf> {
    if agents_dir.is_absolute() {
        Ok(agents_dir.to_path_buf())
    } else {
        Ok(std::env::current_dir()
            .context("получение текущего каталога для agents_dir")?
            .join(agents_dir))
    }
}

fn ensure_reloadable_dir(agents_dir: &Path) -> Result<()> {
    if !agents_dir.exists() {
        return Err(anyhow!(
            "каталог агентов не найден: {}",
            agents_dir.display()
        ));
    }
    if !agents_dir.is_dir() {
        return Err(anyhow!("agents_dir не каталог: {}", agents_dir.display()));
    }
    Ok(())
}

fn load_all_agents(
    agents_dir: &Path,
    previous: Option<&HashMap<PathBuf, Arc<AgentDefinition>>>,
) -> Result<LoadedAgents> {
    let mut agents_by_dir = HashMap::new();
    if !agents_dir.exists() {
        warn!(dir = %agents_dir.display(), "каталог агентов не существует — реестр пуст");
        return Ok(LoadedAgents {
            agents: HashMap::new(),
            agents_by_dir,
        });
    }
    if !agents_dir.is_dir() {
        return Err(anyhow!("agents_dir не каталог: {}", agents_dir.display()));
    }

    let mut entries = Vec::new();
    for entry in std::fs::read_dir(agents_dir)
        .with_context(|| format!("чтение каталога агентов {}", agents_dir.display()))?
    {
        entries.push(entry?.path());
    }
    entries.sort();

    for path in entries {
        if !path.is_dir() {
            continue;
        }
        let dir_name = match path.file_name().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        // Скрытые и служебные папки игнорируем.
        if dir_name.starts_with('.') || dir_name.starts_with('_') {
            continue;
        }

        match load_agent(&path) {
            Ok(def) => {
                agents_by_dir.insert(path, Arc::new(def));
            }
            Err(e) => {
                if let Some(previous) = previous.and_then(|agents| agents.get(&path)) {
                    warn!(
                        agent = %previous.name,
                        agent_dir = %path.display(),
                        error = %e,
                        "ошибка повторной загрузки агента — сохраняю прежнюю годную версию"
                    );
                    agents_by_dir.insert(path, previous.clone());
                } else {
                    warn!(
                        agent = %dir_name,
                        agent_dir = %path.display(),
                        error = %e,
                        "пропускаю агента из-за ошибки загрузки"
                    );
                }
            }
        }
    }

    let agents = select_active_agents(&agents_by_dir);
    Ok(LoadedAgents {
        agents,
        agents_by_dir,
    })
}

fn select_active_agents(
    agents_by_dir: &HashMap<PathBuf, Arc<AgentDefinition>>,
) -> HashMap<String, Arc<AgentDefinition>> {
    let mut candidates: HashMap<String, Vec<(&PathBuf, &Arc<AgentDefinition>)>> = HashMap::new();
    for (path, agent) in agents_by_dir {
        candidates
            .entry(agent.name.clone())
            .or_default()
            .push((path, agent));
    }

    let mut agents = HashMap::new();
    for (name, mut definitions) in candidates {
        definitions.sort_by_key(|(path, _)| *path);
        let winner = definitions
            .iter()
            .position(|(path, _)| {
                path.file_name().and_then(|part| part.to_str()) == Some(name.as_str())
            })
            .unwrap_or(0);
        let (winner_path, winner_agent) = definitions[winner];
        for (path, _) in definitions.iter().copied() {
            if path != winner_path {
                let message = duplicate_agent_error(&name, winner_path, path);
                error!(
                    agent = %name,
                    winner_dir = %winner_path.display(),
                    duplicate_dir = %path.display(),
                    "{message}"
                );
            }
        }
        agents.insert(name, winner_agent.clone());
    }
    agents
}

fn duplicate_agent_error(name: &str, winner_path: &Path, duplicate_path: &Path) -> String {
    format!(
        "два каталога содержат агента с одинаковым именем '{name}': {} и {}; выбран {}",
        winner_path.display(),
        duplicate_path.display(),
        winner_path.display()
    )
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

/// Неизвестные поля не выключают агента, но должны быть видны в журнале.
fn unknown_agent_config_keys(raw: &str) -> Result<Vec<String>, toml::de::Error> {
    let value: toml::Value = toml::from_str(raw)?;
    let Some(root) = value.as_table() else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    unknown_keys_in_table(
        root,
        "",
        &[
            "name",
            "description",
            "version",
            "tags",
            "kind",
            "model",
            "response",
            "input",
            "limits",
            "cache",
            "execution",
            "claude_cli",
            "skill_scope",
            "skill_include_1c",
            "skill_bodies_top",
        ],
        &mut out,
    );
    for (section, known) in [
        (
            "model",
            &[
                "provider",
                "name",
                "temperature",
                "max_tokens",
                "top_p",
                "extra_body",
            ][..],
        ),
        ("response", &["format", "schema_file", "schema_strict"]),
        ("input", &["required", "optional"]),
        (
            "limits",
            &["max_input_tokens", "max_cost_usd", "timeout_sec"],
        ),
        ("cache", &["enabled", "ttl_sec", "key_fields"]),
        (
            "execution",
            &[
                "allowed_tools",
                "disallowed_tools",
                "permission_mode",
                "cwd_template",
                "allowed_roots",
                "mcp_config",
                "max_turns",
                "extra_args",
            ],
        ),
        (
            "claude_cli",
            &[
                "allowed_tools",
                "disallowed_tools",
                "permission_mode",
                "cwd_template",
                "allowed_roots",
                "mcp_config",
                "max_turns",
                "extra_args",
            ],
        ),
    ] {
        if let Some(table) = root.get(section).and_then(toml::Value::as_table) {
            unknown_keys_in_table(table, &format!("{section}."), known, &mut out);
        }
    }
    out.sort();
    Ok(out)
}

/// Поля общего договора, которые адаптер codex-cli пока не применяет.
fn codex_cli_ignored_fields_warning(config: &AgentConfig) -> Option<String> {
    if config.model.provider != "codex-cli" {
        return None;
    }
    let execution = config.execution.as_ref()?;
    let mut fields = Vec::new();
    if execution.cwd_template.is_some() {
        fields.push("cwd");
    }
    if execution.mcp_config.is_some() {
        fields.push("mcp_config");
    }
    if execution.max_turns.is_some() {
        fields.push("max_turns");
    }
    if !execution.allowed_tools.is_empty() {
        fields.push("allowed_tools");
    }
    if !execution.disallowed_tools.is_empty() {
        fields.push("disallowed_tools");
    }
    if execution.permission_mode.is_some() {
        fields.push("permission_mode");
    }
    (!fields.is_empty()).then(|| {
        format!(
            "codex-cli игнорирует поля [execution]: {}",
            fields.join(", ")
        )
    })
}

fn load_agent(dir: &Path) -> Result<AgentDefinition> {
    let config_path = dir.join("config.toml");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("чтение {}", config_path.display()))?;
    let config: AgentConfig =
        toml::from_str(&raw).with_context(|| format!("парсинг {}", config_path.display()))?;

    for key in unknown_agent_config_keys(&raw)
        .with_context(|| format!("поиск неизвестных ключей в {}", config_path.display()))?
    {
        warn!(
            agent = config.name,
            config_key = key,
            "неизвестный ключ в config.toml агента — значение проигнорировано"
        );
    }

    if let Some(raw_mcp) = config
        .execution
        .as_ref()
        .and_then(|execution| execution.mcp_config.as_deref())
    {
        crate::providers::mcp_client::parse_mcp_config(raw_mcp).with_context(|| {
            format!(
                "невалидный JSON в mcp_config агента '{}' ({})",
                config.name,
                config_path.display()
            )
        })?;
    }

    if let Some(roots) = config
        .execution
        .as_ref()
        .and_then(|execution| execution.allowed_roots.as_ref())
    {
        if roots.is_empty() {
            return Err(anyhow!(
                "у агента '{}' allowed_roots задан пустым: перечислите хотя бы один корень или уберите ключ",
                config.name
            ));
        }
        if let Some(relative) = roots.iter().find(|root| root.is_relative()) {
            return Err(anyhow!(
                "у агента '{}' allowed_roots содержит относительный путь '{}': все корни должны быть абсолютными",
                config.name,
                relative.display()
            ));
        }
    }

    if let Some(message) = codex_cli_ignored_fields_warning(&config) {
        warn!(agent = config.name, "{message}");
    }

    // Сверяем имя агента (config.name) с именем папки. Если не совпало —
    // предупреждаем; имя из config.name становится каноническим.
    if let Some(dir_name) = dir.file_name().and_then(|s| s.to_str()) {
        if dir_name != config.name {
            warn!(
                dir_name,
                config_name = config.name,
                "имя папки агента не совпадает с config.name — использую config.name"
            );
        }
    }

    let prompts = load_prompts(dir)?;
    if prompts.is_empty() {
        return Err(anyhow!(
            "у агента '{}' нет ни одного prompt.md в {}",
            config.name,
            dir.display()
        ));
    }

    let last_modified = newest_mtime(dir)?;
    let schema = load_schema(dir, &config)?;

    Ok(AgentDefinition {
        name: config.name.clone(),
        config,
        prompts,
        last_modified,
        schema,
    })
}

/// Загрузить JSON-схему ответа из `response.schema_file` (если задана).
/// Путь резолвится от каталога агента. Ошибка чтения/парсинга схемы —
/// фатальна для агента (лучше явно упасть на старте, чем молча не
/// валидировать ответы в проде).
fn load_schema(dir: &Path, config: &AgentConfig) -> Result<Option<serde_json::Value>> {
    let Some(file) = &config.response.schema_file else {
        return Ok(None);
    };
    let path = dir.join(file);
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("чтение схемы {}", path.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("парсинг схемы {}", path.display()))?;
    Ok(Some(json))
}

/// Собрать варианты промптов: prompt.md (default), prompt.v2.md и т.п.
fn load_prompts(dir: &Path) -> Result<HashMap<String, String>> {
    let mut out = HashMap::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = match path.file_name().and_then(|s| s.to_str()) {
            Some(s) => s,
            None => continue,
        };
        let variant = match prompt_variant(name) {
            Some(v) => v,
            None => continue,
        };
        let body =
            std::fs::read_to_string(&path).with_context(|| format!("чтение {}", path.display()))?;
        out.insert(variant, body);
    }
    Ok(out)
}

/// `prompt.md` → "default", `prompt.v2.md` → "v2", `prompt.experimental.md` → "experimental".
fn prompt_variant(filename: &str) -> Option<String> {
    if filename == "prompt.md" {
        return Some("default".to_string());
    }
    let stem = filename.strip_suffix(".md")?;
    let variant = stem.strip_prefix("prompt.")?;
    if variant.is_empty() {
        None
    } else {
        Some(variant.to_string())
    }
}

fn newest_mtime(dir: &Path) -> Result<chrono::DateTime<chrono::Utc>> {
    let mut newest: Option<std::time::SystemTime> = None;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if let Ok(meta) = entry.metadata() {
            if let Ok(modified) = meta.modified() {
                newest = Some(match newest {
                    Some(prev) if prev > modified => prev,
                    _ => modified,
                });
            }
        }
    }
    let st = newest.unwrap_or_else(std::time::SystemTime::now);
    Ok(chrono::DateTime::<chrono::Utc>::from(st))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_md_is_default() {
        assert_eq!(prompt_variant("prompt.md").as_deref(), Some("default"));
    }

    #[test]
    fn versioned_prompts() {
        assert_eq!(prompt_variant("prompt.v2.md").as_deref(), Some("v2"));
        assert_eq!(
            prompt_variant("prompt.experimental.md").as_deref(),
            Some("experimental")
        );
    }

    #[test]
    fn non_prompt_files_ignored() {
        assert_eq!(prompt_variant("config.toml"), None);
        assert_eq!(prompt_variant("schema.json"), None);
        assert_eq!(prompt_variant("README.md"), None);
    }

    #[test]
    fn empty_variant_ignored() {
        // "prompt..md" → вариант пустой → None.
        assert_eq!(prompt_variant("prompt..md"), None);
    }

    #[test]
    fn default_kind_is_prompt_template() {
        // Безопасный дефолт: бот не получит recursion-aware агента случайно.
        assert_eq!(AgentKind::default(), AgentKind::PromptTemplate);
    }

    #[test]
    fn execution_section_reads_old_name_too() {
        // Секция переименована [claude_cli] → [execution]; прежнее имя обязано
        // читаться, иначе чужие config.toml молча теряют инструменты.
        let основа = "name = \"a\"\n[model]\nprovider = \"mock\"\nname = \"m\"\n";
        for имя in ["execution", "claude_cli"] {
            let текст = format!("{основа}\n[{имя}]\nallowed_tools = [\"t1\"]\nmax_turns = 7\n");
            let cfg: AgentConfig = toml::from_str(&текст).expect("конфиг разобран");
            let секция = cfg.execution.expect("секция прочитана");
            assert_eq!(секция.allowed_tools, vec!["t1".to_string()], "имя {имя}");
            assert_eq!(секция.max_turns, Some(7), "имя {имя}");
        }
    }

    #[test]
    fn schema_strict_defaults_to_off() {
        let текст = "name = \"a\"\n[model]\nprovider = \"mock\"\nname = \"m\"\n[response]\nformat = \"json\"\nschema_file = \"schema.json\"\n";
        let config: AgentConfig = toml::from_str(текст).expect("конфиг разобран");
        assert!(
            !config.response.schema_strict,
            "по умолчанию проверка мягкая"
        );
    }

    #[test]
    fn schema_strict_key_is_known_and_parsed() {
        // Ключ добавлен в список разрешённых: иначе конфиг агента с ним был бы
        // отвергнут как содержащий неизвестный ключ.
        let текст = "name = \"a\"\n[model]\nprovider = \"mock\"\nname = \"m\"\n[response]\nformat = \"json\"\nschema_file = \"schema.json\"\nschema_strict = true\n";
        let config: AgentConfig = toml::from_str(текст).expect("конфиг разобран");
        assert!(config.response.schema_strict, "schema_strict прочитан");
        assert!(
            unknown_agent_config_keys(текст).unwrap().is_empty(),
            "schema_strict не должен попадать в неизвестные ключи"
        );
    }

    #[test]
    fn agent_config_with_schema_strict_loads() {
        // Ключ не отвергает агента при загрузке и действительно читается.
        let base = temp_dir("schema-strict-load");
        let agent_dir = base.join("strict-agent");
        std::fs::create_dir_all(&agent_dir).expect("каталог агента");
        std::fs::write(
            agent_dir.join("config.toml"),
            "name = \"strict-agent\"\n[model]\nprovider = \"mock\"\nname = \"m\"\n[response]\nformat = \"json\"\nschema_file = \"schema.json\"\nschema_strict = true\n",
        )
        .expect("config.toml агента");
        std::fs::write(agent_dir.join("prompt.md"), "ответ").expect("prompt.md агента");
        std::fs::write(
            agent_dir.join("schema.json"),
            r#"{"type":"object","required":["summary"]}"#,
        )
        .expect("schema.json агента");
        let def = load_agent(&agent_dir).expect("агент с schema_strict загружен");
        assert!(def.config.response.schema_strict);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn unknown_agent_keys_are_reported_without_rejecting_config() {
        let текст = r#"
name = "a"
unknown_top = true
skill_bodies_top = 1

[model]
provider = "mock"
name = "m"
skill_bodies_top = 2

[model.extra_body]
custom_provider_option = true

[limits]
timeout = 1800
max_output_tokens = 100
"#;
        let config: AgentConfig = toml::from_str(текст).expect("конфиг разбирается");
        assert_eq!(config.skill_bodies_top, 1);
        assert_eq!(
            unknown_agent_config_keys(текст).unwrap(),
            vec![
                "limits.max_output_tokens".to_string(),
                "limits.timeout".to_string(),
                "model.skill_bodies_top".to_string(),
                "unknown_top".to_string(),
            ]
        );
    }

    #[test]
    fn codex_cli_ignored_execution_fields_have_readable_warning() {
        let текст = r#"
name = "codex-agent"
[model]
provider = "codex-cli"
name = "gpt-5.6-terra"
[execution]
cwd_template = "C:/work"
mcp_config = "{}"
max_turns = 3
allowed_tools = ["mcp__fixture"]
disallowed_tools = ["mcp__fixture__write"]
permission_mode = "default"
"#;
        let config: AgentConfig = toml::from_str(текст).expect("конфиг разбирается");
        assert_eq!(
            codex_cli_ignored_fields_warning(&config).as_deref(),
            Some(
                "codex-cli игнорирует поля [execution]: cwd, mcp_config, max_turns, allowed_tools, disallowed_tools, permission_mode"
            )
        );
    }

    /// Временный каталог для одного теста: в имени — метка и наносекунды,
    /// чтобы параллельные тесты друг с другом не пересекались.
    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("время")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("agents-mcp-registry-{tag}-{nanos}"));
        std::fs::create_dir_all(&dir).expect("временный каталог");
        dir
    }

    /// Положить в каталог одного агента (config.toml + prompt.md) — минимальный
    /// агент на провайдере mock.
    fn write_agent(dir: &Path, name: &str) {
        let agent_dir = dir.join(name);
        write_agent_at(&agent_dir, name, "Тема: {{topic}}");
    }

    fn write_agent_at(agent_dir: &Path, name: &str, prompt: &str) {
        std::fs::create_dir_all(agent_dir).expect("каталог агента");
        std::fs::write(
            agent_dir.join("config.toml"),
            format!(
                "name = \"{name}\"\n\n[model]\nprovider = \"mock\"\nname = \"mock-model-v0\"\n"
            ),
        )
        .expect("config.toml агента");
        std::fs::write(agent_dir.join("prompt.md"), prompt).expect("prompt.md агента");
    }

    #[test]
    fn malformed_mcp_config_rejects_agent_load() {
        let base = temp_dir("malformed-mcp-config");
        write_agent(&base, "broken-agent");
        let agent_dir = base.join("broken-agent");
        std::fs::write(
            agent_dir.join("config.toml"),
            "name = \"broken-agent\"\n[model]\nprovider = \"mock\"\nname = \"m\"\n[execution]\nmcp_config = \"{broken\"\n",
        )
        .expect("config.toml агента");
        let error = load_agent(&agent_dir).expect_err("негодный JSON обязан отклонить агента");
        assert!(error.to_string().contains("невалидный JSON в mcp_config"));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn allowed_roots_parsed_and_not_reported_unknown() {
        let основа = "name = \"a\"\n[model]\nprovider = \"mock\"\nname = \"m\"\n";
        for имя in ["execution", "claude_cli"] {
            let текст =
                format!("{основа}\n[{имя}]\nallowed_roots = [\"C:/work\", \"D:/sandbox\"]\n");
            let cfg: AgentConfig = toml::from_str(&текст).expect("конфиг разобран");
            let секция = cfg.execution.expect("секция прочитана");
            assert_eq!(
                секция.allowed_roots,
                Some(vec![PathBuf::from("C:/work"), PathBuf::from("D:/sandbox")]),
                "имя {имя}"
            );
            assert!(
                unknown_agent_config_keys(&текст).unwrap().is_empty(),
                "allowed_roots не должен попадать в неизвестные ключи (имя {имя})"
            );
        }
        // Ключ не задан — None, и с секцией [execution], и вовсе без неё.
        let без_ключа: AgentConfig =
            toml::from_str(&format!("{основа}\n[execution]\nmax_turns = 3\n"))
                .expect("конфиг разобран");
        assert_eq!(
            без_ключа.execution.expect("секция прочитана").allowed_roots,
            None
        );
        let совсем_без: AgentConfig = toml::from_str(основа).expect("конфиг разобран");
        assert!(совсем_без.execution.is_none());
    }

    #[test]
    fn empty_allowed_roots_rejects_agent_load() {
        let base = temp_dir("empty-allowed-roots");
        write_agent(&base, "empty-roots");
        std::fs::write(
            base.join("empty-roots").join("config.toml"),
            "name = \"empty-roots\"\n[model]\nprovider = \"mock\"\nname = \"m\"\n[execution]\nallowed_roots = []\n",
        )
        .expect("config.toml агента");
        let error = load_agent(&base.join("empty-roots"))
            .expect_err("пустой allowed_roots обязан отклонить агента");
        assert!(error.to_string().contains("allowed_roots"));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn relative_allowed_roots_rejects_agent_load() {
        let base = temp_dir("relative-allowed-roots");
        write_agent(&base, "relative-roots");
        std::fs::write(
            base.join("relative-roots").join("config.toml"),
            "name = \"relative-roots\"\n[model]\nprovider = \"mock\"\nname = \"m\"\n[execution]\nallowed_roots = [\"C:/work\", \"sandbox\"]\n",
        )
        .expect("config.toml агента");
        let error = load_agent(&base.join("relative-roots"))
            .expect_err("относительный путь в allowed_roots обязан отклонить агента");
        assert!(error.to_string().contains("относительный путь"));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn all_repository_agents_load() {
        let agents_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("agents");
        // В публичной поставке каталога agents нет: проверять нечего.
        if !agents_dir.is_dir() {
            return;
        }
        let mut loaded = 0;
        for entry in std::fs::read_dir(&agents_dir).expect("каталог agents") {
            let path = entry.expect("элемент каталога").path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if !path.is_dir() || name.starts_with('.') || name.starts_with('_') {
                continue;
            }
            load_agent(&path)
                .unwrap_or_else(|error| panic!("агент {} не загрузился: {error}", path.display()));
            loaded += 1;
        }
        assert!(loaded > 0, "каталог agents не должен быть пустым");
    }

    #[test]
    fn reload_from_swaps_dir_and_keeps_state_on_error() {
        let base = temp_dir("reload-from");
        let dir_a = base.join("agents-a");
        let dir_b = base.join("agents-b");
        write_agent(&dir_a, "agent-a");
        write_agent(&dir_b, "agent-b");
        write_agent(&dir_b, "agent-c");

        let registry = Registry::load(dir_a.clone()).expect("реестр загружен");
        assert_eq!(registry.len(), 1);

        // Каталога-агента нет (путь занят файлом) — ошибка, прежние агенты на месте.
        let not_a_dir = base.join("файл-вместо-каталога");
        std::fs::write(&not_a_dir, "x").expect("файл");
        assert!(registry.reload_from(not_a_dir).is_err());
        // Несуществующий каталог (опечатка в пути) — тоже ошибка, а не пустой реестр.
        assert!(registry
            .reload_from(base.join("нет-такого-каталога"))
            .is_err());
        assert_eq!(registry.len(), 1);
        assert!(registry.get("agent-a").is_some());

        // Другой валидный каталог — набор подменён целиком.
        registry.reload_from(dir_b.clone()).expect("reload_from");
        assert_eq!(registry.len(), 2);
        assert!(registry.get("agent-b").is_some());
        assert!(registry.get("agent-a").is_none());

        // После смены каталога reload() читает уже НОВЫЙ каталог.
        write_agent(&dir_b, "agent-d");
        registry.reload().expect("reload");
        assert_eq!(registry.len(), 3);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn reload_keeps_last_good_agent_after_broken_config() {
        let base = temp_dir("last-good");
        write_agent_at(&base.join("stable"), "stable", "прежний промпт");
        let registry = Registry::load(base.clone()).expect("реестр загружен");

        std::fs::write(base.join("stable/config.toml"), "это не toml = [")
            .expect("сломанный config.toml");
        registry
            .reload()
            .expect("ошибка агента не ломает перечитку");

        let agent = registry.get("stable").expect("прежний агент сохранён");
        assert_eq!(agent.prompts.get("default").unwrap(), "прежний промпт");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn reload_missing_agents_dir_keeps_registry() {
        let base = temp_dir("missing-dir");
        let agents_dir = base.join("agents");
        write_agent(&agents_dir, "stable");
        let registry = Registry::load(agents_dir.clone()).expect("реестр загружен");

        std::fs::remove_dir_all(&agents_dir).expect("каталог удалён");
        assert!(registry.reload().is_err());
        assert!(registry.get("stable").is_some());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn duplicate_name_prefers_matching_directory_and_keeps_its_last_good_version() {
        let base = temp_dir("duplicate-name");
        write_agent_at(&base.join("a"), "a", "победитель");
        write_agent_at(&base.join("b"), "a", "дубль");
        let duplicate_error = duplicate_agent_error("a", &base.join("a"), &base.join("b"));
        assert!(duplicate_error.contains(&base.join("a").display().to_string()));
        assert!(duplicate_error.contains(&base.join("b").display().to_string()));
        let registry = Registry::load(base.clone()).expect("реестр загружен");
        assert_eq!(
            registry.get("a").unwrap().prompts.get("default").unwrap(),
            "победитель"
        );

        std::fs::write(base.join("a/config.toml"), "сломано = [")
            .expect("сломанный config.toml победителя");
        write_agent_at(&base.join("b"), "a", "обновлённый дубль");
        registry.reload().expect("перечитка");
        assert_eq!(
            registry.get("a").unwrap().prompts.get("default").unwrap(),
            "победитель"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn hidden_or_removed_agent_directory_removes_agent() {
        let base = temp_dir("hidden-agent");
        write_agent(&base, "agent");
        let registry = Registry::load(base.clone()).expect("реестр загружен");

        std::fs::rename(base.join("agent"), base.join("_agent")).expect("агент скрыт");
        registry.reload().expect("перечитка");
        assert!(registry.get("agent").is_none());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn stale_generation_does_not_replace_newer_registry() {
        let base = temp_dir("generation");
        let dir_a = base.join("agents-a");
        let dir_b = base.join("agents-b");
        write_agent(&dir_a, "agent-a");
        write_agent(&dir_b, "agent-b");
        let registry = Registry::load(dir_a).expect("реестр загружен");

        let (old_generation, old_dir, previous) = registry.begin_reload();
        let stale = load_all_agents(&old_dir, Some(&previous)).expect("старый результат");
        registry
            .reload_from(dir_b)
            .expect("новое поколение применено");

        assert!(registry
            .apply_loaded(old_generation, old_dir, stale)
            .is_err());
        assert!(registry.get("agent-b").is_some());
        assert!(registry.get("agent-a").is_none());
        let _ = std::fs::remove_dir_all(&base);
    }
}
