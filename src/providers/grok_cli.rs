//! Провайдер grok-cli: вызов локального `grok.exe` (Grok CLI) через
//! `tokio::process::Command`.
//!
//! Авторизация — вход в Grok CLI на этой машине (подписка), API-ключ службе не
//! нужен, поштучной стоимости вызова нет: `cost_usd` не заполняется.
//!
//! По устройству — прямой аналог `codex_cli.rs`: subprocess, семафор на
//! конкурентность, один вызов = один запуск CLI. Отличия:
//!
//!   * задание уходит ФАЙЛОМ (`--prompt-file` + `--verbatim`), а не потоком
//!     stdin: CLI принимает готовый промпт без своей обвязки;
//!   * MCP-серверы приносит не командная строка, а сгенерированный на каждый
//!     вызов `.grok/config.toml` в рабочем каталоге процесса. Только так
//!     перекрытие `mcp.<сервер>.url` доходит до CLI: runtime переписывает поле
//!     `url` в `hints.mcp_config` (`apply_mcp_url_overrides`), а `extra_args` он
//!     не трогает вовсе;
//!   * заголовки записей mcp_config (в том числе `x-agents-mcp-call`, который
//!     служба проставляет перед вызовом) переносятся в TOML инлайн-таблицей
//!     `headers` — иначе ключ вызова не доехал бы до службы и файловые
//!     инструменты упирались бы в отказ;
//!   * белый список инструментов проверяется по `tools/list` ДО запуска:
//!     всё, чего нет в `[execution] allowed_tools` агента, уходит в `--deny`.
//!
//! Формат ответа снят живыми прогонами 21.09.2026 (CLI 1.0.40): один объект
//! JSON в stdout
//!   `{"text":"...","thought":"...","sessionId":"...","stopReason":"end_turn",
//!     "usage":{"input_tokens":..,"output_tokens":..,"cache_read_input_tokens":..,
//!     "cache_creation_input_tokens":..},"total_cost_usd":0.0}`
//! Потокового режима у CLI нет, поэтому построчный транскрипт берётся не из
//! вывода процесса, а из файла сессии CLI (`chat_history.jsonl`) — ходы туда
//! дописывает сам grok.
//!
//! Схему ответа проверяет служба (`[response] schema_file`), ключ CLI
//! `--json-schema` не используется: проверка провайдеро-независима.
//!
//! Рабочий каталог процесса — свой временный каталог на вызов. `hints.cwd`
//! (`cwd_template` агента) рабочим каталогом процесса НЕ становится: он нужен
//! службе, чтобы ограничить область ключа вызова (и через него `fs_*`), а
//! путь задания доходит до модели текстом промпта.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

use crate::proc_tree;

use super::mcp_client::{self, McpServer};
use super::{LlmError, LlmProvider, LlmRequest, LlmResponse};

/// Встроенные инструменты CLI, которых у агента-исполнителя быть не должно:
/// файлы он правит через `fs_*`, по чужому коду ищет через `code-index`, сеть
/// ему не нужна. Список привязан к версии CLI (1.0.40) и обязан совпадать у
/// ВСЕХ grok-cli агентов, поэтому живёт в коде: ошибка в конфиге агента
/// оставила бы модель без инструментов, а разбираться в этом пришлось бы по
/// журналу провала.
///
/// `search_tool` и `use_tool` здесь НЕТ и быть не должно: это ровно те два
/// встроенных инструмента, которыми CLI вызывает MCP-инструменты (полное имя
/// вида `<сервер>__<инструмент>`). Попади они в `--disallowed-tools` — у модели
/// не осталось бы ни одного инструмента вовсе.
///
/// Имена в этом списке — ВНУТРЕННИЕ имена CLI, а не те, что объявлены модели:
/// запуск команд снимается как `run_terminal_cmd`, хотя модели инструмент
/// объявлен под именем `run_terminal_command` (проверено 06.09.2026). Инструменты
/// подагентов этот список не снимает — их снимает `--tools no_builtin_tools` в
/// `build_args`. Список проверен живым прогоном
/// 21.09.2026: в `tool_definitions.json` сессии остались только `search_tool`
/// и `use_tool`.
pub const BUILTIN_DISALLOWED_TOOLS: &[&str] = &[
    "run_terminal_cmd",
    "run_terminal_command",
    "bash",
    "terminal",
    "shell",
    "read_file",
    "search_replace",
    "list_dir",
    "grep",
    "write",
    "todo_write",
    "scheduler_create",
    "scheduler_delete",
    "scheduler_list",
    "monitor",
    "workflow",
    "enter_plan_mode",
    "exit_plan_mode",
    "ask_user_question",
    "image_gen",
    "image_edit",
    "image_to_video",
    "reference_to_video",
    "spawn_subagent",
    "kill_command_or_subagent",
    "get_command_or_subagent_output",
    "send_feedback",
];

/// Префикс временных каталогов вызова: по нему же чистятся залежавшиеся.
const TMP_PREFIX: &str = "agents-mcp-grok-";

/// Сколько каталог провалившегося вызова живёт до уборки. В провале он остаётся
/// намеренно (там промпт и stderr), но копиться им нельзя.
const STALE_DIR_AGE: Duration = Duration::from_secs(6 * 60 * 60);

/// Потолок на длину одной записи хода в базе. Строка CLI бывает длинной
/// (целый ответ модели), в диагностику столько не нужно.
const ZAPIS_MAX: usize = 4000;

/// Сколько знаков stdout/stderr держать для сообщения об ошибке: отказ CLI
/// приходит текстом, и это единственный источник причины.
const HVOST_MAX: usize = 4000;

pub struct GrokCliProvider {
    executable: PathBuf,
    /// Путь к пользовательскому `config.toml` CLI. Из него читаются имена
    /// ЛИЧНЫХ серверов (подписок Claude/Cursor и локальных) — их надо погасить,
    /// иначе CLI подтянул бы чужие серверы в вызов агента. None — берётся
    /// `<USERPROFILE|HOME>/.grok/config.toml`.
    user_config: Option<PathBuf>,
    /// Предел ходов, если агент не задал свой.
    default_max_turns: u32,
    /// Предел времени одного вызова MCP-инструмента (`tool_timeout_sec` в TOML
    /// сервера) — тот же порядок, что `RPC_TIMEOUT` в `mcp_client`.
    tool_timeout_sec: u64,
    semaphore: Arc<Semaphore>,
    /// Клиент нужен только для `initialize`/`tools/list` по серверам агента —
    /// по тем же адресам, что и в остальных провайдерах (локальная сеть, мимо
    /// посредника).
    http: reqwest::Client,
}

/// Описание MCP-сервера агента для генерации TOML: имя в терминах CLI, адрес и
/// заголовки записей (в них лежит ключ вызова).
#[derive(Debug, Clone)]
struct AgentServer {
    name: String,
    url: String,
    headers: Vec<(String, String)>,
}

/// Разобранный ответ CLI.
#[derive(Debug, Default)]
struct GrokAnswer {
    text: String,
    thought: Option<String>,
    session_id: Option<String>,
    stop_reason: String,
    input_tokens: u32,
    output_tokens: u32,
    cache_creation_input_tokens: u32,
    cache_read_input_tokens: u32,
    total_cost_usd: Option<f64>,
}

/// Имя сервера в терминах CLI: дефис заменяется на подчёркивание
/// (`code-index` → `code_index`), поэтому и `--deny`, и `allowed_tools` агента
/// пишутся в этой форме.
fn cli_server_name(alias: &str) -> String {
    alias.replace('-', "_")
}

/// Ключ TOML: без кавычек, если имя подходит под `^[A-Za-z_][A-Za-z0-9_-]*$`,
/// иначе — строка в кавычках. Имена вроде `1c` без кавычек TOML не примет.
fn toml_key(name: &str) -> String {
    let mut chars = name.chars();
    let bare = match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {
            chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        }
        _ => false,
    };
    if bare {
        name.to_string()
    } else {
        toml_string(name)
    }
}

/// Строка TOML в кавычках с экранированием `\` и `"`.
fn toml_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

/// Имена личных серверов из пользовательского конфига CLI: ключи таблицы
/// `mcp_servers` плюс элементы `disabled_mcp_servers`, без повторов. Текста нет
/// либо он не разобран — пустой список (личных серверов просто не гасим).
fn personal_server_names(user_config_text: &str) -> Vec<String> {
    let Ok(value) = user_config_text.parse::<toml::Value>() else {
        return Vec::new();
    };
    let mut out: Vec<String> = Vec::new();
    if let Some(table) = value.get("mcp_servers").and_then(toml::Value::as_table) {
        for name in table.keys() {
            if !out.contains(name) {
                out.push(name.clone());
            }
        }
    }
    if let Some(items) = value
        .get("disabled_mcp_servers")
        .and_then(toml::Value::as_array)
    {
        for item in items {
            if let Some(name) = item.as_str() {
                if !out.iter().any(|seen| seen == name) {
                    out.push(name.to_string());
                }
            }
        }
    }
    out
}

/// Сгенерировать `.grok/config.toml` вызова: серверы агента плюс погашенные
/// личные серверы. Личным считается сервер, которого агент не принёс сам —
/// запись агента личной записью не затирается.
fn render_grok_config(
    agent_servers: &[AgentServer],
    personal: &[String],
    tool_timeout_sec: u64,
) -> String {
    let mut out = String::new();
    out.push_str("# Файл создан службой agents-mcp на время одного вызова.\n\n");
    for server in agent_servers {
        let name = cli_server_name(&server.name);
        out.push_str(&format!("[mcp_servers.{}]\n", toml_key(&name)));
        out.push_str(&format!("url = {}\n", toml_string(&server.url)));
        out.push_str(&format!("tool_timeout_sec = {tool_timeout_sec}\n"));
        if !server.headers.is_empty() {
            let pairs: Vec<String> = server
                .headers
                .iter()
                .map(|(name, value)| format!("{} = {}", toml_string(name), toml_string(value)))
                .collect();
            out.push_str(&format!("headers = {{ {} }}\n", pairs.join(", ")));
        }
        out.push('\n');
    }
    for name in personal {
        if agent_servers
            .iter()
            .any(|server| cli_server_name(&server.name) == *name)
        {
            continue;
        }
        out.push_str(&format!("[mcp_servers.{}]\n", toml_key(name)));
        out.push_str("enabled = false\n");
        out.push_str("url = \"http://127.0.0.1:9/disabled\"\n\n");
    }
    out
}

/// Аргументы `--deny` для инструментов вне белого списка. `all_tools` — пары
/// (алиас сервера, имя инструмента) в том порядке, в каком их отдал `tools/list`
/// (порядок сохраняем, чтобы тест был устойчив). Имя из `allowed_tools`, не
/// найденное ни у одного сервера, — ошибка: опечатка в конфиге агента не должна
/// молча ни сужать, ни расширять доступ.
fn deny_args(all_tools: &[(String, String)], allowed: &[String]) -> Result<Vec<String>, String> {
    let mut args = Vec::new();
    let mut matched = vec![false; allowed.len()];
    for (alias, tool) in all_tools {
        let full = format!("{}__{}", cli_server_name(alias), tool);
        match allowed.iter().position(|name| *name == full) {
            Some(index) => matched[index] = true,
            None => {
                args.push("--deny".to_string());
                args.push(format!("MCPTool({full})"));
            }
        }
    }
    for (index, name) in allowed.iter().enumerate() {
        if !matched[index] {
            return Err(format!(
                "инструмент '{name}' из allowed_tools не найден ни у одного MCP-сервера — проверьте имя и список серверов в mcp_config"
            ));
        }
    }
    Ok(args)
}

/// Фиксированная часть командной строки CLI: задание файлом, JSON-ответ, ходы,
/// отключённые планировщик/подагенты/сеть, отсутствие встроенных инструментов.
/// Затем идут `deny` (по одному инструменту вне белого списка), а в конце —
/// `extra` из `[execution] extra_args` агента (оттуда приходит
/// `--reasoning-effort`).
fn build_args(
    prompt_file: &Path,
    model: &str,
    max_turns: u32,
    deny: &[String],
    extra: &[String],
    disallowed: &[String],
) -> Vec<String> {
    let mut args = vec![
        "--prompt-file".to_string(),
        prompt_file.display().to_string(),
        "--verbatim".to_string(),
        "-m".to_string(),
        model.to_string(),
        "--output-format".to_string(),
        "json".to_string(),
        "--max-turns".to_string(),
        max_turns.to_string(),
        "--no-plan".to_string(),
        "--no-subagents".to_string(),
        "--disable-web-search".to_string(),
        "--trust".to_string(),
        "--always-approve".to_string(),
        "--tools".to_string(),
        "no_builtin_tools".to_string(),
        "--disallowed-tools".to_string(),
        disallowed.join(","),
    ];
    args.extend(deny.iter().cloned());
    args.extend(extra.iter().cloned());
    args
}

fn chislo(obj: &Value, field: &str) -> u32 {
    obj.get(field).and_then(Value::as_u64).unwrap_or(0) as u32
}

/// Разобрать JSON-ответ CLI. Ошибка — не JSON, пустой `text` либо `stopReason`,
/// отличный от `end_turn` (обрыв по ходам или отказ: разбирать как готовый
/// ответ нельзя).
fn parse_answer(stdout: &str) -> Result<GrokAnswer, String> {
    let value: Value = serde_json::from_str(stdout.trim())
        .map_err(|e| format!("stdout не разобран как JSON: {e}"))?;
    let obj = value
        .as_object()
        .ok_or_else(|| "ответ CLI не объект JSON".to_string())?;
    let stop_reason = obj
        .get("stopReason")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let text = obj
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if stop_reason != "end_turn" {
        return Err(format!(
            "stopReason ответа — «{stop_reason}», ожидалось end_turn"
        ));
    }
    if text.trim().is_empty() {
        return Err("в ответе пустое поле text".to_string());
    }
    let usage = obj.get("usage").cloned().unwrap_or(Value::Null);
    let thought = obj
        .get("thought")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string);
    Ok(GrokAnswer {
        text,
        thought,
        session_id: obj
            .get("sessionId")
            .and_then(Value::as_str)
            .map(str::to_string),
        stop_reason,
        input_tokens: chislo(&usage, "input_tokens"),
        output_tokens: chislo(&usage, "output_tokens"),
        cache_creation_input_tokens: chislo(&usage, "cache_creation_input_tokens"),
        cache_read_input_tokens: chislo(&usage, "cache_read_input_tokens"),
        total_cost_usd: obj.get("total_cost_usd").and_then(Value::as_f64),
    })
}

/// Хвост текста для сообщения об ошибке.
fn hvost_tail(text: &str) -> String {
    let trimmed = text.trim();
    let chars: Vec<char> = trimmed.chars().collect();
    if chars.len() <= HVOST_MAX {
        return trimmed.to_string();
    }
    chars[chars.len() - HVOST_MAX..].iter().collect()
}

fn hvost(stdout: &str, stderr: &str) -> String {
    format!(
        "stdout={} stderr={}",
        hvost_tail(stdout),
        hvost_tail(stderr)
    )
}

/// Обрезать строку транскрипта перед записью в базу.
fn obrezat_zapis(zapis: &str) -> String {
    if zapis.len() <= ZAPIS_MAX {
        return zapis.to_string();
    }
    let nachalo: String = zapis.chars().take(ZAPIS_MAX).collect();
    format!("{nachalo}…[обрезано, всего {} знаков]", zapis.len())
}

/// Убрать залежавшиеся каталоги провалившихся вызовов: иначе они копятся в
/// temp бесконечно. Best-effort: любая ошибка глотается.
fn cleanup_stale_dirs(older_than: Duration) {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    let now = std::time::SystemTime::now();
    for entry in entries.flatten() {
        if !entry.file_name().to_string_lossy().starts_with(TMP_PREFIX) {
            continue;
        }
        let old = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .map(|modified| now.duration_since(modified).unwrap_or_default() >= older_than)
            .unwrap_or(false);
        if old {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// Найти каталог сессии CLI: url-кодирование рабочего каталога (по которому CLI
/// раскладывает сессии) мы не воспроизводим, поэтому ищем по совпадению имени
/// подкаталога с `session_id` — на уровень ниже корня сессий.
fn find_session_dir(root: &Path, session_id: &str) -> Option<PathBuf> {
    let direct = root.join(session_id);
    if direct.join("chat_history.jsonl").is_file() {
        return Some(direct);
    }
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        let candidate = entry.path().join(session_id);
        if candidate.join("chat_history.jsonl").is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Best-effort выгрузка ходов сессии CLI в `agent_turns`. Провал чтения — только
/// `debug!`: транскрипт нужен для разбора, но из-за него вызов падать не должен.
async fn push_session_transcript(
    sessions_dir: &Path,
    session_id: &str,
    sink: &UnboundedSender<Value>,
) {
    let Some(dir) = find_session_dir(sessions_dir, session_id) else {
        debug!(
            session_id,
            sessions_dir = %sessions_dir.display(),
            "файл сессии grok-cli не найден — транскрипт не собран"
        );
        return;
    };
    let history = dir.join("chat_history.jsonl");
    let text = match tokio::fs::read_to_string(&history).await {
        Ok(text) => text,
        Err(e) => {
            debug!(path = %history.display(), error = %e, "транскрипт grok-cli не прочитан");
            return;
        }
    };
    let mut seq: i64 = 0;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let event = serde_json::from_str::<Value>(line)
            .ok()
            .and_then(|value| {
                value
                    .get("type")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_default();
        seq += 1;
        let sent = sink.send(json!({
            "seq": seq,
            "event": event,
            "ts_ms": chrono::Utc::now().timestamp_millis(),
            "zapis": obrezat_zapis(line),
        }));
        if sent.is_err() {
            return;
        }
    }
    debug!(session_id, hodov = seq, "транскрипт grok-cli отправлен");
}

/// Пользовательский конфиг CLI по умолчанию: `<USERPROFILE|HOME>/.grok/config.toml`.
/// Крейта для разворачивания `~` в проекте нет — путь собираем из переменных.
fn default_user_config() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(|home| PathBuf::from(home).join(".grok").join("config.toml"))
}

impl GrokCliProvider {
    pub fn new(
        executable: PathBuf,
        user_config: Option<PathBuf>,
        max_concurrent: u32,
        default_max_turns: u32,
        tool_timeout_sec: u64,
    ) -> Self {
        let permits = max_concurrent.max(1) as usize;
        Self {
            executable,
            user_config,
            default_max_turns,
            tool_timeout_sec,
            semaphore: Arc::new(Semaphore::new(permits)),
            http: super::build_http_client("grok-cli", None, None),
        }
    }

    /// Doctor self-test: `grok --version`. Вызывается из reload.rs при старте
    /// службы для health-report.
    pub async fn doctor(executable: &PathBuf) -> Result<String, String> {
        let output = match tokio::time::timeout(
            Duration::from_secs(15),
            Command::new(executable)
                .arg("--version")
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true)
                .output(),
        )
        .await
        {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => {
                return Err(format!(
                    "не удалось запустить '{}': {e}",
                    executable.display()
                ))
            }
            Err(_) => {
                return Err(format!(
                    "'{} --version' не завершился за 15 с",
                    executable.display()
                ))
            }
        };
        if !output.status.success() {
            return Err(format!(
                "'{} --version' rc={:?}, stderr={}",
                executable.display(),
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Полный перечень инструментов серверов агента в порядке `tools/list`.
    /// Провал любого сервера — отказ целиком (`ToolsUnavailable`): агент без
    /// заявленных инструментов задачу не выполнит, а прогон уже стоил бы денег.
    async fn list_server_tools(
        &self,
        servers: &[McpServer],
        deadline: tokio::time::Instant,
    ) -> Result<Vec<(String, String)>, LlmError> {
        let mut out = Vec::new();
        for server in servers {
            let listed = tokio::time::timeout_at(deadline, async {
                let session = mcp_client::initialize_session(&self.http, server).await?;
                mcp_client::list_tools(&self.http, server, &session).await
            })
            .await
            .map_err(|_| LlmError::Timeout)?;
            let tools = listed.map_err(|e| {
                LlmError::ToolsUnavailable(format!(
                    "MCP-сервер '{}' ({}) не отдал список инструментов: {e}",
                    server.alias, server.url
                ))
            })?;
            for tool in tools {
                out.push((server.alias.clone(), tool.tool_name));
            }
        }
        Ok(out)
    }
}

#[async_trait]
impl LlmProvider for GrokCliProvider {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError> {
        let deadline = tokio::time::Instant::now() + req.timeout;
        cleanup_stale_dirs(STALE_DIR_AGE);

        let permit = self.semaphore.clone().acquire_owned();
        let permit = tokio::time::timeout_at(deadline, permit)
            .await
            .map_err(|_| LlmError::Timeout)?
            .map_err(|e| LlmError::Subprocess(format!("семафор закрыт: {e}")))?;

        let hints = req.cli_hints.clone().unwrap_or_default();

        // Список серверов и их инструментов нужен ДО запуска: белый список
        // агента сверяется с тем, что серверы отдают на самом деле.
        let servers = match hints.mcp_config.as_deref() {
            Some(raw) => mcp_client::parse_mcp_config(raw).map_err(|e| {
                LlmError::Provider(format!("mcp_config агента не разобран как JSON: {e}"))
            })?,
            None => Vec::new(),
        };
        let all_tools = self.list_server_tools(&servers, deadline).await?;
        let deny = deny_args(&all_tools, &hints.allowed_tools).map_err(LlmError::Provider)?;

        // Свой временный каталог на вызов: там и промпт, и `.grok/config.toml`.
        // Уникальность — чтобы параллельные вызовы под общим семафором не
        // делили один файл конфигурации.
        let call_dir = std::env::temp_dir().join(format!("{TMP_PREFIX}{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(call_dir.join(".grok")).map_err(|e| {
            LlmError::Subprocess(format!(
                "каталог вызова '{}' не создан: {e}",
                call_dir.display()
            ))
        })?;

        let combined_prompt = if req.user_input.is_empty() {
            req.system_prompt.clone()
        } else {
            format!("{}\n\n## Запрос\n{}", req.system_prompt, req.user_input)
        };
        let prompt_file = call_dir.join("prompt.txt");
        std::fs::write(&prompt_file, combined_prompt.as_bytes()).map_err(|e| {
            LlmError::Subprocess(format!(
                "файл задания '{}' не записан: {e}",
                prompt_file.display()
            ))
        })?;

        // Личные серверы CLI гасим при КАЖДОМ вызове: пользовательский конфиг
        // может меняться между вызовами, а подтянутый чужой сервер — это и
        // лишние токены в промпте, и доступ к данным вне области ключа.
        let user_config = self.user_config.clone().or_else(default_user_config);
        let personal = match user_config.as_deref() {
            Some(path) => match std::fs::read_to_string(path) {
                Ok(text) => personal_server_names(&text),
                Err(e) => {
                    debug!(path = %path.display(), error = %e, "пользовательский конфиг grok-cli не прочитан");
                    Vec::new()
                }
            },
            None => Vec::new(),
        };
        let agent_servers: Vec<AgentServer> = servers
            .iter()
            .map(|server| AgentServer {
                name: server.alias.clone(),
                url: server.url.clone(),
                headers: server.headers.clone(),
            })
            .collect();
        let config_text = render_grok_config(&agent_servers, &personal, self.tool_timeout_sec);
        let config_path = call_dir.join(".grok").join("config.toml");
        std::fs::write(&config_path, config_text.as_bytes()).map_err(|e| {
            LlmError::Subprocess(format!(
                "конфиг '{}' не записан: {e}",
                config_path.display()
            ))
        })?;

        // Список запрещённых встроенных инструментов — константа кода плюс то,
        // что добавил агент своим `[execution] disallowed_tools`.
        let mut disallowed: Vec<String> = BUILTIN_DISALLOWED_TOOLS
            .iter()
            .map(|name| name.to_string())
            .collect();
        for tool in &hints.disallowed_tools {
            if !disallowed.contains(tool) {
                disallowed.push(tool.clone());
            }
        }
        let max_turns = hints.max_turns.unwrap_or(self.default_max_turns);
        let args = build_args(
            &prompt_file,
            &req.model,
            max_turns,
            &deny,
            &hints.extra_args,
            &disallowed,
        );

        let mut cmd = Command::new(&self.executable);
        cmd.args(&args);
        // Рабочий каталог процесса — временный каталог вызова, а не `hints.cwd`:
        // `cwd_template` нужен службе для области ключа (см. doc-комментарий).
        cmd.current_dir(&call_dir);
        // Личные подписки Claude/Cursor не подтягиваем: в вызове агента видны
        // только серверы из mcp_config.
        cmd.env("GROK_CLAUDE_MCPS_ENABLED", "false")
            .env("GROK_CURSOR_MCPS_ENABLED", "false");
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        info!(
            model = %req.model,
            prompt_chars = combined_prompt.len(),
            tools = all_tools.len(),
            denied = deny.len() / 2,
            max_turns,
            call_dir = %call_dir.display(),
            "grok-cli: запускаю grok"
        );

        let sink = req.turn_sink.clone();
        let sessions_dir = user_config
            .as_deref()
            .and_then(|path| path.parent())
            .map(|dir| dir.join("sessions"));

        let outcome: Result<LlmResponse, LlmError> = async {
            let (mut child, _process_tree) = proc_tree::spawn(&mut cmd)
                .map_err(|e| LlmError::Subprocess(format!("spawn grok: {e}")))?;
            // Задание уходит файлом (`--prompt-file`), stdin процессу не нужен.
            drop(child.stdin.take());

            let stderr_pipe = child.stderr.take();
            let stderr_task = tokio::spawn(async move {
                let mut buf = String::new();
                if let Some(mut e) = stderr_pipe {
                    let _ = e.read_to_string(&mut buf).await;
                }
                buf
            });

            let stdout_pipe = child.stdout.take();
            let chtenie = async {
                let mut buf = Vec::new();
                if let Some(mut so) = stdout_pipe {
                    so.read_to_end(&mut buf)
                        .await
                        .map_err(|e| LlmError::Subprocess(format!("чтение stdout: {e}")))?;
                }
                let status = child
                    .wait()
                    .await
                    .map_err(|e| LlmError::Subprocess(format!("wait: {e}")))?;
                Ok::<_, LlmError>((status, buf))
            };

            let (status, stdout_bytes) = match tokio::time::timeout_at(deadline, chtenie).await {
                Ok(Ok(x)) => x,
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err(LlmError::Timeout),
            };
            let stderr_text = stderr_task.await.unwrap_or_default();
            let stdout_text = String::from_utf8_lossy(&stdout_bytes).to_string();

            if !status.success() {
                return Err(LlmError::Provider(format!(
                    "grok-cli завершился неуспешно (rc={:?}). {}",
                    status.code(),
                    hvost(&stdout_text, &stderr_text)
                )));
            }

            let answer = parse_answer(&stdout_text).map_err(|e| {
                LlmError::InvalidResponse(format!(
                    "grok-cli: {e}. {}",
                    hvost(&stdout_text, &stderr_text)
                ))
            })?;

            // Ходы уже записаны CLI в свой файл сессии — выгружаем их в базу
            // сразу, чтобы разбирать и зависший, и удачный вызов.
            if let (Some(sink), Some(session_id), Some(sessions_dir)) = (
                sink.as_ref(),
                answer.session_id.as_deref(),
                sessions_dir.as_deref(),
            ) {
                push_session_transcript(sessions_dir, session_id, sink).await;
            }

            let tokens_in = answer.input_tokens;
            let cache_read = answer.cache_read_input_tokens;
            let cache_creation = answer.cache_creation_input_tokens;
            let raw_input_tokens = tokens_in
                .saturating_sub(cache_read)
                .saturating_sub(cache_creation);

            debug!(
                chars = answer.text.len(),
                tokens_in,
                tokens_out = answer.output_tokens,
                // Стоимость подписки: в `agent_calls` её не пишем (цена вызова
                // подписочная), но в журнал — чтобы видеть, что служба вернула.
                total_cost_usd = answer.total_cost_usd.unwrap_or(0.0),
                stop_reason = %answer.stop_reason,
                "grok-cli ok"
            );

            Ok(LlmResponse {
                content: answer.text.trim().to_string(),
                tokens_in,
                tokens_out: answer.output_tokens,
                // Подписка Grok CLI: поштучной стоимости вызова нет.
                cost_usd: None,
                finish_reason: "stop".into(),
                reasoning: answer.thought,
                session_id: answer.session_id,
                raw_input_tokens,
                cache_creation_input_tokens: cache_creation,
                cache_read_input_tokens: cache_read,
                // Ходы ушли в базу через turn_sink — пакетом не дублируем.
                transcript: Vec::new(),
            })
        }
        .await;

        match outcome {
            Ok(response) => {
                if let Err(e) = std::fs::remove_dir_all(&call_dir) {
                    warn!(
                        call_dir = %call_dir.display(),
                        error = %e,
                        "не удалось удалить каталог вызова grok-cli"
                    );
                }
                drop(permit);
                Ok(response)
            }
            Err(error) => {
                // Каталог провала оставляем: в нём промпт, конфиг и всё, что
                // успел сказать CLI. Путь — в журнал, иначе искать нечего.
                warn!(
                    call_dir = %call_dir.display(),
                    error = %error,
                    "вызов grok-cli провален, каталог оставлен для разбора"
                );
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::ClaudeCliHints;

    #[test]
    fn deny_args_zapreshchaet_vsyo_vne_spiska() {
        let all = vec![
            ("code-index".to_string(), "read_file".to_string()),
            ("code-index".to_string(), "get_stats".to_string()),
            ("agents".to_string(), "fs_write_file".to_string()),
        ];
        let allowed = vec![
            "code_index__read_file".to_string(),
            "agents__fs_write_file".to_string(),
        ];
        assert_eq!(
            deny_args(&all, &allowed).expect("белый список известен серверам"),
            vec![
                "--deny".to_string(),
                "MCPTool(code_index__get_stats)".to_string()
            ],
            "в deny обязан попасть только инструмент вне белого списка"
        );

        let error = deny_args(&all, &["code_index__unknown".to_string()])
            .expect_err("имя из белого списка, которого нет у серверов, — ошибка конфига");
        assert!(error.contains("code_index__unknown"), "error={error}");
    }

    #[test]
    fn build_args_zapreshchaet_vstroennye_bez_meta_instrumentov() {
        let deny = vec![
            "--deny".to_string(),
            "MCPTool(code_index__get_stats)".to_string(),
        ];
        let extra = vec!["--reasoning-effort".to_string(), "high".to_string()];
        let disallowed: Vec<String> = BUILTIN_DISALLOWED_TOOLS
            .iter()
            .map(|name| name.to_string())
            .collect();
        let args = build_args(
            Path::new("C:/tmp/prompt.txt"),
            "grok-4.7",
            40,
            &deny,
            &extra,
            &disallowed,
        );
        let line = args.join(" ");
        assert!(line.contains("--prompt-file C:/tmp/prompt.txt"), "{line}");
        assert!(line.contains("--verbatim"), "{line}");
        assert!(line.contains("-m grok-4.7"), "{line}");
        assert!(line.contains("--output-format json"), "{line}");
        assert!(line.contains("--max-turns 40"), "{line}");
        assert!(line.contains("--no-plan"), "{line}");
        assert!(line.contains("--no-subagents"), "{line}");
        assert!(line.contains("--disable-web-search"), "{line}");
        assert!(line.contains("--trust"), "{line}");
        assert!(line.contains("--always-approve"), "{line}");
        assert!(line.contains("--tools no_builtin_tools"), "{line}");
        assert!(
            line.contains("--deny MCPTool(code_index__get_stats)"),
            "{line}"
        );
        assert!(line.contains("--reasoning-effort high"), "{line}");
        assert!(
            !line.contains("search_tool"),
            "search_tool в запретах лишает модель MCP-инструментов: {line}"
        );
        assert!(
            !line.contains("use_tool"),
            "use_tool в запретах лишает модель MCP-инструментов: {line}"
        );

        // Значение после `--disallowed-tools` сверяем по элементам списка, а не
        // подстрокой: `write_file` — подстрока `fs_write_file`, и подстрочная
        // проверка дала бы ложный результат.
        let position = args
            .iter()
            .position(|arg| arg == "--disallowed-tools")
            .expect("--disallowed-tools обязан быть в командной строке");
        let elements: Vec<&str> = args[position + 1].split(',').collect();
        assert!(
            elements.contains(&"run_terminal_cmd"),
            "внутреннее имя запуска команд обязано быть в запретах: {elements:?}"
        );
        assert!(
            elements.contains(&"spawn_subagent"),
            "подагенты обязаны быть в запретах: {elements:?}"
        );
        assert!(
            elements.contains(&"send_feedback"),
            "обратная связь подагентам обязана быть в запретах: {elements:?}"
        );
        for vymyshlennoe in ["write_file", "glob", "task"] {
            assert!(
                !elements.contains(&vymyshlennoe),
                "выдуманного имени '{vymyshlennoe}' в запретах быть не должно: {elements:?}"
            );
        }
    }

    #[test]
    fn render_grok_config_peredayot_headers_i_glushit_lichnye_servery() {
        let servers = vec![
            AgentServer {
                name: "agents".to_string(),
                url: "http://127.0.0.1:8025/mcp".to_string(),
                headers: vec![("x-agents-mcp-call".to_string(), "key-1".to_string())],
            },
            AgentServer {
                name: "code-index".to_string(),
                url: "http://127.0.0.1:8037/mcp".to_string(),
                headers: Vec::new(),
            },
        ];
        let personal = vec!["1c".to_string(), "agents".to_string()];
        let config = render_grok_config(&servers, &personal, 120);

        assert!(config.contains("[mcp_servers.code_index]"), "{config}");
        assert!(
            config.contains("url = \"http://127.0.0.1:8037/mcp\""),
            "адрес из перекрытия обязан доехать до TOML: {config}"
        );
        assert!(config.contains("tool_timeout_sec = 120"), "{config}");
        assert!(
            config.contains("headers = { \"x-agents-mcp-call\" = \"key-1\" }"),
            "без заголовков ключ вызова не дойдёт до службы: {config}"
        );
        assert!(
            config.contains("[mcp_servers.\"1c\"]"),
            "имя, начинающееся с цифры, обязано быть в кавычках: {config}"
        );
        assert!(config.contains("enabled = false"), "{config}");
        assert_eq!(
            config.matches("[mcp_servers.agents]").count(),
            1,
            "запись агента не должна затираться личной: {config}"
        );
        assert!(
            !config.contains("[mcp_servers.agents]\nenabled = false"),
            "сервер агента погашен личной записью: {config}"
        );
    }

    #[test]
    fn personal_server_names_sobirayutsya_iz_dvuh_mest() {
        let text = r#"
disabled_mcp_servers = ["postgres", "1c"]

[mcp_servers.1c]
url = "http://127.0.0.1:8037/mcp"

[mcp_servers.code-index]
url = "http://127.0.0.1:8011/mcp"
"#;
        let names = personal_server_names(text);
        assert!(names.contains(&"1c".to_string()), "{names:?}");
        assert!(names.contains(&"postgres".to_string()), "{names:?}");
        assert!(names.contains(&"code-index".to_string()), "{names:?}");
        assert_eq!(
            names.iter().filter(|name| *name == "1c").count(),
            1,
            "повторы не нужны: {names:?}"
        );

        assert!(personal_server_names("").is_empty());
        assert!(personal_server_names("не toml вовсе = =").is_empty());
    }

    #[test]
    fn parse_answer_chitaet_usage_sessiyu_i_razmyshleniya() {
        let stdout = r#"{"text":"готово","thought":"подумал","sessionId":"s-1","stopReason":"end_turn","usage":{"input_tokens":100,"output_tokens":20,"cache_read_input_tokens":30,"cache_creation_input_tokens":10},"total_cost_usd":0.5}"#;
        let answer = parse_answer(stdout).expect("годный ответ");
        assert_eq!(answer.text, "готово");
        assert_eq!(answer.thought.as_deref(), Some("подумал"));
        assert_eq!(answer.session_id.as_deref(), Some("s-1"));
        assert_eq!(answer.input_tokens, 100);
        assert_eq!(answer.output_tokens, 20);
        assert_eq!(answer.cache_read_input_tokens, 30);
        assert_eq!(answer.cache_creation_input_tokens, 10);
        assert_eq!(answer.total_cost_usd, Some(0.5));
    }

    #[test]
    fn parse_answer_otklonyaet_ne_end_turn() {
        let stdout = r#"{"text":"обрыв","stopReason":"max_tokens","usage":{"input_tokens":1}}"#;
        let error = parse_answer(stdout).expect_err("обрыв по ходам — не готовый ответ");
        assert!(error.contains("max_tokens"), "error={error}");
    }

    #[test]
    fn parse_answer_otklonyaet_pustoy_text() {
        let stdout = r#"{"text":"   ","stopReason":"end_turn"}"#;
        let error = parse_answer(stdout).expect_err("пустой ответ бесполезен");
        assert!(error.contains("text"), "error={error}");
    }

    #[test]
    fn parse_answer_otklonyaet_ne_json() {
        let error = parse_answer("начал отвечать словами").expect_err("не JSON — ошибка");
        assert!(error.contains("JSON"), "error={error}");
    }

    /// Прогон с поддельным исполняемым файлом: настоящий `grok.exe` в тестах не
    /// запускаем. Проверяем то, что видно только изнутри процесса, — окружение и
    /// рабочий каталог.
    #[tokio::test]
    async fn complete_zapuskaet_cli_vo_vremennom_kataloge_i_bez_lichnyh_podpisok() {
        let dir =
            std::env::temp_dir().join(format!("agents-mcp-grok-fake-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("каталог fake grok");
        let source = dir.join("fake_cli.rs");
        std::fs::write(
            &source,
            r#"fn main() {
    use std::io::Write;
    let exe = std::env::current_exe().unwrap();
    let probe = exe.with_file_name("probe.txt");
    let cwd = std::env::current_dir().unwrap();
    let mut f = std::fs::File::create(&probe).unwrap();
    writeln!(f, "GROK_CLAUDE_MCPS_ENABLED={}", std::env::var("GROK_CLAUDE_MCPS_ENABLED").unwrap_or_default()).unwrap();
    writeln!(f, "GROK_CURSOR_MCPS_ENABLED={}", std::env::var("GROK_CURSOR_MCPS_ENABLED").unwrap_or_default()).unwrap();
    writeln!(f, "cwd={}", cwd.display()).unwrap();
    print!("{{\"text\":\"готово\",\"thought\":\"\",\"sessionId\":\"fixture-session\",\"stopReason\":\"end_turn\",\"usage\":{{\"input_tokens\":100,\"output_tokens\":20,\"cache_read_input_tokens\":30,\"cache_creation_input_tokens\":10}},\"total_cost_usd\":0.25}}");
}
"#,
        )
        .expect("исходник fake grok");
        let executable = dir.join(if cfg!(windows) {
            "fake_cli.exe"
        } else {
            "fake_cli"
        });
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let compiled = std::process::Command::new(rustc)
            .arg(&source)
            .arg("-o")
            .arg(&executable)
            .status()
            .expect("запуск rustc");
        assert!(compiled.success(), "fake grok должен собраться");

        let provider = GrokCliProvider::new(executable.clone(), None, 1, 40, 120);
        let request = LlmRequest {
            model: "fake-model".into(),
            system_prompt: "prompt".into(),
            user_input: String::new(),
            temperature: 0.0,
            max_tokens: 10,
            top_p: None,
            extra_body: serde_json::Map::new(),
            timeout: Duration::from_secs(30),
            // Пустой mcp_config: `tools/list` не зовётся вовсе, тест не ходит в сеть.
            cli_hints: Some(ClaudeCliHints {
                cwd: Some(dir.clone()),
                mcp_config: None,
                allowed_tools: Vec::new(),
                ..Default::default()
            }),
            turn_sink: None,
            fallback_skill_names: Vec::new(),
            prompt_skill_names: Vec::new(),
            skills: None,
        };
        let response = provider
            .complete(request)
            .await
            .expect("поддельный CLI печатает готовый ответ");
        assert_eq!(response.content, "готово");
        assert_eq!(response.session_id.as_deref(), Some("fixture-session"));
        assert_eq!(response.tokens_in, 100);
        assert_eq!(response.tokens_out, 20);
        assert_eq!(response.cache_read_input_tokens, 30);
        assert_eq!(response.cache_creation_input_tokens, 10);
        assert_eq!(response.raw_input_tokens, 60);
        assert!(
            response.reasoning.is_none(),
            "пустое thought — не размышления"
        );

        let probe = std::fs::read_to_string(executable.with_file_name("probe.txt"))
            .expect("поддельный CLI обязан оставить пробу");
        assert!(
            probe.contains("GROK_CLAUDE_MCPS_ENABLED=false"),
            "личные подписки Claude обязаны быть выключены: {probe}"
        );
        assert!(
            probe.contains("GROK_CURSOR_MCPS_ENABLED=false"),
            "личные подписки Cursor обязаны быть выключены: {probe}"
        );
        let cwd = probe
            .lines()
            .find_map(|line| line.strip_prefix("cwd="))
            .expect("в пробе нет рабочего каталога");
        assert_ne!(
            cwd,
            dir.to_string_lossy().as_ref(),
            "рабочим каталогом процесса не может быть cwd из hints"
        );
        assert!(
            cwd.contains(TMP_PREFIX),
            "рабочий каталог процесса — временный каталог вызова: {cwd}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }
}
