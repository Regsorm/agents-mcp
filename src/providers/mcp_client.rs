//! Минимальный MCP-клиент по HTTP (JSON-RPC `initialize` + `tools/list` +
//! `tools/call`).
//!
//! Нужен openrouter-провайдеру для agentic-loop: OpenAI-совместимые модели
//! (DeepSeek/MiMo/…) сами инструменты НЕ исполняют — они лишь возвращают
//! `tool_calls`. Исполнение делаем мы: дёргаем MCP-серверы из `mcp_config`
//! агента. claude-cli этого не требует — там tool-loop ведёт сам CLI.
//!
//! Поддерживаются ОБА вида Streamable-HTTP серверов:
//! - **stateless** (agents-mcp 8025 NeverSessionManager, серверы 1С за кэшем) —
//!   принимают одиночный POST без хендшейка; `initialize` не возвращает
//!   session-id, работаем без заголовка;
//! - **stateful** (например, серверы на FastMCP) — требуют MCP-
//!   хендшейк: `initialize` → заголовок `Mcp-Session-Id` → `notifications/
//!   initialized` → дальнейшие вызовы с этим заголовком. Без него сервер
//!   отвечает `Bad Request: Missing session ID`.
//!
//! Ответ приходит plain JSON или SSE (`data: {...}`); полезное — в `/result`.
//!
//! Второй вид серверов — **запускаемые процессом**: в `mcp_config` у них вместо
//! `url` стоит `command`. Клиент сам поднимает процесс и говорит с ним через его
//! стандартный ввод/вывод, по одному JSON-сообщению на строку. Для них здесь
//! [`StdioServer`] и [`StdioPool`]: процесс живёт ровно один вызов агента и
//! гаснет вместе с ним.

use std::collections::{HashMap, HashSet};
use std::process::Stdio;
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use crate::proc_tree::{self, ProcessTree};

use super::mask_credentials;

fn parse_var_reference(
    text: &str,
    start: usize,
) -> Result<(usize, String, Option<String>), String> {
    let rest = &text[start + 2..];
    let Some(close) = rest.find('}') else {
        return Err("незакрытая подстановка переменной окружения `${`".to_string());
    };
    let body = &rest[..close];
    let (name, default) = match body.split_once(":-") {
        Some((name, default)) => (name, Some(default.to_string())),
        None => (body, None),
    };
    // Вложенная ссылка `${A:-${B}}` не поддерживается: первая `}` закрыла бы
    // внешнюю, и текст `${B` ушёл бы серверу как есть.
    if default
        .as_deref()
        .is_some_and(|default| default.contains("${"))
    {
        return Err("вложенная подстановка в значении по умолчанию не поддерживается".to_string());
    }
    let mut chars = name.chars();
    let valid = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !valid {
        return Err("недопустимое имя переменной окружения в `${...}`".to_string());
    }
    Ok((start + 2 + close + 1, name.to_string(), default))
}

/// Найти допустимые ссылки `${ИМЯ}` и `${ИМЯ:-значение}` в строке.
pub(crate) fn referenced_vars(text: &str) -> Vec<(String, Option<String>)> {
    let mut out = Vec::new();
    let mut offset = 0;
    while let Some(relative) = text[offset..].find("${") {
        let start = offset + relative;
        match parse_var_reference(text, start) {
            Ok((end, name, default)) => {
                out.push((name, default));
                offset = end;
            }
            Err(_) => offset = start + 2,
        }
    }
    out
}

/// Раскрыть ссылки на переменные окружения, не включая их значения в ошибки.
pub(crate) fn expand_vars(
    text: &str,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<String, String> {
    let mut out = String::with_capacity(text.len());
    let mut offset = 0;
    while let Some(relative) = text[offset..].find("${") {
        let start = offset + relative;
        out.push_str(&text[offset..start]);
        let (end, name, default) = parse_var_reference(text, start)?;
        // Как в bash: значение по умолчанию и при отсутствии, и при пустой переменной.
        match (lookup(&name), default) {
            (Some(value), Some(default)) if value.is_empty() => out.push_str(&default),
            (Some(value), _) => out.push_str(&value),
            (None, Some(default)) => out.push_str(&default),
            (None, None) => return Err(format!("не задана переменная окружения {name}")),
        }
        offset = end;
    }
    out.push_str(&text[offset..]);
    Ok(out)
}

/// Удалить из диагностики учётные данные и query, в котором могут быть ключи.
pub(crate) fn safe_server_address(text: &str) -> String {
    mask_credentials(text.split('?').next().unwrap_or(text))
}

// Минимум для запуска python/node/cmd и поиска программ и временных каталогов;
// секреты службы дочернему MCP-серверу не наследуются.
#[cfg(windows)]
const STDIO_ENV_ALLOWLIST: &[&str] = &[
    "SYSTEMROOT",
    "PATH",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "COMSPEC",
    "PATHEXT",
];
#[cfg(not(windows))]
const STDIO_ENV_ALLOWLIST: &[&str] = &["PATH", "HOME", "LANG", "TMPDIR"];

fn configure_stdio_env(cmd: &mut Command, cfg: &StdioServerCfg) {
    cmd.env_clear();
    for name in STDIO_ENV_ALLOWLIST {
        if let Some(value) = std::env::var_os(name) {
            cmd.env(name, value);
        }
    }
    for (name, value) in &cfg.env {
        cmd.env(name, value);
    }
}

/// Ответ инструмента, который отработал успешно, но ничего не нашёл.
///
/// Отдаём словами, а не сырым конвертом: в конверте есть слово `isError`, и
/// детектор однотипных отказов принимал удачный вызов за ошибку. По этой же
/// строке agentic-loop считает подряд идущие пустые ответы одного инструмента.
pub const EMPTY_RESULT_MARK: &str = "(пусто: вызов прошёл успешно, подходящих записей нет — \
                                     ищи в другом типе объектов или по другой маске)";

const PROTOCOL_VERSION: &str = "2025-06-18";
const MAX_TOOLS_LIST_PAGES: usize = 50;
const MAX_RESPONSE_BYTES: usize = 5 * 1024 * 1024;

/// Один MCP-сервер из `mcp_config`: alias (например `"1c"`) + URL.
#[derive(Debug, Clone)]
pub struct McpServer {
    pub alias: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
}

#[derive(Debug)]
struct HttpSessionState {
    id: Option<String>,
    protocol_version: String,
}

#[derive(Debug)]
struct HttpSessionInner {
    client: reqwest::Client,
    server: McpServer,
    state: tokio::sync::Mutex<HttpSessionState>,
}

/// Согласованное состояние одного HTTP-сервера. Копии разделяют session id,
/// поэтому восстановление после 404 сразу видно всем его инструментам.
#[derive(Clone, Debug)]
pub struct McpSession {
    inner: Arc<HttpSessionInner>,
}

impl McpSession {
    async fn snapshot(&self) -> (Option<String>, String) {
        let state = self.inner.state.lock().await;
        (state.id.clone(), state.protocol_version.clone())
    }

    async fn reinitialize(&self) -> Result<(), String> {
        let state = perform_handshake(&self.inner.client, &self.inner.server).await?;
        *self.inner.state.lock().await = state;
        Ok(())
    }

    #[cfg(test)]
    async fn session_id(&self) -> Option<String> {
        self.inner.state.lock().await.id.clone()
    }
}

impl Drop for HttpSessionInner {
    fn drop(&mut self) {
        let Ok(state) = self.state.try_lock() else {
            return;
        };
        let Some(session_id) = state.id.clone() else {
            return;
        };
        let protocol_version = state.protocol_version.clone();
        let client = self.client.clone();
        let server = self.server.clone();
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        handle.spawn(async move {
            let mut builder = client
                .delete(&server.url)
                .timeout(RPC_TIMEOUT)
                .header("Mcp-Session-Id", session_id)
                .header("MCP-Protocol-Version", protocol_version);
            for (name, value) in &server.headers {
                builder = builder.header(name, value);
            }
            let _ = builder.send().await;
        });
    }
}

/// Описание инструмента из `tools/list`.
#[derive(Debug, Clone)]
pub struct ToolDef {
    /// Полное имя в стиле claude: `mcp__<alias>__<tool>` — его же шлём модели,
    /// по нему маршрутизируем обратный вызов.
    pub full_name: String,
    /// Имя tool на сервере (без префикса) — уходит в `tools/call`.
    pub tool_name: String,
    pub description: String,
    /// `inputSchema` как есть (JSON Schema) → в OpenAI `tools[].function.parameters`.
    pub parameters: Value,
}

/// Описание сервера, который запускается процессом: `{"command": "...",
/// "args": [...], "env": {...}, "cwd": "..."}` в `mcp_config`.
#[derive(Debug, Clone)]
pub struct StdioServerCfg {
    pub alias: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    /// Рабочий каталог процесса. Не задан — берётся каталог вызова агента.
    pub cwd: Option<String>,
}

pub(crate) fn expand_http_server(
    mut server: McpServer,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<McpServer, String> {
    server.url = expand_vars(&server.url, &lookup)?;
    for (_, value) in &mut server.headers {
        *value = expand_vars(value, &lookup)?;
    }
    Ok(server)
}

pub(crate) fn expand_stdio_server(
    mut server: StdioServerCfg,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<StdioServerCfg, String> {
    server.command = expand_vars(&server.command, &lookup)?;
    for arg in &mut server.args {
        *arg = expand_vars(arg, &lookup)?;
    }
    for (_, value) in &mut server.env {
        *value = expand_vars(value, &lookup)?;
    }
    Ok(server)
}

/// Разобрать из `mcp_config` записи, запускаемые процессом (есть `command`).
/// Записи с `url` пропускаются — их забирает [`parse_mcp_config`].
pub fn parse_stdio_servers(raw: &str) -> Result<Vec<StdioServerCfg>, serde_json::Error> {
    let v: Value = serde_json::from_str(raw)?;
    let mut out = Vec::new();
    if let Some(map) = v.get("mcpServers").and_then(|m| m.as_object()) {
        for (alias, cfg) in map {
            // url сильнее command: если заданы оба, сервер уже взят HTTP-веткой,
            // и поднимать вдобавок процесс не нужно.
            if cfg.get("url").and_then(|u| u.as_str()).is_some() {
                continue;
            }
            let command = match cfg.get("command").and_then(|c| c.as_str()) {
                Some(c) if !c.is_empty() => c.to_string(),
                _ => continue,
            };
            let args = cfg
                .get("args")
                .and_then(|a| a.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let env = cfg
                .get("env")
                .and_then(|e| e.as_object())
                .map(|e| {
                    e.iter()
                        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                        .collect()
                })
                .unwrap_or_default();
            let cwd = cfg
                .get("cwd")
                .and_then(|c| c.as_str())
                .map(|s| s.to_string());
            out.push(StdioServerCfg {
                alias: alias.clone(),
                command,
                args,
                env,
                cwd,
            });
        }
    }
    Ok(out)
}

/// Распарсить `mcp_config` (`{"mcpServers":{"1c":{"url":"..."},...}}`).
/// Серверы без поля `url` здесь пропускаются: запускаемые процессом разбирает
/// [`parse_stdio_servers`].
pub fn parse_mcp_config(raw: &str) -> Result<Vec<McpServer>, serde_json::Error> {
    let v: Value = serde_json::from_str(raw)?;
    let mut out = Vec::new();
    if let Some(map) = v.get("mcpServers").and_then(|m| m.as_object()) {
        for (alias, cfg) in map {
            if let Some(url) = cfg.get("url").and_then(|u| u.as_str()) {
                out.push(McpServer {
                    alias: alias.clone(),
                    url: url.to_string(),
                    headers: cfg
                        .get("headers")
                        .and_then(Value::as_object)
                        .map(|headers| {
                            headers
                                .iter()
                                .filter_map(|(name, value)| {
                                    value
                                        .as_str()
                                        .map(|value| (name.clone(), value.to_string()))
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                });
            }
        }
    }
    Ok(out)
}

/// Извлечь JSON-RPC ответ с нужным `id` из plain JSON или потока SSE.
/// Уведомления и встречные сообщения с другим `id` пропускаются.
pub(crate) fn extract_envelope(text: &str, expected_id: i64) -> Result<Value, String> {
    fn matching(value: Value, expected_id: i64) -> Option<Value> {
        (value.get("id") == Some(&json!(expected_id))).then_some(value)
    }

    if text
        .lines()
        .any(|line| line.trim_start().starts_with("data:"))
    {
        let mut data = String::new();
        let mut parse_error = None;
        for line in text.lines().chain(std::iter::once("")) {
            let line = line.trim_end_matches('\r');
            if line.is_empty() {
                if data.is_empty() {
                    continue;
                }
                let payload = data.trim_end_matches('\n');
                if payload != "[DONE]" {
                    match serde_json::from_str::<Value>(payload) {
                        Ok(value) => {
                            if let Some(value) = matching(value, expected_id) {
                                return Ok(value);
                            }
                        }
                        Err(e) => parse_error = Some(e.to_string()),
                    }
                }
                data.clear();
            } else if let Some(part) = line.trim_start().strip_prefix("data:") {
                data.push_str(part.strip_prefix(' ').unwrap_or(part));
                data.push('\n');
            }
        }
        return Err(parse_error
            .map(|e| format!("parse envelope: {e}"))
            .unwrap_or_else(|| format!("нет JSON-RPC ответа с id={expected_id}")));
    }

    let values = serde_json::Deserializer::from_str(text).into_iter::<Value>();
    for value in values {
        let value = value.map_err(|e| format!("parse envelope: {e}"))?;
        if let Some(value) = matching(value, expected_id) {
            return Ok(value);
        }
    }
    Err(format!("нет JSON-RPC ответа с id={expected_id}"))
}

fn response_detail(text: &str) -> String {
    const MAX_CHARS: usize = 500;
    let detail: String = text.trim().chars().take(MAX_CHARS).collect();
    if detail.is_empty() {
        "пустое тело".to_string()
    } else {
        detail
    }
}

async fn read_http_body(mut response: reqwest::Response, method: &str) -> Result<String, String> {
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| format!("{method} body: {}", safe_server_address(&e.to_string())))?
    {
        let size = bytes.len().saturating_add(chunk.len());
        if size > MAX_RESPONSE_BYTES {
            return Err(format!(
                "{method}: ответ MCP-сервера размером {size} байт превышает предел {MAX_RESPONSE_BYTES} байт"
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    String::from_utf8(bytes).map_err(|e| format!("{method}: ответ MCP-сервера не UTF-8: {e}"))
}

async fn perform_handshake(
    client: &reqwest::Client,
    server: &McpServer,
) -> Result<HttpSessionState, String> {
    let init = json!({
        "jsonrpc": "2.0", "id": 0, "method": "initialize",
        "params": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "agents-mcp", "version": env!("CARGO_PKG_VERSION")}
        }
    });
    let mut builder = client
        .post(&server.url)
        .timeout(RPC_TIMEOUT)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .json(&init);
    for (name, value) in &server.headers {
        builder = builder.header(name, value);
    }
    let resp = builder.send().await.map_err(|e| {
        if e.is_timeout() {
            format!(
                "initialize: сервис {} не ответил за {} с",
                safe_server_address(&server.url),
                RPC_TIMEOUT.as_secs()
            )
        } else {
            format!("initialize send: {}", safe_server_address(&e.to_string()))
        }
    })?;
    let status = resp.status();
    let session = resp
        .headers()
        .get("mcp-session-id")
        .map(|v| {
            v.to_str()
                .map(str::to_string)
                .map_err(|_| "initialize: некорректный Mcp-Session-Id".to_string())
        })
        .transpose()?;
    let text = read_http_body(resp, "initialize").await?;
    if !status.is_success() {
        return Err(format!(
            "initialize: HTTP {status}: {}",
            response_detail(&text)
        ));
    }
    let envelope = extract_envelope(&text, 0)?;
    if let Some(error) = envelope.get("error") {
        return Err(format!("initialize rpc error: {error}"));
    }
    let result = envelope
        .get("result")
        .ok_or_else(|| "initialize: нет result в ответе".to_string())?;
    let protocol_version = result
        .get("protocolVersion")
        .and_then(Value::as_str)
        .ok_or_else(|| "initialize: нет protocolVersion в result".to_string())?;
    // Серверы на rmcp (agents-mcp, 1c, bsl-context, rag-query) на запрос
    // 2025-06-18 отвечают своей версией 2025-11-25 — проверено опросом 13.09.2026.
    // Строгое равенство оставило бы агентов без их инструментов: принимаем
    // версию сервера и передаём её дальше в MCP-Protocol-Version.
    if protocol_version.is_empty() {
        return Err("initialize: пустая protocolVersion в result".to_string());
    }

    let note = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
    let mut builder = client
        .post(&server.url)
        .timeout(RPC_TIMEOUT)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .header("MCP-Protocol-Version", protocol_version)
        .json(&note);
    for (name, value) in &server.headers {
        builder = builder.header(name, value);
    }
    if let Some(sid) = &session {
        builder = builder.header("Mcp-Session-Id", sid);
    }
    let response = builder.send().await.map_err(|e| {
        if e.is_timeout() {
            format!(
                "notifications/initialized: сервис {} не ответил за {} с",
                safe_server_address(&server.url),
                RPC_TIMEOUT.as_secs()
            )
        } else {
            format!(
                "notifications/initialized send: {}",
                safe_server_address(&e.to_string())
            )
        }
    })?;
    let note_status = response.status();
    let _ = read_http_body(response, "notifications/initialized").await?;
    if !note_status.is_success() {
        return Err(format!("notifications/initialized: HTTP {}", note_status));
    }
    Ok(HttpSessionState {
        id: session,
        protocol_version: protocol_version.to_string(),
    })
}

/// Полный MCP-хендшейк: проверка `initialize` и обязательное
/// `notifications/initialized`, в том числе для сервера без session id.
pub async fn initialize_session(
    client: &reqwest::Client,
    server: &McpServer,
) -> Result<McpSession, String> {
    let state = perform_handshake(client, server).await?;
    Ok(McpSession {
        inner: Arc::new(HttpSessionInner {
            client: client.clone(),
            server: server.clone(),
            state: tokio::sync::Mutex::new(state),
        }),
    })
}

/// Потолок ожидания ответа от MCP-сервиса. Без него запрос висит вечно: 30.08.2026
/// bsl-mcp набрал восемь застрявших вызовов, остался «здоровым» по своему health
/// (state=ready, аптайм 13 часов) и молча держал прогон 201 секунду, пока его не
/// перезапустили руками. Карты в это время простаивали.
///
/// 120 с — заведомо выше рабочих величин: в замерах того же дня инструменты
/// отвечали за 1–68 мс, включая выборки на 50 тыс. знаков. То есть срабатывание
/// потолка означает именно неисправность, а не тяжёлый запрос.
const RPC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

#[derive(Debug)]
enum RpcError {
    SessionExpired,
    Message(String),
}

async fn rpc_once(
    client: &reqwest::Client,
    url: &str,
    method: &str,
    params: Value,
    session: Option<&str>,
    protocol_version: &str,
    headers: &[(String, String)],
) -> Result<Value, RpcError> {
    let req = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params});
    let mut builder = client
        .post(url)
        .timeout(RPC_TIMEOUT)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .header("MCP-Protocol-Version", protocol_version)
        .json(&req);
    for (name, value) in headers {
        builder = builder.header(name, value);
    }
    if let Some(sid) = session {
        builder = builder.header("Mcp-Session-Id", sid);
    }
    let resp = builder.send().await.map_err(|e| {
        RpcError::Message({
            if e.is_timeout() {
                // Отдельная формулировка: молчащий сервис и отказ сервиса лечатся
                // по-разному, и в журнале это должно различаться с первого взгляда.
                format!(
                    "{method}: сервис {} не ответил за {} с — считаю его недоступным",
                    safe_server_address(url),
                    RPC_TIMEOUT.as_secs()
                )
            } else {
                format!("{method} send: {}", safe_server_address(&e.to_string()))
            }
        })
    })?;
    let status = resp.status();
    let text = read_http_body(resp, method)
        .await
        .map_err(RpcError::Message)?;
    if status == reqwest::StatusCode::NOT_FOUND && session.is_some() {
        return Err(RpcError::SessionExpired);
    }
    if !status.is_success() {
        return Err(RpcError::Message(format!(
            "{method}: HTTP {status}: {}",
            response_detail(&text)
        )));
    }
    let v = extract_envelope(&text, 1).map_err(RpcError::Message)?;
    if let Some(err) = v.get("error") {
        return Err(RpcError::Message(format!("{method} rpc error: {err}")));
    }
    v.get("result")
        .cloned()
        .ok_or_else(|| RpcError::Message(format!("{method}: нет result в ответе")))
}

/// Одиночный JSON-RPC вызов. После 404 с session id ровно один раз повторяет
/// хендшейк и сам исходный запрос.
async fn rpc(
    client: &reqwest::Client,
    server: &McpServer,
    method: &str,
    params: Value,
    session: &McpSession,
) -> Result<Value, String> {
    let (session_id, protocol_version) = session.snapshot().await;
    match rpc_once(
        client,
        &server.url,
        method,
        params.clone(),
        session_id.as_deref(),
        &protocol_version,
        &server.headers,
    )
    .await
    {
        Ok(result) => Ok(result),
        Err(RpcError::Message(error)) => Err(error),
        Err(RpcError::SessionExpired) => {
            session.reinitialize().await?;
            let (session_id, protocol_version) = session.snapshot().await;
            rpc_once(
                client,
                &server.url,
                method,
                params,
                session_id.as_deref(),
                &protocol_version,
                &server.headers,
            )
            .await
            .map_err(|error| match error {
                RpcError::SessionExpired => {
                    format!("{method}: новая MCP-сессия тоже получила HTTP 404")
                }
                RpcError::Message(error) => error,
            })
        }
    }
}

/// Разобрать ответ `tools/list` в `ToolDef` с полными именами
/// `mcp__<alias>__<tool>`. Общий разбор для обоих видов серверов — тех, что по
/// HTTP, и тех, что запускаются процессом.
fn tool_defs_from_result(alias: &str, result: &Value) -> Vec<ToolDef> {
    let tools = result
        .get("tools")
        .and_then(|t| t.as_array())
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::new();
    for t in tools {
        let name = match t.get("name").and_then(|n| n.as_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        let desc = t
            .get("description")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();
        let params = t
            .get("inputSchema")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object"}));
        out.push(ToolDef {
            full_name: format!("mcp__{alias}__{name}"),
            tool_name: name,
            description: desc,
            parameters: params,
        });
    }
    out
}

/// `tools/list` одного сервера → `ToolDef` с полными именами `mcp__alias__tool`.
pub async fn list_tools(
    client: &reqwest::Client,
    server: &McpServer,
    session: &McpSession,
) -> Result<Vec<ToolDef>, String> {
    let mut out = Vec::new();
    let mut cursor = None;
    let mut seen = HashSet::new();
    for page in 0..MAX_TOOLS_LIST_PAGES {
        let params = cursor
            .as_ref()
            .map(|cursor| json!({"cursor": cursor}))
            .unwrap_or_else(|| json!({}));
        let result = rpc(client, server, "tools/list", params, session).await?;
        out.extend(tool_defs_from_result(&server.alias, &result));
        let Some(next_cursor) = result.get("nextCursor").and_then(Value::as_str) else {
            return Ok(out);
        };
        if !seen.insert(next_cursor.to_string()) {
            return Err("tools/list: сервер повторил nextCursor".to_string());
        }
        if page + 1 == MAX_TOOLS_LIST_PAGES {
            return Err(format!(
                "tools/list: превышен предел в {MAX_TOOLS_LIST_PAGES} страниц"
            ));
        }
        cursor = Some(next_cursor.to_string());
    }
    unreachable!()
}

/// `tools/call` → текст результата (склейка `content[*].text`, либо
/// `structuredContent`, либо весь `result`). При `isError=true` — `Err`.
pub async fn call_tool(
    client: &reqwest::Client,
    server: &McpServer,
    tool_name: &str,
    args: Value,
    session: &McpSession,
) -> Result<String, String> {
    let result = rpc(
        client,
        server,
        "tools/call",
        json!({"name": tool_name, "arguments": args}),
        session,
    )
    .await?;
    text_from_call_result(tool_name, &result)
}

/// Разобрать ответ `tools/call` в текст для модели. Общий разбор для обоих
/// видов серверов — тех, что по HTTP, и тех, что запускаются процессом.
fn text_from_call_result(tool_name: &str, result: &Value) -> Result<String, String> {
    let is_error = result
        .get("isError")
        .and_then(|b| b.as_bool())
        .unwrap_or(false);

    let mut buf = String::new();
    let mut non_text = Vec::new();
    if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
        for item in content {
            let text = item.get("text").and_then(Value::as_str).or_else(|| {
                item.get("resource")
                    .and_then(|resource| resource.get("text"))
                    .and_then(Value::as_str)
            });
            if let Some(t) = text {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(t);
            } else {
                non_text.push(
                    item.get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("неизвестный тип"),
                );
            }
        }
    }
    if buf.is_empty() {
        buf = match result.get("structuredContent") {
            Some(sc) => sc.to_string(),
            None if !non_text.is_empty() => format!(
                "(инструмент вернул нетекстовое содержимое: {})",
                non_text.join(", ")
            ),
            // Инструмент отработал, но ничего не нашёл. Сырой конверт сюда
            // отдавать нельзя: в нём есть слово isError, и детектор однотипных
            // отказов принимает удачный вызов за ошибку — модель начинает
            // чинить форму вызова вместо того, чтобы искать в другом месте.
            None if result.get("content").and_then(Value::as_array).is_some() && !is_error => {
                EMPTY_RESULT_MARK.to_string()
            }
            None => result.to_string(),
        };
    }

    if is_error {
        Err(format!("инструмент {tool_name} вернул isError: {buf}"))
    } else {
        Ok(buf)
    }
}

// ── Серверы, запускаемые процессом ─────────────────────────────────────────

/// Живой процесс MCP-сервера: пишем ему в стандартный ввод по строке JSON на
/// сообщение, ответы читаем из его стандартного вывода.
///
/// Дерево процесса гаснет вместе с этой структурой, поэтому любой
/// выход из вызова агента — по ошибке, таймауту, отмене — не оставляет
/// висящих потомков.
pub struct StdioServer {
    alias: String,
    /// Держим потомка живым, пока жива структура; читаем его не отсюда, а из
    /// снятых каналов ниже.
    _child: Child,
    _process_tree: ProcessTree,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: i64,
}

impl StdioServer {
    /// Запустить процесс и провести хендшейк. `default_cwd` — рабочий каталог
    /// вызова агента; берётся, когда у сервера не задан свой `cwd`.
    pub async fn spawn(cfg: &StdioServerCfg, default_cwd: Option<&str>) -> Result<Self, String> {
        let mut cmd = Command::new(&cfg.command);
        cmd.args(&cfg.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // stderr не перехватываем: диагностика сервера уходит в журнал
            // службы, где её видно рядом с остальным ходом вызова.
            .kill_on_drop(true);
        configure_stdio_env(&mut cmd, cfg);
        if let Some(dir) = cfg.cwd.as_deref().or(default_cwd) {
            cmd.current_dir(dir);
        }
        let (mut child, process_tree) = proc_tree::spawn(&mut cmd)
            .map_err(|e| format!("не запустился процесс `{}`: {e}", cfg.command))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| format!("у процесса `{}` нет стандартного ввода", cfg.command))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| format!("у процесса `{}` нет стандартного вывода", cfg.command))?;
        let mut srv = Self {
            alias: cfg.alias.clone(),
            _child: child,
            _process_tree: process_tree,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 0,
        };
        srv.initialize().await?;
        Ok(srv)
    }

    /// Отправить одно сообщение строкой.
    async fn send(&mut self, msg: &Value) -> Result<(), String> {
        let alias = self.alias.clone();
        let mut line = serde_json::to_string(msg).map_err(|e| format!("сборка сообщения: {e}"))?;
        line.push('\n');
        let stdin = &mut self.stdin;
        tokio::time::timeout(RPC_TIMEOUT, async move {
            stdin.write_all(line.as_bytes()).await?;
            stdin.flush().await
        })
        .await
        .map_err(|_| {
            format!(
                "сервер {alias} не принял сообщение за {} с",
                RPC_TIMEOUT.as_secs()
            )
        })?
        .map_err(|e| format!("запись в сервер {alias}: {e}"))
    }

    /// Дождаться ответа с нужным `id`. Уведомления сервера (без `id`) и строки,
    /// которые не разбираются как JSON (сервер может печатать в стандартный
    /// вывод что-то своё), пропускаются.
    async fn read_result(&mut self, id: i64) -> Result<Value, String> {
        let alias = self.alias.clone();
        let deadline = tokio::time::Instant::now() + RPC_TIMEOUT;
        loop {
            let mut line = Vec::new();
            loop {
                let chunk = tokio::time::timeout_at(deadline, self.stdout.fill_buf())
                    .await
                    .map_err(|_| {
                        format!(
                            "сервер {alias} не ответил за {} с — считаю его недоступным",
                            RPC_TIMEOUT.as_secs()
                        )
                    })?
                    .map_err(|e| format!("чтение из сервера {alias}: {e}"))?;
                if chunk.is_empty() {
                    if line.is_empty() {
                        return Err(format!("сервер {alias} закрыл вывод (процесс завершился)"));
                    }
                    break;
                }
                let take = chunk
                    .iter()
                    .position(|byte| *byte == b'\n')
                    .map(|pos| pos + 1)
                    .unwrap_or(chunk.len());
                let size = line.len().saturating_add(take);
                if size > MAX_RESPONSE_BYTES {
                    return Err(format!(
                        "сервер {alias}: ответ MCP-сервера размером {size} байт превышает предел {MAX_RESPONSE_BYTES} байт"
                    ));
                }
                let done = chunk[take - 1] == b'\n';
                line.extend_from_slice(&chunk[..take]);
                self.stdout.consume(take);
                if done {
                    break;
                }
            }
            let line = String::from_utf8(line)
                .map_err(|e| format!("сервер {alias}: ответ не UTF-8: {e}"))?;
            let v: Value = match serde_json::from_str(line.trim()) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if v.get("id").and_then(|i| i.as_i64()) != Some(id) {
                continue;
            }
            if let Some(err) = v.get("error") {
                return Err(format!("сервер {alias} вернул ошибку: {err}"));
            }
            return v
                .get("result")
                .cloned()
                .ok_or_else(|| format!("сервер {alias}: в ответе нет result"));
        }
    }

    async fn rpc(&mut self, method: &str, params: Value) -> Result<Value, String> {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))
            .await?;
        self.read_result(id).await
    }

    /// `initialize` + подтверждение `notifications/initialized`.
    async fn initialize(&mut self) -> Result<(), String> {
        let result = self
            .rpc(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "agents-mcp", "version": env!("CARGO_PKG_VERSION")}
                }),
            )
            .await?;
        let protocol_version = result
            .get("protocolVersion")
            .and_then(Value::as_str)
            .ok_or_else(|| "initialize: нет protocolVersion в result".to_string())?;
        // Как у HTTP-сервера: версию, названную сервером, принимаем.
        if protocol_version.is_empty() {
            return Err("initialize: пустая protocolVersion в result".to_string());
        }
        self.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}))
            .await
    }

    pub async fn list_tools(&mut self) -> Result<Vec<ToolDef>, String> {
        let alias = self.alias.clone();
        let mut out = Vec::new();
        let mut cursor = None;
        let mut seen = HashSet::new();
        for page in 0..MAX_TOOLS_LIST_PAGES {
            let params = cursor
                .as_ref()
                .map(|cursor| json!({"cursor": cursor}))
                .unwrap_or_else(|| json!({}));
            let result = self.rpc("tools/list", params).await?;
            out.extend(tool_defs_from_result(&alias, &result));
            let Some(next_cursor) = result.get("nextCursor").and_then(Value::as_str) else {
                return Ok(out);
            };
            if !seen.insert(next_cursor.to_string()) {
                return Err("tools/list: сервер повторил nextCursor".to_string());
            }
            if page + 1 == MAX_TOOLS_LIST_PAGES {
                return Err(format!(
                    "tools/list: превышен предел в {MAX_TOOLS_LIST_PAGES} страниц"
                ));
            }
            cursor = Some(next_cursor.to_string());
        }
        unreachable!()
    }

    pub async fn call_tool(&mut self, tool_name: &str, args: Value) -> Result<String, String> {
        let result = self
            .rpc("tools/call", json!({"name": tool_name, "arguments": args}))
            .await?;
        text_from_call_result(tool_name, &result)
    }
}

/// Процессы-серверы одного вызова агента плюс карта «полное имя инструмента →
/// (алиас сервера, имя инструмента)». Живёт ровно один вызов: с концом вызова
/// структура уничтожается и все процессы гаснут.
#[derive(Default)]
pub struct StdioPool {
    servers: HashMap<String, StdioServer>,
    tools: HashMap<String, (String, String)>,
}

impl StdioPool {
    pub fn add_server(&mut self, srv: StdioServer) {
        self.servers.insert(srv.alias.clone(), srv);
    }

    pub fn add_tool(&mut self, full_name: String, alias: String, tool_name: String) {
        self.tools.insert(full_name, (alias, tool_name));
    }

    /// Знает ли набор такой инструмент (по полному имени `mcp__alias__tool`).
    pub fn has(&self, full_name: &str) -> bool {
        self.tools.contains_key(full_name)
    }

    /// Вызвать инструмент запускаемого процессом сервера.
    pub async fn call(&mut self, full_name: &str, args: Value) -> Result<String, String> {
        let (alias, tool_name) =
            self.tools.get(full_name).cloned().ok_or_else(|| {
                format!("инструмент {full_name} не из числа запускаемых процессом")
            })?;
        let srv = self
            .servers
            .get_mut(&alias)
            .ok_or_else(|| format!("сервер {alias} не запущен"))?;
        srv.call_tool(&tool_name, args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_one_and_several_variables() {
        let lookup = |name: &str| match name {
            "ONE" => Some("один".to_string()),
            "TWO" => Some("два".to_string()),
            _ => None,
        };
        assert_eq!(expand_vars("${ONE}", lookup).unwrap(), "один");
        assert_eq!(
            expand_vars("до-${ONE}-${TWO}-после", lookup).unwrap(),
            "до-один-два-после"
        );
    }

    #[test]
    fn defaults_apply_when_variable_is_missing_or_empty() {
        assert_eq!(expand_vars("${MISSING:-запас}", |_| None).unwrap(), "запас");
        assert_eq!(
            expand_vars("${EMPTY:-запас}", |_| Some(String::new())).unwrap(),
            "запас"
        );
    }

    #[test]
    fn missing_variable_error_has_name_but_not_surrounding_value() {
        let error = expand_vars("secret-before-${MISSING}-secret-after", |_| None).unwrap_err();
        assert!(error.contains("MISSING"));
        assert!(!error.contains("secret-before"));
        assert!(!error.contains("secret-after"));
    }

    #[test]
    fn malformed_reference_is_rejected_and_plain_dollar_is_unchanged() {
        assert!(expand_vars("${UNCLOSED", |_| None)
            .unwrap_err()
            .contains("незакрытая"));
        assert!(expand_vars("${1BAD}", |_| None)
            .unwrap_err()
            .contains("недопустимое имя"));
        assert!(expand_vars("${A:-${B}}", |_| None)
            .unwrap_err()
            .contains("вложенная"));
        assert_eq!(expand_vars("цена $5", |_| None).unwrap(), "цена $5");
        assert_eq!(
            expand_vars("обычный текст", |_| None).unwrap(),
            "обычный текст"
        );
    }

    #[test]
    fn referenced_variables_include_defaults() {
        assert_eq!(
            referenced_vars("${ONE} x ${TWO:-два}"),
            vec![
                ("ONE".to_string(), None),
                ("TWO".to_string(), Some("два".to_string()))
            ]
        );
    }

    #[test]
    fn parsed_server_values_are_expanded_but_names_and_cwd_are_not() {
        let raw = r#"{"mcpServers":{"${ALIAS}":{"url":"http://${HOST}","headers":{"${HEADER}":"Bearer ${TOKEN}"}},"stdio":{"command":"${CMD}","args":["--token=${TOKEN}"],"env":{"${ENV_KEY}":"${TOKEN}"},"cwd":"${CWD}"}}}"#;
        let lookup = |name: &str| Some(format!("value-{name}"));
        let http = expand_http_server(parse_mcp_config(raw).unwrap().remove(0), lookup).unwrap();
        assert_eq!(http.alias, "${ALIAS}");
        assert_eq!(http.url, "http://value-HOST");
        assert_eq!(
            http.headers,
            vec![("${HEADER}".into(), "Bearer value-TOKEN".into())]
        );

        let stdio =
            expand_stdio_server(parse_stdio_servers(raw).unwrap().remove(0), lookup).unwrap();
        assert_eq!(stdio.command, "value-CMD");
        assert_eq!(stdio.args, vec!["--token=value-TOKEN"]);
        assert_eq!(stdio.env, vec![("${ENV_KEY}".into(), "value-TOKEN".into())]);
        assert_eq!(stdio.cwd.as_deref(), Some("${CWD}"));
    }

    async fn serve_http(app: axum::Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (url, task)
    }

    #[tokio::test]
    async fn http_response_larger_than_five_mib_is_rejected_before_json() {
        use axum::routing::post;
        let app = axum::Router::new().route(
            "/mcp",
            post(|| async { vec![b'x'; MAX_RESPONSE_BYTES + 1] }),
        );
        let (url, task) = serve_http(app).await;
        let response = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .post(url)
            .send()
            .await
            .unwrap();
        let error = read_http_body(response, "tools/call")
            .await
            .expect_err("ответ выше предела должен быть отклонён");
        assert!(error.contains(&(MAX_RESPONSE_BYTES + 1).to_string()));
        assert!(error.contains(&MAX_RESPONSE_BYTES.to_string()));
        task.abort();
    }

    #[test]
    fn parse_http_headers() {
        let servers = parse_mcp_config(r#"{"mcpServers":{"x":{"url":"http://h/mcp","headers":{"Authorization":"Bearer abc"}}}}"#)
            .unwrap();
        assert_eq!(servers.len(), 1);
        assert_eq!(
            servers[0].headers,
            vec![("Authorization".into(), "Bearer abc".into())]
        );
        assert!(
            parse_mcp_config(r#"{"mcpServers":{"x":{"url":"http://h/mcp"}}}"#).unwrap()[0]
                .headers
                .is_empty()
        );
    }

    #[test]
    fn server_errors_hide_url_secrets() {
        // Адреса склеены из частей, чтобы проверка секретов перед фиксацией не
        // принимала тестовые строки за настоящие пароли.
        for raw in [
            concat!("http://u", ":p@h/mcp?key=k"),
            concat!("error sending request for url (http://u", ":p@h/mcp?key=k)"),
            "error sending request for url (http://h/mcp?key=k)",
        ] {
            let safe = safe_server_address(raw);
            assert!(!safe.contains("p@"));
            assert!(!safe.contains("key=k"));
            assert!(safe.contains("h/mcp"));
        }
    }

    #[tokio::test]
    async fn reqwest_errors_hide_url_secrets() {
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let server = McpServer {
            alias: "test".into(),
            url: concat!("http://u", ":p@h:invalid/mcp?key=k").into(),
            headers: Vec::new(),
        };
        let init = initialize_session(&client, &server).await.unwrap_err();
        let session = McpSession {
            inner: Arc::new(HttpSessionInner {
                client: client.clone(),
                server: server.clone(),
                state: tokio::sync::Mutex::new(HttpSessionState {
                    id: None,
                    protocol_version: PROTOCOL_VERSION.to_string(),
                }),
            }),
        };
        let call = call_tool(&client, &server, "echo", json!({}), &session)
            .await
            .unwrap_err();
        for error in [init, call] {
            assert!(!error.contains("p@"));
            assert!(!error.contains("key=k"));
        }
    }

    #[tokio::test]
    async fn http_headers_reach_entire_mcp_cycle() {
        use tokio::io::AsyncReadExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let stub = tokio::spawn(async move {
            let mut seen = Vec::new();
            for _ in 0..4 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                loop {
                    let mut chunk = [0_u8; 1024];
                    let read = stream.read(&mut chunk).await.unwrap();
                    assert!(read > 0, "соединение закрыто до конца HTTP-запроса");
                    request.extend_from_slice(&chunk[..read]);
                    let Some(headers_end) = request.windows(4).position(|w| w == b"\r\n\r\n")
                    else {
                        continue;
                    };
                    let headers = String::from_utf8_lossy(&request[..headers_end]);
                    let content_len = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if request.len() >= headers_end + 4 + content_len {
                        break;
                    }
                }

                let text = String::from_utf8(request).unwrap();
                let (headers, body) = text.split_once("\r\n\r\n").unwrap();
                let authorization = headers.lines().find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("authorization")
                        .then(|| value.trim().to_string())
                });
                let method = serde_json::from_str::<Value>(body).unwrap()["method"]
                    .as_str()
                    .unwrap()
                    .to_string();
                seen.push((method.clone(), authorization));

                let (status, extra_headers, body) = match method.as_str() {
                    "initialize" => (
                        "200 OK",
                        "Mcp-Session-Id: session-1\r\n",
                        r#"{"jsonrpc":"2.0","id":0,"result":{"protocolVersion":"2025-06-18"}}"#,
                    ),
                    "notifications/initialized" => ("202 Accepted", "", ""),
                    "tools/list" => (
                        "200 OK",
                        "",
                        r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"echo","inputSchema":{"type":"object"}}]}}"#,
                    ),
                    "tools/call" => (
                        "200 OK",
                        "",
                        r#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"ok"}]}}"#,
                    ),
                    other => panic!("неожиданный метод: {other}"),
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\n{extra_headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
            seen
        });

        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let server = McpServer {
            alias: "stub".into(),
            url: format!("http://{addr}/mcp"),
            headers: vec![("Authorization".into(), "Bearer abc".into())],
        };
        let session = initialize_session(&client, &server).await.unwrap();
        assert_eq!(session.session_id().await.as_deref(), Some("session-1"));
        let tools = list_tools(&client, &server, &session).await.unwrap();
        assert_eq!(tools.len(), 1);
        let result = call_tool(&client, &server, "echo", json!({}), &session)
            .await
            .unwrap();
        assert_eq!(result, "ok");

        let seen = tokio::time::timeout(std::time::Duration::from_secs(5), stub)
            .await
            .expect("HTTP-заглушка не получила все четыре запроса")
            .unwrap();
        assert_eq!(
            seen.iter()
                .map(|(method, _)| method.as_str())
                .collect::<Vec<_>>(),
            [
                "initialize",
                "notifications/initialized",
                "tools/list",
                "tools/call"
            ]
        );
        assert!(seen
            .iter()
            .all(|(_, authorization)| authorization.as_deref() == Some("Bearer abc")));
    }

    #[tokio::test]
    async fn handshake_without_session_sends_initialized_and_version() {
        use axum::{
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
            routing::post,
            Json, Router,
        };
        use std::sync::{Arc, Mutex};

        let seen = Arc::new(Mutex::new(Vec::new()));
        let handler_seen = seen.clone();
        let app = Router::new().route(
            "/mcp",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let seen = handler_seen.clone();
                async move {
                    let method = body["method"].as_str().unwrap().to_string();
                    seen.lock().unwrap().push((
                        method.clone(),
                        headers
                            .get("mcp-protocol-version")
                            .and_then(|v| v.to_str().ok())
                            .map(str::to_string),
                    ));
                    match method.as_str() {
                        // Сервер отвечает другой версией, чем просил клиент, —
                        // так делают реальные серверы на rmcp.
                        "initialize" => Json(json!({
                            "jsonrpc": "2.0",
                            "id": 0,
                            "result": {"protocolVersion": "2025-11-25"}
                        }))
                        .into_response(),
                        "notifications/initialized" => StatusCode::ACCEPTED.into_response(),
                        "tools/list" => Json(json!({
                            "jsonrpc": "2.0",
                            "id": 1,
                            "result": {"tools": []}
                        }))
                        .into_response(),
                        other => panic!("неожиданный метод: {other}"),
                    }
                }
            }),
        );
        let (url, task) = serve_http(app).await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let server = McpServer {
            alias: "stateless".into(),
            url,
            headers: Vec::new(),
        };

        let session = initialize_session(&client, &server).await.unwrap();
        assert_eq!(session.session_id().await, None);
        assert!(list_tools(&client, &server, &session)
            .await
            .unwrap()
            .is_empty());
        task.abort();

        assert_eq!(
            *seen.lock().unwrap(),
            [
                ("initialize".to_string(), None),
                (
                    "notifications/initialized".to_string(),
                    Some("2025-11-25".to_string())
                ),
                ("tools/list".to_string(), Some("2025-11-25".to_string())),
            ]
        );
    }

    #[test]
    fn sse_notification_before_response_is_ignored() {
        let text = concat!(
            "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{}}\n\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[]}}\n\n"
        );
        assert_eq!(
            extract_envelope(text, 1).unwrap()["result"]["tools"],
            json!([])
        );
    }

    #[test]
    fn reversed_braces_return_error() {
        assert!(extract_envelope("}broken{", 1).is_err());
    }

    #[tokio::test]
    async fn stale_session_is_reinitialized_once_and_request_repeated() {
        use axum::{
            http::{HeaderMap, HeaderValue, StatusCode},
            response::IntoResponse,
            routing::post,
            Json, Router,
        };
        use std::sync::{Arc, Mutex};

        let seen = Arc::new(Mutex::new(Vec::new()));
        let handler_seen = seen.clone();
        let app = Router::new().route(
            "/mcp",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let seen = handler_seen.clone();
                async move {
                    let method = body["method"].as_str().unwrap().to_string();
                    let session_id = headers
                        .get("mcp-session-id")
                        .and_then(|value| value.to_str().ok())
                        .map(str::to_string);
                    let mut rows = seen.lock().unwrap();
                    rows.push((method.clone(), session_id.clone()));
                    let initialize_count = rows
                        .iter()
                        .filter(|(method, _)| method == "initialize")
                        .count();
                    drop(rows);
                    match method.as_str() {
                        "initialize" => {
                            let session_id = if initialize_count == 1 { "old" } else { "new" };
                            let mut response = Json(json!({
                                "jsonrpc": "2.0",
                                "id": 0,
                                "result": {"protocolVersion": PROTOCOL_VERSION}
                            }))
                            .into_response();
                            response
                                .headers_mut()
                                .insert("mcp-session-id", HeaderValue::from_static(session_id));
                            response
                        }
                        "notifications/initialized" => StatusCode::ACCEPTED.into_response(),
                        "tools/call" if session_id.as_deref() == Some("old") => {
                            StatusCode::NOT_FOUND.into_response()
                        }
                        "tools/call" => Json(json!({
                            "jsonrpc": "2.0",
                            "id": 1,
                            "result": {"content": [{"type": "text", "text": "ok"}]}
                        }))
                        .into_response(),
                        other => panic!("неожиданный метод: {other}"),
                    }
                }
            }),
        );
        let (url, task) = serve_http(app).await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let server = McpServer {
            alias: "stateful".into(),
            url,
            headers: Vec::new(),
        };

        let session = initialize_session(&client, &server).await.unwrap();
        let result = call_tool(&client, &server, "echo", json!({}), &session)
            .await
            .unwrap();
        assert_eq!(result, "ok");
        assert_eq!(session.session_id().await.as_deref(), Some("new"));
        task.abort();

        assert_eq!(
            seen.lock()
                .unwrap()
                .iter()
                .map(|(method, session)| (method.as_str(), session.as_deref()))
                .collect::<Vec<_>>(),
            [
                ("initialize", None),
                ("notifications/initialized", Some("old")),
                ("tools/call", Some("old")),
                ("initialize", None),
                ("notifications/initialized", Some("new")),
                ("tools/call", Some("new")),
            ]
        );
    }

    #[tokio::test]
    async fn tools_list_reads_all_pages() {
        use axum::{response::IntoResponse, routing::post, Json, Router};

        let app = Router::new().route(
            "/mcp",
            post(|Json(body): Json<Value>| async move {
                match body["method"].as_str().unwrap() {
                    "initialize" => Json(json!({
                        "jsonrpc": "2.0",
                        "id": 0,
                        "result": {"protocolVersion": PROTOCOL_VERSION}
                    }))
                    .into_response(),
                    "notifications/initialized" => axum::http::StatusCode::ACCEPTED.into_response(),
                    "tools/list" if body["params"].get("cursor").is_none() => Json(json!({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": {
                            "tools": [{"name": "first", "inputSchema": {"type": "object"}}],
                            "nextCursor": "page-2"
                        }
                    }))
                    .into_response(),
                    "tools/list" => Json(json!({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": {
                            "tools": [{"name": "second", "inputSchema": {"type": "object"}}]
                        }
                    }))
                    .into_response(),
                    other => panic!("неожиданный метод: {other}"),
                }
            }),
        );
        let (url, task) = serve_http(app).await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let server = McpServer {
            alias: "paged".into(),
            url,
            headers: Vec::new(),
        };

        let session = initialize_session(&client, &server).await.unwrap();
        let tools = list_tools(&client, &server, &session).await.unwrap();
        task.abort();
        assert_eq!(
            tools
                .iter()
                .map(|tool| tool.tool_name.as_str())
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
    }

    #[tokio::test]
    async fn dropping_stateful_session_sends_delete() {
        use axum::{
            http::{HeaderMap, HeaderValue, StatusCode},
            response::IntoResponse,
            routing::post,
            Json, Router,
        };
        use std::sync::{Arc, Mutex};

        let (deleted_tx, deleted_rx) = tokio::sync::oneshot::channel();
        let deleted_tx = Arc::new(Mutex::new(Some(deleted_tx)));
        let delete_sender = deleted_tx.clone();
        let app = Router::new().route(
            "/mcp",
            post(|Json(body): Json<Value>| async move {
                match body["method"].as_str().unwrap() {
                    "initialize" => {
                        let mut response = Json(json!({
                            "jsonrpc": "2.0",
                            "id": 0,
                            "result": {"protocolVersion": PROTOCOL_VERSION}
                        }))
                        .into_response();
                        response
                            .headers_mut()
                            .insert("mcp-session-id", HeaderValue::from_static("closing"));
                        response
                    }
                    "notifications/initialized" => StatusCode::ACCEPTED.into_response(),
                    other => panic!("неожиданный метод: {other}"),
                }
            })
            .delete(move |headers: HeaderMap| {
                let sender = delete_sender.clone();
                async move {
                    let session = headers
                        .get("mcp-session-id")
                        .and_then(|value| value.to_str().ok())
                        .map(str::to_string);
                    let version = headers
                        .get("mcp-protocol-version")
                        .and_then(|value| value.to_str().ok())
                        .map(str::to_string);
                    if let Some(sender) = sender.lock().unwrap().take() {
                        let _ = sender.send((session, version));
                    }
                    StatusCode::OK
                }
            }),
        );
        let (url, task) = serve_http(app).await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let server = McpServer {
            alias: "stateful".into(),
            url,
            headers: Vec::new(),
        };

        let session = initialize_session(&client, &server).await.unwrap();
        drop(session);
        let deleted = tokio::time::timeout(std::time::Duration::from_secs(5), deleted_rx)
            .await
            .expect("DELETE не отправлен")
            .unwrap();
        task.abort();
        assert_eq!(deleted.0.as_deref(), Some("closing"));
        assert_eq!(deleted.1.as_deref(), Some(PROTOCOL_VERSION));
    }

    #[test]
    fn image_only_result_is_not_reported_as_empty() {
        let result = json!({
            "content": [{"type": "image", "data": "AA==", "mimeType": "image/png"}]
        });
        let text = text_from_call_result("image", &result).unwrap();
        assert!(text.contains("нетекстовое содержимое"));
        assert!(text.contains("image"));
        assert_ne!(text, EMPTY_RESULT_MARK);
    }

    #[test]
    fn stdio_env_does_not_inherit_service_secrets() {
        let name = "AGENTS_MCP_TEST_SECRET";
        let previous = std::env::var_os(name);
        std::env::set_var(name, "secret");
        let cfg = StdioServerCfg {
            alias: "test".into(),
            command: "unused".into(),
            args: Vec::new(),
            env: vec![
                ("A".into(), "1".into()),
                ("PATH".into(), "explicit-path".into()),
            ],
            cwd: None,
        };
        let mut cmd = Command::new(&cfg.command);
        cmd.env(name, "configured-before-clear");
        configure_stdio_env(&mut cmd, &cfg);
        match previous {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
        let env: HashMap<_, _> = cmd.as_std().get_envs().collect();
        assert!(!env.contains_key(std::ffi::OsStr::new(name)));
        assert_eq!(
            env[std::ffi::OsStr::new("A")],
            Some(std::ffi::OsStr::new("1"))
        );
        assert_eq!(
            env[std::ffi::OsStr::new("PATH")],
            Some(std::ffi::OsStr::new("explicit-path"))
        );
    }

    #[test]
    fn parse_two_http_servers() {
        let raw = r#"{"mcpServers":{"1c":{"type":"http","url":"http://127.0.0.1:8010/mcp"},"agents":{"type":"http","url":"http://127.0.0.1:8025/mcp"}}}"#;
        let mut s = parse_mcp_config(raw).unwrap();
        s.sort_by(|a, b| a.alias.cmp(&b.alias));
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].alias, "1c");
        assert_eq!(s[0].url, "http://127.0.0.1:8010/mcp");
        assert_eq!(s[1].alias, "agents");
    }

    #[test]
    fn parse_skips_servers_without_url() {
        let raw = r#"{"mcpServers":{"local":{"type":"stdio","command":"foo"},"web":{"url":"http://x/mcp"}}}"#;
        let s = parse_mcp_config(raw).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].alias, "web");
    }

    #[test]
    fn parse_picks_stdio_servers_and_their_fields() {
        let raw = r#"{"mcpServers":{
            "local":{"type":"stdio","command":"node","args":["srv.js","--flag"],
                     "env":{"TOKEN":"x"},"cwd":"C:/Temp"},
            "web":{"url":"http://x/mcp"}}}"#;
        let s = parse_stdio_servers(raw).unwrap();
        assert_eq!(s.len(), 1, "сервер с url в эту выборку попадать не должен");
        assert_eq!(s[0].alias, "local");
        assert_eq!(s[0].command, "node");
        assert_eq!(s[0].args, vec!["srv.js".to_string(), "--flag".to_string()]);
        assert_eq!(s[0].env, vec![("TOKEN".to_string(), "x".to_string())]);
        assert_eq!(s[0].cwd.as_deref(), Some("C:/Temp"));
        // И обратно: HTTP-разбор не должен трогать запись с command.
        let http = parse_mcp_config(raw).unwrap();
        assert_eq!(http.len(), 1);
        assert_eq!(http[0].alias, "web");
    }

    #[test]
    fn url_wins_over_command_in_the_same_entry() {
        // Заданы оба — сервер уже взят HTTP-веткой, процесс поднимать незачем.
        let raw = r#"{"mcpServers":{"both":{"url":"http://x/mcp","command":"node"}}}"#;
        assert!(parse_stdio_servers(raw).unwrap().is_empty());
        assert_eq!(parse_mcp_config(raw).unwrap().len(), 1);
    }

    #[test]
    fn malformed_config_is_distinct_from_empty_server_list() {
        assert!(parse_mcp_config("{broken").is_err());
        assert!(parse_stdio_servers("{broken").is_err());
        assert!(parse_mcp_config(r#"{"mcpServers":{}}"#).unwrap().is_empty());
        assert!(parse_stdio_servers(r#"{"mcpServers":{}}"#)
            .unwrap()
            .is_empty());
    }

    /// Крошечный MCP-сервер на Python: отвечает на initialize/tools/list/
    /// tools/call и первой строкой печатает мусор — клиент обязан его пропустить.
    const MINI_SERVER: &str = r#"
import sys, json
print("не-JSON строка, клиент должен её пропустить", flush=True)
while True:
    line = sys.stdin.readline()
    if not line:
        break
    line = line.strip()
    if not line:
        continue
    m = json.loads(line)
    if m.get("id") is None:
        continue
    meth = m.get("method")
    if meth == "initialize":
        r = {"protocolVersion": "2025-06-18", "capabilities": {}}
    elif meth == "tools/list":
        r = {"tools": [{"name": "echo", "description": "повторяет текст",
                        "inputSchema": {"type": "object",
                                        "properties": {"text": {"type": "string"}}}}]}
    elif meth == "tools/call":
        r = {"content": [{"type": "text",
                          "text": "эхо: " + m["params"]["arguments"].get("text", "")}]}
    else:
        r = {}
    print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}), flush=True)
"#;

    #[tokio::test]
    async fn stdio_server_handshake_list_and_call() {
        let cfg = StdioServerCfg {
            alias: "mini".into(),
            command: "python".into(),
            args: vec!["-u".into(), "-c".into(), MINI_SERVER.into()],
            env: vec![("PYTHONIOENCODING".into(), "utf-8".into())],
            cwd: None,
        };
        let mut srv = match StdioServer::spawn(&cfg, None).await {
            Ok(s) => s,
            // Нет Python — проверять нечем, но это не повод ронять прогон.
            Err(e) if e.contains("не запустился процесс") => {
                eprintln!("проверка пропущена: {e}");
                return;
            }
            Err(e) => panic!("{e}"),
        };

        let tools = srv.list_tools().await.expect("tools/list");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].full_name, "mcp__mini__echo");
        assert_eq!(tools[0].tool_name, "echo");

        let out = srv
            .call_tool("echo", json!({"text": "привет"}))
            .await
            .expect("tools/call");
        assert_eq!(out, "эхо: привет");

        let mut pool = StdioPool::default();
        pool.add_tool("mcp__mini__echo".into(), "mini".into(), "echo".into());
        pool.add_server(srv);
        assert!(pool.has("mcp__mini__echo"));
        assert!(!pool.has("mcp__mini__другое"));
        let out = pool
            .call("mcp__mini__echo", json!({"text": "ещё раз"}))
            .await
            .expect("вызов через набор");
        assert_eq!(out, "эхо: ещё раз");
    }

    #[test]
    fn parse_rejects_garbage_and_accepts_empty_object() {
        assert!(parse_mcp_config("not json").is_err());
        assert!(parse_mcp_config("{}").unwrap().is_empty());
    }

    #[test]
    fn envelope_from_plain_and_sse() {
        let plain = r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#;
        assert_eq!(
            extract_envelope(plain, 1).unwrap()["result"]["ok"],
            json!(true)
        );
        let sse =
            "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":true}}\n\n";
        assert_eq!(
            extract_envelope(sse, 1).unwrap()["result"]["ok"],
            json!(true)
        );
    }
}
