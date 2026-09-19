//! Anthropic Messages API провайдер.
//! С поддержкой MCP-инструментов.
//!
//! Anthropic не отдаёт `cost_usd` в ответе — считаем по цене модели из секции
//! провайдера главного конфига.
//! Точная схема и заголовки:
//!   POST https://api.anthropic.com/v1/messages
//!   x-api-key: <ANTHROPIC_API_KEY>
//!   anthropic-version: 2023-06-01
//!   content-type: application/json
//!
//! Особенность: `system` уезжает в отдельном поле верхнего уровня, а в
//! `messages` обязательно должен быть хотя бы один user. Если у нас пустой
//! `user_input` — кидаем fallback-просьбу «выполни задачу из system».

use async_trait::async_trait;
use reqwest::{header, Client};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::config::ModelPrice;

use super::tool_loop::{
    build_mcp_tools, clamp_tool_result, execute_tool_call, truncate_str, McpTools, ToolCallInput,
    ToolCallState, Transcript, DEFAULT_MAX_TOOL_TURNS,
};
use super::{pricing, LlmError, LlmProvider, LlmRequest, LlmResponse};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_MAX_TOKENS: u32 = 4096;

pub struct AnthropicProvider {
    name: String,
    client: Client,
    api_key: String,
    base_url: String,
    prompt_cache: bool,
    prices: BTreeMap<String, ModelPrice>,
    semaphore: Option<Arc<tokio::sync::Semaphore>>,
    active: Arc<AtomicUsize>,
}

pub(crate) struct AnthropicOptions {
    pub name: String,
    pub api_key: String,
    pub base_url: Option<String>,
    pub proxy: Option<String>,
    pub proxy_bypass: Option<String>,
    pub max_concurrent: Option<u32>,
    pub prompt_cache: bool,
    pub prices: BTreeMap<String, ModelPrice>,
}

struct ActiveCall(Arc<AtomicUsize>);

impl Drop for ActiveCall {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Deserialize, Debug, Default)]
struct MessagesResponse {
    #[serde(default)]
    content: Vec<Value>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<Usage>,
    /// Ошибка от Anthropic: `{"error": {"type": "...", "message": "..."}}`.
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Deserialize, Debug, Default, Clone, Copy)]
struct Usage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
}

impl Usage {
    fn tokens_in(self) -> u32 {
        self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens
    }
}

#[derive(Debug)]
struct ToolUse {
    id: String,
    name: String,
    input: Value,
    arguments: String,
}

#[derive(Debug)]
struct MessagesTurn {
    blocks: Vec<Value>,
    content: String,
    reasoning: Option<String>,
    tool_uses: Vec<ToolUse>,
    finish_reason: String,
    usage: Usage,
    cost: Option<f64>,
}

impl AnthropicProvider {
    pub fn new(options: AnthropicOptions) -> Self {
        let AnthropicOptions {
            name,
            api_key,
            base_url,
            proxy,
            proxy_bypass,
            max_concurrent,
            prompt_cache,
            prices,
        } = options;
        let client = super::build_http_client(&name, proxy.as_deref(), proxy_bypass.as_deref());
        Self {
            name,
            client,
            api_key,
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.into()),
            prompt_cache,
            prices,
            semaphore: max_concurrent
                .map(|n| Arc::new(tokio::sync::Semaphore::new(n.max(1) as usize))),
            active: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn request_body(
        &self,
        req: &LlmRequest,
        messages: &[Value],
        tools: &[super::mcp_client::ToolDef],
        temperature: f32,
    ) -> Value {
        let mut body = Map::new();
        body.insert("model".into(), Value::String(req.model.clone()));
        body.insert(
            "max_tokens".into(),
            Value::Number(if req.max_tokens == 0 {
                DEFAULT_MAX_TOKENS.into()
            } else {
                req.max_tokens.into()
            }),
        );
        if let Some(value) = request_temperature(temperature) {
            body.insert("temperature".into(), json!(value));
        }
        if let Some(value) = req.top_p {
            body.insert("top_p".into(), json!(value));
        }
        if !req.system_prompt.is_empty() {
            let system = if self.prompt_cache {
                json!([{
                    "type": "text",
                    "text": req.system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }])
            } else {
                Value::String(req.system_prompt.clone())
            };
            body.insert("system".into(), system);
        }
        body.insert("messages".into(), Value::Array(messages.to_vec()));
        if !tools.is_empty() {
            let last = tools.len() - 1;
            let definitions = tools
                .iter()
                .enumerate()
                .map(|(index, def)| {
                    let schema = if def.parameters.is_object() {
                        def.parameters.clone()
                    } else {
                        json!({"type": "object", "properties": {}})
                    };
                    let mut tool = json!({
                        "name": def.full_name,
                        "description": def.description,
                        "input_schema": schema,
                    });
                    if self.prompt_cache && index == last {
                        tool["cache_control"] = json!({"type": "ephemeral"});
                    }
                    tool
                })
                .collect();
            body.insert("tools".into(), Value::Array(definitions));
        }
        for (name, value) in &req.extra_body {
            body.insert(name.clone(), value.clone());
        }
        Value::Object(body)
    }

    async fn messages_once(
        &self,
        req: &LlmRequest,
        messages: &[Value],
        tools: &[super::mcp_client::ToolDef],
        temperature: f32,
        deadline: tokio::time::Instant,
    ) -> Result<MessagesTurn, LlmError> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let body = self.request_body(req, messages, tools, temperature);
        let mut retry = 0u64;
        let (status, raw) = loop {
            let left = deadline.saturating_duration_since(tokio::time::Instant::now());
            if left.is_zero() {
                return Err(LlmError::Timeout);
            }
            let resp = self
                .client
                .post(&url)
                .timeout(left)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION)
                .header(header::CONTENT_TYPE, "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|error| map_reqwest_err(&self.name, error))?;
            let status = resp.status();
            let retry_after = retry_after_seconds(resp.headers());
            let raw = resp.text().await.map_err(|error| {
                LlmError::Provider(format!("{}: ошибка чтения тела: {error}", self.name))
            })?;
            if status.as_u16() != 429 {
                break (status, raw);
            }
            let delay = retry_after.unwrap_or(2u64 << retry.min(2));
            tracing::warn!(
                provider = %self.name,
                model = %req.model,
                active = self.active.load(Ordering::SeqCst),
                retry = retry + 1,
                delay_sec = delay,
                "провайдер ответил 429"
            );
            if retry >= 3 {
                return Err(LlmError::RateLimited);
            }
            let sleep = Duration::from_secs(delay);
            if sleep >= deadline.saturating_duration_since(tokio::time::Instant::now()) {
                return Err(LlmError::Timeout);
            }
            tokio::time::sleep(sleep).await;
            retry += 1;
        };

        if !status.is_success() {
            // Authentication / token-expired специально маппим.
            if status.as_u16() == 401 {
                return Err(LlmError::Provider(format!(
                    "{}: 401 Unauthorized — проверь ключ из api_key_env провайдера '{}'. body={}",
                    self.name,
                    self.name,
                    truncate(&raw, 200)
                )));
            }
            return Err(LlmError::Provider(format!(
                "{} HTTP {status}: {}",
                self.name,
                truncate(&raw, 500)
            )));
        }

        let parsed: MessagesResponse = serde_json::from_str(&raw).map_err(|error| {
            LlmError::InvalidResponse(format!(
                "{}: не удалось распарсить JSON: {error}; body={}",
                self.name,
                truncate(&raw, 200)
            ))
        })?;
        if let Some(error) = parsed.error {
            return Err(LlmError::Provider(format!("{} API: {error}", self.name)));
        }

        let mut text = Vec::new();
        let mut thinking = Vec::new();
        let mut tool_uses = Vec::new();
        for block in &parsed.content {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(value) = block.get("text").and_then(Value::as_str) {
                        text.push(value.to_string());
                    }
                }
                Some("thinking") => {
                    if let Some(value) = block.get("thinking").and_then(Value::as_str) {
                        thinking.push(value.to_string());
                    }
                }
                Some("tool_use") => {
                    let Some(id) = block.get("id").and_then(Value::as_str) else {
                        continue;
                    };
                    let Some(name) = block.get("name").and_then(Value::as_str) else {
                        continue;
                    };
                    let input = block.get("input").cloned().unwrap_or(Value::Null);
                    tool_uses.push(ToolUse {
                        id: id.to_string(),
                        name: name.to_string(),
                        arguments: input.to_string(),
                        input,
                    });
                }
                _ => {}
            }
        }
        if tools.is_empty() && text.is_empty() {
            return Err(LlmError::InvalidResponse(format!(
                "{}: пустой text content в ответе; body={}",
                self.name,
                truncate(&raw, 200)
            )));
        }
        let usage = parsed.usage.unwrap_or_default();
        let cost = pricing::anthropic_cost(
            self.prices.get(&req.model),
            usage.input_tokens,
            usage.cache_creation_input_tokens,
            usage.cache_read_input_tokens,
            usage.output_tokens,
        );
        Ok(MessagesTurn {
            blocks: parsed.content,
            // Склеиваем все text-блоки контента.
            content: text.join(""),
            reasoning: (!thinking.is_empty()).then(|| thinking.join("")),
            tool_uses,
            finish_reason: parsed.stop_reason.unwrap_or_else(|| "unknown".into()),
            usage,
            cost,
        })
    }

    fn response(
        &self,
        content: String,
        finish_reason: String,
        reasoning: Option<String>,
        usage: Usage,
        cost_usd: Option<f64>,
        transcript: Vec<Value>,
    ) -> LlmResponse {
        LlmResponse {
            content,
            tokens_in: usage.tokens_in(),
            tokens_out: usage.output_tokens,
            cost_usd,
            finish_reason,
            reasoning,
            session_id: None,
            raw_input_tokens: usage.input_tokens,
            cache_creation_input_tokens: usage.cache_creation_input_tokens,
            cache_read_input_tokens: usage.cache_read_input_tokens,
            transcript,
        }
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError> {
        let deadline = tokio::time::Instant::now() + req.timeout;
        let _permit = match &self.semaphore {
            Some(semaphore) => Some(
                tokio::time::timeout_at(deadline, semaphore.clone().acquire_owned())
                    .await
                    .map_err(|_| LlmError::Timeout)?
                    .map_err(|error| {
                        LlmError::Provider(format!("семафор провайдера закрыт: {error}"))
                    })?,
            ),
            None => None,
        };
        self.active.fetch_add(1, Ordering::SeqCst);
        let _active = ActiveCall(self.active.clone());

        let user_content = if req.user_input.is_empty() {
            "Выполни задачу из system-промпта и верни результат."
        } else {
            req.user_input.as_str()
        };
        let mut messages = vec![json!({"role": "user", "content": user_content})];
        let mut tools = match &req.cli_hints {
            Some(hints) if hints.mcp_config.is_some() => {
                build_mcp_tools(&self.client, hints).await?
            }
            _ => McpTools::default(),
        };
        let transcript = Transcript::open(req.turn_sink.clone());
        if transcript.enabled() {
            transcript.write(&json!({
                "event": "start",
                "provider": self.name,
                "model": req.model,
                "tools": tools.defs.len(),
                "system_prompt": req.system_prompt,
                "user_input": user_content,
            }));
        }

        if tools.defs.is_empty() {
            let turn = self
                .messages_once(&req, &messages, &[], req.temperature, deadline)
                .await?;
            transcript.write(&json!({
                "event": "single_shot",
                "content": turn.content,
                "reasoning": turn.reasoning,
                "finish": turn.finish_reason,
                "tokens_in": turn.usage.tokens_in(),
                "tokens_out": turn.usage.output_tokens,
            }));
            return Ok(self.response(
                turn.content,
                turn.finish_reason,
                turn.reasoning,
                turn.usage,
                turn.cost,
                transcript.take(),
            ));
        }

        let max_turns = req
            .cli_hints
            .as_ref()
            .and_then(|hints| hints.max_turns)
            .unwrap_or(DEFAULT_MAX_TOOL_TURNS)
            .max(1);
        let mut totals = Usage::default();
        let mut total_cost = Some(0.0);
        let mut last_finish = String::from("unknown");
        let mut temperature = req.temperature;
        let mut state = ToolCallState::default();

        for turn_idx in 0..max_turns {
            let started = std::time::Instant::now();
            let turn = match self
                .messages_once(&req, &messages, &tools.defs, temperature, deadline)
                .await
            {
                Ok(turn) => turn,
                Err(error) if turn_idx > 0 => {
                    transcript.write(&json!({
                        "event": "provider_error",
                        "turn": turn_idx,
                        "error": truncate(&error.to_string(), 1000),
                        "request_shape": {
                            "messages": messages.len(),
                            "bytes": serde_json::to_vec(&messages).map(|v| v.len()).unwrap_or(0),
                        },
                    }));
                    return Err(LlmError::WithUsage {
                        error: Box::new(error),
                        tokens_in: totals.tokens_in(),
                        tokens_out: totals.output_tokens,
                        cost: total_cost,
                    });
                }
                Err(error) => return Err(error),
            };
            temperature = req.temperature;
            totals.input_tokens += turn.usage.input_tokens;
            totals.output_tokens += turn.usage.output_tokens;
            totals.cache_creation_input_tokens += turn.usage.cache_creation_input_tokens;
            totals.cache_read_input_tokens += turn.usage.cache_read_input_tokens;
            total_cost = pricing::add_cost(total_cost, turn.cost);
            last_finish = turn.finish_reason.clone();
            transcript.write(&json!({
                "event": "turn",
                "turn": turn_idx,
                "content": turn.content,
                "reasoning": turn.reasoning,
                "finish": last_finish,
                "tool_calls": turn.tool_uses.iter().map(|call| json!({
                    "name": call.name,
                    "arguments": call.arguments,
                })).collect::<Vec<_>>(),
                "tokens_in": turn.usage.tokens_in(),
                "tokens_out": turn.usage.output_tokens,
                "duration_ms": started.elapsed().as_millis() as u64,
            }));

            if turn.tool_uses.is_empty() || turn.finish_reason == "max_tokens" {
                transcript.write(&json!({"event": "final", "turn": turn_idx}));
                return Ok(self.response(
                    turn.content,
                    turn.finish_reason,
                    turn.reasoning,
                    totals,
                    total_cost,
                    transcript.take(),
                ));
            }

            messages.push(json!({"role": "assistant", "content": turn.blocks}));
            let mut results = Vec::with_capacity(turn.tool_uses.len());
            for call in turn.tool_uses {
                let parsed = if call.input.is_object() {
                    Ok(call.input)
                } else {
                    Err("input не объект".to_string())
                };
                let tool_started = std::time::Instant::now();
                let result = execute_tool_call(
                    &self.client,
                    &mut tools,
                    &mut state,
                    &mut temperature,
                    ToolCallInput {
                        provider: &self.name,
                        hints: req.cli_hints.as_ref(),
                        tool_name: &call.name,
                        raw_arguments: &call.arguments,
                        parsed_arguments: parsed,
                    },
                )
                .await;
                transcript.write(&json!({
                    "event": "tool_result",
                    "turn": turn_idx,
                    "tool": call.name,
                    "tool_call_id": call.id,
                    "duration_ms": tool_started.elapsed().as_millis() as u64,
                    "result_chars": result.chars().count(),
                    "result": truncate_str(&result, 50_000),
                }));
                results.push(json!({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": clamp_tool_result(&result),
                }));
            }
            messages.push(json!({"role": "user", "content": results}));
        }

        transcript.write(&json!({
            "event": "max_turns",
            "max_turns": max_turns,
            "finish": last_finish,
            "tokens_in": totals.tokens_in(),
            "tokens_out": totals.output_tokens,
        }));
        Err(LlmError::MaxTurns {
            provider: self.name.clone(),
            turns: max_turns,
            finish_reason: last_finish,
            tokens_in: totals.tokens_in(),
            tokens_out: totals.output_tokens,
            cost: total_cost,
            transcript: transcript.take(),
        })
    }
}

// Runtime пока хранит уже подставленный default, поэтому отличить
// явно заданные 0.7 нельзя: значение по умолчанию не отправляем.
fn request_temperature(temperature: f32) -> Option<f32> {
    (temperature != DEFAULT_TEMPERATURE).then_some(temperature)
}

fn map_reqwest_err(provider: &str, error: reqwest::Error) -> LlmError {
    if error.is_timeout() {
        LlmError::Timeout
    } else {
        LlmError::Provider(format!("{provider}: {error}"))
    }
}

fn retry_after_seconds(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let value = headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)?;
    value.parse().ok().or_else(|| {
        let retry_at = chrono::DateTime::parse_from_rfc2822(value).ok()?;
        Some(
            (retry_at.with_timezone(&chrono::Utc) - chrono::Utc::now())
                .num_seconds()
                .max(0) as u64,
        )
    })
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        value.to_string()
    } else {
        let mut out: String = value.chars().take(max_chars).collect();
        out.push_str("…[truncated]");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderValue, StatusCode};
    use axum::response::IntoResponse;
    use axum::routing::post;
    use axum::{Json, Router};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    struct TestApi {
        base: String,
        seen: Arc<Mutex<Vec<Value>>>,
        calls: Arc<AtomicUsize>,
        task: tokio::task::JoinHandle<()>,
    }

    impl Drop for TestApi {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    async fn test_api(responses: Vec<(u16, Value)>) -> TestApi {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let calls = Arc::new(AtomicUsize::new(0));
        let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
        let seen_requests = seen.clone();
        let calls_server = calls.clone();
        let app = Router::new()
            .route(
                "/v1/messages",
                post(move |Json(body): Json<Value>| {
                    let seen = seen_requests.clone();
                    let queue = queue.clone();
                    async move {
                        seen.lock().unwrap().push(body);
                        let (status, body) = queue
                            .lock()
                            .unwrap()
                            .pop_front()
                            .expect("неожиданный запрос к модели");
                        let mut response = Json(body).into_response();
                        *response.status_mut() = StatusCode::from_u16(status).unwrap();
                        if status == 429 {
                            response
                                .headers_mut()
                                .insert("retry-after", HeaderValue::from_static("0"));
                        }
                        response
                    }
                }),
            )
            .route(
                "/mcp",
                post(move |Json(body): Json<Value>| {
                    let calls = calls_server.clone();
                    async move {
                        let result = match body["method"].as_str().unwrap_or("") {
                            "initialize" => json!({
                                "protocolVersion": "2025-06-18",
                                "capabilities": {},
                                "serverInfo": {"name": "fixture", "version": "1"}
                            }),
                            "tools/list" => json!({"tools": [{
                                "name": "poll",
                                "description": "Read status",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"value": {"type": "integer"}}
                                }
                            }]}),
                            "tools/call" => {
                                let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
                                json!({"content": [{
                                    "type": "text",
                                    "text": format!("result-{call}")
                                }]})
                            }
                            _ => json!({}),
                        };
                        Json(json!({
                            "jsonrpc": "2.0",
                            "id": body["id"],
                            "result": result
                        }))
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        TestApi {
            base,
            seen,
            calls,
            task,
        }
    }

    fn response(content: Vec<Value>, stop_reason: &str, usage: Value) -> Value {
        json!({
            "content": content,
            "stop_reason": stop_reason,
            "usage": usage,
        })
    }

    fn text_response(text: &str) -> Value {
        response(
            vec![json!({"type": "text", "text": text})],
            "end_turn",
            json!({"input_tokens": 2, "output_tokens": 3}),
        )
    }

    fn test_provider(base: String, prompt_cache: bool) -> AnthropicProvider {
        AnthropicProvider::new(AnthropicOptions {
            name: "fixture-anthropic".into(),
            api_key: "synthetic-key".into(),
            base_url: Some(base),
            proxy: None,
            proxy_bypass: None,
            max_concurrent: None,
            prompt_cache,
            prices: BTreeMap::new(),
        })
    }

    fn test_provider_with_price(base: String) -> AnthropicProvider {
        let mut prices = BTreeMap::new();
        prices.insert(
            "claude-test".into(),
            ModelPrice {
                input: 2.0,
                output: 8.0,
                cache_read: Some(0.5),
                cache_write: Some(3.0),
            },
        );
        AnthropicProvider::new(AnthropicOptions {
            name: "fixture-anthropic".into(),
            api_key: "synthetic-key".into(),
            base_url: Some(base),
            proxy: None,
            proxy_bypass: None,
            max_concurrent: None,
            prompt_cache: false,
            prices,
        })
    }

    fn test_request() -> LlmRequest {
        LlmRequest {
            model: "claude-test".into(),
            system_prompt: "Synthetic system".into(),
            user_input: "Synthetic user".into(),
            temperature: DEFAULT_TEMPERATURE,
            max_tokens: 100,
            top_p: None,
            extra_body: Default::default(),
            timeout: Duration::from_secs(3),
            cli_hints: None,
            turn_sink: None,
            fallback_skill_names: Vec::new(),
            prompt_skill_names: Vec::new(),
            skills: None,
        }
    }

    fn agentic_request(base: &str, max_turns: u32) -> LlmRequest {
        let mut req = test_request();
        req.cli_hints = Some(super::super::ClaudeCliHints {
            mcp_config: Some(
                json!({"mcpServers": {"fixture": {"url": format!("{base}/mcp")}}}).to_string(),
            ),
            max_turns: Some(max_turns),
            ..Default::default()
        });
        req
    }

    fn tool_turn(id: &str, name: &str, input: Value, stop_reason: &str) -> Value {
        response(
            vec![json!({"type": "tool_use", "id": id, "name": name, "input": input})],
            stop_reason,
            json!({"input_tokens": 2, "output_tokens": 1}),
        )
    }

    #[tokio::test]
    async fn single_shot_body_and_cache_usage_follow_contract() {
        let api = test_api(vec![(
            200,
            response(
                vec![json!({"type": "text", "text": "done"})],
                "end_turn",
                json!({
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cache_creation_input_tokens": 3,
                    "cache_read_input_tokens": 5
                }),
            ),
        )])
        .await;
        let mut req = test_request();
        req.max_tokens = 0;
        let result = test_provider(api.base.clone(), false)
            .complete(req)
            .await
            .unwrap();
        assert_eq!(result.content, "done");
        assert_eq!(result.tokens_in, 18);
        assert_eq!(result.raw_input_tokens, 10);
        assert_eq!(result.cache_creation_input_tokens, 3);
        assert_eq!(result.cache_read_input_tokens, 5);
        assert_eq!(result.tokens_out, 4);
        assert_eq!(result.cost_usd, None);
        let seen = api.seen.lock().unwrap();
        assert_eq!(seen[0]["max_tokens"], 4096);
        assert_eq!(seen[0]["system"], "Synthetic system");
        assert!(seen[0].get("tools").is_none());
        assert_eq!(seen[0]["messages"].as_array().unwrap().len(), 1);
        assert_eq!(seen[0]["messages"][0]["role"], "user");
    }

    #[tokio::test]
    async fn configured_price_counts_anthropic_cache_fields() {
        let api = test_api(vec![(
            200,
            response(
                vec![json!({"type": "text", "text": "done"})],
                "end_turn",
                json!({
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cache_creation_input_tokens": 3,
                    "cache_read_input_tokens": 5
                }),
            ),
        )])
        .await;
        let result = test_provider_with_price(api.base.clone())
            .complete(test_request())
            .await
            .unwrap();
        let cost = result.cost_usd.expect("стоимость известна из конфига");
        assert!((cost - 0.0000635).abs() < 1e-12, "cost={cost}");
    }

    #[tokio::test]
    async fn tool_cycle_preserves_blocks_usage_and_transcript() {
        let first_blocks = vec![
            json!({"type": "thinking", "thinking": "plan", "signature": "signed"}),
            json!({
                "type": "tool_use",
                "id": "tool-1",
                "name": "mcp__fixture__poll",
                "input": {"value": 7}
            }),
        ];
        let api = test_api(vec![
            (
                200,
                response(
                    first_blocks.clone(),
                    "tool_use",
                    json!({
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "cache_creation_input_tokens": 2,
                        "cache_read_input_tokens": 3
                    }),
                ),
            ),
            (
                200,
                response(
                    vec![json!({"type": "text", "text": "final"})],
                    "end_turn",
                    json!({
                        "input_tokens": 5,
                        "output_tokens": 6,
                        "cache_creation_input_tokens": 1,
                        "cache_read_input_tokens": 2
                    }),
                ),
            ),
        ])
        .await;
        let (sink, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut req = agentic_request(&api.base, 3);
        req.turn_sink = Some(sink);
        let result = test_provider(api.base.clone(), false)
            .complete(req)
            .await
            .unwrap();
        assert_eq!(result.content, "final");
        assert_eq!(result.reasoning, None);
        assert_eq!(result.tokens_in, 23);
        assert_eq!(result.raw_input_tokens, 15);
        assert_eq!(result.cache_creation_input_tokens, 3);
        assert_eq!(result.cache_read_input_tokens, 5);
        assert_eq!(result.tokens_out, 10);
        let events: Vec<&str> = result
            .transcript
            .iter()
            .filter_map(|event| event["event"].as_str())
            .collect();
        assert_eq!(events, ["start", "turn", "tool_result", "turn", "final"]);
        let seen = api.seen.lock().unwrap();
        assert_eq!(
            seen[1]["messages"][1]["content"],
            Value::Array(first_blocks)
        );
        assert_eq!(seen[1]["messages"][2]["role"], "user");
        assert_eq!(
            seen[1]["messages"][2]["content"][0]["tool_use_id"],
            "tool-1"
        );
        assert_eq!(seen[0]["tools"][0]["name"], "mcp__fixture__poll");
        assert_eq!(seen[0]["tools"][0]["description"], "Read status");
        assert!(seen[0]["tools"][0].get("input_schema").is_some());
        assert!(seen[0]["tools"][0].get("function").is_none());
    }

    #[tokio::test]
    async fn two_tool_uses_become_one_user_message_in_order() {
        let api = test_api(vec![
            (
                200,
                response(
                    vec![
                        json!({"type": "tool_use", "id": "first", "name": "mcp__fixture__poll", "input": {"value": 1}}),
                        json!({"type": "tool_use", "id": "second", "name": "mcp__fixture__poll", "input": {"value": 2}}),
                    ],
                    "tool_use",
                    json!({"input_tokens": 2, "output_tokens": 1}),
                ),
            ),
            (200, text_response("done")),
        ])
        .await;
        test_provider(api.base.clone(), false)
            .complete(agentic_request(&api.base, 2))
            .await
            .unwrap();
        assert_eq!(api.calls.load(Ordering::SeqCst), 2);
        let seen = api.seen.lock().unwrap();
        let results = seen[1]["messages"][2]["content"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["tool_use_id"], "first");
        assert_eq!(results[1]["tool_use_id"], "second");
    }

    #[tokio::test]
    async fn max_turns_keeps_provider_name_and_usage() {
        let api = test_api(vec![(
            200,
            tool_turn("tool-1", "mcp__fixture__poll", json!({}), "tool_use"),
        )])
        .await;
        let error = test_provider(api.base.clone(), false)
            .complete(agentic_request(&api.base, 1))
            .await
            .expect_err("лимит обязан завершить цикл ошибкой");
        match error {
            LlmError::MaxTurns {
                provider,
                tokens_in,
                tokens_out,
                ..
            } => {
                assert_eq!(provider, "fixture-anthropic");
                assert_eq!(tokens_in, 2);
                assert_eq!(tokens_out, 1);
            }
            other => panic!("неожиданная ошибка: {other}"),
        }
    }

    #[tokio::test]
    async fn disallowed_tool_is_not_called() {
        let api = test_api(vec![
            (
                200,
                tool_turn("blocked", "mcp__fixture__blocked", json!({}), "tool_use"),
            ),
            (200, text_response("done")),
        ])
        .await;
        let mut req = agentic_request(&api.base, 2);
        req.cli_hints
            .as_mut()
            .unwrap()
            .disallowed_tools
            .push("mcp__fixture__blocked".into());
        test_provider(api.base.clone(), false)
            .complete(req)
            .await
            .unwrap();
        assert_eq!(api.calls.load(Ordering::SeqCst), 0);
        let seen = api.seen.lock().unwrap();
        assert!(seen[1]["messages"][2]["content"][0]["content"]
            .as_str()
            .unwrap()
            .contains("disallowed_tools"));
    }

    #[tokio::test]
    async fn max_tokens_with_tool_use_finishes_without_calling_tool() {
        let api = test_api(vec![(
            200,
            tool_turn("tool-1", "mcp__fixture__poll", json!({}), "max_tokens"),
        )])
        .await;
        let result = test_provider(api.base.clone(), false)
            .complete(agentic_request(&api.base, 2))
            .await
            .unwrap();
        assert_eq!(result.finish_reason, "max_tokens");
        assert_eq!(api.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn prompt_cache_marks_system_and_last_tool_only() {
        for enabled in [false, true] {
            let api = test_api(vec![(
                200,
                tool_turn("tool-1", "mcp__fixture__poll", json!({}), "max_tokens"),
            )])
            .await;
            test_provider(api.base.clone(), enabled)
                .complete(agentic_request(&api.base, 1))
                .await
                .unwrap();
            let seen = api.seen.lock().unwrap();
            if enabled {
                assert!(seen[0]["system"].is_array());
                assert_eq!(seen[0]["system"][0]["cache_control"]["type"], "ephemeral");
                assert_eq!(seen[0]["tools"][0]["cache_control"]["type"], "ephemeral");
            } else {
                assert_eq!(seen[0]["system"], "Synthetic system");
                assert!(seen[0]["tools"][0].get("cache_control").is_none());
            }
        }
    }

    #[tokio::test]
    async fn extra_body_is_flattened_at_top_level() {
        let api = test_api(vec![(200, text_response("done"))]).await;
        let mut req = test_request();
        req.extra_body.insert(
            "thinking".into(),
            json!({"type": "enabled", "budget_tokens": 1000}),
        );
        test_provider(api.base.clone(), false)
            .complete(req)
            .await
            .unwrap();
        let seen = api.seen.lock().unwrap();
        assert_eq!(seen[0]["thinking"]["type"], "enabled");
        assert_eq!(seen[0]["thinking"]["budget_tokens"], 1000);
    }

    #[tokio::test]
    async fn rate_limit_with_zero_retry_after_retries_once() {
        let api = test_api(vec![
            (429, json!({"error": {"message": "slow down"}})),
            (200, text_response("done")),
        ])
        .await;
        let result = test_provider(api.base.clone(), false)
            .complete(test_request())
            .await
            .unwrap();
        assert_eq!(result.content, "done");
        assert_eq!(api.seen.lock().unwrap().len(), 2);
    }

    #[test]
    fn default_temperature_is_not_serialized() {
        let provider = test_provider("http://unused".into(), false);
        let body = provider.request_body(&test_request(), &[], &[], DEFAULT_TEMPERATURE);
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn nonobject_tool_schema_gets_safe_object_fallback() {
        let provider = test_provider("http://unused".into(), false);
        let tool = super::super::mcp_client::ToolDef {
            full_name: "mcp__fixture__broken".into(),
            tool_name: "broken".into(),
            description: "Broken schema".into(),
            parameters: Value::String("not-an-object".into()),
        };
        let body = provider.request_body(&test_request(), &[], &[tool], DEFAULT_TEMPERATURE);
        assert_eq!(
            body["tools"][0]["input_schema"],
            json!({"type": "object", "properties": {}})
        );
    }
}
