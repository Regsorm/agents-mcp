//! Mock-провайдер: возвращает детерминированный ответ без сетевых вызовов.
//! Нужен для smoke-тестов рантайма и unit-тестов агентов без LLM-зависимостей.

use async_trait::async_trait;

use super::{LlmError, LlmProvider, LlmRequest, LlmResponse};

pub struct MockProvider;

impl MockProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError> {
        // Грубая оценка токенов — 4 символа/токен. На реальных провайдерах
        // заменяется ответом API.
        let tokens_in = (req.system_prompt.len() / 4 + req.user_input.len() / 4) as u32;
        let content = format!(
            "{{\"mock\":true,\"prompt_chars\":{},\"prompt_lines\":{},\"model\":{:?}}}",
            req.system_prompt.len(),
            req.system_prompt.lines().count(),
            req.model
        );
        Ok(LlmResponse {
            tokens_out: (content.len() / 4) as u32,
            content,
            tokens_in,
            cost_usd: Some(0.0),
            finish_reason: "stop".into(),
            reasoning: None,
            session_id: None,
            raw_input_tokens: tokens_in,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            transcript: Vec::new(),
        })
    }
}
