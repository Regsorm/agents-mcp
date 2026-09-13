-- 003_reasoning.sql
-- Добавляет колонку для размышлений reasoning-моделей (поле reasoning_content
-- из ответа OpenAI-совместимого API: gemma 4, deepseek, glm и т.п.).
-- Провайдер openrouter.rs читает reasoning_content, а рантайм сохраняет его
-- через Store::update_call. NULL у не-reasoning моделей.
ALTER TABLE agents_mcp.agent_calls
    ADD COLUMN IF NOT EXISTS reasoning_content text;
