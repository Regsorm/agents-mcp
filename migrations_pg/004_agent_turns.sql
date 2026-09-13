-- agents-mcp 004: пошаговый транскрипт агентного цикла прямых провайдеров.
--
-- Провайдер openrouter.rs (в т.ч. локальная модель на llama-server через секцию
-- [providers.direct.<имя>]) при AGENTS_MCP_TRANSCRIPT=1 копит записи каждого хода
-- (start / turn / tool_result / final / max_turns), а рантайм (execute_ready)
-- пишет их сюда — по одной строке на запись, с жёсткой привязкой к
-- agent_calls.id через call_id. Нужен, чтобы изучать рассуждения локальной
-- модели и дорабатывать промпты: у llama-server своего транскрипта нет, а
-- анализ идём SQL'ом (сравнение с прогонами Opus по task_id/agent_calls), а не
-- чтением файлов.
--
-- record (jsonb) несёт поля хода: content, reasoning, finish, tool_calls,
-- tool/tool_result и т.п. Колонки seq/event вынесены наружу для сортировки и
-- фильтрации без разбора jsonb.
--
-- Применение (по порядку, 001–006):
--   psql "$AGENTS_MCP_TASK_STORE_DSN" -f migrations_pg/004_agent_turns.sql
--
-- Reversible:
--   DROP TABLE agents_mcp.agent_turns;

CREATE TABLE IF NOT EXISTS agents_mcp.agent_turns (
    id         bigserial PRIMARY KEY,
    call_id    bigint NOT NULL REFERENCES agents_mcp.agent_calls(id) ON DELETE CASCADE,
    seq        integer NOT NULL,               -- порядок записи в рамках вызова
    event      text    NOT NULL,               -- start|turn|single_shot|tool_result|final|max_turns
    record     jsonb   NOT NULL,               -- полная запись хода (content/reasoning/tool_calls/…)
    created_at bigint  NOT NULL                -- unixepoch миллисекунды
);
CREATE INDEX IF NOT EXISTS idx_agent_turns_call
    ON agents_mcp.agent_turns(call_id, seq);
CREATE INDEX IF NOT EXISTS idx_agent_turns_event
    ON agents_mcp.agent_turns(event);

-- Если служба подключается не владельцем схемы, права роли выдаются так
-- (замените <роль_службы> на имя своей роли):
-- GRANT SELECT, INSERT, UPDATE, DELETE
--     ON agents_mcp.agent_turns
--     TO <роль_службы>;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA agents_mcp TO <роль_службы>;
