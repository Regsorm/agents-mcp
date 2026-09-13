-- agents-mcp 002: PostgreSQL-схема рантайм-слоя (schema agents_mcp).
--
-- При выборе PostgreSQL журнал вызовов агентов, кеш ответов и журнал логов
-- сервиса живут в той же БД, что и доска задач (tasks / task_artifacts /
-- task_events из 001). Вызовы привязаны к задаче через agent_calls.task_id,
-- доска и дерево вызовов читаются одним запросом по task_id. SQLite остаётся
-- поддерживаемым встроенным вариантом хранилища.
--
-- created_at / expires_at / ts хранятся как bigint unixepoch (created_at,
-- expires_at — секунды; events.ts — миллисекунды), 1:1 с прежней SQLite-схемой,
-- чтобы не менять контракт history(since) и формат метрик.
--
-- Применение (по порядку, 001–006):
--   psql "$AGENTS_MCP_TASK_STORE_DSN" -f migrations_pg/002_agent_runtime.sql
--
-- Reversible:
--   DROP TABLE agents_mcp.agent_calls, agents_mcp.agent_cache, agents_mcp.events;

CREATE SCHEMA IF NOT EXISTS agents_mcp;

-- Журнал вызовов агентов: метрики, аудит, дерево (parent_call_id), session_id
-- для --resume claude-cli, разбивка input-токенов, статус async-режима.
CREATE TABLE IF NOT EXISTS agents_mcp.agent_calls (
    id                bigserial PRIMARY KEY,
    agent_name        text NOT NULL,
    variant           text NOT NULL DEFAULT 'default',
    input_hash        text NOT NULL,                 -- sha256 нормализованного input
    output_json       text,                          -- результат, либо NULL при ошибке
    error             text,                          -- текст ошибки если была
    model_used        text NOT NULL,
    provider          text NOT NULL,
    tokens_in         bigint,                        -- сумма raw+creation+read
    tokens_out        bigint,
    cost_usd          double precision,
    latency_ms        bigint,
    cached            boolean NOT NULL DEFAULT false,
    created_at        bigint NOT NULL,               -- unixepoch (секунды)
    parent_call_id    bigint,                        -- id родительского вызова (дерево оркестрации)
    session_id        text,                          -- claude-cli session для --resume
    raw_input_in      bigint NOT NULL DEFAULT 0,     -- новые input-токены (не из кеша)
    cache_creation_in bigint NOT NULL DEFAULT 0,     -- записанные в кеш на этом запросе
    cache_read_in     bigint NOT NULL DEFAULT 0,     -- прочитанные из кеша (×10 дешевле)
    status            text NOT NULL DEFAULT 'running',-- running|done|error
    task_id           bigint REFERENCES agents_mcp.tasks(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_calls_agent_created
    ON agents_mcp.agent_calls(agent_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_calls_input_hash
    ON agents_mcp.agent_calls(input_hash);
CREATE INDEX IF NOT EXISTS idx_calls_agent_session
    ON agents_mcp.agent_calls(agent_name, variant, created_at DESC)
    WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_calls_status
    ON agents_mcp.agent_calls(status);
CREATE INDEX IF NOT EXISTS idx_calls_task
    ON agents_mcp.agent_calls(task_id);

-- Кеш ответов для идемпотентных вызовов. cache_key = hash(agent+variant+task_id+input_subset).
CREATE TABLE IF NOT EXISTS agents_mcp.agent_cache (
    cache_key     text PRIMARY KEY,
    output_json   text NOT NULL,
    metadata_json text NOT NULL,                     -- model/tokens/cost для возврата
    expires_at    bigint NOT NULL                    -- unixepoch (секунды)
);
CREATE INDEX IF NOT EXISTS idx_cache_expires
    ON agents_mcp.agent_cache(expires_at);

-- Журнал логов сервиса (структурный лог вместо текстовых файлов).
-- context — zstd-сжатый JSON структурированных полей.
CREATE TABLE IF NOT EXISTS agents_mcp.events (
    id       bigserial PRIMARY KEY,
    ts       bigint NOT NULL,                        -- unixepoch миллисекунды
    level    smallint NOT NULL,                      -- 0=trace 1=debug 2=info 3=warn 4=error
    target   text NOT NULL,
    message  text NOT NULL,
    context  bytea                                   -- zstd(JSON) или NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts
    ON agents_mcp.events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_level_ts
    ON agents_mcp.events(level, ts DESC);

-- Если служба подключается не владельцем схемы, права роли выдаются тем же
-- набором команд, что в 001 (замените <роль_службы> на имя своей роли):
-- GRANT USAGE ON SCHEMA agents_mcp TO <роль_службы>;
-- GRANT SELECT, INSERT, UPDATE, DELETE
--     ON agents_mcp.agent_calls, agents_mcp.agent_cache, agents_mcp.events
--     TO <роль_службы>;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA agents_mcp TO <роль_службы>;
