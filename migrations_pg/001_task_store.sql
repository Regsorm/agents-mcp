-- agents-mcp PG task-store (schema agents_mcp).
--
-- Externalized task context, replacing the claude-cli --resume transcript as the
-- carrier of task state. Lets sub-agents read curated slices independently
-- (parallel fan-out) or via dependency chains (DAG over task_artifacts.depends_on),
-- and removes the "prompt too long" failure class of the shared resume session.
--
-- Применение (по порядку, 001–006):
--   psql "$AGENTS_MCP_TASK_STORE_DSN" -f migrations_pg/001_task_store.sql
--
-- Reversible: DROP SCHEMA agents_mcp CASCADE;

CREATE SCHEMA IF NOT EXISTS agents_mcp;

-- Root task = identity + lifecycle. Key correlates to agent_calls.id (root call)
-- via root_call_id (weak reference inside the selected store backend).
CREATE TABLE IF NOT EXISTS agents_mcp.tasks (
    id               bigserial PRIMARY KEY,
    root_call_id     bigint,                        -- agent_calls.id of the orchestrator root call
    external_task_id text,                          -- client-side correlation id (от внешнего клиента)
    status           text NOT NULL DEFAULT 'queued',-- queued|running|needs_input|completed|failed
    task_kind        text,                          -- build_artifact|data_check|explain
    goal             text,                          -- goal anchor / success criteria
    target_base      text,
    working_dir      text,
    sandbox_path     text,
    progress         text,                          -- short running summary of work done
    final_output     text,                          -- outgoing answer to the user
    attachments      jsonb NOT NULL DEFAULT '[]',   -- outgoing artifacts (paths/kind)
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    finished_at      timestamptz
);
CREATE INDEX IF NOT EXISTS idx_tasks_status    ON agents_mcp.tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_root_call ON agents_mcp.tasks(root_call_id);

-- Blackboard: inputs/outputs of each step + curated context slices.
-- depends_on encodes the DAG (independent artifacts fan out in parallel,
-- dependent ones run in chain).
CREATE TABLE IF NOT EXISTS agents_mcp.task_artifacts (
    id               bigserial PRIMARY KEY,
    task_id          bigint NOT NULL REFERENCES agents_mcp.tasks(id) ON DELETE CASCADE,
    kind             text NOT NULL,                 -- metadata|query|bsl_module|form_def|build_path|review|deliverable
    key              text NOT NULL,                 -- logical name, e.g. "ObjectModule.source"
    content          text,                          -- the slice (compact) or a pointer to full
    summary          text,                          -- purpose, so an agent orients without reading all
    producer_agent   text,
    producer_call_id bigint,
    depends_on       bigint[] NOT NULL DEFAULT '{}',-- artifact ids this one depends on (DAG edges)
    status           text NOT NULL DEFAULT 'ready', -- ready|stale|pending
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX        IF NOT EXISTS idx_artifacts_task      ON agents_mcp.task_artifacts(task_id);
CREATE INDEX        IF NOT EXISTS idx_artifacts_task_kind ON agents_mcp.task_artifacts(task_id, kind);
CREATE UNIQUE INDEX IF NOT EXISTS uq_artifacts_task_key   ON agents_mcp.task_artifacts(task_id, key);

-- Durable event journal (incoming/outgoing, audit): the event stream of a task.
CREATE TABLE IF NOT EXISTS agents_mcp.task_events (
    id          bigserial PRIMARY KEY,
    task_id     bigint NOT NULL REFERENCES agents_mcp.tasks(id) ON DELETE CASCADE,
    seq         integer NOT NULL,
    event_type  text NOT NULL,                      -- user_input|agent_start|agent_done|artifact_written|error|status_change
    agent       text,
    call_id     bigint,
    payload     jsonb,
    ts          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_events_task_seq ON agents_mcp.task_events(task_id, seq);

-- Если служба подключается к базе НЕ владельцем схемы, выдайте этой роли права
-- на схему agents_mcp такими командами (замените <роль_службы> на имя своей
-- роли). Роль конкретной установки здесь не прописана: у вас её нет, и такой
-- GRANT упал бы с ошибкой.
-- GRANT USAGE ON SCHEMA agents_mcp TO <роль_службы>;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA agents_mcp TO <роль_службы>;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA agents_mcp TO <роль_службы>;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA agents_mcp
--     GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO <роль_службы>;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA agents_mcp
--     GRANT USAGE, SELECT ON SEQUENCES TO <роль_службы>;
