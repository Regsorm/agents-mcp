-- agents-mcp: встроенное хранилище на SQLite (таблицы без схемы-префикса).
--
-- Те же семь таблиц, что в PostgreSQL (migrations_pg/001_task_store.sql —
-- 006_agent_calls_result_path.sql): колонки, индексы и уникальность
-- (task_id, key) у артефактов совпадают.
--
-- Соответствие типов PostgreSQL → SQLite:
--   bigserial → INTEGER PRIMARY KEY AUTOINCREMENT; bigint → INTEGER;
--   text → TEXT; double precision → REAL; boolean → INTEGER (0/1);
--   bytea → BLOB; jsonb → TEXT; bigint[] (depends_on) → TEXT с JSON-массивом;
--   timestamptz → INTEGER (секунды unixepoch).
--
-- Единицы времени как в PG: agent_calls.created_at и agent_cache.expires_at —
-- секунды; events.ts и agent_turns.created_at — миллисекунды.
--
-- Всё через IF NOT EXISTS: схема применяется при каждом открытии хранилища.
-- Грантов нет — доступ к файлу базы определяется файловой системой.

-- Корневая задача: идентичность и жизненный цикл. root_call_id слабо связан
-- с agent_calls.id корневого вызова.
CREATE TABLE IF NOT EXISTS tasks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    root_call_id     INTEGER,                        -- agent_calls.id вызова-оркестратора
    external_task_id TEXT,                           -- id задачи на стороне клиента
    status           TEXT NOT NULL DEFAULT 'queued', -- queued|running|needs_input|completed|failed
    task_kind        TEXT,                           -- build_artifact|data_check|explain
    goal             TEXT,
    target_base      TEXT,
    working_dir      TEXT,
    sandbox_path     TEXT,
    progress         TEXT,
    final_output     TEXT,
    attachments      TEXT NOT NULL DEFAULT '[]',     -- JSON-массив исходящих артефактов
    created_at       INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
    updated_at       INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
    finished_at      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tasks_status    ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_root_call ON tasks(root_call_id);

-- Доска задачи: входы/выходы шагов и срезы контекста. depends_on кодирует DAG
-- (JSON-массив id артефактов, по умолчанию пустой).
CREATE TABLE IF NOT EXISTS task_artifacts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id          INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    kind             TEXT NOT NULL,                  -- metadata|query|bsl_module|form_def|build_path|review|deliverable
    "key"            TEXT NOT NULL,                  -- логическое имя, напр. "ObjectModule.source"
    content          TEXT,
    summary          TEXT,
    producer_agent   TEXT,
    producer_call_id INTEGER,
    depends_on       TEXT NOT NULL DEFAULT '[]',     -- JSON-массив id артефактов (DAG)
    status           TEXT NOT NULL DEFAULT 'ready',  -- ready|stale|pending
    created_at       INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
    updated_at       INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER))
);
CREATE INDEX        IF NOT EXISTS idx_artifacts_task      ON task_artifacts(task_id);
CREATE INDEX        IF NOT EXISTS idx_artifacts_task_kind ON task_artifacts(task_id, kind);
CREATE UNIQUE INDEX IF NOT EXISTS uq_artifacts_task_key   ON task_artifacts(task_id, "key");

-- Долговечный журнал событий задачи. ts — секунды unixepoch.
CREATE TABLE IF NOT EXISTS task_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    event_type  TEXT NOT NULL,                       -- user_input|agent_start|agent_done|artifact_written|error|status_change
    agent       TEXT,
    call_id     INTEGER,
    payload     TEXT,                                -- JSON события
    ts          INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER))
);
CREATE INDEX IF NOT EXISTS idx_events_task_seq ON task_events(task_id, seq);

-- Журнал вызовов агентов: метрики, аудит, дерево (parent_call_id), session_id,
-- разбивка input-токенов. created_at — секунды unixepoch.
CREATE TABLE IF NOT EXISTS agent_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name        TEXT NOT NULL,
    variant           TEXT NOT NULL DEFAULT 'default',
    input_hash        TEXT NOT NULL,                 -- sha256 нормализованного input
    output_json       TEXT,                          -- результат, либо NULL при ошибке
    error             TEXT,
    model_used        TEXT NOT NULL,
    provider          TEXT NOT NULL,
    tokens_in         INTEGER,                       -- сумма raw+creation+read
    tokens_out        INTEGER,
    cost_usd          REAL,
    latency_ms        INTEGER,
    cached            INTEGER NOT NULL DEFAULT 0,    -- boolean 0/1
    created_at        INTEGER NOT NULL,              -- unixepoch (секунды)
    parent_call_id    INTEGER,
    session_id        TEXT,
    raw_input_in      INTEGER NOT NULL DEFAULT 0,
    cache_creation_in INTEGER NOT NULL DEFAULT 0,
    cache_read_in     INTEGER NOT NULL DEFAULT 0,
    status            TEXT NOT NULL DEFAULT 'running', -- running|done|error
    task_id           INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
    reasoning_content TEXT,                          -- размышления reasoning-моделей (NULL у прочих)
    instance          TEXT,                          -- экземпляр службы (имя машины:порт, NULL у старых сборок)
    result_path       TEXT                           -- файл-итог фонового agent_run (NULL у invoke)
);
CREATE INDEX IF NOT EXISTS idx_calls_agent_created
    ON agent_calls(agent_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_calls_input_hash
    ON agent_calls(input_hash);
CREATE INDEX IF NOT EXISTS idx_calls_agent_session
    ON agent_calls(agent_name, variant, created_at DESC)
    WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_calls_status
    ON agent_calls(status);
CREATE INDEX IF NOT EXISTS idx_calls_task
    ON agent_calls(task_id);

-- Кеш ответов для идемпотентных вызовов: cache_key = hash(agent+variant+task_id+input).
-- expires_at — секунды unixepoch.
CREATE TABLE IF NOT EXISTS agent_cache (
    cache_key     TEXT PRIMARY KEY,
    output_json   TEXT NOT NULL,
    metadata_json TEXT NOT NULL,                     -- model/tokens/cost для возврата
    expires_at    INTEGER NOT NULL                   -- unixepoch (секунды)
);
CREATE INDEX IF NOT EXISTS idx_cache_expires
    ON agent_cache(expires_at);

-- Журнал логов службы: структурный лог вместо текстовых файлов.
-- context — zstd-сжатый JSON структурированных полей; ts — МИЛЛИСЕКУНДЫ.
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       INTEGER NOT NULL,                       -- unixepoch миллисекунды
    level    INTEGER NOT NULL,                       -- 0=trace 1=debug 2=info 3=warn 4=error
    target   TEXT NOT NULL,
    message  TEXT NOT NULL,
    context  BLOB                                    -- zstd(JSON) или NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts
    ON events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_level_ts
    ON events(level, ts DESC);

-- Пошаговый транскрипт агентного цикла прямых провайдеров: по строке на запись
-- хода. record — полная запись хода (JSON), created_at — МИЛЛИСЕКУНДЫ.
CREATE TABLE IF NOT EXISTS agent_turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id    INTEGER NOT NULL REFERENCES agent_calls(id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,               -- порядок записи в рамках вызова
    event      TEXT    NOT NULL,               -- start|turn|single_shot|tool_result|final|max_turns
    record     TEXT    NOT NULL,               -- JSON записи хода
    created_at INTEGER NOT NULL                -- unixepoch миллисекунды
);
CREATE INDEX IF NOT EXISTS idx_agent_turns_call
    ON agent_turns(call_id, seq);
CREATE INDEX IF NOT EXISTS idx_agent_turns_event
    ON agent_turns(event);
