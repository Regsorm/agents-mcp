//! SQLite-реализация хранилища службы — встроенный вариант: файл базы
//! создаётся сам при первом запуске, внешняя СУБД не нужна.
//!
//! Схема — `migrations_sqlite/001_init.sql`, применяется при каждом открытии.
//! Семантика методов согласована с PostgreSQL-реализацией
//! ([`super::pg::PgStore`]): те же статусы, единицы времени, условия и порядок
//! сортировки.
//! Отличия только диалектные — подстановки `?N`, `CAST(strftime('%s','now') AS
//! INTEGER)` вместо `now()`, JSON-массив в TEXT вместо `bigint[]`, `json_each`
//! вместо `= ANY(...)`.
//!
//! rusqlite синхронный, поэтому каждый метод типажа уводит работу в
//! `tokio::task::spawn_blocking`: блокировка соединения берётся и отпускается
//! внутри блокирующей задачи и через `await` не удерживается.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension};

use super::{
    validate_task_status_transition, Artifact, CachedEntry, CallRow, CallStatus, HistoryEntry,
    LogEvent, NewTask, OrphanedCall, Store, StoreError,
};

/// Схема SQLite — та же, что в migrations_pg, в диалекте SQLite.
const SCHEMA: &str = include_str!("../../migrations_sqlite/001_init.sql");
const RESULT_PATH_MIGRATION: &str =
    include_str!("../../migrations_sqlite/002_agent_calls_result_path.sql");

fn visible_error(error: anyhow::Error) -> anyhow::Error {
    anyhow::anyhow!("{error:#}")
}

/// Хранилище на SQLite. Соединение одно на всю службу (файл-то один): так его
/// WAL-режим, `busy_timeout` и `foreign_keys` действуют для всех запросов, а
/// одновременный доступ разводится мьютексом.
pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
    #[cfg(test)]
    fail_next_update_call: std::sync::atomic::AtomicBool,
    #[cfg(test)]
    fail_get_call_row: std::sync::atomic::AtomicBool,
}

impl SqliteStore {
    /// Открыть базу, создав при необходимости родительский каталог файла, и
    /// применить схему (`IF NOT EXISTS` — повторное применение безопасно).
    /// Путь `:memory:` (для тестов) каталога не требует и WAL не включает:
    /// журнал WAL в памяти не поддерживается.
    pub fn open(path: &Path) -> Result<Self> {
        let memory = path == Path::new(":memory:");
        if !memory {
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent).map_err(|error| {
                    anyhow::anyhow!(
                        "task-store: каталог {} не создан: {error}",
                        parent.display()
                    )
                })?;
            }
        }

        let conn = Connection::open(path).map_err(|error| {
            anyhow::anyhow!("task-store: открытие SQLite {}: {error}", path.display())
        })?;

        // WAL — читатели не блокируют писателя; busy_timeout — короткая
        // параллельная запись ждёт, а не падает «database is locked»;
        // foreign_keys — SQLite по умолчанию ссылочную целостность не проверяет.
        conn.execute_batch(if memory {
            "PRAGMA busy_timeout=5000; PRAGMA foreign_keys=ON;"
        } else {
            "PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000; PRAGMA foreign_keys=ON;"
        })
        .map_err(|error| anyhow::anyhow!("task-store: PRAGMA SQLite: {error}"))?;
        conn.execute_batch(SCHEMA)
            .map_err(|error| anyhow::anyhow!("task-store: применение схемы SQLite: {error}"))?;
        ensure_agent_calls_columns(&conn).map_err(visible_error)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            #[cfg(test)]
            fail_next_update_call: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            fail_get_call_row: std::sync::atomic::AtomicBool::new(false),
        })
    }

    #[cfg(test)]
    pub(crate) fn fail_next_update_call(&self) {
        self.fail_next_update_call
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    #[cfg(test)]
    pub(crate) fn set_get_call_row_failure(&self, fail: bool) {
        self.fail_get_call_row
            .store(fail, std::sync::atomic::Ordering::SeqCst);
    }

    /// Выполнить блокирующую работу на соединении. Мьютекс берётся внутри
    /// `spawn_blocking`, поэтому через `await` он не удерживается.
    async fn run<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let conn = self.conn.clone();
        let result = tokio::task::spawn_blocking(move || {
            let mut guard = conn
                .lock()
                .map_err(|_| anyhow::anyhow!("task-store: блокировка SQLite отравлена"))?;
            f(&mut guard)
        })
        .await
        .map_err(|error| anyhow::anyhow!("task-store: ожидание блокирующего запроса: {error}"))?;
        result.map_err(visible_error)
    }
}

/// Дополнить уже существующий файл БД новыми колонками `agent_calls`:
/// `CREATE TABLE IF NOT EXISTS` старую таблицу не меняет, поэтому без этой
/// правки вставка вызова в базу, созданную прежней сборкой, упала бы.
fn ensure_agent_calls_columns(conn: &Connection) -> Result<()> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(agent_calls)")
        .context("task-store: PRAGMA table_info(agent_calls)")?;
    let columns = stmt
        .query_map([], |r| r.get::<_, String>(1))
        .and_then(|m| m.collect::<rusqlite::Result<Vec<String>>>())
        .context("task-store: PRAGMA table_info(agent_calls)")?;
    if !columns.iter().any(|c| c == "instance") {
        conn.execute("ALTER TABLE agent_calls ADD COLUMN instance TEXT", [])
            .context("task-store: добавление колонки agent_calls.instance")?;
    }
    if !columns.iter().any(|c| c == "result_path") {
        conn.execute_batch(RESULT_PATH_MIGRATION)
            .context("task-store: добавление колонки agent_calls.result_path")?;
    }
    Ok(())
}

/// Строка `task_artifacts` → [`Artifact`]. `depends_on` лежит в TEXT как
/// JSON-массив; пустое или битое значение читается пустым вектором.
fn artifact_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Artifact> {
    let deps: String = r.get::<_, Option<String>>(7)?.unwrap_or_default();
    Ok(Artifact {
        id: r.get(0)?,
        kind: r.get(1)?,
        key: r.get(2)?,
        content: r.get(3)?,
        summary: r.get(4)?,
        producer_agent: r.get(5)?,
        producer_call_id: r.get(6)?,
        depends_on: serde_json::from_str(&deps).unwrap_or_default(),
        status: r.get(8)?,
    })
}

#[async_trait]
impl Store for SqliteStore {
    /// Проверка живости хранилища настоящим запросом.
    async fn health(&self) -> Result<()> {
        self.run(|conn| {
            conn.query_row("SELECT 1", [], |_| Ok(()))
                .context("task-store: SELECT 1")?;
            Ok(())
        })
        .await
    }

    /// Создать корневую задачу (статус running), вернуть её id.
    async fn create_task(&self, t: &NewTask) -> Result<i64> {
        let root_call_id = t.root_call_id;
        let external_task_id = t.external_task_id.clone();
        let task_kind = t.task_kind.clone();
        let goal = t.goal.clone();
        let target_base = t.target_base.clone();
        let working_dir = t.working_dir.clone();
        let sandbox_path = t.sandbox_path.clone();
        self.run(move |conn| {
            let id = conn
                .query_row(
                    "INSERT INTO tasks \
                     (root_call_id, external_task_id, status, task_kind, goal, target_base, working_dir, sandbox_path) \
                     VALUES (?1,?2,'running',?3,?4,?5,?6,?7) RETURNING id",
                    rusqlite::params![
                        root_call_id,
                        external_task_id,
                        task_kind,
                        goal,
                        target_base,
                        working_dir,
                        sandbox_path
                    ],
                    |r| r.get::<_, i64>(0),
                )
                .context("task-store: INSERT tasks")?;
            Ok(id)
        })
        .await
    }

    /// Сменить статус задачи; для completed/failed/cancelled выставить finished_at.
    async fn set_task_status(&self, task_id: i64, status: &str) -> Result<()> {
        let status = status.to_string();
        self.run(move |conn| {
            let current = conn
                .query_row(
                    "SELECT status FROM tasks WHERE id=?1",
                    rusqlite::params![task_id],
                    |row| row.get::<_, String>(0),
                )
                .optional()
                .context("task-store: SELECT status")?
                .ok_or(StoreError::TaskNotFound(task_id))?;
            validate_task_status_transition(&current, &status)?;
            let changed = conn
                .execute(
                    "UPDATE tasks SET status=?2, \
                 updated_at = CAST(strftime('%s','now') AS INTEGER), \
                 finished_at = CASE WHEN ?2 IN ('completed','failed','cancelled') \
                   THEN CAST(strftime('%s','now') AS INTEGER) ELSE NULL END \
                 WHERE id=?1",
                    rusqlite::params![task_id, status],
                )
                .context("task-store: UPDATE status")?;
            if changed != 1 {
                return Err(StoreError::TaskNotFound(task_id).into());
            }
            Ok(())
        })
        .await
    }

    /// Текущий статус задачи; None — такой задачи нет.
    async fn get_task_status(&self, task_id: i64) -> Result<Option<String>> {
        self.run(move |conn| {
            conn.query_row(
                "SELECT status FROM tasks WHERE id=?1",
                rusqlite::params![task_id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .context("task-store: SELECT status")
        })
        .await
    }

    /// Записать/обновить артефакт (upsert по (task_id,key)). Вернуть его id.
    #[allow(clippy::too_many_arguments)]
    async fn write_artifact(
        &self,
        task_id: i64,
        kind: &str,
        key: &str,
        content: Option<&str>,
        summary: Option<&str>,
        producer_agent: Option<&str>,
        producer_call_id: Option<i64>,
        depends_on: &[i64],
    ) -> Result<i64> {
        let kind = kind.to_string();
        let key = key.to_string();
        let content = content.map(str::to_string);
        let summary = summary.map(str::to_string);
        let producer_agent = producer_agent.map(str::to_string);
        let deps = serde_json::to_string(depends_on).context("task-store: depends_on в JSON")?;
        self.run(move |conn| {
            let id = conn
                .query_row(
                    "INSERT INTO task_artifacts \
                     (task_id, kind, \"key\", content, summary, producer_agent, producer_call_id, depends_on) \
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8) \
                     ON CONFLICT (task_id, \"key\") DO UPDATE SET \
                       kind=excluded.kind, content=excluded.content, summary=excluded.summary, \
                       producer_agent=excluded.producer_agent, producer_call_id=excluded.producer_call_id, \
                       depends_on=excluded.depends_on, status='ready', \
                       updated_at=CAST(strftime('%s','now') AS INTEGER) \
                     RETURNING id",
                    rusqlite::params![
                        task_id,
                        kind,
                        key,
                        content,
                        summary,
                        producer_agent,
                        producer_call_id,
                        deps
                    ],
                    |r| r.get::<_, i64>(0),
                )
                .context("task-store: upsert artifact")?;
            Ok(id)
        })
        .await
    }

    /// Прочитать артефакты задачи; опционально только указанных `kind`.
    async fn read_artifacts(
        &self,
        task_id: i64,
        kinds: Option<&[String]>,
    ) -> Result<Vec<Artifact>> {
        let kinds_json = match kinds {
            Some(ks) => Some(serde_json::to_string(ks).context("task-store: kinds в JSON")?),
            None => None,
        };
        self.run(move |conn| {
            let sql = "SELECT id, kind, \"key\", content, summary, producer_agent, producer_call_id, depends_on, status \
                       FROM task_artifacts WHERE task_id=?1 ORDER BY id";
            // kind = ANY($2) из PG в SQLite выражается через json_each: список
            // kind приезжает одной JSON-строкой.
            let sql_kinds = "SELECT id, kind, \"key\", content, summary, producer_agent, producer_call_id, depends_on, status \
                             FROM task_artifacts \
                             WHERE task_id=?1 AND kind IN (SELECT value FROM json_each(?2)) ORDER BY id";
            let rows = match kinds_json {
                Some(json) => {
                    let mut stmt = conn
                        .prepare(sql_kinds)
                        .context("task-store: подготовка SELECT artifacts")?;
                    stmt.query_map(rusqlite::params![task_id, json], artifact_from_row)
                        .and_then(|m| m.collect::<rusqlite::Result<Vec<Artifact>>>())
                        .context("task-store: SELECT artifacts")?
                }
                None => {
                    let mut stmt = conn
                        .prepare(sql)
                        .context("task-store: подготовка SELECT artifacts")?;
                    stmt.query_map(rusqlite::params![task_id], artifact_from_row)
                        .and_then(|m| m.collect::<rusqlite::Result<Vec<Artifact>>>())
                        .context("task-store: SELECT artifacts")?
                }
            };
            Ok(rows)
        })
        .await
    }

    /// Добавить событие в журнал задачи (seq = max+1). `payload_json` — JSON-строка или None.
    async fn append_task_event(
        &self,
        task_id: i64,
        event_type: &str,
        agent: Option<&str>,
        call_id: Option<i64>,
        payload_json: Option<&str>,
    ) -> Result<()> {
        if let Some(payload) = payload_json {
            serde_json::from_str::<serde_json::Value>(payload).map_err(|error| {
                anyhow::anyhow!("task-store: payload события содержит невалидный JSON: {error}")
            })?;
        }
        let event_type = event_type.to_string();
        let agent = agent.map(str::to_string);
        let payload_json = payload_json.map(str::to_string);
        self.run(move |conn| {
            conn.execute(
                "INSERT INTO task_events (task_id, seq, event_type, agent, call_id, payload) \
                 VALUES (?1, \
                   COALESCE((SELECT MAX(seq)+1 FROM task_events WHERE task_id=?1), 0), \
                   ?2, ?3, ?4, ?5)",
                rusqlite::params![task_id, event_type, agent, call_id, payload_json],
            )
            .context("task-store: INSERT event")?;
            Ok(())
        })
        .await
    }

    /// Записать один ход прогона в `agent_turns` (потоковая запись ходов).
    async fn append_turn(
        &self,
        call_id: i64,
        seq: i32,
        event: &str,
        record_json: &str,
        ts_millis: i64,
    ) -> Result<()> {
        let event = event.to_string();
        let record_json = record_json.to_string();
        self.run(move |conn| {
            conn.execute(
                "INSERT INTO agent_turns (call_id, seq, event, record, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![call_id, seq, event, record_json, ts_millis],
            )
            .context("task-store: INSERT agent_turn")?;
            Ok(())
        })
        .await
    }

    /// Пометить ошибкой осиротевшие вызовы ЭТОГО экземпляра службы:
    /// `status='running'`, начатые раньше `service_started_at` (unixepoch
    /// запуска этой службы) и созданные тем же `instance`. Вызовы другой
    /// службы на общей базе и строки без экземпляра (старые сборки) не
    /// трогаются. Возвращает помеченные строки.
    async fn mark_orphaned_running(
        &self,
        service_started_at: i64,
        reason: &str,
        instance: &str,
    ) -> Result<Vec<OrphanedCall>> {
        let reason = reason.to_string();
        let instance = instance.to_string();
        self.run(move |conn| {
            let mut stmt = conn
                .prepare(
                    "UPDATE agent_calls \
                        SET status = 'error', error = ?2 \
                      WHERE status = 'running' AND created_at < ?1 AND instance = ?3 \
                      RETURNING id, agent_name, created_at, result_path",
                )
                .context("task-store: подготовка UPDATE осиротевших agent_calls")?;
            let rows = stmt
                .query_map(
                    rusqlite::params![service_started_at, reason, instance],
                    |r| {
                        Ok(OrphanedCall {
                            id: r.get(0)?,
                            agent_name: r.get(1)?,
                            created_at: r.get(2)?,
                            result_path: r.get(3)?,
                        })
                    },
                )
                .and_then(|m| m.collect::<rusqlite::Result<Vec<OrphanedCall>>>())
                .context("task-store: UPDATE осиротевших agent_calls")?;
            Ok(rows)
        })
        .await
    }

    /// Retention: чистка завершённых задач и старых вызовов (старше
    /// `calls_days`), логов (старше `events_days`) и истёкшего кеша.
    /// Возвращает счётчики удалённых вызовов, логов и строк кеша.
    /// `events.ts` хранится в МИЛЛИСЕКУНДАХ — порог домножается на 1000.
    async fn retain(&self, calls_days: i64, events_days: i64) -> Result<(u64, u64, u64)> {
        self.run(move |conn| {
            let now = chrono::Utc::now().timestamp();
            let calls_cutoff = now - calls_days * 86_400;
            let events_cutoff_ms = (now - events_days * 86_400) * 1000;

            let tx = conn.transaction().context("retention: BEGIN")?;
            tx.execute(
                "DELETE FROM tasks AS task \
                 WHERE task.status IN ('completed', 'failed', 'cancelled') \
                   AND task.updated_at < ?1 \
                   AND NOT EXISTS ( \
                       SELECT 1 FROM task_artifacts AS artifact \
                       WHERE artifact.task_id = task.id AND artifact.updated_at >= ?1 \
                   ) \
                   AND NOT EXISTS ( \
                       SELECT 1 FROM task_events AS event \
                       WHERE event.task_id = task.id AND event.ts >= ?1 \
                   )",
                rusqlite::params![calls_cutoff],
            )
            .context("retention: DELETE tasks")?;
            let deleted_calls = tx
                .execute(
                    "DELETE FROM agent_calls WHERE created_at < ?1",
                    rusqlite::params![calls_cutoff],
                )
                .context("retention: DELETE agent_calls")?;
            let deleted_events = tx
                .execute(
                    "DELETE FROM events WHERE ts < ?1",
                    rusqlite::params![events_cutoff_ms],
                )
                .context("retention: DELETE events")?;
            let deleted_cache = tx
                .execute(
                    "DELETE FROM agent_cache WHERE expires_at < ?1",
                    rusqlite::params![now],
                )
                .context("retention: DELETE agent_cache")?;
            tx.commit().context("retention: COMMIT")?;
            Ok((
                deleted_calls as u64,
                deleted_events as u64,
                deleted_cache as u64,
            ))
        })
        .await
    }

    /// Последние вызовы из `agent_calls` (для MCP-tool `agent_history`).
    async fn list_calls(
        &self,
        agent: Option<&str>,
        since: i64,
        limit: u32,
    ) -> Result<Vec<HistoryEntry>> {
        let agent = agent.map(str::to_string);
        let limit = limit.clamp(1, 1000) as i64;
        self.run(move |conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT id, agent_name, variant, model_used, provider, \
                            tokens_in, tokens_out, cost_usd, latency_ms, \
                            cached, created_at, error, parent_call_id, instance \
                     FROM agent_calls \
                     WHERE (?1 IS NULL OR agent_name = ?1) \
                       AND created_at >= ?2 \
                     ORDER BY id DESC \
                     LIMIT ?3",
                )
                .context("task-store: подготовка SELECT agent_calls")?;
            let rows = stmt
                .query_map(rusqlite::params![agent, since, limit], |r| {
                    Ok(HistoryEntry {
                        id: r.get(0)?,
                        agent_name: r.get(1)?,
                        variant: r.get(2)?,
                        model_used: r.get(3)?,
                        provider: r.get(4)?,
                        tokens_in: r.get::<_, Option<i64>>(5)?.map(|v| v as u32),
                        tokens_out: r.get::<_, Option<i64>>(6)?.map(|v| v as u32),
                        cost_usd: r.get(7)?,
                        latency_ms: r.get::<_, Option<i64>>(8)?.map(|v| v as u64),
                        cached: r.get(9)?,
                        created_at: r.get(10)?,
                        error: r.get(11)?,
                        parent_call_id: r.get(12)?,
                        instance: r.get(13)?,
                    })
                })
                .and_then(|m| m.collect::<rusqlite::Result<Vec<HistoryEntry>>>())
                .context("task-store: SELECT agent_calls")?;
            Ok(rows)
        })
        .await
    }

    /// `created_at` строки вызова (unixepoch); None — строки нет.
    async fn get_call_created_at(&self, call_id: i64) -> Result<Option<i64>> {
        self.run(move |conn| {
            conn.query_row(
                "SELECT created_at FROM agent_calls WHERE id = ?1",
                rusqlite::params![call_id],
                |r| r.get::<_, i64>(0),
            )
            .optional()
            .context("task-store: SELECT created_at")
        })
        .await
    }

    /// Создать заготовку строки вызова (`agent_calls`) до старта провайдера.
    #[allow(clippy::too_many_arguments)]
    async fn insert_call_stub(
        &self,
        agent: &str,
        variant: &str,
        input_hash: &str,
        model_used: &str,
        provider: &str,
        parent_call_id: Option<i64>,
        task_id: Option<i64>,
        instance: &str,
    ) -> Result<i64> {
        let agent = agent.to_string();
        let variant = variant.to_string();
        let input_hash = input_hash.to_string();
        let model_used = model_used.to_string();
        let provider = provider.to_string();
        let instance = instance.to_string();
        self.run(move |conn| {
            let now = chrono::Utc::now().timestamp();
            let id = conn
                .query_row(
                    "INSERT INTO agent_calls \
                     (agent_name, variant, input_hash, model_used, provider, created_at, \
                      parent_call_id, task_id, instance) \
                     VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9) RETURNING id",
                    rusqlite::params![
                        agent,
                        variant,
                        input_hash,
                        model_used,
                        provider,
                        now,
                        parent_call_id,
                        task_id,
                        instance
                    ],
                    |r| r.get::<_, i64>(0),
                )
                .context("task-store: INSERT agent_calls")?;
            Ok(id)
        })
        .await
    }

    async fn set_call_result_path(&self, call_id: i64, result_path: &Path) -> Result<()> {
        let result_path = result_path.to_string_lossy().into_owned();
        self.run(move |conn| {
            conn.execute(
                "UPDATE agent_calls SET result_path=?1 WHERE id=?2",
                rusqlite::params![result_path, call_id],
            )
            .context("task-store: UPDATE result_path")?;
            Ok(())
        })
        .await
    }

    async fn mark_call_cached(&self, call_id: i64) -> Result<()> {
        self.run(move |conn| {
            conn.execute(
                "UPDATE agent_calls SET cached=1 WHERE id=?1",
                rusqlite::params![call_id],
            )
            .context("task-store: UPDATE cached")?;
            Ok(())
        })
        .await
    }

    /// Дописать заготовку вызова финальными данными провайдера (UPDATE по id).
    #[allow(clippy::too_many_arguments)]
    async fn update_call(
        &self,
        call_id: i64,
        status: CallStatus,
        output_json: Option<&str>,
        error: Option<&str>,
        tokens_in: u32,
        tokens_out: u32,
        cost_usd: Option<f64>,
        latency_ms: u64,
        session_id: Option<&str>,
        raw_input_in: u32,
        cache_creation_in: u32,
        cache_read_in: u32,
        reasoning: Option<&str>,
    ) -> Result<()> {
        #[cfg(test)]
        if self
            .fail_next_update_call
            .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            return Err(anyhow::anyhow!("тестовый отказ записи итога"));
        }
        let output_json = output_json.map(str::to_string);
        let error = error.map(str::to_string);
        let session_id = session_id.map(str::to_string);
        let reasoning = reasoning.map(str::to_string);
        let status = status.as_str().to_string();
        // Метрики в PG — bigint; в SQLite это INTEGER, тип тот же i64.
        let tokens_in = tokens_in as i64;
        let tokens_out = tokens_out as i64;
        let latency_ms = latency_ms as i64;
        let raw_input_in = raw_input_in as i64;
        let cache_creation_in = cache_creation_in as i64;
        let cache_read_in = cache_read_in as i64;
        self.run(move |conn| {
            conn.execute(
                "UPDATE agent_calls SET \
                    output_json=?1, error=?2, tokens_in=?3, tokens_out=?4, cost_usd=?5, \
                    latency_ms=?6, session_id=?7, raw_input_in=?8, cache_creation_in=?9, \
                    cache_read_in=?10, reasoning_content=?11, status=?12 \
                 WHERE id=?13",
                rusqlite::params![
                    output_json,
                    error,
                    tokens_in,
                    tokens_out,
                    cost_usd,
                    latency_ms,
                    session_id,
                    raw_input_in,
                    cache_creation_in,
                    cache_read_in,
                    reasoning,
                    status,
                    call_id
                ],
            )
            .context("task-store: UPDATE agent_calls")?;
            Ok(())
        })
        .await
    }

    /// Ранний UPDATE: записать `session_id` в заготовку вызова до окончания
    /// provider.complete() — дочерние invoke иначе прочитают NULL.
    async fn set_call_session_id(&self, call_id: i64, session_id: &str) -> Result<()> {
        let session_id = session_id.to_string();
        self.run(move |conn| {
            conn.execute(
                "UPDATE agent_calls SET session_id=?1 WHERE id=?2",
                rusqlite::params![session_id, call_id],
            )
            .context("task-store: UPDATE session_id")?;
            Ok(())
        })
        .await
    }

    /// `session_id` вызова по его id; None — NULL или строки нет.
    async fn get_call_session_id(&self, call_id: i64) -> Result<Option<String>> {
        self.run(move |conn| {
            let row = conn
                .query_row(
                    "SELECT session_id FROM agent_calls WHERE id=?1",
                    rusqlite::params![call_id],
                    |r| r.get::<_, Option<String>>(0),
                )
                .optional()
                .context("task-store: SELECT session_id")?;
            Ok(row.flatten())
        })
        .await
    }

    /// Строка вызова для сборки итога (long-poll и мгновенная проверка).
    async fn get_call_row(&self, call_id: i64) -> Result<Option<CallRow>> {
        #[cfg(test)]
        if self
            .fail_get_call_row
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            return Err(anyhow::anyhow!("тестовый отказ чтения строки вызова"));
        }
        self.run(move |conn| {
            conn.query_row(
                "SELECT status, output_json, error, agent_name, variant, model_used, \
                        provider, tokens_in, tokens_out, cost_usd, latency_ms, cached, \
                        parent_call_id \
                 FROM agent_calls WHERE id = ?1",
                rusqlite::params![call_id],
                |r| {
                    Ok(CallRow {
                        status: r.get(0)?,
                        output_json: r.get(1)?,
                        error: r.get(2)?,
                        agent_name: r.get(3)?,
                        variant: r.get(4)?,
                        model_used: r.get(5)?,
                        provider: r.get(6)?,
                        tokens_in: r.get(7)?,
                        tokens_out: r.get(8)?,
                        cost_usd: r.get(9)?,
                        latency_ms: r.get(10)?,
                        cached: r.get(11)?,
                        parent_call_id: r.get(12)?,
                    })
                },
            )
            .optional()
            .context("task-store: SELECT call row")
        })
        .await
    }

    /// Найти в кеше живую запись по ключу. None — нет или истекла.
    async fn cache_lookup(&self, key: &str) -> Result<Option<CachedEntry>> {
        let key = key.to_string();
        self.run(move |conn| {
            let now = chrono::Utc::now().timestamp();
            conn.query_row(
                "SELECT output_json, metadata_json FROM agent_cache \
                 WHERE cache_key = ?1 AND expires_at > ?2",
                rusqlite::params![key, now],
                |r| {
                    Ok(CachedEntry {
                        output_json: r.get(0)?,
                        metadata_json: r.get(1)?,
                    })
                },
            )
            .optional()
            .context("task-store: SELECT cache")
        })
        .await
    }

    /// Записать ответ в кеш с TTL (upsert по cache_key).
    async fn cache_store(
        &self,
        key: &str,
        output_json: &str,
        metadata_json: &str,
        ttl_sec: u64,
    ) -> Result<()> {
        let key = key.to_string();
        let output_json = output_json.to_string();
        let metadata_json = metadata_json.to_string();
        self.run(move |conn| {
            let expires_at = chrono::Utc::now().timestamp() + ttl_sec as i64;
            conn.execute(
                "INSERT INTO agent_cache \
                 (cache_key, output_json, metadata_json, expires_at) \
                 VALUES (?1, ?2, ?3, ?4) \
                 ON CONFLICT (cache_key) DO UPDATE SET \
                   output_json = excluded.output_json, \
                   metadata_json = excluded.metadata_json, \
                   expires_at = excluded.expires_at",
                rusqlite::params![key, output_json, metadata_json, expires_at],
            )
            .context("task-store: upsert cache")?;
            Ok(())
        })
        .await
    }

    /// Записать пачку событий журнала одной транзакцией: либо вся пачка, либо
    /// ничего — частично записанный батч хуже потерянного.
    async fn write_events(&self, batch: &[LogEvent]) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let rows: Vec<(i64, i16, String, String, Option<Vec<u8>>)> = batch
            .iter()
            .map(|r| {
                (
                    r.ts_millis,
                    r.level,
                    r.target.clone(),
                    r.message.clone(),
                    r.context_zstd.clone(),
                )
            })
            .collect();
        self.run(move |conn| {
            let tx = conn.transaction().context("task-store: BEGIN events")?;
            {
                let mut stmt = tx
                    .prepare(
                        "INSERT INTO events (ts, level, target, message, context) \
                         VALUES (?1, ?2, ?3, ?4, ?5)",
                    )
                    .context("task-store: подготовка INSERT events")?;
                for (ts, level, target, message, context) in &rows {
                    stmt.execute(rusqlite::params![ts, level, target, message, context])
                        .context("task-store: INSERT events")?;
                }
            }
            tx.commit().context("task-store: COMMIT events")?;
            Ok(())
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Хранилище в памяти: схема применяется при открытии, файлов нет.
    async fn mem_store() -> SqliteStore {
        SqliteStore::open(Path::new(":memory:")).expect("open :memory:")
    }

    #[tokio::test]
    async fn health_ok() {
        let store = mem_store().await;
        store.health().await.expect("health");
    }

    #[tokio::test]
    async fn open_creates_parent_dir_and_file() {
        let dir = std::env::temp_dir().join(format!("agents-mcp-sqlite-{}", std::process::id()));
        let file = dir.join("nested").join("agents-mcp.sqlite");
        let store = SqliteStore::open(&file).expect("open файла");
        store.health().await.expect("health");
        assert!(file.exists(), "файл базы создаётся сам");
        drop(store);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn call_stub_lifecycle_round_trip() {
        let store = mem_store().await;
        let call_id = store
            .insert_call_stub(
                "mock-agent",
                "default",
                "hash-1",
                "mock",
                "mock",
                None,
                None,
                "test:1",
            )
            .await
            .expect("insert_call_stub");

        let row = store
            .get_call_row(call_id)
            .await
            .unwrap()
            .expect("строка есть");
        assert_eq!(row.status, "running");
        assert_eq!(row.agent_name, "mock-agent");
        assert_eq!(row.variant, "default");
        assert_eq!(row.tokens_in, None);
        assert_eq!(row.cost_usd, None);
        store
            .set_call_session_id(call_id, "sess-1")
            .await
            .expect("set_call_session_id");
        assert_eq!(
            store.get_call_session_id(call_id).await.unwrap().as_deref(),
            Some("sess-1")
        );

        store
            .update_call(
                call_id,
                CallStatus::Done,
                Some("{\"ok\":true}"),
                None,
                11,
                22,
                Some(0.25),
                321,
                Some("sess-1"),
                5,
                3,
                3,
                Some("рассуждения"),
            )
            .await
            .expect("update_call");

        let row = store
            .get_call_row(call_id)
            .await
            .unwrap()
            .expect("строка есть");
        assert_eq!(row.status, "done");
        assert_eq!(row.output_json.as_deref(), Some("{\"ok\":true}"));
        assert_eq!(row.tokens_in, Some(11));
        assert_eq!(row.tokens_out, Some(22));
        assert_eq!(row.cost_usd, Some(0.25));
        assert_eq!(row.latency_ms, Some(321));

        // История: фильтр по агенту и предел строк.
        let history = store.list_calls(Some("mock-agent"), 0, 10).await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].id, call_id);
        assert_eq!(history[0].tokens_in, Some(11));
        assert_eq!(history[0].latency_ms, Some(321));
        assert!(!history[0].cached);
        assert!(store
            .list_calls(Some("другого-агента"), 0, 10)
            .await
            .unwrap()
            .is_empty());
        assert_eq!(store.list_calls(None, 0, 1).await.unwrap().len(), 1);
        assert!(store
            .list_calls(None, i64::MAX, 10)
            .await
            .unwrap()
            .is_empty());

        assert!(store.get_call_created_at(call_id).await.unwrap().is_some());
        assert!(store
            .get_call_created_at(call_id + 1000)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn update_call_with_error_sets_error_status() {
        let store = mem_store().await;
        let call_id = store
            .insert_call_stub(
                "mock-agent",
                "default",
                "hash-2",
                "mock",
                "mock",
                None,
                None,
                "test:1",
            )
            .await
            .unwrap();
        store
            .update_call(
                call_id,
                CallStatus::Failed,
                None,
                Some("провайдер отказал"),
                0,
                0,
                None,
                5,
                None,
                0,
                0,
                0,
                None,
            )
            .await
            .expect("update_call с ошибкой");
        let row = store
            .get_call_row(call_id)
            .await
            .unwrap()
            .expect("строка есть");
        assert_eq!(row.status, "error");
        assert_eq!(row.error.as_deref(), Some("провайдер отказал"));
        assert_eq!(row.output_json, None);
        assert_eq!(row.cost_usd, None);
    }

    #[tokio::test]
    async fn append_turn_writes_rows() {
        let store = mem_store().await;
        let call_id = store
            .insert_call_stub(
                "mock-agent",
                "default",
                "hash-3",
                "mock",
                "mock",
                None,
                None,
                "test:1",
            )
            .await
            .unwrap();
        for (seq, event) in ["start", "turn", "tool_result", "final"].iter().enumerate() {
            store
                .append_turn(
                    call_id,
                    seq as i32,
                    event,
                    "{\"content\":\"ход\"}",
                    1_700_000_000_000 + seq as i64,
                )
                .await
                .expect("append_turn");
        }
    }

    #[tokio::test]
    async fn cache_store_lookup_replaces_and_expires() {
        let store = mem_store().await;
        assert!(store.cache_lookup("key-1").await.unwrap().is_none());

        store
            .cache_store("key-1", "{\"a\":1}", "{\"model\":\"mock\"}", 60)
            .await
            .expect("cache_store");
        let hit = store
            .cache_lookup("key-1")
            .await
            .unwrap()
            .expect("живая запись");
        assert_eq!(hit.output_json, "{\"a\":1}");
        assert_eq!(hit.metadata_json, "{\"model\":\"mock\"}");

        // Повторная запись тем же ключом заменяет содержимое (upsert).
        store
            .cache_store("key-1", "{\"a\":2}", "{\"model\":\"mock-2\"}", 60)
            .await
            .expect("cache_store повторно");
        let hit = store
            .cache_lookup("key-1")
            .await
            .unwrap()
            .expect("живая запись");
        assert_eq!(hit.output_json, "{\"a\":2}");
        assert_eq!(hit.metadata_json, "{\"model\":\"mock-2\"}");

        // TTL 0 → срок истёк сразу, запись не находится.
        store
            .cache_store("key-expired", "{}", "{}", 0)
            .await
            .expect("cache_store истёкший");
        assert!(store.cache_lookup("key-expired").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn missing_task_status_update_returns_not_found() {
        let store = mem_store().await;
        let error = store
            .set_task_status(123_456, "completed")
            .await
            .expect_err("несуществующая задача не должна считаться обновлённой");
        assert!(error.to_string().contains("NotFound"), "error={error}");
    }

    #[tokio::test]
    async fn task_status_validates_transition_and_clears_finished_at() {
        let store = mem_store().await;
        let task_id = store
            .create_task(&NewTask::default())
            .await
            .expect("создание задачи");

        store
            .set_task_status(task_id, "completed")
            .await
            .expect("завершение задачи");
        let finished = store
            .run(move |conn| {
                conn.query_row(
                    "SELECT finished_at FROM tasks WHERE id=?1",
                    rusqlite::params![task_id],
                    |row| row.get::<_, Option<i64>>(0),
                )
                .context("чтение finished_at")
            })
            .await
            .expect("finished_at читается");
        assert!(finished.is_some());

        let transition_error = store
            .set_task_status(task_id, "needs_input")
            .await
            .expect_err("из completed нельзя сразу перейти в needs_input");
        assert!(transition_error
            .to_string()
            .contains("недопустимый переход"));

        store
            .set_task_status(task_id, "running")
            .await
            .expect("повторный запуск задачи");
        let finished = store
            .run(move |conn| {
                conn.query_row(
                    "SELECT finished_at FROM tasks WHERE id=?1",
                    rusqlite::params![task_id],
                    |row| row.get::<_, Option<i64>>(0),
                )
                .context("чтение сброшенного finished_at")
            })
            .await
            .expect("finished_at читается");
        assert_eq!(finished, None);

        assert!(store.set_task_status(task_id, "unknown").await.is_err());
    }

    #[tokio::test]
    async fn board_task_artifacts_and_events() {
        let store = mem_store().await;
        let task_id = store
            .create_task(&NewTask {
                goal: Some("rust-smoke".into()),
                task_kind: Some("build_artifact".into()),
                ..Default::default()
            })
            .await
            .expect("create_task");

        let first = store
            .write_artifact(
                task_id,
                "query",
                "main.query",
                Some("SELECT 1"),
                Some("первый вариант"),
                Some("mock-agent"),
                None,
                &[],
            )
            .await
            .expect("write_artifact");
        // Тот же (task_id, key) — строка остаётся одна, содержимое обновляется.
        let second = store
            .write_artifact(
                task_id,
                "query",
                "main.query",
                Some("SELECT 2"),
                None,
                None,
                None,
                &[first],
            )
            .await
            .expect("write_artifact повторно");
        assert_eq!(first, second);

        let all = store.read_artifacts(task_id, None).await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].content.as_deref(), Some("SELECT 2"));
        assert_eq!(all[0].summary, None);
        assert_eq!(all[0].status, "ready");

        let by_kind = store
            .read_artifacts(task_id, Some(&["query".to_string()][..]))
            .await
            .unwrap();
        assert_eq!(by_kind.len(), 1);
        // depends_on читается обратно из JSON-массива.
        assert_eq!(by_kind[0].depends_on, vec![first]);
        assert!(store
            .read_artifacts(task_id, Some(&["deliverable".to_string()][..]))
            .await
            .unwrap()
            .is_empty());

        store
            .append_task_event(
                task_id,
                "agent_start",
                Some("mock-agent"),
                Some(first),
                None,
            )
            .await
            .expect("append_task_event");
        store
            .append_task_event(
                task_id,
                "agent_done",
                Some("mock-agent"),
                Some(first),
                Some("{\"ok\":true}"),
            )
            .await
            .expect("append_task_event с payload");
        store
            .set_task_status(task_id, "completed")
            .await
            .expect("set_task_status");
    }

    #[tokio::test]
    async fn task_event_rejects_invalid_json_payload() {
        let store = mem_store().await;
        let task_id = store
            .create_task(&NewTask::default())
            .await
            .expect("create_task");
        let error = store
            .append_task_event(task_id, "broken", None, None, Some("{oops"))
            .await
            .expect_err("невалидный JSON должен быть отвергнут");
        assert!(
            error.to_string().contains("невалидный JSON"),
            "error={error}"
        );

        let events = store
            .run(move |conn| {
                conn.query_row(
                    "SELECT COUNT(*) FROM task_events WHERE task_id=?1",
                    rusqlite::params![task_id],
                    |row| row.get::<_, i64>(0),
                )
                .context("чтение task_events")
            })
            .await
            .expect("task_events читаются");
        assert_eq!(events, 0);
    }

    #[tokio::test]
    async fn retention_removes_only_stale_finished_tasks() {
        let store = mem_store().await;

        let stale_finished = store.create_task(&NewTask::default()).await.unwrap();
        store
            .write_artifact(
                stale_finished,
                "query",
                "stale",
                Some("SELECT 1"),
                None,
                None,
                None,
                &[],
            )
            .await
            .expect("старый артефакт");
        store
            .append_task_event(stale_finished, "done", None, None, Some("{}"))
            .await
            .expect("старое событие");
        store
            .set_task_status(stale_finished, "completed")
            .await
            .unwrap();

        let stale_running = store.create_task(&NewTask::default()).await.unwrap();
        let fresh_artifact_task = store.create_task(&NewTask::default()).await.unwrap();
        store
            .write_artifact(
                fresh_artifact_task,
                "query",
                "fresh",
                Some("SELECT 2"),
                None,
                None,
                None,
                &[],
            )
            .await
            .expect("свежий артефакт");
        store
            .set_task_status(fresh_artifact_task, "failed")
            .await
            .unwrap();

        let fresh_event_task = store.create_task(&NewTask::default()).await.unwrap();
        store
            .append_task_event(fresh_event_task, "fresh", None, None, Some("{}"))
            .await
            .expect("свежее событие");
        store
            .set_task_status(fresh_event_task, "completed")
            .await
            .unwrap();

        let fresh_ts = chrono::Utc::now().timestamp();
        store
            .run(move |conn| {
                conn.execute(
                    "UPDATE tasks SET updated_at=0 WHERE id IN (?1, ?2, ?3, ?4)",
                    rusqlite::params![
                        stale_finished,
                        stale_running,
                        fresh_artifact_task,
                        fresh_event_task
                    ],
                )
                .context("состаривание задач")?;
                conn.execute(
                    "UPDATE task_artifacts \
                     SET created_at=0, updated_at=CASE WHEN task_id=?2 THEN ?3 ELSE 0 END \
                     WHERE task_id IN (?1, ?2)",
                    rusqlite::params![stale_finished, fresh_artifact_task, fresh_ts],
                )
                .context("состаривание артефактов")?;
                conn.execute(
                    "UPDATE task_events SET ts=CASE WHEN task_id=?2 THEN ?3 ELSE 0 END \
                     WHERE task_id IN (?1, ?2)",
                    rusqlite::params![stale_finished, fresh_event_task, fresh_ts],
                )
                .context("состаривание событий")?;
                Ok(())
            })
            .await
            .expect("подготовка retention");

        store.retain(1, 1).await.expect("retention");

        let counts = store
            .run(move |conn| {
                let count = |table: &str, task_id: i64| -> Result<i64> {
                    conn.query_row(
                        &format!("SELECT COUNT(*) FROM {table} WHERE task_id=?1"),
                        rusqlite::params![task_id],
                        |row| row.get(0),
                    )
                    .context("подсчёт дочерних строк")
                };
                let task_count = |task_id: i64| -> Result<i64> {
                    conn.query_row(
                        "SELECT COUNT(*) FROM tasks WHERE id=?1",
                        rusqlite::params![task_id],
                        |row| row.get(0),
                    )
                    .context("подсчёт задач")
                };
                Ok((
                    task_count(stale_finished)?,
                    count("task_artifacts", stale_finished)?,
                    count("task_events", stale_finished)?,
                    task_count(stale_running)?,
                    task_count(fresh_artifact_task)?,
                    task_count(fresh_event_task)?,
                ))
            })
            .await
            .expect("проверка retention");
        assert_eq!(counts, (0, 0, 0, 1, 1, 1));
    }

    #[tokio::test]
    async fn sqlite_error_display_keeps_original_cause() {
        let store = mem_store().await;
        let error = store
            .run(|conn| {
                conn.execute("INSERT INTO missing_table VALUES (1)", [])
                    .context("task-store: проверочная запись")?;
                Ok(())
            })
            .await
            .expect_err("таблица отсутствует");
        let message = error.to_string();
        assert!(message.contains("проверочная запись"), "error={message}");
        assert!(message.contains("no such table"), "error={message}");
    }

    #[tokio::test]
    async fn write_events_batch_with_and_without_context() {
        let store = mem_store().await;
        let batch = vec![
            LogEvent {
                ts_millis: 1_700_000_000_000,
                level: 2,
                target: "agents_mcp".into(),
                message: "первое событие".into(),
                context_zstd: Some(vec![1, 2, 3, 4]),
            },
            LogEvent {
                ts_millis: 1_700_000_001_000,
                level: 4,
                target: "agents_mcp".into(),
                message: "второе событие".into(),
                context_zstd: None,
            },
        ];
        store.write_events(&batch).await.expect("write_events");
        store.write_events(&[]).await.expect("пустая пачка");
    }

    #[tokio::test]
    async fn mark_orphaned_running_marks_only_old_calls() {
        let store = mem_store().await;
        let old_id = store
            .insert_call_stub(
                "mock-agent",
                "default",
                "hash-old",
                "mock",
                "mock",
                None,
                None,
                "test:1",
            )
            .await
            .unwrap();
        let result_path = Path::new("C:/Temp/orphaned-result.json");
        store
            .set_call_result_path(old_id, result_path)
            .await
            .expect("сохранение result_path осиротевшего вызова");
        let fresh_id = store
            .insert_call_stub(
                "mock-agent",
                "default",
                "hash-new",
                "mock",
                "mock",
                None,
                None,
                "test:1",
            )
            .await
            .unwrap();

        // Заготовка всегда пишет created_at «сейчас», поэтому старый вызов
        // получаем сдвигом метки времени в прошлое.
        store
            .run(move |conn| {
                conn.execute(
                    "UPDATE agent_calls SET created_at = created_at - 3600 WHERE id = ?1",
                    rusqlite::params![old_id],
                )
                .context("task-store: сдвиг created_at")?;
                Ok(())
            })
            .await
            .expect("сдвиг created_at");
        let old_created_at = store.get_call_created_at(old_id).await.unwrap().unwrap();

        let now = chrono::Utc::now().timestamp();
        let marked = store
            .mark_orphaned_running(now, "служба перезапущена во время вызова", "test:1")
            .await
            .expect("mark_orphaned_running");
        assert_eq!(marked.len(), 1, "помечается только старый вызов");
        assert_eq!(marked[0].id, old_id);
        assert_eq!(marked[0].agent_name, "mock-agent");
        assert_eq!(marked[0].created_at, old_created_at);
        assert_eq!(
            marked[0].result_path.as_deref(),
            Some("C:/Temp/orphaned-result.json")
        );

        let old_row = store
            .get_call_row(old_id)
            .await
            .unwrap()
            .expect("строка есть");
        assert_eq!(old_row.status, "error");
        let fresh_row = store
            .get_call_row(fresh_id)
            .await
            .unwrap()
            .expect("строка есть");
        assert_eq!(fresh_row.status, "running", "свежий вызов не помечается");

        // Повторный вызов ничего не находит: осиротевших больше нет.
        assert!(store
            .mark_orphaned_running(now, "повтор", "test:1")
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn orphan_marking_touches_only_own_instance() {
        let store = mem_store().await;
        let own_id = store
            .insert_call_stub(
                "a-agent", "default", "h-a", "mock", "mock", None, None, "a:1",
            )
            .await
            .unwrap();
        let other_id = store
            .insert_call_stub(
                "b-agent", "default", "h-b", "mock", "mock", None, None, "b:1",
            )
            .await
            .unwrap();
        let legacy_id = store
            .insert_call_stub(
                "c-agent", "default", "h-c", "mock", "mock", None, None, "c:1",
            )
            .await
            .unwrap();
        // Строка без экземпляра — так её оставила старая сборка.
        store
            .run(move |conn| {
                conn.execute(
                    "UPDATE agent_calls SET instance = NULL WHERE id = ?1",
                    rusqlite::params![legacy_id],
                )
                .context("task-store: обнуление instance")?;
                Ok(())
            })
            .await
            .expect("обнуление instance");
        // Все три строки начаты раньше запуска службы.
        store
            .run(move |conn| {
                conn.execute("UPDATE agent_calls SET created_at = created_at - 3600", [])
                    .context("task-store: сдвиг created_at")?;
                Ok(())
            })
            .await
            .expect("сдвиг created_at");

        let now = chrono::Utc::now().timestamp();
        let marked = store
            .mark_orphaned_running(now, "служба перезапущена во время вызова", "a:1")
            .await
            .expect("mark_orphaned_running");
        assert_eq!(marked.len(), 1, "помечается только свой экземпляр");
        assert_eq!(marked[0].id, own_id);

        let own = store
            .get_call_row(own_id)
            .await
            .unwrap()
            .expect("строка есть");
        assert_eq!(own.status, "error");
        let other = store
            .get_call_row(other_id)
            .await
            .unwrap()
            .expect("строка есть");
        assert_eq!(other.status, "running", "чужой экземпляр не трогаем");
        let legacy = store
            .get_call_row(legacy_id)
            .await
            .unwrap()
            .expect("строка есть");
        assert_eq!(legacy.status, "running", "строку без экземпляра не трогаем");

        // История показывает экземпляр; у старой строки его нет.
        let own_history = store.list_calls(Some("a-agent"), 0, 10).await.unwrap();
        assert_eq!(own_history[0].instance.as_deref(), Some("a:1"));
        let legacy_history = store.list_calls(Some("c-agent"), 0, 10).await.unwrap();
        assert_eq!(legacy_history[0].instance, None);
    }

    #[tokio::test]
    async fn open_adds_agent_calls_columns_to_existing_file() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("время")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("agents-mcp-sqlite-old-{nanos}.sqlite"));

        // База СТАРОЙ схемы: определение agent_calls из migrations_sqlite/001_init.sql
        // без колонок instance и result_path (индексы не создаём — они не нужны).
        {
            let conn = rusqlite::Connection::open(&path).expect("создание старой базы");
            conn.execute_batch(
                "CREATE TABLE agent_calls (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name        TEXT NOT NULL,
                    variant           TEXT NOT NULL DEFAULT 'default',
                    input_hash        TEXT NOT NULL,
                    output_json       TEXT,
                    error             TEXT,
                    model_used        TEXT NOT NULL,
                    provider          TEXT NOT NULL,
                    tokens_in         INTEGER,
                    tokens_out        INTEGER,
                    cost_usd          REAL,
                    latency_ms        INTEGER,
                    cached            INTEGER NOT NULL DEFAULT 0,
                    created_at        INTEGER NOT NULL,
                    parent_call_id    INTEGER,
                    session_id        TEXT,
                    raw_input_in      INTEGER NOT NULL DEFAULT 0,
                    cache_creation_in INTEGER NOT NULL DEFAULT 0,
                    cache_read_in     INTEGER NOT NULL DEFAULT 0,
                    status            TEXT NOT NULL DEFAULT 'running',
                    task_id           INTEGER,
                    reasoning_content TEXT
                );",
            )
            .expect("создание старой схемы");
        }

        let store = SqliteStore::open(&path).expect("open старой базы");
        let call_id = store
            .insert_call_stub(
                "x-agent", "default", "h-x", "mock", "mock", None, None, "x:1",
            )
            .await
            .expect("insert_call_stub в обновлённую базу");
        let history = store.list_calls(Some("x-agent"), 0, 10).await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].id, call_id);
        assert_eq!(history[0].instance.as_deref(), Some("x:1"));
        let result_path = Path::new("C:/Temp/orphan-result.json");
        store
            .set_call_result_path(call_id, result_path)
            .await
            .expect("result_path в обновлённой базе");
        assert_eq!(
            store
                .run(move |conn| {
                    conn.query_row(
                        "SELECT result_path FROM agent_calls WHERE id=?1",
                        rusqlite::params![call_id],
                        |r| r.get::<_, Option<String>>(0),
                    )
                    .context("чтение result_path из обновлённой базы")
                })
                .await
                .unwrap()
                .as_deref(),
            Some("C:/Temp/orphan-result.json"),
        );

        drop(store);
        // WAL-файлы базы остаются после соединения — убираем всё.
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("sqlite-wal"));
        let _ = std::fs::remove_file(path.with_extension("sqlite-shm"));
    }
}
