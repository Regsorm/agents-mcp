//! PostgreSQL-реализация хранилища службы (`agents_mcp`).
//!
//! Здесь лежит весь SQL службы: доска задач (`tasks`, `task_artifacts`,
//! `task_events`), журнал вызовов (`agent_calls`, `agent_turns`), кеш ответов
//! (`agent_cache`) и лог событий (`events`). Схема создаётся отдельно
//! (`migrations_pg/001_task_store.sql` — `006_agent_calls_result_path.sql`),
//! тут — клиент.
//!
//! Подключение — по DSN из конфига (`[storage].task_store_dsn`). Наружу
//! модуль отдаётся типажом [`crate::store::Store`]; SQL за его пределами нет.
#![allow(dead_code)]

use std::error::Error;
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use deadpool_postgres::{Manager, ManagerConfig, Pool, PoolError, RecyclingMethod, TimeoutType};
use tokio_postgres::Config as PgConfig;
use tokio_postgres_rustls::MakeRustlsConnect;

use super::{
    validate_task_status_transition, Artifact, CallRow, CallStatus, CachedEntry, HistoryEntry,
    LogEvent, NewTask, OrphanedCall, Store, StoreError,
};

/// Пул соединений к PG.
#[derive(Clone)]
pub struct PgStore {
    pool: Pool,
}

fn error_chain(error: &(dyn Error + 'static)) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}

fn postgres_error(context: &str, error: &tokio_postgres::Error) -> anyhow::Error {
    anyhow::anyhow!("{context}: {}", error_chain(error))
}

fn pool_error(error: PoolError) -> anyhow::Error {
    match error {
        PoolError::Timeout(TimeoutType::Wait) => {
            anyhow::anyhow!("task-store: истекло ожидание свободного соединения в пуле")
        }
        PoolError::Timeout(TimeoutType::Create) => {
            anyhow::anyhow!("task-store: нет соединения: истекло время подключения к PostgreSQL")
        }
        PoolError::Timeout(TimeoutType::Recycle) => anyhow::anyhow!(
            "task-store: нет соединения: истекло время проверки соединения PostgreSQL"
        ),
        PoolError::Backend(error) => {
            anyhow::anyhow!("task-store: нет соединения: {}", error_chain(&error))
        }
        error => anyhow::anyhow!("task-store: ошибка пула: {}", error_chain(&error)),
    }
}

impl PgStore {
    /// Построить пул из DSN (`postgres://user:pass@host:port/db`).
    pub fn connect(dsn: &str, max_size: usize) -> Result<Self> {
        let pg_config = PgConfig::from_str(dsn)
            .map_err(|error| anyhow::anyhow!("разбор DSN task-store: {error}"))?;
        let tls = MakeRustlsConnect::with_webpki_roots();
        let mgr = Manager::from_config(
            pg_config,
            tls,
            ManagerConfig {
                // Соединение перед выдачей проверяется настоящим запросом. При
                // Fast пул отдавал взятое из кармана соединение не глядя, и
                // мёртвое оставалось там навсегда: 20.08.2026 служба 5.4 суток
                // отвечала на пробу «ok», а каждый запрос падал с «connection
                // closed». Verified в этом случае выбрасывает негодное
                // соединение и открывает новое — переподключение выходит само.
                recycling_method: RecyclingMethod::Verified,
            },
        );
        let pool = Pool::builder(mgr)
            .max_size(max_size)
            .wait_timeout(Some(Duration::from_secs(10)))
            .create_timeout(Some(Duration::from_secs(10)))
            .recycle_timeout(Some(Duration::from_secs(5)))
            .runtime(deadpool_postgres::Runtime::Tokio1)
            .build()
            .map_err(|error| anyhow::anyhow!("сборка пула task-store: {error}"))?;
        Ok(Self { pool })
    }

    /// Соединение из пула. Приватное: SQL наружу не отдаётся, все запросы
    /// живут методами типажа [`Store`] ниже. Публичным этот метод был только
    /// ради участков рантайма, чей SQL ещё не переехал сюда, — сырое
    /// соединение и есть причина размазывания запросов по коду.
    async fn client(&self) -> Result<deadpool_postgres::Object> {
        self.pool.get().await.map_err(pool_error)
    }
}

#[async_trait]
impl Store for PgStore {
    /// Проверка живости соединения (для healthcheck).
    async fn health(&self) -> Result<()> {
        let client = self.client().await?;
        client
            .query_one("SELECT 1", &[])
            .await
            .map_err(|error| postgres_error("task-store: SELECT 1", &error))?;
        Ok(())
    }

    /// Создать корневую задачу (статус running), вернуть её id.
    async fn create_task(&self, t: &NewTask) -> Result<i64> {
        let client = self.client().await?;
        let row = client
            .query_one(
                "INSERT INTO agents_mcp.tasks \
                 (root_call_id, external_task_id, status, task_kind, goal, target_base, working_dir, sandbox_path) \
                 VALUES ($1,$2,'running',$3,$4,$5,$6,$7) RETURNING id",
                &[
                    &t.root_call_id,
                    &t.external_task_id,
                    &t.task_kind,
                    &t.goal,
                    &t.target_base,
                    &t.working_dir,
                    &t.sandbox_path,
                ],
            )
            .await
            .map_err(|error| postgres_error("task-store: INSERT tasks", &error))?;
        Ok(row.get::<_, i64>(0))
    }

    /// Сменить статус задачи; для completed/failed выставить finished_at.
    async fn set_task_status(&self, task_id: i64, status: &str) -> Result<()> {
        let client = self.client().await?;
        let current = client
            .query_opt(
                "SELECT status FROM agents_mcp.tasks WHERE id=$1",
                &[&task_id],
            )
            .await
            .map_err(|error| postgres_error("task-store: SELECT status", &error))?
            .ok_or(StoreError::TaskNotFound(task_id))?
            .get::<_, String>(0);
        validate_task_status_transition(&current, status)?;
        let changed = client
            .execute(
                "UPDATE agents_mcp.tasks SET status=$2, updated_at=now(), \
                 finished_at = CASE WHEN $2 IN ('completed','failed') THEN now() ELSE NULL END \
                 WHERE id=$1",
                &[&task_id, &status],
            )
            .await
            .map_err(|error| postgres_error("task-store: UPDATE status", &error))?;
        if changed != 1 {
            return Err(StoreError::TaskNotFound(task_id).into());
        }
        Ok(())
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
        let client = self.client().await?;
        let deps: Vec<i64> = depends_on.to_vec();
        let row = client
            .query_one(
                "INSERT INTO agents_mcp.task_artifacts \
                 (task_id, kind, key, content, summary, producer_agent, producer_call_id, depends_on) \
                 VALUES ($1,$2,$3,$4,$5,$6,$7,$8) \
                 ON CONFLICT (task_id, key) DO UPDATE SET \
                   kind=EXCLUDED.kind, content=EXCLUDED.content, summary=EXCLUDED.summary, \
                   producer_agent=EXCLUDED.producer_agent, producer_call_id=EXCLUDED.producer_call_id, \
                   depends_on=EXCLUDED.depends_on, status='ready', updated_at=now() \
                 RETURNING id",
                &[
                    &task_id,
                    &kind,
                    &key,
                    &content,
                    &summary,
                    &producer_agent,
                    &producer_call_id,
                    &deps,
                ],
            )
            .await
            .map_err(|error| postgres_error("task-store: upsert artifact", &error))?;
        Ok(row.get::<_, i64>(0))
    }

    /// Прочитать артефакты задачи; опционально только указанных `kind`.
    async fn read_artifacts(
        &self,
        task_id: i64,
        kinds: Option<&[String]>,
    ) -> Result<Vec<Artifact>> {
        let client = self.client().await?;
        let rows = match kinds {
            Some(ks) => {
                let ks: Vec<String> = ks.to_vec();
                client
                    .query(
                        "SELECT id, kind, key, content, summary, producer_agent, producer_call_id, depends_on, status \
                         FROM agents_mcp.task_artifacts WHERE task_id=$1 AND kind = ANY($2) ORDER BY id",
                        &[&task_id, &ks],
                    )
                    .await
            }
            None => {
                client
                    .query(
                        "SELECT id, kind, key, content, summary, producer_agent, producer_call_id, depends_on, status \
                         FROM agents_mcp.task_artifacts WHERE task_id=$1 ORDER BY id",
                        &[&task_id],
                    )
                    .await
            }
        }
        .map_err(|error| postgres_error("task-store: SELECT artifacts", &error))?;

        Ok(rows
            .into_iter()
            .map(|r| Artifact {
                id: r.get(0),
                kind: r.get(1),
                key: r.get(2),
                content: r.get(3),
                summary: r.get(4),
                producer_agent: r.get(5),
                producer_call_id: r.get(6),
                depends_on: r.get(7),
                status: r.get(8),
            })
            .collect())
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
        let client = self.client().await?;
        client
            .execute(
                "INSERT INTO agents_mcp.task_events (task_id, seq, event_type, agent, call_id, payload) \
                 VALUES ($1, \
                   COALESCE((SELECT MAX(seq)+1 FROM agents_mcp.task_events WHERE task_id=$1), 0), \
                   $2, $3, $4, $5::text::jsonb)",
                &[&task_id, &event_type, &agent, &call_id, &payload_json],
            )
            .await
            .map_err(|error| postgres_error("task-store: INSERT event", &error))?;
        Ok(())
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
        let client = self.client().await?;
        client
            .execute(
                "INSERT INTO agents_mcp.agent_turns \
                 (call_id, seq, event, record, created_at) \
                 VALUES ($1, $2, $3, $4::text::jsonb, $5)",
                &[&call_id, &seq, &event, &record_json, &ts_millis],
            )
            .await
            .map_err(|error| postgres_error("task-store: INSERT agent_turn", &error))?;
        Ok(())
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
        let client = self.client().await?;
        let rows = client
            .query(
                "UPDATE agents_mcp.agent_calls \
                    SET status = 'error', error = $2 \
                  WHERE status = 'running' AND created_at < $1 AND instance = $3 \
                  RETURNING id, agent_name, created_at, result_path",
                &[&service_started_at, &reason, &instance],
            )
            .await
            .map_err(|error| {
                postgres_error("task-store: UPDATE осиротевших agent_calls", &error)
            })?;
        Ok(rows
            .into_iter()
            .map(|r| OrphanedCall {
                id: r.get(0),
                agent_name: r.get(1),
                created_at: r.get(2),
                result_path: r.get(3),
            })
            .collect())
    }

    /// Retention: чистка завершённых задач и старых вызовов (старше
    /// `calls_days`), логов (старше `events_days`) и истёкшего кеша.
    /// Возвращает счётчики удалённых вызовов, логов и строк кеша.
    /// `events.ts` хранится в МИЛЛИСЕКУНДАХ — порог домножается на 1000.
    async fn retain(&self, calls_days: i64, events_days: i64) -> Result<(u64, u64, u64)> {
        let mut client = self.client().await?;
        let now = chrono::Utc::now().timestamp();
        let calls_cutoff = now - calls_days * 86_400;
        let events_cutoff_ms = (now - events_days * 86_400) * 1000;

        let tx = client
            .transaction()
            .await
            .map_err(|error| postgres_error("retention: BEGIN", &error))?;
        tx.execute(
            "DELETE FROM agents_mcp.tasks AS task \
             WHERE task.status IN ('completed', 'failed') \
               AND task.updated_at < to_timestamp($1::bigint) \
               AND NOT EXISTS ( \
                   SELECT 1 FROM agents_mcp.task_artifacts AS artifact \
                   WHERE artifact.task_id = task.id \
                     AND artifact.updated_at >= to_timestamp($1::bigint) \
               ) \
               AND NOT EXISTS ( \
                   SELECT 1 FROM agents_mcp.task_events AS event \
                   WHERE event.task_id = task.id \
                     AND event.ts >= to_timestamp($1::bigint) \
               )",
            &[&calls_cutoff],
        )
        .await
        .map_err(|error| postgres_error("retention: DELETE tasks", &error))?;
        let deleted_calls = tx
            .execute(
                "DELETE FROM agents_mcp.agent_calls WHERE created_at < $1",
                &[&calls_cutoff],
            )
            .await
            .map_err(|error| postgres_error("retention: DELETE agent_calls", &error))?;
        let deleted_events = tx
            .execute(
                "DELETE FROM agents_mcp.events WHERE ts < $1",
                &[&events_cutoff_ms],
            )
            .await
            .map_err(|error| postgres_error("retention: DELETE events", &error))?;
        let deleted_cache = tx
            .execute(
                "DELETE FROM agents_mcp.agent_cache WHERE expires_at < $1",
                &[&now],
            )
            .await
            .map_err(|error| postgres_error("retention: DELETE agent_cache", &error))?;
        tx.commit()
            .await
            .map_err(|error| postgres_error("retention: COMMIT", &error))?;
        Ok((deleted_calls, deleted_events, deleted_cache))
    }

    /// Последние вызовы из `agent_calls` (для MCP-tool `agent_history`).
    async fn list_calls(
        &self,
        agent: Option<&str>,
        since: i64,
        limit: u32,
    ) -> Result<Vec<HistoryEntry>> {
        let limit = limit.clamp(1, 1000) as i64;
        let client = self.client().await?;
        let rows = client
            .query(
                "SELECT id, agent_name, variant, model_used, provider, \
                        tokens_in, tokens_out, cost_usd, latency_ms, \
                        cached, created_at, error, parent_call_id, instance \
                 FROM agents_mcp.agent_calls \
                 WHERE ($1::text IS NULL OR agent_name = $1) \
                   AND created_at >= $2 \
                 ORDER BY id DESC \
                 LIMIT $3",
                &[&agent, &since, &limit],
            )
            .await
            .map_err(|error| postgres_error("task-store: SELECT agent_calls", &error))?;
        Ok(rows
            .into_iter()
            .map(|r| HistoryEntry {
                id: r.get(0),
                agent_name: r.get(1),
                variant: r.get(2),
                model_used: r.get(3),
                provider: r.get(4),
                tokens_in: r.get::<_, Option<i64>>(5).map(|v| v as u32),
                tokens_out: r.get::<_, Option<i64>>(6).map(|v| v as u32),
                cost_usd: r.get(7),
                latency_ms: r.get::<_, Option<i64>>(8).map(|v| v as u64),
                cached: r.get::<_, bool>(9),
                created_at: r.get(10),
                error: r.get(11),
                parent_call_id: r.get(12),
                instance: r.get(13),
            })
            .collect())
    }

    /// `created_at` строки вызова (unixepoch); None — строки нет.
    async fn get_call_created_at(&self, call_id: i64) -> Result<Option<i64>> {
        let client = self.client().await?;
        let row = client
            .query_opt(
                "SELECT created_at FROM agents_mcp.agent_calls WHERE id = $1",
                &[&call_id],
            )
            .await
            .map_err(|error| postgres_error("task-store: SELECT created_at", &error))?;
        Ok(row.map(|r| r.get::<_, i64>(0)))
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
        let now = chrono::Utc::now().timestamp();
        let client = self.client().await?;
        let row = client
            .query_one(
                "INSERT INTO agents_mcp.agent_calls \
                 (agent_name, variant, input_hash, model_used, provider, created_at, \
                  parent_call_id, task_id, instance) \
                 VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING id",
                &[
                    &agent,
                    &variant,
                    &input_hash,
                    &model_used,
                    &provider,
                    &now,
                    &parent_call_id,
                    &task_id,
                    &instance,
                ],
            )
            .await
            .map_err(|error| postgres_error("task-store: INSERT agent_calls", &error))?;
        Ok(row.get::<_, i64>(0))
    }

    async fn set_call_result_path(&self, call_id: i64, result_path: &Path) -> Result<()> {
        let result_path = result_path.to_string_lossy().into_owned();
        let client = self.client().await?;
        client
            .execute(
                "UPDATE agents_mcp.agent_calls SET result_path=$1 WHERE id=$2",
                &[&result_path, &call_id],
            )
            .await
            .map_err(|error| postgres_error("task-store: UPDATE result_path", &error))?;
        Ok(())
    }

    async fn mark_call_cached(&self, call_id: i64) -> Result<()> {
        let client = self.client().await?;
        client
            .execute(
                "UPDATE agents_mcp.agent_calls SET cached=true WHERE id=$1",
                &[&call_id],
            )
            .await
            .map_err(|error| postgres_error("task-store: UPDATE cached", &error))?;
        Ok(())
    }

    /// Дописать заготовку вызова финальными данными провайдера (UPDATE по id).
    /// Безопасно вызывать и при успехе (output/tokens/cost/session_id), и при
    /// ошибке (error, метрики 0).
    ///
    /// Детализация input-токенов (миграция 004):
    ///   raw_input_in       — новые, не из кеша
    ///   cache_creation_in  — записанные в кеш на этом запросе
    ///   cache_read_in      — прочитанные из кеша (×10 дешевле)
    /// tokens_in остаётся как сумма (для обратной совместимости и быстрых
    /// агрегатных запросов).
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
        let status = status.as_str();

        let client = self.client().await?;
        client
            .execute(
                "UPDATE agents_mcp.agent_calls SET \
                    output_json=$1, error=$2, tokens_in=$3, tokens_out=$4, cost_usd=$5, \
                    latency_ms=$6, session_id=$7, raw_input_in=$8, cache_creation_in=$9, \
                    cache_read_in=$10, reasoning_content=$11, status=$12 \
                 WHERE id=$13",
                &[
                    &output_json,
                    &error,
                    &(tokens_in as i64),
                    &(tokens_out as i64),
                    &cost_usd,
                    &(latency_ms as i64),
                    &session_id,
                    &(raw_input_in as i64),
                    &(cache_creation_in as i64),
                    &(cache_read_in as i64),
                    &reasoning,
                    &status,
                    &call_id,
                ],
            )
            .await
            .map_err(|error| postgres_error("task-store: UPDATE agent_calls", &error))?;
        Ok(())
    }

    /// Ранний UPDATE: записать `session_id` в заготовку вызова ДО окончания
    /// provider.complete() — дочерние invoke внутри subprocess parent'а иначе
    /// прочитают NULL.
    async fn set_call_session_id(&self, call_id: i64, session_id: &str) -> Result<()> {
        let client = self.client().await?;
        client
            .execute(
                "UPDATE agents_mcp.agent_calls SET session_id=$1 WHERE id=$2",
                &[&session_id, &call_id],
            )
            .await
            .map_err(|error| postgres_error("task-store: UPDATE session_id", &error))?;
        Ok(())
    }

    /// `session_id` вызова по его id; None — NULL или строки нет.
    async fn get_call_session_id(&self, call_id: i64) -> Result<Option<String>> {
        let client = self.client().await?;
        let row = client
            .query_opt(
                "SELECT session_id FROM agents_mcp.agent_calls WHERE id=$1",
                &[&call_id],
            )
            .await
            .map_err(|error| postgres_error("task-store: SELECT session_id", &error))?;
        Ok(row.and_then(|r| r.get::<_, Option<String>>(0)))
    }

    /// Строка вызова для сборки итога (long-poll и мгновенная проверка).
    async fn get_call_row(&self, call_id: i64) -> Result<Option<CallRow>> {
        let client = self.client().await?;
        let row = client
            .query_opt(
                "SELECT status, output_json, error, agent_name, variant, model_used, \
                        provider, tokens_in, tokens_out, cost_usd, latency_ms, cached, \
                        parent_call_id \
                 FROM agents_mcp.agent_calls WHERE id = $1",
                &[&call_id],
            )
            .await
            .map_err(|error| postgres_error("task-store: SELECT call outcome", &error))?;
        Ok(row.map(|r| CallRow {
            status: r.get(0),
            output_json: r.get(1),
            error: r.get(2),
            agent_name: r.get(3),
            variant: r.get(4),
            model_used: r.get(5),
            provider: r.get(6),
            tokens_in: r.get(7),
            tokens_out: r.get(8),
            cost_usd: r.get(9),
            latency_ms: r.get(10),
            cached: r.get(11),
            parent_call_id: r.get(12),
        }))
    }

    /// Найти в кеше живую запись по ключу. None — нет или истекла.
    async fn cache_lookup(&self, key: &str) -> Result<Option<CachedEntry>> {
        let now = chrono::Utc::now().timestamp();
        let client = self.client().await?;
        let row = client
            .query_opt(
                "SELECT output_json, metadata_json FROM agents_mcp.agent_cache \
                 WHERE cache_key = $1 AND expires_at > $2",
                &[&key, &now],
            )
            .await
            .map_err(|error| postgres_error("task-store: SELECT agent_cache", &error))?;
        Ok(row.map(|r| CachedEntry {
            output_json: r.get(0),
            metadata_json: r.get(1),
        }))
    }

    /// Записать ответ в кеш с TTL (upsert по cache_key).
    async fn cache_store(
        &self,
        key: &str,
        output_json: &str,
        metadata_json: &str,
        ttl_sec: u64,
    ) -> Result<()> {
        let expires_at = chrono::Utc::now().timestamp() + ttl_sec as i64;
        let client = self.client().await?;
        client
            .execute(
                "INSERT INTO agents_mcp.agent_cache \
                 (cache_key, output_json, metadata_json, expires_at) \
                 VALUES ($1, $2, $3, $4) \
                 ON CONFLICT (cache_key) DO UPDATE SET \
                   output_json = EXCLUDED.output_json, \
                   metadata_json = EXCLUDED.metadata_json, \
                   expires_at = EXCLUDED.expires_at",
                &[&key, &output_json, &metadata_json, &expires_at],
            )
            .await
            .map_err(|error| postgres_error("task-store: UPSERT agent_cache", &error))?;
        Ok(())
    }

    /// Записать пачку событий журнала одной транзакцией: либо вся пачка, либо
    /// ничего — частично записанный батч хуже потерянного.
    async fn write_events(&self, batch: &[LogEvent]) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let mut client = self.client().await?;
        let tx = client
            .transaction()
            .await
            .map_err(|error| postgres_error("task-store: BEGIN events", &error))?;
        let stmt = tx
            .prepare(
                "INSERT INTO agents_mcp.events (ts, level, target, message, context) \
                 VALUES ($1, $2, $3, $4, $5)",
            )
            .await
            .map_err(|error| postgres_error("task-store: PREPARE events", &error))?;
        for r in batch {
            tx.execute(
                &stmt,
                &[
                    &r.ts_millis,
                    &r.level,
                    &r.target,
                    &r.message,
                    &r.context_zstd,
                ],
            )
            .await
            .map_err(|error| postgres_error("task-store: INSERT events", &error))?;
        }
        tx.commit()
            .await
            .map_err(|error| postgres_error("task-store: COMMIT events", &error))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::time::{timeout, Duration};

    const SSL_REQUEST: [u8; 8] = [0, 0, 0, 8, 4, 210, 22, 47];

    #[test]
    fn pool_has_bounded_wait_create_and_recycle() {
        let store = PgStore::connect("postgres://user:pass@127.0.0.1/db", 3).unwrap();
        let timeouts = store.pool.timeouts();
        assert_eq!(timeouts.wait, Some(Duration::from_secs(10)));
        assert_eq!(timeouts.create, Some(Duration::from_secs(10)));
        assert_eq!(timeouts.recycle, Some(Duration::from_secs(5)));
    }

    #[test]
    fn pool_wait_timeout_has_distinct_message() {
        let error = pool_error(PoolError::Timeout(TimeoutType::Wait));
        let message = error.to_string();
        assert!(message.contains("истекло ожидание"), "error={message}");
        assert!(!message.contains("нет соединения"), "error={message}");
    }

    #[tokio::test]
    async fn unavailable_postgres_keeps_connection_cause_in_display() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("local_addr");
        drop(listener);

        let store = PgStore::connect(
            &format!("postgres://user:pass@{address}/agents?connect_timeout=1"),
            1,
        )
        .expect("connect");
        let error = store.health().await.expect_err("порт закрыт");
        let message = error.to_string();
        let cause = message
            .strip_prefix("task-store: нет соединения: ")
            .expect("ошибка классифицирована как отказ соединения");
        assert!(cause.contains(":"), "исходная цепочка причин потеряна: {message}");
    }

    async fn tls_rejecting_server(
        sslmode: &str,
    ) -> (String, tokio::task::JoinHandle<Option<Vec<u8>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("local_addr");
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept");
            let mut request = [0_u8; 8];
            stream.read_exact(&mut request).await.expect("SSLRequest");
            assert_eq!(request, SSL_REQUEST);
            stream.write_all(b"N").await.expect("ответ N");

            let mut length = [0_u8; 4];
            match timeout(Duration::from_secs(2), stream.read_exact(&mut length)).await {
                Ok(Ok(_)) => {
                    let message_len = u32::from_be_bytes(length) as usize;
                    assert!(message_len >= 8, "слишком короткий StartupMessage");
                    let mut message = vec![0_u8; message_len];
                    message[..4].copy_from_slice(&length);
                    stream
                        .read_exact(&mut message[4..])
                        .await
                        .expect("StartupMessage");
                    Some(message)
                }
                Ok(Err(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof => None,
                Ok(Err(error)) => panic!("чтение StartupMessage: {error}"),
                Err(_) => panic!("клиент не закрыл соединение и не прислал StartupMessage"),
            }
        });
        (
            format!("postgres://user@{address}/agents?sslmode={sslmode}"),
            server,
        )
    }

    #[test]
    fn tls_manager_builds_for_require() {
        PgStore::connect("postgres://user@127.0.0.1:9/agents?sslmode=require", 1)
            .expect("сборка TLS-пула");
    }

    #[tokio::test]
    async fn sslmode_require_rejects_server_without_tls() {
        let (dsn, server) = tls_rejecting_server("require").await;
        let store = PgStore::connect(&dsn, 1).expect("connect");
        let error = match store.pool.get().await {
            Ok(_) => panic!("соединение без TLS принято при sslmode=require"),
            Err(error) => error,
        };
        assert!(error.to_string().to_lowercase().contains("tls"), "{error}");
        assert!(server.await.expect("server task").is_none());
    }

    #[tokio::test]
    async fn sslmode_prefer_falls_back_to_plaintext() {
        let (dsn, server) = tls_rejecting_server("prefer").await;
        let store = PgStore::connect(&dsn, 1).expect("connect");
        assert!(store.pool.get().await.is_err());

        let startup = server
            .await
            .expect("server task")
            .expect("StartupMessage после ответа N");
        assert_eq!(&startup[4..8], &196_608_u32.to_be_bytes());
    }

    /// Живой round-trip против реального PG. Игнорируется по умолчанию.
    /// Запуск (нужна доступная база PostgreSQL со схемой из migrations_pg):
    ///   AGENTS_MCP_TEST_PG_DSN='postgres://user:pass@127.0.0.1:5432/agents' \
    ///   cargo test -- --ignored pg_round_trip
    #[tokio::test]
    #[ignore = "нужен живой PG через AGENTS_MCP_TEST_PG_DSN"]
    async fn pg_round_trip() {
        let dsn =
            std::env::var("AGENTS_MCP_TEST_PG_DSN").expect("AGENTS_MCP_TEST_PG_DSN не задан");
        let store = PgStore::connect(&dsn, 2).expect("connect");
        store.health().await.expect("health");

        let task_id = store
            .create_task(&NewTask {
                goal: Some("rust-smoke".into()),
                task_kind: Some("build_artifact".into()),
                ..Default::default()
            })
            .await
            .expect("create_task");

        let aid = store
            .write_artifact(
                task_id,
                "metadata",
                "meta.x",
                Some("content-x"),
                Some("purpose-x"),
                Some("metadata-analyst"),
                None,
                &[],
            )
            .await
            .expect("write_artifact");
        assert!(aid > 0);

        // upsert по тому же (task_id,key) — id не меняется, content обновляется.
        let aid2 = store
            .write_artifact(
                task_id,
                "metadata",
                "meta.x",
                Some("content-x2"),
                None,
                None,
                None,
                &[aid],
            )
            .await
            .expect("write_artifact upsert");
        assert_eq!(aid, aid2, "upsert по (task_id,key) должен вернуть тот же id");

        let kinds = vec!["metadata".to_string()];
        let arts = store
            .read_artifacts(task_id, Some(kinds.as_slice()))
            .await
            .expect("read_artifacts");
        assert_eq!(arts.len(), 1);
        assert_eq!(arts[0].content.as_deref(), Some("content-x2"));
        assert_eq!(arts[0].depends_on, vec![aid]);

        store
            .append_task_event(
                task_id,
                "agent_done",
                Some("metadata-analyst"),
                None,
                Some("{\"k\":\"v\"}"),
            )
            .await
            .expect("append_task_event");
        store
            .set_task_status(task_id, "completed")
            .await
            .expect("set_task_status");

        let stale_running = store
            .create_task(&NewTask::default())
            .await
            .expect("старая выполняющаяся задача");
        let fresh_artifact_task = store
            .create_task(&NewTask::default())
            .await
            .expect("задача со свежим артефактом");
        store
            .write_artifact(
                fresh_artifact_task,
                "metadata",
                "fresh",
                Some("fresh"),
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
            .expect("завершение задачи со свежим артефактом");

        let task_ids = vec![task_id, stale_running, fresh_artifact_task];
        let client = store.client().await.expect("client");
        client
            .execute(
                "UPDATE agents_mcp.tasks SET updated_at=to_timestamp(0) WHERE id=ANY($1)",
                &[&task_ids],
            )
            .await
            .expect("состаривание задач");
        client
            .execute(
                "UPDATE agents_mcp.task_artifacts \
                 SET created_at=to_timestamp(0), \
                     updated_at=CASE WHEN task_id=$2 THEN now() ELSE to_timestamp(0) END \
                 WHERE task_id=ANY($1)",
                &[&task_ids, &fresh_artifact_task],
            )
            .await
            .expect("состаривание артефактов");
        client
            .execute(
                "UPDATE agents_mcp.task_events SET ts=to_timestamp(0) WHERE task_id=$1",
                &[&task_id],
            )
            .await
            .expect("состаривание событий");

        store.retain(90, 90).await.expect("retention");

        let rows = client
            .query(
                "SELECT id FROM agents_mcp.tasks WHERE id=ANY($1) ORDER BY id",
                &[&task_ids],
            )
            .await
            .expect("проверка оставшихся задач");
        let remaining: Vec<i64> = rows.into_iter().map(|row| row.get(0)).collect();
        assert_eq!(remaining, vec![stale_running, fresh_artifact_task]);
        let child_counts = client
            .query_one(
                "SELECT \
                   (SELECT COUNT(*) FROM agents_mcp.task_artifacts WHERE task_id=$1), \
                   (SELECT COUNT(*) FROM agents_mcp.task_events WHERE task_id=$1)",
                &[&task_id],
            )
            .await
            .expect("проверка каскада");
        assert_eq!(child_counts.get::<_, i64>(0), 0);
        assert_eq!(child_counts.get::<_, i64>(1), 0);

        client
            .execute("DELETE FROM agents_mcp.tasks WHERE id=ANY($1)", &[&task_ids])
            .await
            .expect("очистка тестовых задач");
    }
}
