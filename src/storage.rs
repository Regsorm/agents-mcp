//! Фоновая чистка старых строк хранилища — основного и файла журнала.
//!
//! Этот модуль держит только периодическую чистку старых строк; сама чистка —
//! в [`crate::store::Store::retain`].

use std::sync::Arc;
use std::time::Duration;

use tracing::{info, warn};

use crate::store::Store;

/// Раз в сутки чистит старые записи (завершённые задачи и вызовы старше
/// `agent_calls_days`, логи старше `events_days`, истёкший кеш). Первый проход
/// — через 60 сек после старта, чтобы не конкурировать с инициализацией.
/// Запускать через `tokio::spawn`; крутится до отмены задачи.
pub async fn retention_loop(store: Arc<dyn Store>, agent_calls_days: i64, events_days: i64) {
    tokio::time::sleep(Duration::from_secs(60)).await;
    let mut interval = tokio::time::interval(Duration::from_secs(24 * 3600));
    interval.tick().await; // первый tick идёт сразу — пропускаем

    loop {
        match store.retain(agent_calls_days, events_days).await {
            Ok((calls, events, cache)) => {
                if calls + events + cache > 0 {
                    info!(
                        deleted_calls = calls,
                        deleted_events = events,
                        deleted_cache = cache,
                        "retention pass завершён"
                    );
                }
            }
            Err(e) => warn!(error = %e, "retention pass упал"),
        }
        interval.tick().await;
    }
}
