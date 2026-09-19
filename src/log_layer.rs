//! Custom tracing Layer, пишущий события в отдельный файл SQLite журнала
//! `<[storage] log_dir>/agents-mcp.logs.db` (его открывает main.rs).
//!
//! Журнал живёт в собственном файле и не зависит от основного хранилища
//! (в работе это PostgreSQL): отказ основной базы записи журнала не теряет.
//!
//! Архитектура:
//!   * Layer (синхронный) кладёт каждое событие в unbounded mpsc.
//!   * Writer task (отдельный tokio-таск) батчит INSERT'ы пачками до 100 шт.
//!     или раз в 500 мс, чтобы не молотить БД на каждом log!().
//!   * На graceful shutdown main посылает writer'у команду, ждёт финальный
//!     flush с ограничением по времени и только затем завершает процесс.
//!   * Если запись батча упала — батч теряется (eprintln), канал не копится
//!     бесконечно: писать в свой же журнал из writer'а нельзя (рекурсия).
//!
//! Structured fields (`info!(error = %e, port = 8025, ...)`) сериализуются в
//! JSON и сжимаются zstd → колонка `context BYTEA`. ~10x компрессия на
//! повторяющемся JSON.

use std::sync::Arc;
use std::time::Duration;

use serde_json::{Map, Value};
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::sync::oneshot;
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::{layer::Context as LayerContext, registry::LookupSpan, Layer};

use crate::store::{LogEvent, Store};

/// События одной записи перед записью в БД.
struct EventRecord {
    ts_millis: i64,
    level: u8,
    target: String,
    message: String,
    fields: Option<Value>,
}

/// Layer, кладущий события в unbounded канал.
pub struct LogLayer {
    tx: UnboundedSender<WriterMessage>,
}

enum WriterMessage {
    Event(EventRecord),
    Shutdown(oneshot::Sender<bool>),
}

pub struct LogWriterHandle {
    tx: UnboundedSender<WriterMessage>,
    task: tokio::task::JoinHandle<()>,
}

impl LogLayer {
    pub fn install(store: Arc<dyn Store>) -> (Self, LogWriterHandle) {
        let (tx, rx) = mpsc::unbounded_channel::<WriterMessage>();
        let task = tokio::spawn(writer_task(store, rx));
        (Self { tx: tx.clone() }, LogWriterHandle { tx, task })
    }
}

impl LogWriterHandle {
    /// Записать все события, поставленные в очередь до этой команды, и
    /// дождаться завершения writer'а не дольше заданного срока.
    pub async fn shutdown(mut self, timeout: Duration) -> bool {
        let started = tokio::time::Instant::now();
        let (ack_tx, ack_rx) = oneshot::channel();
        if self.tx.send(WriterMessage::Shutdown(ack_tx)).is_err() {
            let finished = matches!(
                tokio::time::timeout(timeout, &mut self.task).await,
                Ok(Ok(()))
            );
            if !finished {
                self.task.abort();
            }
            return finished;
        }
        match tokio::time::timeout(timeout, ack_rx).await {
            Ok(Ok(true)) => {}
            Ok(Ok(false)) => {
                eprintln!("event-writer: финальный flush завершился ошибкой");
                let _ =
                    tokio::time::timeout(timeout.saturating_sub(started.elapsed()), &mut self.task)
                        .await;
                return false;
            }
            Ok(Err(_)) => {
                eprintln!("event-writer: канал подтверждения shutdown закрыт");
                self.task.abort();
                return false;
            }
            Err(_) => {
                eprintln!("event-writer: финальный flush не завершился за {timeout:?}");
                self.task.abort();
                return false;
            }
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        if !matches!(
            tokio::time::timeout(remaining, &mut self.task).await,
            Ok(Ok(()))
        ) {
            eprintln!("event-writer: задача не завершилась после финального flush");
            self.task.abort();
            return false;
        }
        true
    }
}

impl<S> Layer<S> for LogLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: LayerContext<'_, S>) {
        let mut visitor = FieldVisitor {
            message: None,
            fields: Map::new(),
        };
        event.record(&mut visitor);

        let lvl = match *event.metadata().level() {
            Level::ERROR => 4,
            Level::WARN => 3,
            Level::INFO => 2,
            Level::DEBUG => 1,
            Level::TRACE => 0,
        };

        let rec = EventRecord {
            ts_millis: chrono::Utc::now().timestamp_millis(),
            level: lvl,
            target: event.metadata().target().to_string(),
            message: visitor.message.unwrap_or_default(),
            fields: if visitor.fields.is_empty() {
                None
            } else {
                Some(Value::Object(visitor.fields))
            },
        };

        // Не блокируем тред логгера — если канал закрыт, событие теряется.
        // Это OK: канал закрывается только на shutdown.
        let _ = self.tx.send(WriterMessage::Event(rec));
    }
}

/// Парсер полей события в JSON Map. Поле `message` обрабатывается отдельно.
struct FieldVisitor {
    message: Option<String>,
    fields: Map<String, Value>,
}

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let s = format!("{:?}", value);
        if field.name() == "message" {
            self.message = Some(s);
        } else {
            self.fields
                .insert(field.name().to_string(), Value::String(s));
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        } else {
            self.fields
                .insert(field.name().to_string(), Value::String(value.to_string()));
        }
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.fields
            .insert(field.name().to_string(), Value::Number(value.into()));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.fields
            .insert(field.name().to_string(), Value::Number(value.into()));
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.fields
            .insert(field.name().to_string(), Value::Bool(value));
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        let n = serde_json::Number::from_f64(value).unwrap_or_else(|| 0i64.into());
        self.fields
            .insert(field.name().to_string(), Value::Number(n));
    }
}

async fn writer_task(store: Arc<dyn Store>, mut rx: mpsc::UnboundedReceiver<WriterMessage>) {
    const BATCH_SIZE: usize = 100;
    const FLUSH_INTERVAL_MS: u64 = 500;

    let mut batch: Vec<EventRecord> = Vec::with_capacity(BATCH_SIZE);
    let mut tick = tokio::time::interval(Duration::from_millis(FLUSH_INTERVAL_MS));
    tick.tick().await; // первый tick срабатывает сразу — пропускаем

    loop {
        tokio::select! {
            maybe_rec = rx.recv() => {
                match maybe_rec {
                    Some(WriterMessage::Event(r)) => {
                        batch.push(r);
                        if batch.len() >= BATCH_SIZE {
                            flush(&store, &mut batch).await;
                        }
                    }
                    Some(WriterMessage::Shutdown(ack)) => {
                        let flushed = flush(&store, &mut batch).await;
                        let _ = ack.send(flushed);
                        break;
                    }
                    None => {
                        // Writer потерял всех отправителей — дописываем хвост.
                        flush(&store, &mut batch).await;
                        break;
                    }
                }
            }
            _ = tick.tick() => {
                if !batch.is_empty() {
                    flush(&store, &mut batch).await;
                }
            }
        }
    }
}

async fn flush(store: &Arc<dyn Store>, batch: &mut Vec<EventRecord>) -> bool {
    if batch.is_empty() {
        return true;
    }
    let records: Vec<EventRecord> = std::mem::take(batch);

    // Поля события сериализуются в JSON и сжимаются zstd ЗДЕСЬ: хранилищу
    // достаётся уже готовое к INSERT значение (колонка context BYTEA).
    let events: anyhow::Result<Vec<LogEvent>> = records
        .iter()
        .map(|r| {
            let context_zstd: Option<Vec<u8>> = match &r.fields {
                Some(v) => {
                    let json = serde_json::to_vec(v)?;
                    Some(zstd::encode_all(json.as_slice(), 3)?)
                }
                None => None,
            };
            Ok(LogEvent {
                ts_millis: r.ts_millis,
                level: r.level as i16,
                target: r.target.clone(),
                message: r.message.clone(),
                context_zstd,
            })
        })
        .collect();

    let res = match events {
        Ok(events) => store.write_events(&events).await,
        Err(e) => Err(e),
    };

    // Логировать ошибку через eprintln, чтобы не зациклиться (writer не пишет
    // в свой же журнал — иначе бесконечная рекурсия). Батч при ошибке теряется.
    match res {
        Ok(()) => true,
        Err(e) => {
            eprintln!("event-writer: запись журнала упала: {e}");
            false
        }
    }
}

// Заглушка чтобы Arc-обёртка не понадобилась (Layer должен быть Sync+Send).
// UnboundedSender уже Sync+Send.
static _ASSERT_SEND_SYNC: fn() -> Arc<()> = || Arc::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::prelude::*;

    #[tokio::test]
    async fn shutdown_flushes_last_event_to_sqlite() {
        let dir =
            std::env::temp_dir().join(format!("agents-mcp-log-shutdown-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("logs.db");
        let store: Arc<dyn Store> = Arc::new(crate::store::SqliteStore::open(&db_path).unwrap());
        let (layer, writer) = LogLayer::install(store.clone());
        let subscriber = tracing_subscriber::registry().with(layer);

        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(target: "p21b_test", "последнее событие перед остановкой");
        });
        assert!(writer.shutdown(Duration::from_secs(2)).await);
        drop(store);

        let connection = rusqlite::Connection::open(&db_path).unwrap();
        let count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM events WHERE target = ?1 AND message = ?2",
                rusqlite::params!["p21b_test", "последнее событие перед остановкой"],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
        drop(connection);
        let _ = std::fs::remove_dir_all(dir);
    }
}
