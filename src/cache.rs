//! Кеш ответов агентов во встроенном SQLite или PostgreSQL.
//!
//! Ключ — sha256 от агента, варианта, фактических провайдера/модели,
//! отпечатков prompt.md и контекста задачи, task_id и input_subset.
//! Подмножество input строится по `config.cache.key_fields`: если пусто —
//! берётся весь input. Если поле из `key_fields` отсутствует в input — оно
//! тихо пропускается (не делает ключ "недействительным" — это позволяет
//! кешировать частичные запросы).
//!
//! TTL — `config.cache.ttl_sec`. expires_at записывается как unixepoch + ttl.
//! Истёкшие записи удаляет фоновый retention через общий `Store::retain`.

use std::collections::BTreeMap;

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::store::Store;

/// Что лежит в кеше: сам ответ + metadata вызова (model, tokens, cost).
/// Тип определён в хранилище — кеш и есть его таблица.
pub use crate::store::CachedEntry;

pub struct CacheKeyParts<'a> {
    pub agent_name: &'a str,
    pub variant: &'a str,
    pub provider_name: &'a str,
    pub model_name: &'a str,
    pub prompt: &'a str,
    pub task_context: &'a str,
    pub input: &'a Map<String, Value>,
    pub key_fields: &'a [String],
    pub task_id: Option<i64>,
}

pub fn compute_key(parts: CacheKeyParts<'_>) -> String {
    let CacheKeyParts {
        agent_name,
        variant,
        provider_name,
        model_name,
        prompt,
        task_context,
        input,
        key_fields,
        task_id,
    } = parts;
    // Сортированное подмножество для детерминированности.
    let subset: BTreeMap<&str, &Value> = if key_fields.is_empty() {
        input.iter().map(|(k, v)| (k.as_str(), v)).collect()
    } else {
        key_fields
            .iter()
            .filter_map(|k| input.get(k).map(|v| (k.as_str(), v)))
            .collect()
    };
    let raw = serde_json::to_string(&subset).unwrap_or_default();
    // Сам prompt и срез доски в ключ не кладём: достаточно их отпечатков.
    // task_id дополнительно разделяет разные задачи с одинаковым контекстом.
    let prompt_hash = sha256_hex(prompt);
    let task_context_hash = sha256_hex(task_context);
    let payload = format!(
        "{agent_name}|{variant}|{provider_name}|{model_name}|{prompt_hash}|{task_context_hash}|{task_id:?}|{raw}"
    );
    sha256_hex(&payload)
}

fn sha256_hex(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    let mut hex = String::with_capacity(digest.len() * 2);
    for b in digest.iter() {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}

/// Найти в кеше живую запись по ключу. `None` если нет или истёк.
pub async fn lookup(store: &dyn Store, key: String) -> anyhow::Result<Option<CachedEntry>> {
    store.cache_lookup(&key).await
}

/// Записать ответ в кеш с TTL (upsert по cache_key).
pub async fn store(
    store: &dyn Store,
    key: String,
    output_json: String,
    metadata_json: String,
    ttl_sec: u64,
) -> anyhow::Result<()> {
    store
        .cache_store(&key, &output_json, &metadata_json, ttl_sec)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(v: Value) -> Map<String, Value> {
        v.as_object().expect("ожидался JSON-объект").clone()
    }

    fn key(
        agent: &str,
        variant: &str,
        input: &Map<String, Value>,
        key_fields: &[String],
        task_id: Option<i64>,
    ) -> String {
        compute_key(CacheKeyParts {
            agent_name: agent,
            variant,
            provider_name: "provider",
            model_name: "model",
            prompt: "prompt",
            task_context: "task context",
            input,
            key_fields,
            task_id,
        })
    }

    #[test]
    fn key_is_order_independent() {
        // serde_json::Map хранит порядок вставки, но compute_key нормализует
        // через BTreeMap → ключ не зависит от порядка полей.
        let a = obj(json!({"brief": "x", "src": "y"}));
        let b = obj(json!({"src": "y", "brief": "x"}));
        assert_eq!(
            key("agent", "default", &a, &[], None),
            key("agent", "default", &b, &[], None)
        );
    }

    #[test]
    fn different_value_different_key() {
        let a = obj(json!({"brief": "x"}));
        let b = obj(json!({"brief": "z"}));
        assert_ne!(
            key("agent", "default", &a, &[], None),
            key("agent", "default", &b, &[], None)
        );
    }

    #[test]
    fn agent_and_variant_affect_key() {
        let input = obj(json!({"brief": "x"}));
        assert_ne!(
            key("agent-a", "default", &input, &[], None),
            key("agent-b", "default", &input, &[], None)
        );
        assert_ne!(
            key("agent", "default", &input, &[], None),
            key("agent", "v2", &input, &[], None)
        );
    }

    #[test]
    fn key_fields_subset_ignores_other_fields() {
        // С key_fields=["brief"] поле "noise" не влияет на ключ.
        let a = obj(json!({"brief": "same", "noise": "1"}));
        let b = obj(json!({"brief": "same", "noise": "2"}));
        let kf = vec!["brief".to_string()];
        assert_eq!(
            key("agent", "default", &a, &kf, None),
            key("agent", "default", &b, &kf, None)
        );
    }

    #[test]
    fn missing_key_field_skipped_not_fatal() {
        // Отсутствующее поле из key_fields пропускается, а не делает ключ
        // «недействительным»: если все key_fields отсутствуют в обоих input —
        // ключи совпадают.
        let with = obj(json!({"brief": "x", "extra": "y"}));
        let without = obj(json!({"brief": "x"}));
        let kf = vec!["nonexistent".to_string()];
        assert_eq!(
            key("agent", "default", &with, &kf, None),
            key("agent", "default", &without, &kf, None)
        );
    }

    #[test]
    fn key_is_hex_sha256() {
        let input = obj(json!({"brief": "x"}));
        let k = key("agent", "default", &input, &[], None);
        assert_eq!(k.len(), 64);
        assert!(k.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn task_id_affects_key() {
        // task_id входит в ключ: один и тот же input при разных task_id даёт
        // разные ключи (контекст под-агента живёт в доске задачи).
        let input = obj(json!({"brief": "x"}));
        assert_ne!(
            key("agent", "default", &input, &[], Some(1)),
            key("agent", "default", &input, &[], Some(2))
        );
        // None (вне задачи) и Some тоже различаются.
        assert_ne!(
            key("agent", "default", &input, &[], None),
            key("agent", "default", &input, &[], Some(1))
        );
    }

    #[test]
    fn execution_identity_affects_key() {
        let input = obj(json!({"brief": "x"}));
        let key_for = |provider_name, model_name, prompt, task_context| {
            compute_key(CacheKeyParts {
                agent_name: "agent",
                variant: "default",
                provider_name,
                model_name,
                prompt,
                task_context,
                input: &input,
                key_fields: &[],
                task_id: None,
            })
        };
        let base = key_for("provider", "model", "prompt", "context");
        for changed in [
            key_for("other", "model", "prompt", "context"),
            key_for("provider", "other", "prompt", "context"),
            key_for("provider", "model", "other", "context"),
            key_for("provider", "model", "prompt", "other"),
        ] {
            assert_ne!(base, changed);
        }
    }

    /// Живой round-trip против реального PG. Игнорируется по умолчанию.
    /// Запуск (нужна доступная база PostgreSQL со схемой из migrations_pg):
    ///   AGENTS_MCP_TEST_PG_DSN='postgres://user:pass@127.0.0.1:5432/agents' \
    ///   cargo test -- --ignored cache_pg_round_trip
    #[tokio::test]
    #[ignore]
    async fn cache_pg_round_trip() {
        let dsn = std::env::var("AGENTS_MCP_TEST_PG_DSN").expect("AGENTS_MCP_TEST_PG_DSN не задан");
        let st = crate::store::PgStore::connect(&dsn, 2).expect("connect");
        let key = format!(
            "test-{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        );

        // Пусто — None.
        assert!(lookup(&st, key.clone()).await.unwrap().is_none());

        // Записали — нашли.
        store(
            &st,
            key.clone(),
            "{\"r\":1}".into(),
            "{\"m\":2}".into(),
            3600,
        )
        .await
        .unwrap();
        let entry = lookup(&st, key.clone())
            .await
            .unwrap()
            .expect("запись должна быть");
        assert_eq!(entry.output_json, "{\"r\":1}");
        assert_eq!(entry.metadata_json, "{\"m\":2}");

        // ttl=0 → expires_at = now, условие expires_at > now ложно сразу.
        store(&st, key.clone(), "{\"r\":1}".into(), "{\"m\":2}".into(), 0)
            .await
            .unwrap();
        assert!(lookup(&st, key.clone()).await.unwrap().is_none());
    }
}
