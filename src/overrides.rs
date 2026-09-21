//! Перекрытия настроек агента параметрами вызова (`overrides`).
//!
//! Список ключей закрытый: любой не названный здесь — отказ вызова. Значения
//! действуют на ОДИН вызов и в шаблон промпта не попадают. Старшинство:
//! `force_override` (главный конфиг) старше `overrides`, а те старше
//! per-agent `config.toml`.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

/// Разрешённые ключи перекрытий (закрытый список). Любой другой — отказ с
/// перечнем этих ключей в тексте.
pub const ALLOWED_KEYS: &[&str] = &[
    "model.name",
    "model.temperature",
    "model.max_tokens",
    "execution.max_turns",
    "limits.timeout_sec",
    "effort",
    "cache.enabled",
    "mcp.<сервер>.url",
];

/// Перекрытия одного вызова. Пустой набор — вызов идёт как прежде.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CallOverrides {
    pub model_name: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub max_turns: Option<u32>,
    pub timeout_sec: Option<u64>,
    pub effort: Option<String>,
    pub cache_enabled: Option<bool>,
    /// Адреса MCP-серверов агента, названных по имени в ключе `mcp.<сервер>.url`.
    pub mcp_urls: BTreeMap<String, String>,
}

impl CallOverrides {
    /// Разобрать плоский объект перекрытий. `mcp_servers` — имена серверов,
    /// объявленных агентом; `allowed_urls` — список разрешённых адресов из
    /// `[agents] allowed_mcp_urls` (пустой → действует правило по умолчанию).
    pub fn parse(
        raw: &Map<String, Value>,
        mcp_servers: &[String],
        allowed_urls: &[String],
    ) -> Result<Self, String> {
        let mut out = CallOverrides::default();
        for (key, value) in raw {
            if let Some(server) = server_of_mcp_key(key) {
                let url = string_value(key, value)?;
                if !mcp_servers.iter().any(|name| same_server(name, server)) {
                    return Err(format!(
                        "перекрытие '{key}': сервер '{}' не объявлен агентом (объявлены: {})",
                        server,
                        if mcp_servers.is_empty() {
                            "нет".to_string()
                        } else {
                            mcp_servers.join(", ")
                        }
                    ));
                }
                if !url_allowed(&url, allowed_urls) {
                    return Err(format!(
                        "перекрытие '{key}': адрес '{url}' не разрешён (список [agents] allowed_mcp_urls)"
                    ));
                }
                out.mcp_urls.insert(server.to_string(), url);
                continue;
            }
            match key.as_str() {
                "model.name" => out.model_name = Some(string_value(key, value)?),
                "model.temperature" => {
                    let v = number_value(key, value)?;
                    if !(0.0..=2.0).contains(&v) {
                        return Err(format!(
                            "перекрытие '{key}': значение {v} вне допустимого диапазона 0…2"
                        ));
                    }
                    out.temperature = Some(v);
                }
                "model.max_tokens" => out.max_tokens = Some(positive_u32(key, value)?),
                "execution.max_turns" => out.max_turns = Some(positive_u32(key, value)?),
                "limits.timeout_sec" => out.timeout_sec = Some(positive_u64(key, value)?),
                "cache.enabled" => {
                    out.cache_enabled = Some(value.as_bool().ok_or_else(|| {
                        format!("перекрытие '{key}': ожидалось логическое значение")
                    })?);
                }
                "effort" => out.effort = Some(string_value(key, value)?),
                other => return Err(forbidden_key(other)),
            }
        }
        Ok(out)
    }

    /// Все поля пусты — перекрытий нет.
    pub fn is_empty(&self) -> bool {
        self.model_name.is_none()
            && self.temperature.is_none()
            && self.max_tokens.is_none()
            && self.max_turns.is_none()
            && self.timeout_sec.is_none()
            && self.effort.is_none()
            && self.cache_enabled.is_none()
            && self.mcp_urls.is_empty()
    }

    /// Нормализованный вид перекрытий (ключи отсортированы) — для ключа кэша,
    /// журнала и metadata. Пустой набор → `None`.
    pub fn canonical_json(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }
        let mut obj: BTreeMap<String, Value> = BTreeMap::new();
        if let Some(v) = &self.model_name {
            obj.insert("model.name".to_string(), Value::String(v.clone()));
        }
        if let Some(v) = self.temperature {
            obj.insert("model.temperature".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.max_tokens {
            obj.insert("model.max_tokens".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.max_turns {
            obj.insert("execution.max_turns".to_string(), serde_json::json!(v));
        }
        if let Some(v) = self.timeout_sec {
            obj.insert("limits.timeout_sec".to_string(), serde_json::json!(v));
        }
        if let Some(v) = &self.effort {
            obj.insert("effort".to_string(), Value::String(v.clone()));
        }
        if let Some(v) = self.cache_enabled {
            obj.insert("cache.enabled".to_string(), Value::Bool(v));
        }
        for (server, url) in &self.mcp_urls {
            obj.insert(format!("mcp.{server}.url"), Value::String(url.clone()));
        }
        serde_json::to_string(&obj).ok()
    }
}

/// Адрес разрешён: список непуст — точное совпадение; список пуст — умолчание:
/// схема `http`, хост `127.0.0.1`, путь `/mcp`, порт задан.
pub fn url_allowed(url: &str, allowed: &[String]) -> bool {
    if !allowed.is_empty() {
        return allowed.iter().any(|item| item == url);
    }
    let Some(rest) = url.strip_prefix("http://127.0.0.1:") else {
        return false;
    };
    let Some((port, path)) = rest.split_once('/') else {
        return false;
    };
    path == "mcp" && !port.is_empty() && port.parse::<u16>().is_ok()
}

/// Записать в аргументы CLI пару «флаг + значение»: уже стоящая пара заменяется
/// на месте, отсутствующая — добавляется в конец. Общая логика веток claude-cli
/// (`--effort`) и grok-cli (`--reasoning-effort`).
fn set_pair_arg(extra_args: &mut Vec<String>, flag: &str, value: &str) {
    if let Some(pos) = extra_args.iter().position(|arg| arg == flag) {
        if pos + 1 < extra_args.len() {
            extra_args[pos + 1] = value.to_string();
        } else {
            extra_args.push(value.to_string());
        }
    } else {
        extra_args.push(flag.to_string());
        extra_args.push(value.to_string());
    }
}

/// Записать усилие рассуждений в аргументы CLI. `true` — усилие ушло в
/// аргументы; `false` — провайдер не CLI, значение кладёт вызывающий (в
/// `extra_body`).
pub fn set_effort_arg(provider: &str, extra_args: &mut Vec<String>, effort: &str) -> bool {
    match provider {
        // claude-cli: пара `--effort <значение>`.
        "claude-cli" => {
            set_pair_arg(extra_args, "--effort", effort);
            true
        }
        // grok-cli: пара `--reasoning-effort <значение>` — логика та же.
        "grok-cli" => {
            set_pair_arg(extra_args, "--reasoning-effort", effort);
            true
        }
        // codex-cli: `-c model_reasoning_effort="<значение>"`.
        "codex-cli" => {
            let replacement = format!("model_reasoning_effort=\"{effort}\"");
            if let Some(pos) = extra_args
                .iter()
                .position(|arg| arg.starts_with("model_reasoning_effort="))
            {
                extra_args[pos] = replacement;
            } else {
                extra_args.push("-c".to_string());
                extra_args.push(replacement);
            }
            true
        }
        _ => false,
    }
}

/// Записать адрес MCP-сервера в аргументы codex-cli:
/// `-c mcp_servers.<имя>.url="<адрес>"`. Имя сервера — с `-` → `_` (codex-cli
/// объявляет серверы с подчёркиваниями). Уже заданный агентом ключ заменяется,
/// а не дублируется.
pub fn set_mcp_url_args(extra_args: &mut Vec<String>, server: &str, url: &str) {
    let name = server.replace('-', "_");
    let prefix = format!("mcp_servers.{name}.url=");
    let replacement = format!("mcp_servers.{name}.url=\"{url}\"");
    if let Some(pos) = extra_args.iter().position(|arg| arg.starts_with(&prefix)) {
        extra_args[pos] = replacement;
    } else {
        extra_args.push("-c".to_string());
        extra_args.push(replacement);
    }
}

/// Имена серверов, объявленных агентом в `extra_args` (ключи
/// `mcp_servers.<имя>.url=...`). Порядок первого появления, без повторов.
pub fn mcp_servers_of_extra_args(extra_args: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for arg in extra_args {
        let Some(rest) = arg.trim_start().strip_prefix("mcp_servers.") else {
            continue;
        };
        let Some((name, setting)) = rest.split_once('.') else {
            continue;
        };
        if name.is_empty() || !setting.starts_with("url=") {
            continue;
        }
        let name = name.to_string();
        if !out.contains(&name) {
            out.push(name);
        }
    }
    out
}

/// Имя сервера из ключа `mcp.<сервер>.url`. `None` — ключ не этого вида.
fn server_of_mcp_key(key: &str) -> Option<&str> {
    let rest = key.strip_prefix("mcp.")?.strip_suffix(".url")?;
    (!rest.is_empty()).then_some(rest)
}

/// Имена серверов совпадают как есть либо после замены `-` на `_`: codex-cli
/// объявляет серверы с подчёркиваниями, а ключ перекрытия пишут с дефисом.
fn same_server(declared: &str, wanted: &str) -> bool {
    declared == wanted || declared.replace('-', "_") == wanted.replace('-', "_")
}

fn string_value(key: &str, value: &Value) -> Result<String, String> {
    match value.as_str() {
        Some(s) if !s.trim().is_empty() => Ok(s.to_string()),
        Some(_) => Err(format!("перекрытие '{key}': строка не должна быть пустой")),
        None => Err(format!("перекрытие '{key}': ожидалась строка")),
    }
}

fn number_value(key: &str, value: &Value) -> Result<f32, String> {
    value
        .as_f64()
        .map(|v| v as f32)
        .ok_or_else(|| format!("перекрытие '{key}': ожидалось число"))
}

fn positive_u32(key: &str, value: &Value) -> Result<u32, String> {
    let v = value
        .as_i64()
        .ok_or_else(|| format!("перекрытие '{key}': ожидалось целое число"))?;
    if v <= 0 {
        return Err(format!("перекрытие '{key}': значение должно быть > 0"));
    }
    u32::try_from(v).map_err(|_| format!("перекрытие '{key}': значение {v} слишком велико"))
}

fn positive_u64(key: &str, value: &Value) -> Result<u64, String> {
    let v = value
        .as_i64()
        .ok_or_else(|| format!("перекрытие '{key}': ожидалось целое число"))?;
    if v <= 0 {
        return Err(format!("перекрытие '{key}': значение должно быть > 0"));
    }
    Ok(v as u64)
}

fn forbidden_key(key: &str) -> String {
    format!(
        "ключ перекрытия '{key}' не разрешён; допустимы: {}",
        ALLOWED_KEYS.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn map(v: Value) -> Map<String, Value> {
        v.as_object().expect("ожидался объект").clone()
    }

    fn servers() -> Vec<String> {
        vec!["code_index".to_string()]
    }

    #[test]
    fn parses_every_allowed_key() {
        let raw = map(json!({
            "model.name": "m-1",
            "model.temperature": 0.2,
            "model.max_tokens": 1000,
            "execution.max_turns": 4,
            "limits.timeout_sec": 30,
            "effort": "high",
            "cache.enabled": true,
            "mcp.code-index.url": "http://127.0.0.1:8011/mcp"
        }));
        let ov = CallOverrides::parse(&raw, &servers(), &[]).expect("разбор");
        assert_eq!(ov.model_name.as_deref(), Some("m-1"));
        assert_eq!(ov.temperature, Some(0.2));
        assert_eq!(ov.max_tokens, Some(1000));
        assert_eq!(ov.max_turns, Some(4));
        assert_eq!(ov.timeout_sec, Some(30));
        assert_eq!(ov.effort.as_deref(), Some("high"));
        assert_eq!(ov.cache_enabled, Some(true));
        assert_eq!(
            ov.mcp_urls.get("code-index").map(String::as_str),
            Some("http://127.0.0.1:8011/mcp")
        );
    }

    #[test]
    fn forbidden_key_names_allowed_keys() {
        for key in [
            "model.provider",
            "execution.allowed_tools",
            "response.format",
            "cwd_template",
        ] {
            let raw = map(json!({ key: "x" }));
            let err = CallOverrides::parse(&raw, &servers(), &[]).expect_err("ключ запрещён");
            assert!(err.contains(key), "текст называет ключ: {err}");
            assert!(
                err.contains("model.name") && err.contains("mcp.<сервер>.url"),
                "текст перечисляет разрешённые ключи: {err}"
            );
        }
    }

    #[test]
    fn rejects_wrong_types() {
        for raw in [
            json!({"model.name": ""}),
            json!({"model.name": 5}),
            json!({"model.temperature": 3}),
            json!({"model.max_tokens": 0}),
            json!({"execution.max_turns": -1}),
            json!({"limits.timeout_sec": "60"}),
            json!({"cache.enabled": "yes"}),
            json!({"effort": ""}),
        ] {
            assert!(
                CallOverrides::parse(&map(raw.clone()), &servers(), &[]).is_err(),
                "должен быть отказ: {raw}"
            );
        }
    }

    #[test]
    fn rejects_unknown_server_and_disallowed_url() {
        let unknown = map(json!({"mcp.other.url": "http://127.0.0.1:8011/mcp"}));
        assert!(CallOverrides::parse(&unknown, &servers(), &[]).is_err());

        let allowed = vec!["http://127.0.0.1:8037/mcp".to_string()];
        let wrong = map(json!({"mcp.code_index.url": "http://127.0.0.1:8011/mcp"}));
        assert!(CallOverrides::parse(&wrong, &servers(), &allowed).is_err());
        let right = map(json!({"mcp.code_index.url": "http://127.0.0.1:8037/mcp"}));
        assert!(CallOverrides::parse(&right, &servers(), &allowed).is_ok());
    }

    #[test]
    fn server_name_matches_with_dash_and_underscore() {
        let raw = map(json!({"mcp.code-index.url": "http://127.0.0.1:8011/mcp"}));
        assert!(CallOverrides::parse(&raw, &servers(), &[]).is_ok());
    }

    #[test]
    fn default_url_rule_requires_loopback_http_mcp_with_port() {
        assert!(url_allowed("http://127.0.0.1:8011/mcp", &[]));
        assert!(url_allowed("http://127.0.0.1:80/mcp", &[]));
        assert!(!url_allowed("http://127.0.0.1/mcp", &[]));
        assert!(!url_allowed("http://10.0.0.5:8011/mcp", &[]));
        assert!(!url_allowed("https://127.0.0.1:8011/mcp", &[]));
        assert!(!url_allowed("http://127.0.0.1:8011/other", &[]));
        assert!(!url_allowed("http://user@127.0.0.1:8011/mcp", &[]));
        assert!(!url_allowed("http://127.0.0.1:8011/mcp?x=1", &[]));
    }

    #[test]
    fn canonical_json_is_sorted_and_empty_is_none() {
        assert!(CallOverrides::default().canonical_json().is_none());
        let raw = map(json!({
            "limits.timeout_sec": 5,
            "model.name": "z",
            "mcp.code-index.url": "http://127.0.0.1:8011/mcp"
        }));
        let ov = CallOverrides::parse(&raw, &servers(), &[]).unwrap();
        let json = ov.canonical_json().expect("непустой");
        assert_eq!(
            json,
            "{\"limits.timeout_sec\":5,\"mcp.code-index.url\":\"http://127.0.0.1:8011/mcp\",\"model.name\":\"z\"}"
        );
    }

    #[test]
    fn effort_arg_replaces_not_duplicates() {
        let mut args = vec!["--effort".to_string(), "low".to_string()];
        assert!(set_effort_arg("claude-cli", &mut args, "high"));
        assert_eq!(args, vec!["--effort".to_string(), "high".to_string()]);

        let mut args = vec!["--allowed-tools".to_string(), "Read".to_string()];
        assert!(set_effort_arg("claude-cli", &mut args, "high"));
        assert_eq!(
            args,
            vec![
                "--allowed-tools".to_string(),
                "Read".to_string(),
                "--effort".to_string(),
                "high".to_string()
            ]
        );

        let mut args = vec![
            "-c".to_string(),
            "model_reasoning_effort=\"low\"".to_string(),
        ];
        assert!(set_effort_arg("codex-cli", &mut args, "high"));
        assert_eq!(
            args,
            vec![
                "-c".to_string(),
                "model_reasoning_effort=\"high\"".to_string()
            ]
        );

        let mut args: Vec<String> = Vec::new();
        assert!(!set_effort_arg("openrouter", &mut args, "high"));
        assert!(args.is_empty());
    }

    #[test]
    fn mcp_url_arg_replaces_not_duplicates() {
        let mut args = vec![
            "-c".to_string(),
            "mcp_servers.code_index.url=\"http://127.0.0.1:8011/mcp\"".to_string(),
        ];
        set_mcp_url_args(&mut args, "code-index", "http://127.0.0.1:8037/mcp");
        assert_eq!(args.len(), 2);
        assert_eq!(
            args[1],
            "mcp_servers.code_index.url=\"http://127.0.0.1:8037/mcp\""
        );

        let mut args: Vec<String> = Vec::new();
        set_mcp_url_args(&mut args, "code-index", "http://127.0.0.1:8037/mcp");
        assert_eq!(
            args,
            vec![
                "-c".to_string(),
                "mcp_servers.code_index.url=\"http://127.0.0.1:8037/mcp\"".to_string()
            ]
        );
    }

    #[test]
    fn mcp_servers_of_extra_args_reads_names() {
        let args = vec![
            "-c".to_string(),
            "mcp_servers.code_index.url=\"http://127.0.0.1:8011/mcp\"".to_string(),
            "mcp_servers.rag.url=\"http://127.0.0.1:8019/mcp\"".to_string(),
            "model_reasoning_effort=\"low\"".to_string(),
        ];
        assert_eq!(
            mcp_servers_of_extra_args(&args),
            vec!["code_index".to_string(), "rag".to_string()]
        );
        assert!(mcp_servers_of_extra_args(&[]).is_empty());
    }
}
