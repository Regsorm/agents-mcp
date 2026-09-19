//! Провайдер claude-cli: вызов локального `claude -p --output-format json`
//! через tokio::process::Command.
//!
//! Авторизация — через локальный OAuth-логин Claude Code (тот же что
//! используется десктоп-приложением). API-ключ не нужен.
//!
//! Конкурентность ограничена `tokio::sync::Semaphore` (см. ClaudeCliConfig
//! `max_concurrent`). `--max-turns` обязателен в каждом вызове — защищает от
//! зависания в tool-loop'е.
//!
//! Бюджет-чекер по решению пользователя не реализован: расход пишется в
//! `agent_calls.cost_usd` для ретроспективной аналитики, но не блокирует вызовы.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

use crate::proc_tree;

use super::{LlmError, LlmProvider, LlmRequest, LlmResponse};

/// Путь к `.credentials.json` в реальном (авторизованном) ~/.claude. Основной
/// клиент Claude Code периодически делает silent OAuth-refresh и пишет новый
/// accessToken ТОЛЬКО сюда.
fn real_claude_credentials() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .map(|h| h.join(".claude").join(".credentials.json"))
}

/// Копирует свежий OAuth-токен из реального ~/.claude в fake-home перед каждым
/// вызовом claude-cli. Без этого токен в fake-home протухает при silent refresh
/// основного клиента → субагент падает с 401 (инциденты 27.05 и 02.06). Файл
/// ~0.5 КБ, overhead копии на каждом вызове незаметен. Ошибку не делаем
/// фатальной: если токена нет совсем — пусть claude сам вернёт понятный 401.
/// Долгоживущий токен (`claude setup-token`, живёт год): из окружения, иначе из
/// файла. Это ОСНОВНОЙ способ входа под-агента.
///
/// Файл берётся только из переменной CLAUDE_OAUTH_TOKEN_FILE — пути по
/// умолчанию нет. Зачем файл, а не только переменная с токеном: переменная
/// окружения службы теряется при переносе или ручном запуске, а файл вне
/// репозиториев одинаково доступен и службе, и человеку.
///
/// Токен отвечает ТОЛЬКО за вход. «403 Request not allowed» к токену отношения
/// не имеет: его даёт fake-home БЕЗ настроек выхода в сеть — даже с заведомо
/// рабочими учётными данными, скопированными из настоящего ~/.claude. Если
/// прямое обращение к api.anthropic.com закрыто сетью, настоящий дом проходит
/// благодаря HTTP_PROXY/HTTPS_PROXY в своём `settings.json`; тогда те же
/// переменные прописываются в `settings.json` каталога fake-home (config_dir) —
/// claude применяет их сам.
/// Увидел 403 снова — проверяй сетевой путь, а не токен: подлинность токена
/// проверяется запросом к API напрямую (негодный даёт 401, а не 403).
fn long_lived_token() -> Option<String> {
    if let Ok(tok) = std::env::var("CLAUDE_CODE_OAUTH_TOKEN") {
        if !tok.trim().is_empty() {
            return Some(tok.trim().to_string());
        }
    }
    let path = std::env::var("CLAUDE_OAUTH_TOKEN_FILE").ok()?;
    match std::fs::read_to_string(&path) {
        Ok(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        _ => None,
    }
}

fn sync_credentials_into(config_dir: &std::path::Path) {
    let Some(src) = real_claude_credentials() else {
        return;
    };
    if !src.exists() {
        return;
    }
    let dst = config_dir.join(".credentials.json");
    if let Err(e) = std::fs::copy(&src, &dst) {
        warn!(
            error = %e,
            src = %src.display(),
            "не удалось синхронизировать .credentials.json в fake-home — субагент может упасть с 401"
        );
    }
}

pub struct ClaudeCliProvider {
    executable: PathBuf,
    semaphore: Arc<Semaphore>,
    default_max_turns: u32,
    /// Каталог fake-home (CLAUDE_CONFIG_DIR) для под-агентов. None — fallback
    /// на env родителя в complete().
    config_dir: Option<PathBuf>,
}

impl ClaudeCliProvider {
    pub fn new(
        executable: PathBuf,
        max_concurrent: u32,
        default_max_turns: u32,
        config_dir: Option<PathBuf>,
    ) -> Self {
        // Fail-fast при старте: если fake-home задан, но каталога или
        // .credentials.json нет — под-агенты молча начнут тащить полный
        // ~/.claude (раздув промпта, +~350k токенов, инцидент 2026-05-31).
        // Громкий WARN делает деградацию видимой ещё на старте сервиса.
        match &config_dir {
            Some(dir) if !dir.is_dir() => warn!(
                config_dir = %dir.display(),
                "claude-cli config_dir не существует — под-агенты будут читать полный ~/.claude (раздув промпта, рост стоимости)"
            ),
            Some(dir) if !dir.join(".credentials.json").exists() => warn!(
                config_dir = %dir.display(),
                "claude-cli config_dir без .credentials.json — под-агенты упадут на авторизации (401)"
            ),
            Some(dir) => {
                info!(config_dir = %dir.display(), "claude-cli fake-home подключён (CLAUDE_CONFIG_DIR)")
            }
            None => warn!(
                "claude-cli config_dir не задан в конфиге — fallback на env CLAUDE_CONFIG_DIR (легко потерять при переносе)"
            ),
        }
        let permits = max_concurrent.max(1) as usize;
        Self {
            executable,
            semaphore: Arc::new(Semaphore::new(permits)),
            default_max_turns,
            config_dir,
        }
    }

    /// Doctor self-test: `claude --version`. Возвращает строку версии либо
    /// ошибку. Вызывается из main.rs при старте сервиса для health-report.
    pub async fn doctor(executable: &PathBuf) -> Result<String, String> {
        let output = match tokio::time::timeout(
            Duration::from_secs(15),
            Command::new(executable)
                .arg("--version")
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true)
                .output(),
        )
        .await
        {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => {
                return Err(format!(
                    "не удалось запустить '{}': {e}",
                    executable.display()
                ))
            }
            Err(_) => {
                return Err(format!(
                    "'{} --version' не завершился за 15 с",
                    executable.display()
                ))
            }
        };
        if !output.status.success() {
            return Err(format!(
                "'{} --version' rc={:?}, stderr={}",
                executable.display(),
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

// ── JSON-схема ответа `claude -p --output-format json` ─────────────────────

#[derive(Debug, Deserialize)]
struct CliResult {
    /// "result" — основной итоговый текст ассистента.
    #[serde(default)]
    result: String,
    #[serde(default)]
    is_error: bool,
    #[serde(default, alias = "duration_ms")]
    duration_ms: Option<u64>,
    #[serde(default, alias = "cost_usd", alias = "total_cost_usd")]
    cost_usd: Option<f64>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    usage: Option<Usage>,
    /// Поле может содержать другие ключи (num_turns, type, subtype, ...) —
    /// они нам не нужны, попадут под flatten отброса.
    #[serde(default, flatten)]
    _other: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize, Default)]
struct Usage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
}

fn configure_command(
    cmd: &mut Command,
    model: &str,
    hints: &super::ClaudeCliHints,
    default_max_turns: u32,
) -> u32 {
    let max_turns = hints.max_turns.unwrap_or(default_max_turns);
    // Базовая команда: claude -p --output-format json --model <name> --max-turns N
    // Объединённый prompt передаём позиционным аргументом либо через stdin.
    cmd.arg("-p")
        .arg("--output-format")
        .arg("json")
        .arg("--model")
        .arg(model)
        .arg("--max-turns")
        .arg(max_turns.to_string());

    if !hints.allowed_tools.is_empty() {
        cmd.arg("--allowed-tools")
            .arg(hints.allowed_tools.join(","));
    }
    if !hints.disallowed_tools.is_empty() {
        cmd.arg("--disallowed-tools")
            .arg(hints.disallowed_tools.join(","));
    }
    if let Some(mode) = &hints.permission_mode {
        cmd.arg("--permission-mode").arg(mode);
    }
    if let Some(mcp_cfg) = &hints.mcp_config {
        // При явном mcp-config — строгий режим (игнорим глобальные .mcp.json).
        cmd.arg("--strict-mcp-config")
            .arg("--mcp-config")
            .arg(mcp_cfg);
    }
    // session_id: либо --resume (продолжаем сессию orchestrator'a или
    // прогретую сессию того же агента), либо --session-id (стартуем новую
    // сессию с заранее известным UUID — runtime генерит для root claude-cli,
    // чтобы дочерние invoke могли её сразу --resume). resume имеет приоритет
    // над new — на одном вызове активна только одна из веток.
    if let Some(sid) = &hints.resume_session_id {
        cmd.arg("--resume").arg(sid);
    } else if let Some(sid) = &hints.new_session_id {
        cmd.arg("--session-id").arg(sid);
    }
    for extra in &hints.extra_args {
        cmd.arg(extra);
    }

    if let Some(cwd) = &hints.cwd {
        cmd.current_dir(cwd);
    }
    // Значения переменных из mcp_config: Claude Code раскрывает ${ИМЯ} сам.
    // Ставятся раньше служебных переменных claude-cli (CLAUDE_CONFIG_DIR, токен
    // входа — в complete), чтобы совпадающее имя не подменило fake-home и вход.
    for (name, value) in &hints.mcp_env {
        cmd.env(name, value);
    }
    // Упомянутые в mcp_config, но не найденные службой переменные убираются из
    // унаследованного окружения: иначе claude подставил бы значение, оставшееся
    // в окружении процесса со старта (например удалённое из .env), вместо
    // значения по умолчанию из ${ИМЯ:-…}.
    for name in &hints.mcp_env_missing {
        cmd.env_remove(name);
    }
    max_turns
}

// ── Provider impl ──────────────────────────────────────────────────────────

#[async_trait]
impl LlmProvider for ClaudeCliProvider {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError> {
        let deadline = tokio::time::Instant::now() + req.timeout;
        let permit = self.semaphore.clone().acquire_owned();
        let permit = tokio::time::timeout_at(deadline, permit)
            .await
            .map_err(|_| LlmError::Timeout)?
            .map_err(|e| LlmError::Subprocess(format!("семафор закрыт: {e}")))?;

        let hints = req.cli_hints.clone().unwrap_or_default();
        let mut cmd = Command::new(&self.executable);
        let max_turns = configure_command(&mut cmd, &req.model, &hints, self.default_max_turns);

        // CLAUDE_CONFIG_DIR — подмена user-scope каталога claude-cli на fake-home:
        // пустой CLAUDE.md, без rules/skills/глобального .mcp.json, но с
        // .credentials.json для авторизации. Срезает ~350k токенов на cold-старте
        // под-агента.
        // Источник пути: ПРИОРИТЕТ у config_dir из конфига (version-controlled, идёт
        // через AGENTS_MCP_CONFIG), затем — env родителя. Раньше путь брался ТОЛЬКО
        // из env, и при переносе под supervisor 2026-05-31 его потеряли — под-агенты
        // молча начали тащить полный ~/.claude, цена data_check выросла в 6×
        // ($0.6 → $3.6). Конфиг как источник переживает перенос.
        let config_dir = self
            .config_dir
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
            .or_else(|| std::env::var("CLAUDE_CONFIG_DIR").ok())
            .filter(|s| !s.is_empty());
        if let Some(dir) = &config_dir {
            cmd.env("CLAUDE_CONFIG_DIR", dir);
        }
        // Вход под-агента: сначала долгоживущий токен, и только если его нет —
        // копия подписочных учётных данных в fake-home. Токен передаём явно, а не
        // надеемся на наследование окружения: сервис под supervisor запускается со
        // своим набором переменных, и молчаливая потеря одной из них выглядит как
        // «провайдер неисправен» (так и было расценено с 24.07.2026, пока не
        // выяснилось, что дело в отсутствии токена у процесса).
        match long_lived_token() {
            Some(tok) => {
                info!(
                    token_len = tok.len(),
                    "claude-cli: вход по долгоживущему токену (fake-home только под CLAUDE.md)"
                );
                cmd.env("CLAUDE_CODE_OAUTH_TOKEN", tok);
            }
            None => {
                warn!(
                    "долгоживущего токена нет ни в окружении, ни в файле — вход пойдёт по \
                     подписочным учётным данным fake-home, что даёт 403 Request not allowed"
                );
                if let Some(dir) = &self.config_dir {
                    sync_credentials_into(dir);
                }
            }
        }

        // Сам prompt: всё, что лежит в system_prompt+user_input — отправляем как
        // позиционный аргумент. Если общий размер больше 16 KiB, используем
        // stdin, не дожидаясь системного предела командной строки Windows.
        let combined_prompt = if req.user_input.is_empty() {
            req.system_prompt.clone()
        } else {
            format!("{}\n\n## Запрос\n{}", req.system_prompt, req.user_input)
        };

        // Полная командная строка в журнал: без неё отладка «вручную работает, из
        // сервиса 403» превращается в угадывание отличий (08.08.2026 на это ушло
        // шесть экспериментов). Промпт заменяем длиной — в журнал он не нужен.
        {
            let shown: Vec<String> = cmd
                .as_std()
                .get_args()
                .map(|a| {
                    let s = a.to_string_lossy();
                    if s.len() > 120 {
                        format!("<текст {} знаков>", s.len())
                    } else {
                        s.into_owned()
                    }
                })
                .collect();
            info!(
                args = %shown.join(" "),
                cwd = ?hints.cwd,
                "claude-cli: командная строка"
            );
        }

        const ARGV_THRESHOLD: usize = 16 * 1024;
        // ВАЖНО: когда задан --mcp-config, ОБЯЗАТЕЛЬНО шлём prompt через stdin.
        // У claude CLI флаг `--mcp-config <configs...>` объявлен variadic — он
        // жадно съедает ВСЕ следующие позиционные токены как mcp-config файлы,
        // включая наш текст промпта. В результате claude ругается «MCP config
        // file not found: <первая строка промпта>» и падает rc=1. stdin-режим
        // обходит проблему: позиционного аргумента просто нет.
        //
        // То же самое даёт ЛЮБОЙ variadic-флаг в extra_args, если он оказался
        // последним: `--tools ""` съедает промпт, и claude падает с «Input must be
        // provided either through stdin or as a prompt argument». Так молча стояли
        // skill-writer, skill-rewriter и question-merger (у всех троих `--tools`
        // последний), тогда как atom-synthesizer с тем же флагом, но не последним,
        // работал. Перечислять variadic-флаги по именам бессмысленно — такие списки
        // в этом проекте уже протухали (см. disallowed_tools в конфигах агентов).
        // Поэтому при любых extra_args сразу идём через stdin: там позиционного
        // аргумента нет вовсе, и порядок флагов перестаёт что-либо значить.
        let use_stdin = combined_prompt.len() > ARGV_THRESHOLD
            || hints.mcp_config.is_some()
            || !hints.extra_args.is_empty();

        append_prompt_argument(&mut cmd, &combined_prompt, use_stdin);

        cmd.stdin(if use_stdin {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

        debug!(
            model = %req.model,
            max_turns,
            prompt_chars = combined_prompt.len(),
            use_stdin,
            "запускаю claude -p"
        );

        let (mut child, _process_tree) = proc_tree::spawn(&mut cmd)
            .map_err(|e| LlmError::Subprocess(format!("spawn claude: {e}")))?;

        if use_stdin {
            if let Some(stdin) = child.stdin.as_mut() {
                match tokio::time::timeout_at(deadline, stdin.write_all(combined_prompt.as_bytes()))
                    .await
                {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => return Err(LlmError::Subprocess(format!("write stdin: {e}"))),
                    Err(_) => return Err(LlmError::Timeout),
                }
            }
            // drop stdin → EOF
            drop(child.stdin.take());
        }

        let timeout_fut = tokio::time::timeout_at(deadline, child.wait_with_output());
        let output = match timeout_fut.await {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => return Err(LlmError::Subprocess(format!("wait: {e}"))),
            Err(_) => return Err(LlmError::Timeout),
        };

        drop(permit);

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        // Диагностика claude может повторить раскрытые подстановки mcp_config:
        // значения убираются из текстов ошибок, успешный ответ не трогается.
        let secrets: Vec<String> = hints.mcp_env.values().cloned().collect();
        let redact = |text: &str| super::tool_loop::redact_mcp_secrets(text, &secrets);

        if !output.status.success() {
            // claude headless возвращает rc=1 когда в JSON `is_error=true`.
            // Полезная диагностика — в stdout (поле "result"). stderr обычно пуст.
            // Сначала пробуем извлечь её из stdout, иначе — общий subprocess error.
            let rc = output.status.code();
            if let Ok(json) = serde_json::from_str::<Value>(&stdout) {
                let result_msg = json
                    .get("result")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let terminal = json
                    .get("terminal_reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let api_status = json.get("api_error_status").and_then(|v| v.as_u64());
                let stop_reason = json.get("stop_reason").and_then(|v| v.as_str());

                if matches!(api_status, Some(429)) {
                    return Err(LlmError::RateLimited);
                }
                if result_msg.eq_ignore_ascii_case("prompt is too long") {
                    return Err(LlmError::Provider(format!(
                        "claude-cli: модель отклонила запрос как 'prompt too long' \
                         (api_error_status={:?}, terminal_reason={}, stop_reason={:?}). \
                         Проверьте размер system prompt + user input.",
                        api_status, terminal, stop_reason
                    )));
                }
                if !result_msg.is_empty() {
                    return Err(LlmError::Provider(format!(
                        "claude-cli (rc={:?}, terminal_reason={}): {}",
                        rc,
                        terminal,
                        truncate(&redact(&result_msg), 500)
                    )));
                }
            }

            // Fallback на stderr.
            let lower = stderr.to_lowercase();
            if lower.contains("please log in") || lower.contains("not logged in") {
                return Err(LlmError::Provider(format!(
                    "claude-cli: требуется логин (`claude /login`). stderr={}",
                    truncate(&redact(&stderr), 300)
                )));
            }
            if lower.contains("rate limit") {
                return Err(LlmError::RateLimited);
            }
            return Err(LlmError::Subprocess(format!(
                "claude rc={:?}, stderr={}, stdout={}",
                rc,
                truncate(&redact(&stderr), 300),
                truncate(&redact(&stdout), 300)
            )));
        }

        // Парсинг output. Формат может слегка меняться по версиям; пробуем
        // строгий парсинг в CliResult, при неудаче извлекаем `result` руками.
        let parsed: CliResult = match serde_json::from_str(&stdout) {
            Ok(p) => p,
            Err(e) => {
                warn!(error = %e, "не удалось распарсить claude JSON по схеме, fallback");
                let value: Value = serde_json::from_str(&stdout).map_err(|e2| {
                    LlmError::InvalidResponse(format!(
                        "claude-cli: ни схема, ни сырой JSON не парсятся: {e2}; out={}",
                        truncate(&redact(&stdout), 200)
                    ))
                })?;
                CliResult {
                    result: value
                        .get("result")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    is_error: value
                        .get("is_error")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    duration_ms: value.get("duration_ms").and_then(|v| v.as_u64()),
                    cost_usd: value
                        .get("cost_usd")
                        .or_else(|| value.get("total_cost_usd"))
                        .and_then(|v| v.as_f64()),
                    session_id: value
                        .get("session_id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                    usage: serde_json::from_value(
                        value.get("usage").cloned().unwrap_or(Value::Null),
                    )
                    .ok(),
                    _other: Default::default(),
                }
            }
        };

        if parsed.is_error {
            return Err(LlmError::Provider(format!(
                "claude-cli: модель вернула is_error=true, result={}",
                truncate(&redact(&parsed.result), 300)
            )));
        }

        let usage = parsed.usage.unwrap_or_default();
        let tokens_in =
            usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens;

        debug!(
            session_id = ?parsed.session_id,
            duration_ms = ?parsed.duration_ms,
            cost_usd = ?parsed.cost_usd,
            "claude -p ok"
        );

        Ok(LlmResponse {
            content: parsed.result,
            tokens_in,
            tokens_out: usage.output_tokens,
            cost_usd: parsed.cost_usd,
            finish_reason: "stop".into(),
            reasoning: None,
            session_id: parsed.session_id,
            raw_input_tokens: usage.input_tokens,
            cache_creation_input_tokens: usage.cache_creation_input_tokens,
            cache_read_input_tokens: usage.cache_read_input_tokens,
            transcript: Vec::new(),
        })
    }
}

fn append_prompt_argument(cmd: &mut Command, prompt: &str, use_stdin: bool) {
    if !use_stdin {
        cmd.arg("--").arg(prompt);
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max_chars).collect();
        out.push_str("…[truncated]");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positional_prompt_is_preceded_by_double_dash() {
        let prompt = "- пункт";
        let mut cmd = Command::new("claude");

        append_prompt_argument(&mut cmd, prompt, false);

        let args: Vec<_> = cmd.as_std().get_args().collect();
        assert_eq!(args, ["--", prompt]);
    }

    #[test]
    fn command_keeps_mcp_config_unexpanded_and_sets_mcp_environment() {
        let raw = r#"{"mcpServers":{"fixture":{"url":"https://example/${MCP_TOKEN}"}}}"#;
        let hints = super::super::ClaudeCliHints {
            mcp_config: Some(raw.to_string()),
            mcp_env: std::collections::HashMap::from([(
                "MCP_TOKEN".to_string(),
                "resolved-secret".to_string(),
            )]),
            ..Default::default()
        };
        let mut cmd = Command::new("claude");

        configure_command(&mut cmd, "model", &hints, 8);

        let args: Vec<_> = cmd.as_std().get_args().collect();
        assert!(args.iter().any(|arg| *arg == raw));
        assert!(!args.iter().any(|arg| *arg == "resolved-secret"));
        let value = cmd
            .as_std()
            .get_envs()
            .find(|(name, _)| *name == std::ffi::OsStr::new("MCP_TOKEN"))
            .and_then(|(_, value)| value);
        assert_eq!(value, Some(std::ffi::OsStr::new("resolved-secret")));
    }
}
