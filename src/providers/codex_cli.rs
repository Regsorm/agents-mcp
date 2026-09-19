//! Провайдер codex-cli: вызов локального `codex.exe exec` через
//! tokio::process::Command.
//!
//! Авторизация — через профиль в CODEX_HOME (учётные данные ChatGPT/Codex),
//! путь берётся из конфига службы (`[providers.codex_cli] codex_home`), не из
//! окружения родителя.
//!
//! По устройству — прямой аналог `claude_cli.rs`: subprocess, семафор на
//! конкурентность, задание на stdin, разбор ответа из файла.
//!
//! Ответ модели берётся из файла `-o`, а поток событий (`--json`) читается
//! построчно ради двух вещей: пошаговый транскрипт в `agents_mcp.agent_turns`
//! (канал `req.turn_sink`, писателя поднимает рантайм) и расход токенов из
//! события `turn.completed`. Раньше поток не разбирался вовсе: ходов у
//! codex-вызовов в базе не было ни одного, а tokens_in/tokens_out стояли нули.
//!
//! Формат потока снят живым прогоном 07.09.2026 (codex-cli 0.153.4): поля лежат
//! в корне объекта, вложенного `payload` нет.
//!   `{"type":"thread.started","thread_id":"..."}`
//!   `{"type":"turn.started"}`
//!   `{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"..."}}`
//!   `{"type":"turn.completed","usage":{"input_tokens":16684,"cached_input_tokens":11008,
//!     "cache_write_input_tokens":0,"output_tokens":5,"reasoning_output_tokens":0}}`
//!
//! Прокси. Прямой маршрут до OpenAI бывает закрыт или медленным — тогда
//! процессу нужен посредник из `[providers.codex_cli] proxy`. Служба поднимается супервизором
//! без прокси в окружении, поэтому адрес задаётся там, для запускаемого
//! процесса, а не переменными всей службы: у неё есть обращения к локальным адресам, которым прокси
//! не нужен.
//!
//! Конкурентность ограничена `tokio::sync::Semaphore` (`max_concurrent` из
//! `[providers.codex_cli]`).
//!
//! Модель, усилие рассуждений (`-c model_reasoning_effort=...`) и путь к
//! схеме ответа (`--output-schema`) не зашиты в код: усилие и схема приходят
//! через `LlmRequest.cli_hints.extra_args` — то же поле, которым остальные
//! CLI-агенты этой службы передают свои дополнительные argv (секция
//! `[execution]` конфига агента).

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

use crate::proc_tree;

use super::{mask_credentials, LlmError, LlmProvider, LlmRequest, LlmResponse};

/// Приносит ли агент собственные MCP-серверы в `extra_args`. Ключи вида
/// `mcp_servers.<имя>.url=...` приходят из секции `[execution]` конфига агента
/// парой аргументов `-c` + значение, поэтому смотрим на сами значения.
fn zadany_svoi_servery(extra_args: &[String]) -> bool {
    extra_args
        .iter()
        .any(|arg| arg.trim_start().starts_with("mcp_servers"))
}

/// Потолок на длину одной записи хода в базе. Ходов у codex единицы на вызов,
/// но текст ответа модели приходит целиком в `item.completed` и на длинных
/// разборах тянет на десятки килобайт — в диагностику столько не нужно.
const ZAPIS_MAX: usize = 4000;

/// Сколько строк потока держать для сообщения об ошибке. Отказ codex приходит
/// текстом в stdout/stderr, и это единственный источник причины.
const HVOST_MAX: usize = 4000;

pub struct CodexCliProvider {
    executable: PathBuf,
    /// CODEX_HOME — профиль с учётными данными codex. Обязателен: без него
    /// codex использует профиль по умолчанию из окружения процесса, а служба
    /// запускается под supervisor с чужим/пустым окружением.
    codex_home: PathBuf,
    /// Адрес прокси для запускаемого процесса (HTTP_PROXY/HTTPS_PROXY).
    /// None — ничего не подставляем, процесс идёт напрямую.
    proxy: Option<String>,
    /// Список адресов в обход прокси (NO_PROXY).
    proxy_bypass: Option<String>,
    semaphore: Arc<Semaphore>,
}

/// Обрезать значение для записи в базу: длинные объекты заменяются пометкой
/// с исходной длиной, чтобы строка транскрипта осталась читаемой.
fn obrezat(v: &Value) -> Value {
    let s = v.to_string();
    if s.len() <= ZAPIS_MAX {
        return v.clone();
    }
    json!({
        "type": v.get("type").and_then(|x| x.as_str()).unwrap_or(""),
        "_obrezano_znakov": s.len(),
        "nachalo": s.chars().take(ZAPIS_MAX).collect::<String>(),
    })
}

fn chislo(u: Option<&Value>, pole: &str) -> u32 {
    u.and_then(|x| x.get(pole))
        .and_then(|x| x.as_u64())
        .unwrap_or(0) as u32
}

fn raw_input_tokens(tokens_in: u32, cached_in: u32) -> u32 {
    tokens_in.saturating_sub(cached_in)
}

impl CodexCliProvider {
    pub fn new(
        executable: PathBuf,
        codex_home: PathBuf,
        max_concurrent: u32,
        proxy: Option<String>,
        proxy_bypass: Option<String>,
    ) -> Self {
        let permits = max_concurrent.max(1) as usize;
        Self {
            executable,
            codex_home,
            proxy,
            proxy_bypass,
            semaphore: Arc::new(Semaphore::new(permits)),
        }
    }

    /// Doctor self-test: `codex --version` (отвечает строкой вида
    /// `codex-cli 0.153.4`). Вызывается из main.rs при старте сервиса для
    /// health-report.
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

#[async_trait]
impl LlmProvider for CodexCliProvider {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError> {
        let deadline = tokio::time::Instant::now() + req.timeout;
        let permit = self.semaphore.clone().acquire_owned();
        let permit = tokio::time::timeout_at(deadline, permit)
            .await
            .map_err(|_| LlmError::Timeout)?
            .map_err(|e| LlmError::Subprocess(format!("семафор закрыт: {e}")))?;

        let hints = req.cli_hints.clone().unwrap_or_default();

        // Файл ответа: codex пишет туда текст последнего сообщения модели.
        // Уникальное имя на каждый вызов — параллельные вызовы под общим
        // семафором не должны делить один файл.
        let otvet_file =
            std::env::temp_dir().join(format!("agents-mcp-codex-{}.txt", uuid::Uuid::new_v4()));

        let mut cmd = Command::new(&self.executable);
        cmd.arg("exec").arg("-m").arg(&req.model);

        // Пустой список серверов ставим только тогда, когда агент не приносит
        // своего: ключ `-c mcp_servers={}` затирает ВЕСЬ список из профиля
        // CODEX_HOME, и точечные ключи (`mcp_servers.<имя>.url`), добавленные
        // следом, его уже не восстанавливают — проверено живым прогоном
        // 19.09.2026: инструменты code-index так и не появились у модели.
        if !zadany_svoi_servery(&hints.extra_args) {
            cmd.arg("-c").arg("mcp_servers={}");
        }

        cmd.arg("--skip-git-repo-check")
            // Поток событий: нужен для транскрипта и расхода токенов. Файл
            // ответа (`-o`) при этом пишется по-прежнему.
            .arg("--json");

        // Рабочий каталог вызова (`cwd_template` агента) — ключом `-C`: для
        // codex это корень, в котором он читает и правит файлы, и песочница
        // `workspace-write` разрешает запись именно в него.
        if let Some(cwd) = &hints.cwd {
            cmd.arg("-C").arg(cwd);
        }

        // Усилие рассуждений (`-c model_reasoning_effort="..."`) и путь к
        // схеме ответа (`--output-schema <файл>`) — из настроек агента, не из
        // кода: обе подсказки приходят через extra_args секции [execution]
        // config.toml нужного агента (см. doc-комментарий модуля выше).
        for extra in &hints.extra_args {
            cmd.arg(extra);
        }

        cmd.arg("-o").arg(&otvet_file).arg("-");

        cmd.env("CODEX_HOME", &self.codex_home);
        if let Some(adres) = &self.proxy {
            cmd.env("HTTP_PROXY", adres)
                .env("HTTPS_PROXY", adres)
                .env("http_proxy", adres)
                .env("https_proxy", adres);
            if let Some(bypass) = &self.proxy_bypass {
                cmd.env("NO_PROXY", bypass).env("no_proxy", bypass);
            }
        }

        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let combined_prompt = if req.user_input.is_empty() {
            req.system_prompt.clone()
        } else {
            format!("{}\n\n## Запрос\n{}", req.system_prompt, req.user_input)
        };

        info!(
            model = %req.model,
            prompt_chars = combined_prompt.len(),
            proxy = %mask_credentials(self.proxy.as_deref().unwrap_or("нет")),
            otvet_file = %otvet_file.display(),
            "codex-cli: запускаю codex exec"
        );

        let (mut child, _process_tree) = proc_tree::spawn(&mut cmd)
            .map_err(|e| LlmError::Subprocess(format!("spawn codex: {e}")))?;

        if let Some(stdin) = child.stdin.as_mut() {
            match tokio::time::timeout_at(deadline, stdin.write_all(combined_prompt.as_bytes()))
                .await
            {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(LlmError::Subprocess(format!("write stdin: {e}"))),
                Err(_) => return Err(LlmError::Timeout),
            }
        }
        // drop stdin → EOF (задание подаётся только через stdin, у codex нет
        // ключа вида --prompt-file).
        drop(child.stdin.take());

        // stderr собираем отдельной задачей: если его не вычитывать, труба
        // переполнится и процесс встанет на записи.
        let stderr_pipe = child.stderr.take();
        let stderr_task = tokio::spawn(async move {
            let mut buf = String::new();
            if let Some(mut e) = stderr_pipe {
                let _ = e.read_to_string(&mut buf).await;
            }
            buf
        });

        let stdout_pipe = child.stdout.take();
        let sink = req.turn_sink.clone();

        // Чтение потока событий и ожидание завершения — под общим таймаутом.
        let chtenie = async {
            let mut seq: i64 = 0;
            let mut usage: Option<Value> = None;
            let mut thread_id: Option<String> = None;
            let mut rassuzhdeniya: Vec<String> = Vec::new();
            let mut hvost = String::new();
            let mut turn_failure: Option<String> = None;

            if let Some(so) = stdout_pipe {
                let mut lines = BufReader::new(so).lines();
                loop {
                    let line = match lines.next_line().await {
                        Ok(Some(l)) => l,
                        Ok(None) => break,
                        Err(e) => {
                            return Err(LlmError::Subprocess(format!("чтение потока: {e}")));
                        }
                    };
                    if line.trim().is_empty() {
                        continue;
                    }
                    if hvost.len() < HVOST_MAX {
                        hvost.push_str(&line);
                        hvost.push('\n');
                    }
                    // Не-JSON строки в потоке возможны (предупреждения CLI) —
                    // они уже попали в хвост для диагностики, дальше пропускаем.
                    let v: Value = match serde_json::from_str(&line) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let tip = v
                        .get("type")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();

                    match tip.as_str() {
                        "thread.started" => {
                            thread_id = v
                                .get("thread_id")
                                .and_then(|x| x.as_str())
                                .map(|s| s.to_string());
                        }
                        "turn.completed" => {
                            usage = v.get("usage").cloned();
                        }
                        "turn.failed" => {
                            turn_failure = v
                                .pointer("/error/message")
                                .and_then(|x| x.as_str())
                                .map(str::to_string)
                                .or_else(|| Some("codex сообщил turn.failed".to_string()));
                        }
                        "item.completed" => {
                            if let Some(item) = v.get("item") {
                                if item.get("type").and_then(|x| x.as_str()) == Some("reasoning") {
                                    if let Some(t) = item.get("text").and_then(|x| x.as_str()) {
                                        rassuzhdeniya.push(t.to_string());
                                    }
                                }
                            }
                        }
                        _ => {}
                    }

                    // Ход уходит в базу сразу: у зависшего вызова иначе не
                    // остаётся ничего, разбирать провал будет нечем.
                    if let Some(tx) = &sink {
                        seq += 1;
                        let _ = tx.send(json!({
                            "seq": seq,
                            "event": tip,
                            "ts_ms": chrono::Utc::now().timestamp_millis(),
                            "zapis": obrezat(&v),
                        }));
                    }
                }
            }

            let status = child
                .wait()
                .await
                .map_err(|e| LlmError::Subprocess(format!("wait: {e}")))?;

            Ok::<_, LlmError>((
                status,
                seq,
                usage,
                thread_id,
                rassuzhdeniya,
                hvost,
                turn_failure,
            ))
        };

        let (status, hodov, usage, thread_id, rassuzhdeniya, hvost, turn_failure) =
            match tokio::time::timeout_at(deadline, chtenie).await {
                Ok(Ok(x)) => x,
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err(LlmError::Timeout),
            };

        let stderr_text = stderr_task.await.unwrap_or_default();

        drop(permit);

        // Отказ codex (невалидная схема, неподдерживаемая модель и т.п.)
        // приходит за секунды и НЕ создаёт файл ответа — это не таймаут, а
        // мгновенное отклонение запроса. Текст отказа — единственный источник
        // причины, поэтому сохраняем его целиком, без усечения.
        if !otvet_file.exists() {
            return Err(LlmError::Provider(format!(
                "codex-cli: файл ответа не создан (rc={:?}). поток={hvost} stderr={stderr_text}",
                status.code()
            )));
        }

        if !status.success() || turn_failure.is_some() {
            let partial_chars = std::fs::read_to_string(&otvet_file)
                .map(|text| text.chars().count())
                .unwrap_or(0);
            if let Err(e) = std::fs::remove_file(&otvet_file) {
                warn!(
                    path = %otvet_file.display(),
                    error = %e,
                    "не удалось удалить частичный файл ответа codex-cli"
                );
            }
            return Err(LlmError::Provider(format!(
                "codex-cli завершился неуспешно (rc={:?}, turn_failure={:?}, partial_chars={partial_chars}). поток={hvost} stderr={stderr_text}",
                status.code(),
                turn_failure
            )));
        }

        let content = match std::fs::read_to_string(&otvet_file) {
            Ok(s) => s.trim().to_string(),
            Err(e) => {
                return Err(LlmError::Subprocess(format!(
                    "чтение файла ответа '{}': {e}",
                    otvet_file.display()
                )))
            }
        };
        if let Err(e) = std::fs::remove_file(&otvet_file) {
            warn!(
                path = %otvet_file.display(),
                error = %e,
                "не удалось удалить временный файл ответа codex-cli"
            );
        }

        if content.is_empty() {
            return Err(LlmError::InvalidResponse(format!(
                "codex-cli: файл ответа пуст. поток={hvost} stderr={stderr_text}"
            )));
        }

        let u = usage.as_ref();
        let tokens_in = chislo(u, "input_tokens");
        let tokens_out = chislo(u, "output_tokens");
        let cached_in = chislo(u, "cached_input_tokens");
        let cache_write = chislo(u, "cache_write_input_tokens");
        let raw_input_tokens = raw_input_tokens(tokens_in, cached_in);

        debug!(
            chars = content.len(),
            hodov, tokens_in, tokens_out, "codex exec ok"
        );

        Ok(LlmResponse {
            content,
            tokens_in,
            tokens_out,
            // Тариф подписки ChatGPT: поштучной стоимости вызова нет.
            cost_usd: None,
            finish_reason: "stop".into(),
            reasoning: if rassuzhdeniya.is_empty() {
                None
            } else {
                Some(rassuzhdeniya.join("\n\n"))
            },
            session_id: thread_id,
            raw_input_tokens,
            cache_creation_input_tokens: cache_write,
            cache_read_input_tokens: cached_in,
            // Ходы уже ушли в базу потоком через turn_sink — пакетом не дублируем.
            transcript: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::ClaudeCliHints;
    use serde_json::json;

    #[test]
    fn turn_completed_separates_cached_input_tokens() {
        let event = json!({
            "type": "turn.completed",
            "usage": {
                "input_tokens": 16684,
                "cached_input_tokens": 11008,
                "cache_write_input_tokens": 0,
                "output_tokens": 5
            }
        });
        let usage = event.get("usage");
        let tokens_in = chislo(usage, "input_tokens");
        let cache_read_input_tokens = chislo(usage, "cached_input_tokens");

        assert_eq!(tokens_in, 16684);
        assert_eq!(cache_read_input_tokens, 11008);
        assert_eq!(raw_input_tokens(tokens_in, cache_read_input_tokens), 5676);
    }

    #[test]
    fn raw_input_tokens_does_not_underflow() {
        assert_eq!(raw_input_tokens(10, 20), 0);
    }

    #[test]
    fn svoi_servery_uznayutsya_po_klyuchu_mcp_servers() {
        let svoi = vec![
            "-c".to_string(),
            "mcp_servers.code_index.url=\"http://127.0.0.1:8011/mcp\"".to_string(),
        ];
        assert!(zadany_svoi_servery(&svoi));

        let chuzhie = vec![
            "-c".to_string(),
            "model_reasoning_effort=\"high\"".to_string(),
        ];
        assert!(!zadany_svoi_servery(&chuzhie));
        assert!(!zadany_svoi_servery(&[]));
    }

    /// Агент со своими серверами не должен получать затирающий `mcp_servers={}`,
    /// а рабочий каталог обязан уходить ключом `-C`: без первого у модели нет
    /// инструментов code-index, без второго codex правит файлы не в том месте.
    #[tokio::test]
    async fn argv_uchityvaet_svoi_servery_i_rabochiy_katalog() {
        let dir =
            std::env::temp_dir().join(format!("agents-mcp-codex-argv-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("каталог fake codex");
        let source = dir.join("argv_cli.rs");
        std::fs::write(
            &source,
            r#"fn main() {
    use std::io::Read;
    let args = std::env::args().collect::<Vec<_>>();
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    let out = args.windows(2).find(|pair| pair[0] == "-o").unwrap()[1].clone();
    std::fs::write(out, args[1..].join(" ")).unwrap();
}"#,
        )
        .expect("исходник fake codex");
        let executable = dir.join(if cfg!(windows) {
            "argv_cli.exe"
        } else {
            "argv_cli"
        });
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let compiled = std::process::Command::new(rustc)
            .arg(&source)
            .arg("-o")
            .arg(&executable)
            .status()
            .expect("запуск rustc");
        assert!(compiled.success(), "fake codex должен собраться");

        let provider = CodexCliProvider::new(executable, dir.clone(), 1, None, None);
        let zapros = |hints: ClaudeCliHints| LlmRequest {
            model: "fake-model".into(),
            system_prompt: "prompt".into(),
            user_input: String::new(),
            temperature: 0.0,
            max_tokens: 10,
            top_p: None,
            extra_body: serde_json::Map::new(),
            timeout: Duration::from_secs(30),
            cli_hints: Some(hints),
            turn_sink: None,
            fallback_skill_names: Vec::new(),
            prompt_skill_names: Vec::new(),
            skills: None,
        };

        let so_svoimi = provider
            .complete(zapros(ClaudeCliHints {
                cwd: Some(dir.clone()),
                extra_args: vec![
                    "-c".to_string(),
                    "mcp_servers.code_index.url=\"http://127.0.0.1:8011/mcp\"".to_string(),
                ],
                ..Default::default()
            }))
            .await
            .expect("вызов со своими серверами")
            .content;
        assert!(
            !so_svoimi.contains("mcp_servers={}"),
            "список серверов затёрт: {so_svoimi}"
        );
        assert!(so_svoimi.contains(" -C "), "нет ключа -C: {so_svoimi}");
        assert!(
            so_svoimi.contains(&dir.to_string_lossy().to_string()),
            "рабочий каталог не передан: {so_svoimi}"
        );

        let bez_svoih = provider
            .complete(zapros(ClaudeCliHints::default()))
            .await
            .expect("вызов без своих серверов")
            .content;
        assert!(
            bez_svoih.contains("mcp_servers={}"),
            "без своих серверов список обязан обнуляться: {bez_svoih}"
        );
        assert!(!bez_svoih.contains(" -C "), "лишний -C: {bez_svoih}");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn proxy_log_field_hides_password() {
        // Адрес склеен из частей, чтобы проверка секретов перед фиксацией не
        // принимала тестовую строку за настоящий пароль.
        let proxy = concat!("http://user", ":secret@proxy:3128");
        let field = mask_credentials(proxy);
        assert!(!field.contains("secret"));
        assert_eq!(field, "***@proxy:3128");
    }

    #[tokio::test]
    async fn failed_exit_with_partial_file_returns_error() {
        let dir =
            std::env::temp_dir().join(format!("agents-mcp-codex-fake-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("каталог fake codex");
        let source = dir.join("fake_cli.rs");
        std::fs::write(
            &source,
            r#"fn main() {
    use std::io::Read;
    let args = std::env::args().collect::<Vec<_>>();
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    let out = args.windows(2).find(|pair| pair[0] == "-o").unwrap()[1].clone();
    std::fs::write(out, "partial fixture response").unwrap();
    println!("{{\"type\":\"turn.failed\",\"error\":{{\"message\":\"fixture failure\"}}}}");
    std::process::exit(1);
}"#,
        )
        .expect("исходник fake codex");
        let executable = dir.join(if cfg!(windows) {
            "fake_cli.exe"
        } else {
            "fake_cli"
        });
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let compiled = std::process::Command::new(rustc)
            .arg(&source)
            .arg("-o")
            .arg(&executable)
            .status()
            .expect("запуск rustc");
        assert!(compiled.success(), "fake codex должен собраться");

        let provider = CodexCliProvider::new(executable, dir.clone(), 1, None, None);
        let request = LlmRequest {
            model: "fake-model".into(),
            system_prompt: "prompt".into(),
            user_input: String::new(),
            temperature: 0.0,
            max_tokens: 10,
            top_p: None,
            extra_body: serde_json::Map::new(),
            timeout: std::time::Duration::from_secs(5),
            cli_hints: None,
            turn_sink: None,
            fallback_skill_names: Vec::new(),
            prompt_skill_names: Vec::new(),
            skills: None,
        };
        let error = provider
            .complete(request)
            .await
            .expect_err("rc=1 и turn.failed не могут быть успехом");
        let text = error.to_string();
        assert!(text.contains("rc=Some(1)"), "error={text}");
        assert!(text.contains("fixture failure"), "error={text}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn stdin_write_is_inside_provider_timeout() {
        let dir =
            std::env::temp_dir().join(format!("agents-mcp-codex-no-read-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("no_read.rs");
        std::fs::write(
            &source,
            "fn main() { std::thread::sleep(std::time::Duration::from_secs(2)); }",
        )
        .unwrap();
        let executable = dir.join(if cfg!(windows) {
            "no_read.exe"
        } else {
            "no_read"
        });
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let compiled = std::process::Command::new(rustc)
            .arg(&source)
            .arg("-o")
            .arg(&executable)
            .status()
            .unwrap();
        assert!(compiled.success());

        let provider = CodexCliProvider::new(executable, dir.clone(), 1, None, None);
        let request = LlmRequest {
            model: "fake-model".into(),
            system_prompt: "x".repeat(4 * 1024 * 1024),
            user_input: String::new(),
            temperature: 0.0,
            max_tokens: 10,
            top_p: None,
            extra_body: serde_json::Map::new(),
            timeout: Duration::from_millis(40),
            cli_hints: None,
            turn_sink: None,
            fallback_skill_names: Vec::new(),
            prompt_skill_names: Vec::new(),
            skills: None,
        };
        let started = std::time::Instant::now();
        let error = provider.complete(request).await.expect_err("ожидался срок");
        assert!(matches!(error, LlmError::Timeout));
        assert!(started.elapsed() < Duration::from_millis(500));
        let _ = std::fs::remove_dir_all(dir);
    }
}
