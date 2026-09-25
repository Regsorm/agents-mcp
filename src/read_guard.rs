//! Проверка чтения файлов агентом, которому доступен индекс кода.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tracing::debug;

use crate::config::FsConfig;
use crate::runtime::CallScope;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ReadGuard {
    pub program: PathBuf,
    pub mcp_prefix: Option<String>,
}

impl ReadGuard {
    pub(crate) fn from_config(fs: &FsConfig) -> Option<Self> {
        fs.read_guard.as_ref().map(|program| Self {
            program: program.clone(),
            mcp_prefix: fs.read_guard_mcp_prefix.clone(),
        })
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum GuardVerdict {
    Deny(String),
    Pass,
    Broken(String),
}

pub(crate) const READ_GUARD_TIMEOUT: Duration = Duration::from_secs(5);

fn parse_guard_reply(stdout: &str) -> GuardVerdict {
    if stdout.trim().is_empty() {
        return GuardVerdict::Pass;
    }
    let value: serde_json::Value = match serde_json::from_str(stdout) {
        Ok(value) => value,
        Err(error) => return GuardVerdict::Broken(format!("ответ гарда не JSON: {error}")),
    };
    let output = &value["hookSpecificOutput"];
    match output["permissionDecision"].as_str() {
        Some("deny") => {
            let reason = output["permissionDecisionReason"]
                .as_str()
                .map(str::trim)
                .filter(|reason| !reason.is_empty())
                .unwrap_or("чтение запрещено гардом индекса: читайте через code-index");
            GuardVerdict::Deny(reason.to_string())
        }
        Some("allow") => GuardVerdict::Pass,
        _ => GuardVerdict::Broken("в ответе гарда нет допустимого решения".to_string()),
    }
}

pub(crate) async fn ask_read_guard(
    guard: &ReadGuard,
    file_path: &str,
    cwd: Option<&Path>,
    timeout: Duration,
) -> GuardVerdict {
    let result = tokio::time::timeout(timeout, async {
        let mut command = Command::new(&guard.program);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(prefix) = &guard.mcp_prefix {
            command.arg("--mcp-prefix").arg(prefix);
        }
        let mut child = command
            .spawn()
            .map_err(|error| format!("запуск гарда: {error}"))?;
        let guard_cwd = cwd.or_else(|| Path::new(file_path).parent());
        let input = serde_json::json!({
            "tool_name": "Read",
            "tool_input": {"file_path": file_path},
            "cwd": guard_cwd.map(|path| path.to_string_lossy().into_owned()),
        });
        if let Some(mut stdin) = child.stdin.take() {
            if let Err(error) = stdin.write_all(input.to_string().as_bytes()).await {
                debug!(error = %error, "не удалось записать запрос гарду индекса");
            }
            drop(stdin);
        }
        let output = child
            .wait_with_output()
            .await
            .map_err(|error| format!("ожидание гарда: {error}"))?;
        if !output.status.success() {
            return Err(format!("гард завершился с кодом {}", output.status));
        }
        String::from_utf8(output.stdout).map_err(|error| format!("stdout гарда не UTF-8: {error}"))
    })
    .await;
    match result {
        Ok(Ok(stdout)) => parse_guard_reply(&stdout),
        Ok(Err(error)) => GuardVerdict::Broken(error),
        Err(_) => GuardVerdict::Broken("время ожидания гарда истекло".to_string()),
    }
}

pub(crate) fn read_guard_applies(scope: Option<&CallScope>, guard: Option<&ReadGuard>) -> bool {
    guard.is_some() && scope.is_some_and(|scope| scope.reads_code_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;
    use std::time::Instant;

    fn fake_guard_program() -> &'static PathBuf {
        static PROGRAM: OnceLock<PathBuf> = OnceLock::new();
        PROGRAM.get_or_init(|| {
            let dir = std::env::temp_dir().join(format!(
                "agents-mcp-read-guard-{}",
                uuid::Uuid::new_v4()
            ));
            std::fs::create_dir_all(&dir).expect("каталог подставного гарда");
            let source = dir.join("fake_read_guard.rs");
            std::fs::write(
                &source,
                r##"fn main() {
    use std::io::Read;
    let args = std::env::args().collect::<Vec<_>>();
    let prefix = args.windows(2).find(|pair| pair[0] == "--mcp-prefix")
        .map(|pair| pair[1].as_str()).unwrap_or("silent|");
    let (mode, mark) = prefix.split_once('|').unwrap();
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    if !mark.is_empty() {
        std::fs::write(mark, input).unwrap();
    }
    match mode {
        "deny" => println!("{}", r#"{"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"fixture deny"}}"#),
        "allow" => println!("{}", r#"{"hookSpecificOutput":{"permissionDecision":"allow"}}"#),
        "silent" => (),
        "garbage" => println!("not json"),
        "crash" => std::process::exit(1),
        "hang" => std::thread::sleep(std::time::Duration::from_secs(10)),
        _ => panic!("неизвестный режим"),
    }
}"##,
            )
            .expect("исходник подставного гарда");
            let program = dir.join(if cfg!(windows) {
                "fake_read_guard.exe"
            } else {
                "fake_read_guard"
            });
            let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
            let status = std::process::Command::new(rustc)
                .arg(&source)
                .arg("-o")
                .arg(&program)
                .status()
                .expect("запуск rustc");
            assert!(status.success(), "подставной гард должен собраться");
            program
        })
    }

    fn fixture(mode: &str, mark: &Path) -> ReadGuard {
        ReadGuard {
            program: fake_guard_program().clone(),
            mcp_prefix: Some(format!("{mode}|{}", mark.display())),
        }
    }

    #[tokio::test]
    async fn guard_denies_and_receives_request() {
        let mark = std::env::temp_dir().join(format!("guard-mark-{}.json", uuid::Uuid::new_v4()));
        let cwd = std::env::temp_dir().join("guard-work");
        let file_path = cwd.join("file.rs").to_string_lossy().into_owned();
        let verdict = ask_read_guard(
            &fixture("deny", &mark),
            &file_path,
            Some(&cwd),
            READ_GUARD_TIMEOUT,
        )
        .await;
        assert_eq!(verdict, GuardVerdict::Deny("fixture deny".to_string()));
        let input: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&mark).expect("метка гарда")).unwrap();
        assert_eq!(input["tool_name"], "Read");
        assert_eq!(input["tool_input"]["file_path"], file_path);
        assert_eq!(input["cwd"].as_str(), Some(cwd.to_string_lossy().as_ref()));
        std::fs::remove_file(mark).ok();
    }

    #[tokio::test]
    async fn guard_passes_silent_and_allow() {
        let mark = std::env::temp_dir().join(format!("guard-mark-{}.json", uuid::Uuid::new_v4()));
        for mode in ["silent", "allow"] {
            assert_eq!(
                ask_read_guard(&fixture(mode, &mark), "file.rs", None, READ_GUARD_TIMEOUT).await,
                GuardVerdict::Pass
            );
        }
        std::fs::remove_file(mark).ok();
    }

    #[tokio::test]
    async fn guard_failures_are_broken() {
        let mark = std::env::temp_dir().join(format!("guard-mark-{}.json", uuid::Uuid::new_v4()));
        for mode in ["garbage", "crash", "hang"] {
            let start = Instant::now();
            let verdict = ask_read_guard(
                &fixture(mode, &mark),
                "file.rs",
                None,
                Duration::from_millis(300),
            )
            .await;
            assert!(matches!(verdict, GuardVerdict::Broken(_)));
            if mode == "hang" {
                assert!(start.elapsed() < Duration::from_secs(2));
            }
        }
        let missing = ReadGuard {
            program: mark.with_extension("missing"),
            mcp_prefix: None,
        };
        assert!(matches!(
            ask_read_guard(&missing, "file.rs", None, READ_GUARD_TIMEOUT).await,
            GuardVerdict::Broken(_)
        ));
        std::fs::remove_file(mark).ok();
    }

    #[test]
    fn parses_guard_replies() {
        assert_eq!(parse_guard_reply("  "), GuardVerdict::Pass);
        assert_eq!(
            parse_guard_reply(r#"{"hookSpecificOutput":{"permissionDecision":"allow"}}"#),
            GuardVerdict::Pass
        );
        assert_eq!(
            parse_guard_reply(r#"{"hookSpecificOutput":{"permissionDecision":"deny"}}"#),
            GuardVerdict::Deny(
                "чтение запрещено гардом индекса: читайте через code-index".to_string()
            )
        );
        assert!(matches!(
            parse_guard_reply("garbage"),
            GuardVerdict::Broken(_)
        ));
    }

    #[test]
    fn guard_requires_index_scope_and_config() {
        let guard = ReadGuard {
            program: PathBuf::from("guard"),
            mcp_prefix: None,
        };
        let mut scope = CallScope {
            call_id: 1,
            cwd: None,
            allowed_roots: None,
            parent_call_id: None,
            orchestration_depth: 0,
            reads_code_index: false,
        };
        assert!(!read_guard_applies(Some(&scope), Some(&guard)));
        assert!(!read_guard_applies(None, Some(&guard)));
        scope.reads_code_index = true;
        assert!(!read_guard_applies(Some(&scope), None));
        assert!(read_guard_applies(Some(&scope), Some(&guard)));
    }

    #[test]
    fn config_reads_optional_guard_fields() {
        let configured: FsConfig = toml::from_str(
            "read_guard = 'C:/guard.exe'\nread_guard_mcp_prefix = 'mcp__code-index__'\n",
        )
        .unwrap();
        assert_eq!(
            ReadGuard::from_config(&configured),
            Some(ReadGuard {
                program: PathBuf::from("C:/guard.exe"),
                mcp_prefix: Some("mcp__code-index__".to_string()),
            })
        );
        let default: FsConfig = toml::from_str("allowed_roots = []\n").unwrap();
        assert_eq!(default.read_guard, None);
        assert_eq!(default.read_guard_mcp_prefix, None);
        assert_eq!(ReadGuard::from_config(&default), None);
    }
}
