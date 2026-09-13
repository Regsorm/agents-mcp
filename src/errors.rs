//! Ошибки чтения и разбора основного конфига службы.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentsMcpError {
    #[error("ошибка чтения конфига: {0}")]
    ConfigRead(#[source] std::io::Error),

    #[error("ошибка парсинга конфига: {0}")]
    ConfigParse(#[source] toml::de::Error),
}

pub type Result<T> = std::result::Result<T, AgentsMcpError>;

/// Только координаты ошибки: сообщение TOML может содержать секретные значения.
pub fn safe_config_error(error: &AgentsMcpError) -> String {
    match error {
        AgentsMcpError::ConfigParse(error) => {
            // Из диагностического заголовка берём только число, не текст ошибки.
            let diagnostic = error.to_string();
            let line = diagnostic
                .lines()
                .next()
                .and_then(|s| s.strip_prefix("TOML parse error at line "))
                .and_then(|s| s.split(',').next())
                .and_then(|s| s.parse::<usize>().ok());
            match line {
                Some(line) => format!("ошибка разбора конфига в строке {line} (текст строки скрыт)"),
                None => match error.span() {
                    Some(span) => format!(
                        "ошибка разбора конфига, байтовая позиция {} (текст строки скрыт)",
                        span.start
                    ),
                    None => "ошибка разбора конфига (текст строки скрыт)".to_string(),
                },
            }
        }
        _ => error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_parse_error_hides_secret_and_source_chain() {
        for raw in [
            "[storage]\ntask_store_dsn = \"postgres://u:hunter2@h/db\n",
            "[server]\nport = \"hunter2\"\n",
        ] {
            let error = toml::from_str::<crate::config::Config>(raw).unwrap_err();
            let safe = safe_config_error(&AgentsMcpError::ConfigParse(error));
            assert!(safe.contains("строке 2"));
            assert!(!safe.contains("hunter2"));
            let startup_error = anyhow::Error::msg(safe);
            assert!(!format!("{startup_error:?}").contains("hunter2"));
        }
    }
}
