//! Структура ответа `/health` и MCP-tool `health`.
//!
//! Поля заполняются из текущего состояния службы (см. `server::AppState`).
//! Верхние поля ответа сохраняют стабильный контракт для супервизора, а
//! `providers` показывает регистрацию и итог последней CLI-проверки.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
    pub started_at: String,
    pub uptime_sec: i64,
    /// Количество загруженных агентов.
    pub agents_loaded: usize,
    /// Статус зарегистрированных провайдеров LLM.
    pub providers: std::collections::BTreeMap<String, ProviderStatus>,
    /// Состояние хранилища задач. Заполняется настоящим запросом к БД:
    /// живой процесс ещё не означает рабочее соединение — 20.08.2026 служба
    /// 5.4 суток отвечала «ok», пока каждый запрос падал с «connection closed».
    pub database: ProviderStatus,
    /// Осиротевшие вызовы, помеченные ошибкой при старте службы (только
    /// транспорт http). None — помечать было нечего либо экземпляр запущен на
    /// транспорте stdio и вызовов службы не касается.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orphaned_at_start: Option<OrphanedAtStart>,
    /// Экземпляр службы, как он пишется в строки вызовов (`[server] instance`,
    /// по умолчанию «имя машины:порт»).
    pub instance: String,
}

/// Что пометили при старте: сколько всего осиротевших вызовов и разбивка по
/// агентам (`summarize_orphans`).
#[derive(Debug, Clone, Serialize)]
pub struct OrphanedAtStart {
    pub count: usize,
    pub agents: String,
}

/// Заполняется один раз в `main` после стартовой пометки осиротевших вызовов.
static ORPHANED_AT_START: std::sync::OnceLock<OrphanedAtStart> = std::sync::OnceLock::new();

/// Сохранить итог стартовой пометки (вызывает `main`).
pub fn set_orphaned_at_start(value: OrphanedAtStart) {
    let _ = ORPHANED_AT_START.set(value);
}

/// Прочитать итог стартовой пометки; None — помеченных при старте не было.
pub fn orphaned_at_start() -> Option<OrphanedAtStart> {
    ORPHANED_AT_START.get().cloned()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProviderStatus {
    /// "registered" — зарегистрирован без проверки; "ok" — последняя проверка
    /// успешна; "down" — последняя проверка завершилась ошибкой.
    pub status: &'static str,
    /// Причина неуспешной проверки.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl ProviderStatus {
    pub fn registered() -> Self {
        Self {
            status: "registered",
            message: None,
        }
    }

    pub fn ok() -> Self {
        Self {
            status: "ok",
            message: None,
        }
    }

    pub fn down(message: impl Into<String>) -> Self {
        Self {
            status: "down",
            message: Some(message.into()),
        }
    }
}

impl HealthResponse {
    pub fn new(started_at: chrono::DateTime<chrono::Utc>, agents_loaded: usize) -> Self {
        let now = chrono::Utc::now();
        Self {
            status: "ok",
            version: env!("CARGO_PKG_VERSION"),
            started_at: started_at.to_rfc3339(),
            uptime_sec: (now - started_at).num_seconds(),
            agents_loaded,
            providers: std::collections::BTreeMap::new(),
            // Значение до опроса БД; оба места сборки ответа его перезаписывают.
            database: ProviderStatus {
                status: "unknown",
                message: None,
            },
            // Осиротевшие вызовы помечены один раз при старте — берём готовый итог.
            orphaned_at_start: orphaned_at_start(),
            // Значение до сборки ответа: build_health перезаписывает его именем
            // экземпляра из Runtime.
            instance: String::new(),
        }
    }
}
