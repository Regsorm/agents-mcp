//! Слой LLM-провайдеров: общий trait и реализации.
//!
//! День 2 — `mock` (для smoke-тестов без сети).
//! День 3 — `openrouter`, `anthropic`.
//! День 4 — `claude_cli` (subprocess через `claude -p`).

pub mod anthropic;
pub mod claude_cli;
pub mod codex_cli;
pub mod mcp_client;
pub mod mock;
pub mod openrouter;
pub mod pricing;
pub(crate) mod tool_loop;

use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub model: String,
    /// Отрендеренный prompt.md: у прямых провайдеров это system-сообщение, а CLI
    /// объединяют его с user_input и передают как обычный prompt.
    pub system_prompt: String,
    /// Доп. user-сообщение, если нужно. Phase 1 — обычно пусто: всё в prompt.md.
    pub user_input: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub top_p: Option<f32>,
    /// Произвольные поля тела запроса из `[model.extra_body]` конфига агента.
    /// Прямые OpenAI-совместимые и Anthropic-провайдеры подмешивают их в тело
    /// запроса на верхнем уровне. Пустая карта ничего не добавляет.
    pub extra_body: serde_json::Map<String, Value>,
    pub timeout: Duration,
    /// Условия исполнения вызова (allowed_tools, mcp_config, cwd, max_turns).
    /// Заполняются runtime'ом из секции `[execution]` конфига агента и нужны
    /// всем провайдерам: claude-cli передаёт их ключами командной строки,
    /// прямые провайдеры — набором инструментов в запросе к модели.
    #[allow(dead_code)]
    pub cli_hints: Option<ClaudeCliHints>,
    /// Куда провайдер шлёт ходы ПО МЕРЕ выполнения, а не пачкой в конце.
    /// Без этого канала разбор провалов невозможен: транскрипт копился в памяти
    /// и попадал в БД только при успехе либо при `MaxTurns`, а при таймауте
    /// задача вообще не доходит до кода записи и остаётся `running` навсегда —
    /// именно так 30.08.2026 бесследно пропали пять зависших вызовов генератора.
    pub turn_sink: Option<tokio::sync::mpsc::UnboundedSender<Value>>,
    /// Имена навыков «в запасе» — те, что нашлись по фразе задания следом за
    /// вложенными в промпт. Тела здесь НЕ лежат: они грузятся только если
    /// понадобились, то есть при петле и только когда поиск по месту затыка
    /// (см. `skills`) ничего не дал.
    ///
    /// Почему запасной путь именно такой: каталог по фразе задания собран в
    /// начале вызова, и темы, на которой модель встанет на двадцатом ходу, в нём
    /// может не быть вовсе — замер 31.08.2026: навык про пустой отбор по статусу
    /// перечисления по фразе задания не входит и в первую восьмёрку, зато по
    /// хвосту размышления встаёт первым. Поэтому это подстраховка, а не основной
    /// источник.
    pub fallback_skill_names: Vec<String>,
    /// Имена навыков, чьи тела уже лежат в системном промпте. Провайдеру нужны,
    /// чтобы при петле не подложить в диалог то, что модель и так видит перед
    /// собой на каждом ходу.
    pub prompt_skill_names: Vec<String>,
    /// Клиент библиотеки навыков — чтобы искать навык по тому месту, где модель
    /// зациклилась. Промпт к этому моменту собран давно, и своего доступа к
    /// навыкам у провайдера нет. None — поиск выключен, работает только запас.
    pub skills: Option<crate::skills::SkillsClient>,
}

/// Подсказки для ClaudeCliProvider (см. `providers/claude_cli.rs`).
#[derive(Debug, Clone, Default)]
pub struct ClaudeCliHints {
    pub allowed_tools: Vec<String>,
    pub disallowed_tools: Vec<String>,
    pub permission_mode: Option<String>,
    /// Уже отрендеренный путь к рабочей директории (tera-подстановки сделаны
    /// в runtime).
    pub cwd: Option<std::path::PathBuf>,
    pub mcp_config: Option<String>,
    /// Разрешённые значения переменных из `mcp_config`; секреты, в журнал не писать.
    pub mcp_env: std::collections::HashMap<String, String>,
    /// Переменные из `mcp_config`, которых служба не нашла: claude-cli убирает их
    /// из унаследованного окружения, чтобы действовало значение по умолчанию.
    pub mcp_env_missing: Vec<String>,
    pub max_turns: Option<u32>,
    pub extra_args: Vec<String>,
    /// Если задан — передать `--resume <id>` в claude headless. Заполняется
    /// runtime'ом либо из parent.session_id (shared-session), либо из последнего
    /// session_id того же агента+варианта за TTL.
    pub resume_session_id: Option<String>,
    /// Если задан И resume_session_id не задан — передать `--session-id <uuid>`
    /// в claude headless. Pre-generated в runtime для корневого claude-cli вызова,
    /// чтобы дочерние invoke сразу видели его в БД и делали --resume (shared-session).
    pub new_session_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    /// Сырой текст ответа модели (если format=json у агента, парсится дальше в runtime).
    pub content: String,
    /// Суммарный input: raw + cache_creation + cache_read. Удобный «общий счёт».
    pub tokens_in: u32,
    pub tokens_out: u32,
    /// Стоимость в долларах; `None` означает, что цена модели неизвестна.
    pub cost_usd: Option<f64>,
    pub finish_reason: String,
    /// Размышления reasoning-модели (`reasoning_content` OpenAI-совместимого API
    /// либо блоки `thinking` Anthropic). None у не-reasoning моделей.
    pub reasoning: Option<String>,
    /// session_id для последующего --resume (только у claude-cli провайдера).
    /// У других провайдеров — None.
    pub session_id: Option<String>,
    /// Детализация input (для диагностики где сидят деньги). Заполняется
    /// claude-cli и Anthropic; у остальных cache-поля могут быть нулевыми.
    pub raw_input_tokens: u32,
    pub cache_creation_input_tokens: u32,
    pub cache_read_input_tokens: u32,
    /// Пошаговый транскрипт прогона (start/turn/tool_result/final/max_turns) —
    /// заполняется прямыми OpenRouter и Anthropic провайдерами при наличии
    /// turn_sink либо `AGENTS_MCP_TRANSCRIPT=1`. Рантайм пишет его в
    /// `agents_mcp.agent_turns` по call_id.
    pub transcript: Vec<Value>,
}

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("timeout провайдера")]
    Timeout,
    #[error("rate limit от провайдера")]
    RateLimited,
    #[error("rate limit от провайдера")]
    RateLimitedRetry(Option<u64>),
    #[error("ошибка провайдера: {0}")]
    Provider(String),
    #[error("невалидный ответ провайдера: {0}")]
    InvalidResponse(String),
    #[error("CLI/subprocess: {0}")]
    Subprocess(String),
    /// Заявленный в `mcp_config` сервер инструментов не ответил (не поднялся
    /// процесс либо упал initialize/tools/list). Вызов отклоняется целиком, до
    /// обращения к модели: агент без заявленных инструментов задачу всё равно
    /// не выполнит, а прогон уже стоил бы денег (случай 12.09.2026 — 40 минут
    /// работы без code-index).
    #[error("{0}")]
    ToolsUnavailable(String),
    /// Ошибка agentic-loop вместе со стоимостью уже завершённых ходов.
    #[error("{error}")]
    WithUsage {
        error: Box<LlmError>,
        tokens_in: u32,
        tokens_out: u32,
        /// Стоимость завершённых ходов; `None` означает неизвестную цену.
        cost: Option<f64>,
    },
    /// Исчерпан лимит tool-use ходов в agentic-цикле. Несёт накопленные токены,
    /// чтобы runtime записал реальную стоимость провала в agent_calls (раньше
    /// ветка ошибки писала нули — стоимость провалов недосчитывалась).
    #[error("ошибка провайдера: {provider}: превышен лимит tool-use turns ({turns}); последний finish_reason={finish_reason}")]
    MaxTurns {
        provider: String,
        turns: u32,
        finish_reason: String,
        tokens_in: u32,
        tokens_out: u32,
        /// Стоимость завершённых ходов; `None` означает неизвестную цену.
        cost: Option<f64>,
        /// Транскрипт частичного (провалившегося) прогона — для разбора, почему
        /// цикл не сошёлся. Пуст, если сбор транскрипта выключен.
        transcript: Vec<Value>,
    },
    /// Ход оборван предохранителем: модель гоняет один абзац по кругу либо
    /// пишет без остановки и до ответа не доходит.
    ///
    /// Отдельной разновидностью, а не строкой, ради `tail` — по хвосту канала
    /// ищется навык на ту тему, где модель встала. Разбирать это из текста
    /// ошибки нельзя: хвост в него не влезает, а обрезанной строки в 120 знаков
    /// для поиска мало — замер 31.08.2026 показал, что по одной строке находится
    /// смежный навык, а по хвосту — тот, что бьёт в корень.
    #[error("ошибка провайдера: {provider}: ход прерван в {channel} — {reason}; генерация прервана на {reasoning_chars} знаках размышления и {content_chars} знаках ответа. Причина обычно в жадной выборке без штрафа за повтор (--temp 0 --top-k 1 при выключенных --repeat-penalty / --dry-multiplier)")]
    Loop {
        /// Все строковые поля этой разновидности — в `Box<str>`: иначе сам
        /// `LlmError` вырастает за порог clippy `result_large_err` (128 байт), и
        /// признак висит на каждой функции, возвращающей `Result<_, LlmError>`.
        /// Тексты ошибок от боксирования не меняются: `Box<str>` и печатается,
        /// и сериализуется как обычная строка.
        provider: Box<str>,
        /// Канал, где случилась петля: «размышлении» или «ответе».
        channel: Box<str>,
        /// Готовая фраза, что именно сработало: повтор строки или объём хода.
        /// Разновидность одна на оба случая — дальше по коду они лечатся
        /// одинаково (навык по хвосту + повтор хода), различать их нужно только
        /// в журнале и в отчёте.
        reason: Box<str>,
        /// Строка, повторившаяся дословно (усечённая — для журнала и отчёта).
        /// При обрыве по объёму — самая частая длинная строка хода.
        line: Box<str>,
        repeats: u32,
        reasoning_chars: usize,
        content_chars: usize,
        /// Хвост оборванного канала — поисковая фраза для навыка.
        tail: Box<str>,
    },
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError>;
}

// ── Сетевой посредник провайдера ───────────────────────────────────────────

/// Адреса, которые НИКОГДА не идут через посредника: петлевые и частные сети.
/// Посредник нужен для выхода наружу; внутренние адреса (серверы инструментов,
/// база, видеокарта) обязаны оставаться прямыми. Клиент провайдера ходит не
/// только к модели, но и к своим MCP-серверам — и по петлевым адресам, и по
/// адресам внутренней сети, — так что без этого списка агент молча остался бы
/// без инструментов.
pub(crate) const PROXY_BYPASS_CORE: &str =
    "127.0.0.1,localhost,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16";

/// Убрать учётные данные из адреса перед записью в журнал: всё до последней `@`
/// заменяется на `***@`. Пароль посредника в файле журнала — утечка.
pub(crate) fn mask_credentials(s: &str) -> String {
    match s.rfind('@') {
        Some(i) => format!("***@{}", &s[i + 1..]),
        None => s.to_string(),
    }
}

/// Разобрать настройку посредника провайдера.
///
/// `Ok(None)` — посредник не задан (поля нет либо оно пустое): поведение
/// остаётся прежним. `Err(текст)` — задан, но негоден; вызывающий пишет
/// предупреждение и собирает клиент без посредника.
///
/// Схему проверяем сами: `Proxy::all` берёт `socks5://…` молча, а возможность
/// socks в сборку не включена — клиент потом обращался бы к socks-порту как к
/// обычному HTTP-посреднику и получал ошибку соединения либо ожидание до предела
/// времени.
pub(crate) fn parse_proxy_setting(
    proxy: Option<&str>,
    bypass: Option<&str>,
) -> Result<Option<reqwest::Proxy>, String> {
    let url = match proxy.map(str::trim) {
        None | Some("") => return Ok(None),
        Some(u) => u,
    };

    let lower = url.to_ascii_lowercase();
    if !(lower.starts_with("http://") || lower.starts_with("https://")) {
        return Err(format!(
            "адрес посредника «{url}»: допустимы только схемы http и https"
        ));
    }

    let proxy = reqwest::Proxy::all(url)
        .map_err(|e| format!("адрес посредника «{url}» не разобран: {e}"))?;

    // Свой список только ДОПОЛНЯЕТ обязательное ядро, заменить его нельзя.
    let bypass = bypass.map(str::trim).unwrap_or("");
    let list = if bypass.is_empty() {
        PROXY_BYPASS_CORE.to_string()
    } else {
        format!("{PROXY_BYPASS_CORE},{bypass}")
    };

    // from_string в reqwest 0.12 всегда отдаёт Some: разбор ленивый и живёт
    // внутри hyper-util. Ошибок здесь не бывает, поэтому ветки на None нет.
    Ok(Some(proxy.no_proxy(reqwest::NoProxy::from_string(&list))))
}

/// Собрать HTTP-клиент провайдера. `provider` нужен только для внятного
/// предупреждения в журнал.
///
/// Кривая строка в конфиге не должна ронять службу, поэтому при негодном адресе
/// клиент собирается БЕЗ нашего посредника — то есть ведёт себя как провайдер
/// без настройки: библиотека сама смотрит переменные окружения (`HTTPS_PROXY` и
/// прочие). Системную настройку посредника Windows она не читает.
pub(crate) fn build_http_client(
    provider: &str,
    proxy: Option<&str>,
    bypass: Option<&str>,
) -> reqwest::Client {
    let mut builder =
        reqwest::Client::builder().user_agent(concat!("agents-mcp/", env!("CARGO_PKG_VERSION")));

    match parse_proxy_setting(proxy, bypass) {
        Ok(Some(p)) => builder = builder.proxy(p),
        Ok(None) => {
            // Список исключений без самого посредника ничего не делает — молчать
            // о такой настройке нельзя, иначе её будут искать часами.
            if bypass.map(str::trim).is_some_and(|b| !b.is_empty()) {
                tracing::warn!(
                    provider = %provider,
                    "proxy_bypass задан, а proxy — нет: список исключений ни на что не влияет"
                );
            }
        }
        Err(_) => tracing::warn!(
            provider = %provider,
            "посредник не применён: адрес недопустим (значение скрыто)"
        ),
    }

    builder.build().expect("reqwest client build")
}

#[cfg(test)]
mod tests {
    use super::*;

    // reqwest::Proxy печатается, но не сравнивается (нет PartialEq), поэтому
    // разбираем итог через matches!, а не assert_eq!.

    #[test]
    fn proxy_not_set_means_direct() {
        assert!(matches!(parse_proxy_setting(None, None), Ok(None)));
    }

    #[test]
    fn empty_proxy_is_not_an_error() {
        assert!(matches!(parse_proxy_setting(Some(""), None), Ok(None)));
        assert!(matches!(parse_proxy_setting(Some("   "), None), Ok(None)));
    }

    #[test]
    fn http_proxy_accepted() {
        assert!(matches!(
            parse_proxy_setting(Some("http://127.0.0.1:9"), None),
            Ok(Some(_))
        ));
    }

    #[test]
    fn spaces_around_address_do_not_break_it() {
        assert!(matches!(
            parse_proxy_setting(Some("  http://127.0.0.1:9  "), None),
            Ok(Some(_))
        ));
    }

    #[test]
    fn garbage_address_rejected() {
        let res = parse_proxy_setting(Some("не адрес вовсе"), None);
        let err = res.expect_err("мусорный адрес должен быть отклонён");
        assert!(!err.is_empty(), "текст ошибки не должен быть пустым");
    }

    #[test]
    fn socks_address_rejected_by_our_own_check() {
        // Сама библиотека такой адрес принимает молча — проверка наша.
        let res = parse_proxy_setting(Some("socks5://127.0.0.1:9"), None);
        let err = res.expect_err("socks-адрес должен быть отклонён");
        assert!(
            err.contains("socks5"),
            "в тексте ошибки нужен сам адрес: {err}"
        );
    }

    #[test]
    fn bypass_core_keeps_own_networks() {
        // Проверка ТЕКСТА константы: куда реально пойдёт запрос, публичный
        // интерфейс библиотеки посмотреть не даёт. Смысл — чтобы ядро исключений
        // нельзя было потерять при будущих правках.
        for addr in ["127.0.0.1", "localhost", "10.0.0.0/8"] {
            assert!(
                PROXY_BYPASS_CORE.contains(addr),
                "{addr} обязан остаться в ядре исключений"
            );
        }
    }

    #[test]
    fn broken_address_does_not_panic_the_client() {
        let _client = build_http_client("тест", Some("не адрес вовсе"), None);
    }

    #[test]
    fn credentials_are_masked_for_the_log() {
        assert_eq!(
            mask_credentials("http://user:pass@proxy:3128"),
            "***@proxy:3128"
        );
        assert_eq!(
            mask_credentials("http://proxy:3128"),
            "http://proxy:3128",
            "адрес без учётных данных не трогаем"
        );
    }
}
