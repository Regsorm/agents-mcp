//! Общие части agentic-loop для прямых HTTP-провайдеров.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use reqwest::Client;
use serde_json::Value;

use super::mcp_client;
use super::{ClaudeCliHints, LlmError};

/// Потолок итераций tool-use, если у агента не задан `max_turns`.
pub(crate) const DEFAULT_MAX_TOOL_TURNS: u32 = 12;
/// Потолок результата инструмента, попадающего в ИСТОРИЮ диалога.
/// История пересылается целиком на каждом ходу, поэтому один крупный ответ
/// дорожает не разово, а умножается на число оставшихся ходов. Замер 30.08.2026
/// (агент-генератор, call_id=8397): `execute_query` вернул 31 318 знаков и поднял
/// запрос с 23 866 до 37 746 токенов — плюс 13 880 к каждому следующему ходу.
/// Полный результат остаётся в транскрипте (`agents_mcp.agent_turns`), модель же
/// получает начало плюс явную пометку об обрезке.
///
/// Порог выбран по замеренным размерам ответов, а не «на глаз»: структура объекта
/// 1С (`get_metadata_structure`) доходит до 16 846 знаков и обязана проходить
/// целиком — это содержательный контекст, обрезка отняла бы у модели реквизиты.
/// Резать надо выборки данных: `execute_query` в том же прогоне вернул 31 318.
pub(crate) const MAX_TOOL_RESULT_IN_HISTORY: usize = 24_000;
/// Сколько одинаковых результатов одного вызова разрешить до блокировки.
pub(crate) const REPEAT_LIMIT: u32 = 3;
const ERROR_HINT_AT: u32 = 2; // со второй — подсказать точную форму вызова
const ERROR_TEMP_AT: u32 = 3; // с третьей — сбить детерминизм температурой
const EMPTY_HINT_AT: u32 = 2; // со второго пустого — сказать сменить признак поиска
const EMPTY_TEMP_AT: u32 = 3; // с третьего — сбить детерминизм, как и на ошибках

/// Инструменты наблюдения не участвуют в защите от повторов: их штатно вызывают
/// многократно, пока состояние фоновой работы или доски задачи не изменится.
/// `task_*` означает только чтение; известные записи исключены ниже.
pub(crate) const OBSERVATION_TOOL_NAMES: &[&str] = &[
    "wait_agent",
    "agent_run",
    "task_*",
    "artifact_read",
    "agent_history",
    "health",
];
const STATE_CHANGING_TASK_TOOLS: &[&str] = &["task_create", "task_set_status"];

/// Частые промахи в языке запросов 1С — по журналу прогонов (301 неудачная
/// проверка на 14.08.2026). Порядок по частоте: `ГРУППИРОВАТЬ` — 39 случаев,
/// одинарные кавычки в литералах — около 30, дальше кальки с SQL. Список
/// пополнять по данным `agents_mcp.agent_turns`, а не по памяти.
const QUERY_SYNTAX_HINT: &str = "\n\
    Чаще всего в наших прогонах ломалось это (слева — как пишут по ошибке, справа — как надо):\n\
    - ГРУППИРОВАТЬ ПО → СГРУППИРОВАТЬ ПО\n\
    - 'строка' → \"строка\": строковые литералы только в ДВОЙНЫХ кавычках\n\
    - WHERE → ГДЕ; JOIN → СОЕДИНЕНИЕ; ORDER BY / ПОРЯДОК → УПОРЯДОЧИТЬ ПО\n\
    - HAVING / ИМЕЮЩИЙ → ИМЕЮЩИЕ; LIMIT / ЛИМИТ → ВЫБРАТЬ ПЕРВЫЕ N\n\
    - LIKE / СОДЕРЖИТ → ПОДОБНО \"%текст%\"\n\
    - COUNT(DISTINCT ...) / СЧЕТ(ПО ...) → КОЛИЧЕСТВО(РАЗЛИЧНЫЕ ...)\n\
    - реквизиты шапки из табличной части — только через .Ссылка. (Товары.Ссылка.Дата)\n\
    Своего случая тут нет — проверь имена полей через get_metadata_structure, а не подбирай.";

fn tool_name_without_server_prefix(tool_name: &str) -> &str {
    tool_name
        .strip_prefix("mcp__")
        .and_then(|name| name.rsplit_once("__").map(|(_, short)| short))
        .unwrap_or(tool_name)
}

/// Совпадает ли имя инструмента с точным именем либо записью «весь сервер».
pub(crate) fn tool_pattern_matches(pattern: &str, full_name: &str) -> bool {
    if pattern == full_name {
        return true;
    }
    let Some((server_name, _)) = full_name.rsplit_once("__") else {
        return false;
    };
    pattern == server_name || pattern.strip_suffix("__*") == Some(server_name)
}

pub(crate) fn tool_list_matches(patterns: &[String], full_name: &str) -> bool {
    patterns
        .iter()
        .any(|pattern| tool_pattern_matches(pattern, full_name))
}

pub(crate) fn is_observation_tool(tool_name: &str, args: &Value) -> bool {
    let tool_name = tool_name_without_server_prefix(tool_name);
    if tool_name == "agent_run" {
        return args.as_object().is_some_and(|args| {
            args.len() == 1
                && args
                    .get("call_id")
                    .is_some_and(|call_id| !call_id.is_null())
        });
    }

    OBSERVATION_TOOL_NAMES.iter().any(|candidate| {
        *candidate == tool_name
            || (*candidate == "task_*"
                && tool_name.starts_with("task_")
                && !STATE_CHANGING_TASK_TOOLS.contains(&tool_name))
    })
}

pub(crate) fn tool_result_hash(result: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    result.hash(&mut hasher);
    hasher.finish()
}

/// Собирать ли пошаговый транскрипт прогона (start/turn/tool_result/final/
/// max_turns). Включается переменной окружения `AGENTS_MCP_TRANSCRIPT=1`;
/// иначе провайдер не копит записи (пустой `transcript` в ответе) — рантайм
/// ничего не пишет в PG, поведение прежнее.
fn transcript_enabled() -> bool {
    matches!(
        std::env::var("AGENTS_MCP_TRANSCRIPT").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// Ин-мемори сборщик пошагового транскрипта прогона. Каждая запись — JSON
/// (`event` + поля хода); в конце `take()` отдаёт их рантайму, который кладёт в
/// `agents_mcp.agent_turns` по call_id (жёсткая привязка к `agent_calls`).
/// Диском не пользуемся — вся аналитика в PostgreSQL через SQL. Выключен → все
/// методы no-op.
pub(crate) struct Transcript {
    records: RefCell<Vec<Value>>,
    enabled: bool,
    /// Канал в PG-писатель: ход уходит в базу сразу, не дожидаясь конца прогона.
    sink: Option<tokio::sync::mpsc::UnboundedSender<Value>>,
}

impl Transcript {
    pub(crate) fn open(sink: Option<tokio::sync::mpsc::UnboundedSender<Value>>) -> Self {
        Self {
            records: RefCell::new(Vec::new()),
            // Канал от рантайма включает сбор сам по себе: раз его дали —
            // ходы нужны в базе, и ждать отдельной переменной окружения незачем.
            enabled: transcript_enabled() || sink.is_some(),
            sink,
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    /// Добавить запись хода. Проставляет порядковый `seq` (для сортировки в SQL)
    /// и метку времени `ts_ms`: без неё все ходы прогона получали один и тот же
    /// `created_at` пакетной вставки, и понять, что именно было долгим — модель
    /// или инструмент — было нельзя.
    pub(crate) fn write(&self, record: &Value) {
        if !self.enabled {
            return;
        }
        let mut rec = record.clone();
        let seq = self.records.borrow().len();
        if let Value::Object(m) = &mut rec {
            m.insert("seq".to_string(), Value::Number(seq.into()));
            m.insert(
                "ts_ms".to_string(),
                Value::Number(chrono::Utc::now().timestamp_millis().into()),
            );
        }
        if let Some(sink) = &self.sink {
            // Канал не ограничен по размеру и не ждёт получателя, поэтому отправка
            // не может задержать сам прогон. Обрыв канала (писатель умер) — не
            // повод валить генерацию: диагностика важна, но не важнее результата.
            let _ = sink.send(rec.clone());
        }
        self.records.borrow_mut().push(rec);
    }

    /// Забрать накопленные записи (для передачи рантайму).
    pub(crate) fn take(&self) -> Vec<Value> {
        std::mem::take(&mut self.records.borrow_mut())
    }
}

/// Усечь строку до `max` символов для записи в транскрипт (результаты
/// инструментов бывают на сотни КБ — метадампы). Возвращает owned-строку.
pub(crate) fn truncate_str(s: &str, max: usize) -> String {
    let total = s.chars().count();
    if total <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max).collect();
    format!("{head}…[+{} символов]", total - max)
}

/// Обрезать результат инструмента перед укладкой в историю диалога.
/// Пометка обязательна: без неё модель считает усечённую выборку полной и
/// делает выводы по обрубку, не зная, что часть строк не показана.
pub(crate) fn clamp_tool_result(result: &str) -> String {
    let total = result.chars().count();
    if total <= MAX_TOOL_RESULT_IN_HISTORY {
        return result.to_string();
    }
    let head: String = result.chars().take(MAX_TOOL_RESULT_IN_HISTORY).collect();
    format!(
        "{head}\n\n[показано {shown} знаков из {total}: результат обрезан, чтобы не \
         раздувать историю диалога. Если нужного в показанной части нет — сузь запрос \
         (отбор, меньше полей, ПЕРВЫЕ N), а не повторяй тот же вызов.]",
        shown = MAX_TOOL_RESULT_IN_HISTORY,
    )
}

/// Имя, которого нет в конфигурации, из сообщения платформы 1С.
///
/// «Таблица не найдена \"РегистрСведений.Нет\"» → `РегистрСведений.Нет`. Нужно,
/// чтобы вернуть модели адресную подсказку: она подбирает написание вслепую,
/// хотя платформа прямым текстом называет промах.
pub(crate) fn missing_name(result: &str) -> Option<String> {
    let at = ["не найдена", "не найдено", "не найден"]
        .iter()
        .filter_map(|p| result.find(p))
        .min()?;
    let tail = &result[at..];
    let start = tail.find('"')? + 1;
    let name = tail[start..].split('"').next()?.trim_matches('\\').trim();
    (!name.is_empty() && name.chars().count() <= 120).then(|| name.to_string())
}

/// Вид ошибки в ответе инструмента либо None, если вызов удался.
///
/// Считаем именно ВИД, а не точный текст: слабая модель коверкает аргументы
/// каждый раз по-новому, и сообщение отличается деталями, оставаясь той же
/// ошибкой по сути. Нормализация до вида делает повтор видимым.
pub(crate) fn error_kind(result: &str) -> Option<String> {
    let head: String = result.chars().take(400).collect::<String>().to_lowercase();
    let is_error = head.starts_with("ошибка")
        || head.contains("ошибка вызова инструмента")
        // Именно поднятый флаг: в конверте удачного вызова тоже есть слово
        // isError (со значением false), и проверка на подстроку принимала
        // пустой, но успешный результат за ошибку.
        || head.contains("\"iserror\": true")
        || head.contains("\"iserror\":true")
        || head.contains("validation error")
        || head.contains("invalid type")
        || head.contains("missing field")
        || head.contains("unknown field")
        // Отказ проверки запроса 1С: {"valid": false, ...} / {"success": false, ...}.
        // Сам вызов инструмента удался, поэтому ни isError, ни «ошибка вызова»
        // здесь нет — и до 14.08.2026 такой ответ не считался ошибкой вовсе.
        // Из-за этого генератор 28 ходов подряд правил один запрос, а счётчик
        // однотипных отказов молчал.
        || head.contains("\"valid\": false")
        || head.contains("\"success\": false");
    if !is_error {
        return None;
    }
    // Тип ошибки грубыми классами: детали (какое именно поле, какое значение)
    // отбрасываем — иначе снова получим уникальные подписи.
    let kind = if head.contains("не разобраны как json") {
        "разбор аргументов"
    } else if head.contains("missing field") || head.contains("unknown field") {
        "имена параметров"
    } else if head.contains("invalid type") || head.contains("validation error") {
        "типы параметров"
    } else if head.contains("недоступен") {
        "инструмент недоступен"
    } else if head.contains("\"valid\": false") || head.contains("\"success\": false") {
        // Разделяем два разных промаха: выдуманное имя лечится обращением к
        // метаданным, кривой синтаксис — сменой конструкции. Подсказки у них
        // тоже разные.
        if head.contains("не найден") {
            "несуществующее имя"
        } else {
            "текст запроса"
        }
    } else {
        "выполнение"
    };
    Some(kind.to_string())
}

/// Список имён параметров инструмента из его JSON-схемы: `base, metaType, nameMask`.
/// Схему мы сами отдали модели в `tools`, поэтому подсказка всегда точная.
pub(crate) fn tool_schema_hint(tools: &[mcp_client::ToolDef], full_name: &str) -> Option<String> {
    let props = tools
        .iter()
        .find(|tool| tool.full_name == full_name)?
        .parameters
        .get("properties")?
        .as_object()?;
    if props.is_empty() {
        return None;
    }
    Some(props.keys().cloned().collect::<Vec<_>>().join(", "))
}

pub(crate) struct McpTools {
    pub(crate) defs: Vec<mcp_client::ToolDef>,
    pub(crate) registry: HashMap<String, (mcp_client::McpServer, String, mcp_client::McpSession)>,
    pub(crate) stdio_pool: mcp_client::StdioPool,
    secret_values: Vec<String>,
}

impl Default for McpTools {
    fn default() -> Self {
        Self {
            defs: Vec::new(),
            registry: HashMap::new(),
            stdio_pool: mcp_client::StdioPool::default(),
            secret_values: Vec::new(),
        }
    }
}

pub(crate) fn redact_mcp_secrets(text: &str, secret_values: &[String]) -> String {
    // Длинные значения раньше коротких: короткое, совпавшее с началом длинного,
    // иначе оставило бы в тексте хвост длинного секрета.
    let mut values: Vec<&String> = secret_values.iter().filter(|value| !value.is_empty()).collect();
    values.sort_by_key(|value| std::cmp::Reverse(value.len()));
    values
        .into_iter()
        .fold(text.to_string(), |safe, value| safe.replace(value.as_str(), "<секрет>"))
}

pub(crate) async fn build_mcp_tools(
    client: &Client,
    hints: &ClaudeCliHints,
) -> Result<McpTools, LlmError> {
    // Серверы, запускаемые процессом: их процессы живут в этом наборе и
    // гаснут вместе с ним, когда вызов агента закончится.
    let mut result = McpTools::default();
    result.secret_values = hints.mcp_env.values().cloned().collect();
    let raw = match &hints.mcp_config {
        Some(raw) => raw,
        None => return Ok(result),
    };
    let allow_all = hints.allowed_tools.is_empty();
    let http_servers = mcp_client::parse_mcp_config(raw).map_err(|e| {
        LlmError::ToolsUnavailable(format!(
            "mcp_config не разобран как JSON: {e}; вызов отклонён, чтобы агент не работал без заявленных инструментов"
        ))
    })?;
    for srv in http_servers {
        let srv = mcp_client::expand_http_server(srv, |name| hints.mcp_env.get(name).cloned())
            .map_err(LlmError::ToolsUnavailable)?;
        let session = mcp_client::initialize_session(client, &srv)
            .await
            .map_err(|e| {
                LlmError::ToolsUnavailable(format!(
                    "сервер инструментов '{}' по адресу {} не прошёл initialize: {}; вызов отклонён, чтобы агент не работал без заявленных инструментов",
                    srv.alias,
                    redact_mcp_secrets(&mcp_client::safe_server_address(&srv.url), &result.secret_values),
                    redact_mcp_secrets(&e, &result.secret_values)
                ))
            })?;
        match mcp_client::list_tools(client, &srv, &session).await {
            Ok(defs) => {
                for def in defs {
                    if (!allow_all && !tool_list_matches(&hints.allowed_tools, &def.full_name))
                        || tool_list_matches(&hints.disallowed_tools, &def.full_name)
                    {
                        continue;
                    }
                    result.registry.insert(
                        def.full_name.clone(),
                        (srv.clone(), def.tool_name.clone(), session.clone()),
                    );
                    result.defs.push(def);
                }
            }
            Err(e) => {
                // Сервер заявлен в mcp_config агента, а инструментов не дал:
                // отказываемся от вызова целиком — до обращения к модели.
                return Err(LlmError::ToolsUnavailable(format!(
                    "сервер инструментов '{}' по адресу {} не ответил: {}; вызов отклонён, чтобы агент не работал без заявленных инструментов",
                    srv.alias,
                    redact_mcp_secrets(&mcp_client::safe_server_address(&srv.url), &result.secret_values),
                    redact_mcp_secrets(&e, &result.secret_values)
                )));
            }
        }
    }

    // Второй вид серверов — запускаемые процессом (в mcp_config у них
    // `command` вместо `url`). Свой процесс на каждый вызов агента: чужого
    // состояния между вызовами не остаётся, а лишний процесс не переживёт
    // конец вызова.
    let stdio_servers = mcp_client::parse_stdio_servers(raw).map_err(|e| {
        LlmError::ToolsUnavailable(format!(
            "mcp_config не разобран как JSON: {e}; вызов отклонён, чтобы агент не работал без заявленных инструментов"
        ))
    })?;
    for scfg in stdio_servers {
        let scfg = mcp_client::expand_stdio_server(scfg, |name| hints.mcp_env.get(name).cloned())
            .map_err(LlmError::ToolsUnavailable)?;
        let alias = scfg.alias.clone();
        let default_cwd = hints.cwd.as_deref().and_then(|p| p.to_str());
        let mut srv = match mcp_client::StdioServer::spawn(&scfg, default_cwd).await {
            Ok(server) => server,
            Err(e) => {
                // Сервер-процесс заявлен, но не поднялся — тот же отказ:
                // инструментов у агента не будет.
                return Err(LlmError::ToolsUnavailable(format!(
                    "сервер инструментов '{}' (команда {}) не поднялся: {}; вызов отклонён, чтобы агент не работал без заявленных инструментов",
                    alias,
                    redact_mcp_secrets(&scfg.command, &result.secret_values),
                    redact_mcp_secrets(&e, &result.secret_values)
                )));
            }
        };
        match srv.list_tools().await {
            Ok(defs) => {
                let mut taken = 0;
                for def in defs {
                    if (!allow_all && !tool_list_matches(&hints.allowed_tools, &def.full_name))
                        || tool_list_matches(&hints.disallowed_tools, &def.full_name)
                    {
                        continue;
                    }
                    result.stdio_pool.add_tool(
                        def.full_name.clone(),
                        alias.clone(),
                        def.tool_name.clone(),
                    );
                    result.defs.push(def);
                    taken += 1;
                }
                // Ни один инструмент не прошёл белый список — держать
                // процесс незачем, он гаснет вместе с `srv`.
                if taken > 0 {
                    result.stdio_pool.add_server(srv);
                }
            }
            Err(e) => {
                return Err(LlmError::ToolsUnavailable(format!(
                    "сервер инструментов '{}' (команда {}) не ответил: tools/list: {}; вызов отклонён, чтобы агент не работал без заявленных инструментов",
                    alias,
                    redact_mcp_secrets(&scfg.command, &result.secret_values),
                    redact_mcp_secrets(&e, &result.secret_values)
                )));
            }
        }
    }
    if !allow_all && result.defs.is_empty() {
        return Err(LlmError::ToolsUnavailable(format!(
            "ни один инструмент из allowed_tools не найден или запрещён: {}; вызов отклонён, чтобы агент не работал без заявленных инструментов",
            hints.allowed_tools.join(", ")
        )));
    }
    Ok(result)
}

#[derive(Default)]
pub(crate) struct ToolCallState {
    // Детектор зацикливания: для каждой подписи вызова (имя + аргументы)
    // храним хеш последнего результата и число одинаковых результатов.
    // Новый результат означает прогресс и сбрасывает счётчик. Инструменты
    // наблюдения сюда не попадают вовсе: повторный опрос для них штатен.
    call_repeats: HashMap<String, (u64, u32)>,
    // Счётчик ОДНОТИПНЫХ ОШИБОК инструмента. Детектор выше считает подпись
    // «имя + аргументы», а слабая модель коверкает аргументы каждый раз
    // по-новому (`"base"` в лишних кавычках, `\base`, обрывок промпта в
    // имени ключа) — подписи разные, счётчик не копится, защита молчит.
    // Поймано 2026-07-23: 30 ходов подряд один и тот же отказ по параметрам.
    // Здесь считаем по ВИДУ ошибки, что бы ни было в аргументах.
    error_counts: HashMap<String, u32>,
    // Счётчик ПУСТЫХ, но успешных ответов по инструменту. Отдельно от ошибок:
    // форма вызова верна, чинить в ней нечего — модель просто ищет то, чего
    // нет. Поймано 14.08.2026: планировщик 12 ходов перебирал написания
    // маски («соглаш», «Соглашени», «индив»…), получая пустой список, и
    // упёрся в лимит ходов, так и не сменив тип объекта.
    empty_counts: HashMap<String, u32>,
}

pub(crate) async fn execute_tool_call(
    client: &Client,
    provider: &str,
    tools: &mut McpTools,
    hints: Option<&ClaudeCliHints>,
    state: &mut ToolCallState,
    tool_name: &str,
    raw_arguments: &str,
    parsed_arguments: Result<Value, String>,
    temperature: &mut f32,
) -> String {
    let forbidden =
        hints.is_some_and(|hints| tool_list_matches(&hints.disallowed_tools, tool_name));
    let (args, args_err) = match parsed_arguments {
        Ok(value) => (value, None),
        Err(error) => (serde_json::json!({}), Some(error)),
    };
    // Подпись включает исходные аргументы: разные запросы не должны
    // влиять друг на друга. Блокируем только после трёх уже
    // проверенных одинаковых результатов этой подписи.
    let sig = format!("{tool_name}::{raw_arguments}");
    let observation = is_observation_tool(tool_name, &args);
    let repeated_results = state
        .call_repeats
        .get(&sig)
        .map(|(_, count)| *count)
        .unwrap_or(0);
    let blocked = !observation && repeated_results >= REPEAT_LIMIT;

    let secret_values = tools.secret_values.clone();
    let mut result = if forbidden {
        tracing::warn!(
            provider,
            tool = tool_name,
            "вызов инструмента заблокирован списком disallowed_tools"
        );
        format!("ОШИБКА: инструмент {tool_name} запрещён настройкой disallowed_tools и НЕ выполнен")
    } else if let Some(err) = &args_err {
        // Звать инструмент с пустыми аргументами бессмысленно: он
        // ответит про нехватку параметров, и модель начнёт чинить не
        // то. Называем настоящую причину — испорченное экранирование.
        tracing::warn!(provider, tool = tool_name, error = %err, "аргументы вызова не разобраны — инструмент не выполнен");
        format!(
            "ОШИБКА: аргументы вызова `{tool_name}` не разобраны как JSON: {err}. Инструмент \
             НЕ выполнен. Причина — испорченное экранирование внутри текстового \
             значения. Повтори вызов: двойные кавычки внутри текста экранируй как \
             \\\", переносы строк — как \\n, обратный слэш — как \\\\. Если значение \
             очень большое, передавай его частями."
        )
    } else if blocked {
        // Жёсткая блокировка: инструмент не исполняем вовсе. Мягкое
        // предупреждение слабые модели игнорируют и продолжают жечь
        // ходы на том же вызове — механический отказ не обойти.
        tracing::warn!(
            provider,
            tool = tool_name,
            count = repeated_results,
            "зацикливание: вызов заблокирован"
        );
        format!(
            "ЗАБЛОКИРОВАНО: три предыдущих выполнения `{tool_name}` с этими же аргументами \
             дали одинаковый результат. Инструмент НЕ выполнен защитой от \
             зацикливания. СМЕНИ ПОДХОД: проверь \
             структуру объекта через get_metadata_structure, упрости запрос (убери \
             проблемное поле/соединение), либо в языке запросов 1С строковые литералы \
             пиши в ДВОЙНЫХ кавычках (\"текст\"), а не одинарных. Повторять этот же \
             вызов бесполезно — он будет заблокирован снова."
        )
    } else {
        match tools.registry.get(tool_name) {
            Some((server, short_name, session)) => mcp_client::call_tool(
                client,
                server,
                short_name,
                args,
                session,
            )
            .await
            .unwrap_or_else(|e| {
                let safe_error = redact_mcp_secrets(&e, &secret_values);
                // Молчащий сервис — это отказ инфраструктуры, а не
                // ошибка модели: сигналим отдельно и погромче, иначе
                // он теряется среди обычных отказов инструментов.
                if e.contains("не ответил за") {
                    let safe_url = redact_mcp_secrets(
                        &mcp_client::safe_server_address(&server.url),
                        &secret_values,
                    );
                    tracing::error!(provider, tool = tool_name, url = %safe_url, error = %safe_error, "MCP-сервис не отвечает — вызов брошен по таймауту");
                } else {
                    tracing::warn!(provider, tool = tool_name, error = %safe_error, "вызов инструмента не удался");
                }
                format!("ОШИБКА вызова инструмента: {safe_error}")
            }),
            // Инструмент сервера, запущенного процессом: разговор
            // идёт по его стандартному вводу/выводу, набор процессов
            // живёт весь вызов агента.
            None if tools.stdio_pool.has(tool_name) => tools
                .stdio_pool
                .call(tool_name, args)
                .await
                .unwrap_or_else(|e| {
                    let safe_error = redact_mcp_secrets(&e, &secret_values);
                    tracing::warn!(provider, tool = tool_name, error = %safe_error, "вызов инструмента сервера-процесса не удался");
                    format!("ОШИБКА вызова инструмента: {safe_error}")
                }),
            None => format!(
                "ОШИБКА: инструмент {tool_name} недоступен (нет в mcp_config/allowed_tools)"
            ),
        }
    };

    // Сравниваем исходный текст ответа инструмента до добавления
    // наших подсказок: они не являются частью результата сервера.
    let repeated_results = if observation || blocked || forbidden {
        0
    } else {
        let hash = tool_result_hash(&result);
        let entry = state.call_repeats.entry(sig).or_insert((hash, 0));
        if entry.0 == hash {
            entry.1 += 1;
        } else {
            *entry = (hash, 1);
        }
        entry.1
    };
    // После третьего подтверждённого одинакового результата
    // предупреждаем, что следующий такой вызов уже не пройдёт.
    if repeated_results == REPEAT_LIMIT {
        tracing::warn!(
            provider,
            tool = tool_name,
            count = repeated_results,
            "детектор зацикливания сработал"
        );
        result.push_str(&format!(
            "\n\n⚠️ ВНИМАНИЕ: ты вызвал `{tool_name}` с теми же аргументами {repeated_results} раз и \
             получаешь тот же результат. Повторять бессмысленно — ошибка в замысле, \
             не в мелочи. СМЕНИ ПОДХОД: проверь структуру объекта через \
             get_metadata_structure, упрости запрос (убери проблемное поле/соединение), \
             либо в языке запросов 1С строковые литералы пиши в ДВОЙНЫХ кавычках \
             (\"текст\"), а не одинарных. Следующий такой же вызов будет ЗАБЛОКИРОВАН."
        ));
    }
    // Однотипный отказ подряд: подсказываем точную форму вызова, а
    // затем сбиваем детерминизм — иначе модель повторяет тот же
    // испорченный вызов до конца лимита ходов.
    if let Some(kind) = error_kind(&result) {
        let key = format!("{tool_name}::{kind}");
        let n = {
            let count = state.error_counts.entry(key).or_insert(0);
            *count += 1;
            *count
        };
        if n >= ERROR_HINT_AT {
            tracing::warn!(provider, tool = tool_name, count = n, kind = %kind, "однотипная ошибка инструмента");
            // Отказ проверки запроса лечится не формой вызова, а текстом
            // запроса — подсказка про имена параметров тут увела бы не туда.
            let hint = if kind == "несуществующее имя" {
                format!(
                    "\n\n⚠️ Такого имени в конфигурации НЕТ{}. Подбирать написание \
                     бессмысленно — в запрос разрешено писать только те имена, что \
                     пришли тебе в ОТВЕТЕ ИНСТРУМЕНТА в этом же диалоге. Возьми \
                     структуру объекта (get_metadata_structure или \
                     get_object_profile) и используй имена оттуда. Не знаешь, какой \
                     объект нужен — сперва найди его (list_metadata_objects), потом \
                     запрашивай структуру, и только потом пиши запрос.",
                    missing_name(&result)
                        .map(|name| format!(": {name}"))
                        .unwrap_or_default()
                )
            } else if kind == "текст запроса" {
                format!(
                    "\n\n⚠️ Запрос не проходит проверку {n} раз подряд. Ошибка в \
                     САМОМ ТЕКСТЕ запроса, а не в форме вызова: переставлять \
                     скобки бесполезно — меняй конструкцию.{QUERY_SYNTAX_HINT}"
                )
            } else {
                format!(
                    "\n\n⚠️ Этот инструмент отвечает одной и той же ошибкой {n} раз \
                     подряд, а ты повторяешь вызов почти без изменений. Дело не в \
                     значениях, а в ФОРМЕ вызова.{}\n\
                     Имена параметров пишутся как есть: без кавычек внутри имени, без \
                     обратных слэшей, без посторонних символов и переносов строк.",
                    tool_schema_hint(&tools.defs, tool_name)
                        .map(|schema| format!(" Требуемые параметры: {schema}."))
                        .unwrap_or_default()
                )
            };
            result.push_str(&hint);
        }
        if n >= ERROR_TEMP_AT && *temperature < 0.1 {
            tracing::warn!(
                provider,
                tool = tool_name,
                count = n,
                "поднимаю температуру следующего хода, чтобы выйти из повтора"
            );
            // Ровно столько, чтобы сойти с детерминированной колеи.
            // Больше — у слабой модели растёт разброс: смешение
            // алфавитов, опечатки в именах (поймано на Gemma-31B).
            *temperature = 0.1;
        }
    }

    // Пустой результат — не ошибка, но топтание такое же. Считаем
    // отдельно и подсказываем по существу: менять надо не форму
    // вызова, а сам признак поиска.
    if result.starts_with(mcp_client::EMPTY_RESULT_MARK) {
        let n = {
            let count = state.empty_counts.entry(tool_name.to_string()).or_insert(0);
            *count += 1;
            *count
        };
        if n >= EMPTY_HINT_AT {
            tracing::warn!(
                provider,
                tool = tool_name,
                count = n,
                "инструмент подряд возвращает пусто"
            );
            result.push_str(&format!(
                "\n\n⚠️ `{tool_name}` вернул пусто {n} раз подряд. Форма вызова верна — \
                 значит объекта с такими признаками НЕТ. Перебирать написания \
                 бессмысленно: смени САМ признак поиска (другой тип объекта, \
                 более короткий корень слова, другой инструмент) либо считай, \
                 что объекта нет, и продолжай работу без него."
            ));
        }
        if n >= EMPTY_TEMP_AT && *temperature < 0.1 {
            *temperature = 0.1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn secret_redaction_replaces_longer_values_first() {
        let secrets = vec!["abc".to_string(), "abcdef".to_string()];
        assert_eq!(redact_mcp_secrets("x=abcdef y=abc", &secrets), "x=<секрет> y=<секрет>");
    }

    #[test]
    fn server_wide_tool_patterns_match_both_supported_forms() {
        assert!(tool_pattern_matches("mcp__fixture", "mcp__fixture__poll"));
        assert!(tool_pattern_matches(
            "mcp__fixture__*",
            "mcp__fixture__poll"
        ));
        assert!(tool_pattern_matches(
            "mcp__fixture__poll",
            "mcp__fixture__poll"
        ));
        assert!(!tool_pattern_matches("mcp__other", "mcp__fixture__poll"));
    }

    #[test]
    fn observation_tools_are_matched_without_server_alias() {
        assert!(is_observation_tool(
            "mcp__any-alias__wait_agent",
            &json!({"call_id": 7, "wait_sec": 0})
        ));
        assert!(is_observation_tool(
            "mcp__any-alias__agent_run",
            &json!({"call_id": 7})
        ));
        assert!(is_observation_tool("mcp__any-alias__task_list", &json!({})));
        assert!(is_observation_tool(
            "mcp__any-alias__artifact_read",
            &json!({"task_id": 1})
        ));
        assert!(!is_observation_tool(
            "mcp__any-alias__agent_run",
            &json!({"agent": "worker"})
        ));
        assert!(!is_observation_tool(
            "mcp__any-alias__task_create",
            &json!({})
        ));
        assert!(!is_observation_tool(
            "mcp__any-alias__artifact_write",
            &json!({})
        ));
        assert!(!is_observation_tool(
            "mcp__any-alias__fs_write_file",
            &json!({})
        ));
        assert!(!is_observation_tool(
            "mcp__any-alias__fs_edit_file",
            &json!({})
        ));
    }

    #[test]
    fn arguments_parse_failure_has_own_error_kind() {
        // Свой класс ошибки: подсказка про экранирование не должна смешиваться
        // с ошибками про имена и типы параметров.
        let msg = "ОШИБКА: аргументы вызова `bsl_analyze_text` не разобраны как JSON: \
                   EOF while parsing a string. Инструмент НЕ выполнен.";
        assert_eq!(error_kind(msg).as_deref(), Some("разбор аргументов"));
        assert_eq!(
            error_kind("ОШИБКА: missing field `base`").as_deref(),
            Some("имена параметров")
        );
        assert_eq!(error_kind("Результат: 42 строки"), None);
    }

    #[test]
    fn successful_empty_result_is_not_an_error() {
        // Дословный конверт из прогона 2026-08-14 (call 5487, планировщик на
        // Qwen3.8): инструмент отработал, но ничего не нашёл. Проверка на
        // подстроку «isError» считала такой ответ отказом, и со второго раза
        // модель получала подсказку «дело в ФОРМЕ вызова» — после чего
        // 14 ходов чинила исправный вызов вместо поиска объекта.
        let ok_empty = r#"{"content":[{"text":"","type":"text"}],"isError":false}"#;
        assert_eq!(error_kind(ok_empty), None);
        // Поднятый флаг по-прежнему ошибка.
        let failed = r#"{"content":[{"text":"нет такого объекта"}],"isError":true}"#;
        assert_eq!(error_kind(failed).as_deref(), Some("выполнение"));
    }

    #[test]
    fn rejected_query_has_own_error_kind() {
        // Дословный ответ validate_query из прогона 2026-08-14 (call 5497):
        // генератор 28 ходов правил один запрос, а счётчик молчал — в ответе нет
        // ни isError, ни «ошибка вызова инструмента», сам вызов ведь удался.
        let rejected = "{\r\n\"valid\": false,\r\n\"message\": \"{Конфигурация \
            Обработка.ВыполнениеЗапросов.МодульМенеджера(55)}: Ошибка при вызове метода \
            контекста (УстановитьТекстЗапроса): {(5, 10)}: Ожидается выражение\"\r\n}";
        assert_eq!(error_kind(rejected).as_deref(), Some("текст запроса"));
        // execute_query отдаёт success вместо valid — тот же класс.
        let failed_exec = "{\r\n\"success\": false,\r\n\"rowCount\": 0,\r\n\"message\": \
            \"Синтаксическая ошибка\"\r\n}";
        assert_eq!(error_kind(failed_exec).as_deref(), Some("текст запроса"));
        // Удачная проверка ошибкой не считается.
        assert_eq!(error_kind("{\r\n\"valid\": true\r\n}"), None);
    }

    #[test]
    fn invented_name_is_recognized_and_extracted() {
        // Дословно из прогона 2026-08-14 (call 5582): генератор выдумал таблицу
        // и подбирал написание вслепую, хотя платформа прямо назвала промах.
        let msg = "{\r\n\"valid\": false,\r\n\"message\": \"{Конфигурация \
            Обработка.ВыполнениеЗапросов.МодульМенеджера(55)}: Ошибка при вызове метода \
            контекста (УстановитьТекстЗапроса): {(16, 20)}: Таблица не найдена \
            \\\"РегистрСведений.СкидкиТорговыхПредложений.СрезПоследних\\\"\"\r\n}";
        assert_eq!(error_kind(msg).as_deref(), Some("несуществующее имя"));
        assert_eq!(
            missing_name(msg).as_deref(),
            Some("РегистрСведений.СкидкиТорговыхПредложений.СрезПоследних")
        );
        // Кривой синтаксис остаётся отдельным классом.
        let syntax = "{\r\n\"valid\": false,\r\n\"message\": \"{(12, 1)}: Синтаксическая ошибка \
            \\\"ГРУППИРОВАТЬ\\\"\"\r\n}";
        assert_eq!(error_kind(syntax).as_deref(), Some("текст запроса"));
        assert_eq!(missing_name("обычный ответ без ошибок"), None);
    }
}
