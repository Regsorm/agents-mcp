//! Доставка навыков под-агенту через внешний MCP-сервис библиотеки навыков.
//!
//! Два слоя прогрессивного раскрытия, оба через ГОТОВЫЕ инструменты этого
//! сервиса (не через сырой SQL — сервис сам эмбеддит запрос и отдаёт чистый
//! текст):
//!   - push-индекс: `skill_search(бриф)` → семантический top-k «имя + описание»
//!     в {{ skills_index }}. Локальная модель получает 3–5 релевантных рецептов,
//!     а не весь домен (масштабируется на тысячи навыков в базе).
//!   - тело по требованию: `skill_load(name)` → полный текст SKILL.md.
//!
//! Транспорт — POST tools/call на этот сервис (stateless Streamable HTTP).

use serde_json::{json, Map, Value};
use std::time::Duration;

use crate::providers::mcp_client::extract_envelope;

/// Clone нужен, чтобы отдать клиента провайдеру в `LlmRequest`: петлю он ловит
/// посреди агентного цикла и ищет навык по тому месту, где модель встала, а не
/// по фразе задания. Внутри `reqwest::Client` уже разделяемый — копия дешёвая.
#[derive(Clone, Debug)]
pub struct SkillsClient {
    client: reqwest::Client,
    url: Option<String>,
}

impl SkillsClient {
    pub fn new(url: Option<String>) -> Self {
        Self::with_limits(url, Duration::from_secs(5), Duration::from_secs(30))
    }

    fn with_limits(url: Option<String>, connect: Duration, request: Duration) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(connect)
            .timeout(request)
            .build()
            .expect("построение HTTP-клиента навыков");
        Self { client, url }
    }

    #[cfg(test)]
    pub(crate) fn with_test_timeouts(url: String, request: Duration) -> Self {
        Self::with_limits(Some(url), request, request)
    }

    pub fn enabled(&self) -> bool {
        self.url.is_some()
    }

    pub fn address(&self) -> &str {
        self.url.as_deref().unwrap_or("не задан")
    }

    /// Один stateless tools/call к сервису навыков → result.content[0].text.
    /// Внешний JSON-конверт выбираем по `id` из plain JSON или потока SSE;
    /// внутренний text отдаём как есть.
    async fn call_tool(&self, tool: &str, args: Value) -> Result<String, String> {
        let url = self
            .url
            .as_deref()
            .ok_or_else(|| "skills: rag_query_url не задан".to_string())?;
        let req = json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool, "arguments": args}
        });
        let resp = self
            .client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .json(&req)
            .send()
            .await
            .map_err(|e| format!("{tool} send: {e}"))?;
        let text = resp.text().await.map_err(|e| format!("{tool} body: {e}"))?;
        let v = extract_envelope(&text, 1).map_err(|e| format!("{tool} {e}"))?;
        v.pointer("/result/content/0/text")
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| format!("{tool}: неожиданный ответ: {v}"))
    }

    pub(crate) async fn skill_catalog_result(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<String, String> {
        if self.url.is_none() || query.trim().is_empty() {
            return Ok(String::new());
        }
        self.call_tool("skill_search", json!({ "query": query, "limit": limit }))
            .await
    }

    pub(crate) async fn skill_body_result(&self, name: &str) -> Result<Option<String>, String> {
        if self.url.is_none() {
            return Ok(None);
        }
        self.call_tool("skill_load", json!({ "names": [name] }))
            .await
            .map(|text| (!text.trim().is_empty()).then_some(text))
    }

    /// Семантический push-индекс: по запросу-брифу берём top-k релевантных
    /// навыков через skill_search. Отдаём СЫРОЙ ответ инструмента: в нём есть
    /// оценки cos и rr, нужные для отбора тел (см. `top_names_by_rerank`).
    /// Для промпта его прогоняют через `format_catalog`.
    /// Пусто при выключенном клиенте, пустом запросе или отсутствии совпадений.
    pub async fn skill_catalog(&self, query: &str, limit: usize) -> String {
        if self.url.is_none() || query.trim().is_empty() {
            return String::new();
        }
        // Порог отсечки СВОЙ не задаём — берётся серверный
        // (RAG_QUERY_SKILL_MIN_RERANK, на 31.08.2026 = -8.0).
        //
        // Раньше здесь стояло min_rerank=0.4 из процентной шкалы 0..1, где «у
        // попаданий 0.9+, у шума <0.3». Шкала давно другая: 13.08.2026 реранкер
        // сменили на bge-reranker-v2-m3, он отдаёт сырые логиты примерно
        // -11..+3, и у ВЕРНЫХ находок они лежат в -6.3..-0.1 при медиане -3.9.
        // Порог 0.4 на этой шкале оставлял от каталога одну позицию: замер
        // 31.08.2026 на задании про скидку по группе номенклатуры — 1 навык
        // против 5 без порога, причём отсекались ровно те, которых не хватало
        // генератору. Числовое значение порога привязано к модели реранкера и
        // не переносится между шкалами, поэтому второй копии его здесь быть не
        // должно: тот же рассинхрон клиента и сервера уже стоил системе пяти
        // дней молчания хука 04.08.2026.
        match self.skill_catalog_result(query, limit).await {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!("skill_catalog: {e}");
                String::new()
            }
        }
    }

    /// Полное тело навыка по имени через skill_load. None если не найдено/ошибка.
    pub async fn skill_body(&self, name: &str) -> Option<String> {
        self.url.as_ref()?;
        match self.skill_body_result(name).await {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!("skill_body: {e}");
                None
            }
        }
    }
}

/// Собирает поисковый запрос к skill_search из входа под-агента: строковые
/// значения полей, несущих суть задачи (бриф/цель/объект), в порядке приоритета.
/// Объектно-якорный запрос точнее сырого текста. Пусто → skill_index не ищет.
pub fn build_skill_query(input: &Map<String, Value>) -> String {
    const KEYS: &[&str] = &[
        "extracted_intent",
        "brief",
        "task",
        "goal",
        "question",
        "target",
        "module_kind",
        "user_message",
    ];
    let mut parts: Vec<String> = Vec::new();
    for k in KEYS {
        if let Some(s) = input.get(*k).and_then(|v| v.as_str()) {
            let s = s.trim();
            if !s.is_empty() {
                parts.push(s.to_string());
            }
        }
    }
    // Ограничиваем длину: эмбеддингу нужен сфокусированный запрос, а не весь
    // бриф со всеми деталями.
    parts.join(" · ").chars().take(600).collect()
}

/// Каталог skill_search ("• name (cos=.., scope: ..)\n  description") →
/// компактный индекс "- name — description" (cos/rr/scope из имени убираем).
/// Пусто, если строк-навыков нет (например "Найдено навыков: 0").
pub fn format_catalog(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let Some(rest) = line.trim_start().strip_prefix("• ") else {
            continue;
        };
        // имя — до " (" (перед cos-скобкой); если скобки нет — вся строка.
        let name = rest.split(" (").next().unwrap_or(rest).trim();
        if name.is_empty() {
            continue;
        }
        // описание — следующая непустая строка каталога (в выдаче она с отступом).
        let desc = lines
            .peek()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with("• "))
            .unwrap_or("");
        if desc.is_empty() {
            out.push(format!("- {name}"));
        } else {
            lines.next(); // потребить строку описания
            out.push(format!("- {name} — {desc}"));
        }
    }
    out.join("\n")
}

/// Имена `n` самых релевантных навыков из СЫРОГО каталога skill_search —
/// для вложения их тел в {{ skills_bodies }}.
///
/// Каталог отсортирован по косинусной близости, а она грубее оценки реранкера
/// (`rr`), который сравнивает запрос с навыком внимательнее. Замер 30.08.2026:
/// на запросе «модуль объекта» порядок по cos дал rr = -7.2, -1.4, +2.1 —
/// самый подходящий навык оказался последним; на запросе по сути задания два
/// верных навыка стояли третьим и четвёртым при rr на два с половиной пункта
/// выше, чем у двух первых. Тело уходит провайдеру заново на КАЖДОМ ходу, до
/// шестидесяти раз за задачу, поэтому ошибка отбора стоит дорого — берём по rr.
///
/// Навык без оценки rr уходит в конец: без неё судить не о чем.
pub fn top_names_by_rerank(catalog: &str, n: usize) -> Vec<String> {
    top_scored_by_rerank(catalog, n)
        .into_iter()
        .map(|(name, _)| name)
        .collect()
}

/// То же, но с оценкой реранкера у каждого имени.
///
/// Оценка нужна там, где решают не только «кто первый», но и «годится ли
/// вообще»: при лечении петли навык с низкой оценкой — это не подсказка по
/// теме, а случайный сосед по каталогу.
pub fn top_scored_by_rerank(catalog: &str, n: usize) -> Vec<(String, f64)> {
    let mut rows: Vec<(f64, String)> = Vec::new();
    for line in catalog.lines() {
        let Some(rest) = line.trim_start().strip_prefix("• ") else {
            continue;
        };
        let name = rest.split(" (").next().unwrap_or(rest).trim().to_string();
        if name.is_empty() {
            continue;
        }
        let rr = rest
            .split_once("rr=")
            .and_then(|(_, tail)| {
                let end = tail.find([',', ')']).unwrap_or(tail.len());
                tail[..end].trim().parse::<f64>().ok()
            })
            .unwrap_or(f64::NEG_INFINITY);
        rows.push((rr, name));
    }
    // sort_by с partial_cmp: NEG_INFINITY сравнивается корректно, NaN в оценках
    // не бывает — на нём порядок просто сохранится.
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    rows.into_iter()
        .take(n)
        .map(|(rr, name)| (name, rr))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ответ skill_search на запрос «модуль объекта» (30.08.2026), имена навыков обезличены:
    /// порядок по cos не совпадает с порядком по rr.
    const REAL_CATALOG: &str = "Найдено навыков: 3 (cos≥0.45).\n\n\
• index-freshness-check (cos=0.593, rr=-7.233, scope: project-a)\n  Как поддерживать актуальность индекса.\n\n\
• bsl-object-member-access-runtime-errors (cos=0.574, rr=-1.364, scope: общее)\n  Три рантайм-ошибки доступа к членам объекта.\n\n\
• external-processing-branches (cos=0.565, rr=2.148, scope: Repo1C)\n  Две ветки исполнения внешней обработки.";

    #[test]
    fn top_names_by_rerank_prefers_rr_over_catalog_order() {
        // По порядку каталога первым шёл бы навык про актуальность индекса (rr=-7.2).
        assert_eq!(
            top_names_by_rerank(REAL_CATALOG, 1),
            vec!["external-processing-branches"]
        );
        assert_eq!(
            top_names_by_rerank(REAL_CATALOG, 2)[1],
            "bsl-object-member-access-runtime-errors"
        );
        assert!(top_names_by_rerank("", 2).is_empty());
        assert!(top_names_by_rerank("Найдено навыков: 0 (cos≥0.45).", 2).is_empty());
    }

    #[test]
    fn top_names_by_rerank_puts_unscored_last() {
        let text =
            "• без-оценки\n  описание\n\n• с-оценкой (cos=0.5, rr=-9.0, scope: X)\n  описание";
        assert_eq!(
            top_names_by_rerank(text, 2),
            vec!["с-оценкой", "без-оценки"]
        );
    }

    #[test]
    fn format_catalog_parses_bullets() {
        let text = "Найдено навыков: 2 (cos≥0.45).\n\n\
• 1c-gtd-registry (cos=0.831, rr=1.000, scope: Repo1C)\n  Структура регистров ГТД в УТ 11.\n\n\
• 1c-pagination-strategies (cos=0.50, rr=0.90, scope: Repo1C)\n  Пагинация в запросах 1С.";
        assert_eq!(
            format_catalog(text),
            "- 1c-gtd-registry — Структура регистров ГТД в УТ 11.\n\
- 1c-pagination-strategies — Пагинация в запросах 1С."
        );
    }

    #[test]
    fn format_catalog_empty_when_no_skills() {
        assert_eq!(format_catalog("Найдено навыков: 0 (cos≥0.45)."), "");
        assert_eq!(format_catalog(""), "");
    }

    #[test]
    fn format_catalog_name_without_parens() {
        assert_eq!(
            format_catalog("• skill-x\n  описание X"),
            "- skill-x — описание X"
        );
    }

    #[test]
    fn build_skill_query_anchors_on_objects() {
        let mut m = Map::new();
        m.insert("brief".into(), json!("сделать отчёт по продажам"));
        m.insert("target".into(), json!("ОтчётПродажи"));
        m.insert("base".into(), json!("demo-base")); // не в KEYS → игнорируется
        let q = build_skill_query(&m);
        assert_eq!(q, "сделать отчёт по продажам · ОтчётПродажи");
    }

    #[test]
    fn build_skill_query_empty_when_no_text_fields() {
        let m = Map::new();
        assert_eq!(build_skill_query(&m), "");
    }

    #[test]
    fn malformed_envelope_returns_error_without_panic() {
        assert!(extract_envelope("}broken{", 1).is_err());
    }
}
