//! OpenRouter-провайдер: вызовы chat completion через OpenAI-совместимый API.
//!
//! OpenRouter раздаёт доступ к десяткам моделей (DeepSeek, Qwen, GLM, Llama,
//! Gemini, GPT-*) через единый endpoint. С параметром `usage.include=true` он
//! возвращает реальный `cost` в долларах. Если его нет, стоимость считается по
//! цене модели из секции провайдера главного конфига.
//!
//! Tool-use (веха #3). Если у вызова есть `cli_hints` с непустым
//! `allowed_tools` И заданным `mcp_config` — провайдер исполняет полноценный
//! agentic-loop: собирает `tools` из MCP-серверов (через `mcp_client`),
//! шлёт их модели, получает `tool_calls`, исполняет их сам (модели
//! OpenAI-формата тулы не исполняют — только просят), дописывает результаты
//! и крутит цикл до финального ответа либо `max_turns`. Без hints/mcp_config —
//! поведение прежнее, single-shot (так работают brief-analyst и пробники).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use reqwest::{header, Client};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::warn;

use crate::config::ModelPrice;

use super::mcp_client;
use super::tool_loop::{
    self, clamp_tool_result, execute_tool_call, truncate_str, ToolCallInput, ToolCallState,
    Transcript, DEFAULT_MAX_TOOL_TURNS,
};
use super::{pricing, ClaudeCliHints, LlmError, LlmProvider, LlmRequest, LlmResponse};

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
const STREAM_MODEL_NAME_MAX_LEN: usize = 80;
const STREAM_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

pub struct OpenRouterProvider {
    /// Имя провайдера, под которым он зарегистрирован (`openrouter`, `mimo`,
    /// `deepseek`, …). Возвращается из `name()` — важно, чтобы метрики в
    /// `agent_calls` и health не схлопывали все прямые клиенты в «openrouter».
    name: String,
    client: Client,
    api_key: String,
    base_url: String,
    /// Опциональный HTTP-Referer для OpenRouter analytics.
    referer: Option<String>,
    prices: BTreeMap<String, ModelPrice>,
    /// Окно контекста сервера, спрошенное у него самого (`GET /props`) и
    /// запомненное на время жизни провайдера. Нужно, чтобы порог вытеснения
    /// истории считался от РЕАЛЬНОГО окна, а не от зашитого числа: окно меняется
    /// при перезапуске llama-server с другими ключами (у нас 30.08.2026 оно
    /// выросло с 131072 до 262144). У облачных провайдеров ручки `/props` нет —
    /// там остаётся None, и вытеснение не работает, что и правильно.
    ctx_window: tokio::sync::OnceCell<Option<u32>>,
    semaphore: Option<Arc<tokio::sync::Semaphore>>,
    active: Arc<AtomicUsize>,
}

pub(crate) struct OpenRouterOptions {
    pub name: String,
    pub api_key: String,
    pub base_url: Option<String>,
    pub referer: Option<String>,
    pub proxy: Option<String>,
    pub proxy_bypass: Option<String>,
    pub max_concurrent: Option<u32>,
    pub prices: BTreeMap<String, ModelPrice>,
}

struct ActiveCall(Arc<AtomicUsize>);

impl Drop for ActiveCall {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl OpenRouterProvider {
    pub fn new(options: OpenRouterOptions) -> Self {
        let OpenRouterOptions {
            name,
            api_key,
            base_url,
            referer,
            proxy,
            proxy_bypass,
            max_concurrent,
            prices,
        } = options;
        let client = super::build_http_client(&name, proxy.as_deref(), proxy_bypass.as_deref());
        Self {
            name,
            client,
            api_key,
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.into()),
            referer,
            prices,
            ctx_window: tokio::sync::OnceCell::new(),
            semaphore: max_concurrent
                .map(|n| Arc::new(tokio::sync::Semaphore::new(n.max(1) as usize))),
            active: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Узнать окно контекста у сервера. Спрашиваем один раз, дальше берём
    /// запомненное. Ошибка или отсутствие ручки — не беда: вернём None, и
    /// вытеснение истории просто не включится.
    async fn context_window(&self) -> Option<u32> {
        *self
            .ctx_window
            .get_or_init(|| async {
                let url = format!("{}/props", self.base_url.trim_end_matches("/v1"));
                let resp =
                    tokio::time::timeout(Duration::from_secs(5), self.client.get(&url).send())
                        .await
                        .ok()?
                        .ok()?;
                let body: Value = tokio::time::timeout(Duration::from_secs(5), resp.json())
                    .await
                    .ok()?
                    .ok()?;
                let n = body
                    .pointer("/default_generation_settings/n_ctx")
                    .and_then(|v| v.as_u64())?;
                tracing::info!(provider = %self.name, n_ctx = n, "окно контекста сервера");
                Some(n as u32)
            })
            .await
    }
}

/// Доля окна, после которой начинаем вытеснять из истории старые результаты
/// инструментов. 0.6 — не догма, а точка, до которой замеры далеко: в прогонах
/// 30.08.2026 пик занятости был 61 130 токенов из 262 144, то есть 23%. Порог
/// поставлен выше наблюдаемого пика, чтобы в обычной работе механизм молчал и
/// включался только на нетипично длинных цепочках.
const HISTORY_TRIM_AT_DEFAULT: f64 = 0.6;

/// Порог вытеснения, заданный при старте службы переменной окружения
/// `AGENTS_MCP_HISTORY_TRIM_AT` (доля окна). Ноль или отрицательное значение —
/// вытеснение выключено полностью, история идёт модели как есть.
///
/// Переменная нужна для честного сравнения моделей с разным окном: у модели с
/// вдвое меньшим окном тот же порог в долях срабатывает вдвое раньше, и тогда
/// сравниваются уже не модели, а два разных режима работы с историей.
fn history_trim_at() -> f64 {
    std::env::var("AGENTS_MCP_HISTORY_TRIM_AT")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(HISTORY_TRIM_AT_DEFAULT)
}

/// Вытеснить из истории самые старые результаты инструментов, оставив вместо них
/// пометку. Возвращает число вытесненных сообщений.
///
/// Почему вытесняем результаты инструментов, а не пересказываем историю: агент
/// работает с ТОЧНЫМИ именами (`АналитикаУчетаНоменклатуры`, тексты запросов), и
/// пересказ своими словами их портит — модель потом пишет код по испорченному
/// имени. Старая выборка данных своё уже отработала: по ней модель либо приняла
/// решение, либо нет. Системный запрос, задание и свежие ходы не трогаем.
fn trim_history(messages: &mut [Value], keep_last: usize) -> usize {
    let n = messages.len();
    if n <= keep_last {
        return 0;
    }
    let mut evicted = 0;
    for m in messages.iter_mut().take(n - keep_last) {
        if m["role"].as_str() != Some("tool") {
            continue;
        }
        let len = m["content"]
            .as_str()
            .map(|s| s.chars().count())
            .unwrap_or(0);
        // Пометку ставим один раз: повторный проход не должен «вытеснять»
        // уже вытесненное и раздувать счётчик.
        if len == 0
            || m["content"]
                .as_str()
                .is_some_and(|s| s.starts_with("[вытеснено"))
        {
            continue;
        }
        m["content"] = Value::String(format!(
            "[вытеснено из контекста: результат инструмента на {len} знаков. \
             Он уже отработал на прошлых ходах. Если данные снова нужны — вызови \
             инструмент заново, сузив запрос.]"
        ));
        evicted += 1;
    }
    evicted
}

// ── Request / Response payloads ────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    /// Сообщения собираются как `Value` (system/user/assistant+tool_calls/tool) —
    /// в tool-loop состав динамический, owned-форма проще borrow-структур.
    messages: &'a [Value],
    temperature: f32,
    /// Потолок выдачи. `None` — поле НЕ уходит в запрос, и сервер сам отдаёт
    /// столько, сколько влезает в остаток окна (`n_predict = -1` по умолчанию).
    /// Так задаётся «не ограничивать»: у reasoning-модели мысль и ответ идут в
    /// один поток, и жёсткое число режет ответ, если мысль вышла длинной —
    /// 30.08.2026 агент разбора критериев отдал 4096 токенов рассуждений и пустой ответ.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    /// OpenAI tools (function-calling). None → обычный single-shot.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [Value]>,
    /// OpenRouter-расширение: попросить детальный usage с реальной стоимостью.
    usage: UsageOption,
    /// Потоковая отдача ответа: включается, когда задан каталог для записи
    /// генерации по мере поступления (см. `stream_dir`).
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
    /// В потоковом режиме usage приходит отдельным последним событием только
    /// если попросить явно.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    /// Поля из `[model.extra_body]` конфига агента — уходят в тело как есть,
    /// на верхний уровень. Пустая карта не добавляет ничего.
    #[serde(flatten)]
    extra: &'a serde_json::Map<String, Value>,
}

fn is_false(v: &bool) -> bool {
    !*v
}

/// Ниже какой длины строку не считаем признаком зацикливания: короткие
/// (`КонецЕсли;`, отступы, пустые) законно повторяются десятки раз.
const LOOP_LINE_MIN_CHARS: usize = 40;

/// Сколько дословных повторов ОДНОЙ длинной строки считать петлёй.
/// Замер по двум сорванным прогонам 30.08.2026: максимум повторов одной строки
/// был 23 и 29 раз. Законная перезапись кода в размышлении даёт две-три копии,
/// так что шесть — с запасом в обе стороны.
///
/// Порог берётся только когда до него дошли МИНИМУМ ДВЕ разные строки (см.
/// `note_repeats`): одна строка с разными соседями — код вроде структуры с
/// повторяющимся атрибутом, её ловит лишь `LOOP_TOTAL_LIMIT`.
const LOOP_REPEAT_LIMIT: u32 = 6;

/// Сколько знаков между двумя вхождениями строки ещё считается «подряд».
///
/// Без этого порога детектор считает вхождения по всему тексту и ловит не
/// петлю, а законную работу: 31.08.2026 он оборвал генерацию на строке
/// `Результат = Запрос.Выполнить().Выгрузить();`, которая встретилась шесть раз
/// в ШЕСТИ РАЗНЫХ редакциях функции — между ними было 722, 15 386, 674, 3152 и
/// 8826 знаков рассуждений. Настоящая петля выглядит иначе: блок в 3243 знака
/// шёл 28 раз с постоянным шагом и побайтно совпадающими копиями. Отсюда порог:
/// период настоящей петли — тысячи знаков, разрыв законного повтора — десятки
/// тысяч. Разрыв больше порога сбрасывает счётчик.
const LOOP_GAP_MAX: usize = 5000;

/// Сколько повторов одной длинной строки за ВЕСЬ ход считать петлёй — сколько бы
/// текста между ними ни лежало.
///
/// Второй сторож рядом с `LOOP_REPEAT_LIMIT`: тот считает повторы подряд и
/// намеренно обнуляется на разрыве больше `LOOP_GAP_MAX`, поэтому пропускает
/// медленную петлю. Замер 31.08.2026 по сорванному прогону: ход на 298 727
/// знаков, где строка `- Remove the JOIN with the group.` встретилась 92 раза, а
/// подряд — не больше двух: между вхождениями лежало по 6 300 знаков
/// рассуждений. Порог взят по тому же замеру: на 165 законных ходах максимум
/// повторов одной строки — 7, на двух сорванных прогонах 30.08.2026 — 23 и 29.
///
/// Ниже восьми опускать нельзя — там начинается здоровая работа: порог 3 оборвал
/// бы 30 законных ходов из 165, порог 5 — тринадцать, порог 7 — один. А выигрыш
/// от снижения крошечный: на разносе 31.08.2026 порог 3 сработал бы на 1.9
/// минуте против 3.8 у десятки. Цена ложного обрыва несоизмеримо выше: ход
/// выбрасывается целиком и идёт заново, а после `LOOP_FIX_LIMIT` таких обрывов
/// падает весь прогон.
const LOOP_TOTAL_LIMIT: u32 = 10;

/// Предел объёма одного хода в знаках: и размышление, и ответ вместе.
///
/// Ловит разнос без дословных повторов — когда модель пишет много и всё время
/// разное.
///
/// Порог откалиброван 01.09.2026 по `agents_mcp.agent_turns` — по-ходовым
/// записям в базе, а не по потоковому логу. Лог для этого не годится:
/// параллельные вызовы одной модели пишут в общий файл вперемешку, и прежний
/// замер по нему склеил чужие куски в мнимые ходы на 199 497 и 298 727 знаков.
/// По базе за все 5622 хода самый большой — 100 626 знаков (только размышление,
/// `finish=stop`, то есть ход законный и договорённый), следующий — 79 306.
/// Порог оставляет над потолком запас в четверть.
///
/// Проверенного разноса ПО ОБЪЁМУ в базе нет ни одного: все известные разносы
/// дословные и ловятся счётчиками повторов, а тот, что уходил внутрь одной
/// строки, берёт `LOOP_LINE_CHARS_MAX`. Этот предел — последняя подстраховка на
/// случай, которого пока не видели.
const TURN_CHARS_MAX: usize = 125_000;

/// Предел длины ОДНОЙ строки без перевода — в знаках.
///
/// Четвёртый сторож. Три предыдущих считают по строкам и слепы к петле, которая
/// целиком укладывается внутрь одной строки: для них она — единственное
/// вхождение, повторов ноль. Замер 01.09.2026 по 467 ходам двух моделей: ход
/// 01:26:09 начался осмысленно («Criterion 4: only reads, no writes…») и ушёл в
/// повтор слова `Записать` — 8294 раза подряд без единого перевода, строка на
/// 99 624 знака. Прогон встал на приёмке и упал по таймауту провайдера через
/// 600 секунд; предел объёма хода сработать не успел — до 150 000 знаков дело
/// не дошло.
///
/// Порог взят с большим запасом: у остальных 466 ходов самая длинная строка —
/// 6941 знак, третье место 3062, медиана 370, девяносто девятый процентиль
/// 2100. Отрыв разноса от ближайшего законного соседа — четырнадцатикратный,
/// поэтому 10 000 не заденет ни один известный законный ход, а разнос оборвёт
/// на первой же сотой доле его длины.
const LOOP_LINE_CHARS_MAX: usize = 10_000;

/// Учесть очередной кусок потока в счётчике дословных повторов.
///
/// Зачем: при жадной выборке без штрафа за повтор модель попадает в петлю и
/// гоняет один и тот же абзац, пока не выдаст признак конца. 30.08.2026 такой
/// прогон крутил блок в 3243 знака 28 раз подряд — девять минут, 36 тысяч
/// токенов и ноль знаков ответа, потому что до канала ответа дело не дошло.
/// Ловим по строкам: период петли — сотни знаков, но строки внутри него
/// повторяются дословно, и это видно уже на шестом обороте.
///
/// `tail` — незавершённый остаток строки между кусками потока (границы кусков
/// не совпадают с переводами строк). `seen` хранит по строке тройку
/// (сколько раз подряд, на каком знаке встретилась в последний раз, сколько раз
/// всего за ход), `pos` — сколько знаков канала уже прошло. Возвращает
/// (строка, число повторов) при превышении любого из двух порогов: подряд
/// (`LOOP_REPEAT_LIMIT`) или всего за ход (`LOOP_TOTAL_LIMIT`).
fn note_repeats(
    chunk: &str,
    tail: &mut String,
    seen: &mut HashMap<String, (u32, usize, u32)>,
    pos: &mut usize,
) -> Option<(String, u32)> {
    tail.push_str(chunk);
    let mut hit = None;
    while let Some(idx) = tail.find('\n') {
        let raw: String = tail.drain(..=idx).collect();
        *pos += raw.chars().count();
        let line = raw.trim();
        if line.chars().count() < LOOP_LINE_MIN_CHARS {
            continue;
        }
        let here = *pos;
        let entry = seen.entry(line.to_string()).or_insert((0, here, 0));
        // Разрыв больше порога — прошлые вхождения к нынешнему отношения не
        // имеют: это не петля, а возврат к той же мысли через страницу текста.
        entry.0 = if here - entry.1 > LOOP_GAP_MAX {
            1
        } else {
            entry.0 + 1
        };
        entry.1 = here;
        // Счётчик за весь ход разрывом не сбрасывается: медленная петля тем и
        // отличается, что между вхождениями лежат страницы текста.
        entry.2 += 1;
        let (streak, total) = (entry.0, entry.2);
        if hit.is_some() {
            continue;
        }
        if total >= LOOP_TOTAL_LIMIT {
            hit = Some((line.to_string(), streak.max(total)));
            continue;
        }
        // Порог «подряд» одна строка не берёт: настоящая петля крутит блок
        // целиком, и до шести оборотов доходят ВСЕ его длинные строки. Одна и
        // та же строка шесть раз с разными соседями — законный код: 19.09.2026
        // детектор четыре раза подряд обрывал исполнителя на первом же ходу,
        // потому что тот описывал структуру с шестью полями-списками и перед
        // каждым ставил один и тот же атрибут serde (48 знаков, вся структура —
        // в тысяче знаков). Одиночную строку ловит только счётчик за весь ход.
        if streak >= LOOP_REPEAT_LIMIT
            && seen
                .iter()
                .any(|(other, e)| other != line && e.0 >= LOOP_REPEAT_LIMIT)
        {
            hit = Some((line.to_string(), streak.max(total)));
        }
    }
    hit
}

/// Причина обрыва по объёму — готовой фразой для отчёта и журнала.
fn runaway_reason(chars: usize) -> String {
    format!("объём хода {chars} знаков превысил предел {TURN_CHARS_MAX}")
}

/// Причина обрыва по длине одной строки — готовой фразой.
///
/// Слово в фразе называем не для красоты: у этого вида петли повторяется именно
/// слово, и по нему сразу видно, на чём модель встала, — без него в отчёте
/// осталась бы одна голая длина.
fn long_line_reason(chars: usize, word: &str, repeats: u32) -> String {
    format!(
        "строка без перевода разрослась до {chars} знаков при пределе \
         {LOOP_LINE_CHARS_MAX}: слово «{word}» повторено {repeats} раз"
    )
}

/// Самое частое слово незавершённой строки и сколько раз оно встретилось.
///
/// Короткие обрывки (меньше трёх знаков) пропускаем: разделители и предлоги
/// частотнее любого осмысленного слова и забили бы собой отчёт.
fn top_word(line: &str) -> (String, u32) {
    let mut seen: HashMap<&str, u32> = HashMap::new();
    for word in line.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if word.chars().count() >= 3 {
            *seen.entry(word).or_insert(0) += 1;
        }
    }
    seen.into_iter()
        .max_by_key(|(_, n)| *n)
        .map(|(w, n)| (w.to_string(), n))
        .unwrap_or_default()
}

/// Самая частая длинная строка хода и сколько раз она встретилась.
///
/// Нужна при обрыве по объёму: сама строка идёт в поисковую фразу навыка и в
/// отчёт — по ней видно, вокруг чего модель ходила, когда её прервали.
fn top_line(seen: &HashMap<String, (u32, usize, u32)>) -> (String, u32) {
    seen.iter()
        .max_by_key(|(_, v)| v.2)
        .map(|(k, v)| (k.clone(), v.2))
        .unwrap_or_default()
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct UsageOption {
    include: bool,
}

/// Каталог для потоковой записи генерации: `AGENTS_MCP_STREAM_DIR`.
/// Не задан — работаем как раньше, ответ забираем целиком.
fn stream_dir() -> Option<std::path::PathBuf> {
    let raw = std::env::var("AGENTS_MCP_STREAM_DIR").ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    let path = std::path::PathBuf::from(raw);
    if let Err(e) = std::fs::create_dir_all(&path) {
        warn!(path = %path.display(), error = %e, "каталог живой записи потока не создан");
        return None;
    }
    Some(path)
}

fn sanitize_stream_model_name(model: &str) -> String {
    let sanitized: String = model
        .chars()
        .take(STREAM_MODEL_NAME_MAX_LEN)
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "model".to_string()
    } else {
        sanitized
    }
}

fn cleanup_old_stream_files(dir: &std::path::Path, now: SystemTime) {
    let cutoff = now.checked_sub(STREAM_RETENTION).unwrap_or(UNIX_EPOCH);
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            warn!(path = %dir.display(), error = %e, "каталог живой записи потока не прочитан для очистки");
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => {
                warn!(path = %dir.display(), error = %e, "элемент каталога живой записи потока не прочитан");
                continue;
            }
        };
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.ends_with(".stream.bsl") && !name.ends_with(".raw.jsonl") {
            continue;
        }
        let metadata = match entry.metadata() {
            Ok(metadata) if metadata.is_file() => metadata,
            Ok(_) => continue,
            Err(e) => {
                warn!(path = %entry.path().display(), error = %e, "метаданные файла живой записи потока не прочитаны");
                continue;
            }
        };
        if metadata.modified().is_ok_and(|modified| modified < cutoff) {
            if let Err(e) = std::fs::remove_file(entry.path()) {
                warn!(path = %entry.path().display(), error = %e, "старый файл живой записи потока не удалён");
            }
        }
    }
}

fn open_stream_files(
    dir: &std::path::Path,
    model: &str,
) -> (Option<std::fs::File>, Option<std::fs::File>) {
    let now = SystemTime::now();
    cleanup_old_stream_files(dir, now);
    let timestamp_ms = now
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let model = sanitize_stream_model_name(model);

    for suffix in 0..=u32::MAX {
        let stem = if suffix == 0 {
            format!("{timestamp_ms}-{model}")
        } else {
            format!("{timestamp_ms}-{model}-{suffix}")
        };
        let stream_path = dir.join(format!("{stem}.stream.bsl"));
        let raw_path = dir.join(format!("{stem}.raw.jsonl"));
        let stream = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&stream_path)
        {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                warn!(path = %stream_path.display(), error = %e, "файл живой записи потока не открыт");
                return (None, None);
            }
        };
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&raw_path)
        {
            Ok(raw) => return (Some(stream), Some(raw)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                drop(stream);
                if let Err(remove_error) = std::fs::remove_file(&stream_path) {
                    warn!(path = %stream_path.display(), error = %remove_error, "неполная пара файлов живой записи потока не удалена");
                }
            }
            Err(e) => {
                warn!(path = %raw_path.display(), error = %e, "файл сырых событий потока не открыт");
                return (Some(stream), None);
            }
        }
    }
    warn!(path = %dir.display(), "не удалось подобрать уникальное имя живой записи потока");
    (None, None)
}

#[derive(Deserialize, Debug)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<UsageInfo>,
    /// Если запрос упал на стороне OpenRouter — здесь будет ошибка.
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: ChoiceMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ChoiceMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct ToolCall {
    #[serde(default)]
    id: String,
    function: ToolCallFunction,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct ToolCallFunction {
    name: String,
    /// Аргументы приходят строкой с JSON внутри (OpenAI-контракт).
    #[serde(default)]
    arguments: String,
}

fn openai_tool_definitions(defs: &[mcp_client::ToolDef]) -> Vec<Value> {
    defs.iter()
        .map(|def| {
            json!({
                "type": "function",
                "function": {
                    "name": def.full_name,
                    "description": def.description,
                    "parameters": def.parameters,
                }
            })
        })
        .collect()
}

#[derive(Deserialize, Debug, Default)]
struct UsageInfo {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    /// OpenRouter возвращает реальную стоимость в долларах (если usage.include=true).
    #[serde(default)]
    cost: Option<f64>,
    /// Сколько токенов входа обслужено из кеша префикса. DeepSeek кладёт это в
    /// собственное поле, остальные OpenAI-совместимые — во вложенное
    /// `prompt_tokens_details.cached_tokens`. Читаем оба: у кого поля нет,
    /// останется 0 и поведение не изменится.
    #[serde(default)]
    prompt_cache_hit_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Deserialize, Debug, Default)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u32,
}

impl UsageInfo {
    /// Токены входа из кеша — по тому полю, которое заполнил провайдер.
    fn cached_in(&self) -> u32 {
        self.prompt_cache_hit_tokens.max(
            self.prompt_tokens_details
                .as_ref()
                .map(|d| d.cached_tokens)
                .unwrap_or(0),
        )
    }
}

/// Результат одного round-trip к модели.
struct ChatTurn {
    content: Option<String>,
    tool_calls: Vec<ToolCall>,
    finish_reason: String,
    tokens_in: u32,
    /// Часть `tokens_in`, пришедшая из кеша префикса (не вычитается из него).
    cached_in: u32,
    tokens_out: u32,
    cost: Option<f64>,
    reasoning: Option<String>,
}

/// Распознать текстовый fallback tool-call одинаково для обычного и потокового ответа.
fn normalize_text_tool_calls(provider: &str, turn: &mut ChatTurn) {
    if !turn.tool_calls.is_empty() {
        return;
    }
    let Some(content) = turn.content.as_deref() else {
        return;
    };
    let parsed = parse_text_tool_calls(content);
    if !parsed.is_empty() {
        let cleaned = strip_text_tool_calls(content);
        turn.content = (!cleaned.trim().is_empty()).then_some(cleaned);
        turn.tool_calls = parsed;
    } else if turn.finish_reason == "tool_calls" {
        tracing::warn!(
            provider,
            content_head = %truncate(content, 400),
            "finish=tool_calls, но вызов не распознан (ни нативно, ни текстом)"
        );
    }
}

/// Сервер не смог разобрать вызов инструмента, сгенерированный моделью.
///
/// llama-server отвечает HTTP 500 с телом `Failed to parse tool call arguments
/// as JSON: [json.exception.parse_error.101] parse error at line 1, column N:
/// … missing closing quote`, когда модель испортила экранирование в большом
/// текстовом аргументе (текст запроса, целый модуль). Ответ модели при этом
/// потерян, но контекст диалога цел — повтор хода обычно проходит.
/// Аргументы вызова для эха в историю диалога: строку, которая сама не
/// разбирается как JSON, подменяем пустым объектом.
///
/// llama-server разбирает `arguments` не только в ответе модели, но и во всех
/// сообщениях входящего запроса. Один испорченный вызов, попавший в историю,
/// делает неразбираемым КАЖДЫЙ следующий ход — прогон умирает целиком, и повтор
/// не спасает. Настоящие аргументы всё равно потеряны, а с пустым объектом
/// диалог остаётся рабочим: модель получает объяснение и вызывает инструмент
/// заново.
fn safe_arguments(raw: &str) -> &str {
    if serde_json::from_str::<Value>(raw).is_ok() {
        raw
    } else {
        "{}"
    }
}

fn is_tool_call_parse_error(e: &LlmError) -> bool {
    match e {
        LlmError::Provider(msg) => msg.contains("parse tool call arguments"),
        _ => false,
    }
}

/// Сколько знаков хвоста оборванного канала брать как поисковую фразу.
///
/// Замер 31.08.2026 на живом затыке (модель не могла объяснить, почему
/// `ГДЕ Статус = "Действует"` даёт 0 строк): по хвосту такого размера нужный
/// навык встаёт в выдаче первым, а по одной повторяющейся строке — только
/// смежный, не про корень. Больше брать смысла нет: фразу всё равно сравнивают
/// с описанием навыка в пару строк.
const LOOP_TAIL_CHARS: usize = 1500;

/// Сколько раз за прогон лечить петлю подкладкой навыка.
///
/// Пока резерв был конечным (одно припасённое тело), пределом служил он сам.
/// Поиск по месту затыка резерв не ограничивает, поэтому предел нужен явный:
/// каждое лечение стоит оборванного хода плюс тело навыка в истории диалога.
const LOOP_FIX_LIMIT: u32 = 3;

/// Последние `n` знаков строки. По символам, не по байтам: в размышлении есть
/// и кириллица, и обрезка по байту развалила бы UTF-8.
fn tail_chars(s: &str, n: usize) -> String {
    let total = s.chars().count();
    if total <= n {
        return s.to_string();
    }
    s.chars().skip(total - n).collect()
}

/// Поисковая фраза для навыка по месту затыка: хвост канала плюс сама
/// повторявшаяся строка. Строку добавляем отдельно — она уже усечена до 120
/// знаков и в хвост попадает не всегда (петля могла оборваться на другом).
fn loop_query(line: &str, tail: &str) -> String {
    let mut q = String::with_capacity(tail.len() + line.len() + 2);
    q.push_str(tail.trim());
    if !line.trim().is_empty() {
        if !q.is_empty() {
            q.push('\n');
        }
        q.push_str(line.trim());
    }
    q
}

/// Имя навыка из каталога, которое ещё не подкладывали и которого нет в
/// промпте. Порядок — по оценке реранкера, как и при отборе тел для промпта.
fn pick_skill(catalog: &str, used: &[String]) -> Option<(String, f64)> {
    crate::skills::top_scored_by_rerank(catalog, used.len() + 1)
        .into_iter()
        .find(|(n, _)| !used.iter().any(|u| u == n))
}

/// Ниже какой оценки реранкера найденный навык считаем «не по теме».
///
/// Шкала bge-reranker-v2-m3 идёт от -11 до +3, медиана верных попаданий около
/// -4. Порог отсекает хвост: 31.08.2026 прогон упал, израсходовав все три
/// попытки лечения на навыки, к затыку отношения не имевшие («поиск по
/// контактной информации», «производительность виртуальных таблиц»,
/// «регистрация обработки в БСП») — поиск по хвосту всегда возвращает
/// что-нибудь, даже когда подходящего навыка в базе нет.
///
/// Значение предварительное: оценок тех подкладок в журнале не было, подбирать
/// было не по чему. Теперь оценка пишется в журнал при каждом лечении — после
/// двух-трёх прогонов порог уточняется по данным, а не по рассуждению.
const LOOP_SKILL_MIN_RR: f64 = -5.0;

/// Температура, до которой поднимаем выборку, если подсказать нечем.
///
/// При жадной выборке (--temp 0 --top-k 1) повтор хода на неизменившемся
/// контексте даёт ту же петлю знак в знак. Тело навыка контекст меняет само;
/// когда навыка нет, единственный рычаг — сдвиг выборки. Та же мера уже
/// применяется в ветке с неразобранным вызовом инструмента.
const LOOP_NUDGE_TEMP: f32 = 0.2;

struct ChatParams<'a> {
    model: &'a str,
    messages: &'a [Value],
    tools: Option<&'a [Value]>,
    temperature: f32,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    extra: &'a serde_json::Map<String, Value>,
}

struct ResponseParts {
    content: String,
    finish_reason: String,
    tokens_in: u32,
    cached_in: u32,
    tokens_out: u32,
    cost_usd: Option<f64>,
    reasoning: Option<String>,
}

impl OpenRouterProvider {
    /// Один HTTP-вызов chat/completions. Разбирает первый choice + usage.
    async fn chat_once_attempt(
        &self,
        params: &ChatParams<'_>,
        timeout: Duration,
    ) -> Result<ChatTurn, LlmError> {
        let sink = stream_dir();
        let body = ChatRequest {
            model: params.model,
            messages: params.messages,
            temperature: params.temperature,
            max_tokens: params.max_tokens,
            top_p: params.top_p,
            tools: params.tools,
            usage: UsageOption { include: true },
            stream: sink.is_some(),
            stream_options: sink.as_ref().map(|_| StreamOptions {
                include_usage: true,
            }),
            extra: params.extra,
        };

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let mut builder = self.client.post(&url);
        // В потоковом режиме общий срок ставим не здесь, а в `read_stream`.
        // Свой `timeout` у запроса при обрыве по времени рапортует ошибкой
        // чтения тела — 31.08.2026 вызов 8599 упёрся в свои 1800 секунд, а в
        // отчёт попало «обрыв потока: error decoding response body», и разбор
        // ушёл искать сетевой сбой вместо исчерпанного срока.
        if sink.is_none() {
            builder = builder.timeout(timeout);
        }
        builder = builder
            .bearer_auth(&self.api_key)
            .header(header::CONTENT_TYPE, "application/json")
            .header("X-Title", "agents-mcp")
            .json(&body);
        if let Some(referer) = &self.referer {
            builder = builder.header("HTTP-Referer", referer);
        }

        if let Some(dir) = sink {
            // Своего срока у запроса здесь нет, поэтому им накрыты обе стадии:
            // ожидание заголовков и чтение потока — вместе, а не по отдельности.
            let started = tokio::time::Instant::now();
            let resp = match tokio::time::timeout(timeout, builder.send()).await {
                Err(_) => return Err(LlmError::Timeout),
                Ok(r) => r.map_err(map_reqwest_err)?,
            };
            if resp.status().as_u16() == 429 {
                return Err(LlmError::RateLimitedRetry(retry_after_seconds(
                    resp.headers(),
                )));
            }
            let left = timeout.saturating_sub(started.elapsed());
            let mut turn = self.read_stream(resp, params.model, &dir, left).await?;
            normalize_text_tool_calls(&self.name, &mut turn);
            return Ok(turn);
        }
        let resp = builder.send().await.map_err(map_reqwest_err)?;
        let status = resp.status();
        let retry_after = retry_after_seconds(resp.headers());
        let raw = resp
            .text()
            .await
            .map_err(|e| LlmError::Provider(format!("{}: ошибка чтения тела: {e}", self.name)))?;

        if !status.is_success() {
            if status.as_u16() == 429 {
                return Err(LlmError::RateLimitedRetry(retry_after));
            }
            return Err(LlmError::Provider(format!(
                "{} HTTP {status}: {}",
                self.name,
                truncate(&raw, 500)
            )));
        }

        let parsed: ChatResponse = serde_json::from_str(&raw).map_err(|e| {
            LlmError::InvalidResponse(format!(
                "{}: не удалось распарсить JSON: {e}; body={}",
                self.name,
                truncate(&raw, 200)
            ))
        })?;

        if let Some(err) = parsed.error {
            return Err(LlmError::Provider(format!("{} API: {err}", self.name)));
        }

        let (tokens_in, cached_in, tokens_out, cost) = match &parsed.usage {
            Some(u) => {
                let cached = u.cached_in();
                let cost = u.cost.or_else(|| {
                    pricing::openai_cost(
                        self.prices.get(params.model),
                        u.prompt_tokens,
                        cached,
                        u.completion_tokens,
                    )
                });
                (u.prompt_tokens, cached, u.completion_tokens, cost)
            }
            None => (0, 0, 0, None),
        };

        let choice =
            parsed.choices.into_iter().next().ok_or_else(|| {
                LlmError::InvalidResponse(format!("{}: пустой choices", self.name))
            })?;
        let finish_reason = choice.finish_reason.unwrap_or_else(|| "unknown".into());

        let reasoning = choice
            .message
            .reasoning_content
            .or(choice.message.reasoning);
        let mut turn = ChatTurn {
            content: choice.message.content,
            tool_calls: choice.message.tool_calls.unwrap_or_default(),
            finish_reason,
            tokens_in,
            cached_in,
            tokens_out,
            cost,
            reasoning,
        };
        normalize_text_tool_calls(&self.name, &mut turn);
        Ok(turn)
    }

    async fn chat_once(
        &self,
        params: ChatParams<'_>,
        timeout: Duration,
    ) -> Result<ChatTurn, LlmError> {
        let deadline = tokio::time::Instant::now() + timeout;
        for retry in 0..=3u64 {
            let left = deadline.saturating_duration_since(tokio::time::Instant::now());
            if left.is_zero() {
                return Err(LlmError::Timeout);
            }
            match self.chat_once_attempt(&params, left).await {
                Err(LlmError::RateLimitedRetry(header_delay)) => {
                    let delay = header_delay.unwrap_or(2u64 << retry);
                    tracing::warn!(
                        provider = %self.name,
                        model = params.model,
                        active = self.active.load(Ordering::SeqCst),
                        retry = retry + 1,
                        delay_sec = delay,
                        "провайдер ответил 429"
                    );
                    if retry >= 3 {
                        return Err(LlmError::RateLimited);
                    }
                    let sleep = Duration::from_secs(delay);
                    if sleep >= deadline.saturating_duration_since(tokio::time::Instant::now()) {
                        return Err(LlmError::Timeout);
                    }
                    tokio::time::sleep(sleep).await;
                }
                other => return other,
            }
        }
        unreachable!()
    }

    /// Прочитать потоковый ответ, дописывая текст в файл по мере поступления.
    ///
    /// Смысл — видеть генерацию вживую, а не после её конца: срыв (повтор одной
    /// строки, развал слов) заметен сразу, а не через шесть минут ожидания.
    /// Вызовы инструментов в потоке приходят по частям — отдельно идентификатор,
    /// имя и куски аргументов, — поэтому их собираем вручную по индексу.
    async fn read_stream(
        &self,
        mut resp: reqwest::Response,
        model: &str,
        dir: &std::path::Path,
        timeout: Duration,
    ) -> Result<ChatTurn, LlmError> {
        use std::io::Write;

        let status = resp.status();
        if !status.is_success() {
            let raw = match tokio::time::timeout(timeout, resp.text()).await {
                Ok(Ok(text)) => text,
                Ok(Err(e)) => {
                    return Err(LlmError::Provider(format!(
                        "{}: ошибка чтения тела: {e}",
                        self.name
                    )))
                }
                Err(_) => return Err(LlmError::Timeout),
            };
            if status.as_u16() == 429 {
                return Err(LlmError::RateLimited);
            }
            return Err(LlmError::Provider(format!(
                "{} HTTP {status}: {}",
                self.name,
                truncate(&raw, 500)
            )));
        }

        let (mut file, mut raw_file) = open_stream_files(dir, model);
        if let Some(f) = file.as_mut() {
            let _ = writeln!(f, "\n=== {} ===", chrono::Local::now().format("%H:%M:%S"));
        }

        // Сырые пакеты сервера, как они пришли по сети, до всякого разбора.
        // Нужны, чтобы отличить «модель ничего не выдала» от «ответ приехал в
        // поле, которое мы не читаем». По разобранной записи это неразличимо:
        // 30.08.2026 мы приняли пустое поле content за молчание модели, ни разу
        // не посмотрев, что было в самом ответе.
        if let Some(f) = raw_file.as_mut() {
            let _ = writeln!(
                f,
                "{{\"событие\":\"начало вызова\",\"время\":\"{}\",\"модель\":\"{model}\"}}",
                chrono::Local::now().format("%H:%M:%S%.3f")
            );
        }

        let mut buf = Vec::<u8>::new();
        let (mut content, mut reasoning) = (String::new(), String::new());
        // Что писали в файл последним: 0 — ничего, 1 — размышление, 2 — ответ.
        // Нужно, чтобы поставить заголовок только при переключении канала, а не
        // перед каждым куском: иначе живой текст утонет в пометках.
        let mut last_kind: u8 = 0;
        let mut finish_reason = String::from("unknown");
        let mut usage = UsageInfo::default();
        let mut usage_seen = false;
        // (id, имя, аргументы) по индексу вызова — части приходят вразбивку.
        let mut calls: Vec<(String, String, String)> = Vec::new();
        // Счётчик дословных повторов: отдельный на каждый канал, чтобы код,
        // законно повторённый в размышлении и в ответе, не складывался.
        let mut reason_lines: HashMap<String, (u32, usize, u32)> = HashMap::new();
        let (mut reason_tail, mut reason_pos) = (String::new(), 0usize);
        let mut content_lines: HashMap<String, (u32, usize, u32)> = HashMap::new();
        let (mut content_tail, mut content_pos) = (String::new(), 0usize);
        // (строка, повторов, канал, что сработало) — заполняется предохранителем.
        let mut loop_hit: Option<(String, u32, &str, String)> = None;
        // Знаков хода: размышление и ответ вместе. Считаем сами, а не по
        // `reasoning.len()`, чтобы мерить в знаках, а не в байтах (кириллица в
        // UTF-8 занимает по два байта, и байтовый предел сработал бы вдвое раньше).
        let mut turn_chars: usize = 0;
        // Предел времени на весь поток. Без него ход живёт дольше, чем разрешено
        // агенту: `timeout` у запроса до чтения тела не дотягивается — замер
        // 31.08.2026, вызов 8599 при `timeout_sec = 900` шёл 1845 секунд и
        // кончился обрывом связи, а не остановкой по времени.
        let deadline = tokio::time::Instant::now() + timeout;

        'stream: loop {
            let left = deadline.saturating_duration_since(tokio::time::Instant::now());
            if left.is_zero() {
                return Err(LlmError::Timeout);
            }
            let chunk = match tokio::time::timeout(left, resp.chunk()).await {
                // Кусок не пришёл до срока — это именно исчерпанное время, а не
                // разрыв связи; в отчёте эти два случая раньше выглядели
                // одинаково («обрыв потока») и путали разбор.
                Err(_) => return Err(LlmError::Timeout),
                Ok(Err(e)) => {
                    return Err(LlmError::Provider(format!(
                        "{}: обрыв потока: {e}",
                        self.name
                    )))
                }
                Ok(Ok(None)) => break 'stream,
                Ok(Ok(Some(c))) => c,
            };
            buf.extend_from_slice(&chunk);
            while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                // Декодируем только законченную строку: незавершённый UTF-8-хвост
                // остаётся в buf до следующей сетевой порции.
                let line_bytes: Vec<u8> = buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line_bytes);
                let Some(data) = line.trim().strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if let Some(f) = raw_file.as_mut() {
                    let _ = writeln!(f, "{data}");
                    let _ = f.flush();
                }
                // «[DONE]» — конец потока по спецификации OpenAI, и на нём цикл
                // заканчивается. Раньше маркер просто пропускался, а выход был
                // один — закрытие тела сервером. Провайдеры, которые держат
                // соединение открытым (мост Grok на отдельной машине, keep-alive), из-за
                // этого висели до конца отведённого времени: 06.09.2026 вызов
                // 9136 получил полный ответ за 21 с и всё равно был записан как
                // «timeout провайдера» с нулём токенов на выходе через 600 с.
                // Данные при этом не теряются: чанк с usage приходит ДО «[DONE]».
                if data == "[DONE]" {
                    break 'stream;
                }
                if data.is_empty() {
                    continue;
                }
                let Ok(v) = serde_json::from_str::<Value>(data) else {
                    continue;
                };
                if let Some(error) = v.get("error").filter(|error| !error.is_null()) {
                    let text = error["message"]
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| error.to_string());
                    return Err(LlmError::Provider(format!("{} API: {text}", self.name)));
                }
                if let Some(u) = v.get("usage") {
                    if !u.is_null() {
                        usage = serde_json::from_value(u.clone()).unwrap_or_default();
                        usage_seen = true;
                    }
                }
                let Some(choice) = v["choices"].get(0) else {
                    continue;
                };
                if let Some(fr) = choice["finish_reason"].as_str() {
                    finish_reason = fr.to_string();
                    if fr == "error" {
                        return Err(LlmError::Provider(format!(
                            "{} API: поток завершён с finish_reason=error",
                            self.name
                        )));
                    }
                }
                let delta = &choice["delta"];
                if let Some(c) = delta["content"].as_str() {
                    content.push_str(c);
                    if let Some(f) = file.as_mut() {
                        if last_kind != 2 {
                            let _ = write!(f, "\n\n--- ОТВЕТ ---\n");
                            last_kind = 2;
                        }
                        let _ = write!(f, "{c}");
                        let _ = f.flush();
                    }
                    turn_chars += c.chars().count();
                    if let Some((line, n)) =
                        note_repeats(c, &mut content_tail, &mut content_lines, &mut content_pos)
                    {
                        let reason = format!("строка «{}» повторена {n} раз", truncate(&line, 120));
                        loop_hit = Some((line, n, "ответе", reason));
                        break 'stream;
                    }
                    // Незавершённая строка канала: `note_repeats` слил из неё всё
                    // до последнего перевода, поэтому в остатке лежит ровно та
                    // строка, которую модель пишет сейчас. Пухнет она — петля
                    // внутри строки, счётчикам по строкам не видная.
                    let line_chars = content_tail.chars().count();
                    if line_chars >= LOOP_LINE_CHARS_MAX {
                        let (word, n) = top_word(&content_tail);
                        let head = truncate(content_tail.trim_start(), 120);
                        loop_hit =
                            Some((head, n, "ответе", long_line_reason(line_chars, &word, n)));
                        break 'stream;
                    }
                    if turn_chars >= TURN_CHARS_MAX {
                        let (line, n) = top_line(&content_lines);
                        loop_hit = Some((line, n, "ответе", runaway_reason(turn_chars)));
                        break 'stream;
                    }
                }
                if let Some(r) = delta["reasoning_content"]
                    .as_str()
                    .or_else(|| delta["reasoning"].as_str())
                {
                    reasoning.push_str(r);
                    // Размышления пишем в тот же файл наравне с ответом: именно
                    // в них видно, на чём модель ходит по кругу. Без этого живая
                    // картина бесполезна — у моделей, которые думают долго, до
                    // канала ответа дело может не дойти вовсе.
                    if let Some(f) = file.as_mut() {
                        if last_kind != 1 {
                            let _ = write!(f, "\n\n--- РАЗМЫШЛЕНИЕ ---\n");
                            last_kind = 1;
                        }
                        let _ = write!(f, "{r}");
                        let _ = f.flush();
                    }
                    turn_chars += r.chars().count();
                    if let Some((line, n)) =
                        note_repeats(r, &mut reason_tail, &mut reason_lines, &mut reason_pos)
                    {
                        let reason = format!("строка «{}» повторена {n} раз", truncate(&line, 120));
                        loop_hit = Some((line, n, "размышлении", reason));
                        break 'stream;
                    }
                    let line_chars = reason_tail.chars().count();
                    if line_chars >= LOOP_LINE_CHARS_MAX {
                        let (word, n) = top_word(&reason_tail);
                        let head = truncate(reason_tail.trim_start(), 120);
                        loop_hit = Some((
                            head,
                            n,
                            "размышлении",
                            long_line_reason(line_chars, &word, n),
                        ));
                        break 'stream;
                    }
                    if turn_chars >= TURN_CHARS_MAX {
                        let (line, n) = top_line(&reason_lines);
                        loop_hit = Some((line, n, "размышлении", runaway_reason(turn_chars)));
                        break 'stream;
                    }
                }
                if let Some(tcs) = delta["tool_calls"].as_array() {
                    for tc in tcs {
                        const MAX_TOOL_CALL_INDEX: u64 = 128;
                        let idx = match tc["index"].as_u64() {
                            Some(idx) if idx > MAX_TOOL_CALL_INDEX => {
                                return Err(LlmError::InvalidResponse(format!(
                                    "{}: index вызова инструмента {idx} превышает предел {MAX_TOOL_CALL_INDEX}",
                                    self.name
                                )));
                            }
                            Some(idx) => idx as usize,
                            None if tc["id"].as_str().is_some()
                                || tc["function"]["name"].as_str().is_some() =>
                            {
                                calls.len()
                            }
                            None => calls.len().saturating_sub(1),
                        };
                        while calls.len() <= idx {
                            calls.push(Default::default());
                        }
                        if let Some(id) = tc["id"].as_str() {
                            calls[idx].0 = id.to_string();
                        }
                        if let Some(n) = tc["function"]["name"].as_str() {
                            if calls[idx].1 != n {
                                calls[idx].1.push_str(n);
                            }
                        }
                        if let Some(a) = tc["function"]["arguments"].as_str() {
                            calls[idx].2.push_str(a);
                        }
                    }
                }
            }
        }

        // Петля: соединение рвём, сервер прекращает генерацию. Отдаём ошибку, а
        // не пустой ответ: пустой ответ выглядит как «модель промолчала» и
        // уводит разбор в сторону — 30.08.2026 на это ушло полдня.
        if let Some((line, n, channel, reason)) = loop_hit {
            if let Some(f) = file.as_mut() {
                let _ = write!(f, "\n\n--- ПРЕРВАНО: {reason} ---\n");
            }
            tracing::warn!(
                provider = %self.name,
                model = %model,
                repeats = n,
                channel = %channel,
                reason = %reason,
                turn_chars,
                reasoning_chars = reasoning.len(),
                content_chars = content.len(),
                line = %truncate(&line, 120),
                "генерация прервана предохранителем"
            );
            // Хвост берём из того канала, где случилась петля: в нём и лежит то
            // место, на котором модель встала.
            let source = if channel == "ответе" {
                &content
            } else {
                &reasoning
            };
            return Err(LlmError::Loop {
                provider: self.name.clone(),
                channel: channel.to_string(),
                reason,
                line: truncate(&line, 120),
                repeats: n,
                reasoning_chars: reasoning.len(),
                content_chars: content.len(),
                tail: tail_chars(source, LOOP_TAIL_CHARS),
            });
        }

        let tool_calls: Vec<ToolCall> = calls
            .into_iter()
            .filter(|(_, name, _)| !name.is_empty())
            .map(|(id, name, arguments)| ToolCall {
                id,
                function: ToolCallFunction { name, arguments },
            })
            .collect();
        let cached_in = usage.cached_in();
        let cost = usage.cost.or_else(|| {
            if usage_seen {
                pricing::openai_cost(
                    self.prices.get(model),
                    usage.prompt_tokens,
                    cached_in,
                    usage.completion_tokens,
                )
            } else {
                None
            }
        });
        Ok(ChatTurn {
            content: (!content.is_empty()).then_some(content),
            tool_calls,
            finish_reason,
            tokens_in: usage.prompt_tokens,
            cached_in,
            tokens_out: usage.completion_tokens,
            cost,
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
        })
    }

    /// Собрать OpenAI-`tools` + реестр маршрутизации
    /// (`full_name → (url, tool, session)`) из `mcp_config` агента. Пустой
    /// `allowed_tools` = все инструменты подключённых серверов (конвенция
    /// claude-cli); непустой — белый список. Для каждого сервера делается
    /// полный MCP-хендшейк.
    /// Сервер, заявленный в `mcp_config`, но не ответивший, — не «пропустим и
    /// поедем»: вызов отклоняется с [`LlmError::ToolsUnavailable`] до обращения
    /// к модели, иначе агент работает без заявленных инструментов, а прогон
    /// уже оплачен.
    #[allow(clippy::type_complexity)]
    #[allow(dead_code)]
    async fn build_tools(
        &self,
        hints: &ClaudeCliHints,
    ) -> Result<
        (
            Vec<Value>,
            HashMap<String, (mcp_client::McpServer, String, mcp_client::McpSession)>,
            mcp_client::StdioPool,
        ),
        LlmError,
    > {
        let built = tool_loop::build_mcp_tools(&self.client, hints).await?;
        let tools = openai_tool_definitions(&built.defs);
        Ok((tools, built.registry, built.stdio_pool))
    }

    fn to_response(&self, parts: ResponseParts) -> LlmResponse {
        LlmResponse {
            content: parts.content,
            tokens_in: parts.tokens_in,
            tokens_out: parts.tokens_out,
            cost_usd: parts.cost_usd,
            finish_reason: parts.finish_reason,
            reasoning: parts.reasoning,
            session_id: None,
            // Кеш префикса у OpenAI-совместимых провайдеров только читается:
            // отдельной операции «создать кеш» (и отдельной цены за неё, как у
            // Anthropic) тут нет, поэтому creation всегда 0. Провайдер, не
            // присылающий кеш-полей, даёт cached_in = 0 — тогда весь вход
            // ложится в raw, как было раньше.
            raw_input_tokens: parts.tokens_in.saturating_sub(parts.cached_in),
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: parts.cached_in,
            // Наполняется в complete() перед возвратом (из Transcript::take).
            transcript: Vec::new(),
        }
    }
}

// ── Provider impl ──────────────────────────────────────────────────────────

#[async_trait]
impl LlmProvider for OpenRouterProvider {
    async fn complete(&self, mut req: LlmRequest) -> Result<LlmResponse, LlmError> {
        let deadline = tokio::time::Instant::now() + req.timeout;
        let _permit = match &self.semaphore {
            Some(semaphore) => Some(
                tokio::time::timeout_at(deadline, semaphore.clone().acquire_owned())
                    .await
                    .map_err(|_| LlmError::Timeout)?
                    .map_err(|e| LlmError::Provider(format!("семафор провайдера закрыт: {e}")))?,
            ),
            None => None,
        };
        self.active.fetch_add(1, Ordering::SeqCst);
        let _active = ActiveCall(self.active.clone());
        req.timeout = deadline.saturating_duration_since(tokio::time::Instant::now());
        // Начальные сообщения: system (если есть) + user (с фолбэком).
        let mut messages: Vec<Value> = Vec::with_capacity(2);
        if !req.system_prompt.is_empty() {
            messages.push(json!({"role": "system", "content": req.system_prompt}));
        }
        let user_content = if req.user_input.is_empty() {
            "Выполни задачу из system-промпта и верни результат."
        } else {
            req.user_input.as_str()
        };
        messages.push(json!({"role": "user", "content": user_content}));

        // Собираем тулы, если у агента задан mcp_config. allowed_tools пустой —
        // это «все инструменты подключённых MCP-серверов» (конвенция claude-cli),
        // непустой — белый список. Без mcp_config → single-shot (brief-analyst,
        // пробники).
        let mut mcp_tools = match &req.cli_hints {
            Some(h) if h.mcp_config.is_some() => {
                tool_loop::build_mcp_tools(&self.client, h).await?
            }
            _ => tool_loop::McpTools::default(),
        };
        let tools = openai_tool_definitions(&mcp_tools.defs);

        // Транскрипт прогона (opt-in через env). enabled() → копим, иначе no-op.
        let transcript = Transcript::open(req.turn_sink.clone());
        if transcript.enabled() {
            transcript.write(&json!({
                "event": "start",
                "provider": self.name,
                "model": req.model,
                "tools": tools.len(),
                "system_prompt": req.system_prompt,
                "user_input": user_content,
            }));
        }

        // Single-shot: тулов нет (обычные роли — brief-analyst, пробники).
        if tools.is_empty() {
            let turn = self
                .chat_once(
                    ChatParams {
                        model: &req.model,
                        messages: &messages,
                        tools: None,
                        temperature: req.temperature,
                        max_tokens: cap_tokens(req.max_tokens),
                        top_p: req.top_p,
                        extra: &req.extra_body,
                    },
                    deadline.saturating_duration_since(tokio::time::Instant::now()),
                )
                .await?;
            transcript.write(&json!({
                "event": "single_shot",
                "content": turn.content,
                "reasoning": turn.reasoning,
                "finish": turn.finish_reason,
                "tokens_in": turn.tokens_in,
                "tokens_out": turn.tokens_out,
            }));
            // Фолбэк-оценка токенов, если usage не пришёл.
            let usage_missing = turn.tokens_in == 0 && turn.tokens_out == 0;
            let (tin, tout) = if usage_missing {
                let tin = (req.system_prompt.len() / 4 + req.user_input.len() / 4) as u32;
                let tout = (turn.content.as_deref().unwrap_or("").len() / 4) as u32;
                (tin, tout)
            } else {
                (turn.tokens_in, turn.tokens_out)
            };
            let cost = if usage_missing && turn.cost.is_none() {
                pricing::openai_cost(self.prices.get(&req.model), tin, 0, tout)
            } else {
                turn.cost
            };
            let mut resp = self.to_response(ResponseParts {
                content: turn.content.unwrap_or_default(),
                finish_reason: turn.finish_reason,
                tokens_in: tin,
                cached_in: turn.cached_in,
                tokens_out: tout,
                cost_usd: cost,
                reasoning: turn.reasoning,
            });
            resp.transcript = transcript.take();
            return Ok(resp);
        }

        // Agentic-loop: модель просит tool_calls — мы исполняем и возвращаем
        // результаты, пока не получим финальный ответ без tool_calls.
        let max_turns = req
            .cli_hints
            .as_ref()
            .and_then(|h| h.max_turns)
            .unwrap_or(DEFAULT_MAX_TOOL_TURNS)
            .max(1);

        let (mut total_in, mut total_out, mut total_cost) = (0u32, 0u32, Some(0.0));
        // Сколько входа за весь прогон обслужил кеш префикса. В agent-loop это
        // основная часть расхода: каждый ход везёт всю историю заново, и со
        // второго хода она приходит из кеша по цене в десятки раз ниже.
        let mut total_cached_in = 0u32;
        let mut last_finish = String::from("unknown");

        let mut tool_state = ToolCallState::default();
        // При temperature=0 модель на тот же контекст обязана выдать тот же
        // ответ — она физически не может «попробовать иначе». Один поднятый ход
        // выбивает её из колеи; дальше возвращаемся к исходному значению.
        let mut temperature = req.temperature;

        // Сбой РАЗБОРА вызова на стороне сервера — не повод ронять весь прогон.
        // llama-server отдаёт HTTP 500 «Failed to parse tool call arguments as
        // JSON», когда модель испортила экранирование в большом текстовом
        // аргументе. Раньше такая ошибка уходила через `?` и убивала прогон
        // целиком: 2026-07-23 так потеряны три прогона по 20-30 минут, каждый —
        // на последнем шаге. Считаем ПОДРЯД идущие сбои: успешный ход обнуляет.
        let mut parse_fail_streak: u32 = 0;
        const PARSE_FAIL_LIMIT: u32 = 3;

        // Лечение петли навыком. Модель зацикливается там, где не может сойтись,
        // — обычно ей не хватает знания о предметной части, а не ходов. Ронять
        // из-за этого весь прогон расточительно: к моменту петли позади бывает
        // два десятка ходов с доведённым до зелёного запросом. Поэтому ищем
        // навык по тому месту, где она встала, подкладываем тело в диалог и
        // повторяем ход.
        let mut loop_fixes: u32 = 0;
        // Что уже перед глазами модели: тела из промпта плюс всё, что подложили
        // по ходу. Второй раз то же самое давать бессмысленно.
        let mut skills_shown: Vec<String> = req.prompt_skill_names.clone();

        // Окно контекста сервера и сколько его занял прошлый ход. Размер берём
        // не оценкой по символам, а фактическим `tokens_in`, который сервер
        // вернул за предыдущее обращение — это точная величина, а не догадка.
        let ctx_window = self.context_window().await;
        let trim_at = history_trim_at();
        // Условия прогона пишем в журнал явно: по отчёту потом видно, с каким
        // окном и с каким порогом вытеснения работала модель, а не «наверное».
        tracing::info!(
            provider = %self.name, model = %req.model,
            ctx_window = ctx_window.unwrap_or(0), history_trim_at = trim_at,
            "условия прогона: окно сервера и порог вытеснения истории"
        );
        let mut last_tokens_in: u32 = 0;

        for turn_idx in 0..max_turns {
            // Вытеснение старых результатов инструментов, когда история подошла
            // к порогу окна. Свежие ходы не трогаем: там лежит то, с чем модель
            // работает прямо сейчас.
            if let (Some(window), true) = (ctx_window, trim_at > 0.0) {
                let threshold = (window as f64 * trim_at) as u32;
                if last_tokens_in > threshold {
                    let evicted = trim_history(&mut messages, 6);
                    if evicted > 0 {
                        tracing::warn!(
                            provider = %self.name, turn = turn_idx,
                            tokens_in = last_tokens_in, threshold, evicted,
                            "история подошла к порогу окна — вытеснил старые результаты"
                        );
                        transcript.write(&json!({
                            "event": "history_trim",
                            "turn": turn_idx,
                            "tokens_in_before": last_tokens_in,
                            "threshold": threshold,
                            "window": window,
                            "evicted": evicted,
                        }));
                    }
                }
            }
            // Длительность самого обращения к модели — тоже явным замером.
            // Вместе с duration_ms у tool_result это даёт полный разбор хода:
            // сколько думала модель и сколько ждали инструменты.
            let turn_started = std::time::Instant::now();
            let turn = match self
                .chat_once(
                    ChatParams {
                        model: &req.model,
                        messages: &messages,
                        tools: Some(&tools),
                        temperature,
                        max_tokens: cap_tokens(req.max_tokens),
                        top_p: req.top_p,
                        extra: &req.extra_body,
                    },
                    deadline.saturating_duration_since(tokio::time::Instant::now()),
                )
                .await
            {
                Ok(t) => {
                    parse_fail_streak = 0;
                    t
                }
                Err(e) if is_tool_call_parse_error(&e) => {
                    parse_fail_streak += 1;
                    if parse_fail_streak > PARSE_FAIL_LIMIT {
                        tracing::error!(
                            provider = %self.name, turn = turn_idx,
                            streak = parse_fail_streak,
                            "вызов не разбирается подряд — прекращаю прогон"
                        );
                        return Err(LlmError::WithUsage {
                            error: Box::new(e),
                            tokens_in: total_in,
                            tokens_out: total_out,
                            cost: total_cost,
                        });
                    }
                    tracing::warn!(
                        provider = %self.name, turn = turn_idx,
                        streak = parse_fail_streak, error = %e,
                        "сервер не разобрал вызов инструмента — повторяю ход"
                    );
                    transcript.write(&json!({
                        "event": "tool_call_parse_retry",
                        "turn": turn_idx,
                        "streak": parse_fail_streak,
                        "error": truncate(&e.to_string(), 500),
                    }));
                    // Срезаем последний ход целиком: сервер спотыкается о то, что
                    // уже лежит в истории, и без этого повтор шлёт ему ровно тот
                    // же неразбираемый текст — три попытки лягут за миллисекунды.
                    // Убираем результаты инструментов и породивший их вызов;
                    // остальная работа агента сохраняется.
                    while matches!(
                        messages.last().and_then(|m| m["role"].as_str()),
                        Some("tool")
                    ) {
                        messages.pop();
                    }
                    if matches!(
                        messages.last().and_then(|m| m["role"].as_str()),
                        Some("assistant")
                    ) {
                        messages.pop();
                    }
                    messages.push(json!({
                        "role": "user",
                        "content": "Твой прошлый вызов инструмента не удалось разобрать: \
                             в аргументе поехало экранирование (незакрытая кавычка). Ответ \
                             потерян — повтори вызов заново, замысел не меняй. Внутри \
                             текстового аргумента двойные кавычки экранируй как \\\", \
                             переносы строк — как \\n, обратный слэш — как \\\\."
                    }));
                    // При temperature=0 модель на тот же контекст выдаст тот же
                    // испорченный вызов — повтор без сдвига бесполезен.
                    if temperature < 0.1 {
                        temperature = 0.1;
                    }
                    continue;
                }
                Err(LlmError::Loop {
                    // Имя провайдера здесь уже не нужно: ход не пробрасывается
                    // наружу, а лечится на месте — навыком либо просьбой сменить
                    // замысел. Наружу ошибка уходит уже из общей ветки, когда
                    // попытки исчерпаны.
                    provider: _,
                    channel,
                    reason,
                    line,
                    repeats,
                    reasoning_chars,
                    content_chars,
                    tail,
                }) if loop_fixes < LOOP_FIX_LIMIT => {
                    loop_fixes += 1;
                    // Ищем навык по хвосту оборванного канала. Каталог из начала
                    // вызова тут не годится: он собран по фразе задания, а модель
                    // могла встать на совсем другой теме — замер 31.08.2026:
                    // навык про пустой отбор по статусу перечисления по фразе
                    // задания не входит и в первую восьмёрку, а по хвосту
                    // размышления встаёт первым.
                    let found = match &req.skills {
                        Some(sk) => {
                            let catalog = sk.skill_catalog(&loop_query(&line, &tail), 5).await;
                            pick_skill(&catalog, &skills_shown)
                        }
                        None => None,
                    };
                    // Оценка реранкера у найденного: ниже порога — это не
                    // подсказка по теме, а случайный сосед по каталогу.
                    let rr = found.as_ref().map(|(_, rr)| *rr);
                    let found = found.filter(|(_, rr)| *rr >= LOOP_SKILL_MIN_RR);
                    // Поиск не дал ничего (сервис навыков молчит, каталог пуст, либо
                    // всё найденное не по теме) — берём то, что припасено по
                    // фразе задания.
                    let by_tail = found.is_some();
                    let name = found.map(|(n, _)| n).or_else(|| {
                        req.fallback_skill_names
                            .iter()
                            .find(|n| !skills_shown.iter().any(|u| u == *n))
                            .cloned()
                    });
                    let body = match (&req.skills, &name) {
                        (Some(sk), Some(n)) => sk.skill_body(n).await,
                        _ => None,
                    };
                    let (Some(name), Some(body)) = (name, body) else {
                        // Подкладывать нечего. Раньше прогон здесь падал, но
                        // отсутствие навыка — не повод бросать ход: при жадной
                        // выборке повтор на неизменившемся контексте даст ту же
                        // петлю, а вот прямая просьба сменить замысел вместе со
                        // сдвигом температуры даёт модели выход. Попытка при
                        // этом расходуется — иначе ход повторялся бы без конца.
                        tracing::warn!(
                            provider = %self.name, turn = turn_idx,
                            fix = loop_fixes, rr = rr.unwrap_or(f64::NAN),
                            channel = %channel, reason = %reason,
                            "навыка по теме не нашлось — прошу сменить замысел и повторяю"
                        );
                        transcript.write(&json!({
                            "event": "loop_nudge_retry",
                            "turn": turn_idx,
                            "fix": loop_fixes,
                            "rerank": rr,
                            "channel": channel,
                            "reason": reason,
                            "line": line,
                        }));
                        if temperature < LOOP_NUDGE_TEMP {
                            temperature = LOOP_NUDGE_TEMP;
                        }
                        messages.push(json!({
                            "role": "user",
                            "content": format!(
                                "Твой прошлый ход прерван: {reason}. Справки по этой теме \
                                 у меня нет, и повторять прежний ход бессмысленно — тот же \
                                 подход даст тот же результат. Смени замысел: если бился \
                                 над одним запросом, возьми другой источник данных или \
                                 другой способ отбора; если не хватает сведений о базе, \
                                 запроси структуру объекта инструментом, а не гадай."
                            )
                        }));
                        continue;
                    };
                    tracing::warn!(
                        provider = %self.name, turn = turn_idx, skill = %name,
                        repeats, channel = %channel, line = %line, reason = %reason,
                        reasoning_chars, content_chars, fix = loop_fixes,
                        found_by_tail = by_tail, rr = rr.unwrap_or(f64::NAN),
                        "ход прерван предохранителем — подкладываю тело навыка и повторяю"
                    );
                    transcript.write(&json!({
                        "event": "loop_skill_retry",
                        "turn": turn_idx,
                        "skill": name,
                        "fix": loop_fixes,
                        "found_by_tail": by_tail,
                        // Оценка реранкера у выбранного навыка: по ней потом
                        // видно, чем полезная подкладка отличалась от мусорной,
                        // и уточняется LOOP_SKILL_MIN_RR.
                        "rerank": rr,
                        "channel": channel,
                        "reason": reason,
                        "repeats": repeats,
                        "line": line,
                    }));
                    skills_shown.push(name.clone());
                    // История цела: прерванный ход в неё не попал (сообщение
                    // модели добавляется только после успешного обращения), —
                    // срезать, в отличие от ветки с неразобранным вызовом,
                    // нечего. Достаточно добавить справку и повторить ход: при
                    // temperature=0 контекст обязан измениться, иначе модель
                    // выдаст ту же петлю знак в знак.
                    messages.push(json!({
                        "role": "user",
                        "content": format!(
                            "Твой прошлый ход прерван: одна и та же строка повторялась \
                             десятки раз подряд. Переписывать её заново бессмысленно — \
                             менять надо замысел. Ниже справка по теме задания; прочитай \
                             её и продолжай с учётом написанного.\n\n### Навык: {name}\n\n{body}"
                        )
                    }));
                    continue;
                }
                Err(e) => {
                    // Снимок того, ИЗ ЧЕГО сложился отвергнутый запрос. Без него
                    // от падения оставался лишь текст ошибки: 30.08.2026 вызов
                    // упал на 140 797 токенах, и разобрать, что именно раздуло
                    // историю, было нечем — ходы в базу не попадали вовсе.
                    // Пишем не сам текст (он огромен), а состав: сколько
                    // сообщений, какого размера, по ролям.
                    transcript.write(&json!({
                        "event": "provider_error",
                        "turn": turn_idx,
                        "error": truncate(&e.to_string(), 1000),
                        "request_shape": describe_messages(&messages),
                    }));
                    return Err(LlmError::WithUsage {
                        error: Box::new(e),
                        tokens_in: total_in,
                        tokens_out: total_out,
                        cost: total_cost,
                    });
                }
            };
            // Подъём температуры действует ровно на один ход — тот, что должен
            // выйти из колеи. Дальше снова детерминированно.
            temperature = req.temperature;
            total_in += turn.tokens_in;
            total_cached_in += turn.cached_in;
            total_out += turn.tokens_out;
            total_cost = pricing::add_cost(total_cost, turn.cost);
            last_finish = turn.finish_reason.clone();
            // Занятость окна на этом ходу — основание для решения о вытеснении
            // перед следующим.
            last_tokens_in = turn.tokens_in;

            // Per-turn наблюдаемость: имена инструментов и индекс хода уходят в
            // журнал отдельной строкой. Ходы целиком пишутся в таблицу
            // agent_turns, поэтому здесь — только отладочный след, по умолчанию
            // выключенный.
            let tool_names: Vec<&str> = turn
                .tool_calls
                .iter()
                .map(|tc| tc.function.name.as_str())
                .collect();
            tracing::debug!(
                provider = %self.name,
                model = %req.model,
                turn = turn_idx,
                max_turns = max_turns,
                tool_calls = turn.tool_calls.len(),
                tools = ?tool_names,
                finish = %last_finish,
                "agentic turn"
            );

            transcript.write(&json!({
                "event": "turn",
                "turn": turn_idx,
                "content": turn.content,
                "reasoning": turn.reasoning,
                "finish": last_finish,
                "tool_calls": turn
                    .tool_calls
                    .iter()
                    .map(|tc| json!({"name": tc.function.name, "arguments": tc.function.arguments}))
                    .collect::<Vec<_>>(),
                "tokens_in": turn.tokens_in,
                "tokens_out": turn.tokens_out,
                "duration_ms": turn_started.elapsed().as_millis() as u64,
            }));

            if turn.tool_calls.is_empty() {
                transcript.write(&json!({"event": "final", "turn": turn_idx}));
                // Финальный ответ.
                let mut resp = self.to_response(ResponseParts {
                    content: turn.content.unwrap_or_default(),
                    finish_reason: last_finish,
                    tokens_in: total_in,
                    cached_in: total_cached_in,
                    tokens_out: total_out,
                    cost_usd: total_cost,
                    reasoning: turn.reasoning,
                });
                resp.transcript = transcript.take();
                return Ok(resp);
            }

            // Эхо assistant-сообщения с tool_calls (обязательно для контракта).
            // Аргументы, которые сами не разбираются как JSON, в историю НЕ
            // кладём: llama-server парсит их и у входящего запроса тоже, и
            // отвечает HTTP 500 на КАЖДЫЙ следующий ход — диалог отравлен
            // навсегда, повтор бесполезен (поймано 2026-07-23: три повтора
            // подряд легли за 5 мс, даже не дойдя до модели). Вместо испорченной
            // строки кладём пустой объект, а модели ниже объясняем, что вызов не
            // разобрался.
            let tc_echo: Vec<Value> = turn
                .tool_calls
                .iter()
                .map(|tc| {
                    json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": safe_arguments(&tc.function.arguments),
                        }
                    })
                })
                .collect();
            let mut assistant = json!({
                "role": "assistant",
                "content": turn.content,
                "tool_calls": tc_echo,
            });
            if let Some(reasoning) = turn.reasoning.as_deref().filter(|s| !s.is_empty()) {
                assistant["reasoning_content"] = Value::String(reasoning.to_string());
            }
            messages.push(assistant);

            // Исполняем каждый tool_call и дописываем результат как role=tool.
            for tc in &turn.tool_calls {
                let parsed = serde_json::from_str::<Value>(&tc.function.arguments)
                    .map_err(|error| error.to_string());
                // Длительность вызова инструмента меряем явно, а не вычитанием
                // соседних меток времени: если следующий ход не состоится
                // (таймаут, отказ провайдера), вычитать будет не из чего — а это
                // ровно те случаи, ради разбора которых журнал и ведётся.
                let tool_started = std::time::Instant::now();
                let result = execute_tool_call(
                    &self.client,
                    &mut mcp_tools,
                    &mut tool_state,
                    &mut temperature,
                    ToolCallInput {
                        provider: &self.name,
                        hints: req.cli_hints.as_ref(),
                        tool_name: &tc.function.name,
                        raw_arguments: &tc.function.arguments,
                        parsed_arguments: parsed,
                    },
                )
                .await;

                if transcript.enabled() {
                    transcript.write(&json!({
                        "event": "tool_result",
                        "turn": turn_idx,
                        "tool": tc.function.name,
                        "tool_call_id": tc.id,
                        "duration_ms": tool_started.elapsed().as_millis() as u64,
                        "result_chars": result.chars().count(),
                        "result": truncate_str(&result, 50_000),
                    }));
                }
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": clamp_tool_result(&result),
                }));
            }
        }

        transcript.write(&json!({
            "event": "max_turns",
            "max_turns": max_turns,
            "finish": last_finish,
            "tokens_in": total_in,
            "tokens_out": total_out,
        }));
        Err(LlmError::MaxTurns {
            provider: self.name.clone(),
            turns: max_turns,
            finish_reason: last_finish,
            tokens_in: total_in,
            tokens_out: total_out,
            cost: total_cost,
            transcript: transcript.take(),
        })
    }
}

fn map_reqwest_err(e: reqwest::Error) -> LlmError {
    if e.is_timeout() {
        LlmError::Timeout
    } else {
        LlmError::Provider(format!("openrouter: {e}"))
    }
}

fn retry_after_seconds(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let value = headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)?;
    value.parse().ok().or_else(|| {
        let retry_at = chrono::DateTime::parse_from_rfc2822(value).ok()?;
        Some(
            (retry_at.with_timezone(&chrono::Utc) - chrono::Utc::now())
                .num_seconds()
                .max(0) as u64,
        )
    })
}

/// Перевести потолок выдачи из конфига агента в поле запроса.
/// `0` — условленное «не ограничивать»: поле не уходит на сервер, и тот пишет,
/// пока хватает окна. Любое другое значение отправляется как есть.
fn cap_tokens(max_tokens: u32) -> Option<u32> {
    if max_tokens == 0 {
        None
    } else {
        Some(max_tokens)
    }
}

/// Разложить историю диалога по ролям: сколько сообщений и какого размера.
/// Нужно, чтобы после отказа сервера (переполнение окна и т.п.) было видно, ЧЕМ
/// набрался запрос, а не только его итоговый размер в токенах. Отдельно —
/// пятёрка самых крупных сообщений с ролью и именем инструмента: обычно
/// раздувает история одна-две тяжёлые выборки, и по этому списку они видны сразу.
fn describe_messages(messages: &[Value]) -> Value {
    use std::collections::BTreeMap;
    let mut by_role: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut sizes: Vec<(usize, &str, String)> = Vec::new();
    for m in messages {
        let role = m["role"].as_str().unwrap_or("?");
        let len = m["content"]
            .as_str()
            .map(|s| s.chars().count())
            .unwrap_or(0);
        let e = by_role.entry(role).or_insert((0, 0));
        e.0 += 1;
        e.1 += len;
        let tool = m["tool_call_id"].as_str().unwrap_or("").to_string();
        sizes.push((len, role, tool));
    }
    sizes.sort_by(|a, b| b.0.cmp(&a.0));
    let roles: Vec<Value> = by_role
        .iter()
        .map(|(role, (n, chars))| json!({"роль": role, "сообщений": n, "знаков": chars}))
        .collect();
    let largest: Vec<Value> = sizes
        .iter()
        .take(5)
        .map(|(len, role, tool)| json!({"роль": role, "знаков": len, "tool_call_id": tool}))
        .collect();
    json!({
        "сообщений_всего": messages.len(),
        "знаков_всего": messages.iter().map(|m| m["content"].as_str().map(|s| s.chars().count()).unwrap_or(0)).sum::<usize>(),
        "по_ролям": roles,
        "самые_крупные": largest,
    })
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

/// Распарсить текстовый формат tool-call (MiMo/Qwen/Hermes-стиль) из content.
/// Формат блока:
/// `<tool_call><function=ИМЯ><parameter=КЛЮЧ>ЗНАЧЕНИЕ</parameter>…</function></tool_call>`.
/// Значение параметра, если оно валидный JSON, кладётся как JSON; иначе как строка.
/// Возвращает пустой вектор, если ни одного корректного блока не найдено.
fn parse_text_tool_calls(content: &str) -> Vec<ToolCall> {
    let mut out = Vec::new();
    let mut rest = content;
    let mut idx = 0usize;
    while let Some(start) = rest.find("<tool_call>") {
        let after_open = &rest[start + "<tool_call>".len()..];
        let Some(end_rel) = after_open.find("</tool_call>") else {
            break;
        };
        let block = &after_open[..end_rel];
        rest = &after_open[end_rel + "</tool_call>".len()..];

        // Имя функции: <function=ИМЯ>
        let Some(fn_start) = block.find("<function=") else {
            continue;
        };
        let after_fn = &block[fn_start + "<function=".len()..];
        let Some(name_end) = after_fn.find('>') else {
            continue;
        };
        let name = after_fn[..name_end].trim().to_string();
        if name.is_empty() {
            continue;
        }

        // Параметры: <parameter=КЛЮЧ>ЗНАЧЕНИЕ</parameter>
        let mut args = serde_json::Map::new();
        let mut prest = &after_fn[name_end + 1..];
        while let Some(p_start) = prest.find("<parameter=") {
            let after_p = &prest[p_start + "<parameter=".len()..];
            let Some(key_end) = after_p.find('>') else {
                break;
            };
            let key = after_p[..key_end].trim().to_string();
            let val_part = &after_p[key_end + 1..];
            let Some(val_end) = val_part.find("</parameter>") else {
                break;
            };
            let raw_val = val_part[..val_end].trim();
            let value = serde_json::from_str::<Value>(raw_val)
                .unwrap_or_else(|_| Value::String(raw_val.to_string()));
            args.insert(key, value);
            prest = &val_part[val_end + "</parameter>".len()..];
        }

        out.push(ToolCall {
            id: format!("call_text_{idx}"),
            function: ToolCallFunction {
                name,
                arguments: Value::Object(args).to_string(),
            },
        });
        idx += 1;
    }
    out
}

/// Вырезать из текста все блоки `<tool_call>…</tool_call>` — оставшийся текст
/// (рассуждения модели вокруг вызова) сохраняем как content.
fn strip_text_tool_calls(content: &str) -> String {
    let mut out = String::new();
    let mut rest = content;
    while let Some(start) = rest.find("<tool_call>") {
        out.push_str(&rest[..start]);
        let after_open = &rest[start..];
        if let Some(end_rel) = after_open.find("</tool_call>") {
            rest = &after_open[end_rel + "</tool_call>".len()..];
        } else {
            rest = "";
            break;
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_provider(url: String) -> OpenRouterProvider {
        OpenRouterProvider::new(OpenRouterOptions {
            name: "fixture".into(),
            api_key: "synthetic-key".into(),
            base_url: Some(url),
            referer: None,
            proxy: None,
            proxy_bypass: None,
            max_concurrent: None,
            prices: BTreeMap::new(),
        })
    }

    fn test_provider_with_price(url: String) -> OpenRouterProvider {
        let mut prices = BTreeMap::new();
        prices.insert(
            "review-model".into(),
            ModelPrice {
                input: 2.0,
                output: 8.0,
                cache_read: None,
                cache_write: None,
            },
        );
        OpenRouterProvider::new(OpenRouterOptions {
            name: "fixture".into(),
            api_key: "synthetic-key".into(),
            base_url: Some(url),
            referer: None,
            proxy: None,
            proxy_bypass: None,
            max_concurrent: None,
            prices,
        })
    }

    fn test_temp_dir() -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("agents-mcp-openrouter-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn stream_files_are_sanitized_unique_and_inside_directory() {
        let dir = test_temp_dir();
        let first = open_stream_files(&dir, "qwen/qwen3-coder:free");
        let second = open_stream_files(&dir, r"C:\models\x.gguf");
        let third = open_stream_files(&dir, "qwen/qwen3-coder:free");
        drop((first, second, third));

        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| {
                let path = entry.unwrap().path();
                assert_eq!(path.parent(), Some(dir.as_path()));
                path.file_name().unwrap().to_string_lossy().into_owned()
            })
            .collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 6, "каждый вызов создаёт отдельную пару файлов");
        assert!(names
            .iter()
            .all(|name| !name.chars().any(|ch| matches!(ch, ':' | '/' | '\\'))));
        assert!(names
            .iter()
            .any(|name| name.contains("qwen_qwen3-coder_free")));
        assert!(names.iter().any(|name| name.contains("C__models_x.gguf")));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn creating_stream_files_removes_only_expired_recordings() {
        let dir = test_temp_dir();
        let old = dir.join("old.stream.bsl");
        let fresh = dir.join("fresh.raw.jsonl");
        std::fs::write(&old, "old").unwrap();
        std::fs::write(&fresh, "fresh").unwrap();
        let old_time = SystemTime::now() - STREAM_RETENTION - Duration::from_secs(1);
        std::fs::File::options()
            .write(true)
            .open(&old)
            .and_then(|file| file.set_modified(old_time))
            .unwrap();

        let files = open_stream_files(&dir, "model");
        drop(files);
        assert!(!old.exists(), "просроченная запись должна быть удалена");
        assert!(fresh.exists(), "свежая запись должна остаться");
        let _ = std::fs::remove_dir_all(dir);
    }

    async fn test_stream(chunks: Vec<Vec<u8>>) -> reqwest::Response {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 4096];
            let _ = socket.read(&mut request).await.unwrap();
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            for chunk in chunks {
                socket
                    .write_all(format!("{:x}\r\n", chunk.len()).as_bytes())
                    .await
                    .unwrap();
                socket.write_all(&chunk).await.unwrap();
                socket.write_all(b"\r\n").await.unwrap();
                socket.flush().await.unwrap();
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            let _ = socket.write_all(b"0\r\n\r\n").await;
        });
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .get(url)
            .send()
            .await
            .unwrap()
    }

    struct TestApi {
        base: String,
        seen: std::sync::Arc<std::sync::Mutex<Vec<Value>>>,
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        task: tokio::task::JoinHandle<()>,
    }

    impl Drop for TestApi {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    async fn test_api(turns: Vec<Value>) -> TestApi {
        test_api_with_tool(turns, "poll", vec!["progress=1".into()]).await
    }

    async fn test_api_with_tool(
        turns: Vec<Value>,
        tool_name: &str,
        tool_results: Vec<String>,
    ) -> TestApi {
        use axum::{
            routing::{get, post},
            Json, Router,
        };

        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let queue = std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::from(
            turns,
        )));
        let tool_results = std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::VecDeque::from(tool_results),
        ));
        let tool_name = tool_name.to_string();
        let seen_requests = seen.clone();
        let calls_server = calls.clone();
        let app = Router::new()
            .route(
                "/props",
                get(|| async { Json(json!({"default_generation_settings": {"n_ctx": 100_000}})) }),
            )
            .route(
                "/chat/completions",
                post(move |Json(body): Json<Value>| {
                    let seen_requests = seen_requests.clone();
                    let queue = queue.clone();
                    async move {
                        seen_requests.lock().unwrap().push(body);
                        Json(
                            queue
                                .lock()
                                .unwrap()
                                .pop_front()
                                .expect("неожиданный запрос к модели"),
                        )
                    }
                }),
            )
            .route(
                "/mcp",
                post(move |Json(body): Json<Value>| {
                    let calls = calls_server.clone();
                    let tool_results = tool_results.clone();
                    let tool_name = tool_name.clone();
                    async move {
                        let result = match body["method"].as_str().unwrap_or("") {
                            "initialize" => json!({
                                "protocolVersion": "2025-06-18",
                                "capabilities": {},
                                "serverInfo": {"name": "fixture", "version": "1"}
                            }),
                            "tools/list" => json!({"tools": [{
                                "name": tool_name,
                                "description": "Read status",
                                "inputSchema": {"type": "object"}
                            }]}),
                            "tools/call" => {
                                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                let result = tool_results
                                    .lock()
                                    .unwrap()
                                    .pop_front()
                                    .expect("неожиданный вызов инструмента");
                                json!({"content": [{"type": "text", "text": result}]})
                            }
                            _ => json!({}),
                        };
                        Json(json!({"jsonrpc": "2.0", "id": body["id"], "result": result}))
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        TestApi {
            base,
            seen,
            calls,
            task,
        }
    }

    fn test_request() -> LlmRequest {
        LlmRequest {
            model: "review-model".into(),
            system_prompt: "Synthetic review fixture".into(),
            user_input: String::new(),
            temperature: 0.0,
            max_tokens: 100,
            top_p: None,
            extra_body: Default::default(),
            timeout: Duration::from_secs(3),
            cli_hints: None,
            turn_sink: None,
            fallback_skill_names: Vec::new(),
            prompt_skill_names: Vec::new(),
            skills: None,
        }
    }

    fn priced_response(cost: Option<f64>) -> Value {
        let mut usage = json!({"prompt_tokens": 1_000_000, "completion_tokens": 100_000});
        if let Some(cost) = cost {
            usage["cost"] = json!(cost);
        }
        json!({
            "choices": [{
                "message": {"content": "finished"},
                "finish_reason": "stop"
            }],
            "usage": usage
        })
    }

    #[tokio::test]
    async fn response_cost_has_priority_over_config_price() {
        let api = test_api(vec![priced_response(Some(0.25))]).await;
        let result = test_provider_with_price(api.base.clone())
            .complete(test_request())
            .await
            .unwrap();
        assert_eq!(result.cost_usd, Some(0.25));
    }

    #[tokio::test]
    async fn config_price_is_used_without_response_cost() {
        let api = test_api(vec![priced_response(None)]).await;
        let result = test_provider_with_price(api.base.clone())
            .complete(test_request())
            .await
            .unwrap();
        let cost = result.cost_usd.expect("стоимость известна из конфига");
        assert!((cost - 2.8).abs() < 1e-12, "cost={cost}");
    }

    #[tokio::test]
    async fn cost_is_unknown_without_response_or_config_price() {
        let api = test_api(vec![priced_response(None)]).await;
        let result = test_provider(api.base.clone())
            .complete(test_request())
            .await
            .unwrap();
        assert_eq!(result.cost_usd, None);
    }

    #[tokio::test]
    async fn rate_limit_retries_but_forbidden_does_not() {
        use axum::response::IntoResponse;
        use axum::routing::post;
        use axum::Router;

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_server = calls.clone();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let calls = calls_server.clone();
                async move {
                    if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                        (
                            axum::http::StatusCode::TOO_MANY_REQUESTS,
                            [("Retry-After", "0")],
                            "{}",
                        )
                            .into_response()
                    } else {
                        axum::Json(test_final_turn()).into_response()
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let provider = OpenRouterProvider::new(OpenRouterOptions {
            name: "retry".into(),
            api_key: "key".into(),
            base_url: Some(base),
            referer: None,
            proxy: None,
            proxy_bypass: None,
            max_concurrent: None,
            prices: BTreeMap::new(),
        });
        assert_eq!(
            provider.complete(test_request()).await.unwrap().content,
            "finished"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        task.abort();

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_server = calls.clone();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let calls = calls_server.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    axum::http::StatusCode::FORBIDDEN
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let provider = OpenRouterProvider::new(OpenRouterOptions {
            name: "forbidden".into(),
            api_key: "key".into(),
            base_url: Some(base),
            referer: None,
            proxy: None,
            proxy_bypass: None,
            max_concurrent: None,
            prices: BTreeMap::new(),
        });
        assert!(provider.complete(test_request()).await.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        task.abort();
    }

    #[tokio::test]
    async fn provider_max_concurrent_one_serializes_calls() {
        use axum::routing::post;
        use axum::{Json, Router};

        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let active_server = active.clone();
        let peak_server = peak.clone();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let active = active_server.clone();
                let peak = peak_server.clone();
                async move {
                    let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(80)).await;
                    active.fetch_sub(1, Ordering::SeqCst);
                    Json(test_final_turn())
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let provider = OpenRouterProvider::new(OpenRouterOptions {
            name: "limited".into(),
            api_key: "key".into(),
            base_url: Some(base),
            referer: None,
            proxy: None,
            proxy_bypass: None,
            max_concurrent: Some(1),
            prices: BTreeMap::new(),
        });
        let (one, two) = tokio::join!(
            provider.complete(test_request()),
            provider.complete(test_request())
        );
        one.unwrap();
        two.unwrap();
        assert_eq!(peak.load(Ordering::SeqCst), 1);
        task.abort();
    }

    fn test_tool_turn() -> Value {
        test_tool_turn_named(1, "mcp__fixture__poll", "{}")
    }

    fn test_tool_turn_named(id: usize, tool_name: &str, arguments: &str) -> Value {
        json!({
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": "Synthetic reasoning field",
                    "tool_calls": [{
                        "id": format!("t{id}"),
                        "type": "function",
                        "function": {"name": tool_name, "arguments": arguments}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7, "cost": 0.25}
        })
    }

    fn test_final_turn() -> Value {
        json!({"choices": [{
            "message": {"content": "finished"},
            "finish_reason": "stop"
        }]})
    }

    fn test_agentic_request(base: &str, max_turns: u32) -> LlmRequest {
        let mut req = test_request();
        req.cli_hints = Some(ClaudeCliHints {
            mcp_config: Some(
                json!({"mcpServers": {"fixture": {"url": format!("{base}/mcp")}}}).to_string(),
            ),
            max_turns: Some(max_turns),
            ..Default::default()
        });
        req
    }

    fn history_contains_blocked(api: &TestApi) -> bool {
        api.seen.lock().unwrap().last().unwrap()["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|message| {
                message["content"]
                    .as_str()
                    .is_some_and(|content| content.starts_with("ЗАБЛОКИРОВАНО"))
            })
    }

    #[tokio::test]
    async fn server_wide_allowed_tools_publish_all_server_tools() {
        for allowed in ["mcp__fixture", "mcp__fixture__*"] {
            let api = test_api(Vec::new()).await;
            let hints = ClaudeCliHints {
                mcp_config: Some(
                    json!({"mcpServers": {"fixture": {"url": format!("{}/mcp", api.base)}}})
                        .to_string(),
                ),
                allowed_tools: vec![allowed.to_string()],
                ..Default::default()
            };
            let (tools, registry, _) = test_provider(api.base.clone())
                .build_tools(&hints)
                .await
                .unwrap();
            assert_eq!(tools.len(), 1, "шаблон {allowed}");
            assert!(registry.contains_key("mcp__fixture__poll"));
        }
    }

    #[tokio::test]
    async fn disallowed_tools_are_not_published() {
        let api = test_api(Vec::new()).await;
        let hints = ClaudeCliHints {
            mcp_config: Some(
                json!({"mcpServers": {"fixture": {"url": format!("{}/mcp", api.base)}}})
                    .to_string(),
            ),
            disallowed_tools: vec!["mcp__fixture__poll".into()],
            ..Default::default()
        };
        let (tools, registry, _) = test_provider(api.base.clone())
            .build_tools(&hints)
            .await
            .unwrap();
        assert!(tools.is_empty());
        assert!(!registry.contains_key("mcp__fixture__poll"));
    }

    #[tokio::test]
    async fn unknown_allowed_tool_rejects_run() {
        let api = test_api(Vec::new()).await;
        let hints = ClaudeCliHints {
            mcp_config: Some(
                json!({"mcpServers": {"fixture": {"url": format!("{}/mcp", api.base)}}})
                    .to_string(),
            ),
            allowed_tools: vec!["mcp__fixture__missing".into()],
            ..Default::default()
        };
        let error = match test_provider(api.base.clone()).build_tools(&hints).await {
            Ok(_) => panic!("пустой белый список обязан отклонить вызов"),
            Err(error) => error,
        };
        assert!(matches!(error, LlmError::ToolsUnavailable(_)));
        assert!(error.to_string().contains("mcp__fixture__missing"));
    }

    #[tokio::test]
    async fn malformed_mcp_config_rejects_before_model_call() {
        let api = test_api(vec![test_final_turn()]).await;
        let mut req = test_request();
        req.cli_hints = Some(ClaudeCliHints {
            mcp_config: Some("{broken".into()),
            allowed_tools: vec!["mcp__fixture__poll".into()],
            ..Default::default()
        });
        let error = test_provider(api.base.clone())
            .complete(req)
            .await
            .expect_err("негодный JSON обязан отклонить вызов");
        assert!(matches!(error, LlmError::ToolsUnavailable(_)));
        assert!(api.seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn self_named_disallowed_tool_is_not_executed() {
        let api = test_api(vec![
            test_tool_turn_named(1, "mcp__fixture__blocked", "{}"),
            test_final_turn(),
        ])
        .await;
        let mut req = test_agentic_request(&api.base, 2);
        req.cli_hints
            .as_mut()
            .unwrap()
            .disallowed_tools
            .push("mcp__fixture__blocked".into());
        test_provider(api.base.clone()).complete(req).await.unwrap();
        assert_eq!(api.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        let seen = api.seen.lock().unwrap();
        assert!(seen[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|message| {
                message["content"]
                    .as_str()
                    .is_some_and(|content| content.contains("disallowed_tools"))
            }));
    }

    #[tokio::test]
    async fn stream_keeps_utf8_split_between_chunks() {
        let bytes = "data: {\"choices\":[{\"delta\":{\"content\":\"Я\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".as_bytes();
        let split = bytes.iter().position(|b| *b == 0xd0).unwrap() + 1;
        let resp = test_stream(vec![bytes[..split].to_vec(), bytes[split..].to_vec()]).await;
        let turn = test_provider("unused".into())
            .read_stream(resp, "test", &test_temp_dir(), Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(turn.content.as_deref(), Some("Я"));
    }

    #[tokio::test]
    async fn stream_error_event_is_provider_error() {
        let body = b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\ndata: {\"error\":{\"message\":\"failed\"},\"choices\":[{\"finish_reason\":\"error\",\"delta\":{}}]}\n\ndata: [DONE]\n\n";
        let result = test_provider("unused".into())
            .read_stream(
                test_stream(vec![body.to_vec()]).await,
                "test",
                &test_temp_dir(),
                Duration::from_secs(2),
            )
            .await;
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("событие error не должно считаться успехом"),
        };
        assert!(matches!(error, LlmError::Provider(message) if message.contains("failed")));
    }

    #[tokio::test]
    async fn text_tool_call_is_parsed_after_stream() {
        let text = "<tool_call><function=mcp__fixture__poll><parameter=x>1</parameter></function></tool_call>";
        let body = format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]})
        );
        let mut turn = test_provider("unused".into())
            .read_stream(
                test_stream(vec![body.into_bytes()]).await,
                "test",
                &test_temp_dir(),
                Duration::from_secs(2),
            )
            .await
            .unwrap();
        normalize_text_tool_calls("fixture", &mut turn);
        assert_eq!(turn.tool_calls.len(), 1);
        assert!(turn.content.is_none());
    }

    #[tokio::test]
    async fn reasoning_is_returned_in_next_request_history() {
        let api = test_api(vec![
            test_tool_turn(),
            json!({"choices": [{
                "message": {"content": "finished"},
                "finish_reason": "stop"
            }]}),
        ])
        .await;
        let mut req = test_request();
        req.cli_hints = Some(ClaudeCliHints {
            mcp_config: Some(
                json!({"mcpServers": {"fixture": {"url": format!("{}/mcp", api.base)}}})
                    .to_string(),
            ),
            ..Default::default()
        });
        test_provider(api.base.clone()).complete(req).await.unwrap();
        let seen = api.seen.lock().unwrap();
        let assistant = seen[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|message| message["role"] == "assistant")
            .unwrap();
        assert_eq!(assistant["reasoning_content"], "Synthetic reasoning field");
    }

    #[tokio::test]
    async fn fourth_observation_poll_with_changing_results_is_not_blocked() {
        let mut turns = (1..=4)
            .map(|id| {
                test_tool_turn_named(
                    id,
                    "mcp__fixture__wait_agent",
                    r#"{"call_id":17,"wait_sec":0}"#,
                )
            })
            .collect::<Vec<_>>();
        turns.push(test_final_turn());
        let api = test_api_with_tool(
            turns,
            "wait_agent",
            (1..=4).map(|n| format!("progress={n}")).collect(),
        )
        .await;

        test_provider(api.base.clone())
            .complete(test_agentic_request(&api.base, 5))
            .await
            .unwrap();

        assert_eq!(api.calls.load(std::sync::atomic::Ordering::SeqCst), 4);
        assert!(!history_contains_blocked(&api));
    }

    #[tokio::test]
    async fn fourth_identical_non_observation_result_is_blocked() {
        let mut turns = (1..=4)
            .map(|id| test_tool_turn_named(id, "mcp__fixture__poll", "{}"))
            .collect::<Vec<_>>();
        turns.push(test_final_turn());
        let api = test_api_with_tool(
            turns,
            "poll",
            vec!["same".into(), "same".into(), "same".into()],
        )
        .await;

        test_provider(api.base.clone())
            .complete(test_agentic_request(&api.base, 5))
            .await
            .unwrap();

        assert_eq!(api.calls.load(std::sync::atomic::Ordering::SeqCst), 3);
        assert!(history_contains_blocked(&api));
    }

    #[tokio::test]
    async fn changing_non_observation_results_reset_repeat_counter() {
        let mut turns = (1..=4)
            .map(|id| test_tool_turn_named(id, "mcp__fixture__poll", "{}"))
            .collect::<Vec<_>>();
        turns.push(test_final_turn());
        let api = test_api_with_tool(
            turns,
            "poll",
            (1..=4).map(|n| format!("progress={n}")).collect(),
        )
        .await;

        test_provider(api.base.clone())
            .complete(test_agentic_request(&api.base, 5))
            .await
            .unwrap();

        assert_eq!(api.calls.load(std::sync::atomic::Ordering::SeqCst), 4);
        assert!(!history_contains_blocked(&api));
    }

    #[tokio::test]
    async fn non_stream_response_reads_reasoning_alias() {
        let api = test_api(vec![json!({"choices": [{
            "message": {"content": "finished", "reasoning": "thinking"},
            "finish_reason": "stop"
        }]})])
        .await;
        let messages = vec![json!({"role": "user", "content": "test"})];
        let extra = serde_json::Map::new();
        let turn = test_provider(api.base.clone())
            .chat_once(
                ChatParams {
                    model: "review-model",
                    messages: &messages,
                    tools: None,
                    temperature: 0.0,
                    max_tokens: Some(100),
                    top_p: None,
                    extra: &extra,
                },
                Duration::from_secs(2),
            )
            .await
            .unwrap();
        assert_eq!(turn.reasoning.as_deref(), Some("thinking"));
    }

    #[tokio::test]
    async fn stream_tool_calls_without_index_stay_separate() {
        let body = format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({"choices": [{
                "delta": {"tool_calls": [
                    {"id": "a", "function": {"name": "one", "arguments": "{}"}},
                    {"id": "b", "function": {"name": "two", "arguments": "{}"}}
                ]},
                "finish_reason": "tool_calls"
            }]})
        );
        let turn = test_provider("unused".into())
            .read_stream(
                test_stream(vec![body.into_bytes()]).await,
                "test",
                &test_temp_dir(),
                Duration::from_secs(2),
            )
            .await
            .unwrap();
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(turn.tool_calls[0].id, "a");
        assert_eq!(turn.tool_calls[1].id, "b");
    }

    #[tokio::test]
    async fn huge_stream_tool_call_index_is_rejected() {
        let body = format!(
            "data: {}\n\n",
            json!({"choices": [{
                "delta": {"tool_calls": [{
                    "index": 100_000,
                    "id": "a",
                    "function": {"name": "one", "arguments": "{}"}
                }]}
            }]})
        );
        let result = test_provider("unused".into())
            .read_stream(
                test_stream(vec![body.into_bytes()]).await,
                "test",
                &test_temp_dir(),
                Duration::from_secs(2),
            )
            .await;
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("огромный index должен быть отклонён"),
        };
        assert!(matches!(error, LlmError::InvalidResponse(message) if message.contains("100000")));
    }

    #[tokio::test]
    async fn provider_error_after_first_turn_keeps_usage() {
        let api = test_api(vec![
            test_tool_turn(),
            json!({"error": {"message": "second turn failed"}}),
        ])
        .await;
        let mut req = test_request();
        req.cli_hints = Some(ClaudeCliHints {
            mcp_config: Some(
                json!({"mcpServers": {"fixture": {"url": format!("{}/mcp", api.base)}}})
                    .to_string(),
            ),
            ..Default::default()
        });
        let error = test_provider(api.base.clone())
            .complete(req)
            .await
            .unwrap_err();
        match error {
            LlmError::WithUsage {
                error,
                tokens_in,
                tokens_out,
                cost,
            } => {
                assert!(matches!(*error, LlmError::Provider(_)));
                assert_eq!(tokens_in, 10);
                assert_eq!(tokens_out, 7);
                assert_eq!(cost, Some(0.25));
            }
            other => panic!("ожидалась ошибка со статистикой, получена: {other}"),
        }
    }

    /// Кусок реальной петли из сорванного прогона 30.08.2026 (call_id=8564):
    /// один и тот же блок шёл 28 раз подряд, отпечаток совпадал побайтно.
    const LOOP_BLOCK: &str = "\
Actually, should I even validate? The query was already validated in the previous round.\n\
Let me write the final code:\n\
\tЗапрос.УстановитьПараметр(\"ОсновнойМенеджер\", Строка(ОсновнойМенеджер));\n\
\tРезультатЗапроса = Запрос.Выполнить().Выгрузить();\n";

    fn feed(chunks: &[&str]) -> Option<(String, u32)> {
        let (mut tail, mut seen, mut pos) = (String::new(), HashMap::new(), 0usize);
        for c in chunks {
            if let Some(hit) = note_repeats(c, &mut tail, &mut seen, &mut pos) {
                return Some(hit);
            }
        }
        None
    }

    #[test]
    fn note_repeats_catches_loop_on_sixth_turn() {
        // Пять оборотов — ещё не петля: модель имеет право переписать код.
        let five: Vec<&str> = vec![LOOP_BLOCK; 5];
        assert!(feed(&five).is_none());
        let six: Vec<&str> = vec![LOOP_BLOCK; 6];
        let (line, n) = feed(&six).expect("шестой оборот должен быть пойман");
        assert_eq!(n, LOOP_REPEAT_LIMIT);
        assert!(line.chars().count() >= LOOP_LINE_MIN_CHARS);
    }

    #[test]
    fn note_repeats_survives_chunk_boundaries() {
        // Поток режется где угодно, в том числе посреди строки.
        let joined: String = LOOP_BLOCK.repeat(6);
        let (mut tail, mut seen, mut pos) = (String::new(), HashMap::new(), 0usize);
        let mut hit = None;
        let mut rest = joined.as_str();
        while !rest.is_empty() {
            // Режем по 13 символов, не по байтам — иначе разрежем кириллицу.
            let take = rest
                .char_indices()
                .nth(13)
                .map(|(i, _)| i)
                .unwrap_or(rest.len());
            let (head, tail_str) = rest.split_at(take);
            rest = tail_str;
            if let Some(h) = note_repeats(head, &mut tail, &mut seen, &mut pos) {
                hit = Some(h);
                break;
            }
        }
        assert!(
            hit.is_some(),
            "петля должна ловиться при любой нарезке потока"
        );
    }

    /// Случай ложного срабатывания 31.08.2026: строка
    /// `Результат = Запрос.Выполнить().Выгрузить();` встретилась шесть раз в
    /// шести РАЗНЫХ редакциях функции, разделённых тысячами знаков рассуждений
    /// (реальные разрывы: 722, 15 386, 674, 3152, 8826). Прогон был оборван зря.
    #[test]
    fn note_repeats_ignores_distant_repeats_of_same_line() {
        let line = "\tРезультат = Запрос.Выполнить().Выгрузить();\n";
        let gaps = [722usize, 15386, 674, 3152, 8826];
        let (mut tail, mut seen, mut pos) = (String::new(), HashMap::new(), 0usize);
        let mut hit = note_repeats(line, &mut tail, &mut seen, &mut pos);
        for gap in gaps {
            // Между вхождениями — рассуждения на gap знаков, разбитые на строки
            // длиннее порога, чтобы они сами не выглядели повтором.
            let filler: String = (0..gap / 60)
                .map(|i| format!("рассуждение номер {i} о том, как ещё можно построить запрос\n"))
                .collect();
            hit = hit.or(note_repeats(&filler, &mut tail, &mut seen, &mut pos));
            hit = hit.or(note_repeats(line, &mut tail, &mut seen, &mut pos));
        }
        assert!(
            hit.is_none(),
            "разнесённые повторы одной строки кода — это не петля, а перебор вариантов"
        );
    }

    /// Ложное срабатывание 19.09.2026: исполнитель описывал в размышлении
    /// структуру с шестью полями-списками, перед каждым — один и тот же атрибут
    /// serde длиннее порога. Шесть одинаковых строк в тысяче знаков, но соседи
    /// у них разные — это код, а не петля. Четыре запуска подряд были оборваны
    /// на первом ходу, и повтор ничего не менял.
    #[test]
    fn note_repeats_allows_one_repeated_line_between_distinct_ones() {
        let fields = [
            "data_sets",
            "links",
            "calculated_fields",
            "totals",
            "parameters",
            "variants",
        ];
        let text: String = fields
            .iter()
            .map(|f| {
                format!(
                    "    #[serde(skip_serializing_if = \"Vec::is_empty\")]\n    pub {f}: Vec<Dcs{f}Row>, // список из разбора схемы компоновки\n"
                )
            })
            .collect();
        assert!(
            feed(&[&text]).is_none(),
            "одна повторяющаяся строка с разными соседями — не петля"
        );
        // А та же строка десять раз ловится счётчиком за весь ход.
        let ten: String = text.repeat(2);
        let (line, n) = feed(&[&ten]).expect("десятый повтор одиночной строки — петля");
        assert!(line.contains("skip_serializing_if"));
        assert_eq!(n, LOOP_TOTAL_LIMIT);
    }

    #[test]
    fn note_repeats_ignores_short_lines() {
        // Короткие строки законно повторяются: отступы, КонецЕсли, пустые.
        let short = "\tКонецЕсли;\n\n\tКонецЦикла;\n";
        let many: Vec<&str> = vec![short; 30];
        assert!(feed(&many).is_none());
    }

    /// Строка, повторённая `times` раз с разрывом `gap` знаков между вхождениями.
    fn feed_with_gaps(line: &str, times: usize, gap: usize) -> Option<(String, u32)> {
        let (mut tail, mut seen, mut pos) = (String::new(), HashMap::new(), 0usize);
        let mut hit = None;
        for i in 0..times {
            hit = hit.or(note_repeats(line, &mut tail, &mut seen, &mut pos));
            let filler: String = (0..gap / 60)
                .map(|k| format!("рассуждение {i}-{k} о том, как ещё можно построить запрос\n"))
                .collect();
            hit = hit.or(note_repeats(&filler, &mut tail, &mut seen, &mut pos));
        }
        hit
    }

    /// Медленная петля 31.08.2026 (вызов 8599): строка встретилась 92 раза за ход
    /// на 298 727 знаков, но подряд — не больше двух, между вхождениями лежало по
    /// 6 300 знаков. Счётчик «подряд» её пропускал, ловит счётчик за весь ход.
    #[test]
    fn note_repeats_catches_slow_loop_by_total() {
        let line = "- Remove the JOIN with the group and filter the hierarchy in WHERE.\n";
        // Девять повторов — ещё не срабатывает: законный максимум замерен на 7.
        assert!(feed_with_gaps(line, 9, 6300).is_none());
        let (caught, n) =
            feed_with_gaps(line, 10, 6300).expect("десятый повтор должен быть пойман");
        assert_eq!(n, LOOP_TOTAL_LIMIT);
        assert!(caught.starts_with("- Remove the JOIN"));
    }

    /// Верхняя граница законной работы: по 165 ходам прогона 31.08.2026 самая
    /// частая длинная строка встречалась не более семи раз за ход.
    #[test]
    fn note_repeats_allows_seven_rewrites_per_turn() {
        let line =
            "\tЗапрос.УстановитьПараметр(\"Статус\", Перечисления.СтатусыСоглашений.Действует);\n";
        assert!(feed_with_gaps(line, 7, 6300).is_none());
    }

    /// Самая частая строка нужна при обрыве по объёму: она идёт в отчёт и в
    /// поисковую фразу навыка.
    #[test]
    fn top_line_returns_most_frequent() {
        let (mut tail, mut seen, mut pos) = (String::new(), HashMap::new(), 0usize);
        let rare = "редкая строка рассуждения про соединение таблиц регистра\n";
        let often = "частая строка рассуждения про иерархию номенклатуры\n";
        note_repeats(rare, &mut tail, &mut seen, &mut pos);
        for _ in 0..4 {
            note_repeats(often, &mut tail, &mut seen, &mut pos);
        }
        let (line, n) = top_line(&seen);
        assert_eq!(n, 4);
        assert!(line.starts_with("частая строка"));
    }

    /// Порог объёма стоит выше самого большого законного хода. Потолок взят по
    /// `agents_mcp.agent_turns`, а не по потоковому логу: в логе параллельные
    /// вызовы склеиваются, и прежние числа (58 654 законный, 298 727 разнос)
    /// оказались чужими кусками в одном блоке. По базе за 5622 хода максимум —
    /// 100 626 знаков с `finish=stop`, то есть ход законный и договорённый.
    ///
    /// Верхней границы у теста нет намеренно: разноса ПО ОБЪЁМУ в базе не
    /// нашлось ни одного, придумывать для него число — значит вернуть ту же
    /// ложную калибровку, только с другой стороны.
    #[test]
    fn runaway_threshold_sits_above_longest_legal_turn() {
        const {
            assert!(
                TURN_CHARS_MAX > 100_626,
                "самый длинный законный ход не должен обрываться"
            );
        }
        assert!(runaway_reason(298_727).contains("298727"));
    }

    /// Порог длины одной строки стоит между самой длинной законной строкой
    /// замера 01.09.2026 (6941 знак на 466 ходах) и разносом (99 624).
    #[test]
    fn long_line_threshold_sits_between_legal_and_runaway() {
        const {
            assert!(
                LOOP_LINE_CHARS_MAX > 6_941,
                "законная длинная строка не должна обрываться"
            );
            assert!(
                LOOP_LINE_CHARS_MAX < 99_624,
                "разнос внутри строки обязан обрываться"
            );
        }
    }

    /// Тот самый разнос 01.09.2026: строка начинается осмысленно, а дальше идёт
    /// одно слово тысячами повторов без перевода. Счётчики по строкам видят
    /// единственное вхождение и молчат — поймать такую петлю можно только по
    /// длине строки, а назвать её — по самому частому слову.
    #[test]
    fn top_word_names_the_looped_word() {
        let head = "Criterion 4: only reads, no writes. Module only `Выполнить()` queries, no ";
        let line = format!("{head}{}", "`Записать`, ".repeat(900));
        assert!(line.chars().count() >= LOOP_LINE_CHARS_MAX);
        let (word, n) = top_word(&line);
        assert_eq!(word, "Записать");
        assert_eq!(n, 900);
        assert!(long_line_reason(line.chars().count(), &word, n).contains("Записать"));
        // Перевода строки в разносе нет, поэтому счётчик повторов строк слеп.
        let (mut tail, mut seen, mut pos) = (String::new(), HashMap::new(), 0usize);
        assert!(note_repeats(&line, &mut tail, &mut seen, &mut pos).is_none());
        // А в остатке лежит вся строка целиком — по нему и срабатывает сторож.
        assert_eq!(tail.chars().count(), line.chars().count());
    }

    #[test]
    fn parse_text_tool_call_mimo_style() {
        let content = "Начинаю оркестрацию.<tool_call>\n\
            <function=mcp__agents__invoke_agent>\n\
            <parameter=agent>brief-analyst</parameter>\n\
            <parameter=input>{\"user_message\": \"привет\"}</parameter>\n\
            <parameter=orchestration_depth>1</parameter>\n\
            </function>\n\
            </tool_call>";
        let tcs = parse_text_tool_calls(content);
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "mcp__agents__invoke_agent");
        let args: Value = serde_json::from_str(&tcs[0].function.arguments).unwrap();
        assert_eq!(args["agent"], "brief-analyst");
        // Числовое значение распарсилось как JSON-число, а не строка.
        assert_eq!(args["orchestration_depth"], serde_json::json!(1));
        // Вложенный JSON-объект сохранён как объект.
        assert_eq!(args["input"]["user_message"], "привет");
    }

    #[test]
    fn parse_text_multiple_blocks() {
        let content = "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>\
            <tool_call><function=b><parameter=y>\"s\"</parameter></function></tool_call>";
        let tcs = parse_text_tool_calls(content);
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0].function.name, "a");
        assert_eq!(tcs[1].function.name, "b");
    }

    #[test]
    fn parse_text_no_tool_call_returns_empty() {
        assert!(parse_text_tool_calls("обычный финальный ответ без вызовов").is_empty());
    }

    #[test]
    fn strip_removes_blocks_keeps_surrounding_text() {
        let c = "до <tool_call>X</tool_call> после";
        assert_eq!(strip_text_tool_calls(c), "до  после");
    }

    #[test]
    fn parse_text_task_create_with_backslash_path() {
        // Реальный turn-1 content из сквозного прогона на mimo (call 32):
        // task_create с backslash-путём и кириллицей в goal.
        let content = "Брифинг завершён, clarity=clear, task_kind=build_artifact. \
            Создаю задачу на доске.<tool_call>\n\
            <function=mcp__agents__task_create>\n\
            <parameter=task_kind>build_artifact</parameter>\n\
            <parameter=goal>Создать внешнюю обработку для УТ с формой выбора склада</parameter>\n\
            <parameter=target_base>demo-base</parameter>\n\
            <parameter=sandbox_path>C:\\Temp\\agents-e2e-sandbox</parameter>\n\
            <parameter=root_call_id>32</parameter>\n\
            </function>\n\
            </tool_call>";
        let tcs = parse_text_tool_calls(content);
        assert_eq!(tcs.len(), 1, "task_create должен распознаться");
        assert_eq!(tcs[0].function.name, "mcp__agents__task_create");
        let args: Value = serde_json::from_str(&tcs[0].function.arguments).unwrap();
        assert_eq!(args["task_kind"], "build_artifact");
        // backslash-путь не валидный JSON → сохраняется строкой как есть.
        assert_eq!(args["sandbox_path"], "C:\\Temp\\agents-e2e-sandbox");
        assert_eq!(args["root_call_id"], serde_json::json!(32));
    }

    #[test]
    fn tool_call_parse_error_recognized() {
        // Дословный текст из прогона 2026-07-23 (report.json одной из
        // сгенерированных обработок): валидатор передавал текст запроса.
        let real = LlmError::Provider(
            "local-llm HTTP 500 Internal Server Error: {\"error\":{\"code\":500,\"message\":\
             \"Failed to parse tool call arguments as JSON: [json.exception.parse_error.101] \
             parse error at line 1, column 794: syntax error while parsing value - invalid \
             string: missing closing quote"
                .to_string(),
        );
        assert!(is_tool_call_parse_error(&real));

        // Прочие сбои провайдера повтором хода не лечатся — прогон падает как раньше.
        assert!(!is_tool_call_parse_error(&LlmError::Provider(
            "local-llm HTTP 503 Service Unavailable".to_string()
        )));
        assert!(!is_tool_call_parse_error(&LlmError::RateLimited));
        assert!(!is_tool_call_parse_error(&LlmError::Timeout));
    }

    #[test]
    fn tail_chars_cuts_by_symbols_not_bytes() {
        // Кириллица: обрезка по байтам развалила бы строку.
        let s = "абвгд";
        assert_eq!(tail_chars(s, 3), "вгд");
        assert_eq!(tail_chars(s, 99), "абвгд");
        assert_eq!(tail_chars("", 5), "");
    }

    #[test]
    fn loop_query_joins_tail_and_line() {
        // Дословный затык 31.08.2026: по такой фразе нужный навык встаёт первым,
        // а по одной строке — только смежный.
        let tail =
            "There ARE 3082 rows with Статус = \"Действует\" but the WHERE clause returns 0.";
        let line = "Actually, let me try passing the enum value as a parameter.";
        let q = loop_query(line, tail);
        assert!(q.contains("WHERE clause returns 0"));
        assert!(q.contains("enum value as a parameter"));
        // Пустая строка не добавляет пустых хвостов.
        assert_eq!(loop_query("", tail), tail);
    }

    /// Ответ skill_search (имена навыков обезличены) — такой же, какой разбирает отбор тел для
    /// промпта (см. одноимённую константу в skills.rs).
    const REAL_CATALOG: &str = "Найдено навыков: 3 (cos≥0.45).\n\n\
• debugger-cache-invalidate (cos=0.601, rr=-7.338, scope: Repo1C)\n  Сброс кэша отладчика.\n\n\
• bsl-object-member-access-runtime-errors (cos=0.574, rr=-1.364, scope: общее)\n  Три рантайм-ошибки доступа к членам объекта.\n\n\
• external-processing-branches (cos=0.565, rr=2.148, scope: Repo1C)\n  Две ветки исполнения внешней обработки.";

    #[test]
    fn pick_skill_skips_already_shown() {
        // Тот же дословный каталог, что и в тестах отбора тел для промпта.
        let shown = vec!["external-processing-branches".to_string()];
        assert_eq!(
            pick_skill(REAL_CATALOG, &shown).map(|(n, _)| n).as_deref(),
            Some("bsl-object-member-access-runtime-errors")
        );
        let (first, rr) = pick_skill(REAL_CATALOG, &[]).expect("каталог не пуст");
        assert_eq!(first, "external-processing-branches");
        // Оценка возвращается вместе с именем: по ней ветка лечения решает,
        // навык это по теме или случайный сосед по каталогу.
        assert!(
            rr > f64::NEG_INFINITY,
            "оценка реранкера должна разбираться"
        );
        assert!(pick_skill("", &[]).is_none());
    }

    /// Порог отсева стоит между верным попаданием и случайным соседом. Замер
    /// 31.08.2026: верные находки лежали в -6.3…-0.1 при медиане -3.9, а
    /// подкладки, из-за которых прогон израсходовал все попытки впустую, шли
    /// заметно ниже.
    #[test]
    fn loop_skill_threshold_between_hit_and_neighbour() {
        const {
            assert!(
                LOOP_SKILL_MIN_RR > -8.0,
                "серверную отсечку порог не дублирует"
            );
            assert!(
                LOOP_SKILL_MIN_RR < -3.9,
                "медиану верных попаданий не отсекаем"
            );
            // Сдвиг выборки нужен именно потому, что штатный режим — жадный.
            assert!(LOOP_NUDGE_TEMP > 0.0);
        }
    }

    #[test]
    fn loop_error_keeps_readable_text() {
        // Ошибка петли — отдельная разновидность (ветку выбирает сопоставление
        // по типу), но её текст уходит в report.json и в журнал, поэтому в нём
        // должны остаться все числа живого обрыва 31.08.2026.
        let real = LlmError::Loop {
            provider: "local-llm".to_string(),
            channel: "размышлении".to_string(),
            reason: "строка «Результат = Запрос.Выполнить().Выгрузить();» повторена 6 раз"
                .to_string(),
            line: "\tРезультат = Запрос.Выполнить().Выгрузить();".to_string(),
            repeats: 6,
            reasoning_chars: 33_128,
            content_chars: 0,
            tail: "хвост размышления".to_string(),
        };
        let text = real.to_string();
        assert!(text.contains("ход прерван в размышлении"));
        assert!(text.contains("повторена 6 раз"));
        assert!(text.contains("33128 знаках размышления"));
        assert!(text.contains("Запрос.Выполнить().Выгрузить()"));

        // Обрыв по объёму — та же разновидность с другой причиной: лечится он
        // так же, а различать их нужно только в отчёте.
        let runaway = LlmError::Loop {
            provider: "local-llm".to_string(),
            channel: "размышлении".to_string(),
            reason: runaway_reason(298_727),
            line: "- Remove the JOIN with the group.".to_string(),
            repeats: 92,
            reasoning_chars: 298_727,
            content_chars: 0,
            tail: "хвост размышления".to_string(),
        };
        assert!(runaway.to_string().contains("объём хода 298727 знаков"));

        // Соседний сбой лечится своим способом — повтором того же хода, без
        // навыка; перепутать ветки нельзя.
        assert!(!is_tool_call_parse_error(&real));
        assert!(matches!(real, LlmError::Loop { .. }));
    }

    #[test]
    fn broken_arguments_never_reach_history() {
        // Целый модуль в аргументе с оборванным экранированием — ровно то, что
        // отравляло диалог: сервер спотыкался о него на каждом следующем ходе.
        let broken = r#"{"code": "Функция Сведения() Экспорт\n\tД.Вставить(\"Вид\", "#;
        assert_eq!(safe_arguments(broken), "{}");
        // Целые аргументы проходят как есть, включая экранированные кавычки.
        let good = r#"{"code": "Д.Вставить(\"Вид\", \"Обработка\");"}"#;
        assert_eq!(safe_arguments(good), good);
        assert_eq!(safe_arguments("{}"), "{}");
    }

    /// Поля из `[model.extra_body]` должны уходить в тело запроса на верхнем
    /// уровне, а пустая карта — не менять его вовсе.
    #[test]
    fn extra_body_goes_to_request_top_level() {
        let messages: Vec<Value> = vec![json!({"role": "user", "content": "проба"})];
        let extra: serde_json::Map<String, Value> = serde_json::from_value(json!({
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .expect("карта разбирается");
        let body = ChatRequest {
            model: "проверочная-модель",
            messages: &messages,
            temperature: 0.0,
            max_tokens: None,
            top_p: None,
            tools: None,
            usage: UsageOption { include: true },
            stream: false,
            stream_options: None,
            extra: &extra,
        };
        let собранное: Value = serde_json::to_value(&body).expect("тело сериализуется");
        assert_eq!(
            собранное["chat_template_kwargs"]["enable_thinking"],
            Value::Bool(false)
        );

        let пусто = serde_json::Map::new();
        let body = ChatRequest {
            model: "проверочная-модель",
            messages: &messages,
            temperature: 0.0,
            max_tokens: None,
            top_p: None,
            tools: None,
            usage: UsageOption { include: true },
            stream: false,
            stream_options: None,
            extra: &пусто,
        };
        let собранное: Value = serde_json::to_value(&body).expect("тело сериализуется");
        assert!(собранное.get("chat_template_kwargs").is_none());
    }
}
