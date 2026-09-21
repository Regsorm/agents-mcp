//! Исполнитель invoke_agent: рендер prompt.md → вызов провайдера → парсинг →
//! запись в `agent_calls`.
//!
//! Phase 1 (День 2) — minimal viable flow:
//!   1. Валидация (агент существует, required-поля).
//!   2. Нормализация input + sha256 → input_hash.
//!   3. Рендер prompt через tera (`{{brief}}`, `{{src_files}}` и т.д.).
//!   4. Вызов провайдера через trait LlmProvider.
//!   5. Парсинг ответа (если format=json) — без retry на Дне 2.
//!   6. INSERT в agent_calls.
//!
//! Кеш (День 5) и retry с подсказкой при невалидном JSON (День 3-4) — позже.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tracing::{error, info, warn};

use crate::cache;
use crate::health::ProviderStatus;
use crate::providers::mcp_client;
use crate::providers::{ClaudeCliHints, LlmError, LlmProvider, LlmRequest};
use crate::registry::{AgentDefinition, ExecutionConfig, Registry, ResponseFormat};
use crate::reload::ProviderEnv;
use crate::store::{CallStatus, OrphanedCall, Store};

pub(crate) const CALL_KEY_HEADER: &str = "x-agents-mcp-call";

/// Доверенный контекст идущего вызова, найденный по выданному службой ключу.
#[derive(Debug, Clone)]
pub(crate) struct CallScope {
    pub call_id: i64,
    pub cwd: Option<PathBuf>,
    /// Корни fs_* инструментов агента (allowed_roots в [execution]); None —
    /// действует общий [fs].allowed_roots службы.
    pub allowed_roots: Option<Vec<PathBuf>>,
    pub parent_call_id: Option<i64>,
    pub orchestration_depth: u32,
}

type CallScopes = Arc<std::sync::Mutex<HashMap<String, CallScope>>>;

/// Ключ живёт ровно столько же, сколько `ReadyCall`: Drop срабатывает и при
/// отмене либо панике фоновой задачи.
struct CallKeyGuard {
    key: String,
    scopes: CallScopes,
}

impl Drop for CallKeyGuard {
    fn drop(&mut self) {
        self.scopes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.key);
    }
}

/// Параметры вызова агента.
#[derive(Debug, Clone, Deserialize)]
pub struct InvokeRequest {
    pub agent: String,
    #[serde(default)]
    pub input: Map<String, Value>,
    #[serde(default)]
    pub variant: Option<String>,
    /// ID родительского вызова (из `agent_calls.id`). Заполняется когда
    /// invoke инициирован агентом-оркестратором — мы прокидываем call_id
    /// внешнего вызова, чтобы видеть дерево вложенности и контролировать
    /// глубину рекурсии.
    #[serde(default)]
    pub parent_call_id: Option<i64>,
    /// Текущая глубина в дереве оркестрации (0 — корневой вызов от клиента).
    /// На каждом invoke_agent через `mcp__agents__invoke_agent` оркестратор
    /// прокидывает увеличенное значение; runtime сравнивает с
    /// `max_orchestration_depth` и отказывает при превышении.
    #[serde(default)]
    pub orchestration_depth: u32,
    /// Режим ожидания результата (async-режим, из-за MCP-timeout клиента 60 с):
    ///   None    — синхронно: ждать завершения, вернуть полный {result, metadata}
    ///             (прежнее поведение, для прямых внешних вызовов и тестов).
    ///   Some(0) — запустить job в фоне и сразу вернуть call_id (status=running).
    ///   Some(n) — запустить в фоне и опрашивать до n секунд; готово — результат,
    ///             иначе running. n зажимается в [0, 55], чтобы сам ответ успел
    ///             вернуться под 60-секундным таймаутом MCP-клиента оркестратора.
    #[serde(default)]
    pub wait_sec: Option<u64>,
    /// id задачи в PG task-store, к которой принадлежит вызов. Прокидывается
    /// оркестратором; при наличии runtime собирает срез артефактов задачи в
    /// {{ task_context }}. None — вызов вне задачи (срез пуст).
    #[serde(default)]
    pub task_id: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvokeResponse {
    pub result: Value,
    pub metadata: InvokeMetadata,
}

/// Результат `Runtime::invoke` — разделяет синхронный путь (прежний контракт)
/// и async-конверт. Сериализацию в JSON-конверт делает слой server.rs.
pub enum InvokeOutcome {
    /// Синхронный режим (wait_sec=None): bare-ответ {result, metadata}.
    Sync(InvokeResponse),
    /// Async: задача завершилась (в т.ч. cache hit) — конверт done.
    Done(InvokeResponse),
    /// Модель вернула обрезанный либо неразобранный результат.
    Incomplete {
        response: InvokeResponse,
        error: String,
    },
    /// Async: задача ещё бежит, опрашивать через wait_agent(call_id).
    Running { call_id: i64 },
    /// Задача завершилась ошибкой (ошибка уже записана в строку).
    Failed { call_id: i64, error: String },
    /// Вызов отменён вручную.
    Cancelled { call_id: i64, error: String },
    /// Модель успела ответить, но штатная запись итога в хранилище отказала.
    PersistenceFailed {
        call_id: i64,
        response: Option<InvokeResponse>,
        error: String,
    },
}

enum CompletedCall {
    Done(InvokeResponse),
    Incomplete {
        response: InvokeResponse,
        error: String,
    },
}

/// Промежуточный результат `Runtime::prepare`: либо готовый ответ из кеша
/// (job не создаём), либо подготовленный к исполнению вызов с зарезервированным
/// call_id. Boxed — ReadyCall крупный (несёт LlmRequest + Arc-и).
enum Prepared {
    CacheHit(InvokeResponse),
    Ready(Box<ReadyCall>),
}

/// Всё, что нужно `execute_ready` для медленной части (provider.complete →
/// парсинг → UPDATE строки → кеш). Полностью owned/Arc — спокойно уезжает
/// в `tokio::spawn` для async-режима.
struct ReadyCall {
    call_id: i64,
    agent_name: String,
    agent: Arc<AgentDefinition>,
    provider: Arc<dyn LlmProvider>,
    provider_name: String,
    model_name: String,
    variant: String,
    parent_call_id: Option<i64>,
    orchestration_depth: u32,
    /// Готовый cache_key (если кеш включён) — чтобы execute_ready не тащил
    /// req.input для повторного compute_key.
    cache_key: Option<String>,
    llm_req: LlmRequest,
    start: Instant,
    deadline: tokio::time::Instant,
    timeout_sec: u64,
    call_key_guard: CallKeyGuard,
    /// Снимок `[storage] runs_dir` на момент подготовки: туда кладётся сырой
    /// ответ модели, если разобрать его как JSON не удалось.
    runs_dir: PathBuf,
}

/// Строка истории вызовов (`agent_calls`). Тип живёт в хранилище: рантайм
/// отдаёт его наружу как есть (MCP-tool `agent_history`).
pub use crate::store::HistoryEntry;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvokeMetadata {
    pub agent: String,
    pub variant: String,
    pub model_used: String,
    pub provider: String,
    pub tokens_in: u32,
    pub tokens_out: u32,
    /// Стоимость в долларах; `None` означает, что цена модели неизвестна.
    pub cost_usd: Option<f64>,
    pub latency_ms: u64,
    pub cached: bool,
    pub call_id: i64,
    /// ID родительского вызова (NULL для корневого вызова от клиента).
    pub parent_call_id: Option<i64>,
    /// Глубина в дереве оркестрации (0 для корневого вызова).
    pub orchestration_depth: u32,
}

#[derive(Debug, Error)]
pub enum InvokeError {
    #[error("агент '{0}' не найден")]
    AgentNotFound(String),

    #[error("вариант промпта '{variant}' у агента '{agent}' не найден")]
    VariantNotFound { agent: String, variant: String },

    #[error("отсутствуют обязательные поля input: {0:?}")]
    MissingFields(Vec<String>),

    #[error("провайдер '{0}' не подключён в этом сервисе")]
    UnknownProvider(String),

    #[error("ошибка рендера prompt.md: {0}")]
    PromptRender(String),

    #[error("ошибка mcp_config: {0}")]
    McpConfig(String),

    #[error("ошибка провайдера: {0}")]
    Llm(#[source] Box<LlmError>),

    #[error("превышен общий срок прогона ({timeout_sec} с)")]
    RunTimeout { timeout_sec: u64 },

    #[error("ошибка БД: {0}")]
    Db(String),

    #[error("запись итога в хранилище не удалась: {error}")]
    PersistenceFailed {
        call_id: i64,
        error: String,
        response: Option<Box<InvokeResponse>>,
    },

    #[error("фоновая задача завершилась паникой: {0}")]
    BackgroundPanic(String),

    #[error("превышена глубина оркестрации: {depth} >= {max} — возможна циклическая рекурсия")]
    MaxDepthExceeded { depth: u32, max: u32 },

    #[error("оценочный размер входа {estimated} токенов превышает лимит агента {max}")]
    InputTooLarge { estimated: u32, max: u32 },

    #[error("у агента задан allowed_roots, но рабочий каталог вызова не задан (cwd_template)")]
    AgentRootsWorkDirMissing,

    #[error("рабочий каталог вызова '{cwd}' вне корней allowed_roots агента: {roots:?}")]
    WorkDirOutsideAgentRoots { cwd: String, roots: Vec<String> },

    #[error("служба готовится к остановке: новые вызовы не принимаются")]
    ShuttingDown,
}

impl From<LlmError> for InvokeError {
    fn from(error: LlmError) -> Self {
        Self::Llm(Box::new(error))
    }
}

/// Запись реестра живых фоновых вызовов: чем остановить задачу и что нужно,
/// чтобы закрыть её итог.
struct LiveCall {
    handle: tokio::task::JoinHandle<()>,
    agent: String,
    /// Файл-итог фонового вызова (agent_run); None — итог ждут по call_id.
    result_path: Option<PathBuf>,
    /// Задача task-store, к которой привязан вызов (`task_id` запроса); None —
    /// вызов вне задачи. По ней работает tool `chain_cancel`: у остановки
    /// цепочки нет call_id, есть только задача.
    task_id: Option<i64>,
    started: Instant,
}

/// Запись реестра синхронных вызовов (invoke без wait_sec): агент, начало и
/// задача, если она известна из запроса.
struct SyncCall {
    agent: String,
    started: Instant,
    task_id: Option<i64>,
}

/// Реестр живых вызовов: `call_id` → задача + данные для отмены (фоновые) либо
/// агент и начало (синхронные). Видит и фоновые вызовы, и синхронные — по ним
/// [`Runtime::drain_status`] решает, можно ли останавливать службу
/// (tool `prepare_shutdown`).
///
/// За std::sync::Mutex — блокировка не удерживается через await: критические
/// участки здесь только вставка и изъятие записи. Живёт в Arc, потому что
/// задача снимает себя из реестра сама, последним своим действием.
#[derive(Default)]
pub struct CallRegistry {
    inner: std::sync::Mutex<HashMap<i64, LiveCall>>,
    /// Синхронные вызовы (invoke_agent без wait_sec): `call_id` → агент, начало
    /// и задача. Запись держит [`SyncCallGuard`], поэтому снимается и при обрыве
    /// HTTP-запроса вместе с будущим.
    sync: std::sync::Mutex<HashMap<i64, SyncCall>>,
}

/// Живой вызов, каким его видно снаружи (tool `agent_cancel`, tool
/// `prepare_shutdown` — кого дожидаться вместо обрыва всех).
#[derive(Debug, Clone)]
pub struct LiveCallInfo {
    pub call_id: i64,
    pub agent: String,
    pub elapsed_sec: u64,
    /// true — фоновый вызов (agent_run или invoke_agent с wait_sec): его можно
    /// отменить через agent_cancel. false — синхронный: отменить нельзя.
    pub background: bool,
    /// Задача task-store, к которой привязан вызов; None — вызов вне задачи
    /// либо задача синхронного вызова неизвестна.
    pub task_id: Option<i64>,
}

/// Сторож синхронного вызова: снимает запись из реестра при своём Drop.
/// Drop срабатывает и при обрыве HTTP-запроса вместе с будущим — запись не
/// остаётся фантомом.
pub(crate) struct SyncCallGuard {
    registry: Arc<CallRegistry>,
    call_id: i64,
}

/// Снимает фоновый вызов из реестра при любом выходе из задачи, включая
/// отмену и панику во вложенной задаче исполнения.
struct BackgroundCallGuard {
    registry: Arc<CallRegistry>,
    call_id: i64,
}

/// Сторож финализации: после изъятия вызова из реестра отмена ещё должна
/// дождаться задачи, закрыть строку в хранилище и записать файл-итог.
/// Счётчик снимается по Drop на любом пути, включая ошибку и отмену future.
struct FinalizationGuard(Arc<AtomicUsize>);

impl FinalizationGuard {
    fn new(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Self(counter)
    }
}

impl Drop for FinalizationGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Drop for BackgroundCallGuard {
    fn drop(&mut self) {
        self.registry.remove(self.call_id);
    }
}

/// `JoinHandle` сам не отменяет задачу при Drop. Эта обёртка нужна, чтобы
/// отмена внешней зарегистрированной задачи не оставляла исполнение detached.
struct AbortOnDrop<T> {
    handle: tokio::task::JoinHandle<T>,
}

impl<T> AbortOnDrop<T> {
    fn new(handle: tokio::task::JoinHandle<T>) -> Self {
        Self { handle }
    }

    async fn join(&mut self) -> Result<T, tokio::task::JoinError> {
        (&mut self.handle).await
    }

    fn abort(&self) {
        self.handle.abort();
    }
}

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

impl Drop for SyncCallGuard {
    fn drop(&mut self) {
        self.registry
            .sync
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.call_id);
    }
}

/// Исход `Runtime::cancel`.
#[derive(Debug)]
pub enum CancelOutcome {
    /// Задача оборвана, строка вызова помечена ошибкой, файл-итог записан.
    Cancelled {
        call_id: i64,
        agent: String,
        elapsed_sec: u64,
    },
    /// Такого живого вызова нет: ни в реестре, ни строки в хранилище.
    NotFound { call_id: i64 },
    /// Строка вызова есть, а фоновая задача не бежит: вызов уже завершён либо
    /// идёт внутри синхронного запроса (такой отменить нельзя).
    Finished { call_id: i64, status: String },
}

/// Исход `Runtime::cancel_task` (tool `chain_cancel`): остановка всей цепочки
/// работ по задаче, а не одного вызова.
#[derive(Debug)]
pub enum TaskCancelOutcome {
    /// Живые вызовы задачи отменены, сама задача помечена `cancelled`.
    Cancelled {
        task_id: i64,
        /// call_id вызовов, оборванных этой отменой (те, что успели завершиться
        /// сами между выборкой и отменой, сюда не попадают).
        cancelled_calls: Vec<i64>,
        previous_status: String,
        status: String,
    },
    /// Задача уже закрыта (completed/failed/cancelled): её статус не меняем.
    /// Совпавшие живые фоновые вызовы к этому моменту уже остановлены.
    AlreadyClosed { task_id: i64, status: String },
    /// Задачи с таким id в task-store нет.
    NotFound { task_id: i64 },
}

impl CallRegistry {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<i64, LiveCall>> {
        // Отравление мьютекса тут ничего не сообщает: в критической секции
        // только вставка/изъятие записи, восстанавливать после паники нечего.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Зарегистрировать синхронный вызов (invoke без wait_sec): пока он идёт,
    /// служба не считает себя свободной — `prepare_shutdown` его дожидается.
    /// Запись снимает [`SyncCallGuard`] при своём Drop.
    pub(crate) fn register_sync(
        self: &Arc<Self>,
        call_id: i64,
        agent: String,
        task_id: Option<i64>,
    ) -> SyncCallGuard {
        let call = SyncCall {
            agent,
            started: Instant::now(),
            task_id,
        };
        let mut map = self.sync.lock().unwrap_or_else(|e| e.into_inner());
        map.insert(call_id, call);
        SyncCallGuard {
            registry: Arc::clone(self),
            call_id,
        }
    }

    /// Есть ли вызов с таким call_id в любом из наборов реестра — фоновом или
    /// синхронном.
    pub(crate) fn contains(&self, call_id: i64) -> bool {
        if self.lock().contains_key(&call_id) {
            return true;
        }
        self.sync
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&call_id)
    }

    /// Запустить фоновую задачу, зарегистрировав её под той же блокировкой:
    /// тогда запись появляется раньше, чем задача успеет снять себя. Иначе
    /// мгновенно завершившийся вызов оставил бы в реестре фантом — вызов,
    /// которого уже нет, но который числится отменяемым.
    fn spawn_registered<F: FnOnce() -> tokio::task::JoinHandle<()>>(
        &self,
        call_id: i64,
        agent: String,
        result_path: Option<PathBuf>,
        task_id: Option<i64>,
        job: F,
    ) {
        let mut map = self.lock();
        let handle = job();
        map.insert(
            call_id,
            LiveCall {
                handle,
                agent,
                result_path,
                task_id,
                started: Instant::now(),
            },
        );
    }

    /// Снять вызов с учёта — задача зовёт это своим последним действием.
    fn remove(&self, call_id: i64) {
        self.lock().remove(&call_id);
    }

    /// Забрать запись из реестра. Нет записи — вызова отменять нечем.
    fn take(&self, call_id: i64) -> Option<LiveCall> {
        self.lock().remove(&call_id)
    }

    /// Забрать все фоновые вызовы для штатной остановки службы.
    fn take_all(&self) -> Vec<(i64, LiveCall)> {
        self.lock().drain().collect()
    }

    /// Список живых вызовов (фоновые + синхронные), отсортированный по call_id.
    /// `take()`/`remove()`/`spawn_registered()` работают только с фоновыми:
    /// синхронный вызов отменить нельзя (cancel для него вернёт Finished).
    pub fn live_calls(&self) -> Vec<LiveCallInfo> {
        let mut out: Vec<LiveCallInfo> = {
            let map = self.lock();
            map.iter()
                .map(|(call_id, c)| LiveCallInfo {
                    call_id: *call_id,
                    agent: c.agent.clone(),
                    elapsed_sec: c.started.elapsed().as_secs(),
                    background: true,
                    task_id: c.task_id,
                })
                .collect()
        };
        {
            let map = self.sync.lock().unwrap_or_else(|e| e.into_inner());
            out.extend(map.iter().map(|(call_id, c)| LiveCallInfo {
                call_id: *call_id,
                agent: c.agent.clone(),
                elapsed_sec: c.started.elapsed().as_secs(),
                background: false,
                task_id: c.task_id,
            }));
        }
        out.sort_by_key(|c| c.call_id);
        out
    }

    /// `call_id` живых фоновых вызовов указанной задачи, по возрастанию. По ним
    /// [`Runtime::cancel_task`] останавливает цепочку целиком: у неё на руках
    /// только task_id. Синхронные вызовы сюда не попадают — их отменить нельзя,
    /// они держат своего вызывающего.
    pub fn calls_of_task(&self, task_id: i64) -> Vec<i64> {
        let mut out: Vec<i64> = self
            .lock()
            .iter()
            .filter(|(_, call)| call.task_id == Some(task_id))
            .map(|(call_id, _)| *call_id)
            .collect();
        out.sort_unstable();
        out
    }
}

pub struct Runtime {
    /// Единый вход в хранилище: SQL и драйвер живут за типажом [`Store`].
    pg: Arc<dyn Store>,
    registry: Arc<Registry>,
    /// Набор провайдеров LLM. За RwLock: перечитка главного конфига подменяет
    /// его на лету, а идущие вызовы держат свой Arc и доживают на прежнем наборе.
    providers: std::sync::RwLock<HashMap<String, Arc<dyn LlmProvider>>>,
    /// Итог последней проверки каждого зарегистрированного провайдера.
    provider_statuses: std::sync::RwLock<HashMap<String, ProviderStatus>>,
    /// Максимальная глубина рекурсии оркестраторов. При превышении — отказ
    /// с InvokeError::MaxDepthExceeded. По плану раздел про защиту от циклов.
    max_orchestration_depth: u32,
    /// Срок для агентов без собственного `[limits] timeout_sec`.
    default_timeout_sec: AtomicU64,
    /// Актуальный снимок окружения для подстановок в `mcp_config`.
    provider_env: std::sync::RwLock<ProviderEnv>,
    /// Клиент библиотеки навыков (push-инъекция skills_index + skill_load).
    /// За RwLock: перечитка `[skills] rag_query_url` подменяет его на лету.
    skills: std::sync::RwLock<crate::skills::SkillsClient>,
    /// Тест-override модели на ВСЕХ агентах ([agents] force_provider/force_model).
    /// Если оба Some — любой invoke идёт через этого провайдера с этой моделью,
    /// игнорируя per-agent [model]. None — каждый агент по своему config.toml.
    /// За RwLock: watcher главного конфига перечитывает его на лету (без рестарта).
    force_override: SharedOverride,
    /// Каталог файлов-итогов фоновых вызовов ([storage] runs_dir).
    /// За RwLock: перечитка `[storage] runs_dir` подменяет его на лету.
    runs_dir: std::sync::RwLock<PathBuf>,
    /// Реестр живых вызовов: кого можно отменить (`cancel`) и кто сейчас идёт
    /// (`live_calls`) — фоновые и синхронные.
    calls: Arc<CallRegistry>,
    /// Момент закрытия приёма новых вызовов (tool `prepare_shutdown`).
    /// None — приём открыт. За std::sync::Mutex: guard не держится через await.
    draining_since: std::sync::Mutex<Option<Instant>>,
    /// Вызовы в стадии подготовки: прошли проверку приёма (admit), но ещё не
    /// видны в реестре (идёт prepare — поиск навыков, запись в базу). Вместе с
    /// `calls` образует множество идущих вызовов для `wait_drained`.
    preparing: Arc<AtomicUsize>,
    /// Вызовы уже сняты с реестра, но их отмена и запись итогов ещё идут.
    finalizing: Arc<AtomicUsize>,
    /// Будит `wait_drained`, когда другой prepare_shutdown снова открыл приём.
    drain_changed: tokio::sync::Notify,
    /// Одноразовые ключи идущих вызовов. Значение удаляет `CallKeyGuard`.
    call_scopes: CallScopes,
    /// Порт собственного HTTP `/mcp`; задаётся при сборке axum-роутера.
    own_mcp_port: std::sync::RwLock<Option<u16>>,
    /// Экземпляр службы (`[server] instance`, по умолчанию «имя машины:порт»):
    /// пишется в строки вызовов, по нему при старте закрываются свои
    /// осиротевшие вызовы.
    instance: String,
}

/// Итог подготовки службы к остановке (tool `prepare_shutdown`).
#[derive(Debug, Clone)]
pub struct DrainStatus {
    pub draining: bool,
    /// Сколько секунд приём закрыт (0, если открыт).
    pub draining_sec: u64,
    pub preparing: usize,
    pub finalizing: usize,
    pub live: Vec<LiveCallInfo>,
}

impl DrainStatus {
    /// Службу можно останавливать: приём закрыт, идущих вызовов нет.
    pub fn ready(&self) -> bool {
        self.draining && self.preparing == 0 && self.finalizing == 0 && self.live.is_empty()
    }
}

/// Сторож подготовки вызова: пока он жив, вызов числится в `preparing` и
/// `prepare_shutdown` его видит. Счётчик уменьшается по Drop — в том числе
/// когда подготовка закончилась ошибкой или вызов уже уехал в реестр.
pub(crate) struct AdmissionGuard(Arc<AtomicUsize>);

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Итог пуска фонового вызова (`Runtime::start_background`, бэкенд tool
/// `agent_run` в режиме пуска). Возвращается СРАЗУ, работа агента продолжается
/// в фоне сервиса.
pub enum StartedJob {
    /// Вызов уехал в фон: итог появится в `result_path` и доступен проверкой
    /// по `call_id`.
    Running { call_id: i64, result_path: PathBuf },
    /// Ответ нашёлся в кеше — готов сразу, файл-итог уже записан.
    Done {
        response: InvokeResponse,
        result_path: PathBuf,
    },
}

/// Тест-override модели, перечитываемый watcher'ом главного конфига на лету.
#[derive(Debug, Clone, Default)]
pub struct ModelOverride {
    pub provider: Option<String>,
    pub model: Option<String>,
}

/// Разделяемый override: Runtime читает в execute, watcher конфига пишет при
/// изменении `[agents] force_provider/force_model` в главном конфиге.
pub type SharedOverride = std::sync::Arc<std::sync::RwLock<ModelOverride>>;

/// Компактный текстовый срез доски задачи для системного промпта под-агента:
/// по одному блоку на артефакт —
/// kind/key + summary; короткий content вливаем целиком, длинный — только
/// намёк с размером, чтобы агент знал про артефакт и дёрнул artifact_read.
fn format_task_context(artifacts: &[crate::store::Artifact]) -> String {
    const MAX_CONTENT: usize = 600;
    const MAX_HEAD: usize = 300;
    const MAX_TOTAL: usize = 6000;
    const TAIL_RESERVE: usize = 60;
    if artifacts.is_empty() {
        return String::new();
    }
    let mut b = String::from("Контекст задачи (артефакты доски):\n");
    // Жёсткий суммарный предел среза — MAX_TOTAL символов. Сначала резервируем
    // место под заголовки всех артефактов (каждый не длиннее MAX_HEAD), чтобы
    // агент видел всю доску; content идёт только в оставшийся запас. Заголовки,
    // не поместившиеся даже без content, заменяются строкой «ещё N». Полный
    // контекст отдельно учитывается в ключе кеша.
    let heads: Vec<String> = artifacts
        .iter()
        .map(|a| {
            let mut head = format!("- [{}] {}", a.kind, a.key);
            if let Some(s) = a.summary.as_deref().filter(|s| !s.is_empty()) {
                head.push_str(" — ");
                head.push_str(s);
            }
            if head.chars().count() > MAX_HEAD {
                head = head
                    .chars()
                    .take(MAX_HEAD - 1)
                    .chain(std::iter::once('…'))
                    .collect();
            }
            head.push('\n');
            head
        })
        .collect();
    let mut used = b.chars().count();
    let mut shown = 0;
    for (i, head) in heads.iter().enumerate() {
        let len = head.chars().count();
        let tail = if i + 1 < heads.len() { TAIL_RESERVE } else { 0 };
        if used + len + tail > MAX_TOTAL {
            break;
        }
        used += len;
        shown += 1;
    }
    let hidden = heads.len() - shown;
    let tail = if hidden > 0 {
        format!("- … ещё {hidden} артефактов не показаны (artifact_read)\n")
    } else {
        String::new()
    };
    used += tail.chars().count();
    for (a, head) in artifacts.iter().zip(&heads).take(shown) {
        b.push_str(head);
        if let Some(c) = &a.content {
            let len = c.chars().count();
            let line = if len <= MAX_CONTENT {
                format!("  {c}\n")
            } else {
                format!("  (content {len} символов — прочитать через artifact_read)\n")
            };
            let line_len = line.chars().count();
            if used + line_len <= MAX_TOTAL {
                used += line_len;
                b.push_str(&line);
            }
        }
    }
    b.push_str(&tail);
    b
}

/// Стабильный отпечаток полного, не ограниченного для промпта состояния доски.
fn task_context_fingerprint(artifacts: &[crate::store::Artifact]) -> String {
    let raw = serde_json::to_vec(artifacts).unwrap_or_default();
    let digest = Sha256::digest(raw);
    let mut hex = String::with_capacity(digest.len() * 2);
    for b in digest.iter() {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}

impl Runtime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pg: Arc<dyn Store>,
        registry: Arc<Registry>,
        providers: HashMap<String, Arc<dyn LlmProvider>>,
        skills: crate::skills::SkillsClient,
        force_override: SharedOverride,
        runs_dir: PathBuf,
        instance: String,
        default_timeout_sec: u64,
    ) -> Self {
        let provider_statuses = providers
            .keys()
            .map(|name| (name.clone(), ProviderStatus::registered()))
            .collect();
        Self {
            pg,
            registry,
            providers: std::sync::RwLock::new(providers),
            provider_statuses: std::sync::RwLock::new(provider_statuses),
            max_orchestration_depth: 5,
            default_timeout_sec: AtomicU64::new(default_timeout_sec),
            provider_env: std::sync::RwLock::new(ProviderEnv::default()),
            skills: std::sync::RwLock::new(skills),
            force_override,
            runs_dir: std::sync::RwLock::new(runs_dir),
            calls: Arc::new(CallRegistry::default()),
            draining_since: std::sync::Mutex::new(None),
            preparing: Arc::new(AtomicUsize::new(0)),
            finalizing: Arc::new(AtomicUsize::new(0)),
            drain_changed: tokio::sync::Notify::new(),
            call_scopes: Arc::new(std::sync::Mutex::new(HashMap::new())),
            own_mcp_port: std::sync::RwLock::new(None),
            instance,
        }
    }

    /// Запомнить порт собственного HTTP-транспорта до начала приёма вызовов.
    pub(crate) fn set_own_mcp_port(&self, port: u16) {
        *self.own_mcp_port.write().unwrap_or_else(|e| e.into_inner()) = Some(port);
    }

    /// Найти доверенный контекст по ключу из HTTP-заголовка.
    pub(crate) fn call_scope(&self, key: &str) -> Option<CallScope> {
        self.call_scopes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
            .cloned()
    }

    fn issue_call_key(
        &self,
        call_id: i64,
        cwd: Option<PathBuf>,
        allowed_roots: Option<Vec<PathBuf>>,
        parent_call_id: Option<i64>,
        orchestration_depth: u32,
    ) -> (String, CallKeyGuard) {
        let key = uuid::Uuid::new_v4().to_string();
        self.call_scopes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                key.clone(),
                CallScope {
                    call_id,
                    cwd,
                    allowed_roots,
                    parent_call_id,
                    orchestration_depth,
                },
            );
        let guard = CallKeyGuard {
            key: key.clone(),
            scopes: Arc::clone(&self.call_scopes),
        };
        (key, guard)
    }

    /// Экземпляр службы, как он пишется в строки вызовов (для пробы здоровья).
    pub fn instance(&self) -> &str {
        &self.instance
    }

    pub fn provider_statuses(&self) -> std::collections::BTreeMap<String, ProviderStatus> {
        self.provider_statuses
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|(name, status)| (name.clone(), status.clone()))
            .collect()
    }

    /// Подменить набор провайдеров — перечитка главного конфига. Идущие вызовы
    /// держат свой Arc из прежнего набора и доживают на нём, новые берут новый.
    #[cfg(test)]
    pub fn set_providers(&self, providers: HashMap<String, Arc<dyn LlmProvider>>) {
        let statuses = providers
            .keys()
            .map(|name| (name.clone(), ProviderStatus::registered()))
            .collect();
        *self.providers.write().unwrap_or_else(|e| e.into_inner()) = providers;
        *self
            .provider_statuses
            .write()
            .unwrap_or_else(|e| e.into_inner()) = statuses;
    }

    /// Подменить провайдеры вместе с результатами их стартовой проверки.
    pub fn set_provider_set(
        &self,
        providers: HashMap<String, Arc<dyn LlmProvider>>,
        statuses: HashMap<String, ProviderStatus>,
    ) {
        *self.providers.write().unwrap_or_else(|e| e.into_inner()) = providers;
        *self
            .provider_statuses
            .write()
            .unwrap_or_else(|e| e.into_inner()) = statuses;
    }

    /// Подменить клиент навыков — перечитка `[skills] rag_query_url`.
    pub fn set_skills(&self, skills: crate::skills::SkillsClient) {
        *self.skills.write().unwrap_or_else(|e| e.into_inner()) = skills;
    }

    /// Подменить каталог файлов-итогов — перечитка `[storage] runs_dir`.
    pub fn set_runs_dir(&self, dir: PathBuf) {
        *self.runs_dir.write().unwrap_or_else(|e| e.into_inner()) = dir;
    }

    pub fn set_default_timeout_sec(&self, seconds: u64) {
        self.default_timeout_sec.store(seconds, Ordering::SeqCst);
    }

    /// Подменить окружение провайдеров — старт службы и перечитка `.env`.
    pub fn set_provider_env(&self, provider_env: ProviderEnv) {
        *self.provider_env.write().unwrap_or_else(|e| e.into_inner()) = provider_env;
    }

    fn run_timeout(&self, req: &InvokeRequest) -> Result<u64, InvokeError> {
        let agent = self
            .registry
            .get(&req.agent)
            .ok_or_else(|| InvokeError::AgentNotFound(req.agent.clone()))?;
        Ok(agent
            .config
            .limits
            .timeout_sec
            .unwrap_or_else(|| self.default_timeout_sec.load(Ordering::SeqCst)))
    }

    /// Клон клиента навыков под коротким захватом: guard не живёт через await.
    fn skills(&self) -> crate::skills::SkillsClient {
        self.skills
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Полное тело навыка по имени (для MCP-tool skill_load).
    pub async fn skill_load(&self, name: &str) -> Option<String> {
        let skills = self.skills();
        skills.skill_body(name).await
    }

    // ── PG task-store (внешнее хранилище контекста задач) ────────────────

    fn require_store(&self) -> anyhow::Result<&dyn Store> {
        Ok(&*self.pg)
    }

    /// Живость хранилища — настоящим запросом к БД, для пробы здоровья.
    pub async fn pg_health(&self) -> anyhow::Result<()> {
        tokio::time::timeout(Duration::from_secs(5), self.pg.health())
            .await
            .map_err(|_| anyhow::anyhow!("task-store не ответил за 5 с"))?
    }

    /// Создать корневую задачу, вернуть её id.
    pub async fn task_create(&self, t: crate::store::NewTask) -> anyhow::Result<i64> {
        self.require_store()?.create_task(&t).await
    }

    /// Сменить статус задачи (running/needs_input/completed/failed/cancelled).
    pub async fn task_set_status(&self, task_id: i64, status: &str) -> anyhow::Result<()> {
        self.require_store()?.set_task_status(task_id, status).await
    }

    /// Текущий статус задачи (бэкенд tool `task_get`); None — задачи нет.
    pub async fn task_status(&self, task_id: i64) -> anyhow::Result<Option<String>> {
        self.require_store()?.get_task_status(task_id).await
    }

    /// Записать/обновить артефакт доски задачи (upsert по task_id+key).
    #[allow(clippy::too_many_arguments)]
    pub async fn task_write_artifact(
        &self,
        task_id: i64,
        kind: &str,
        key: &str,
        content: Option<&str>,
        summary: Option<&str>,
        producer_agent: Option<&str>,
        producer_call_id: Option<i64>,
        depends_on: &[i64],
    ) -> anyhow::Result<i64> {
        self.require_store()?
            .write_artifact(
                task_id,
                kind,
                key,
                content,
                summary,
                producer_agent,
                producer_call_id,
                depends_on,
            )
            .await
    }

    /// Прочитать артефакты задачи (опционально только указанных kind).
    pub async fn task_read_artifacts(
        &self,
        task_id: i64,
        kinds: Option<&[String]>,
    ) -> anyhow::Result<Vec<crate::store::Artifact>> {
        self.require_store()?.read_artifacts(task_id, kinds).await
    }

    /// Добавить событие в журнал задачи.
    pub async fn task_append_event(
        &self,
        task_id: i64,
        event_type: &str,
        agent: Option<&str>,
        call_id: Option<i64>,
        payload_json: Option<&str>,
    ) -> anyhow::Result<()> {
        self.require_store()?
            .append_task_event(task_id, event_type, agent, call_id, payload_json)
            .await
    }

    /// Последние записи из `agent_calls` (для MCP-tool `agent_history`).
    pub async fn history(
        &self,
        agent: Option<String>,
        since: Option<i64>,
        limit: u32,
    ) -> anyhow::Result<Vec<HistoryEntry>> {
        self.pg
            .list_calls(agent.as_deref(), since.unwrap_or(0), limit)
            .await
    }

    /// Публичный вход. Диспетчер sync/async на основе `req.wait_sec`.
    ///   None    → синхронно (прежний контракт InvokeOutcome::Sync).
    ///   Some(n) → запустить job в фоне, опросить до n сек (Done/Running/Failed).
    pub async fn invoke(&self, req: InvokeRequest) -> Result<InvokeOutcome, InvokeError> {
        let start = Instant::now();
        let timeout_sec = self.run_timeout(&req)?;
        let deadline = tokio::time::Instant::from_std(start) + Duration::from_secs(timeout_sec);
        let wait_sec = req.wait_sec;
        // Задача запроса: по ней вызов привязывается к цепочке (chain_cancel).
        let task_id = req.task_id;

        // Приём вызова: при закрытом приёме (prepare_shutdown) отказ ДО любой
        // работы — строка вызова не создаётся.
        let admission = self.admit(&req)?;
        let preparing_call_id = Arc::new(AtomicI64::new(0));
        let prepared = match tokio::time::timeout_at(
            deadline,
            self.prepare(
                &req,
                start,
                deadline,
                timeout_sec,
                preparing_call_id.clone(),
            ),
        )
        .await
        {
            Ok(result) => result?,
            Err(_) => {
                let call_id = preparing_call_id.load(Ordering::SeqCst);
                if call_id > 0 {
                    finish_timed_out_call(self.pg.clone(), call_id, &req.agent, start, timeout_sec)
                        .await;
                }
                return Err(InvokeError::RunTimeout { timeout_sec });
            }
        };

        match wait_sec {
            // Синхронный режим — прежнее поведение: ждём завершения inline.
            None => match prepared {
                Prepared::CacheHit(resp) => Ok(InvokeOutcome::Sync(resp)),
                Prepared::Ready(ready) => {
                    // Синхронный вызов виден реестру: пока он идёт,
                    // prepare_shutdown его дожидается.
                    let _sync_guard =
                        self.calls
                            .register_sync(ready.call_id, ready.agent_name.clone(), task_id);
                    drop(admission);
                    match execute_ready_guarded(self.pg.clone(), *ready).await {
                        Ok(CompletedCall::Done(resp)) => Ok(InvokeOutcome::Sync(resp)),
                        Ok(CompletedCall::Incomplete { response, error }) => {
                            Ok(InvokeOutcome::Incomplete { response, error })
                        }
                        Err(InvokeError::PersistenceFailed {
                            call_id,
                            error,
                            response,
                        }) => {
                            let response = response.map(|value| *value);
                            Ok(InvokeOutcome::PersistenceFailed {
                                call_id,
                                response,
                                error,
                            })
                        }
                        Err(e) => Err(e),
                    }
                }
            },
            // Async-режим: исполнение уезжает в фон, сразу есть call_id.
            // Долгий provider.complete() больше не блокирует MCP-ответ —
            // оркестратор опрашивает результат через wait_agent(call_id).
            Some(w) => match prepared {
                Prepared::CacheHit(resp) => Ok(InvokeOutcome::Done(resp)),
                Prepared::Ready(ready) => {
                    let call_id = ready.call_id;
                    let agent_name = ready.agent_name.clone();
                    let pool = self.pg.clone();
                    let calls = self.calls.clone();
                    // Регистрация идёт под блокировкой реестра, а задача снимает
                    // себя из него последним действием: пока запись жива, вызов
                    // отменяем, а завершённые в реестре не копятся.
                    self.calls
                        .spawn_registered(call_id, agent_name, None, task_id, move || {
                            tokio::spawn(async move {
                                let _guard = BackgroundCallGuard {
                                    registry: calls,
                                    call_id,
                                };
                                // Провал журналирует сам execute_ready (и в
                                // журнал, и в строку вызова) — дублировать warn
                                // здесь не нужно.
                                let _ = execute_ready_guarded(pool, *ready).await;
                            })
                        });
                    // Вызов виден реестру — счётчик подготовки больше не нужен.
                    drop(admission);
                    // Long-poll до w секунд (зажат в poll_call), затем — что есть.
                    Ok(poll_call(self.pg.clone(), call_id, w).await)
                }
            },
        }
    }

    /// Long-poll результата ранее запущенного async-вызова (бэкенд tool
    /// `wait_agent`). Блокирует максимум `wait_sec` (зажат ≤55с в poll_call).
    pub async fn wait(&self, call_id: i64, wait_sec: u64) -> InvokeOutcome {
        poll_call(self.pg.clone(), call_id, wait_sec).await
    }

    /// Пустить вызов в фон и вернуться сразу, ничего не ожидая (бэкенд tool
    /// `agent_run` в режиме пуска). Отличие от `invoke(wait_sec=Some(0))` —
    /// итог дополнительно кладётся файлом в `result_path`: его можно прочитать
    /// без обращения к серверу, а появления файла — дождаться средствами
    /// клиента, не занимая беседу.
    ///
    /// `result_path` = None → путь по умолчанию в `[storage] runs_dir`.
    /// Ошибки подготовки (нет агента, не хватает обязательных полей input,
    /// негодный вариант промпта) возвращаются здесь же, до пуска в фон.
    pub async fn start_background(
        &self,
        req: InvokeRequest,
        result_path: Option<PathBuf>,
    ) -> Result<StartedJob, InvokeError> {
        let start = Instant::now();
        let timeout_sec = self.run_timeout(&req)?;
        let deadline = tokio::time::Instant::from_std(start) + Duration::from_secs(timeout_sec);
        // Приём вызова: при закрытом приёме (prepare_shutdown) отказ ДО любой
        // работы — строка вызова не создаётся.
        let admission = self.admit(&req)?;
        let preparing_call_id = Arc::new(AtomicI64::new(0));
        let prepared = match tokio::time::timeout_at(
            deadline,
            self.prepare(
                &req,
                start,
                deadline,
                timeout_sec,
                preparing_call_id.clone(),
            ),
        )
        .await
        {
            Ok(result) => result?,
            Err(_) => {
                let call_id = preparing_call_id.load(Ordering::SeqCst);
                if call_id > 0 {
                    finish_timed_out_call(self.pg.clone(), call_id, &req.agent, start, timeout_sec)
                        .await;
                }
                return Err(InvokeError::RunTimeout { timeout_sec });
            }
        };
        match prepared {
            // Кеш-попадание: работы нет, ответ уже есть. Файл всё равно пишем —
            // читателю итога не нужно знать, был вызов кеширован или нет.
            Prepared::CacheHit(resp) => {
                let path = result_path.unwrap_or_else(|| {
                    self.default_result_path(resp.metadata.call_id, &resp.metadata.agent)
                });
                remove_old_result_file(&path);
                self.pg
                    .set_call_result_path(resp.metadata.call_id, &path)
                    .await
                    .map_err(|e| InvokeError::Db(format!("{e}")))?;
                if let Err(e) = write_result_file(&path, &envelope_done(&resp)) {
                    warn!(path = %path.display(), error = %e, "не записал файл-итог (кеш-попадание)");
                }
                Ok(StartedJob::Done {
                    response: resp,
                    result_path: path,
                })
            }
            Prepared::Ready(ready) => {
                let call_id = ready.call_id;
                let agent_name = ready.agent_name.clone();
                let path =
                    result_path.unwrap_or_else(|| self.default_result_path(call_id, &agent_name));
                remove_old_result_file(&path);
                self.pg
                    .set_call_result_path(call_id, &path)
                    .await
                    .map_err(|e| InvokeError::Db(format!("{e}")))?;
                let pool = self.pg.clone();
                let path_in_job = path.clone();
                let name_in_job = agent_name.clone();
                let finalization_deadline = ready.deadline + Duration::from_secs(10);
                let calls = self.calls.clone();
                // Задача запроса: по ней вызов отменяет chain_cancel.
                let task_id = req.task_id;
                // Файл-итог известен заранее — храним его в реестре вместе с
                // задачей: отмена обязана закрыть файл, иначе тот, кто ждёт его
                // появления, зависнет навсегда.
                self.calls.spawn_registered(
                    call_id,
                    agent_name,
                    Some(path.clone()),
                    task_id,
                    move || {
                        tokio::spawn(async move {
                            let _guard = BackgroundCallGuard {
                                registry: calls,
                                call_id,
                            };
                            // execute_ready сам дописывает строку agent_calls (в т.ч.
                            // при ошибке). Здесь остаётся только файл-итог.
                            let envelope = match execute_ready_guarded(pool, *ready).await {
                                Ok(CompletedCall::Done(resp)) => envelope_done(&resp),
                                Ok(CompletedCall::Incomplete { response, error }) => {
                                    envelope_incomplete(&response, &error)
                                }
                                Err(InvokeError::PersistenceFailed {
                                    error, response, ..
                                }) => envelope_persistence_failed(
                                    call_id,
                                    &name_in_job,
                                    response.as_deref(),
                                    &error,
                                ),
                                Err(e) => envelope_error(call_id, &name_in_job, &format!("{e}")),
                            };
                            if tokio::time::Instant::now() >= finalization_deadline {
                                warn!(call_id, path = %path_in_job.display(),
                                      "запись файла-итога прервана: исчерпаны дополнительные 10 с");
                            } else if let Err(e) = write_result_file(&path_in_job, &envelope) {
                                warn!(call_id, path = %path_in_job.display(), error = %e,
                                      "не записал файл-итог фонового вызова");
                            }
                        })
                    },
                );
                // Вызов виден реестру — счётчик подготовки больше не нужен.
                drop(admission);
                Ok(StartedJob::Running {
                    call_id,
                    result_path: path,
                })
            }
        }
    }

    /// Мгновенная проверка ранее запущенного вызова, без ожидания (бэкенд tool
    /// `agent_run` в режиме проверки). Второе значение — `created_at` строки
    /// вызова (unixepoch), чтобы показать, сколько вызов уже идёт; None —
    /// строку прочитать не удалось.
    pub async fn check(&self, call_id: i64) -> (InvokeOutcome, Option<i64>) {
        let created_at = match self.pg.get_call_created_at(call_id).await {
            Ok(opt) => opt,
            Err(e) => {
                warn!(call_id, error = %e, "check: соединение с БД не получено");
                return (
                    InvokeOutcome::PersistenceFailed {
                        call_id,
                        response: None,
                        error: format!("хранилище недоступно: {e}"),
                    },
                    None,
                );
            }
        };
        (poll_call(self.pg.clone(), call_id, 0).await, created_at)
    }

    /// Список живых вызовов (фоновые + синхронные) — для tool `agent_cancel`
    /// (отменяются только фоновые) и для tool `prepare_shutdown` (кого
    /// дожидаться вместо обрыва всех).
    pub fn live_calls(&self) -> Vec<LiveCallInfo> {
        self.calls.live_calls()
    }

    // ── Корректное завершение службы (tool prepare_shutdown) ─────────────

    /// Приём новых вызовов закрыт?
    pub fn is_draining(&self) -> bool {
        self.draining_since
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    /// Закрыть приём новых вызовов. Повторный вызов ничего не меняет.
    pub fn begin_drain(&self) {
        {
            let mut since = self
                .draining_since
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if since.is_some() {
                return;
            }
            *since = Some(Instant::now());
        }
        warn!(
            live = self.live_calls().len(),
            "служба готовится к остановке: приём новых вызовов закрыт"
        );
    }

    /// Отменить подготовку к остановке — приём вызовов снова открыт.
    pub fn abort_drain(&self) {
        let mut since = self
            .draining_since
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if since.take().is_some() {
            info!("подготовка к остановке отменена: приём вызовов открыт");
        }
        self.drain_changed.notify_waiters();
    }

    /// Текущее состояние подготовки к остановке.
    pub fn drain_status(&self) -> DrainStatus {
        let since = *self
            .draining_since
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        DrainStatus {
            draining: since.is_some(),
            draining_sec: since.map(|s| s.elapsed().as_secs()).unwrap_or(0),
            preparing: self.preparing.load(Ordering::SeqCst),
            finalizing: self.finalizing.load(Ordering::SeqCst),
            live: self.live_calls(),
        }
    }

    /// Дождаться, пока идущие вызовы дойдут: опрашивает состояние раз в 500 мс
    /// до `ready()` или истечения `wait_sec` (зажат в [0, 55] — потолок ниже
    /// 60-секундного таймаута MCP-клиента, как у poll_call). Возвращает
    /// последний прочитанный статус.
    pub async fn wait_drained(&self, wait_sec: u64) -> DrainStatus {
        let deadline = Instant::now() + Duration::from_secs(wait_sec.min(55));
        loop {
            let mut changed = std::pin::pin!(self.drain_changed.notified());
            changed.as_mut().enable();
            let status = self.drain_status();
            if !status.draining || status.ready() || Instant::now() >= deadline {
                return status;
            }
            tokio::select! {
                _ = changed.as_mut() => {}
                _ = tokio::time::sleep(Duration::from_millis(500)) => {}
            }
        }
    }

    /// Проверка приёма вызова: при закрытом приёме новый вызов отклоняется ДО
    /// любой работы (строка вызова не создаётся). Исключение — дочерний вызов
    /// идущего вызова (`parent_call_id` есть в реестре живых): иначе идущий
    /// оркестратор упал бы на полудороге.
    ///
    /// ПОРЯДОК ВАЖЕН: СНАЧАЛА увеличиваем счётчик подготовки и заводим сторож,
    /// ПОТОМ читаем draining_since. Наоборот была бы гонка с begin_drain:
    /// вызов прочитал «приём открыт», а закрытие случилось раньше, чем вызов
    /// стал виден (в preparing или в реестре) — wait_drained вернул бы ready
    /// при идущем вызове.
    fn admit(&self, req: &InvokeRequest) -> Result<AdmissionGuard, InvokeError> {
        self.preparing.fetch_add(1, Ordering::SeqCst);
        // Сторож снимает счётчик и при ошибке ниже, и в конце вызова.
        let guard = AdmissionGuard(self.preparing.clone());
        if self.is_draining() {
            let child_of_live = req
                .parent_call_id
                .map(|p| self.calls.contains(p))
                .unwrap_or(false);
            if !child_of_live {
                return Err(InvokeError::ShuttingDown);
            }
        }
        Ok(guard)
    }

    /// Остановить идущий фоновый вызов по call_id (бэкенд tool `agent_cancel`).
    ///
    /// Отмену видит и хранилище, и файл-итог: строка закрывается ошибкой
    /// «вызов отменён вручную», туда же кладётся конверт ошибки — иначе тот,
    /// кто ждёт появления файла, завис бы навсегда. Дочерние процессы убивать
    /// не надо: у claude-cli, codex-cli и серверов-процессов стоит
    /// kill_on_drop(true), и они гаснут сами вместе с задачей.
    pub async fn cancel(&self, call_id: i64) -> CancelOutcome {
        // Счётчик ставим ДО изъятия из реестра: drain_status ни на миг не
        // увидит ложную готовность между take() и записью статуса/файла.
        let _finalization = FinalizationGuard::new(self.finalizing.clone());
        let entry = match self.calls.take(call_id) {
            Some(e) => e,
            // В реестре записи нет: строка вызова есть — он уже завершился
            // (либо идёт внутри синхронного запроса), строки нет — вызова нет.
            None => {
                return match self.pg.get_call_row(call_id).await {
                    Ok(Some(row)) => CancelOutcome::Finished {
                        call_id,
                        status: row.status,
                    },
                    _ => CancelOutcome::NotFound { call_id },
                };
            }
        };

        let elapsed_sec = entry.started.elapsed().as_secs();
        entry.handle.abort();
        // abort() лишь просит задачу остановиться на ближайшей точке ожидания.
        // Если она успела дойти до конца сама (строка и файл-итог уже записаны,
        // а снять себя из реестра не успела), её честный итог затирать ошибкой
        // «отменён» нельзя. Дескриптор различает оба случая: Ok — задача
        // завершилась сама, ошибка с признаком отмены — прервана нами.
        if entry.handle.await.is_ok() {
            let status = match self.pg.get_call_row(call_id).await {
                Ok(Some(row)) => row.status,
                _ => "done".to_string(),
            };
            return CancelOutcome::Finished { call_id, status };
        }
        let err_text = "вызов отменён вручную";
        if let Err(e) = self
            .pg
            .update_call(
                call_id,
                CallStatus::Cancelled,
                None,
                Some(err_text),
                0,
                0,
                None,
                elapsed_sec * 1000,
                None, // session_id у отменённых не сохраняем
                0,
                0,
                0,
                None,
            )
            .await
        {
            warn!(call_id, error = %e, "отмена: не удалось пометить строку вызова ошибкой");
        }
        // Файл пишем, только если его кто-то ждёт: у invoke_agent с wait_sec
        // result_path нет, итог забирают по call_id.
        if let Some(path) = entry.result_path.as_ref() {
            if let Err(e) = write_result_file(
                path,
                &envelope_failure("cancelled", call_id, &entry.agent, err_text),
            ) {
                warn!(call_id, path = %path.display(), error = %e,
                      "отмена: не записал файл-итог отменённого вызова");
            }
        }
        warn!(call_id, agent = %entry.agent, "вызов отменён вручную");

        CancelOutcome::Cancelled {
            call_id,
            agent: entry.agent,
            elapsed_sec,
        }
    }

    /// Остановить цепочку по задаче (бэкенд tool `chain_cancel`): отменить все
    /// живые фоновые вызовы задачи и пометить её `cancelled`.
    ///
    /// Сначала вызовы отменяются тем же путём, что `agent_cancel`: строка
    /// закрывается ошибкой «вызов отменён вручную», файл-итог дописывается
    /// конвертом ошибки. Затем читается статус задачи: закрытый итог не
    /// перезаписывается, а неизвестная задача возвращается как `NotFound`.
    pub async fn cancel_task(&self, task_id: i64) -> anyhow::Result<TaskCancelOutcome> {
        let mut cancelled_calls: Vec<i64> = Vec::new();
        for call_id in self.calls.calls_of_task(task_id) {
            match self.cancel(call_id).await {
                // Отчитываемся только об оборванных нами: вызов, успевший
                // завершиться сам между выборкой и отменой, отменять уже нечем.
                CancelOutcome::Cancelled { call_id, .. } => cancelled_calls.push(call_id),
                CancelOutcome::Finished { .. } | CancelOutcome::NotFound { .. } => {}
            }
        }

        let previous_status = match self.task_status(task_id).await? {
            Some(status) => status,
            None => return Ok(TaskCancelOutcome::NotFound { task_id }),
        };
        // Закрытый статус не перезаписываем: честный итог задачи важнее
        // запоздалой команды остановки.
        if matches!(
            previous_status.as_str(),
            "completed" | "failed" | "cancelled"
        ) {
            return Ok(TaskCancelOutcome::AlreadyClosed {
                task_id,
                status: previous_status,
            });
        }

        self.task_set_status(task_id, "cancelled").await?;
        let payload = serde_json::json!({
            "reason": "остановлено вручную",
            "cancelled_calls": cancelled_calls,
        })
        .to_string();
        self.task_append_event(
            task_id,
            "chain_cancelled",
            None,
            None,
            Some(payload.as_str()),
        )
        .await?;
        let cancelled = cancelled_calls.len();
        warn!(task_id, cancelled, "цепочка остановлена вручную");

        Ok(TaskCancelOutcome::Cancelled {
            task_id,
            cancelled_calls,
            previous_status,
            status: "cancelled".to_string(),
        })
    }

    /// Оборвать фоновые вызовы при штатном завершении процесса и закрыть их
    /// строки/файлы итогом `error`. Внешний предел не даёт остановке зависнуть
    /// навсегда на недоступном хранилище.
    pub async fn finalize_background_calls(&self, reason: &str, timeout: Duration) -> usize {
        let entries = self.calls.take_all();
        if entries.is_empty() {
            return 0;
        }

        let total = entries.len();
        let mut tasks = tokio::task::JoinSet::new();
        for (call_id, entry) in entries {
            entry.handle.abort();
            let pool = self.pg.clone();
            let finalizing = self.finalizing.clone();
            let reason = reason.to_string();
            tasks.spawn(async move {
                let _finalization = FinalizationGuard::new(finalizing);
                // Если задача успела честно закончить до abort, её готовый итог
                // не затираем сообщением об остановке службы.
                if entry.handle.await.is_ok() {
                    return false;
                }

                let elapsed_sec = entry.started.elapsed().as_secs();
                if let Err(e) = pool
                    .update_call(
                        call_id,
                        CallStatus::Failed,
                        None,
                        Some(&reason),
                        0,
                        0,
                        None,
                        elapsed_sec * 1000,
                        None,
                        0,
                        0,
                        0,
                        None,
                    )
                    .await
                {
                    warn!(call_id, error = %e, "остановка: не удалось закрыть строку вызова");
                }
                if let Some(path) = entry.result_path.as_ref() {
                    if let Err(e) =
                        write_result_file(path, &envelope_error(call_id, &entry.agent, &reason))
                    {
                        warn!(call_id, path = %path.display(), error = %e,
                              "остановка: не записал файл-итог вызова");
                    }
                }
                warn!(call_id, agent = %entry.agent, "вызов закрыт при остановке службы");
                true
            });
        }

        let completed = tokio::time::timeout(timeout, async {
            let mut stopped = 0;
            while let Some(result) = tasks.join_next().await {
                if matches!(result, Ok(true)) {
                    stopped += 1;
                }
            }
            stopped
        })
        .await;

        match completed {
            Ok(stopped) => stopped,
            Err(_) => {
                tasks.abort_all();
                warn!(
                    total,
                    timeout_ms = timeout.as_millis(),
                    "истёк срок финализации фоновых вызовов при остановке"
                );
                0
            }
        }
    }

    /// Путь файла-итога по умолчанию: `<runs_dir>/<call_id>-<агент>-<uuid>.json`.
    fn default_result_path(&self, call_id: i64, agent: &str) -> PathBuf {
        let runs_dir = self
            .runs_dir
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        runs_dir.join(format!("{call_id}-{agent}-{}.json", uuid::Uuid::new_v4()))
    }

    /// Закрыть строку вызова ошибкой и записать провал в журнал. Для ошибок ПОСЛЕ
    /// резерва call_id и ДО provider.complete: без этого строка оставалась в
    /// 'running' навсегда, а ждущий её опрос не дожидался конца.
    async fn fail_prepared_call(
        &self,
        call_id: i64,
        agent: &str,
        start: Instant,
        err: &InvokeError,
    ) {
        let latency_ms = start.elapsed().as_millis() as u64;
        if let Err(e) = self
            .pg
            .update_call(
                call_id,
                CallStatus::Failed,
                None,
                Some(&format!("{err}")),
                0,
                0,
                None,
                latency_ms,
                None,
                0,
                0,
                0,
                None,
            )
            .await
        {
            warn!(call_id, error = %e, "строку провалившегося вызова закрыть не удалось");
        }
        error!(call_id, agent, latency_ms, error = %err, "вызов завершился ошибкой");
    }

    /// Быстрая часть вызова ДО provider.complete(): валидация, проверка кеша,
    /// INSERT-заглушка (резерв call_id), рендер промпта, ранний UPDATE
    /// session_id. Медленный LLM-вызов выполняет [`execute_ready`].
    async fn prepare(
        &self,
        req: &InvokeRequest,
        start: Instant,
        deadline: tokio::time::Instant,
        timeout_sec: u64,
        preparing_call_id: Arc<AtomicI64>,
    ) -> Result<Prepared, InvokeError> {
        // Защита от рекурсивных циклов оркестраторов: каждый invoke,
        // инициированный изнутри другого агента через mcp__agents__invoke_agent,
        // приходит с увеличенным orchestration_depth. Корневой вызов от клиента
        // = 0. При превышении отказываемся ДО любых тяжёлых действий (рендер,
        // LLM-вызов, запись в БД).
        if req.orchestration_depth >= self.max_orchestration_depth {
            return Err(InvokeError::MaxDepthExceeded {
                depth: req.orchestration_depth,
                max: self.max_orchestration_depth,
            });
        }

        let agent = self
            .registry
            .get(&req.agent)
            .ok_or_else(|| InvokeError::AgentNotFound(req.agent.clone()))?;

        // Проверка required input.
        let missing: Vec<String> = agent
            .config
            .input
            .required
            .iter()
            .filter(|k| !req.input.contains_key(k.as_str()))
            .cloned()
            .collect();
        if !missing.is_empty() {
            return Err(InvokeError::MissingFields(missing));
        }

        // Variant промпта.
        let variant = req.variant.clone().unwrap_or_else(|| "default".to_string());
        let prompt_tmpl =
            agent
                .prompts
                .get(&variant)
                .ok_or_else(|| InvokeError::VariantNotFound {
                    agent: req.agent.clone(),
                    variant: variant.clone(),
                })?;

        // Провайдер + модель. Тест-override ([agents] force_provider/force_model):
        // если заданы оба — ВСЕ агенты идут через одну модель, игнорируя
        // per-agent [model]. Иначе — каждый по своему config.toml.
        let (provider_name, model_name) = {
            // Короткий захват RwLock: читаем override, клонируем строки, отпускаем
            // ДО любого await ниже (guard не живёт через await-точки).
            let ov = self
                .force_override
                .read()
                .unwrap_or_else(|e| e.into_inner());
            match (&ov.provider, &ov.model) {
                (Some(fp), Some(fm)) => (fp.clone(), fm.clone()),
                _ => (
                    agent.config.model.provider.clone(),
                    agent.config.model.name.clone(),
                ),
            }
        };

        // В промпт идёт ограниченный срез, а в ключ кеша — отпечаток полного
        // состояния: изменения за границей среза тоже инвалидируют ответ.
        let (task_context, task_context_fingerprint) = match req.task_id {
            Some(tid) => match self.pg.read_artifacts(tid, None).await {
                Ok(arts) => (format_task_context(&arts), task_context_fingerprint(&arts)),
                Err(e) => {
                    warn!(task_id = tid, error = %e, "task-store: чтение артефактов упало, срез пуст");
                    (String::new(), String::new())
                }
            },
            None => (String::new(), String::new()),
        };

        // Hash input для трассировки. Нормализация — BTreeMap (sorted keys).
        let input_hash = hash_input(&req.input);

        // Один и тот же ключ идёт и в lookup, и в последующий store. В него
        // входят фактические provider/model, исходный prompt.md и отпечаток доски.
        let cache_key = agent.config.cache.enabled.then(|| {
            cache::compute_key(cache::CacheKeyParts {
                agent_name: &req.agent,
                variant: &variant,
                provider_name: &provider_name,
                model_name: &model_name,
                prompt: prompt_tmpl,
                task_context: &task_context_fingerprint,
                input: &req.input,
                key_fields: &agent.config.cache.key_fields,
                task_id: req.task_id,
            })
        });
        if let Some(key) = cache_key.as_ref() {
            match cache::lookup(self.pg.as_ref(), key.clone()).await {
                Ok(Some(entry)) => {
                    if let Some(mut resp) = build_response_from_cache(
                        &req.agent,
                        &variant,
                        &entry,
                        start,
                        req.parent_call_id,
                        req.orchestration_depth,
                    ) {
                        let call_id = self
                            .pg
                            .insert_call_stub(
                                &req.agent,
                                &variant,
                                &input_hash,
                                &model_name,
                                &provider_name,
                                req.parent_call_id,
                                req.task_id,
                                &self.instance,
                            )
                            .await
                            .map_err(|e| InvokeError::Db(format!("{e}")))?;
                        preparing_call_id.store(call_id, Ordering::SeqCst);
                        resp.metadata.call_id = call_id;
                        self.pg
                            .mark_call_cached(call_id)
                            .await
                            .map_err(|e| InvokeError::Db(format!("{e}")))?;
                        self.pg
                            .update_call(
                                call_id,
                                CallStatus::Done,
                                Some(&entry.output_json),
                                None,
                                resp.metadata.tokens_in,
                                resp.metadata.tokens_out,
                                resp.metadata.cost_usd,
                                resp.metadata.latency_ms,
                                None,
                                0,
                                0,
                                0,
                                None,
                            )
                            .await
                            .map_err(|e| InvokeError::Db(format!("{e}")))?;
                        info!(
                            agent = %req.agent,
                            variant = %variant,
                            "cache hit"
                        );
                        return Ok(Prepared::CacheHit(resp));
                    }
                }
                Ok(None) => {}
                Err(e) => warn!(error = %e, "cache lookup упал, продолжаю без кеша"),
            }
        }

        // Короткий захват RwLock: Arc клонируем, guard отпускаем до любого await
        // ниже — идущий вызов доживает на прежнем наборе провайдеров.
        let provider = self
            .providers
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&provider_name)
            .cloned()
            .ok_or_else(|| InvokeError::UnknownProvider(provider_name.clone()))?;

        // Сборка tera-контекста из входа + служебных полей.
        let mut ctx = tera::Context::new();
        for (k, v) in req.input.iter() {
            ctx.insert(k, v);
        }
        // Объявленные опциональные входы, не переданные в этом вызове, вставляем
        // пустой строкой — иначе tera-strict упадёт на их безусловном упоминании
        // в промпте (например `{{ artifact_name }}` в примере формата ответа,
        // когда вызов прошёл без artifact_name). Тот же приём, что ниже для
        // skills_index/task_context. Guard-блоки `{% if x %}` пустую строку
        // трактуют как ложь — поведение не меняется; ловятся strict-режимом
        // только НЕобъявленные переменные (реальные опечатки).
        //
        // ВАЖНО для авторов промптов: раз пропущенный опциональный вход — это
        // ОПРЕДЕЛЁННАЯ пустая строка (а не undefined), фильтр `| default(value="X")`
        // на него НЕ срабатывает (tera default подставляет дефолт только для
        // undefined). Если нужен НЕпустой дефолт — писать
        // `{% if x %}{{ x }}{% else %}X{% endif %}`, а не `{{ x | default(value="X") }}`.
        for opt in &agent.config.input.optional {
            if !req.input.contains_key(opt) {
                ctx.insert(opt, &"");
            }
        }
        // Служебные поля для агентов-оркестраторов: текущая глубина рекурсии
        // и готовое значение для прокидывания в дочерний invoke_agent.
        // Имена с префиксом `_` чтобы не конфликтовать с доменными полями input.
        ctx.insert("_orchestration_depth", &req.orchestration_depth);
        ctx.insert(
            "_orchestration_depth_plus_one",
            &(req.orchestration_depth + 1),
        );

        // Pre-insert pattern (shared-session):
        // Резервируем call_id ДО provider.complete(), чтобы оркестратор мог
        // прокинуть его в дочерние invoke_agent как parent_call_id и все
        // под-агенты заходили в ОДНУ claude-сессию через --resume.
        // После complete — UPDATE этой же строки с финальными метриками.
        let call_id = self
            .pg
            .insert_call_stub(
                &req.agent,
                &variant,
                &input_hash,
                &model_name,
                &provider_name,
                req.parent_call_id,
                req.task_id,
                &self.instance,
            )
            .await
            .map_err(|e| InvokeError::Db(format!("{e}")))?;
        preparing_call_id.store(call_id, Ordering::SeqCst);
        ctx.insert("_my_call_id", &call_id);
        // Номер зарезервирован — пара к этой записи будет в конце вызова
        // (успех: "вызов завершён", провал: "вызов завершился ошибкой").
        info!(
            call_id,
            agent = %req.agent,
            variant = %variant,
            provider = %provider_name,
            model = %model_name,
            parent_call_id = ?req.parent_call_id,
            task_id = ?req.task_id,
            "вызов начат"
        );
        let turn_sink = spawn_turn_writer(self.pg.clone(), call_id);

        // Клон клиента навыков под коротким захватом: перечитка конфига может
        // подменить его в любой момент, а весь участок ниже работает со своим.
        let skills = self.skills();

        // Push-инъекция индекса навыков (семантический top-k): по брифу из входа
        // ищем релевантные навыки через skill_search (внешний сервис навыков) и кладём
        // компактный список в {{ skills_index }}. Локальная модель получает 3–5
        // точных рецептов вместо всего домена. Вставляем ВСЕГДА (пусто если
        // выключено/ничего не нашлось), иначе tera-strict упадёт на шаблонах,
        // ссылающихся на переменную.
        // Сырой каталог держим отдельно от индекса для промпта: в нём остаются
        // оценки cos и rr, по которым ниже отбираются тела.
        let skills_catalog = if skills.enabled() && agent.config.skill_include_1c {
            let query = crate::skills::build_skill_query(&req.input);
            let waited = Instant::now();
            match tokio::time::timeout_at(deadline, skills.skill_catalog_result(&query, 5)).await {
                Ok(Ok(text)) => text,
                Ok(Err(reason)) => {
                    record_skills_unavailable(&skills, &turn_sink, waited.elapsed(), &reason);
                    String::new()
                }
                Err(_) => {
                    record_skills_unavailable(
                        &skills,
                        &turn_sink,
                        waited.elapsed(),
                        "срок прогона",
                    );
                    String::new()
                }
            }
        } else {
            String::new()
        };
        let skills_index = crate::skills::format_catalog(&skills_catalog);
        ctx.insert("skills_index", &skills_index);

        // Тела самых релевантных навыков — сразу в промпт (skill_bodies_top > 0).
        // Оглавление без тела слабой модели мало помогает: до skill_load она
        // обычно не доходит и пишет код по догадке. Вставляем ВСЕГДА (пусто при
        // выключенном режиме), иначе tera-strict уронит шаблон с переменной.
        let mut skills_bodies = String::new();
        // Имена держим отдельно: без них по журналу видно только объём, а
        // проверять надо ПОПАДАНИЕ отбора. 15.08.2026 вложение тел выключили
        // именно потому, что доезжал навык не по теме, — и заметить это можно
        // было только вручную.
        let mut inserted: Vec<String> = Vec::new();
        // Навыки следом за вложенными — не в промпт, а в запас провайдера на
        // случай петли (см. LlmRequest.fallback_skill_names). Только ИМЕНА: тело
        // грузится, лишь когда понадобилось, и обычный вызов за него не платит
        // ни временем, ни лишним запросом. Основной источник при петле другой —
        // поиск по месту затыка; этот запас нужен, если поиск ничего не вернул.
        const FALLBACK_SKILLS: usize = 1;
        let mut fallback_skill_names: Vec<String> = Vec::new();
        if agent.config.skill_bodies_top > 0 && !skills_catalog.is_empty() {
            let names = crate::skills::top_names_by_rerank(
                &skills_catalog,
                agent.config.skill_bodies_top + FALLBACK_SKILLS,
            );
            let (for_prompt, for_fallback) =
                names.split_at(agent.config.skill_bodies_top.min(names.len()));
            for name in for_prompt {
                let waited = Instant::now();
                match tokio::time::timeout_at(deadline, skills.skill_body_result(name)).await {
                    Ok(Ok(Some(body))) => {
                        skills_bodies.push_str(&format!("\n### Навык: {name}\n\n{body}\n"));
                        inserted.push(name.clone());
                    }
                    Ok(Ok(None)) => {}
                    Ok(Err(reason)) => {
                        record_skills_unavailable(&skills, &turn_sink, waited.elapsed(), &reason);
                    }
                    Err(_) => {
                        record_skills_unavailable(
                            &skills,
                            &turn_sink,
                            waited.elapsed(),
                            "срок прогона",
                        );
                        break;
                    }
                }
            }
            fallback_skill_names = for_fallback.to_vec();
            if !skills_bodies.is_empty() {
                tracing::info!(
                    agent = %agent.config.name,
                    chars = skills_bodies.len(),
                    names = %inserted.join(", "),
                    fallback = %fallback_skill_names.join(", "),
                    "skills_bodies: тела навыков вложены в промпт"
                );
            }
        }
        ctx.insert("skills_bodies", &skills_bodies);

        // Вставляем ВСЕГДА (пусто если нет), иначе tera-strict упадёт на
        // шаблонах с этой переменной.
        ctx.insert("task_context", &task_context);
        // task_id в шаблон: под-агенту он нужен, чтобы дочитать длинную наработку
        // через artifact_read(task_id=…). None (вызов вне задачи) → null, шаблоны
        // оборачивают использование в {% if task_id %}.
        ctx.insert("task_id", &req.task_id);

        // Рендер prompt через tera.one_off (после INSERT stub — чтобы
        // _my_call_id уже был доступен в шаблоне). autoescape=false: промпты —
        // это текст для LLM, не HTML. С autoescape=true Tera экранировала
        // подставляемый код (" → &quot;, / → &#x2F;), и валидатор получал
        // нечитаемый модуль с ложными ParseError на «HTML-сущности».
        let rendered = match tera::Tera::one_off(prompt_tmpl, &ctx, false) {
            Ok(r) => r,
            Err(e) => {
                let err = InvokeError::PromptRender(format!("{e}"));
                self.fail_prepared_call(call_id, &req.agent, start, &err)
                    .await;
                return Err(err);
            }
        };

        // Pre-flight проверка лимита размера входа. Оценка грубая (≈4 символа
        // на токен) — верхняя прикидка по отрендеренному промпту. При
        // превышении отказываемся ДО вызова провайдера, закрыв pre-insert
        // строку ошибкой (как в других ветках после резерва call_id).
        if let Some(max_in) = agent.config.limits.max_input_tokens {
            let estimated = estimate_tokens(&rendered);
            if estimated > max_in {
                let err = InvokeError::InputTooLarge {
                    estimated,
                    max: max_in,
                };
                self.fail_prepared_call(call_id, &req.agent, start, &err)
                    .await;
                return Err(err);
            }
        }

        // Подсказки для claude-cli (если агент работает через subprocess).
        // Логика выбора session_id (shared-session):
        //   1. Если есть parent_call_id (дочерний в задаче) → читаем
        //      session_id parent'а из БД (он там УЖЕ NOT NULL — runtime записал
        //      его ДО старта subprocess parent'а через ранний UPDATE). Дочерний
        //      идёт `--resume <parent_sid>` — ВСЯ задача в одной claude-сессии.
        //   2. Если parent_call_id None (корневой вызов) → ВСЕГДА новая сессия:
        //      генерим новый UUID и передаём как `--session-id <uuid>`. НИКАКОГО
        //      переиспользования session_id между разными корневыми задачами —
        //      это смешало бы чужой контекст. UUID сразу пишем в БД, чтобы
        //      дочерние invoke, которые могут произойти ВНУТРИ provider.complete(),
        //      успели прочитать наш session_id (root claude-cli ещё не закончился).
        // hints (allowed_tools + mcp_config + max_turns) нужны и claude-cli, и
        // прямым провайдерам: openrouter.rs использует их для agentic-loop
        // (веха #3). Session-id логика ниже — только для claude-cli.
        let mut cli_hints = match &agent.config.execution {
            Some(cli_cfg) => {
                let mut hints = match build_cli_hints(cli_cfg, &ctx) {
                    Ok(h) => h,
                    Err(err) => {
                        self.fail_prepared_call(call_id, &req.agent, start, &err)
                            .await;
                        return Err(err);
                    }
                };
                let provider_env = self
                    .provider_env
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                hints.mcp_env = match collect_mcp_env(hints.mcp_config.as_deref(), &provider_env) {
                    Ok((values, missing)) => {
                        hints.mcp_env_missing = missing;
                        values
                    }
                    Err(message) => {
                        let err = InvokeError::McpConfig(message);
                        self.fail_prepared_call(call_id, &req.agent, start, &err)
                            .await;
                        return Err(err);
                    }
                };
                // Корни fs_* агента (allowed_roots): рабочий каталог вызова
                // обязан быть задан и лежать внутри одного из них, иначе
                // провайдер не запускаем.
                if let Some(roots) = &cli_cfg.allowed_roots {
                    if let Err(err) = ensure_cwd_within_agent_roots(hints.cwd.as_deref(), roots) {
                        self.fail_prepared_call(call_id, &req.agent, start, &err)
                            .await;
                        return Err(err);
                    }
                }
                if provider_name == "claude-cli" {
                    let resume_sid: Option<String> = match req.parent_call_id {
                        Some(parent_id) => match self.pg.get_call_session_id(parent_id).await {
                            Ok(opt) => opt,
                            Err(e) => {
                                warn!(error = %e, parent_id, "не прочитал parent session_id");
                                None
                            }
                        },
                        // Корневой вызов (parent_call_id == None) НИКОГДА не
                        // переиспользует чужую сессию: ниже сгенерится новый UUID
                        // и уйдёт как --session-id. Переиспользование между разными
                        // корневыми задачами смешивало бы чужой контекст (разные
                        // пользовательские запросы в одной claude-сессии).
                        None => None,
                    };

                    // Финальный session_id текущего вызова — для раннего UPDATE.
                    let final_sid: String = match &resume_sid {
                        Some(sid) => sid.clone(),
                        None => uuid::Uuid::new_v4().to_string(),
                    };
                    if let Some(sid) = resume_sid {
                        info!(
                            agent = %req.agent,
                            parent_call_id = ?req.parent_call_id,
                            session_id = %sid,
                            "переиспользую сессию через --resume"
                        );
                        hints.resume_session_id = Some(sid);
                    } else {
                        info!(
                            agent = %req.agent,
                            session_id = %final_sid,
                            "стартую новую сессию через --session-id"
                        );
                        hints.new_session_id = Some(final_sid.clone());
                    }

                    // Ранний UPDATE pre-insert row: пишем session_id ДО старта
                    // claude-cli, чтобы дочерние invoke сразу его видели. Иначе
                    // shared-session не работает: дочерний попадает к нам по MCP
                    // пока parent.complete() в await'е, и read parent.session_id
                    // вернёт NULL.
                    if let Err(e) = self.pg.set_call_session_id(call_id, &final_sid).await {
                        warn!(error = %e, call_id, "ранний UPDATE session_id упал");
                    }
                } // конец if provider_name == "claude-cli"

                Some(hints)
            }
            None => None,
        };

        let call_cwd = cli_hints.as_ref().and_then(|hints| hints.cwd.clone());
        let agent_roots = agent
            .config
            .execution
            .as_ref()
            .and_then(|execution| execution.allowed_roots.clone());
        let (call_key, call_key_guard) = self.issue_call_key(
            call_id,
            call_cwd,
            agent_roots,
            req.parent_call_id,
            req.orchestration_depth,
        );
        let own_mcp_port = *self.own_mcp_port.read().unwrap_or_else(|e| e.into_inner());
        if let (Some(hints), Some(port)) = (cli_hints.as_mut(), own_mcp_port) {
            if let Some(raw) = hints.mcp_config.as_deref() {
                hints.mcp_config = Some(inject_call_key_into_mcp_config(raw, port, &call_key));
            }
        }

        // provider.complete() выполняет execute_ready (sync inline или в фоне
        // для async). Здесь только собираем LlmRequest и возвращаем
        // подготовленный ReadyCall с уже зарезервированным call_id.
        let llm_req = LlmRequest {
            model: model_name.clone(),
            system_prompt: rendered,
            user_input: String::new(),
            temperature: agent.config.model.temperature.unwrap_or(0.7),
            max_tokens: agent.config.model.max_tokens.unwrap_or(4096),
            top_p: agent.config.model.top_p,
            extra_body: agent.config.model.extra_body.clone(),
            timeout: deadline.saturating_duration_since(tokio::time::Instant::now()),
            cli_hints,
            // Ходы пишутся в PG по мере выполнения. Пакетная запись в конце
            // теряла всё, если вызов не доживал до конца: при таймауте задача
            // остаётся в статусе running и до кода записи не доходит вовсе.
            turn_sink: Some(turn_sink),
            fallback_skill_names,
            prompt_skill_names: inserted.clone(),
            // Клиент навыков нужен провайдеру, чтобы при петле искать навык по
            // тому месту, где модель встала. Выключен поиск — останется запас.
            skills: if skills.enabled() {
                Some(skills.clone())
            } else {
                None
            },
        };

        Ok(Prepared::Ready(Box::new(ReadyCall {
            call_id,
            agent_name: req.agent.clone(),
            agent: agent.clone(),
            provider,
            provider_name,
            model_name,
            variant,
            parent_call_id: req.parent_call_id,
            orchestration_depth: req.orchestration_depth,
            cache_key,
            llm_req,
            start,
            deadline,
            timeout_sec,
            call_key_guard,
            runs_dir: self
                .runs_dir
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
        })))
    }
}

/// Поднять писатель ходов: возвращает канал, всё присланное в него пишется
/// хранилищем сразу, по одной записи. Живёт, пока жив отправитель
/// (то есть пока идёт прогон), и завершается сам, когда канал закрыт.
///
/// Смысл — наблюдаемость незавершённых вызовов. Пакетная запись в конце давала
/// строки только у доживших до финала: у зависших и упавших в базе не было
/// ничего, и разбирать провал было нечем.
fn spawn_turn_writer(
    store: Arc<dyn Store>,
    call_id: i64,
) -> tokio::sync::mpsc::UnboundedSender<Value> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Value>();
    tokio::spawn(async move {
        while let Some(rec) = rx.recv().await {
            let seq = rec.get("seq").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            let event = rec
                .get("event")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            // Метку времени ставит сам провайдер в момент события; здесь она
            // только переносится, иначе задержка записи исказила бы длительности.
            let ts = rec
                .get("ts_ms")
                .and_then(|v| v.as_i64())
                .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
            let record_text = rec.to_string();
            if let Err(e) = store
                .append_turn(call_id, seq, &event, &record_text, ts)
                .await
            {
                tracing::warn!(call_id, seq, error = %e, "потоковая запись хода упала");
            }
        }
    });
    tx
}

fn record_skills_unavailable(
    skills: &crate::skills::SkillsClient,
    sink: &tokio::sync::mpsc::UnboundedSender<Value>,
    waited: Duration,
    reason: &str,
) {
    let address = crate::providers::mcp_client::safe_server_address(skills.address());
    let reason_kind =
        if reason.contains("срок") || reason.to_ascii_lowercase().contains("timed out") {
            "срок"
        } else {
            "ошибка"
        };
    warn!(
        address = %address,
        reason_kind,
        reason = %reason,
        waited_ms = waited.as_millis() as u64,
        "сервис навыков недоступен"
    );
    let _ = sink.send(serde_json::json!({
        "seq": -1,
        "event": "skills_unavailable",
        "ts_ms": chrono::Utc::now().timestamp_millis(),
        "address": address,
        "reason_kind": reason_kind,
        "reason": reason,
        "waited_ms": waited.as_millis() as u64,
    }));
}

async fn finish_timed_out_call(
    pg: Arc<dyn Store>,
    call_id: i64,
    agent_name: &str,
    start: Instant,
    timeout_sec: u64,
) {
    let detail = format!("превышен общий срок прогона ({timeout_sec} с)");
    error!(call_id, agent = %agent_name, timeout_sec, "вызов превысил общий срок");
    let write = pg.update_call(
        call_id,
        CallStatus::Failed,
        None,
        Some(&detail),
        0,
        0,
        None,
        start.elapsed().as_millis() as u64,
        None,
        0,
        0,
        0,
        None,
    );
    match tokio::time::timeout(Duration::from_secs(5), write).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            warn!(call_id, error = %e, "не удалось записать статус после истечения срока")
        }
        Err(_) => {
            warn!(
                call_id,
                "запись статуса прервана: истекли 5 с из дополнительного срока"
            )
        }
    }
}

/// Медленная часть вызова: provider.complete() → парсинг → финальный UPDATE
/// строки agent_calls → запись кеша. Пишет результат и при успехе, и при
/// ошибке. Возвращает полный или неполный результат; в async-пути строку затем
/// читает `poll_call`.
async fn execute_ready_guarded(
    pg: Arc<dyn Store>,
    ready: ReadyCall,
) -> Result<CompletedCall, InvokeError> {
    let call_id = ready.call_id;
    let agent_name = ready.agent_name.clone();
    let start = ready.start;
    let deadline = ready.deadline;
    let timeout_sec = ready.timeout_sec;
    let mut task = AbortOnDrop::new(tokio::spawn(execute_ready(pg.clone(), ready)));
    let joined = tokio::time::timeout_at(deadline, task.join()).await;
    match joined {
        Err(_) => {
            task.abort();
            finish_timed_out_call(pg, call_id, &agent_name, start, timeout_sec).await;
            Err(InvokeError::RunTimeout { timeout_sec })
        }
        Ok(Ok(result)) => result,
        Ok(Err(join_error)) => {
            let detail = if join_error.is_panic() {
                format!("фоновая задача завершилась паникой: {join_error}")
            } else {
                format!("фоновая задача была неожиданно прервана: {join_error}")
            };
            error!(call_id, agent = %agent_name, error = %detail, "фоновый вызов аварийно завершён");
            if let Err(write_error) = pg
                .update_call(
                    call_id,
                    CallStatus::Failed,
                    None,
                    Some(&detail),
                    0,
                    0,
                    None,
                    start.elapsed().as_millis() as u64,
                    None,
                    0,
                    0,
                    0,
                    None,
                )
                .await
            {
                let persistence_error =
                    format!("{detail}; не удалось записать итог: {write_error}");
                let _ = pg
                    .update_call(
                        call_id,
                        CallStatus::PersistenceFailed,
                        None,
                        Some(&persistence_error),
                        0,
                        0,
                        None,
                        start.elapsed().as_millis() as u64,
                        None,
                        0,
                        0,
                        0,
                        None,
                    )
                    .await;
                return Err(InvokeError::PersistenceFailed {
                    call_id,
                    error: persistence_error,
                    response: None,
                });
            }
            Err(InvokeError::BackgroundPanic(detail))
        }
    }
}

async fn execute_ready(pg: Arc<dyn Store>, ready: ReadyCall) -> Result<CompletedCall, InvokeError> {
    let ReadyCall {
        call_id,
        agent_name,
        agent,
        provider,
        provider_name,
        model_name,
        variant,
        parent_call_id,
        orchestration_depth,
        cache_key,
        llm_req,
        start,
        deadline: _,
        timeout_sec: _,
        call_key_guard: _call_key_guard,
        runs_dir,
    } = ready;

    let llm_resp = match provider.complete(llm_req).await {
        Ok(r) => r,
        Err(e) => {
            let err_text = format!("{e}");
            // Ошибки agentic-loop несут накопленные токены завершённых ходов.
            let (ti, to, cost) = match &e {
                LlmError::WithUsage {
                    tokens_in,
                    tokens_out,
                    cost,
                    ..
                }
                | LlmError::MaxTurns {
                    tokens_in,
                    tokens_out,
                    cost,
                    ..
                } => (*tokens_in, *tokens_out, *cost),
                _ => (0, 0, None),
            };
            if let Err(write_error) = pg
                .update_call(
                    call_id,
                    CallStatus::Failed,
                    None,
                    Some(&err_text),
                    ti,
                    to,
                    cost,
                    start.elapsed().as_millis() as u64,
                    None, // session_id у неуспешных не сохраняем
                    ti,   // raw_input ≈ tokens_in (openrouter без cache-разбивки)
                    0,
                    0,
                    None,
                )
                .await
            {
                let persistence_error = format!(
                    "не удалось записать ошибку провайдера: {write_error}; исход провайдера: {err_text}"
                );
                let _ = pg
                    .update_call(
                        call_id,
                        CallStatus::PersistenceFailed,
                        None,
                        Some(&persistence_error),
                        ti,
                        to,
                        cost,
                        start.elapsed().as_millis() as u64,
                        None,
                        ti,
                        0,
                        0,
                        None,
                    )
                    .await;
                error!(
                    call_id,
                    agent = %agent_name,
                    error = %persistence_error,
                    "не удалось записать итог провалившегося вызова"
                );
                return Err(InvokeError::PersistenceFailed {
                    call_id,
                    error: persistence_error,
                    response: None,
                });
            }
            // Транскрипт здесь уже не пишем: ходы легли в agent_turns потоком,
            // по мере выполнения (spawn_turn_writer). Прежняя запись из ветки
            // MaxTurns покрывала лишь один вид провала из шести.
            error!(
                call_id,
                agent = %agent_name,
                provider = %provider_name,
                latency_ms = start.elapsed().as_millis() as u64,
                error = %err_text,
                "вызов завершился ошибкой"
            );
            return Err(e.into());
        }
    };

    // Парсинг ответа: для format=json перебираем кандидатов (ответ как есть →
    // тело обрамления → кусок по скобкам) с проверкой разбором. Без retry к
    // провайдеру.
    // Разбор не удался — ответ негоден: у format=json вызывающий ждёт объект, а
    // получает строку и падает сам. Такой ответ нельзя класть в кеш (см. ниже).
    let mut incomplete_reasons = Vec::new();
    if matches!(
        llm_resp.finish_reason.trim().to_ascii_lowercase().as_str(),
        "length" | "max_tokens" | "content_filter"
    ) {
        incomplete_reasons.push(format!(
            "модель завершила ответ с finish_reason={}",
            llm_resp.finish_reason
        ));
    }
    let result_value: Value = match agent.config.response.format {
        ResponseFormat::Text => Value::String(llm_resp.content.clone()),
        ResponseFormat::Json => match parse_json_tolerant(&llm_resp.content) {
            Ok(v) => v,
            Err(e) => {
                warn!(
                    agent = %agent_name,
                    error = %e,
                    "ответ модели не валидный JSON ни одним из кандидатов, возвращаю как text"
                );
                let saved = write_raw_response(&runs_dir, call_id, &agent_name, &llm_resp.content);
                incomplete_reasons.push(match saved {
                    Some(path) => format!(
                        "ответ format=json не удалось разобрать как JSON (сырой ответ: {})",
                        path.display()
                    ),
                    None => "ответ format=json не удалось разобрать как JSON".to_string(),
                });
                Value::String(llm_resp.content.clone())
            }
        },
    };

    // Лёгкая валидация против schema_file — warn всегда; при schema_strict
    // несоответствие ещё и делает вызов неполным.
    if matches!(agent.config.response.format, ResponseFormat::Json) {
        if let Some(schema) = &agent.schema {
            if let Err(e) = validate_against_schema(&result_value, schema) {
                warn!(
                    agent = %agent_name,
                    error = %e,
                    "ответ не соответствует schema_file"
                );
                if agent.config.response.schema_strict {
                    let saved =
                        write_raw_response(&runs_dir, call_id, &agent_name, &llm_resp.content);
                    incomplete_reasons.push(match saved {
                        Some(path) => format!(
                            "ответ не соответствует schema_file: {} (сырой ответ: {})",
                            e,
                            path.display()
                        ),
                        None => format!("ответ не соответствует schema_file: {e}"),
                    });
                }
            }
        }
    }

    let latency_ms = start.elapsed().as_millis() as u64;

    let response = InvokeResponse {
        result: result_value,
        metadata: InvokeMetadata {
            agent: agent_name.clone(),
            variant: variant.clone(),
            model_used: model_name,
            provider: provider_name.clone(),
            tokens_in: llm_resp.tokens_in,
            tokens_out: llm_resp.tokens_out,
            cost_usd: llm_resp.cost_usd,
            latency_ms,
            cached: false,
            call_id,
            parent_call_id,
            orchestration_depth,
        },
    };
    let incomplete_error = (!incomplete_reasons.is_empty()).then(|| incomplete_reasons.join("; "));
    let call_status = if incomplete_error.is_some() {
        CallStatus::Incomplete
    } else {
        CallStatus::Done
    };

    let output_json =
        serde_json::to_string(&response.result).map_err(|e| InvokeError::Db(format!("{e}")))?;
    if let Err(write_error) = pg
        .update_call(
            call_id,
            call_status,
            Some(&output_json),
            incomplete_error.as_deref(),
            llm_resp.tokens_in,
            llm_resp.tokens_out,
            llm_resp.cost_usd,
            latency_ms,
            llm_resp.session_id.as_deref(),
            llm_resp.raw_input_tokens,
            llm_resp.cache_creation_input_tokens,
            llm_resp.cache_read_input_tokens,
            llm_resp.reasoning.as_deref(),
        )
        .await
    {
        let persistence_error = format!("не удалось записать итог вызова: {write_error}");
        error!(
            call_id,
            agent = %agent_name,
            error = %persistence_error,
            "вызов выполнен, но итог не записан в базу"
        );
        // Первый отказ уже является отдельным исходом. Повторный минимальный
        // UPDATE даёт восстановившемуся хранилищу снять строку с `running`,
        // при этом сохраняет сам ответ модели.
        let _ = pg
            .update_call(
                call_id,
                CallStatus::PersistenceFailed,
                Some(&output_json),
                Some(&persistence_error),
                llm_resp.tokens_in,
                llm_resp.tokens_out,
                llm_resp.cost_usd,
                latency_ms,
                llm_resp.session_id.as_deref(),
                llm_resp.raw_input_tokens,
                llm_resp.cache_creation_input_tokens,
                llm_resp.cache_read_input_tokens,
                llm_resp.reasoning.as_deref(),
            )
            .await;
        return Err(InvokeError::PersistenceFailed {
            call_id,
            error: persistence_error,
            response: Some(Box::new(response)),
        });
    }

    // Транскрипт уже в agent_turns: его пишет spawn_turn_writer по ходу прогона.

    // max_cost_usd — мягкая проверка: предупреждаем, но НЕ блокируем.
    if let (Some(max_cost), Some(cost_usd)) = (agent.config.limits.max_cost_usd, llm_resp.cost_usd)
    {
        if cost_usd > max_cost {
            warn!(
                agent = %agent_name,
                cost_usd = ?llm_resp.cost_usd,
                max_cost_usd = max_cost,
                "вызов превысил max_cost_usd (не блокирую)"
            );
        }
    }

    info!(
        agent = %agent_name,
        variant = %variant,
        provider = %provider_name,
        tokens_in = llm_resp.tokens_in,
        tokens_out = llm_resp.tokens_out,
        cost_usd = ?llm_resp.cost_usd,
        latency_ms,
        call_id,
        "вызов завершён"
    );

    // Сохранение в кеш (после успешного вызова) — но только если ответ разобрался.
    //
    // Негодный ответ в кеше хуже отсутствия кеша: он живёт весь TTL (у части
    // агентов 30 дней) и возвращается за миллисекунду, поэтому повторный заход
    // получает тот же мусор, НЕ обращаясь к модели, и правка настроек агента
    // (предел вывода, промпт) выглядит бесполезной — проверить её нечем.
    // Поймано 24.08.2026: DeepSeek Flash обрывал JSON по пределу вывода, обрывок
    // осел в кеше, и повторы падали мгновенно с тем же обрывом.
    if let Some(key) = cache_key {
        if let Some(error) = incomplete_error {
            warn!(
                agent = %response.metadata.agent,
                "неполный ответ — в кеш не кладу"
            );
            return Ok(CompletedCall::Incomplete { response, error });
        }
        let output_json = serde_json::to_string(&response.result).unwrap_or_default();
        let metadata_json = serde_json::to_string(&response.metadata).unwrap_or_default();
        let ttl = agent.config.cache.ttl_sec;
        let cache_pg = pg.clone();
        tokio::spawn(async move {
            if let Err(e) =
                cache::store(cache_pg.as_ref(), key, output_json, metadata_json, ttl).await
            {
                warn!(error = %e, "cache store упал");
            }
        });
    }

    match incomplete_error {
        Some(error) => Ok(CompletedCall::Incomplete { response, error }),
        None => Ok(CompletedCall::Done(response)),
    }
}

/// Удалить прежний итог до пуска вызова: ждущий появления файла не должен
/// принять старый конверт за результат нового запуска.
fn remove_old_result_file(path: &Path) {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => warn!(path = %path.display(), error = %e,
                        "не удалось удалить старый файл-итог перед запуском"),
    }
}

/// Путь файла-итога для стартовой пометки осиротевшего вызова. Старые строки
/// без сохранённого пути используют прежнее имя, которое было известно им при
/// запуске до добавления UUID.
pub(crate) fn orphan_result_path(runs_dir: &Path, call: &OrphanedCall) -> PathBuf {
    call.result_path
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| runs_dir.join(format!("{}-{}.json", call.id, call.agent_name)))
}

/// Удалить старые JSON-итоги из `runs_dir`. Срок должен совпадать с retention
/// строк `agent_calls`, иначе один из двух журналов снова начнёт расти без меры.
pub(crate) fn cleanup_result_files(runs_dir: &Path, ttl: Duration) -> u64 {
    let cutoff = SystemTime::now().checked_sub(ttl).unwrap_or(UNIX_EPOCH);
    cleanup_result_files_before(runs_dir, cutoff)
}

fn cleanup_result_files_before(runs_dir: &Path, cutoff: SystemTime) -> u64 {
    let entries = match std::fs::read_dir(runs_dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return 0,
        Err(e) => {
            warn!(path = %runs_dir.display(), error = %e,
                  "не удалось прочитать каталог файлов-итогов для чистки");
            return 0;
        }
    };

    let mut deleted = 0;
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => {
                warn!(path = %runs_dir.display(), error = %e,
                      "не удалось прочитать запись каталога файлов-итогов");
                continue;
            }
        };
        let path = entry.path();
        // Кроме самих итогов чистим `-raw.txt` — сырые ответы, сохранённые при
        // неудачном разборе JSON: живут столько же, сколько итоги, иначе
        // каталог растёт без меры. Посторонние файлы не трогаем.
        let is_result = path.extension().and_then(|ext| ext.to_str()) == Some("json");
        let is_raw_answer = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with("-raw.txt"));
        if !is_result && !is_raw_answer {
            continue;
        }
        let modified = match entry.metadata().and_then(|meta| meta.modified()) {
            Ok(modified) => modified,
            Err(e) => {
                warn!(path = %path.display(), error = %e,
                      "не удалось прочитать время файла-итога");
                continue;
            }
        };
        if modified < cutoff {
            match std::fs::remove_file(&path) {
                Ok(()) => deleted += 1,
                Err(e) => warn!(path = %path.display(), error = %e,
                                "не удалось удалить старый файл-итог"),
            }
        }
    }
    deleted
}

/// Конверт готового итога для файла `result_path` — та же форма, что отдаёт
/// tool: {status, call_id, agent, variant, result, metadata}.
fn envelope_done(resp: &InvokeResponse) -> Value {
    serde_json::json!({
        "status": "done",
        "call_id": resp.metadata.call_id,
        "agent": resp.metadata.agent,
        "variant": resp.metadata.variant,
        "result": resp.result,
        "metadata": resp.metadata,
    })
}

fn envelope_incomplete(resp: &InvokeResponse, error: &str) -> Value {
    serde_json::json!({
        "status": "incomplete",
        "call_id": resp.metadata.call_id,
        "agent": resp.metadata.agent,
        "variant": resp.metadata.variant,
        "result": resp.result,
        "metadata": resp.metadata,
        "error": error,
    })
}

fn envelope_failure(status: &str, call_id: i64, agent: &str, error: &str) -> Value {
    serde_json::json!({
        "status": status,
        "call_id": call_id,
        "agent": agent,
        "error": error,
    })
}

fn envelope_persistence_failed(
    call_id: i64,
    agent: &str,
    response: Option<&InvokeResponse>,
    error: &str,
) -> Value {
    match response {
        Some(resp) => serde_json::json!({
            "status": "persistence_failed",
            "call_id": call_id,
            "agent": agent,
            "variant": resp.metadata.variant,
            "result": resp.result,
            "metadata": resp.metadata,
            "error": error,
        }),
        None => envelope_failure("persistence_failed", call_id, agent, error),
    }
}

/// Конверт неудачи для файла `result_path`. `pub(crate)` — им же закрываются
/// осиротевшие вызовы при старте службы (см. `main`).
pub(crate) fn envelope_error(call_id: i64, agent: &str, error: &str) -> Value {
    envelope_failure("error", call_id, agent, error)
}

/// Записать конверт итога: сначала во временный файл рядом, затем
/// переименовать. Читатель, ждущий появления `path`, никогда не увидит
/// половину JSON — файл появляется целиком. `pub(crate)` — этим же путём
/// пишутся файлы-итоги осиротевших вызовов при старте службы.
pub(crate) fn write_result_file(path: &Path, envelope: &Value) -> std::io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "у пути файла-итога нет родительского каталога",
        )
    })?;
    std::fs::create_dir_all(parent)?;
    let file_name = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "у пути файла-итога нет имени файла",
        )
    })?;
    let mut tmp_name = std::ffi::OsString::from(".");
    tmp_name.push(file_name);
    tmp_name.push(format!(".{}.part", uuid::Uuid::new_v4()));
    let tmp = parent.join(tmp_name);
    let body = serde_json::to_vec_pretty(envelope).unwrap_or_default();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)?;
    if let Err(e) = std::io::Write::write_all(&mut file, &body) {
        drop(file);
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    drop(file);
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

/// Опрашивать строку agent_calls по call_id до завершения или истечения
/// wait_sec. wait_sec зажимается в [0, 55] — потолок ниже 60-секундного
/// MCP-таймаута оркестратора, чтобы сам wait-ответ гарантированно вернулся.
/// wait_sec=0 → одна проверка без ожидания.
async fn poll_call(pg: Arc<dyn Store>, call_id: i64, wait_sec: u64) -> InvokeOutcome {
    const POLL_INTERVAL: Duration = Duration::from_millis(500);
    let capped = wait_sec.min(55);
    let deadline = Instant::now() + Duration::from_secs(capped);
    loop {
        match read_call_outcome(pg.clone(), call_id).await {
            Ok(Some(outcome)) => return outcome, // done / error / не найден
            Ok(None) => {}                       // ещё running
            Err(e) => {
                warn!(call_id, error = %e, "poll_call: чтение строки упало");
                return InvokeOutcome::PersistenceFailed {
                    call_id,
                    response: None,
                    error: format!("хранилище недоступно: {e}"),
                };
            }
        }
        if Instant::now() >= deadline {
            return InvokeOutcome::Running { call_id };
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// Прочитать одну строку agent_calls и собрать InvokeOutcome:
///   Ok(Some(Done))   — status='done', результат восстановлен из output_json;
///   Ok(Some(Failed)) — status='failed'/'error' (или строка не найдена);
///   Ok(None)         — status='running', ещё бежит.
async fn read_call_outcome(
    pg: Arc<dyn Store>,
    call_id: i64,
) -> anyhow::Result<Option<InvokeOutcome>> {
    let row = match pg.get_call_row(call_id).await? {
        Some(r) => r,
        None => {
            return Ok(Some(InvokeOutcome::Failed {
                call_id,
                error: format!("call_id {call_id} не найден"),
            }));
        }
    };

    let response = || {
        let result: Value = row
            .output_json
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(Value::Null);
        InvokeResponse {
            result,
            metadata: InvokeMetadata {
                agent: row.agent_name.clone(),
                variant: row.variant.clone(),
                model_used: row.model_used.clone(),
                provider: row.provider.clone(),
                tokens_in: row.tokens_in.unwrap_or(0) as u32,
                tokens_out: row.tokens_out.unwrap_or(0) as u32,
                cost_usd: row.cost_usd,
                latency_ms: row.latency_ms.unwrap_or(0) as u64,
                cached: row.cached,
                call_id,
                // orchestration_depth в строке не хранится; для опрошенного
                // ответа выставляем 0 — узел дерева оркестратору тут не нужен.
                parent_call_id: row.parent_call_id,
                orchestration_depth: 0,
            },
        }
    };

    match row.status.as_str() {
        "done" => Ok(Some(InvokeOutcome::Done(response()))),
        "incomplete" => Ok(Some(InvokeOutcome::Incomplete {
            response: response(),
            error: row
                .error
                .clone()
                .unwrap_or_else(|| "модель вернула неполный результат".to_string()),
        })),
        "failed" | "error" => Ok(Some(InvokeOutcome::Failed {
            call_id,
            error: row
                .error
                .clone()
                .unwrap_or_else(|| "неизвестная ошибка".to_string()),
        })),
        "cancelled" => Ok(Some(InvokeOutcome::Cancelled {
            call_id,
            error: row
                .error
                .clone()
                .unwrap_or_else(|| "вызов отменён вручную".to_string()),
        })),
        "persistence_failed" => Ok(Some(InvokeOutcome::PersistenceFailed {
            call_id,
            response: row.output_json.as_ref().map(|_| response()),
            error: row
                .error
                .clone()
                .unwrap_or_else(|| "запись итога в хранилище не удалась".to_string()),
        })),
        "running" => Ok(None),
        status => Ok(Some(InvokeOutcome::Failed {
            call_id,
            error: format!("неизвестный статус вызова '{status}'"),
        })),
    }
}

/// Разобрать ответ модели как JSON, перебирая кандидатов с проверкой разбором.
///
/// Порядок: ответ как есть → тело обрамления ```` ```json ... ``` ```` → кусок
/// от первой открывающей скобки до последней закрывающей. Побеждает первый
/// кандидат, который разобрался; ни один не подошёл — ошибка разбора самого
/// ответа. Проверка разбором обязательна: одну стратегию вслепую сбивал план,
/// у которого внутри строкового поля лежали собственные блоки кода.
fn parse_json_tolerant(s: &str) -> Result<Value, serde_json::Error> {
    let trimmed = s.trim();
    let as_is = serde_json::from_str::<Value>(trimmed);
    if as_is.is_ok() {
        return as_is;
    }
    for candidate in [fence_body(trimmed), json_span(trimmed)]
        .into_iter()
        .flatten()
    {
        if let Ok(v) = serde_json::from_str::<Value>(candidate) {
            return Ok(v);
        }
    }
    as_is
}

/// Тело обрамления: от первого открывающего ```` ``` ```` до ПОСЛЕДНЕГО
/// закрывающего.
///
/// Обрамление снимается, даже когда перед ним стоит пояснение: модели пишут
/// фразу вроде «Проверил три места» и только потом JSON. Поддерживаются пометки
/// языка json и javascript, а также рамка без пометки. Закрывающее ищется
/// последним, потому что внутри строковых полей JSON бывают свои блоки кода —
/// по первому вхождению ответ обрезался на чужой рамке.
fn fence_body(s: &str) -> Option<&str> {
    let open = s.find("```")?;
    let after_tag = &s[open + 3..];
    let body = after_tag
        .strip_prefix("json")
        .or_else(|| after_tag.strip_prefix("javascript"))
        .unwrap_or(after_tag)
        .trim_start();
    let inner = match body.rfind("```") {
        Some(close) => body[..close].trim(),
        // Рамку открыли и не закрыли — берём всё, что после открытия.
        None => body.trim_end(),
    };
    if inner.is_empty() {
        None
    } else {
        Some(inner)
    }
}

/// Сохранить сырой ответ модели рядом с файлом-итогом вызова, когда разобрать
/// его как JSON не удалось ни одним кандидатом. Без этого причина отказа
/// доставалась только ручным чтением поля `result` из строки вызова.
fn write_raw_response(
    runs_dir: &Path,
    call_id: i64,
    agent: &str,
    content: &str,
) -> Option<PathBuf> {
    let path = runs_dir.join(format!("{call_id}-{agent}-raw.txt"));
    match std::fs::create_dir_all(runs_dir).and_then(|()| std::fs::write(&path, content)) {
        Ok(()) => Some(path),
        Err(e) => {
            warn!(call_id, path = %path.display(), error = %e,
                  "не удалось сохранить сырой ответ модели");
            None
        }
    }
}

/// Кусок от первой открывающей скобки до последней закрывающей.
///
/// Нужен для ответов, где JSON идёт без рамки, но с текстом вокруг.
fn json_span(s: &str) -> Option<&str> {
    let start = s.find(['{', '['])?;
    let end = s.rfind(['}', ']'])?;
    if end > start {
        Some(s[start..=end].trim())
    } else {
        None
    }
}

/// Грубая оценка числа токенов по тексту (≈4 символа на токен). Не точная
/// токенизация — верхняя прикидка для pre-flight проверки max_input_tokens.
/// Для кириллицы реальное число токенов обычно выше, эта оценка занижает —
/// поэтому это «мягкий» барьер, а не точный учёт.
fn estimate_tokens(text: &str) -> u32 {
    (text.chars().count() / 4) as u32
}

/// Лёгкая проверка JSON-ответа против схемы агента. Покрывает верхний уровень:
/// тип, наличие required-полей и грубое соответствие типов top-level
/// properties. НЕ полная JSON Schema (без `$ref`/`anyOf`/вложенной рекурсии) —
/// задача поймать «модель не вернула обязательное поле» или «не тот тип на
/// верхнем уровне», а не заменить валидатор схемы целиком.
fn validate_against_schema(value: &Value, schema: &Value) -> Result<(), String> {
    if let Some(t) = schema.get("type").and_then(|v| v.as_str()) {
        if !json_type_matches(value, t) {
            return Err(format!("ожидался тип '{t}' на верхнем уровне"));
        }
    }
    if let Some(required) = schema.get("required").and_then(|v| v.as_array()) {
        let obj = value.as_object();
        for field in required.iter().filter_map(|v| v.as_str()) {
            let present = obj.map(|o| o.contains_key(field)).unwrap_or(false);
            if !present {
                return Err(format!("отсутствует обязательное поле '{field}'"));
            }
        }
    }
    if let (Some(props), Some(obj)) = (
        schema.get("properties").and_then(|v| v.as_object()),
        value.as_object(),
    ) {
        for (key, pschema) in props {
            // Тип проверяем только если он задан строкой. Массивный тип вида
            // ["integer","null"] (nullable) пропускаем — иначе ложные
            // срабатывания на корректных null/значениях.
            if let (Some(val), Some(Value::String(t))) = (obj.get(key), pschema.get("type")) {
                if !val.is_null() && !json_type_matches(val, t) {
                    return Err(format!("поле '{key}': ожидался тип '{t}'"));
                }
            }
        }
    }
    Ok(())
}

/// Соответствует ли JSON-значение типу из JSON Schema. Неизвестный тип
/// считаем совпавшим (не валим на экзотике вне нашего покрытия).
fn json_type_matches(value: &Value, ty: &str) -> bool {
    match ty {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "integer" => value.is_i64() || value.is_u64(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    }
}

/// Восстановить InvokeResponse из закешированной записи. Подменяем
/// `cached=true` и реальный latency_ms (с момента начала запроса до возврата).
/// `parent_call_id` и `orchestration_depth` берутся из текущего запроса,
/// а не из закешированного метадаты — узел дерева вызовов всегда «здесь и сейчас».
fn build_response_from_cache(
    agent: &str,
    variant: &str,
    entry: &cache::CachedEntry,
    start: Instant,
    parent_call_id: Option<i64>,
    orchestration_depth: u32,
) -> Option<InvokeResponse> {
    let result: Value = serde_json::from_str(&entry.output_json).ok()?;
    let mut metadata: InvokeMetadata = serde_json::from_str(&entry.metadata_json).ok()?;
    metadata.cached = true;
    metadata.latency_ms = start.elapsed().as_millis() as u64;
    metadata.agent = agent.to_string();
    metadata.variant = variant.to_string();
    metadata.parent_call_id = parent_call_id;
    metadata.orchestration_depth = orchestration_depth;
    Some(InvokeResponse { result, metadata })
}

/// Проверка корней fs_* инструментов агента (allowed_roots в [execution]):
/// рабочий каталог вызова обязан быть задан и лежать внутри хотя бы одного
/// корня агента. Корни, которые не удалось канонизировать, пропускаются —
/// как в fs_safe_path.
fn ensure_cwd_within_agent_roots(cwd: Option<&Path>, roots: &[PathBuf]) -> Result<(), InvokeError> {
    let cwd = cwd.ok_or(InvokeError::AgentRootsWorkDirMissing)?;
    let roots_text: Vec<String> = roots
        .iter()
        .map(|root| root.display().to_string())
        .collect();
    let cwd_resolved = crate::server::canonicalize_with_missing(cwd).map_err(|_| {
        InvokeError::WorkDirOutsideAgentRoots {
            cwd: cwd.display().to_string(),
            roots: roots_text.clone(),
        }
    })?;
    let canonical_roots: Vec<PathBuf> = roots
        .iter()
        .filter_map(|root| std::fs::canonicalize(root).ok())
        .collect();
    if !canonical_roots
        .iter()
        .any(|root| cwd_resolved.starts_with(root))
    {
        return Err(InvokeError::WorkDirOutsideAgentRoots {
            cwd: cwd.display().to_string(),
            roots: roots_text,
        });
    }
    Ok(())
}

/// Собрать `ClaudeCliHints` из per-agent конфига. cwd_template (если задан) —
/// рендерится через тот же tera-контекст что и prompt.md, чтобы пути
/// типа `/sandbox/{{user_id}}/{{task_dir}}` подставляли поля из input.
fn build_cli_hints(
    cli_cfg: &ExecutionConfig,
    ctx: &tera::Context,
) -> Result<ClaudeCliHints, InvokeError> {
    let cwd = if let Some(tmpl) = &cli_cfg.cwd_template {
        let rendered = tera::Tera::one_off(tmpl, ctx, false)
            .map_err(|e| InvokeError::PromptRender(format!("cwd_template: {e}")))?;
        if rendered.trim().is_empty() {
            None
        } else {
            Some(std::path::PathBuf::from(rendered))
        }
    } else {
        None
    };

    Ok(ClaudeCliHints {
        allowed_tools: cli_cfg.allowed_tools.clone(),
        disallowed_tools: cli_cfg.disallowed_tools.clone(),
        permission_mode: cli_cfg.permission_mode.clone(),
        cwd,
        mcp_config: cli_cfg.mcp_config.clone(),
        mcp_env: HashMap::new(),
        mcp_env_missing: Vec::new(),
        max_turns: cli_cfg.max_turns,
        extra_args: cli_cfg.extra_args.clone(),
        // resume_session_id / new_session_id заполняются на уровне выше
        // в invoke() (shared-session).
        resume_session_id: None,
        new_session_id: None,
    })
}

fn collect_mcp_env(
    raw: Option<&str>,
    provider_env: &ProviderEnv,
) -> Result<(HashMap<String, String>, Vec<String>), String> {
    let Some(raw) = raw else {
        return Ok((HashMap::new(), Vec::new()));
    };
    let http = mcp_client::parse_mcp_config(raw)
        .map_err(|error| format!("не разобран как JSON: {error}"))?;
    let stdio = mcp_client::parse_stdio_servers(raw)
        .map_err(|error| format!("не разобран как JSON: {error}"))?;
    let mut texts: Vec<&str> = Vec::new();
    for server in &http {
        texts.push(&server.url);
        texts.extend(server.headers.iter().map(|(_, value)| value.as_str()));
    }
    for server in &stdio {
        texts.push(&server.command);
        texts.extend(server.args.iter().map(String::as_str));
        texts.extend(server.env.iter().map(|(_, value)| value.as_str()));
    }

    let mut values = HashMap::new();
    let mut missing = Vec::new();
    for text in texts {
        mcp_client::expand_vars(text, |name| provider_env.var(name))?;
        for (name, _) in mcp_client::referenced_vars(text) {
            match provider_env.var(&name) {
                Some(value) => {
                    values.insert(name, value);
                }
                None => missing.push(name),
            }
        }
    }
    missing.sort();
    missing.dedup();
    Ok((values, missing))
}

fn is_own_mcp_url(raw: &str, port: u16) -> bool {
    let Ok(url) = reqwest::Url::parse(raw) else {
        return false;
    };
    let host_ok = matches!(
        url.host_str()
            .map(|host| host.to_ascii_lowercase())
            .as_deref(),
        Some("127.0.0.1" | "localhost" | "[::1]" | "::1")
    );
    matches!(url.scheme(), "http" | "https")
        && host_ok
        && url.port_or_known_default() == Some(port)
        && url.path() == "/mcp"
}

/// Добавить ключ вызова только в записи, ведущие на собственный HTTP `/mcp`.
/// Алиас записи намеренно не рассматривается: доверяем адресу, а не имени.
fn inject_call_key_into_mcp_config(raw: &str, port: u16, key: &str) -> String {
    let Ok(mut value) = serde_json::from_str::<Value>(raw) else {
        return raw.to_string();
    };
    let Some(servers) = value.get_mut("mcpServers").and_then(Value::as_object_mut) else {
        return raw.to_string();
    };
    let mut changed = false;
    for config in servers.values_mut() {
        let own = config
            .get("url")
            .and_then(Value::as_str)
            .is_some_and(|url| is_own_mcp_url(url, port));
        if !own {
            continue;
        }
        let Some(config) = config.as_object_mut() else {
            continue;
        };
        let headers = config
            .entry("headers")
            .or_insert_with(|| Value::Object(Map::new()));
        if !headers.is_object() {
            *headers = Value::Object(Map::new());
        }
        let headers = headers.as_object_mut().expect("выше создан объект");
        headers.retain(|name, _| !name.eq_ignore_ascii_case(CALL_KEY_HEADER));
        headers.insert(CALL_KEY_HEADER.to_string(), Value::String(key.to_string()));
        changed = true;
    }
    if changed {
        serde_json::to_string(&value).unwrap_or_else(|_| raw.to_string())
    } else {
        raw.to_string()
    }
}

/// Нормализованный JSON входа (sorted keys) + sha256 hex.
fn hash_input(input: &Map<String, Value>) -> String {
    let sorted: BTreeMap<&String, &Value> = input.iter().collect();
    let raw = serde_json::to_string(&sorted).unwrap_or_default();
    let digest = Sha256::digest(raw.as_bytes());
    let mut hex = String::with_capacity(digest.len() * 2);
    for b in digest.iter() {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(v: Value) -> Map<String, Value> {
        v.as_object().expect("ожидался JSON-объект").clone()
    }

    #[test]
    fn call_key_is_added_only_to_own_mcp_address_regardless_of_alias() {
        let raw = json!({
            "mcpServers": {
                "agents-mcp": {
                    "url": "http://localhost:8025/mcp",
                    "headers": {"Authorization": "Bearer keep", "X-Agents-Mcp-Call": "old"}
                },
                "foreign": {
                    "url": "http://127.0.0.1:9000/mcp",
                    "headers": {"Authorization": "Bearer foreign"}
                }
            }
        })
        .to_string();

        let updated: Value =
            serde_json::from_str(&inject_call_key_into_mcp_config(&raw, 8025, "new-key")).unwrap();
        assert_eq!(
            updated["mcpServers"]["agents-mcp"]["headers"][CALL_KEY_HEADER],
            "new-key"
        );
        assert_eq!(
            updated["mcpServers"]["agents-mcp"]["headers"]["Authorization"],
            "Bearer keep"
        );
        assert!(updated["mcpServers"]["foreign"]["headers"]
            .get(CALL_KEY_HEADER)
            .is_none());
    }

    #[test]
    fn call_key_guard_removes_context() {
        let (runtime, _, dir) = test_runtime("call-key", "ok");
        let cwd = dir.join("work");
        let (key, guard) = runtime.issue_call_key(41, Some(cwd.clone()), None, Some(17), 3);
        let scope = runtime.call_scope(&key).expect("ключ зарегистрирован");
        assert_eq!(scope.call_id, 41);
        assert_eq!(scope.cwd.as_deref(), Some(cwd.as_path()));
        assert_eq!(scope.allowed_roots, None);
        assert_eq!(scope.parent_call_id, Some(17));
        assert_eq!(scope.orchestration_depth, 3);
        drop(guard);
        assert!(runtime.call_scope(&key).is_none(), "ключ снят по Drop");
        std::fs::remove_dir_all(dir).ok();
    }

    // ── Файл-итог фонового вызова (agent_run) ───────────────────────────

    #[test]
    fn result_file_appears_whole_and_keeps_envelope() {
        // Ради этого файла и затеян agent_run: читатель ждёт его появления и
        // обязан увидеть JSON целиком. Проверяем, что временного огрызка на
        // месте итога не остаётся и конверт разбирается.
        let dir =
            std::env::temp_dir().join(format!("agents-mcp-result-test-{}", uuid::Uuid::new_v4()));
        let root = dir.join("root");
        let path = root.join("вложенный").join("42-mock-agent.json");
        let old_shared_tmp = root.with_extension("json.part");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&old_shared_tmp, b"sentinel").unwrap();
        write_result_file(&path, &envelope_error(42, "mock-agent", "провайдер молчит"))
            .expect("файл-итог записан вместе с каталогами");

        let body = std::fs::read_to_string(&path).expect("итог читается");
        let v: Value = serde_json::from_str(&body).expect("итог — целый JSON");
        assert_eq!(v["status"], "error");
        assert_eq!(v["call_id"], 42);
        assert_eq!(v["agent"], "mock-agent");
        assert_eq!(v["error"], "провайдер молчит");
        assert!(
            !path.with_extension("json.part").exists(),
            "временный файл должен быть переименован, а не оставлен рядом"
        );
        assert_eq!(
            std::fs::read(&old_shared_tmp).unwrap(),
            b"sentinel",
            "старый общий .part рядом с корнем не должен быть затронут"
        );
        let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".part"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "временные файлы не должны оставаться: {leftovers:?}"
        );

        // Повторная запись поверх существующего итога не должна падать
        // (Windows: rename на занятое имя).
        write_result_file(&path, &envelope_error(42, "mock-agent", "вторая попытка"))
            .expect("перезапись итога проходит");
        let body = std::fs::read_to_string(&path).expect("итог читается");
        assert!(body.contains("вторая попытка"));

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── Реестр живых вызовов и отмена ────────────────────────────────────

    /// Задача-пустышка: висит, пока её не оборвут (модель долгого прогона).
    fn spawn_sleeping() -> tokio::task::JoinHandle<()> {
        tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(600)).await;
        })
    }

    #[tokio::test]
    async fn registry_shows_live_call_and_hides_removed() {
        let reg = CallRegistry::default();
        reg.spawn_registered(7, "mock-agent".to_string(), None, None, spawn_sleeping);

        let live = reg.live_calls();
        assert_eq!(live.len(), 1, "зарегистрированный вызов виден в реестре");
        assert_eq!(live[0].call_id, 7);
        assert_eq!(live[0].agent, "mock-agent");
        assert_eq!(live[0].elapsed_sec, 0);

        // Так задача снимает себя сама своим последним действием.
        reg.remove(7);
        assert!(reg.live_calls().is_empty(), "снятый вызов из списка уходит");
    }

    #[tokio::test]
    async fn taking_unknown_call_id_reports_no_live_call() {
        let reg = CallRegistry::default();
        // Неизвестный call_id — это отсутствие записи, а не паника: тем же
        // путём cancel отвечает «такого живого вызова нет».
        assert!(reg.take(10126).is_none());
        assert!(reg.live_calls().is_empty());
    }

    #[tokio::test]
    async fn take_and_abort_stops_the_task() {
        let reg = CallRegistry::default();
        reg.spawn_registered(11, "mock-agent".to_string(), None, None, spawn_sleeping);

        let entry = reg.take(11).expect("вызов числится живым");
        entry.handle.abort();
        let res = entry.handle.await;
        assert!(
            res.expect_err("оборванная задача завершается ошибкой")
                .is_cancelled(),
            "задача обязана быть отменена, а не дождаться сна"
        );
        assert!(reg.live_calls().is_empty());
    }

    // ── Чистые функции ──────────────────────────────────────────────────

    /// План с двумя блоками кода внутри строкового поля — тот самый ответ, на
    /// котором разбор по первому закрывающему обрамлению терял 9/10 текста.
    fn plan_with_two_code_blocks() -> String {
        let plan = "Шаг 1.\n```rust\nfn a() {}\n```\nШаг 2.\n```rust\nfn b() {}\n```\nГотово.";
        serde_json::json!({"plan": plan, "questions": [], "used_transcript": false}).to_string()
    }

    #[test]
    fn parse_json_fenced_answer_with_inner_code_blocks() {
        let body = plan_with_two_code_blocks();
        let answer = format!("```json\n{body}\n```");
        let v = parse_json_tolerant(&answer).expect("ответ обязан разобраться целиком");
        assert!(v["plan"]
            .as_str()
            .expect("plan — строка")
            .contains("fn b()"));
    }

    #[test]
    fn parse_json_fenced_without_language_tag() {
        let body = plan_with_two_code_blocks();
        let answer = format!("```\n{body}\n```");
        let v = parse_json_tolerant(&answer).expect("рамка без пометки языка тоже снимается");
        assert!(v["plan"]
            .as_str()
            .expect("plan — строка")
            .contains("fn b()"));
    }

    #[test]
    fn parse_json_after_leading_text() {
        // Модель пояснила ответ словами и только потом дала рамку с JSON.
        let v = parse_json_tolerant("Проверил три места.\n\n```json\n{\"a\":1}\n```")
            .expect("пояснение перед рамкой не мешает");
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn parse_json_without_fence() {
        let v = parse_json_tolerant("  {\"a\":1}  ").expect("ответ без рамки разбирается как есть");
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn parse_json_without_fence_but_with_text() {
        // Рамки нет, текст вокруг есть — берём кусок по скобкам.
        let v = parse_json_tolerant("Вот итог: {\"a\":1} — готово.").expect("кусок по скобкам");
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn parse_json_open_fence_without_close() {
        let v = parse_json_tolerant("```json\n{\"a\":1}").expect("незакрытая рамка не мешает");
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn parse_json_broken_json_is_refused() {
        // Скобка не закрыта — честный отказ, а не обрывок.
        assert!(parse_json_tolerant("```json\n{\"a\": 1\n```").is_err());
        assert!(parse_json_tolerant("  просто текст  ").is_err());
    }

    #[test]
    fn write_raw_response_saves_answer_as_is() {
        let dir =
            std::env::temp_dir().join(format!("agents-mcp-raw-answer-{}", uuid::Uuid::new_v4()));
        let content = "```json\n{\"a\": 1\n```";
        let path =
            write_raw_response(&dir, 42, "code-planner", content).expect("сырой ответ сохранён");
        assert_eq!(
            std::fs::read_to_string(&path).expect("файл читается"),
            content
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hash_input_order_independent() {
        let a = obj(json!({"x": 1, "y": 2}));
        let b = obj(json!({"y": 2, "x": 1}));
        assert_eq!(hash_input(&a), hash_input(&b));
    }

    #[test]
    fn hash_input_differs_on_value() {
        assert_ne!(
            hash_input(&obj(json!({"x": 1}))),
            hash_input(&obj(json!({"x": 2})))
        );
    }

    #[test]
    fn hash_input_is_hex_sha256() {
        let h = hash_input(&obj(json!({"x": 1})));
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn cache_response_marks_cached_and_uses_current_node() {
        // Восстановление из кеша: cached=true, а agent/variant/parent/depth
        // берутся из текущего запроса, не из закешированной метадаты.
        let meta = InvokeMetadata {
            agent: "old".into(),
            variant: "old".into(),
            model_used: "m".into(),
            provider: "p".into(),
            tokens_in: 1,
            tokens_out: 2,
            cost_usd: Some(0.5),
            latency_ms: 999,
            cached: false,
            call_id: 7,
            parent_call_id: None,
            orchestration_depth: 0,
        };
        let entry = cache::CachedEntry {
            output_json: "{\"ok\":true}".into(),
            metadata_json: serde_json::to_string(&meta).unwrap(),
        };
        let resp =
            build_response_from_cache("agent-new", "v2", &entry, Instant::now(), Some(42), 3)
                .expect("должен восстановиться");
        assert!(resp.metadata.cached);
        assert_eq!(resp.metadata.agent, "agent-new");
        assert_eq!(resp.metadata.variant, "v2");
        assert_eq!(resp.metadata.parent_call_id, Some(42));
        assert_eq!(resp.metadata.orchestration_depth, 3);
        assert_eq!(resp.result, json!({"ok": true}));
    }

    #[test]
    fn cache_response_invalid_json_returns_none() {
        let entry = cache::CachedEntry {
            output_json: "не json".into(),
            metadata_json: "тоже не json".into(),
        };
        assert!(
            build_response_from_cache("a", "default", &entry, Instant::now(), None, 0).is_none()
        );
    }

    // ── Лимиты и валидация схемы (фича б) ───────────────────────────────

    #[test]
    fn estimate_tokens_roughly_quarter_of_chars() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens(&"x".repeat(400)), 100);
    }

    fn art(
        kind: &str,
        key: &str,
        content: Option<&str>,
        summary: Option<&str>,
    ) -> crate::store::Artifact {
        crate::store::Artifact {
            id: 0,
            kind: kind.into(),
            key: key.into(),
            content: content.map(|s| s.to_string()),
            summary: summary.map(|s| s.to_string()),
            producer_agent: None,
            producer_call_id: None,
            depends_on: vec![],
            status: "ready".into(),
        }
    }

    #[test]
    fn task_context_empty_for_no_artifacts() {
        assert_eq!(format_task_context(&[]), "");
    }

    #[test]
    fn task_context_compact_and_truncates_long_content() {
        let arts = vec![
            art("metadata", "meta.X", Some("короткий"), Some("про X")),
            art("bsl_module", "ObjectModule", Some(&"я".repeat(1000)), None),
        ];
        let s = format_task_context(&arts);
        assert!(s.contains("[metadata] meta.X — про X"), "s={s}");
        assert!(
            s.contains("короткий"),
            "короткий content должен влиться целиком"
        );
        assert!(s.contains("[bsl_module] ObjectModule"), "s={s}");
        assert!(
            s.contains("прочитать через artifact_read"),
            "длинный content должен быть заменён намёком"
        );
        assert!(
            !s.contains(&"я".repeat(1000)),
            "длинный content не должен вливаться целиком"
        );
    }

    #[test]
    fn task_context_limits_total_size() {
        // Длинные kind/key/summary: заголовок обрезан, срез в пределе.
        let long = vec![art(
            &"т".repeat(2500),
            &"к".repeat(2500),
            Some("короткий content"),
            Some(&"с".repeat(2500)),
        )];
        let s = format_task_context(&long);
        assert!(s.chars().count() <= 6000);
        let head = s.lines().nth(1).expect("заголовок");
        assert_eq!(head.chars().count(), 300, "заголовок ограничен");
        assert!(s.contains("короткий content"), "content помещается в запас");

        // 30 артефактов по 600 символов content: заголовки видны у всех, content — пока есть запас.
        let arts: Vec<_> = (0..30)
            .map(|i| {
                art(
                    "metadata",
                    &format!("meta.{i}"),
                    Some(&"я".repeat(600)),
                    Some("сводка"),
                )
            })
            .collect();
        let s = format_task_context(&arts);
        assert!(s.chars().count() <= 6000, "len={}", s.chars().count());
        assert!(
            s.contains("[metadata] meta.29 — сводка"),
            "заголовки не отбрасываются"
        );
        assert!(
            s.contains(&"я".repeat(600)),
            "первые content попадают в срез"
        );

        // Заголовков больше, чем помещается: строгий предел и строка «ещё N».
        let many: Vec<_> = (0..2000)
            .map(|i| art("metadata", &format!("meta.{i}"), None, Some("сводка")))
            .collect();
        let s = format_task_context(&many);
        assert!(s.chars().count() <= 6000, "len={}", s.chars().count());
        assert!(s.contains("артефактов не показаны"));
    }

    fn bsl_reviewer_schema() -> Value {
        json!({
            "type": "object",
            "required": ["verdict", "critical", "minor", "summary"],
            "properties": {
                "verdict": {"type": "string"},
                "critical": {"type": "array"},
                "minor": {"type": "array"},
                "summary": {"type": "string"},
                "line": {"type": ["integer", "null"]}
            }
        })
    }

    #[test]
    fn schema_valid_response_ok() {
        let v = json!({"verdict": "ok", "critical": [], "minor": [], "summary": "норм"});
        assert!(validate_against_schema(&v, &bsl_reviewer_schema()).is_ok());
    }

    #[test]
    fn schema_missing_required_field() {
        let v = json!({"verdict": "ok", "critical": [], "minor": []}); // нет summary
        let err = validate_against_schema(&v, &bsl_reviewer_schema()).unwrap_err();
        assert!(err.contains("summary"), "err={err}");
    }

    #[test]
    fn schema_wrong_top_level_type() {
        let v = json!(["не", "объект"]);
        assert!(validate_against_schema(&v, &bsl_reviewer_schema()).is_err());
    }

    #[test]
    fn schema_wrong_property_type() {
        // critical должен быть array, прислали string.
        let v = json!({"verdict": "ok", "critical": "не массив", "minor": [], "summary": "s"});
        let err = validate_against_schema(&v, &bsl_reviewer_schema()).unwrap_err();
        assert!(err.contains("critical"), "err={err}");
    }

    #[test]
    fn schema_nullable_field_not_flagged() {
        // line объявлен как ["integer","null"] — массивный тип пропускаем:
        // ни null, ни integer не должны вызывать ошибку.
        let with_null =
            json!({"verdict": "ok", "critical": [], "minor": [], "summary": "s", "line": null});
        let with_int =
            json!({"verdict": "ok", "critical": [], "minor": [], "summary": "s", "line": 5});
        assert!(validate_against_schema(&with_null, &bsl_reviewer_schema()).is_ok());
        assert!(validate_against_schema(&with_int, &bsl_reviewer_schema()).is_ok());
    }

    // ── Session-стратегия (БД-хелперы shared-session) ───────────────────

    /// Живой round-trip против PG. Игнорируется по умолчанию (нужна база
    /// PostgreSQL и AGENTS_MCP_TEST_PG_DSN). Покрывает: insert_call_stub →
    /// session_id NULL, ранний set_call_session_id виден get_call_session_id,
    /// неизвестный call_id → None.
    #[tokio::test]
    #[ignore]
    async fn session_helpers_pg_round_trip() {
        let dsn = std::env::var("AGENTS_MCP_TEST_PG_DSN").expect("AGENTS_MCP_TEST_PG_DSN не задан");
        let pg: Arc<dyn Store> =
            Arc::new(crate::store::PgStore::connect(&dsn, 2).expect("connect"));

        let id = pg
            .insert_call_stub(
                "orch",
                "default",
                "h",
                "sonnet",
                "claude-cli",
                None,
                None,
                "test:1",
            )
            .await
            .unwrap();
        // До раннего UPDATE session_id == NULL.
        assert_eq!(pg.get_call_session_id(id).await.unwrap(), None);

        // Ранний UPDATE виден следующему чтению.
        pg.set_call_session_id(id, "sess-ABC").await.unwrap();
        assert_eq!(
            pg.get_call_session_id(id).await.unwrap().as_deref(),
            Some("sess-ABC")
        );

        // Неизвестный call_id → None.
        assert_eq!(pg.get_call_session_id(99_999_999).await.unwrap(), None);
    }

    // ── Помощники для тестов с настоящим рантаймом ───────────────────────

    /// Живой рантайм над SqliteStore :memory: с временным каталогом агентов:
    /// один агент с config.toml (провайдер mock) и prompt.md заданного текста.
    /// Возвращает рантайм, хранилище (для проверок) и каталог (для уборки).
    fn test_runtime(agent_name: &str, prompt: &str) -> (Runtime, Arc<dyn Store>, PathBuf) {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("время")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("agents-mcp-runtime-{agent_name}-{nanos}"));
        let agents_dir = dir.join("agents");
        let agent_dir = agents_dir.join(agent_name);
        std::fs::create_dir_all(&agent_dir).expect("каталог агента");
        std::fs::write(
            agent_dir.join("config.toml"),
            format!(
                "name = \"{agent_name}\"\n\n[model]\nprovider = \"mock\"\nname = \"mock-model-v0\"\n"
            ),
        )
        .expect("config.toml агента");
        std::fs::write(agent_dir.join("prompt.md"), prompt).expect("prompt.md агента");

        let store: Arc<dyn Store> = Arc::new(
            crate::store::SqliteStore::open(Path::new(":memory:")).expect("хранилище журнала"),
        );
        let registry = Arc::new(Registry::load(agents_dir).expect("реестр агентов"));
        let mut providers: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        providers.insert(
            "mock".to_string(),
            Arc::new(crate::providers::mock::MockProvider::new()),
        );
        let runtime = Runtime::new(
            store.clone(),
            registry,
            providers,
            crate::skills::SkillsClient::new(None),
            Arc::new(std::sync::RwLock::new(ModelOverride::default())),
            dir.join("runs"),
            "test:1".to_string(),
            120,
        );
        (runtime, store, dir)
    }

    /// Запрос синхронного вызова (wait_sec=None) с необязательным родителем.
    fn invoke_req(agent: &str, parent_call_id: Option<i64>) -> InvokeRequest {
        InvokeRequest {
            agent: agent.to_string(),
            input: Map::new(),
            variant: None,
            parent_call_id,
            orchestration_depth: 0,
            wait_sec: None,
            task_id: None,
        }
    }

    struct FixedProvider {
        finish_reason: &'static str,
        panic: bool,
    }

    struct McpEnvInspectProvider {
        seen: Arc<std::sync::Mutex<Option<HashMap<String, String>>>>,
    }

    #[async_trait::async_trait]
    impl LlmProvider for McpEnvInspectProvider {
        async fn complete(
            &self,
            req: LlmRequest,
        ) -> Result<crate::providers::LlmResponse, LlmError> {
            *self.seen.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(req.cli_hints.expect("подсказки исполнения").mcp_env);
            Ok(crate::providers::LlmResponse {
                content: "ok".into(),
                tokens_in: 1,
                tokens_out: 1,
                cost_usd: Some(0.0),
                finish_reason: "stop".into(),
                reasoning: None,
                session_id: None,
                raw_input_tokens: 1,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                transcript: vec![],
            })
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for FixedProvider {
        async fn complete(
            &self,
            req: LlmRequest,
        ) -> Result<crate::providers::LlmResponse, LlmError> {
            assert!(!self.panic, "тестовая паника провайдера");
            Ok(crate::providers::LlmResponse {
                content: req.system_prompt,
                tokens_in: 1,
                tokens_out: 1,
                cost_usd: Some(0.0),
                finish_reason: self.finish_reason.into(),
                reasoning: None,
                session_id: None,
                raw_input_tokens: 1,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                transcript: vec![],
            })
        }
    }

    struct PendingProvider;

    #[async_trait::async_trait]
    impl LlmProvider for PendingProvider {
        async fn complete(
            &self,
            _req: LlmRequest,
        ) -> Result<crate::providers::LlmResponse, LlmError> {
            std::future::pending::<Result<crate::providers::LlmResponse, LlmError>>().await
        }
    }

    struct CallKeyInspectProvider {
        scopes: CallScopes,
        seen_key: Arc<std::sync::Mutex<Option<String>>>,
    }

    #[async_trait::async_trait]
    impl LlmProvider for CallKeyInspectProvider {
        async fn complete(
            &self,
            req: LlmRequest,
        ) -> Result<crate::providers::LlmResponse, LlmError> {
            let raw = req
                .cli_hints
                .as_ref()
                .and_then(|hints| hints.mcp_config.as_deref())
                .expect("прямой провайдер получил mcp_config");
            let config: Value = serde_json::from_str(raw).unwrap();
            let key = config["mcpServers"]["not-agents"]["headers"][CALL_KEY_HEADER]
                .as_str()
                .expect("ключ добавлен независимо от алиаса")
                .to_string();
            assert!(
                self.scopes
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .contains_key(&key),
                "во время provider.complete ключ должен быть действителен"
            );
            *self.seen_key.lock().unwrap_or_else(|e| e.into_inner()) = Some(key);
            Ok(crate::providers::LlmResponse {
                content: "ok".to_string(),
                tokens_in: 1,
                tokens_out: 1,
                cost_usd: Some(0.0),
                finish_reason: "stop".into(),
                reasoning: None,
                session_id: None,
                raw_input_tokens: 1,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                transcript: vec![],
            })
        }
    }

    #[tokio::test]
    async fn direct_provider_gets_alias_independent_key_and_completion_revokes_it() {
        let dir = std::env::temp_dir().join(format!(
            "agents-mcp-call-key-provider-{}",
            uuid::Uuid::new_v4()
        ));
        let agent_dir = dir.join("agents/direct-key");
        let cwd = dir.join("work");
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::create_dir(&cwd).unwrap();
        let cwd_text = cwd.to_string_lossy().replace('\\', "/");
        std::fs::write(
            agent_dir.join("config.toml"),
            format!(
                "name = 'direct-key'\n[model]\nprovider = 'direct-fixture'\nname = 'fixture'\n[execution]\ncwd_template = '{cwd_text}'\nmcp_config = '''{{\"mcpServers\":{{\"not-agents\":{{\"url\":\"http://127.0.0.1:8025/mcp\"}}}}}}'''\n"
            ),
        )
        .unwrap();
        std::fs::write(agent_dir.join("prompt.md"), "ok").unwrap();
        let store: Arc<dyn Store> =
            Arc::new(crate::store::SqliteStore::open(Path::new(":memory:")).unwrap());
        let registry = Arc::new(Registry::load(dir.join("agents")).unwrap());
        let runtime = Runtime::new(
            store,
            registry,
            HashMap::new(),
            crate::skills::SkillsClient::new(None),
            Arc::new(std::sync::RwLock::new(ModelOverride::default())),
            dir.join("runs"),
            "test:key".into(),
            120,
        );
        runtime.set_own_mcp_port(8025);
        let seen_key = Arc::new(std::sync::Mutex::new(None));
        runtime.set_providers(HashMap::from([(
            "direct-fixture".to_string(),
            Arc::new(CallKeyInspectProvider {
                scopes: Arc::clone(&runtime.call_scopes),
                seen_key: Arc::clone(&seen_key),
            }) as Arc<dyn LlmProvider>,
        )]));

        runtime
            .invoke(invoke_req("direct-key", None))
            .await
            .unwrap();
        let key = seen_key
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .expect("провайдер увидел ключ");
        assert!(
            runtime.call_scope(&key).is_none(),
            "после завершения ключ снят"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    struct CountingProvider {
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl LlmProvider for CountingProvider {
        async fn complete(
            &self,
            _req: LlmRequest,
        ) -> Result<crate::providers::LlmResponse, LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(crate::providers::LlmResponse {
                content: "ok".to_string(),
                tokens_in: 1,
                tokens_out: 1,
                cost_usd: Some(0.0),
                finish_reason: "stop".into(),
                reasoning: None,
                session_id: None,
                raw_input_tokens: 1,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                transcript: vec![],
            })
        }
    }

    /// Отдельный каталог под cwd/корни теста allowed_roots: в имени — метка
    /// и uuid, чтобы параллельные тесты не пересекались.
    fn temp_call_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "agents-mcp-agent-roots-{tag}-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("временный каталог");
        dir
    }

    /// Секция [execution] для тестов allowed_roots: пути в TOML — через
    /// прямые слэши.
    fn agent_roots_config(cwd: Option<&Path>, roots: &[&Path]) -> String {
        let mut секция = String::from("[execution]\n");
        if let Some(cwd) = cwd {
            секция.push_str(&format!(
                "cwd_template = '{}'\n",
                cwd.to_string_lossy().replace('\\', "/")
            ));
        }
        let roots_list = roots
            .iter()
            .map(|root| format!("'{}'", root.to_string_lossy().replace('\\', "/")))
            .collect::<Vec<_>>()
            .join(", ");
        секция.push_str(&format!("allowed_roots = [{roots_list}]\n"));
        секция
    }

    fn counting_fixed_runtime(
        extra_config: &str,
        calls: &Arc<std::sync::atomic::AtomicUsize>,
    ) -> Runtime {
        let (runtime, _store, _base) = fixed_runtime(extra_config, "ok", "stop", false);
        runtime.set_providers(HashMap::from([(
            "fixed".to_string(),
            Arc::new(CountingProvider {
                calls: Arc::clone(calls),
            }) as Arc<dyn LlmProvider>,
        )]));
        runtime
    }

    #[tokio::test]
    async fn agent_roots_reject_cwd_outside_agent_roots_and_skip_provider() {
        let dir = temp_call_dir("outside");
        let root = dir.join("root");
        let cwd = dir.join("elsewhere").join("work");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let runtime = counting_fixed_runtime(&agent_roots_config(Some(&cwd), &[&root]), &calls);

        let err = runtime
            .invoke(invoke_req("test", None))
            .await
            .err()
            .expect("вызов с cwd вне корней агента обязан быть отклонён");
        assert!(
            matches!(&err, InvokeError::WorkDirOutsideAgentRoots { .. }),
            "получено: {err}"
        );
        let InvokeError::WorkDirOutsideAgentRoots {
            cwd: err_cwd,
            roots: err_roots,
        } = err
        else {
            unreachable!("проверено matches! выше")
        };
        assert!(err_cwd.contains("elsewhere"), "cwd = {err_cwd}");
        assert_eq!(err_roots, vec![root.to_string_lossy().replace('\\', "/")]);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "провайдер не должен вызываться"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test]
    async fn agent_roots_allow_cwd_inside_agent_roots() {
        let dir = temp_call_dir("inside");
        let root = dir.join("root");
        let cwd = root.join("work");
        std::fs::create_dir_all(&cwd).unwrap();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let runtime = counting_fixed_runtime(&agent_roots_config(Some(&cwd), &[&root]), &calls);

        runtime
            .invoke(invoke_req("test", None))
            .await
            .expect("вызов с cwd внутри корней агента проходит");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test]
    async fn agent_roots_without_cwd_template_refuses_call() {
        let dir = temp_call_dir("no-cwd");
        let root = dir.join("root");
        std::fs::create_dir_all(&root).unwrap();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let runtime = counting_fixed_runtime(&agent_roots_config(None, &[&root]), &calls);

        let err = runtime
            .invoke(invoke_req("test", None))
            .await
            .err()
            .expect("вызов без cwd_template при заданном allowed_roots обязан быть отклонён");
        assert!(
            matches!(&err, InvokeError::AgentRootsWorkDirMissing),
            "получено: {err}"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "провайдер не должен вызываться"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    fn cache_runtime(
        extra_config: &str,
        prompt: &str,
    ) -> (Runtime, Arc<dyn Store>, Arc<Registry>, PathBuf) {
        let dir = std::env::temp_dir().join(format!("agents-mcp-cache-{}", uuid::Uuid::new_v4()));
        let agent_dir = dir.join("agents/test");
        std::fs::create_dir_all(&agent_dir).expect("каталог агента");
        std::fs::write(
            agent_dir.join("config.toml"),
            format!(
                "name = \"test\"\n\n[model]\nprovider = \"fixed\"\nname = \"configured\"\n\n{extra_config}\n"
            ),
        )
        .expect("config.toml агента");
        std::fs::write(agent_dir.join("prompt.md"), prompt).expect("prompt.md агента");

        let store: Arc<dyn Store> = Arc::new(
            crate::store::SqliteStore::open(Path::new(":memory:")).expect("хранилище журнала"),
        );
        let registry = Arc::new(Registry::load(dir.join("agents")).expect("реестр агентов"));
        let providers = HashMap::from([(
            "fixed".to_string(),
            Arc::new(FixedProvider {
                finish_reason: "stop",
                panic: false,
            }) as Arc<dyn LlmProvider>,
        )]);
        let runtime = Runtime::new(
            store.clone(),
            registry.clone(),
            providers,
            crate::skills::SkillsClient::new(None),
            Arc::new(std::sync::RwLock::new(ModelOverride::default())),
            dir.join("runs"),
            "test:cache".into(),
            120,
        );
        (runtime, store, registry, dir)
    }

    #[tokio::test]
    async fn missing_mcp_variable_is_rejected_before_provider() {
        let variable = "AGENTS_MCP_TEST_ENV_MISSING_9F2A";
        let config = format!(
            "[execution]\nmcp_config = '''{{\"mcpServers\":{{\"fixture\":{{\"url\":\"https://example/${{{variable}}}\"}}}}}}'''"
        );
        let (runtime, _store, _registry, dir) = cache_runtime(&config, "prompt");
        let calls = Arc::new(AtomicUsize::new(0));
        runtime.set_providers(HashMap::from([(
            "fixed".to_string(),
            Arc::new(CountingProvider {
                calls: Arc::clone(&calls),
            }) as Arc<dyn LlmProvider>,
        )]));

        let error = runtime
            .invoke(cache_req())
            .await
            .err()
            .expect("вызов без переменной обязан быть отклонён");

        assert!(matches!(&error, InvokeError::McpConfig(_)));
        assert!(error.to_string().contains(variable));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn mcp_environment_uses_dotenv_and_ignores_names_and_cwd() {
        let token = "AGENTS_MCP_TEST_DOTENV_TOKEN_7C1B";
        let config = format!(
            "[execution]\nmcp_config = '''{{\"mcpServers\":{{\"${{ALIAS}}\":{{\"url\":\"https://example/mcp\",\"headers\":{{\"${{HEADER}}\":\"Bearer ${{{token}}}\"}}}},\"stdio\":{{\"command\":\"program\",\"cwd\":\"${{CWD}}\"}}}}}}'''"
        );
        let (runtime, _store, _registry, dir) = cache_runtime(&config, "prompt");
        runtime.set_provider_env(ProviderEnv::capture(HashMap::from([(
            token.to_string(),
            "dotenv-secret".to_string(),
        )])));
        let seen = Arc::new(std::sync::Mutex::new(None));
        runtime.set_providers(HashMap::from([(
            "fixed".to_string(),
            Arc::new(McpEnvInspectProvider {
                seen: Arc::clone(&seen),
            }) as Arc<dyn LlmProvider>,
        )]));

        runtime.invoke(cache_req()).await.expect("вызов проходит");

        let values = seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .expect("провайдер получил окружение");
        assert_eq!(values.get(token).map(String::as_str), Some("dotenv-secret"));
        assert!(!values.contains_key("ALIAS"));
        assert!(!values.contains_key("HEADER"));
        assert!(!values.contains_key("CWD"));
        let _ = std::fs::remove_dir_all(dir);
    }

    fn fixed_runtime(
        extra_config: &str,
        prompt: &str,
        finish_reason: &'static str,
        panic: bool,
    ) -> (Runtime, Arc<crate::store::SqliteStore>, PathBuf) {
        let dir =
            std::env::temp_dir().join(format!("agents-mcp-result-status-{}", uuid::Uuid::new_v4()));
        let agent_dir = dir.join("agents/test");
        std::fs::create_dir_all(&agent_dir).expect("каталог агента");
        std::fs::write(
            agent_dir.join("config.toml"),
            format!(
                "name = \"test\"\n\n[model]\nprovider = \"fixed\"\nname = \"fixed-model\"\n\n{extra_config}\n"
            ),
        )
        .expect("config.toml агента");
        std::fs::write(agent_dir.join("prompt.md"), prompt).expect("prompt.md агента");

        let store = Arc::new(
            crate::store::SqliteStore::open(Path::new(":memory:")).expect("хранилище журнала"),
        );
        let registry = Arc::new(Registry::load(dir.join("agents")).expect("реестр агентов"));
        let providers = HashMap::from([(
            "fixed".to_string(),
            Arc::new(FixedProvider {
                finish_reason,
                panic,
            }) as Arc<dyn LlmProvider>,
        )]);
        let runtime = Runtime::new(
            store.clone(),
            registry,
            providers,
            crate::skills::SkillsClient::new(None),
            Arc::new(std::sync::RwLock::new(ModelOverride::default())),
            dir.join("runs"),
            "test:status".into(),
            120,
        );
        (runtime, store, dir)
    }

    async fn wait_for_result_file(path: &Path) -> Value {
        for _ in 0..200 {
            if let Ok(body) = std::fs::read_to_string(path) {
                return serde_json::from_str(&body).expect("файл-итог содержит JSON");
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("файл-итог {} не появился", path.display());
    }

    #[tokio::test]
    async fn invalid_json_and_length_are_incomplete() {
        let (runtime, store, dir) =
            fixed_runtime("[response]\nformat = \"json\"", "{\"cut", "length", false);
        let outcome = runtime.invoke(cache_req()).await.expect("вызов выполнен");
        let response = match outcome {
            InvokeOutcome::Incomplete { response, error } => {
                assert!(error.contains("finish_reason=length"));
                assert!(error.contains("не удалось разобрать как JSON"));
                response
            }
            _ => panic!("ожидался incomplete"),
        };
        assert!(response.result.is_string());
        let row = store
            .get_call_row(response.metadata.call_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(row.status, "incomplete");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Рантайм с агентом format=json: есть schema.json, требующая поле
    /// `summary`, и переключатель `schema_strict`. Модель (FixedProvider)
    /// возвращает объект без этого поля — сырой ответ равен prompt.md.
    fn schema_runtime(strict: bool) -> (Runtime, Arc<crate::store::SqliteStore>, PathBuf) {
        let dir =
            std::env::temp_dir().join(format!("agents-mcp-schema-strict-{}", uuid::Uuid::new_v4()));
        let agent_dir = dir.join("agents/test");
        std::fs::create_dir_all(&agent_dir).expect("каталог агента");
        std::fs::write(
            agent_dir.join("config.toml"),
            format!(
                "name = \"test\"\n\n[model]\nprovider = \"fixed\"\nname = \"fixed-model\"\n\n[response]\nformat = \"json\"\nschema_file = \"schema.json\"\nschema_strict = {strict}\n"
            ),
        )
        .expect("config.toml агента");
        std::fs::write(
            agent_dir.join("schema.json"),
            r#"{"type":"object","required":["summary"],"properties":{"summary":{"type":"string"}}}"#,
        )
        .expect("schema.json агента");
        std::fs::write(agent_dir.join("prompt.md"), r#"{"verdict":"ok"}"#)
            .expect("prompt.md агента");

        let store = Arc::new(
            crate::store::SqliteStore::open(Path::new(":memory:")).expect("хранилище журнала"),
        );
        let registry = Arc::new(Registry::load(dir.join("agents")).expect("реестр агентов"));
        let providers = HashMap::from([(
            "fixed".to_string(),
            Arc::new(FixedProvider {
                finish_reason: "stop",
                panic: false,
            }) as Arc<dyn LlmProvider>,
        )]);
        let runtime = Runtime::new(
            store.clone(),
            registry,
            providers,
            crate::skills::SkillsClient::new(None),
            Arc::new(std::sync::RwLock::new(ModelOverride::default())),
            dir.join("runs"),
            "test:status".into(),
            120,
        );
        (runtime, store, dir)
    }

    /// Файлы сырого ответа (`-raw.txt`) в каталоге прогонов вызова.
    fn raw_response_files(runs_dir: &Path) -> Vec<PathBuf> {
        let Ok(entries) = std::fs::read_dir(runs_dir) else {
            return Vec::new();
        };
        entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with("-raw.txt"))
            })
            .collect()
    }

    #[tokio::test]
    async fn strict_schema_mismatch_makes_call_incomplete_and_saves_raw() {
        let (runtime, store, dir) = schema_runtime(true);
        let outcome = runtime.invoke(cache_req()).await.expect("вызов выполнен");
        let (response, error) = match outcome {
            InvokeOutcome::Incomplete { response, error } => (response, error),
            _ => panic!("ожидался incomplete"),
        };
        assert!(
            error.contains("ответ не соответствует schema_file"),
            "error={error}"
        );
        assert!(
            error.contains("summary"),
            "в причине нет имени недостающего поля: {error}"
        );
        let row = store
            .get_call_row(response.metadata.call_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(row.status, "incomplete");
        let raw = raw_response_files(&dir.join("runs"));
        assert_eq!(raw.len(), 1, "ожидался файл -raw.txt: {raw:?}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn soft_schema_mismatch_keeps_call_done_without_raw_file() {
        let (runtime, store, dir) = schema_runtime(false);
        let outcome = runtime.invoke(cache_req()).await.expect("вызов выполнен");
        let response = sync_response(outcome);
        let row = store
            .get_call_row(response.metadata.call_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(row.status, "done");
        assert!(
            raw_response_files(&dir.join("runs")).is_empty(),
            "при выключенном переключателе сырой ответ не сохраняется"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn persistence_failure_keeps_model_response_in_result_file() {
        let (runtime, store, dir) = fixed_runtime("", "valuable answer", "stop", false);
        store.fail_next_update_call();
        let started = runtime
            .start_background(cache_req(), None)
            .await
            .expect("фоновый вызов запущен");
        let (call_id, path) = match started {
            StartedJob::Running {
                call_id,
                result_path,
            } => (call_id, result_path),
            StartedJob::Done { .. } => panic!("кеш не включён"),
        };
        let envelope = wait_for_result_file(&path).await;
        assert_eq!(envelope["status"], "persistence_failed");
        assert_eq!(envelope["result"], "valuable answer");
        let row = store.get_call_row(call_id).await.unwrap().unwrap();
        assert_eq!(row.status, "persistence_failed");
        assert_eq!(row.output_json.as_deref(), Some("\"valuable answer\""));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn background_panic_finishes_call_and_releases_registry() {
        let (runtime, store, dir) = fixed_runtime("", "unused", "stop", true);
        let started = runtime
            .start_background(cache_req(), None)
            .await
            .expect("фоновый вызов запущен");
        let (call_id, path) = match started {
            StartedJob::Running {
                call_id,
                result_path,
            } => (call_id, result_path),
            StartedJob::Done { .. } => panic!("кеш не включён"),
        };
        let envelope = wait_for_result_file(&path).await;
        assert_eq!(envelope["status"], "error");
        assert!(envelope["error"].as_str().unwrap().contains("паник"));
        let row = store.get_call_row(call_id).await.unwrap().unwrap();
        assert_eq!(row.status, "error");
        assert!(runtime.live_calls().is_empty());
        runtime.begin_drain();
        assert!(runtime.drain_status().ready());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn storage_read_failure_is_not_reported_as_running() {
        let (runtime, store, dir) = fixed_runtime("", "unused", "stop", false);
        let call_id = store
            .insert_call_stub(
                "test",
                "default",
                "hash",
                "fixed-model",
                "fixed",
                None,
                None,
                "test:status",
            )
            .await
            .unwrap();
        store.set_get_call_row_failure(true);

        match runtime.wait(call_id, 0).await {
            InvokeOutcome::PersistenceFailed { error, .. } => {
                assert!(error.contains("хранилище недоступно"));
            }
            _ => panic!("wait_agent не должен маскировать отказ хранилища как running"),
        }
        let (checked, created_at) = runtime.check(call_id).await;
        assert!(created_at.is_some());
        assert!(matches!(checked, InvokeOutcome::PersistenceFailed { .. }));
        let _ = std::fs::remove_dir_all(dir);
    }

    fn cache_req() -> InvokeRequest {
        invoke_req("test", None)
    }

    fn sync_response(outcome: InvokeOutcome) -> InvokeResponse {
        match outcome {
            InvokeOutcome::Sync(response) => response,
            _ => panic!("ожидался синхронный ответ"),
        }
    }

    async fn wait_for_cache(store: &dyn Store, key: &str) {
        for _ in 0..100 {
            if store
                .cache_lookup(key)
                .await
                .expect("чтение кеша")
                .is_some()
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        panic!("запись не появилась в кеше");
    }

    #[tokio::test]
    async fn cache_separates_forced_model() {
        let (runtime, store, _registry, dir) = cache_runtime("[cache]\nenabled = true", "answer");
        let request = cache_req();
        let first = sync_response(runtime.invoke(request.clone()).await.expect("первый вызов"));
        assert!(!first.metadata.cached);
        let first_key = cache::compute_key(cache::CacheKeyParts {
            agent_name: "test",
            variant: "default",
            provider_name: "fixed",
            model_name: "configured",
            prompt: "answer",
            task_context: "",
            input: &request.input,
            key_fields: &[],
            task_id: None,
        });
        wait_for_cache(store.as_ref(), &first_key).await;

        *runtime.force_override.write().expect("override") = ModelOverride {
            provider: Some("fixed".into()),
            model: Some("actual".into()),
        };
        let second = sync_response(runtime.invoke(request).await.expect("вызов с override"));
        assert!(!second.metadata.cached);
        assert_eq!(second.metadata.model_used, "actual");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn cache_separates_reloaded_prompt() {
        let (runtime, store, registry, dir) =
            cache_runtime("[cache]\nenabled = true", "prompt-version-one");
        let request = cache_req();
        let first = sync_response(runtime.invoke(request.clone()).await.expect("первый вызов"));
        assert!(!first.metadata.cached);
        let first_key = cache::compute_key(cache::CacheKeyParts {
            agent_name: "test",
            variant: "default",
            provider_name: "fixed",
            model_name: "configured",
            prompt: "prompt-version-one",
            task_context: "",
            input: &request.input,
            key_fields: &[],
            task_id: None,
        });
        wait_for_cache(store.as_ref(), &first_key).await;

        std::fs::write(dir.join("agents/test/prompt.md"), "prompt-version-two")
            .expect("обновление prompt.md");
        registry.reload().expect("перечитка реестра");
        let second = sync_response(runtime.invoke(request).await.expect("второй вызов"));
        assert!(!second.metadata.cached);
        assert_eq!(second.result, Value::String("prompt-version-two".into()));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn cache_separates_change_beyond_task_context_limit() {
        let (runtime, store, _registry, dir) =
            cache_runtime("[cache]\nenabled = true", "{{ task_context }}");
        let task_id = store
            .create_task(&crate::store::NewTask::default())
            .await
            .expect("создание задачи");
        store
            .write_artifact(
                task_id,
                "query",
                "source",
                None,
                Some(&format!("{}-one", "с".repeat(7000))),
                None,
                None,
                &[],
            )
            .await
            .expect("первый артефакт");
        let mut request = cache_req();
        request.task_id = Some(task_id);
        let first = sync_response(runtime.invoke(request.clone()).await.expect("первый вызов"));
        assert!(!first.metadata.cached);
        let first_artifacts = store
            .read_artifacts(task_id, None)
            .await
            .expect("чтение артефактов");
        let first_context = format_task_context(&first_artifacts);
        assert!(first_context.chars().count() <= 6000);
        assert!(!first_context.contains("-one"));
        assert_eq!(first.result.as_str().expect("текст"), first_context);
        let first_key = cache::compute_key(cache::CacheKeyParts {
            agent_name: "test",
            variant: "default",
            provider_name: "fixed",
            model_name: "configured",
            prompt: "{{ task_context }}",
            task_context: &task_context_fingerprint(&first_artifacts),
            input: &request.input,
            key_fields: &[],
            task_id: Some(task_id),
        });
        wait_for_cache(store.as_ref(), &first_key).await;

        store
            .write_artifact(
                task_id,
                "query",
                "source",
                None,
                Some(&format!("{}-two", "с".repeat(7000))),
                None,
                None,
                &[],
            )
            .await
            .expect("обновление артефакта");
        let second = sync_response(runtime.invoke(request).await.expect("второй вызов"));
        assert!(!second.metadata.cached);
        assert_eq!(second.result.as_str().expect("текст"), first_context);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn cache_hits_for_identical_execution_identity() {
        let (runtime, store, _registry, dir) =
            cache_runtime("[cache]\nenabled = true", "same prompt");
        let request = cache_req();
        let first = sync_response(runtime.invoke(request.clone()).await.expect("первый вызов"));
        let key = cache::compute_key(cache::CacheKeyParts {
            agent_name: "test",
            variant: "default",
            provider_name: "fixed",
            model_name: "configured",
            prompt: "same prompt",
            task_context: "",
            input: &request.input,
            key_fields: &[],
            task_id: None,
        });
        wait_for_cache(store.as_ref(), &key).await;

        let second = sync_response(runtime.invoke(request).await.expect("второй вызов"));
        assert!(second.metadata.cached);
        assert_eq!(second.result, first.result);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn cache_hit_does_not_require_current_mcp_environment() {
        let variable = "AGENTS_MCP_TEST_CACHE_TOKEN_4D8E";
        let config = format!(
            "[cache]\nenabled = true\n[execution]\nmcp_config = '''{{\"mcpServers\":{{\"fixture\":{{\"url\":\"https://example/${{{variable}}}\"}}}}}}'''"
        );
        let (runtime, store, _registry, dir) = cache_runtime(&config, "same prompt");
        runtime.set_provider_env(ProviderEnv::capture(HashMap::from([(
            variable.to_string(),
            "first-secret".to_string(),
        )])));
        let request = cache_req();
        let first = sync_response(runtime.invoke(request.clone()).await.expect("первый вызов"));
        assert!(!first.metadata.cached);
        let key = cache::compute_key(cache::CacheKeyParts {
            agent_name: "test",
            variant: "default",
            provider_name: "fixed",
            model_name: "configured",
            prompt: "same prompt",
            task_context: "",
            input: &request.input,
            key_fields: &[],
            task_id: None,
        });
        wait_for_cache(store.as_ref(), &key).await;
        runtime.set_provider_env(ProviderEnv::default());

        let second = sync_response(runtime.invoke(request).await.expect("ответ из кеша"));

        assert!(second.metadata.cached);
        assert_eq!(second.result, first.result);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn background_cache_hit_gets_own_call_and_result_file() {
        let (runtime, _store, _registry, dir) =
            cache_runtime("[cache]\nenabled = true", "same prompt");

        let first = runtime
            .start_background(cache_req(), None)
            .await
            .expect("первый запуск");
        let (first_id, first_path) = match first {
            StartedJob::Running {
                call_id,
                result_path,
            } => (call_id, result_path),
            StartedJob::Done { .. } => panic!("первый запуск не должен попасть в кеш"),
        };
        let _ = runtime.wait(first_id, 1).await;
        wait_for_result_file(&first_path).await;
        let first_file = std::fs::read(&first_path).expect("первый файл-итог");

        let second = runtime
            .start_background(cache_req(), None)
            .await
            .expect("запуск из кеша");
        let (second_response, second_path) = match second {
            StartedJob::Done {
                response,
                result_path,
            } => (response, result_path),
            StartedJob::Running { .. } => panic!("второй запуск должен попасть в кеш"),
        };

        assert!(second_response.metadata.cached);
        assert_ne!(second_response.metadata.call_id, first_id);
        assert_ne!(second_path, first_path);
        assert_eq!(
            std::fs::read(&first_path).expect("первый файл после кеш-попадания"),
            first_file,
            "кеш-попадание не должно перезаписывать файл исходного вызова"
        );
        assert!(second_path.exists(), "кеш-попадание пишет свой файл-итог");
        let history = runtime.history(None, None, 10).await.expect("история");
        assert_eq!(
            history.len(),
            2,
            "кеш-попадание создаёт новую строку вызова"
        );
        assert!(
            history
                .iter()
                .any(|entry| entry.id == second_response.metadata.call_id && entry.cached),
            "строка кеш-попадания помечена cached"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn default_result_path_does_not_reuse_stale_envelope() {
        let (runtime, _store, dir) = test_runtime("worker", "answer");
        let stale_path = dir.join("runs/1-worker-previous.json");
        std::fs::create_dir_all(stale_path.parent().expect("каталог stale"))
            .expect("создание runs");
        std::fs::write(&stale_path, "старый конверт").expect("старый файл-итог");

        let started = runtime
            .start_background(invoke_req("worker", None), None)
            .await
            .expect("новый запуск");
        let (call_id, result_path) = match started {
            StartedJob::Running {
                call_id,
                result_path,
            } => (call_id, result_path),
            StartedJob::Done { .. } => panic!("кеш выключен"),
        };

        assert_eq!(call_id, 1, "новое хранилище начинает нумерацию с 1");
        assert_ne!(result_path, stale_path);
        assert_eq!(
            std::fs::read_to_string(&stale_path).expect("старый файл"),
            "старый конверт"
        );
        let _ = runtime.wait(call_id, 1).await;
        wait_for_result_file(&result_path).await;

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn cancel_async_invoke_without_result_path_does_not_create_file() {
        let (runtime, _store, _registry, dir) = cache_runtime("", "answer");
        runtime
            .providers
            .write()
            .expect("провайдеры")
            .insert("fixed".into(), Arc::new(PendingProvider));
        let mut request = cache_req();
        request.wait_sec = Some(0);
        let call_id = match runtime.invoke(request).await.expect("асинхронный запуск")
        {
            InvokeOutcome::Running { call_id, .. } => call_id,
            _ => panic!("ожидался запущенный вызов"),
        };

        assert!(matches!(
            runtime.cancel(call_id).await,
            CancelOutcome::Cancelled { .. }
        ));
        assert!(
            !dir.join("runs").exists(),
            "invoke_agent с wait_sec без result_path не должен создавать файл-итог"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Запустить `count` фоновых вызовов агента `test` в задаче task_id. Агент
    /// висит (PendingProvider), пока вызов не отменят; возвращаются call_id.
    async fn start_background_calls(runtime: &Runtime, task_id: i64, count: usize) -> Vec<i64> {
        let mut out = Vec::new();
        for _ in 0..count {
            let mut request = cache_req();
            request.wait_sec = Some(0);
            request.task_id = Some(task_id);
            match runtime.invoke(request).await.expect("запуск звена") {
                InvokeOutcome::Running { call_id } => out.push(call_id),
                _ => panic!("ожидался запущенный вызов"),
            }
        }
        out
    }

    #[tokio::test]
    async fn cancel_task_stops_only_its_calls_and_marks_task_cancelled() {
        let (runtime, store, _registry, dir) = cache_runtime("", "answer");
        runtime
            .providers
            .write()
            .expect("провайдеры")
            .insert("fixed".into(), Arc::new(PendingProvider));
        let blank = crate::store::NewTask::default();
        let task_a = store.create_task(&blank).await.expect("задача A");
        let task_b = store.create_task(&blank).await.expect("задача B");
        let calls_a = start_background_calls(&runtime, task_a, 2).await;
        let calls_b = start_background_calls(&runtime, task_b, 2).await;
        assert_eq!(runtime.calls.calls_of_task(task_a), calls_a);

        let outcome = runtime.cancel_task(task_a).await;
        match outcome.expect("отмена задачи A") {
            TaskCancelOutcome::Cancelled {
                task_id,
                cancelled_calls,
                previous_status,
                status,
            } => {
                assert_eq!(task_id, task_a);
                assert_eq!(cancelled_calls, calls_a);
                assert_eq!(previous_status, "running");
                assert_eq!(status, "cancelled");
            }
            other => panic!("ожидалась отмена задачи, получено {other:?}"),
        }
        let state = store.get_task_status(task_a).await;
        assert_eq!(state.expect("статус A"), Some("cancelled".to_string()));

        // Чужая задача не задета: её вызовы живы, и цепочку по ней вести можно.
        for call_id in &calls_b {
            assert!(runtime.calls.contains(*call_id));
        }
        assert!(
            runtime.calls.calls_of_task(task_a).is_empty(),
            "вызовы отменённой задачи обязаны уйти из реестра"
        );
        assert_eq!(runtime.calls.calls_of_task(task_b), calls_b);

        // Повторная отмена и неизвестная задача — внятные исходы, а не паника.
        let again = runtime.cancel_task(task_a).await;
        assert!(matches!(again, Ok(TaskCancelOutcome::AlreadyClosed { .. })));
        let unknown = runtime.cancel_task(999_999).await;
        assert!(matches!(unknown, Ok(TaskCancelOutcome::NotFound { .. })));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn orphan_result_path_uses_saved_explicit_path() {
        let dir =
            std::env::temp_dir().join(format!("agents-mcp-orphan-result-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("каталог");
        let expected = dir.join("explicit-result.json");
        let call = OrphanedCall {
            id: 42,
            agent_name: "worker".into(),
            created_at: 0,
            result_path: Some(expected.to_string_lossy().into_owned()),
        };

        let path = orphan_result_path(&dir, &call);
        assert_eq!(path, expected);
        write_result_file(
            &path,
            &envelope_error(call.id, &call.agent_name, "перезапуск"),
        )
        .expect("конверт осиротевшего вызова");
        assert!(path.exists());
        assert!(
            !dir.join("42-worker.json").exists(),
            "сохранённый result_path не должен подменяться путём по умолчанию"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn cleanup_result_files_removes_only_expired_results() {
        let dir = std::env::temp_dir().join(format!(
            "agents-mcp-result-cleanup-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("каталог");
        let stale = dir.join("stale.json");
        let fresh = dir.join("fresh.json");
        let other = dir.join("keep.txt");
        let stale_raw = dir.join("7-planner-raw.txt");
        std::fs::write(&stale, "старый").expect("старый итог");
        std::fs::write(&fresh, "свежий").expect("свежий итог");
        std::fs::write(&other, "не итог").expect("посторонний файл");
        std::fs::write(&stale_raw, "сырой ответ").expect("старый сырой ответ");
        // Граница — час назад, старому файлу время ставим явно на два часа
        // назад: время изменения файла на Windows грубее SystemTime::now(),
        // и граница «сейчас» между двумя записями давала плавающий итог.
        let now = SystemTime::now();
        for path in [&stale, &stale_raw] {
            std::fs::File::options()
                .write(true)
                .open(path)
                .and_then(|f| f.set_modified(now - Duration::from_secs(2 * 3600)))
                .expect("время старого файла");
        }
        let cutoff = now - Duration::from_secs(3600);

        assert_eq!(cleanup_result_files_before(&dir, cutoff), 2);
        assert!(!stale.exists());
        assert!(!stale_raw.exists(), "старый сырой ответ тоже убирается");
        assert!(fresh.exists());
        assert!(other.exists(), "посторонний .txt не трогаем");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn forced_model_matches_immediate_and_polled_response() {
        let (runtime, _store, _registry, dir) = cache_runtime("", "answer");
        *runtime.force_override.write().expect("override") = ModelOverride {
            provider: Some("fixed".into()),
            model: Some("actual".into()),
        };
        let response = sync_response(runtime.invoke(cache_req()).await.expect("вызов"));
        assert_eq!(response.metadata.model_used, "actual");

        match runtime.wait(response.metadata.call_id, 0).await {
            InvokeOutcome::Done(polled) => assert_eq!(polled.metadata.model_used, "actual"),
            _ => panic!("ожидался завершённый ответ"),
        }

        let _ = std::fs::remove_dir_all(dir);
    }

    // ── Провал после резерва call_id ─────────────────────────────────────

    /// Ошибка ПОСЛЕ резерва call_id (здесь — рендер промпта) обязана закрыть
    /// строку вызова ошибкой и попасть в журнал. До правки строка навсегда
    /// оставалась в 'running', а ждущий её опрос не дожидался конца.
    #[tokio::test]
    async fn prompt_render_failure_closes_call_row() {
        // Переменной нет в контексте — tera падает на рендере.
        let (runtime, store, dir) =
            test_runtime("broken-prompt", "Тема: {{ нет_такой_переменной }}");

        let invoked = runtime.invoke(invoke_req("broken-prompt", None)).await;
        let err = match invoked {
            Err(e) => e,
            Ok(_) => panic!("рендер с несуществующей переменной обязан упасть"),
        };
        assert!(
            matches!(&err, InvokeError::PromptRender(_)),
            "получили {err:?}"
        );

        let history = runtime.history(None, None, 10).await.expect("история");
        assert_eq!(history.len(), 1, "строка вызова должна быть ровно одна");
        assert!(
            history[0].error.is_some(),
            "провал обязан лежать в строке вызова"
        );
        let row = store
            .get_call_row(history[0].id)
            .await
            .expect("чтение строки вызова")
            .expect("строка вызова есть");
        assert_ne!(
            row.status, "running",
            "строка не должна остаться в 'running'"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Корректное завершение службы (prepare_shutdown) ──────────────────

    #[test]
    fn sync_guard_is_live_and_removed_on_drop() {
        let reg = Arc::new(CallRegistry::default());
        let guard = reg.register_sync(7, "a".to_string(), None);

        let live = reg.live_calls();
        assert_eq!(live.len(), 1, "синхронный вызов виден в реестре");
        assert_eq!(live[0].call_id, 7);
        assert_eq!(live[0].agent, "a");
        assert!(!live[0].background, "синхронный вызов — не фоновый");
        assert!(reg.contains(7), "contains видит синхронный вызов");

        drop(guard);
        assert!(
            reg.live_calls().is_empty(),
            "Drop снимает запись из реестра"
        );
        assert!(!reg.contains(7));
    }

    #[tokio::test]
    async fn drain_refuses_new_calls_but_admits_children_of_live() {
        let (runtime, _store, dir) = test_runtime("drain-agent", "Привет");

        runtime.begin_drain();
        let status = runtime.drain_status();
        assert!(status.draining, "приём закрыт");
        assert!(status.ready(), "живых вызовов нет — останавливать можно");

        let refused = runtime.invoke(invoke_req("drain-agent", None)).await;
        assert!(
            matches!(refused, Err(InvokeError::ShuttingDown)),
            "новый вызов обязан быть отклонён"
        );
        let history = runtime.history(None, None, 10).await.expect("история");
        assert!(history.is_empty(), "строка вызова не должна создаваться");

        let parent = runtime.calls.register_sync(999, "parent".to_string(), None);
        let child = runtime.invoke(invoke_req("drain-agent", Some(999))).await;
        assert!(child.is_ok(), "дочерний вызов идущего принимается");

        let stranger = runtime.invoke(invoke_req("drain-agent", Some(12345))).await;
        assert!(
            matches!(stranger, Err(InvokeError::ShuttingDown)),
            "вызов с неживым родителем отклоняется"
        );

        drop(parent);
        runtime.abort_drain();
        assert!(!runtime.is_draining(), "подготовка отменена");
        let again = runtime.invoke(invoke_req("drain-agent", None)).await;
        assert!(again.is_ok(), "приём снова открыт");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn drain_not_ready_while_preparing() {
        let (runtime, _store, dir) = test_runtime("drain-agent", "Привет");
        runtime.begin_drain();

        runtime.preparing.fetch_add(1, Ordering::SeqCst);
        let guard = AdmissionGuard(runtime.preparing.clone());
        let status = runtime.drain_status();
        assert_eq!(status.preparing, 1, "подготовка видна счётчику");
        assert!(!status.ready(), "идёт подготовка — останавливать нельзя");

        drop(guard);
        assert!(runtime.drain_status().ready(), "подготовка кончилась");
        assert!(runtime.wait_drained(1).await.ready());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn drain_ready_while_cancel_is_pending() {
        let (runtime, _store, dir) = test_runtime("drain-finalizing", "Привет");
        runtime.begin_drain();

        let guard = FinalizationGuard::new(runtime.finalizing.clone());
        let status = runtime.drain_status();
        assert_eq!(status.finalizing, 1, "финализация видна счётчику");
        assert!(
            !status.ready(),
            "пока отмена пишет итог, останавливать службу нельзя"
        );

        drop(guard);
        assert!(runtime.drain_status().ready(), "финализация закончилась");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn wait_drained_returns_when_another_request_aborts_drain() {
        let (runtime, _store, dir) = test_runtime("drain-abort", "Привет");
        let runtime = Arc::new(runtime);
        runtime.begin_drain();
        runtime.preparing.fetch_add(1, Ordering::SeqCst);
        let preparing = AdmissionGuard(runtime.preparing.clone());
        let waiting_runtime = runtime.clone();
        let waiter = tokio::spawn(async move { waiting_runtime.wait_drained(30).await });
        tokio::task::yield_now().await;

        runtime.abort_drain();
        let status = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("ожидание должно завершиться сразу после abort")
            .expect("задача ожидания не паникует");
        assert!(!status.draining, "приём снова открыт");

        drop(preparing);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn shutdown_finalizes_background_call_and_writes_result() {
        let (runtime, store, _registry, dir) = cache_runtime("", "answer");
        runtime
            .providers
            .write()
            .expect("провайдеры")
            .insert("fixed".into(), Arc::new(PendingProvider));
        let started = runtime
            .start_background(cache_req(), None)
            .await
            .expect("фоновый запуск");
        let (call_id, result_path) = match started {
            StartedJob::Running {
                call_id,
                result_path,
            } => (call_id, result_path),
            StartedJob::Done { .. } => panic!("кеш выключен"),
        };

        let stopped = runtime
            .finalize_background_calls("служба остановлена", Duration::from_secs(2))
            .await;
        assert_eq!(stopped, 1);
        let row = store
            .get_call_row(call_id)
            .await
            .expect("чтение строки")
            .expect("строка вызова");
        assert_eq!(row.status, "error");
        assert_eq!(row.error.as_deref(), Some("служба остановлена"));

        let envelope = wait_for_result_file(&result_path).await;
        assert_eq!(envelope["status"], "error");
        assert_eq!(envelope["error"], "служба остановлена");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn agent_timeout_overrides_runtime_default() {
        let (runtime, _, _, dir) = cache_runtime("", "answer");
        runtime.set_default_timeout_sec(7);
        assert_eq!(runtime.run_timeout(&cache_req()).unwrap(), 7);
        let _ = std::fs::remove_dir_all(&dir);

        let (runtime, _, _, dir) = cache_runtime("[limits]\ntimeout_sec = 11", "answer");
        runtime.set_default_timeout_sec(7);
        assert_eq!(runtime.run_timeout(&cache_req()).unwrap(), 11);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn overall_timeout_closes_background_and_sync_calls() {
        let (runtime, store, _, dir) = cache_runtime("[limits]\ntimeout_sec = 1", "answer");
        runtime.set_providers(HashMap::from([(
            "fixed".to_string(),
            Arc::new(PendingProvider) as Arc<dyn LlmProvider>,
        )]));
        let started = runtime.start_background(cache_req(), None).await.unwrap();
        let (call_id, result_path) = match started {
            StartedJob::Running {
                call_id,
                result_path,
            } => (call_id, result_path),
            StartedJob::Done { .. } => panic!("кеш выключен"),
        };
        let envelope = wait_for_result_file(&result_path).await;
        assert_eq!(envelope["status"], "error");
        assert!(envelope["error"]
            .as_str()
            .unwrap()
            .contains("общий срок прогона"));
        let row = store.get_call_row(call_id).await.unwrap().unwrap();
        assert_eq!(row.status, "error");
        let _ = std::fs::remove_dir_all(&dir);

        let (runtime, _, _, dir) = cache_runtime("[limits]\ntimeout_sec = 1", "answer");
        runtime.set_providers(HashMap::from([(
            "fixed".to_string(),
            Arc::new(PendingProvider) as Arc<dyn LlmProvider>,
        )]));
        let error = match runtime.invoke(cache_req()).await {
            Err(error) => error,
            Ok(_) => panic!("синхронный вызов обязан завершиться по сроку"),
        };
        assert!(matches!(error, InvokeError::RunTimeout { timeout_sec: 1 }));
        assert!(!dir.join("runs").exists(), "синхронный вызов не пишет файл");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn silent_skills_service_is_soft_and_writes_turn_event() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(2)).await;
        });

        let dir = std::env::temp_dir().join(format!(
            "agents-mcp-skills-timeout-{}",
            uuid::Uuid::new_v4()
        ));
        let agent_dir = dir.join("agents/test");
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::write(
            agent_dir.join("config.toml"),
            "name = \"test\"\nskill_include_1c = true\n\
             [model]\nprovider = \"fixed\"\nname = \"fixture\"\n\
             [limits]\ntimeout_sec = 2\n",
        )
        .unwrap();
        std::fs::write(agent_dir.join("prompt.md"), "{{ skills_index }}done").unwrap();
        let db_path = dir.join("store.sqlite");
        let store: Arc<dyn Store> = Arc::new(crate::store::SqliteStore::open(&db_path).unwrap());
        let registry = Arc::new(Registry::load(dir.join("agents")).unwrap());
        let providers = HashMap::from([(
            "fixed".to_string(),
            Arc::new(FixedProvider {
                finish_reason: "stop",
                panic: false,
            }) as Arc<dyn LlmProvider>,
        )]);
        let runtime = Runtime::new(
            store,
            registry,
            providers,
            crate::skills::SkillsClient::with_test_timeouts(url, Duration::from_millis(100)),
            Arc::new(std::sync::RwLock::new(ModelOverride::default())),
            dir.join("runs"),
            "test:skills".into(),
            120,
        );
        let mut request = cache_req();
        request
            .input
            .insert("brief".to_string(), Value::String("проверка".to_string()));
        let started = Instant::now();
        let response = match runtime.invoke(request).await.unwrap() {
            InvokeOutcome::Sync(response) => response,
            _ => panic!("ожидался синхронный ответ"),
        };
        assert_eq!(response.result, Value::String("done".to_string()));
        assert!(started.elapsed() < Duration::from_secs(1));

        tokio::time::sleep(Duration::from_millis(50)).await;
        let connection = rusqlite::Connection::open(&db_path).unwrap();
        let event: String = connection
            .query_row(
                "SELECT event FROM agent_turns WHERE call_id=?1 AND event='skills_unavailable'",
                [response.metadata.call_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(event, "skills_unavailable");
        server.abort();
        drop(connection);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
