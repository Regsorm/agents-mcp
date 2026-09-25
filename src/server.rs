//! Транспорты службы: `/health`, MCP по Streamable HTTP без сессий и stdio.
//!
//! MCP-инструменты охватывают вызов и остановку агентов, реестр и историю,
//! доску задач и артефакты, файловые операции, навыки и перечитку конфига.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use axum::{extract::State, response::Json, routing::get, Router};
use rmcp::transport::streamable_http_server::{
    session::never::NeverSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    tool, tool_router, ServerHandler,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::Config;
use crate::health::{HealthResponse, ProviderStatus};
use crate::read_guard::{
    ask_read_guard, read_guard_applies, GuardVerdict, ReadGuard, READ_GUARD_TIMEOUT,
};
use crate::registry::Registry;
use crate::runtime::{
    CallScope, CancelOutcome, DrainStatus, InvokeOutcome, InvokeRequest, Runtime, StartedJob,
    TaskCancelOutcome, CALL_KEY_HEADER,
};

const MAX_EDIT_FILE_SIZE: u64 = 5 * 1024 * 1024;
const MAX_FS_LIST_DEPTH: usize = 32;
const MAX_FS_LIST_ENTRIES: usize = 10_000;

#[derive(Clone)]
pub struct AppState {
    #[allow(dead_code)]
    pub config: Arc<Config>,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub registry: Arc<Registry>,
    pub runtime: Arc<Runtime>,
}

#[derive(Clone)]
pub struct AgentsMcpServer {
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub registry: Arc<Registry>,
    pub runtime: Arc<Runtime>,
    /// Разрешённые корни для файловых инструментов fs_* (из config [fs]).
    /// За RwLock: перечитка главного конфига подменяет список на лету.
    fs_roots: Arc<std::sync::RwLock<Vec<PathBuf>>>,
    read_guard: Arc<std::sync::RwLock<Option<ReadGuard>>>,
    /// Служебные пути защищены независимо от ширины allowed_roots.
    service_paths: Arc<std::sync::RwLock<crate::reload::ServicePaths>>,
    /// Перечитка главного конфига без перезапуска — бэкенд инструмента config_reload.
    reloader: Arc<crate::reload::ConfigReloader>,
    tool_router: ToolRouter<Self>,
}

impl AgentsMcpServer {
    pub fn new(
        started_at: chrono::DateTime<chrono::Utc>,
        registry: Arc<Registry>,
        runtime: Arc<Runtime>,
        reloader: Arc<crate::reload::ConfigReloader>,
    ) -> Self {
        // Корни берём у перечитки: она же подменит их при смене [fs] allowed_roots.
        let fs_roots = reloader.fs_roots();
        let read_guard = reloader.read_guard();
        let service_paths = reloader.service_paths();
        Self {
            started_at,
            registry,
            runtime,
            fs_roots,
            read_guard,
            service_paths,
            reloader,
            tool_router: Self::tool_router(),
        }
    }

    fn request_call_scope(
        &self,
        extensions: &rmcp::model::Extensions,
    ) -> Result<Option<CallScope>, String> {
        let Some(parts) = extensions.get::<axum::http::request::Parts>() else {
            return Ok(None);
        };
        let Some(value) = parts.headers.get(CALL_KEY_HEADER) else {
            return Ok(None);
        };
        let key = value
            .to_str()
            .map_err(|_| "негодный ключ вызова в HTTP-заголовке".to_string())?;
        self.runtime
            .call_scope(key)
            .map(Some)
            .ok_or_else(|| "неизвестный или уже снятый ключ вызова".to_string())
    }

    /// Один раз на запрос: рабочий каталог и корни проверки пути. С ключом
    /// каталог берётся только из доверенного контекста runtime, корни — тоже
    /// (allowed_roots агента вместо общего [fs]); без ключа сохраняется
    /// прежнее поведение внешних клиентов.
    fn fs_access(
        &self,
        extensions: &rmcp::model::Extensions,
        supplied: Option<&str>,
    ) -> Result<(Option<PathBuf>, Vec<PathBuf>), String> {
        let scope = self.request_call_scope(extensions)?;
        let scope_dir = effective_fs_scope(scope.clone(), supplied)?;
        let roots = {
            let service_roots = self.fs_roots.read().unwrap_or_else(|e| e.into_inner());
            effective_fs_roots(scope.as_ref(), &service_roots)
        };
        Ok((scope_dir, roots))
    }

    fn child_lineage(
        &self,
        extensions: &rmcp::model::Extensions,
        supplied_parent: Option<i64>,
        supplied_depth: u32,
    ) -> Result<(Option<i64>, u32), String> {
        let scope = self.request_call_scope(extensions)?;
        if let Some(scope) = &scope {
            tracing::debug!(
                call_id = scope.call_id,
                parent_call_id = ?scope.parent_call_id,
                orchestration_depth = scope.orchestration_depth,
                "родословная дочернего вызова взята из ключа"
            );
        }
        Ok(effective_child_lineage(
            scope.as_ref(),
            supplied_parent,
            supplied_depth,
        ))
    }
}

fn effective_fs_scope(
    call_scope: Option<CallScope>,
    supplied: Option<&str>,
) -> Result<Option<PathBuf>, String> {
    match call_scope {
        Some(scope) => scope.cwd.map(Some).ok_or_else(|| {
            "для этого вызова не задан рабочий каталог: файловые инструменты запрещены".to_string()
        }),
        None => Ok(supplied.map(PathBuf::from)),
    }
}

/// Корни для проверки файловых путей: у вызова с ключом и своим списком
/// allowed_roots агента действуют корни агента ВМЕСТО общего [fs].allowed_roots;
/// без ключа или без своего списка — общий список, как раньше. Пустой список
/// агента отклоняется ещё при загрузке (load_agent), здесь не проверяется.
fn effective_fs_roots(call_scope: Option<&CallScope>, service_roots: &[PathBuf]) -> Vec<PathBuf> {
    match call_scope {
        Some(scope) => scope
            .allowed_roots
            .clone()
            .unwrap_or_else(|| service_roots.to_vec()),
        None => service_roots.to_vec(),
    }
}

fn effective_child_lineage(
    call_scope: Option<&CallScope>,
    supplied_parent: Option<i64>,
    supplied_depth: u32,
) -> (Option<i64>, u32) {
    match call_scope {
        Some(scope) => (
            Some(scope.call_id),
            scope.orchestration_depth.saturating_add(1),
        ),
        None => (supplied_parent, supplied_depth),
    }
}

// ── Параметры tools ────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EmptyParams {}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct FsWriteParams {
    /// Абсолютный путь файла внутри разрешённого корня (config [fs]).
    pub path: String,
    /// Содержимое файла (UTF-8).
    pub content: String,
    /// Служебное поле: рабочий каталог ЭТОГО вызова агента. Подставляется
    /// циклом провайдера (openrouter.rs) из cli_hints.cwd — модель заполнять
    /// его не должна. Отсутствует — путь проверяется только по allowed_roots.
    #[serde(default)]
    pub scope_dir: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct FsPathParams {
    /// Абсолютный путь внутри разрешённого корня (config [fs]).
    pub path: String,
    /// Служебное поле: рабочий каталог ЭТОГО вызова агента. Подставляется
    /// циклом провайдера (openrouter.rs) из cli_hints.cwd — модель заполнять
    /// его не должна. Отсутствует — путь проверяется только по allowed_roots.
    #[serde(default)]
    pub scope_dir: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct FsListParams {
    /// Абсолютный путь каталога внутри разрешённого корня (config [fs]).
    pub path: String,
    /// Рекурсивный обход (по умолчанию false).
    #[serde(default)]
    pub recursive: bool,
    /// Служебное поле: рабочий каталог ЭТОГО вызова агента. Подставляется
    /// циклом провайдера (openrouter.rs) из cli_hints.cwd — модель заполнять
    /// его не должна. Отсутствует — путь проверяется только по allowed_roots.
    #[serde(default)]
    pub scope_dir: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct FsEditParams {
    /// Абсолютный путь файла внутри разрешённого корня (config [fs]).
    pub path: String,
    /// Заменяемый фрагмент текста. Должен быть непустым.
    pub old_string: String,
    /// Текст, на который заменяется old_string.
    pub new_string: String,
    /// Заменить ВСЕ вхождения old_string (по умолчанию false — требуется
    /// ровно одно вхождение, иначе отказ).
    #[serde(default)]
    pub replace_all: bool,
    /// Служебное поле: рабочий каталог ЭТОГО вызова агента. Подставляется
    /// циклом провайдера (openrouter.rs) из cli_hints.cwd — модель заполнять
    /// его не должна. Отсутствует — путь проверяется только по allowed_roots.
    #[serde(default)]
    pub scope_dir: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SkillLoadParams {
    /// Имя навыка (skills.name) из индекса skills_index.
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetAgentParams {
    /// Имя агента (= имя папки в agents/).
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct InvokeAgentParams {
    /// Имя агента из реестра.
    pub agent: String,
    /// Объект с параметрами, подставляется в prompt.md через tera.
    /// На клиенте передаётся как `{"brief": "...", "src_files": "..."}`.
    /// ВАЖНО: тип `Map<String, Value>`, а не `Value` — schemars для `Value`
    /// генерит схему без `type=object`, из-за чего Claude Code framework
    /// сериализует параметр как JSON-строку (вылезает в режиме без auto-discovery
    /// CLAUDE.md, когда модель строго следует schema). `Map` форсирует
    /// `type=object` явно.
    #[serde(default)]
    pub input: serde_json::Map<String, Value>,
    /// Опциональный variant (prompt.v2.md → "v2"). По умолчанию "default".
    #[serde(default)]
    pub variant: Option<String>,
    /// Служебное поле для агентов-оркестраторов: id внешнего вызова из
    /// `agent_calls`, чтобы текущий вызов записался как его child. У корневого
    /// вызова от клиента — отсутствует.
    #[serde(default)]
    pub parent_call_id: Option<i64>,
    /// Служебное поле: текущая глубина в дереве оркестрации (0 для корневого
    /// вызова). Оркестратор должен прокидывать `orchestration_depth + 1` в
    /// каждый рекурсивный invoke_agent. Превышение `max_orchestration_depth`
    /// (по умолчанию 5) — отказ.
    #[serde(default)]
    pub orchestration_depth: u32,
    /// Режим ожидания результата (async, против 60-секундного MCP-таймаута):
    ///   опущено — синхронно: ждать завершения, вернуть {result, metadata};
    ///   0       — запустить в фоне и сразу вернуть {status:"running", call_id};
    ///   N       — запустить в фоне и подождать до N секунд (сервер зажимает ≤55):
    ///             готово — {status:"done", result, metadata}, иначе running.
    /// Оркестраторы должны звать с wait_sec, чтобы под-агенты длительностью
    /// >60с не валили MCP-tool по таймауту; затем опрашивать через wait_agent.
    #[serde(default)]
    pub wait_sec: Option<u64>,
    /// id задачи в PG task-store, к которой относится вызов. Оркестратор
    /// прокидывает его в каждый invoke_agent под-агента; runtime по нему
    /// собирает срез артефактов задачи в {{ task_context }}. Опущено — вне задачи.
    #[serde(default)]
    pub task_id: Option<i64>,
    /// Плоский объект перекрытий настроек агента на ОДИН вызов (`overrides`):
    /// ключи `model.name`, `model.temperature`, `model.max_tokens`,
    /// `execution.max_turns`, `limits.timeout_sec`, `effort`, `cache.enabled`,
    /// `mcp.<сервер>.url`. Список ключей закрытый — неизвестный ключ отказ.
    /// Действует только на этот вызов, в шаблон промпта не попадает; вложенные
    /// вызовы (оркестратор → под-агент) его не наследуют.
    #[serde(default)]
    pub overrides: Option<serde_json::Map<String, Value>>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct WaitAgentParams {
    /// call_id из ответа invoke_agent (status="running").
    pub call_id: i64,
    /// Сколько секунд блокирующе ждать готовности (сервер зажимает ≤55).
    /// По умолчанию 50. Если задача ещё не готова — вернётся status="running",
    /// тогда звать wait_agent повторно с тем же call_id.
    #[serde(default)]
    pub wait_sec: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentRunParams {
    /// РЕЖИМ ПУСКА: имя агента из реестра. Взаимно исключается с `call_id`.
    #[serde(default)]
    pub agent: Option<String>,
    /// Объект подстановок в prompt.md (как у invoke_agent). Только при пуске.
    #[serde(default)]
    pub input: serde_json::Map<String, Value>,
    /// Вариант промпта (prompt.<variant>.md). По умолчанию "default".
    #[serde(default)]
    pub variant: Option<String>,
    /// id задачи в PG task-store — вызов привяжется к её доске ({{ task_context }}).
    #[serde(default)]
    pub task_id: Option<i64>,
    /// Куда положить файл-итог: АБСОЛЮТНЫЙ путь файла внутри разрешённого
    /// корня (config [fs]); родительский каталог уже должен существовать.
    /// Опущено — каталог [storage] runs_dir сервиса.
    #[serde(default)]
    pub result_path: Option<String>,
    /// РЕЖИМ ПРОВЕРКИ: call_id ранее запущенного вызова. Взаимно исключается
    /// с `agent`.
    #[serde(default)]
    pub call_id: Option<i64>,
    /// Плоский объект перекрытий настроек агента на ОДИН вызов (`overrides`).
    /// Тот же закрытый список ключей, что у `invoke_agent`; действует только
    /// на этот вызов, в шаблон промпта не попадает, вложенные вызовы его не
    /// наследуют.
    #[serde(default)]
    pub overrides: Option<serde_json::Map<String, Value>>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentCancelParams {
    /// Живой вызов, который надо остановить (call_id из ответа `agent_run`
    /// либо асинхронного `invoke_agent`). Опущено — вернуть список живых
    /// вызовов.
    #[serde(default)]
    pub call_id: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct PrepareShutdownParams {
    /// Сколько секунд дождаться завершения идущих вызовов, прежде чем ответить
    /// (сервер зажимает ≤55). По умолчанию 0 — ответить сразу текущим
    /// состоянием.
    #[serde(default)]
    pub wait_sec: Option<u64>,
    /// true — отменить подготовку к остановке и снова открыть приём вызовов.
    #[serde(default)]
    pub abort: Option<bool>,
}

fn prepare_shutdown_response(st: &DrainStatus) -> String {
    let live: Vec<Value> = st
        .live
        .iter()
        .map(|c| {
            serde_json::json!({
                "call_id": c.call_id,
                "agent": c.agent,
                "elapsed_sec": c.elapsed_sec,
                "background": c.background,
            })
        })
        .collect();
    let status = if !st.draining {
        "accepting"
    } else if st.ready() {
        "ready"
    } else {
        "draining"
    };
    let hint = if !st.draining {
        "приём вызовов снова открыт другим запросом abort=true"
    } else if st.ready() {
        "идущих вызовов нет, службу можно останавливать"
    } else {
        "приём закрыт, идут вызовы: повторите вызов позже, отмените фоновые через \
         agent_cancel или отмените подготовку abort=true"
    };
    serde_json::json!({
        "status": status,
        "draining": st.draining,
        "draining_sec": st.draining_sec,
        "preparing": st.preparing,
        "finalizing": st.finalizing,
        "live": live,
        "hint": hint,
    })
    .to_string()
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentHistoryParams {
    #[serde(default)]
    pub agent: Option<String>,
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub since: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct TaskCreateParams {
    /// id корневого вызова оркестратора (agent_calls.id), опционально.
    #[serde(default)]
    pub root_call_id: Option<i64>,
    /// Корреляционный id со стороны клиента, опционально.
    #[serde(default)]
    pub external_task_id: Option<String>,
    /// build_artifact | data_check | explain.
    #[serde(default)]
    pub task_kind: Option<String>,
    /// Якорь цели / критерии приёмки.
    #[serde(default)]
    pub goal: Option<String>,
    #[serde(default)]
    pub target_base: Option<String>,
    #[serde(default)]
    pub working_dir: Option<String>,
    #[serde(default)]
    pub sandbox_path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ArtifactWriteParams {
    pub task_id: i64,
    /// metadata | query | bsl_module | form_def | build_path | review | deliverable.
    pub kind: String,
    /// Логическое имя артефакта (уникально в пределах задачи).
    pub key: String,
    #[serde(default)]
    pub content: Option<String>,
    /// Краткое назначение — чтобы агент ориентировался не читая content целиком.
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub producer_agent: Option<String>,
    #[serde(default)]
    pub producer_call_id: Option<i64>,
    /// id артефактов, от которых зависит этот (рёбра DAG).
    #[serde(default)]
    pub depends_on: Option<Vec<i64>>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ArtifactReadParams {
    pub task_id: i64,
    /// Фильтр по kind; опущено — все артефакты задачи.
    #[serde(default)]
    pub kinds: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct TaskStatusParams {
    pub task_id: i64,
    /// running | needs_input | completed | failed | cancelled.
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ChainCancelParams {
    /// Задача task-store, цепочку по которой надо остановить (task_id из
    /// `task_create` / ответа цепочки).
    pub task_id: i64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct TaskGetParams {
    pub task_id: i64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EventAppendParams {
    pub task_id: i64,
    /// user_input | agent_start | agent_done | artifact_written | error |
    /// status_change | chain_cancelled.
    pub event_type: String,
    #[serde(default)]
    pub agent: Option<String>,
    #[serde(default)]
    pub call_id: Option<i64>,
    /// JSON-строка payload (опционально), кладётся как jsonb.
    #[serde(default)]
    pub payload_json: Option<String>,
}

/// Сериализовать InvokeOutcome в JSON-строку ответа MCP-tool.
/// Синхронный путь сохраняет прежний bare-формат {result, metadata}
/// (обратная совместимость с прямыми вызовами и тестами); async-пути
/// заворачиваются в конверт со status.
fn outcome_to_json(outcome: InvokeOutcome) -> String {
    match outcome {
        InvokeOutcome::Sync(resp) => {
            serde_json::to_string_pretty(&resp).unwrap_or_else(|_| "{}".to_string())
        }
        InvokeOutcome::Done(resp) => serde_json::json!({
            "status": "done",
            "result": resp.result,
            "metadata": resp.metadata,
        })
        .to_string(),
        InvokeOutcome::Incomplete { response, error } => serde_json::json!({
            "status": "incomplete",
            "call_id": response.metadata.call_id,
            "result": response.result,
            "metadata": response.metadata,
            "error": error,
        })
        .to_string(),
        InvokeOutcome::Running { call_id } => serde_json::json!({
            "status": "running",
            "call_id": call_id,
        })
        .to_string(),
        InvokeOutcome::Failed { call_id, error } => serde_json::json!({
            "status": "error",
            "call_id": call_id,
            "error": error,
        })
        .to_string(),
        InvokeOutcome::Cancelled { call_id, error } => serde_json::json!({
            "status": "cancelled",
            "call_id": call_id,
            "error": error,
        })
        .to_string(),
        InvokeOutcome::PersistenceFailed {
            call_id,
            response,
            error,
        } => match response {
            Some(resp) => serde_json::json!({
                "status": "persistence_failed",
                "call_id": call_id,
                "result": resp.result,
                "metadata": resp.metadata,
                "error": error,
            })
            .to_string(),
            None => serde_json::json!({
                "status": "persistence_failed",
                "call_id": call_id,
                "error": error,
            })
            .to_string(),
        },
    }
}

// ── MCP Tools ──────────────────────────────────────────────────────────────

#[tool_router]
impl AgentsMcpServer {
    #[tool(
        description = "Статус сервиса agents-mcp: версия, uptime, число загруженных агентов, регистрация и последняя CLI-проверка провайдеров LLM, \
                       доступность хранилища задач (проверяется запросом к БД). Возвращает JSON HealthResponse."
    )]
    pub async fn health(&self, Parameters(_): Parameters<EmptyParams>) -> String {
        let resp = build_health(self.started_at, self.registry.len(), &self.runtime).await;
        serde_json::to_string_pretty(&resp).unwrap_or_else(|_| "{}".to_string())
    }

    #[tool(
        description = "Список всех загруженных агентов с описанием, моделью, тегами. Возвращает JSON-массив."
    )]
    pub async fn list_agents(&self, Parameters(_): Parameters<EmptyParams>) -> String {
        let agents = self.registry.list();
        let items: Vec<Value> = agents
            .iter()
            .map(|a| {
                serde_json::json!({
                    "name": a.name,
                    "description": a.config.description,
                    "version": a.config.version,
                    "tags": a.config.tags,
                    "kind": a.config.kind,
                    "provider": a.config.model.provider,
                    "model": a.config.model.name,
                    "response_format": match a.config.response.format {
                        crate::registry::ResponseFormat::Text => "text",
                        crate::registry::ResponseFormat::Json => "json",
                    },
                    "variants": a.prompts.keys().collect::<Vec<_>>(),
                })
            })
            .collect();
        serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string())
    }

    #[tool(
        description = "Детальная информация об агенте по имени: конфиг, варианты промптов, last_modified. \
                       Если агента нет — возвращает JSON {\"error\": \"...\"}."
    )]
    pub async fn get_agent_info(&self, Parameters(p): Parameters<GetAgentParams>) -> String {
        match self.registry.get(&p.name) {
            Some(a) => {
                let mut config = serde_json::to_value(&a.config).unwrap_or(Value::Null);
                redact_mcp_config_env(&mut config);
                let detail = serde_json::json!({
                    "name": a.name,
                    "config": config,
                    "variants": a.prompts.keys().collect::<Vec<_>>(),
                    "last_modified": a.last_modified.to_rfc3339(),
                });
                serde_json::to_string_pretty(&detail).unwrap_or_else(|_| "{}".to_string())
            }
            None => {
                serde_json::json!({"error": format!("агент '{}' не найден", p.name)}).to_string()
            }
        }
    }

    #[tool(
        description = "Вызвать LLM-агента: рендерит prompt.md с подстановками из input, обращается к провайдеру, \
                       парсит ответ. \
                       Синхронно (wait_sec опущен): возвращает {result, metadata: {agent, variant, model_used, \
                       provider, tokens_in, tokens_out, cost_usd, latency_ms, cached, call_id}}, где \
                       cost_usd = null, если цена модели неизвестна. \
                       Асинхронно (wait_sec задан): запускает job в фоне и возвращает конверт со status — \
                       {status:\"running\", call_id} (опрашивать через wait_agent), либо {status:\"done\", \
                       result, metadata}, либо {status:\"error\", call_id, error}. \
                       Async нужен оркестраторам, чтобы под-агенты дольше 60с не валили MCP-tool по таймауту. \
                       Необязательный overrides — плоский объект из закрытого списка настроек только этого \
                       вызова; в input и вложенные вызовы он не попадает. \
                       При ошибке вызова возвращает {error: \"...\"}."
    )]
    pub async fn invoke_agent(
        &self,
        extensions: rmcp::model::Extensions,
        Parameters(p): Parameters<InvokeAgentParams>,
    ) -> String {
        let (parent_call_id, orchestration_depth) =
            match self.child_lineage(&extensions, p.parent_call_id, p.orchestration_depth) {
                Ok(lineage) => lineage,
                Err(e) => return err_json(&e),
            };
        // input уже Map<String, Value> благодаря явной типизации в схеме
        // (см. комментарий в InvokeAgentParams). Дополнительной нормализации
        // не требуется — required-поля проверяются в runtime ниже.
        let req = InvokeRequest {
            agent: p.agent,
            input: p.input,
            variant: p.variant,
            parent_call_id,
            orchestration_depth,
            wait_sec: p.wait_sec,
            task_id: p.task_id,
            overrides: p.overrides,
        };

        match self.runtime.invoke(req).await {
            Ok(outcome) => outcome_to_json(outcome),
            Err(e) => serde_json::json!({"error": format!("{e}")}).to_string(),
        }
    }

    #[tool(
        description = "Дождаться результата ранее запущенного async-вызова invoke_agent (по call_id из его \
                       ответа со status=\"running\"). Блокирует сервер максимум wait_sec секунд (зажимается \
                       ≤55, дефолт 50) — гарантированно вернётся под 60-секундным MCP-таймаутом. \
                       Возвращает тот же конверт, что async invoke_agent: {status:\"done\", result, metadata}, \
                       {status:\"running\", call_id} (тогда звать wait_agent повторно), либо \
                       {status:\"error\", call_id, error}."
    )]
    pub async fn wait_agent(&self, Parameters(p): Parameters<WaitAgentParams>) -> String {
        let wait_sec = p.wait_sec.unwrap_or(50);
        let outcome = self.runtime.wait(p.call_id, wait_sec).await;
        outcome_to_json(outcome)
    }

    #[tool(
        description = "Запустить агента в фоне и НЕ ждать его — единственный вызов, который никогда не \
                       блокирует вызывающего (ни на секунду). Два режима у одного tool: \
                       ПУСК — передать agent (+ input, variant, task_id, result_path, overrides): агент уходит \
                       работать в фон сервиса, ответ приходит сразу: {status:\"running\", call_id, agent, \
                       variant, result_path}. Если ответ нашёлся в кеше — сразу {status:\"done\", result, \
                       metadata, result_path}. \
                       ПРОВЕРКА — передать только call_id: мгновенный ответ о состоянии того вызова — \
                       {status:\"running\", call_id, elapsed_sec} либо {status:\"done\", result, metadata}, \
                       либо {status:\"error\", call_id, error}. \
                       Забрать итог можно двумя путями: этой же проверкой по call_id или чтением файла \
                       result_path — он появляется целиком (пишется через временный файл) в момент \
                       завершения, поэтому на его появление можно подписаться средствами клиента, \
                       не занимая беседу ожиданием. \
                       Чем отличается от соседей: invoke_agent без wait_sec ждёт агента до конца, \
                       wait_agent держит запрос до 55 с — оба занимают вызывающего; agent_run не ждёт \
                       никогда и файла-итога у тех двух нет. \
                       overrides — плоский объект из того же закрытого списка, что у invoke_agent; он \
                       действует только на запускаемый вызов и не наследуется вложенными вызовами. \
                       Ошибки подготовки (агента нет, не хватает обязательных полей input, нет такого \
                       варианта промпта) возвращаются сразу как {error} — в фон такой вызов не уходит."
    )]
    pub async fn agent_run(
        &self,
        extensions: rmcp::model::Extensions,
        Parameters(p): Parameters<AgentRunParams>,
    ) -> String {
        let call_scope = match self.request_call_scope(&extensions) {
            Ok(scope) => scope,
            Err(e) => return err_json(&e),
        };
        match (p.agent, p.call_id) {
            (Some(_), Some(_)) => {
                err_json("передайте либо agent (пуск), либо call_id (проверка) — не оба сразу")
            }
            (None, None) => {
                err_json("нужен agent (пуск нового вызова) либо call_id (проверка запущенного)")
            }
            // Проверка: один взгляд на строку вызова, без ожидания.
            (None, Some(call_id)) => {
                let (outcome, created_at) = self.runtime.check(call_id).await;
                match outcome {
                    InvokeOutcome::Running { call_id } => {
                        let elapsed =
                            created_at.map(|c| (chrono::Utc::now().timestamp() - c).max(0));
                        serde_json::json!({
                            "status": "running",
                            "call_id": call_id,
                            "elapsed_sec": elapsed,
                        })
                        .to_string()
                    }
                    other => outcome_to_json(other),
                }
            }
            // Пуск: подготовка синхронно (чтобы ошибки входа вернулись сразу),
            // сам вызов — в фон.
            (Some(agent), None) => {
                let (parent_call_id, orchestration_depth) =
                    effective_child_lineage(call_scope.as_ref(), None, 0);
                let result_path = match p.result_path.as_deref() {
                    // Корни читаем под коротким захватом: guard не живёт через
                    // await ниже (start_background).
                    Some(raw) => {
                        let roots = self
                            .fs_roots
                            .read()
                            .unwrap_or_else(|e| e.into_inner())
                            .clone();
                        let service_paths = self
                            .service_paths
                            .read()
                            .unwrap_or_else(|e| e.into_inner())
                            .clone();
                        match fs_safe_result_path(&roots, &service_paths, raw) {
                            Ok(x) => Some(x),
                            Err(e) => return err_json(&e),
                        }
                    }
                    None => None,
                };
                let variant = p.variant.clone().unwrap_or_else(|| "default".to_string());
                let req = InvokeRequest {
                    agent: agent.clone(),
                    input: p.input,
                    variant: p.variant,
                    parent_call_id,
                    orchestration_depth,
                    // Ожиданием здесь не управляют: режим всегда «не ждать».
                    wait_sec: Some(0),
                    task_id: p.task_id,
                    overrides: p.overrides,
                };
                match self.runtime.start_background(req, result_path).await {
                    Ok(StartedJob::Running {
                        call_id,
                        result_path,
                    }) => serde_json::json!({
                        "status": "running",
                        "call_id": call_id,
                        "agent": agent,
                        "variant": variant,
                        "result_path": path_for_client(&result_path),
                    })
                    .to_string(),
                    Ok(StartedJob::Done {
                        response,
                        result_path,
                    }) => serde_json::json!({
                        "status": "done",
                        "call_id": response.metadata.call_id,
                        "agent": agent,
                        "variant": variant,
                        "result": response.result,
                        "metadata": response.metadata,
                        "result_path": path_for_client(&result_path),
                    })
                    .to_string(),
                    Err(e) => err_json(&format!("{e}")),
                }
            }
        }
    }

    #[tool(
        description = "Остановить идущий ФОНОВЫЙ вызов агента — или посмотреть, какие сейчас идут. \
                       С call_id — отменить этот вызов: {status:\"cancelled\", call_id, agent, \
                       elapsed_sec}. Строка вызова в истории при этом закрывается ошибкой «вызов \
                       отменён вручную», а файл-итог дописывается конвертом ошибки — тот, кто ждёт \
                       его появления, не зависает. \
                       Без call_id — список живых вызовов: {live:[{call_id, agent, elapsed_sec, background, task_id}]}. \
                       Если живого вызова нет: {status:\"not_found\", call_id, hint}; если вызов уже \
                       завершён либо идёт внутри синхронного запроса: {status:\"already_finished\", \
                       call_id, call_status, hint}. \
                       Отменяются ТОЛЬКО фоновые вызовы — запущенные agent_run и асинхронным \
                       invoke_agent (wait_sec задан): у них своя задача в службе. Вызов, идущий \
                       внутри синхронного запроса (invoke_agent без wait_sec), отменить нельзя — он \
                       держит вызывающего; его остаётся только переждать. \
                       Список (вызов без call_id) показывает и синхронные вызовы (background=false); \
                       отменить можно только фоновые (background=true)."
    )]
    pub async fn agent_cancel(&self, Parameters(p): Parameters<AgentCancelParams>) -> String {
        // Список живых вызовов отдаёт тот же инструмент: второго (вроде
        // agent_live) не заводим — без call_id он и так никого не отменяет.
        let call_id = match p.call_id {
            Some(id) => id,
            None => {
                let live: Vec<Value> = self
                    .runtime
                    .live_calls()
                    .into_iter()
                    .map(|c| {
                        serde_json::json!({
                            "call_id": c.call_id,
                            "agent": c.agent,
                            "elapsed_sec": c.elapsed_sec,
                            "background": c.background,
                            "task_id": c.task_id,
                        })
                    })
                    .collect();
                return serde_json::json!({ "live": live }).to_string();
            }
        };

        match self.runtime.cancel(call_id).await {
            CancelOutcome::Cancelled {
                call_id,
                agent,
                elapsed_sec,
            } => serde_json::json!({
                "status": "cancelled",
                "call_id": call_id,
                "agent": agent,
                "elapsed_sec": elapsed_sec,
            })
            .to_string(),
            CancelOutcome::NotFound { call_id } => serde_json::json!({
                "status": "not_found",
                "call_id": call_id,
                "hint": "такого живого вызова нет: он не запущен либо давно завершён",
            })
            .to_string(),
            CancelOutcome::Finished { call_id, status } => serde_json::json!({
                "status": "already_finished",
                "call_id": call_id,
                "call_status": status,
                "hint": "фоновой задачи у этого вызова нет: он уже завершён либо идёт внутри \
                         синхронного запроса — такой вызов отменить нельзя",
            })
            .to_string(),
        }
    }

    #[tool(
        description = "Остановить цепочку работ по задаче одним вызовом: отменить все её живые \
                       ФОНОВЫЕ вызовы и пометить саму задачу статусом cancelled. Это штатный способ \
                       остановить scripts/code_chain.py: снятие процессов Windows оставляло задачи \
                       платформы незакрытыми, а result.json — не записанным. Отменяются только вызовы \
                       с этим task_id; вызовы других задач не задеваются. Отменённое звено закрывается \
                       ошибкой «вызов отменён вручную» (файл-итог тоже дописывается конвертом ошибки), \
                       а сам скрипт цепочки видит отмену статусом задачи между звеньями и в цикле \
                       ожидания wait_agent, пишет result.json со статусом cancelled и завершается \
                       кодом 4. \
                       Ответ: {status:\"cancelled\", task_id, cancelled_calls:[call_id], previous_status}; \
                       задача уже закрыта (completed/failed/cancelled) — \
                       {status:\"already_closed\", task_id, task_status}; задачи нет — \
                       {status:\"not_found\", task_id}. Отменяются ТОЛЬКО фоновые вызовы \
                       (agent_run и invoke_agent с wait_sec): вызов внутри синхронного запроса держит \
                       вызывающего, его остаётся только переждать."
    )]
    pub async fn chain_cancel(&self, Parameters(p): Parameters<ChainCancelParams>) -> String {
        match self.runtime.cancel_task(p.task_id).await {
            Ok(TaskCancelOutcome::Cancelled {
                task_id,
                cancelled_calls,
                previous_status,
                status,
            }) => serde_json::json!({
                "status": status,
                "task_id": task_id,
                "cancelled_calls": cancelled_calls,
                "previous_status": previous_status,
            })
            .to_string(),
            Ok(TaskCancelOutcome::AlreadyClosed { task_id, status }) => serde_json::json!({
                "status": "already_closed",
                "task_id": task_id,
                "task_status": status,
            })
            .to_string(),
            Ok(TaskCancelOutcome::NotFound { task_id }) => serde_json::json!({
                "status": "not_found",
                "task_id": task_id,
            })
            .to_string(),
            Err(e) => err_json(&format!("{e}")),
        }
    }

    #[tool(
        description = "Подготовить службу к остановке: закрыть приём НОВЫХ вызовов и дождаться, пока \
                       идущие дойдут. Зовут перед остановкой или перезапуском службы: супервизор гасит \
                       процесс сразу (TerminateProcess) и обрывает идущие вызовы — а этот инструмент \
                       даёт им закончить. После него invoke_agent любого режима и agent_run в режиме \
                       пуска отклоняются ошибкой ДО любой работы (строка вызова не создаётся); \
                       исключение — дочерний вызов идущего сейчас вызова (parent_call_id есть в списке \
                       живых): иначе идущий оркестратор упал бы на полудороге. Всё прочее работает как \
                       обычно: проверка agent_run по call_id, wait_agent, agent_cancel, история, \
                       доска, файловые инструменты, health. \
                       Ответ: {status:\"ready\"|\"draining\"|\"accepting\", draining, draining_sec, \
                       preparing, finalizing, live:[{call_id, \
                       agent, elapsed_sec, background}], hint}. status=ready — идущих вызовов нет, \
                       службу можно останавливать; иначе повторите вызов позже, отмените фоновые \
                       через agent_cancel (background=true) или отмените подготовку abort=true. \
                       wait_sec — сколько секунд ждать завершения идущих перед ответом (0..55, по \
                       умолчанию 0 — ответить сразу). abort=true — отменить подготовку и снова \
                       открыть приём: ответ {status:\"accepting\"}. Повторный вызов безопасен, приём \
                       остаётся закрытым до abort или перезапуска службы."
    )]
    pub async fn prepare_shutdown(
        &self,
        Parameters(p): Parameters<PrepareShutdownParams>,
    ) -> String {
        if p.abort.unwrap_or(false) {
            self.runtime.abort_drain();
            return serde_json::json!({ "status": "accepting", "draining": false }).to_string();
        }

        self.runtime.begin_drain();
        let st = self.runtime.wait_drained(p.wait_sec.unwrap_or(0)).await;
        prepare_shutdown_response(&st)
    }

    #[tool(
        description = "Перечитать главный конфиг службы без перезапуска и применить изменившееся. \
                       Применяются все секции, кроме [server] и базы/журнала в [storage] \
                       (log_dir, sqlite_path, task_store_dsn, task_store_pool) — о них ответ \
                       сообщает в restart_required. Файл .env перечитывается в собственную карту \
                       службы: ключи из api_key_env берутся из неё, окружение процесса главнее \
                       и не меняется. Идущие вызовы доживают на прежнем наборе провайдеров, новые \
                       берут новый. Конфиг с ошибкой разбора не применяется целиком. Ответ: \
                       {applied:[...], restart_required:[...], errors:[...]}. Тот же разбор \
                       запускает наблюдатель файла конфига при его сохранении (если \
                       hot_reload=true)."
    )]
    pub async fn config_reload(&self, Parameters(_): Parameters<EmptyParams>) -> String {
        serde_json::to_string(&self.reloader.reload().await).unwrap_or_else(|_| "{}".to_string())
    }

    #[tool(
        description = "История вызовов агентов из таблицы agent_calls. Возвращает JSON-массив записей \
                       (id, agent_name, variant, model_used, provider, tokens_in/out, cost_usd, latency_ms, \
                       cached, created_at, error, instance — экземпляр службы, пусто у вызова старой \
                       сборки); cost_usd = null, если цена модели неизвестна. Параметры: agent (фильтр по имени), since (unixepoch — \
                       минимальная дата создания), limit (1..1000, дефолт 50)."
    )]
    pub async fn agent_history(&self, Parameters(p): Parameters<AgentHistoryParams>) -> String {
        let limit = p.limit.unwrap_or(50);
        match self.runtime.history(p.agent, p.since, limit).await {
            Ok(rows) => serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".to_string()),
            Err(e) => serde_json::json!({"error": format!("{e}")}).to_string(),
        }
    }

    #[tool(
        description = "Загрузить полное тело навыка (SKILL.md) по имени из библиотеки навыков (Postgres). \
                       Вызывается когда в индексе skills_index найден подходящий навык. \
                       Возвращает markdown-тело либо {\"error\":\"...\"}."
    )]
    pub async fn skill_load(&self, Parameters(p): Parameters<SkillLoadParams>) -> String {
        match self.runtime.skill_load(&p.name).await {
            Some(body) => body,
            None => {
                serde_json::json!({"error": format!("навык '{}' не найден", p.name)}).to_string()
            }
        }
    }

    #[tool(
        description = "Создать корневую задачу в PG task-store (внешнее хранилище контекста задач). \
                       Поля: task_kind, goal (якорь цели), target_base, working_dir, sandbox_path, \
                       external_task_id, root_call_id — все опциональны. Статус задачи = running. \
                       Возвращает {\"task_id\": N} либо {\"error\": \"...\"}."
    )]
    pub async fn task_create(&self, Parameters(p): Parameters<TaskCreateParams>) -> String {
        let t = crate::store::NewTask {
            root_call_id: p.root_call_id,
            external_task_id: p.external_task_id,
            task_kind: p.task_kind,
            goal: p.goal,
            target_base: p.target_base,
            working_dir: p.working_dir,
            sandbox_path: p.sandbox_path,
        };
        match self.runtime.task_create(t).await {
            Ok(id) => serde_json::json!({ "task_id": id }).to_string(),
            Err(e) => serde_json::json!({ "error": format!("{e}") }).to_string(),
        }
    }

    #[tool(
        description = "Записать/обновить артефакт доски задачи (upsert по task_id+key). kind: \
                       metadata|query|bsl_module|form_def|build_path|review|deliverable. \
                       depends_on — id артефактов-зависимостей (рёбра DAG). \
                       Возвращает {\"artifact_id\": N} либо {\"error\": \"...\"}."
    )]
    pub async fn artifact_write(&self, Parameters(p): Parameters<ArtifactWriteParams>) -> String {
        let deps = p.depends_on.unwrap_or_default();
        match self
            .runtime
            .task_write_artifact(
                p.task_id,
                &p.kind,
                &p.key,
                p.content.as_deref(),
                p.summary.as_deref(),
                p.producer_agent.as_deref(),
                p.producer_call_id,
                &deps,
            )
            .await
        {
            Ok(id) => serde_json::json!({ "artifact_id": id }).to_string(),
            Err(e) => serde_json::json!({ "error": format!("{e}") }).to_string(),
        }
    }

    #[tool(
        description = "Прочитать артефакты задачи из task-store (опц. фильтр kinds). \
                       Возвращает JSON-массив {id, kind, key, content, summary, producer_agent, \
                       producer_call_id, depends_on, status} либо {\"error\": \"...\"}."
    )]
    pub async fn artifact_read(&self, Parameters(p): Parameters<ArtifactReadParams>) -> String {
        match self
            .runtime
            .task_read_artifacts(p.task_id, p.kinds.as_deref())
            .await
        {
            Ok(arts) => serde_json::to_string_pretty(&arts).unwrap_or_else(|_| "[]".to_string()),
            Err(e) => serde_json::json!({ "error": format!("{e}") }).to_string(),
        }
    }

    #[tool(
        description = "Сменить статус задачи в task-store: running | needs_input | completed | failed | \
                       cancelled. Статус `cancelled` (остановлено вручную) из закрытых обратно не \
                       открывается — для этого заводится новая задача. \
                       Для completed/failed/cancelled выставляется finished_at. \
                       Возвращает {\"ok\": true} либо {\"error\": \"...\"}."
    )]
    pub async fn task_set_status(&self, Parameters(p): Parameters<TaskStatusParams>) -> String {
        match self.runtime.task_set_status(p.task_id, &p.status).await {
            Ok(()) => serde_json::json!({ "ok": true }).to_string(),
            Err(e) => serde_json::json!({ "error": format!("{e}") }).to_string(),
        }
    }

    #[tool(
        description = "Прочитать состояние задачи из task-store: {task_id, status}. Нужен тому, кто \
                       ведёт работу по задаче (цепочка code_chain.py) — по status=\"cancelled\" он \
                       видит остановку вручную (chain_cancel) и сворачивается сам. Задачи нет — \
                       {status:\"not_found\", task_id, hint}; отказ хранилища — {\"error\": \"...\"}."
    )]
    pub async fn task_get(&self, Parameters(p): Parameters<TaskGetParams>) -> String {
        match self.runtime.task_status(p.task_id).await {
            Ok(Some(status)) => {
                serde_json::json!({ "task_id": p.task_id, "status": status }).to_string()
            }
            Ok(None) => serde_json::json!({
                "status": "not_found",
                "task_id": p.task_id,
                "hint": "задачи с таким id в task-store нет: проверьте task_id",
            })
            .to_string(),
            Err(e) => serde_json::json!({ "error": format!("{e}") }).to_string(),
        }
    }

    #[tool(
        description = "Добавить событие в журнал задачи (durable event-stream task-store). event_type: \
                       user_input | agent_start | agent_done | artifact_written | error | status_change | \
                       chain_cancelled. \
                       seq присваивается автоматически. Возвращает {\"ok\": true} либо {\"error\": \"...\"}."
    )]
    pub async fn event_append(&self, Parameters(p): Parameters<EventAppendParams>) -> String {
        match self
            .runtime
            .task_append_event(
                p.task_id,
                &p.event_type,
                p.agent.as_deref(),
                p.call_id,
                p.payload_json.as_deref(),
            )
            .await
        {
            Ok(()) => serde_json::json!({ "ok": true }).to_string(),
            Err(e) => serde_json::json!({ "error": format!("{e}") }).to_string(),
        }
    }

    // ── Файловые инструменты (fs_*) ────────────────────────────────────────
    // Дают файловые операции через MCP, поэтому доступны на ЛЮБОМ провайдере
    // (deepseek/mimo), а не только на claude-cli со встроенными Write/Bash.
    // Все пути обязаны быть абсолютными и лежать внутри разрешённых корней
    // (config [fs].allowed_roots) — защита от записи в системные каталоги.

    #[tool(
        description = "Записать файл (UTF-8), автоматически создавая родительские каталоги. \
                       Стиль переводов строк существующего файла сохраняется. \
                       path — АБСОЛЮТНЫЙ путь внутри разрешённого корня (config [fs]). \
                       Возвращает {ok, path, bytes} либо {error}."
    )]
    pub async fn fs_write_file(
        &self,
        extensions: rmcp::model::Extensions,
        Parameters(p): Parameters<FsWriteParams>,
    ) -> String {
        let (scope_dir, roots) = match self.fs_access(&extensions, p.scope_dir.as_deref()) {
            Ok(x) => x,
            Err(e) => return err_json(&e),
        };
        let scope_dir_text = scope_dir.as_ref().map(|path| path.to_string_lossy());
        let service_paths = self
            .service_paths
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let path = match fs_safe_service_path(
            &roots,
            &service_paths,
            &p.path,
            scope_dir_text.as_deref(),
            true,
        ) {
            Ok(x) => x,
            Err(e) => return err_json(&e),
        };
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return err_json(&format!("create_dir_all: {e}"));
            }
        }
        match fs_write_apply(&path, &p.content) {
            Ok(bytes) => {
                serde_json::json!({"ok": true, "path": p.path, "bytes": bytes}).to_string()
            }
            Err(e) => err_json(&format!("write: {e}")),
        }
    }

    #[tool(
        description = "Прочитать текстовый файл (UTF-8). path — АБСОЛЮТНЫЙ путь внутри разрешённого \
                       корня. Возвращает {ok, path, content} либо {error}."
    )]
    pub async fn fs_read_file(
        &self,
        extensions: rmcp::model::Extensions,
        Parameters(p): Parameters<FsPathParams>,
    ) -> String {
        let (scope_dir, roots) = match self.fs_access(&extensions, p.scope_dir.as_deref()) {
            Ok(x) => x,
            Err(e) => return err_json(&e),
        };
        let scope_dir_text = scope_dir.as_ref().map(|path| path.to_string_lossy());
        let service_paths = self
            .service_paths
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let path = match fs_safe_service_path(
            &roots,
            &service_paths,
            &p.path,
            scope_dir_text.as_deref(),
            false,
        ) {
            Ok(x) => x,
            Err(e) => return err_json(&e),
        };
        let scope = self.request_call_scope(&extensions).ok().flatten();
        let guard = self
            .read_guard
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if read_guard_applies(scope.as_ref(), guard.as_ref()) {
            match ask_read_guard(
                guard.as_ref().expect("гард проверен выше"),
                &p.path,
                scope_dir.as_deref(),
                READ_GUARD_TIMEOUT,
            )
            .await
            {
                GuardVerdict::Deny(reason) => return err_json(&reason),
                GuardVerdict::Pass => {}
                GuardVerdict::Broken(e) => {
                    tracing::warn!(
                        error = %e,
                        path = %p.path,
                        "гард индекса не ответил, читаю без него"
                    );
                }
            }
        }
        match fs_read_file_content(&path) {
            Ok(content) => {
                serde_json::json!({"ok": true, "path": p.path, "content": content}).to_string()
            }
            Err(e) => err_json(&e),
        }
    }

    #[tool(
        description = "Создать каталог (рекурсивно, mkdir -p). path — АБСОЛЮТНЫЙ путь внутри \
                       разрешённого корня. Возвращает {ok, path} либо {error}."
    )]
    pub async fn fs_mkdir(
        &self,
        extensions: rmcp::model::Extensions,
        Parameters(p): Parameters<FsPathParams>,
    ) -> String {
        let (scope_dir, roots) = match self.fs_access(&extensions, p.scope_dir.as_deref()) {
            Ok(x) => x,
            Err(e) => return err_json(&e),
        };
        let scope_dir_text = scope_dir.as_ref().map(|path| path.to_string_lossy());
        let service_paths = self
            .service_paths
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let path = match fs_safe_service_path(
            &roots,
            &service_paths,
            &p.path,
            scope_dir_text.as_deref(),
            true,
        ) {
            Ok(x) => x,
            Err(e) => return err_json(&e),
        };
        match std::fs::create_dir_all(&path) {
            Ok(()) => serde_json::json!({"ok": true, "path": p.path}).to_string(),
            Err(e) => err_json(&format!("mkdir: {e}")),
        }
    }

    #[tool(
        description = "Список каталога. path — АБСОЛЮТНЫЙ путь внутри разрешённого корня. \
                       recursive=true — рекурсивно. name всегда задаётся относительно path; ссылки \
                       отмечаются is_symlink и не обходятся. Возвращает \
                       {ok, path, entries:[{name,is_dir,is_symlink}], truncated} либо {error}."
    )]
    pub async fn fs_list_dir(
        &self,
        extensions: rmcp::model::Extensions,
        Parameters(p): Parameters<FsListParams>,
    ) -> String {
        let (scope_dir, roots) = match self.fs_access(&extensions, p.scope_dir.as_deref()) {
            Ok(x) => x,
            Err(e) => return err_json(&e),
        };
        let scope_dir_text = scope_dir.as_ref().map(|path| path.to_string_lossy());
        let service_paths = self
            .service_paths
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let path = match fs_safe_service_path(
            &roots,
            &service_paths,
            &p.path,
            scope_dir_text.as_deref(),
            false,
        ) {
            Ok(x) => x,
            Err(e) => return err_json(&e),
        };
        match fs_list_dir_entries(&path, p.recursive) {
            Ok((entries, truncated)) => serde_json::json!({
                "ok": true,
                "path": p.path,
                "entries": entries,
                "truncated": truncated,
            })
            .to_string(),
            Err(e) => err_json(&e),
        }
    }

    #[tool(
        description = "Заменить кусок текста в СУЩЕСТВУЮЩЕМ файле (UTF-8) точечно, не переписывая \
                       файл целиком. path — АБСОЛЮТНЫЙ путь внутри разрешённого корня. old_string — \
                       заменяемый фрагмент, new_string — чем заменить. По умолчанию требуется РОВНО \
                       одно вхождение old_string в файле; если вхождений несколько — либо сузьте \
                       old_string до уникального контекста, либо укажите replace_all=true, чтобы \
                       заменить все. Для создания нового файла или полной перезаписи используйте \
                       fs_write_file. Стиль переводов строк существующего файла сохраняется. \
                       Возвращает {ok, path, replaced, bytes, newline_tolerant} либо {error}."
    )]
    pub async fn fs_edit_file(
        &self,
        extensions: rmcp::model::Extensions,
        Parameters(p): Parameters<FsEditParams>,
    ) -> String {
        let (scope_dir, roots) = match self.fs_access(&extensions, p.scope_dir.as_deref()) {
            Ok(x) => x,
            Err(e) => return err_json(&e),
        };
        let scope_dir_text = scope_dir.as_ref().map(|path| path.to_string_lossy());
        let service_paths = self
            .service_paths
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let path = match fs_safe_service_path(
            &roots,
            &service_paths,
            &p.path,
            scope_dir_text.as_deref(),
            true,
        ) {
            Ok(x) => x,
            Err(e) => return err_json(&e),
        };
        match fs_edit_apply(&path, &p.old_string, &p.new_string, p.replace_all) {
            Ok((replaced, bytes, newline_tolerant)) => serde_json::json!({
                "ok": true,
                "path": p.path,
                "replaced": replaced,
                "bytes": bytes,
                "newline_tolerant": newline_tolerant
            })
            .to_string(),
            Err(e) => err_json(&e),
        }
    }
}

// ── fs-инструменты: хелперы безопасности пути ───────────────────────────────

fn err_json(msg: &str) -> String {
    serde_json::json!({ "error": msg }).to_string()
}

/// Путь для выдачи клиенту: только прямые слэши. PathBuf::join на Windows
/// ставит обратный, и в JSON он ещё и экранируется — получается
/// `C:/agents-mcp/runs\\10114-mock-agent.json`, которое клиент потом
/// подставляет в чтение файла и в команды. Windows принимает прямые слэши
/// везде, поэтому отдаём единый вид.
fn path_for_client(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Скрыть значения переменных окружения во встроенном JSON `mcp_config`.
/// При негодном JSON безопаснее не возвращать строку целиком: она тоже может
/// содержать секрет, который разобрать и адресно замаскировать невозможно.
fn redact_mcp_config_env(config: &mut Value) {
    let Some(raw) = config
        .get_mut("execution")
        .and_then(Value::as_object_mut)
        .and_then(|execution| execution.get_mut("mcp_config"))
    else {
        return;
    };
    let Some(text) = raw.as_str() else {
        return;
    };
    let Ok(mut mcp_config) = serde_json::from_str::<Value>(text) else {
        *raw = Value::String("***".to_string());
        return;
    };
    redact_env_values(&mut mcp_config);
    *raw = Value::String(serde_json::to_string(&mcp_config).unwrap_or_else(|_| "***".to_string()));
}

fn redact_env_values(value: &mut Value) {
    match value {
        Value::Object(object) => {
            if let Some(Value::Object(env)) = object.get_mut("env") {
                for value in env.values_mut() {
                    *value = Value::String("***".to_string());
                }
            }
            for value in object.values_mut() {
                redact_env_values(value);
            }
        }
        Value::Array(items) => {
            for item in items {
                redact_env_values(item);
            }
        }
        _ => {}
    }
}

/// Проверить, что путь абсолютный, без traversal `..`, и лежит внутри одного из
/// разрешённых корней. Если задан `scope_dir` (рабочий каталог ЭТОГО вызова
/// агента — подставляется циклом провайдера из cli_hints.cwd, а не самой
/// моделью), путь ДОПОЛНИТЕЛЬНО обязан лежать внутри него: оба условия
/// обязательны одновременно, проверка общих корней не отменяется.
/// Возвращает PathBuf для операции либо текст ошибки.
fn fs_safe_path(roots: &[PathBuf], raw: &str, scope_dir: Option<&str>) -> Result<PathBuf, String> {
    let p = Path::new(raw);
    if !p.is_absolute() {
        return Err(format!("путь должен быть абсолютным: {raw}"));
    }
    if p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(format!("путь содержит '..' (traversal запрещён): {raw}"));
    }
    if has_current_dir_component(raw) {
        return Err(format!("путь содержит '.' (traversal запрещён): {raw}"));
    }
    if roots.is_empty() {
        return Err("файловые инструменты выключены: config [fs].allowed_roots пуст".to_string());
    }
    let resolved = canonicalize_with_missing(p)?;
    let canonical_roots: Vec<PathBuf> = roots
        .iter()
        .filter_map(|root| std::fs::canonicalize(root).ok())
        .collect();
    let inside = canonical_roots
        .iter()
        .any(|root| resolved.starts_with(root));
    if !inside {
        return Err(format!(
            "путь вне разрешённых корней (config [fs].allowed_roots): {raw}"
        ));
    }
    if let Some(scope) = scope_dir.filter(|s| !s.is_empty()) {
        let sp = Path::new(scope);
        let bad_scope = !sp.is_absolute()
            || sp.components().any(|c| matches!(c, Component::ParentDir))
            || has_current_dir_component(scope)
            || !scope_has_directory_component(sp);
        if bad_scope {
            return Err(format!("негодный рабочий каталог этого вызова: {scope}"));
        }
        let scope_resolved = canonicalize_with_missing(sp)
            .map_err(|e| format!("негодный рабочий каталог этого вызова: {scope} ({e})"))?;
        let scoped_ok = resolved.starts_with(&scope_resolved);
        if !scoped_ok {
            return Err(format!(
                "путь вне рабочего каталога этого вызова ({scope}): {raw}"
            ));
        }
    }
    Ok(without_verbatim_disk_prefix(resolved))
}

/// Проверка служебных путей идёт сразу после общей канонической проверки П1.
/// `runs_dir` доступен на чтение, но запись туда выполняет только сама служба.
fn fs_safe_service_path(
    roots: &[PathBuf],
    service_paths: &crate::reload::ServicePaths,
    raw: &str,
    scope_dir: Option<&str>,
    write: bool,
) -> Result<PathBuf, String> {
    let resolved = fs_safe_path(roots, raw, scope_dir)?;
    for entry in &service_paths.entries {
        if entry.write_only && !write {
            continue;
        }
        let protected = canonicalize_with_missing(&entry.path)
            .map(without_verbatim_disk_prefix)
            .map_err(|_| "путь относится к служебным файлам agents-mcp".to_string())?;
        if resolved.starts_with(protected) {
            return Err("путь относится к служебным файлам agents-mcp".to_string());
        }
    }
    Ok(resolved)
}

fn scope_has_directory_component(path: &Path) -> bool {
    path.components().any(|c| matches!(c, Component::Normal(_)))
}

fn has_current_dir_component(raw: &str) -> bool {
    if cfg!(windows) {
        raw.split(['/', '\\']).any(|component| component == ".")
    } else {
        raw.split('/').any(|component| component == ".")
    }
}

/// Канонизировать ближайшего существующего предка и вернуть путь к нему с
/// добавленными несуществующими обычными компонентами.
pub(crate) fn canonicalize_with_missing(path: &Path) -> Result<PathBuf, String> {
    let mut current = path.to_path_buf();
    let mut missing = Vec::new();
    loop {
        match std::fs::symlink_metadata(&current) {
            Ok(_) => {
                let mut resolved = std::fs::canonicalize(&current)
                    .map_err(|e| format!("не удалось канонизировать {}: {e}", current.display()))?;
                for component in missing.iter().rev() {
                    resolved.push(component);
                }
                return Ok(resolved);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let component = current
                    .components()
                    .next_back()
                    .ok_or_else(|| format!("не найден существующий предок: {}", path.display()))?;
                let Component::Normal(name) = component else {
                    return Err(format!(
                        "несуществующая часть пути содержит не обычное имя: {}",
                        path.display()
                    ));
                };
                missing.push(name.to_os_string());
                if !current.pop() {
                    return Err(format!("не найден существующий предок: {}", path.display()));
                }
            }
            Err(e) => {
                return Err(format!(
                    "не удалось проверить существующий предок {}: {e}",
                    current.display()
                ));
            }
        }
    }
}

fn without_verbatim_disk_prefix(path: PathBuf) -> PathBuf {
    #[cfg(windows)]
    {
        use std::path::Prefix;

        if let Some(Component::Prefix(prefix)) = path.components().next() {
            if let Prefix::VerbatimDisk(drive) = prefix.kind() {
                let mut plain = PathBuf::from(format!("{}:\\", drive as char));
                for component in path.components().skip(2) {
                    plain.push(component.as_os_str());
                }
                return plain;
            }
        }
    }
    path
}

/// Пользовательский result_path обязан обозначать файл в уже существующем
/// каталоге. Сам каталог уже входит в канонически проверенный путь.
fn fs_safe_result_path(
    roots: &[PathBuf],
    service_paths: &crate::reload::ServicePaths,
    raw: &str,
) -> Result<PathBuf, String> {
    let path = fs_safe_service_path(roots, service_paths, raw, None, true)?;
    if path.is_dir() {
        return Err(format!(
            "result_path должен быть путём файла, а не каталога: {raw}"
        ));
    }
    let parent = path
        .parent()
        .ok_or_else(|| format!("у result_path нет родительского каталога: {raw}"))?;
    let meta = std::fs::metadata(parent).map_err(|e| {
        format!(
            "родительский каталог result_path не существует: {} ({e})",
            parent.display()
        )
    })?;
    if !meta.is_dir() {
        return Err(format!(
            "родитель result_path не является каталогом: {}",
            parent.display()
        ));
    }
    Ok(path)
}

fn fs_list_dir_entries(path: &Path, recursive: bool) -> Result<(Vec<Value>, bool), String> {
    let mut entries = Vec::new();
    let mut truncated = false;
    let mut stack = vec![(path.to_path_buf(), 0usize)];

    'directories: while let Some((dir, depth)) = stack.pop() {
        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(e) if depth > 0 => {
                if recursive && entries.len() >= MAX_FS_LIST_ENTRIES {
                    truncated = true;
                    break 'directories;
                }
                let name = path_for_client(dir.strip_prefix(path).unwrap_or(&dir));
                entries.push(serde_json::json!({"name": name, "error": format!("read_dir: {e}")}));
                continue;
            }
            Err(e) => return Err(format!("read_dir: {e}")),
        };
        for item in rd {
            if recursive && entries.len() >= MAX_FS_LIST_ENTRIES {
                truncated = true;
                break 'directories;
            }
            let entry = match item {
                Ok(entry) => entry,
                Err(e) => {
                    entries.push(serde_json::json!({
                        "name": path_for_client(dir.strip_prefix(path).unwrap_or(&dir)),
                        "error": format!("read_dir entry: {e}"),
                    }));
                    continue;
                }
            };
            let entry_path = entry.path();
            let name = path_for_client(entry_path.strip_prefix(path).unwrap_or(&entry_path));
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(e) => {
                    entries.push(
                        serde_json::json!({"name": name, "error": format!("file_type: {e}")}),
                    );
                    continue;
                }
            };
            let is_symlink = file_type.is_symlink();
            let is_dir = file_type.is_dir();
            entries.push(serde_json::json!({
                "name": name,
                "is_dir": is_dir,
                "is_symlink": is_symlink,
            }));
            if recursive && is_dir && !is_symlink {
                if depth < MAX_FS_LIST_DEPTH {
                    stack.push((entry_path, depth + 1));
                } else {
                    truncated = true;
                }
            }
        }
        if !recursive {
            break;
        }
    }
    Ok((entries, truncated))
}

fn fs_read_file_content(path: &Path) -> Result<String, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("metadata: {e}"))?;
    if meta.len() > MAX_EDIT_FILE_SIZE {
        return Err(format!(
            "файл слишком большой для чтения: {} байт (лимит {MAX_EDIT_FILE_SIZE}): {}",
            meta.len(),
            path.display()
        ));
    }
    std::fs::read_to_string(path).map_err(|e| format!("read: {e}"))
}

/// Нормализовать переводы строк (`\r\n` и одиночный `\r` → `\n`) и построить
/// карту смещений: `map[k]` — смещение в ИСХОДНОМ тексте, где НАЧИНАЕТСЯ
/// последовательность, давшая k-й байт нормализованного текста. Длина карты
/// на 1 больше длины нормализованного текста; последний элемент — длина
/// исходного текста (нужен для правой границы вхождения, кончающегося в
/// конце файла). `\r` и `\n` — однобайтовые ASCII-символы и никогда не входят
/// в многобайтовую UTF-8 последовательность, поэтому побайтовый разбор не
/// портит кодировку.
fn normalize_with_map(s: &str) -> (String, Vec<usize>) {
    let bytes = s.as_bytes();
    let mut normalized: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut map: Vec<usize> = Vec::with_capacity(bytes.len() + 1);
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\r' {
            map.push(i);
            normalized.push(b'\n');
            if i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                i += 2;
            } else {
                i += 1;
            }
        } else {
            map.push(i);
            normalized.push(b);
            i += 1;
        }
    }
    map.push(bytes.len());
    let normalized = String::from_utf8(normalized)
        .expect("нормализация переводов строк не меняет корректность UTF-8");
    (normalized, map)
}

/// Привести переводы строк к CRLF: сначала нормализовать (`\r\n`/`\r` → `\n`),
/// затем каждый `\n` превратить в `\r\n`.
fn normalize_newlines_to_crlf(s: &str) -> String {
    normalize_with_map(s).0.replace('\n', "\r\n")
}

/// Записать `content` в `path`, сохранив преобладающий стиль переводов строк
/// существующего файла. Для определения стиля читаются только первые 64 КиБ;
/// ошибка чтения сохраняет прежнее поведение — содержимое записывается как есть.
fn fs_write_apply(path: &std::path::Path, content: &str) -> Result<usize, String> {
    let adapted = std::fs::File::open(path)
        .ok()
        .and_then(|file| {
            let mut sample = Vec::new();
            let mut limited = std::io::Read::take(file, 64 * 1024);
            std::io::Read::read_to_end(&mut limited, &mut sample)
                .ok()
                .map(|_| sample)
        })
        .and_then(|sample| {
            let mut crlf = 0usize;
            let mut lf = 0usize;
            for (i, &byte) in sample.iter().enumerate() {
                if byte == b'\n' {
                    if i > 0 && sample[i - 1] == b'\r' {
                        crlf += 1;
                    } else {
                        lf += 1;
                    }
                }
            }
            if crlf > lf {
                Some(normalize_newlines_to_crlf(content))
            } else if crlf + lf > 0 {
                Some(normalize_with_map(content).0)
            } else {
                None
            }
        })
        .unwrap_or_else(|| content.to_string());

    std::fs::write(path, adapted.as_bytes()).map_err(|e| e.to_string())?;
    Ok(adapted.len())
}

/// Применить точечную замену `old_string` → `new_string` в файле `path`
/// (путь уже проверен fs_safe_path). Вынесена отдельной свободной функцией —
/// без обращения к `&self`/fs_roots/scope_dir, — чтобы тестировать логику
/// подсчёта вхождений и замены без сборки полного AgentsMcpServer (сам
/// fs_edit_file async и требует зарегистрированного сервера).
/// Сначала идёт точный поиск; если он не дал ни одного вхождения, включается
/// запасной путь, терпимый к различиям в переводах строк (см. тело функции).
/// Возвращает (число_замен, размер_файла_после_записи_в_байтах,
/// сработал_ли_запасной_путь) либо текст ошибки.
fn fs_edit_apply(
    path: &std::path::Path,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> Result<(usize, usize, bool), String> {
    if old_string.is_empty() {
        return Err("old_string пуст: нечего искать".to_string());
    }
    if old_string == new_string {
        return Err("old_string и new_string совпадают: замена ничего не изменит".to_string());
    }
    let meta =
        std::fs::metadata(path).map_err(|e| format!("файл не найден: {} ({e})", path.display()))?;
    if meta.len() > MAX_EDIT_FILE_SIZE {
        return Err(format!(
            "файл слишком большой для точечной правки: {} байт (лимит {MAX_EDIT_FILE_SIZE}): {}",
            meta.len(),
            path.display()
        ));
    }
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("файл не читается как UTF-8: {} ({e})", path.display()))?;

    // Шаг 1 — точный поиск. Стиль перевода строк берётся из совпавшего
    // old_string, а для однострочного фрагмента — из всего файла.
    let count = content.matches(old_string).count();
    if count > 0 {
        if count > 1 && !replace_all {
            return Err(format!(
                "фрагмент встречается {count} раз(а) в файле {} — уточните old_string до уникального \
                 контекста либо укажите replace_all: true, чтобы заменить все вхождения",
                path.display()
            ));
        }
        let limit = if replace_all { count } else { 1 };
        let adapted = if old_string.contains('\n') || old_string.contains('\r') {
            if old_string.contains("\r\n") {
                normalize_newlines_to_crlf(new_string)
            } else {
                normalize_with_map(new_string).0
            }
        } else {
            let mut crlf = 0usize;
            let mut lf = 0usize;
            for (i, &byte) in content.as_bytes().iter().enumerate() {
                if byte == b'\n' {
                    if i > 0 && content.as_bytes()[i - 1] == b'\r' {
                        crlf += 1;
                    } else {
                        lf += 1;
                    }
                }
            }
            if crlf > lf {
                normalize_newlines_to_crlf(new_string)
            } else if crlf + lf > 0 {
                normalize_with_map(new_string).0
            } else {
                new_string.to_string()
            }
        };
        let new_content = content.replacen(old_string, &adapted, limit);
        std::fs::write(path, new_content.as_bytes()).map_err(|e| format!("write: {e}"))?;
        return Ok((limit, new_content.len(), false));
    }

    // Шаг 2 — запасной путь, терпимый к различиям в переводах строк.
    // Включается только когда old_string содержит хотя бы один перевод
    // строки — иначе различаться в переводах строк нечему, и это тот же
    // самый "не найден", что и раньше.
    if !old_string.contains('\n') && !old_string.contains('\r') {
        return Err(format!("фрагмент не найден в файле: {}", path.display()));
    }

    let (normalized_content, map) = normalize_with_map(&content);
    let normalized_old = normalize_with_map(old_string).0;

    // Непересекающийся поиск слева направо по нормализованному тексту.
    let mut occurrences: Vec<(usize, usize)> = Vec::new();
    let mut search_from = 0usize;
    while let Some(rel) = normalized_content[search_from..].find(normalized_old.as_str()) {
        let s = search_from + rel;
        let e = s + normalized_old.len();
        occurrences.push((s, e));
        search_from = e;
    }

    if occurrences.is_empty() {
        return Err(format!("фрагмент не найден в файле: {}", path.display()));
    }
    if occurrences.len() > 1 && !replace_all {
        return Err(format!(
            "фрагмент встречается {} раз(а) в файле {} — уточните old_string до уникального \
             контекста либо укажите replace_all: true, чтобы заменить все вхождения (совпадения \
             найдены без учёта различий в переводах строк)",
            occurrences.len(),
            path.display()
        ));
    }

    // Замена: новое содержимое собирается из кусков ИСХОДНОГО содержимого —
    // текст между вхождениями копируется байт в байт, вхождения заменяются.
    let mut new_content = String::with_capacity(content.len());
    let mut last_end = 0usize;
    for &(s, e) in &occurrences {
        let orig_s = map[s];
        let orig_e = map[e];
        new_content.push_str(&content[last_end..orig_s]);
        let excerpt = &content[orig_s..orig_e];
        let adapted = if excerpt.contains("\r\n") {
            normalize_newlines_to_crlf(new_string)
        } else {
            normalize_with_map(new_string).0
        };
        new_content.push_str(&adapted);
        last_end = orig_e;
    }
    new_content.push_str(&content[last_end..]);

    if new_content == content {
        return Err("замена ничего не изменит".to_string());
    }

    std::fs::write(path, new_content.as_bytes()).map_err(|e| format!("write: {e}"))?;
    Ok((occurrences.len(), new_content.len(), true))
}

// ── ServerHandler ──────────────────────────────────────────────────────────

impl ServerHandler for AgentsMcpServer {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        let mut info = rmcp::model::ServerInfo::default();
        info.instructions = Some(
            "MCP-сервер платформы специализированных LLM-агентов: вызов агентов по имени, история, healthcheck.".into(),
        );
        info.capabilities = rmcp::model::ServerCapabilities::builder()
            .enable_tools()
            .build();
        let mut impl_info = rmcp::model::Implementation::default();
        impl_info.name = "agents-mcp".into();
        impl_info.version = env!("CARGO_PKG_VERSION").into();
        info.server_info = impl_info;
        info
    }

    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, rmcp::ErrorData> {
        Ok(rmcp::model::ListToolsResult {
            tools: self.tool_router.list_all(),
            ..Default::default()
        })
    }

    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParams,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        if let Err(e) = self.request_call_scope(&context.extensions) {
            return Ok(rmcp::model::CallToolResult::error(vec![
                rmcp::model::Content::text(e),
            ]));
        }
        let tcc = rmcp::handler::server::tool::ToolCallContext::new(self, request, context);
        self.tool_router.call(tcc).await
    }
}

// ── Router ─────────────────────────────────────────────────────────────────

/// Собрать MCP-службу отдельно от HTTP-обвязки: тот же набор инструментов, но
/// без axum. Нужна транспорту стандартного ввода/вывода, где HTTP не участвует.
pub fn build_mcp_server(
    started_at: chrono::DateTime<chrono::Utc>,
    registry: Arc<Registry>,
    runtime: Arc<Runtime>,
    reloader: Arc<crate::reload::ConfigReloader>,
) -> AgentsMcpServer {
    AgentsMcpServer::new(started_at, registry, runtime, reloader)
}

/// Собрать axum-роутер.
pub fn build_router(
    config: Config,
    started_at: chrono::DateTime<chrono::Utc>,
    registry: Arc<Registry>,
    runtime: Arc<Runtime>,
    reloader: Arc<crate::reload::ConfigReloader>,
) -> Router {
    let allowed_hosts = config.server.allowed_hosts.clone();
    runtime.set_own_mcp_port(config.server.port);
    let mcp = build_mcp_server(started_at, registry.clone(), runtime.clone(), reloader);

    let state = AppState {
        config: Arc::new(config),
        started_at,
        registry,
        runtime,
    };

    let session_manager = Arc::new(NeverSessionManager::default());
    let service_factory = move || Ok(mcp.clone());
    let http_config = StreamableHttpServerConfig::default()
        .with_stateful_mode(false)
        .with_json_response(true)
        .with_allowed_hosts(allowed_hosts);
    let http_service = StreamableHttpService::new(service_factory, session_manager, http_config);

    Router::new()
        .route("/health", get(health_endpoint))
        .with_state(state)
        .nest_service("/mcp", http_service)
}

async fn health_endpoint(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(build_health(state.started_at, state.registry.len(), &state.runtime).await)
}

/// Ответ пробы здоровья: провайдеры и НАСТОЯЩИЙ запрос к БД.
///
/// Проба без обращения к хранилищу отличить здоровую службу от службы с
/// мёртвым соединением не может — 20.08.2026 такая проба 5.4 суток показывала
/// «ok», пока работа стояла. Отказ БД опускает общий статус до `degraded`:
/// служба на ходу, но задачи вести не может.
async fn build_health(
    started_at: chrono::DateTime<chrono::Utc>,
    agents_loaded: usize,
    runtime: &Runtime,
) -> HealthResponse {
    let mut resp = HealthResponse::new(started_at, agents_loaded);
    resp.instance = runtime.instance().to_string();
    // Идёт подготовка к остановке (prepare_shutdown): приём новых вызовов
    // закрыт, но служба на ходу. Отказ базы ниже перекрывает это состояние
    // на "degraded" — он важнее.
    if runtime.is_draining() {
        resp.status = "draining";
    }
    resp.providers = runtime.provider_statuses();
    resp.database = match runtime.pg_health().await {
        Ok(()) => ProviderStatus::ok(),
        Err(e) => {
            resp.status = "degraded";
            ProviderStatus::down(e.to_string())
        }
    };
    resp
}

// ── Тесты fs_safe_path (границы: allowed_roots + scope_dir вызова) ─────────

#[cfg(test)]
mod fs_safe_path_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn prepare_shutdown_reports_accepting_after_foreign_abort() {
        let response = prepare_shutdown_response(&DrainStatus {
            draining: false,
            draining_sec: 0,
            preparing: 1,
            finalizing: 0,
            live: Vec::new(),
        });
        let body: Value = serde_json::from_str(&response).expect("JSON-ответ");
        assert_eq!(body["status"], "accepting");
        assert_eq!(body["draining"], false);
    }

    #[tokio::test]
    async fn health_distinguishes_registered_and_checked_providers() {
        let tree = TempTree::new("health_provider_status");
        std::fs::create_dir_all(tree.0.join("agents")).expect("каталог агентов");
        let store: Arc<dyn crate::store::Store> = Arc::new(
            crate::store::SqliteStore::open(std::path::Path::new(":memory:"))
                .expect("SQLite для health"),
        );
        let registry = Arc::new(Registry::load(tree.0.join("agents")).expect("реестр"));
        let providers: HashMap<String, Arc<dyn crate::providers::LlmProvider>> = HashMap::from([(
            "mock".to_string(),
            Arc::new(crate::providers::mock::MockProvider::new())
                as Arc<dyn crate::providers::LlmProvider>,
        )]);
        let runtime = Runtime::new(
            store,
            registry,
            providers.clone(),
            crate::skills::SkillsClient::new(None),
            Arc::new(std::sync::RwLock::new(
                crate::runtime::ModelOverride::default(),
            )),
            tree.0.join("runs"),
            "test:health".into(),
            120,
        );

        let registered = build_health(chrono::Utc::now(), 0, &runtime).await;
        assert_eq!(registered.status, "ok", "верхний статус не меняется");
        assert_eq!(registered.providers["mock"].status, "registered");

        runtime.set_provider_set(
            providers.clone(),
            HashMap::from([("mock".to_string(), ProviderStatus::ok())]),
        );
        let checked = build_health(chrono::Utc::now(), 0, &runtime).await;
        assert_eq!(checked.status, "ok", "верхний статус не меняется");
        assert_eq!(checked.providers["mock"].status, "ok");

        runtime.set_provider_set(
            providers,
            HashMap::from([("mock".to_string(), ProviderStatus::down("doctor failed"))]),
        );
        let down = build_health(chrono::Utc::now(), 0, &runtime).await;
        assert_eq!(down.status, "ok", "верхний статус не меняется");
        assert_eq!(down.providers["mock"].status, "down");
        assert_eq!(
            down.providers["mock"].message.as_deref(),
            Some("doctor failed")
        );
    }

    struct TempTree(PathBuf);

    impl TempTree {
        fn new(tag: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("agents_mcp_fs_{tag}_{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&path).expect("создать временный каталог");
            Self(path)
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn path_text(path: &Path) -> String {
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn call_key_scope_overrides_supplied_scope_dir() {
        let tree = TempTree::new("call_key_scope");
        let scope = tree.0.join("a");
        let sibling = tree.0.join("b");
        std::fs::create_dir(&scope).unwrap();
        std::fs::create_dir(&sibling).unwrap();
        let trusted = CallScope {
            call_id: 10,
            cwd: Some(scope.clone()),
            allowed_roots: None,
            parent_call_id: None,
            orchestration_depth: 0,
            reads_code_index: false,
        };
        let effective = effective_fs_scope(Some(trusted), Some(&path_text(&sibling)))
            .expect("ключ задаёт рабочий каталог")
            .expect("cwd есть");

        assert!(
            fs_safe_path(
                std::slice::from_ref(&tree.0),
                &path_text(&sibling.join("outside.txt")),
                Some(&path_text(&effective)),
            )
            .is_err(),
            "подложный scope_dir не должен расширять доступ"
        );
        assert!(
            fs_safe_path(
                std::slice::from_ref(&tree.0),
                &path_text(&scope.join("inside.txt")),
                Some(&path_text(&effective)),
            )
            .is_ok(),
            "путь внутри cwd ключа разрешён"
        );
    }

    #[test]
    fn call_key_without_cwd_disables_fs_but_external_client_keeps_old_behavior() {
        let no_cwd = CallScope {
            call_id: 11,
            cwd: None,
            allowed_roots: None,
            parent_call_id: None,
            orchestration_depth: 0,
            reads_code_index: false,
        };
        assert!(effective_fs_scope(Some(no_cwd), Some("C:/подложный")).is_err());
        assert_eq!(effective_fs_scope(None, None).unwrap(), None);
    }

    #[test]
    fn call_key_overrides_fake_child_lineage() {
        let scope = CallScope {
            call_id: 73,
            cwd: None,
            allowed_roots: None,
            parent_call_id: Some(12),
            orchestration_depth: 2,
            reads_code_index: false,
        };
        assert_eq!(
            effective_child_lineage(Some(&scope), Some(999), 0),
            (Some(73), 3)
        );
        assert_eq!(effective_child_lineage(None, Some(999), 4), (Some(999), 4));
    }

    #[test]
    fn agent_roots_replace_service_roots_only_for_keyed_call() {
        let tree = TempTree::new("agent_roots");
        let agent_root = tree.0.join("agent-root");
        let cwd = agent_root.join("work");
        let outside = tree.0.join("outside");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let target = cwd.join("f.txt");
        let trusted = CallScope {
            call_id: 21,
            cwd: Some(cwd.clone()),
            allowed_roots: Some(vec![agent_root.clone()]),
            parent_call_id: None,
            orchestration_depth: 0,
            reads_code_index: false,
        };
        let effective = effective_fs_scope(Some(trusted.clone()), None)
            .expect("ключ задаёт рабочий каталог")
            .expect("cwd есть");

        // С ключом и корнями агента путь внутри рабочего каталога разрешён,
        // хотя общий список службы его не покрывает.
        let roots = effective_fs_roots(Some(&trusted), std::slice::from_ref(&outside));
        assert!(
            fs_safe_path(&roots, &path_text(&target), Some(&path_text(&effective))).is_ok(),
            "корни агента действуют вместо общего списка"
        );

        // Тот же путь без ключа — отказ по общему списку службы.
        assert!(
            fs_safe_path(
                &effective_fs_roots(None, std::slice::from_ref(&outside)),
                &path_text(&target),
                None
            )
            .is_err(),
            "без ключа действует общий список службы"
        );

        // Ключ без корней агента — тоже общий список службы.
        let plain = CallScope {
            call_id: 22,
            cwd: Some(cwd.clone()),
            allowed_roots: None,
            parent_call_id: None,
            orchestration_depth: 0,
            reads_code_index: false,
        };
        assert!(
            fs_safe_path(
                &effective_fs_roots(Some(&plain), std::slice::from_ref(&outside)),
                &path_text(&target),
                Some(&path_text(&effective))
            )
            .is_err(),
            "ключ без allowed_roots не расширяет доступ"
        );
    }

    #[test]
    fn service_path_rejected_even_inside_agent_roots() {
        let tree = TempTree::new("agent_roots_service");
        let agent_root = tree.0.join("agent-root");
        let config_dir = agent_root.join("service");
        std::fs::create_dir_all(&config_dir).unwrap();
        let paths = protected_path("каталог главного конфига", config_dir.clone(), false);

        let err = fs_safe_service_path(
            std::slice::from_ref(&agent_root),
            &paths,
            &path_text(&config_dir.join("agents-mcp.toml")),
            None,
            true,
        )
        .expect_err("служебный путь закрыт и в корнях агента");
        assert_eq!(err, "путь относится к служебным файлам agents-mcp");
    }

    #[cfg(windows)]
    fn create_dir_link(link: &Path, target: &Path) -> Result<(), String> {
        let output = std::process::Command::new("cmd")
            .args(["/c", "mklink", "/J"])
            .arg(link)
            .arg(target)
            .output()
            .map_err(|e| format!("не запустился mklink: {e}"))?;
        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let reason = if stderr.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            };
            Err(format!("mklink завершился с {}: {reason}", output.status))
        }
    }

    #[cfg(unix)]
    fn create_dir_link(link: &Path, target: &Path) -> Result<(), String> {
        std::os::unix::fs::symlink(target, link).map_err(|e| e.to_string())
    }

    #[test]
    fn path_inside_scope_and_roots_ok() {
        let tree = TempTree::new("inside");
        let scope = tree.0.join("proj");
        std::fs::create_dir(&scope).unwrap();
        let path = scope.join("new.txt");
        let res = fs_safe_path(
            std::slice::from_ref(&tree.0),
            &path_text(&path),
            Some(&path_text(&scope)),
        );
        assert!(res.is_ok(), "{res:?}");
    }

    #[test]
    fn path_inside_roots_but_outside_scope_rejected() {
        let tree = TempTree::new("scope_outside");
        let scope = tree.0.join("proj");
        let other = tree.0.join("other");
        std::fs::create_dir(&scope).unwrap();
        std::fs::create_dir(&other).unwrap();
        let path = other.join("file.txt");
        let err = fs_safe_path(
            std::slice::from_ref(&tree.0),
            &path_text(&path),
            Some(&path_text(&scope)),
        )
        .expect_err("путь вне scope_dir должен быть отклонён");
        assert!(err.contains("рабочего каталога"), "{err}");
    }

    #[test]
    fn nonexistent_file_in_existing_subdir_allowed_but_parent_dir_rejected() {
        let tree = TempTree::new("missing");
        let subdir = tree.0.join("subdir");
        std::fs::create_dir(&subdir).unwrap();
        let path = subdir.join("new.txt");
        assert!(fs_safe_path(std::slice::from_ref(&tree.0), &path_text(&path), None).is_ok());

        let traversal = subdir.join("..").join("outside.txt");
        assert!(fs_safe_path(std::slice::from_ref(&tree.0), &path_text(&traversal), None).is_err());

        let current_dir = format!(
            "{}{}.{}new.txt",
            path_text(&subdir),
            std::path::MAIN_SEPARATOR,
            std::path::MAIN_SEPARATOR,
        );
        assert!(fs_safe_path(std::slice::from_ref(&tree.0), &current_dir, None).is_err());
    }

    #[test]
    fn nonexistent_root_allows_nothing() {
        let tree = TempTree::new("missing_root");
        let root = tree.0.join("missing");
        let path = root.join("file.txt");
        assert!(fs_safe_path(&[root], &path_text(&path), None).is_err());
    }

    #[test]
    fn link_outside_root_rejected() {
        let tree = TempTree::new("link_outside");
        let root = tree.0.join("root");
        let outside = tree.0.join("outside");
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(&outside).unwrap();
        let link = root.join("link");
        if let Err(e) = create_dir_link(&link, &outside) {
            eprintln!("тест пропущен: не удалось создать ссылку/junction: {e}");
            return;
        }
        let path = link.join("file.txt");
        assert!(fs_safe_path(&[root], &path_text(&path), None).is_err());
    }

    #[test]
    fn link_to_subdir_inside_root_allowed() {
        let tree = TempTree::new("link_inside");
        let root = tree.0.join("root");
        let target = root.join("target");
        std::fs::create_dir_all(&target).unwrap();
        let link = root.join("link");
        if let Err(e) = create_dir_link(&link, &target) {
            eprintln!("тест пропущен: не удалось создать ссылку/junction: {e}");
            return;
        }
        let path = link.join("new.txt");
        let resolved = fs_safe_path(&[root], &path_text(&path), None).expect("ссылка ведёт внутрь");
        let target = without_verbatim_disk_prefix(std::fs::canonicalize(target).unwrap());
        assert!(resolved.starts_with(target));
    }

    #[cfg(windows)]
    #[test]
    fn safe_paths_do_not_return_verbatim_disk_prefix() {
        let tree = TempTree::new("plain_windows_path");
        let file = tree.0.join("result.json");
        std::fs::write(&file, "{}").unwrap();
        let raw = path_text(&file);

        let safe_path = fs_safe_path(std::slice::from_ref(&tree.0), &raw, None).unwrap();
        assert!(!path_text(&safe_path).starts_with(r"\\?\"));

        let result_path = fs_safe_result_path(
            std::slice::from_ref(&tree.0),
            &crate::reload::ServicePaths::default(),
            &raw,
        )
        .unwrap();
        assert!(!path_text(&result_path).starts_with(r"\\?\"));
    }

    #[cfg(windows)]
    #[test]
    fn unicode_case_folding_distinct_sibling_rejected() {
        let tree = TempTree::new("kelvin");
        let root = tree.0.join("worK");
        let sibling = tree.0.join("wor\u{212a}");
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(&sibling).unwrap();
        let path = sibling.join("file.txt");
        assert!(fs_safe_path(&[root], &path_text(&path), None).is_err());
    }

    #[cfg(windows)]
    #[test]
    fn ascii_case_variation_uses_filesystem_canonical_case() {
        let tree = TempTree::new("ascii_case");
        let root = tree.0.join("Work");
        std::fs::create_dir(&root).unwrap();
        let raw = path_text(&root.join("new.txt")).replace("Work", "work");
        assert!(fs_safe_path(&[root], &raw, None).is_ok());
    }

    #[test]
    fn result_path_equal_to_root_rejected() {
        let tree = TempTree::new("result_root");
        let err = fs_safe_result_path(
            std::slice::from_ref(&tree.0),
            &crate::reload::ServicePaths::default(),
            &path_text(&tree.0),
        )
        .expect_err("каталог нельзя принять как result_path");
        assert!(err.contains("не каталога"), "{err}");
    }

    #[test]
    fn result_path_parent_must_exist() {
        let tree = TempTree::new("result_parent");
        let path = tree.0.join("missing").join("result.json");
        assert!(fs_safe_result_path(
            std::slice::from_ref(&tree.0),
            &crate::reload::ServicePaths::default(),
            &path_text(&path),
        )
        .is_err());
    }

    fn protected_path(
        name: &'static str,
        path: PathBuf,
        write_only: bool,
    ) -> crate::reload::ServicePaths {
        crate::reload::ServicePaths {
            entries: vec![crate::reload::ServicePath {
                name,
                path,
                write_only,
            }],
        }
    }

    #[test]
    fn config_and_env_write_rejected_inside_allowed_root() {
        let tree = TempTree::new("service_config");
        let config_dir = tree.0.join("service");
        std::fs::create_dir(&config_dir).unwrap();
        let paths = protected_path("каталог главного конфига", config_dir.clone(), false);

        for path in [config_dir.join("agents-mcp.toml"), config_dir.join(".env")] {
            let err = fs_safe_service_path(
                std::slice::from_ref(&tree.0),
                &paths,
                &path_text(&path),
                None,
                true,
            )
            .expect_err("служебный файл нельзя перезаписать через fs_write_file");
            assert_eq!(err, "путь относится к служебным файлам agents-mcp");
        }
    }

    #[test]
    fn agents_dir_write_rejected_but_neighbor_allowed() {
        let tree = TempTree::new("agents_dir");
        let agents_dir = tree.0.join("agents");
        let neighbor = tree.0.join("work");
        std::fs::create_dir(&agents_dir).unwrap();
        std::fs::create_dir(&neighbor).unwrap();
        let paths = protected_path("agents_dir", agents_dir.clone(), false);

        let agent_config = agents_dir.join("worker").join("config.toml");
        assert!(fs_safe_service_path(
            std::slice::from_ref(&tree.0),
            &paths,
            &path_text(&agent_config),
            None,
            true,
        )
        .is_err());
        assert!(fs_safe_service_path(
            std::slice::from_ref(&tree.0),
            &paths,
            &path_text(&neighbor.join("result.txt")),
            None,
            true,
        )
        .is_ok());
    }

    #[test]
    fn runs_dir_allows_read_but_rejects_write_and_result_path() {
        let tree = TempTree::new("runs_dir");
        let runs_dir = tree.0.join("runs");
        std::fs::create_dir(&runs_dir).unwrap();
        let paths = protected_path("runs_dir", runs_dir.clone(), true);
        let result = runs_dir.join("result.json");

        assert!(fs_safe_service_path(
            std::slice::from_ref(&tree.0),
            &paths,
            &path_text(&result),
            None,
            false,
        )
        .is_ok());
        assert!(fs_safe_service_path(
            std::slice::from_ref(&tree.0),
            &paths,
            &path_text(&result),
            None,
            true,
        )
        .is_err());
        assert!(
            fs_safe_result_path(std::slice::from_ref(&tree.0), &paths, &path_text(&result))
                .is_err()
        );
    }

    #[test]
    fn linked_path_to_service_directory_rejected() {
        let tree = TempTree::new("service_link");
        let work = tree.0.join("work");
        let config_dir = tree.0.join("service");
        std::fs::create_dir(&work).unwrap();
        std::fs::create_dir(&config_dir).unwrap();
        let link = work.join("config-link");
        create_dir_link(&link, &config_dir).expect("создать ссылку/junction на служебный каталог");
        let paths = protected_path("каталог главного конфига", config_dir, false);
        let err = fs_safe_service_path(
            std::slice::from_ref(&tree.0),
            &paths,
            &path_text(&link.join("agents-mcp.toml")),
            None,
            true,
        )
        .expect_err("канонический служебный путь должен быть запрещён");
        assert_eq!(err, "путь относится к служебным файлам agents-mcp");
    }

    #[test]
    fn mcp_config_env_values_are_redacted() {
        let mut config = serde_json::json!({
            "execution": {
                "mcp_config": r#"{"mcpServers":{"private":{"command":"server","env":{"TOKEN":"secret","MODE":"safe"}}}}"#
            }
        });
        redact_mcp_config_env(&mut config);
        let response = serde_json::to_string(&config).unwrap();
        assert!(!response.contains("secret"));
        let raw = config["execution"]["mcp_config"].as_str().unwrap();
        let mcp: Value = serde_json::from_str(raw).unwrap();
        assert_eq!(mcp["mcpServers"]["private"]["env"]["TOKEN"], "***");
        assert_eq!(mcp["mcpServers"]["private"]["env"]["MODE"], "***");
    }

    #[test]
    fn recursive_list_does_not_follow_link_to_parent() {
        let tree = TempTree::new("list_link");
        let child = tree.0.join("child");
        std::fs::create_dir(&child).unwrap();
        let link = child.join("back");
        if let Err(e) = create_dir_link(&link, &tree.0) {
            eprintln!("тест пропущен: не удалось создать ссылку/junction: {e}");
            return;
        }
        let (entries, truncated) = fs_list_dir_entries(&tree.0, true).unwrap();
        assert!(!truncated);
        assert!(
            entries.len() <= 2,
            "ссылка не должна порождать цикл: {entries:?}"
        );
        let back = entries
            .iter()
            .find(|entry| {
                entry["name"]
                    .as_str()
                    .is_some_and(|name| name.ends_with("back"))
            })
            .expect("ссылка присутствует в выдаче");
        assert_eq!(back["is_symlink"], true);
    }

    #[test]
    fn recursive_list_reports_depth_truncation() {
        let tree = TempTree::new("list_depth");
        let mut current = tree.0.clone();
        for level in 0..=MAX_FS_LIST_DEPTH {
            current = current.join(format!("d{level}"));
            std::fs::create_dir(&current).unwrap();
        }
        let (_, truncated) = fs_list_dir_entries(&tree.0, true).unwrap();
        assert!(truncated);
    }

    #[test]
    fn read_file_over_limit_rejected_before_reading() {
        let tree = TempTree::new("read_limit");
        let path = tree.0.join("large.txt");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(MAX_EDIT_FILE_SIZE + 1).unwrap();
        let err = fs_read_file_content(&path).expect_err("слишком большой файл отклоняется");
        assert!(err.contains(&(MAX_EDIT_FILE_SIZE + 1).to_string()), "{err}");
    }

    #[cfg(windows)]
    #[test]
    fn windows_scope_first_level_allowed_drive_root_rejected() {
        let roots = [PathBuf::from("C:/")];
        assert!(fs_safe_path(&roots, "C:/work/new.txt", Some("C:/work")).is_ok());
        assert!(fs_safe_path(&roots, "C:/work/new.txt", Some("C:/")).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn unix_scope_first_level_allowed_fs_root_rejected() {
        let roots = [PathBuf::from("/")];
        assert!(fs_safe_path(&roots, "/work/new.txt", Some("/work")).is_ok());
        assert!(fs_safe_path(&roots, "/work/new.txt", Some("/")).is_err());
    }
}

#[cfg(test)]
mod fs_edit_apply_tests {
    use super::*;

    /// Уникальный путь во временном каталоге для одного теста — чтобы тесты
    /// не пересекались друг с другом при параллельном запуске.
    fn temp_path(tag: &str) -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("время после эпохи")
            .as_nanos();
        std::env::temp_dir().join(format!("agents_mcp_fs_edit_test_{tag}_{n}.txt"))
    }

    fn write_temp(tag: &str, content: &str) -> PathBuf {
        let path = temp_path(tag);
        std::fs::write(&path, content).expect("write temp file");
        path
    }

    #[test]
    fn single_occurrence_replaced() {
        let path = write_temp("single", "hello world");
        let res = fs_edit_apply(&path, "world", "Rust", false);
        assert!(res.is_ok(), "{res:?}");
        let (replaced, _bytes, newline_tolerant) = res.unwrap();
        assert_eq!(replaced, 1);
        assert!(!newline_tolerant, "точный путь не должен включать запасной");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello Rust");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn multiple_occurrences_without_replace_all_rejected() {
        let path = write_temp("multi_reject", "aa aa aa");
        let res = fs_edit_apply(&path, "aa", "bb", false);
        let err = res.expect_err("несколько вхождений без replace_all должны быть отклонены");
        assert!(
            err.contains('3'),
            "в тексте ошибки должно быть число вхождений: {err}"
        );
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "aa aa aa",
            "файл на диске не должен измениться при отказе"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn multiple_occurrences_with_replace_all_replaced() {
        let path = write_temp("multi_all", "aa aa aa");
        let res = fs_edit_apply(&path, "aa", "bb", true);
        assert!(res.is_ok(), "{res:?}");
        let (replaced, _bytes, _newline_tolerant) = res.unwrap();
        assert_eq!(replaced, 3);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "bb bb bb");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fragment_not_found_rejected() {
        let path = write_temp("not_found", "hello world");
        let res = fs_edit_apply(&path, "missing", "x", false);
        let err = res.expect_err("отсутствующий фрагмент должен быть отклонён");
        assert!(err.contains("не найден"), "{err}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello world");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_old_string_rejected() {
        let path = write_temp("empty_old", "hello world");
        let res = fs_edit_apply(&path, "", "x", false);
        let err = res.expect_err("пустой old_string должен быть отклонён");
        assert!(err.contains("пуст"), "{err}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello world");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn same_old_and_new_string_rejected() {
        let path = write_temp("same", "hello world");
        let res = fs_edit_apply(&path, "hello", "hello", false);
        assert!(
            res.is_err(),
            "old_string == new_string должен быть отклонён: замена ничего не изменит"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello world");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn path_outside_scope_dir_rejected() {
        // Граница scope_dir действует и для правки — fs_edit_file проверяет
        // путь тем же fs_safe_path, что и остальные fs_*-инструменты, ПЕРЕД
        // вызовом fs_edit_apply.
        #[cfg(windows)]
        let root = "C:/Work";
        #[cfg(not(windows))]
        let root = "/work";
        let roots = vec![PathBuf::from(root)];
        let path = format!("{root}/other/file.txt");
        let scope = format!("{root}/proj");
        let res = fs_safe_path(&roots, &path, Some(scope.as_str()));
        assert!(
            res.is_err(),
            "путь вне рабочего каталога вызова должен быть отклонён и для fs_edit_file"
        );
    }

    #[test]
    fn multiline_fragment_replaced() {
        let original = "line1\nline2\nline3\n";
        let path = write_temp("multiline", original);
        let res = fs_edit_apply(&path, "line1\nline2", "lineA\nlineB", false);
        assert!(res.is_ok(), "{res:?}");
        let (replaced, _bytes, _newline_tolerant) = res.unwrap();
        assert_eq!(replaced, 1);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "lineA\nlineB\nline3\n"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn exact_single_line_edit_preserves_crlf_file_style() {
        let path = write_temp("exact_single_crlf", "a\r\nb\r\nc\r\n");
        let res = fs_edit_apply(&path, "b", "b1\nb2", false).unwrap();
        let after = std::fs::read_to_string(&path).unwrap();
        assert_eq!(after, "a\r\nb1\r\nb2\r\nc\r\n");
        assert_eq!(res.1, after.len());
        assert!(!after.as_bytes().windows(2).any(|pair| pair == b"\n\n"));
        assert_eq!(
            after.bytes().filter(|&byte| byte == b'\n').count(),
            after.bytes().filter(|&byte| byte == b'\r').count()
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn exact_single_line_edit_preserves_lf_file_style() {
        let path = write_temp("exact_single_lf", "a\nb\nc\n");
        fs_edit_apply(&path, "b", "b1\nb2", false).unwrap();
        let after = std::fs::read_to_string(&path).unwrap();
        assert_eq!(after, "a\nb1\nb2\nc\n");
        assert!(!after.contains('\r'));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn exact_multiline_edit_uses_old_string_crlf_style() {
        let path = write_temp("exact_multiline_crlf", "a\r\nb\r\nc\r\n");
        fs_edit_apply(&path, "a\r\nb", "x\ny\nz", false).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "x\r\ny\r\nz\r\nc\r\n"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn exact_single_line_edit_uses_prevalent_crlf_style_in_mixed_file() {
        let path = write_temp("exact_mixed_crlf", "a\r\nb\r\nc\ntail");
        fs_edit_apply(&path, "b", "b1\nb2", false).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "a\r\nb1\r\nb2\r\nc\ntail"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_apply_preserves_crlf_file_style() {
        let path = write_temp("write_crlf", "old\r\ncontent\r\n");
        let bytes = fs_write_apply(&path, "a\nb\nc\n").unwrap();
        let after = std::fs::read_to_string(&path).unwrap();
        assert_eq!(after, "a\r\nb\r\nc\r\n");
        assert_eq!(bytes, after.len());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_apply_preserves_lf_file_style() {
        let path = write_temp("write_lf", "old\ncontent\n");
        let bytes = fs_write_apply(&path, "a\r\nb\r\nc\r\n").unwrap();
        let after = std::fs::read_to_string(&path).unwrap();
        assert_eq!(after, "a\nb\nc\n");
        assert_eq!(bytes, after.len());
        assert!(!after.contains('\r'));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_apply_keeps_new_file_content_byte_for_byte() {
        let path = temp_path("write_new");
        let content = "a\r\nb\nc\r";
        let bytes = fs_write_apply(&path, content).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), content.as_bytes());
        assert_eq!(bytes, content.len());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn crlf_file_lf_fragment_replaced_via_fallback() {
        let original = "line1\r\nline2\r\nline3\r\n";
        let path = write_temp("crlf_lf_fragment", original);
        let res = fs_edit_apply(&path, "line1\nline2", "lineA\nlineB", false);
        assert!(res.is_ok(), "{res:?}");
        let (replaced, _bytes, newline_tolerant) = res.unwrap();
        assert_eq!(replaced, 1);
        assert!(newline_tolerant);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "lineA\r\nlineB\r\nline3\r\n"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn lf_file_crlf_fragment_replaced_via_fallback() {
        let original = "line1\nline2\nline3\n";
        let path = write_temp("lf_crlf_fragment", original);
        let res = fs_edit_apply(&path, "line1\r\nline2", "lineA\r\nlineB", false);
        assert!(res.is_ok(), "{res:?}");
        let (replaced, _bytes, newline_tolerant) = res.unwrap();
        assert_eq!(replaced, 1);
        assert!(newline_tolerant);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "lineA\nlineB\nline3\n"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fallback_boundary_newline_no_stray_cr() {
        // old_string начинается и заканчивается переводом строки — ловит
        // ошибку "позиция самого \n" в карте смещений: осиротевший \r или
        // удвоенный \r\r\n после замены.
        let original = "line1\r\nline2\r\nline3\r\n";
        let path = write_temp("fallback_boundary", original);
        let res = fs_edit_apply(&path, "\nline2\n", "\nLINE2\n", false);
        assert!(res.is_ok(), "{res:?}");
        let (replaced, _bytes, newline_tolerant) = res.unwrap();
        assert_eq!(replaced, 1);
        assert!(newline_tolerant);
        let after = std::fs::read_to_string(&path).unwrap();
        assert_eq!(after, "line1\r\nLINE2\r\nline3\r\n");
        let cr_count = after.bytes().filter(|&b| b == b'\r').count();
        let lf_count = after.bytes().filter(|&b| b == b'\n').count();
        assert_eq!(
            cr_count, lf_count,
            "не должно остаться ни одиночного \\r, ни удвоенного \\r\\r\\n"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fallback_multiple_occurrences_without_replace_all_rejected() {
        let original = "aa\r\nbb\r\naa\r\nbb\r\n";
        let path = write_temp("fallback_multi_reject", original);
        let res = fs_edit_apply(&path, "aa\nbb", "AA\nBB", false);
        let err =
            res.expect_err("несколько терпимых вхождений без replace_all должны быть отклонены");
        assert!(err.contains("встречается 2 раз"), "{err}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fallback_multiple_occurrences_with_replace_all_replaced() {
        let original = "aa\r\nbb\r\naa\r\nbb\r\n";
        let path = write_temp("fallback_multi_all", original);
        let res = fs_edit_apply(&path, "aa\nbb", "AA\nBB", true);
        assert!(res.is_ok(), "{res:?}");
        let (replaced, _bytes, newline_tolerant) = res.unwrap();
        assert_eq!(replaced, 2);
        assert!(newline_tolerant);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "AA\r\nBB\r\nAA\r\nBB\r\n"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fallback_multiline_new_string_gets_crlf_style() {
        let original = "line1\r\nline2\r\nline3\r\n";
        let path = write_temp("fallback_multiline_new", original);
        let res = fs_edit_apply(&path, "line1\nline2", "lineA\nlineB\nlineC", false);
        assert!(res.is_ok(), "{res:?}");
        let (replaced, _bytes, newline_tolerant) = res.unwrap();
        assert_eq!(replaced, 1);
        assert!(newline_tolerant);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "lineA\r\nlineB\r\nlineC\r\nline3\r\n"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn mixed_file_only_crlf_occurrences_replaced() {
        let original = "aa\r\nbb\r\nkeep\naa\r\nbb\r\n";
        let path = write_temp("mixed_crlf_lf", original);
        // Старый фрагмент точно не встречается в исходном тексте — только
        // после нормализации переводов строк, поэтому шаг 1 (точный поиск)
        // обязан дать ноль и включить запасной путь; LF-строка "keep" в
        // фрагмент не входит и должна остаться нетронутой.
        assert_eq!(original.matches("aa\nbb").count(), 0);
        let res = fs_edit_apply(&path, "aa\nbb", "AA\nBB", true);
        assert!(res.is_ok(), "{res:?}");
        let (replaced, _bytes, newline_tolerant) = res.unwrap();
        assert_eq!(replaced, 2);
        assert!(newline_tolerant);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "AA\r\nBB\r\nkeep\nAA\r\nBB\r\n"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn single_line_old_string_not_found_skips_fallback() {
        // old_string без переводов строк: запасной путь не включается вовсе,
        // текст ошибки — тот же, что и раньше.
        let path = write_temp("single_line_no_fallback", "hello world");
        let res = fs_edit_apply(&path, "missing", "x", false);
        let err = res.expect_err("отсутствующий фрагмент без переводов строк должен быть отклонён");
        assert_eq!(
            err,
            format!("фрагмент не найден в файле: {}", path.display())
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello world");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fallback_replacement_that_changes_nothing_rejected() {
        let original = "a\r\nb";
        let path = write_temp("noop_fallback", original);
        let res = fs_edit_apply(&path, "a\nb", "a\r\nb", false);
        let err = res.expect_err(
            "замена, приводящая к тому же байтовому содержимому, должна быть отклонена",
        );
        assert_eq!(err, "замена ничего не изменит");
        let after = std::fs::read(&path).unwrap();
        assert_eq!(after, original.as_bytes());
        let _ = std::fs::remove_file(&path);
    }
}
