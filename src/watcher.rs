//! Hot-reload реестра агентов через `notify` + ручной debounce.
//!
//! IDE/редактор обычно генерируют несколько событий на одно «сохранение»
//! (создание .swp, временный файл, переименование). Без debounce будет
//! 3-5 reload подряд. Здесь — таймер 500 мс «тихой паузы» после последнего
//! события, потом единственный `registry.reload()`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::registry::{absolute_agents_dir, Registry};

const DEBOUNCE_MS: u64 = 500;
const TICK_MS: u64 = 200;

/// Стартует watcher на `agents_dir`. Возвращает `RecommendedWatcher` —
/// его нужно удерживать живым в main.rs (иначе watcher остановится).
pub fn spawn(agents_dir: PathBuf, registry: Arc<Registry>) -> Result<RecommendedWatcher> {
    let agents_dir = absolute_agents_dir(&agents_dir)?;
    // Мост std::sync::mpsc (notify требует sync callback) → tokio::sync::mpsc.
    let (tx_tokio, mut rx_tokio) = mpsc::unbounded_channel::<()>();
    let agents_dir_for_filter = agents_dir.clone();

    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
        let event = match res {
            Ok(e) => e,
            Err(e) => {
                warn!(error = %e, "notify error");
                return;
            }
        };
        if !is_interesting(&event) {
            return;
        }
        if !event
            .paths
            .iter()
            .any(|p| is_relevant_path(p, &agents_dir_for_filter))
        {
            return;
        }
        let _ = tx_tokio.send(());
    })?;
    watcher.watch(&agents_dir, RecursiveMode::Recursive)?;

    // Debounce-задача.
    tokio::spawn(async move {
        let mut last_change: Option<Instant> = None;
        let mut interval = tokio::time::interval(Duration::from_millis(TICK_MS));
        interval.tick().await; // первый tick срабатывает сразу
        loop {
            tokio::select! {
                biased;
                maybe_evt = rx_tokio.recv() => {
                    if maybe_evt.is_none() {
                        // Канал закрыт — watcher умер, выходим.
                        break;
                    }
                    last_change = Some(Instant::now());
                }
                _ = interval.tick() => {
                    if let Some(t) = last_change {
                        if t.elapsed() >= Duration::from_millis(DEBOUNCE_MS) {
                            match registry.reload() {
                                Ok(_) => info!("agents/ перечитан (hot-reload)"),
                                Err(e) => warn!(error = %e, "hot-reload reload() упал"),
                            }
                            last_change = None;
                        }
                    }
                }
            }
        }
    });

    info!(dir = %agents_dir.display(), "watcher агентов запущен");
    Ok(watcher)
}

/// Стартует watcher на главный конфиг agents-mcp.toml: при изменении файла
/// запускает перечитку конфига целиком ([`crate::reload::ConfigReloader::reload`])
/// — она сама применяет изменившееся и пишет отчёт в журнал. Смена модели,
/// каталога агентов, разрешённых корней и набора провайдеров — без рестарта.
///
/// Watch ставим на РОДИТЕЛЬСКИЙ каталог с фильтром по имени файла: редакторы
/// часто пересоздают файл (новый inode), и watch на сам файл теряется.
pub fn spawn_config(
    config_path: PathBuf,
    reloader: Arc<crate::reload::ConfigReloader>,
) -> Result<RecommendedWatcher> {
    let watch_dir = config_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let file_name = config_path.file_name().map(|n| n.to_os_string());

    let (tx_tokio, mut rx_tokio) = mpsc::unbounded_channel::<()>();
    let file_name_filter = file_name.clone();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
        let event = match res {
            Ok(e) => e,
            Err(e) => {
                warn!(error = %e, "notify error (config)");
                return;
            }
        };
        if !is_interesting(&event) {
            return;
        }
        // Реагируем только на файл главного конфига (в каталоге могут быть и
        // другие файлы).
        let hit = event
            .paths
            .iter()
            .any(|p| match (&file_name_filter, p.file_name()) {
                (Some(want), Some(got)) => want.as_os_str() == got,
                _ => false,
            });
        if !hit {
            return;
        }
        let _ = tx_tokio.send(());
    })?;
    watcher.watch(&watch_dir, RecursiveMode::NonRecursive)?;

    tokio::spawn(async move {
        let mut last_change: Option<Instant> = None;
        let mut interval = tokio::time::interval(Duration::from_millis(TICK_MS));
        interval.tick().await; // первый tick срабатывает сразу
        loop {
            tokio::select! {
                biased;
                maybe_evt = rx_tokio.recv() => {
                    if maybe_evt.is_none() {
                        break; // канал закрыт — watcher умер
                    }
                    last_change = Some(Instant::now());
                }
                _ = interval.tick() => {
                    if let Some(t) = last_change {
                        if t.elapsed() >= Duration::from_millis(DEBOUNCE_MS) {
                            // Разбор и применение — внутри reload: он сам решает,
                            // что изменилось, применяет это и пишет журнал.
                            let _ = reloader.reload().await;
                            last_change = None;
                        }
                    }
                }
            }
        }
    });

    info!(path = %config_path.display(), "watcher главного конфига запущен");
    Ok(watcher)
}

fn is_interesting(event: &Event) -> bool {
    matches!(
        event.kind,
        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
    )
}

/// Реагируем на *.md/*.toml/*.json и на сам каталог агента внутри agents_dir.
/// Последнее нужно для переименования, переноса и удаления каталога. IDE-временные
/// файлы (`.swp`, `.tmp`, `~`-backup) игнорируются.
fn is_relevant_path(path: &Path, agents_dir: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(agents_dir) else {
        return false;
    };
    if relative.components().count() == 1 {
        return true;
    }
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    matches!(ext, "md" | "toml" | "json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_directory_event_is_relevant_without_extension() {
        let root = Path::new("C:/agents");
        assert!(is_relevant_path(Path::new("C:/agents/renamed-agent"), root));
        assert!(is_relevant_path(
            Path::new("C:/agents/_disabled-agent"),
            root
        ));
        assert!(!is_relevant_path(
            Path::new("C:/agents/agent/temporary-file"),
            root
        ));
        assert!(!is_relevant_path(Path::new("C:/other/agent"), root));
    }

    #[test]
    fn relative_agents_dir_becomes_absolute_for_watcher() {
        let resolved = absolute_agents_dir(Path::new("agents-relative")).expect("абсолютный путь");
        assert!(resolved.is_absolute());
        assert_eq!(
            resolved,
            std::env::current_dir()
                .expect("текущий каталог")
                .join("agents-relative")
        );
    }
}
