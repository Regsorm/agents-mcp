//! Singleton-защита через PID-lock.
//!
//! Гарантирует, что у `agents-mcp` ровно один процесс на каталог логов. Без этого
//! второй экземпляр успевает запустить tokio runtime, прочитать конфиг и
//! только потом упасть на `bind 10048`. С PID-lock второй экземпляр выходит
//! сразу с понятным сообщением о PID работающего инстанса.
//!
//! Владение определяется блокировкой открытого файла средствами ОС. PID внутри
//! нужен только для понятной диагностики второго запуска.
//!
//! Файл-лок — `<log_dir>/agents-mcp.pid`; каталог создаётся при захвате.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{bail, Result};

const PID_FILE_NAME: &str = "agents-mcp.pid";

/// RAII-guard PID-lock. В `Drop` стирает свой PID и снимает блокировку.
pub struct PidLock {
    file: Option<File>,
    pid: u32,
}

impl PidLock {
    /// Захватить PID-lock в каталоге `log_dir`. Владение определяется блокировкой
    /// открытого файла; записанный PID используется только для диагностики.
    pub fn acquire(log_dir: &Path) -> Result<Self> {
        // Каталог может отсутствовать при первом или ручном запуске.
        if let Err(e) = std::fs::create_dir_all(log_dir) {
            bail!(
                "не удалось создать каталог логов {}: {}",
                log_dir.display(),
                e
            );
        }

        let pid_path = log_dir.join(PID_FILE_NAME);
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&pid_path)?;

        if !try_lock_file(&file)? {
            let mut content = String::new();
            let _ = file.read_to_string(&mut content);
            let pid = content.trim();
            let pid = if pid.is_empty() {
                "неизвестен"
            } else {
                pid
            };
            bail!(
                "Сервис agents-mcp уже запущен (PID {}). PID-файл: {}. \
                 Если это ошибочное срабатывание — удалите файл или дождитесь его \
                 автоудаления при graceful shutdown.",
                pid,
                pid_path.display()
            );
        }

        let mut previous = String::new();
        file.read_to_string(&mut previous)?;
        if !previous.trim().is_empty() {
            tracing::warn!(
                "найден устаревший PID-файл {} — перезаписываем",
                pid_path.display()
            );
        }

        let pid = std::process::id();
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(pid.to_string().as_bytes())?;
        file.flush()?;

        Ok(Self {
            file: Some(file),
            pid,
        })
    }
}

impl Drop for PidLock {
    fn drop(&mut self) {
        let Some(mut file) = self.file.take() else {
            return;
        };

        // Файл не удаляем: между снятием блокировки и удалением его успел бы
        // занять новый процесс, а следующий создал бы на том же пути второй файл
        // и тоже получил замок. Свой PID стираем под блокировкой — пустой файл
        // при следующем старте устаревшим не считается.
        let mut content = String::new();
        let owns_file = file.seek(SeekFrom::Start(0)).is_ok()
            && file.read_to_string(&mut content).is_ok()
            && content.trim() == self.pid.to_string();
        if owns_file {
            let _ = file.set_len(0);
        }

        let _ = unlock_file(&file);
    }
}

#[cfg(not(windows))]
fn try_lock_file(file: &File) -> std::io::Result<bool> {
    match file.try_lock() {
        Ok(()) => Ok(true),
        Err(std::fs::TryLockError::WouldBlock) => Ok(false),
        Err(std::fs::TryLockError::Error(error)) => Err(error),
    }
}

#[cfg(not(windows))]
fn unlock_file(file: &File) -> std::io::Result<()> {
    file.unlock()
}

#[cfg(windows)]
fn try_lock_file(file: &File) -> std::io::Result<bool> {
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;

    #[repr(C)]
    struct Overlapped {
        internal: usize,
        internal_high: usize,
        offset: u32,
        offset_high: u32,
        h_event: *mut c_void,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn LockFileEx(
            file: *mut c_void,
            flags: u32,
            reserved: u32,
            bytes_low: u32,
            bytes_high: u32,
            overlapped: *mut Overlapped,
        ) -> i32;
    }

    const LOCKFILE_FAIL_IMMEDIATELY: u32 = 0x0000_0001;
    const LOCKFILE_EXCLUSIVE_LOCK: u32 = 0x0000_0002;
    const ERROR_LOCK_VIOLATION: i32 = 33;
    let mut overlapped = Overlapped {
        internal: 0,
        internal_high: 0,
        // PID лежит в начале файла; дальний байт оставляем только под замок,
        // чтобы второй процесс мог прочитать диагностическое содержимое.
        offset: u32::MAX,
        offset_high: 0,
        h_event: std::ptr::null_mut(),
    };
    let locked = unsafe {
        LockFileEx(
            file.as_raw_handle(),
            LOCKFILE_FAIL_IMMEDIATELY | LOCKFILE_EXCLUSIVE_LOCK,
            0,
            1,
            0,
            &mut overlapped,
        )
    };
    if locked != 0 {
        return Ok(true);
    }

    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(ERROR_LOCK_VIOLATION) {
        Ok(false)
    } else {
        Err(error)
    }
}

#[cfg(windows)]
fn unlock_file(file: &File) -> std::io::Result<()> {
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;

    #[repr(C)]
    struct Overlapped {
        internal: usize,
        internal_high: usize,
        offset: u32,
        offset_high: u32,
        h_event: *mut c_void,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn UnlockFileEx(
            file: *mut c_void,
            reserved: u32,
            bytes_low: u32,
            bytes_high: u32,
            overlapped: *mut Overlapped,
        ) -> i32;
    }

    let mut overlapped = Overlapped {
        internal: 0,
        internal_high: 0,
        offset: u32::MAX,
        offset_high: 0,
        h_event: std::ptr::null_mut(),
    };
    let unlocked = unsafe { UnlockFileEx(file.as_raw_handle(), 0, 1, 0, &mut overlapped) };
    if unlocked != 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let id = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("agents-mcp-pid-lock-{}-{id}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn stale_file_with_own_pid_is_accepted() {
        let dir = TestDir::new();
        std::fs::write(dir.0.join(PID_FILE_NAME), std::process::id().to_string()).unwrap();

        let lock = PidLock::acquire(&dir.0).unwrap();

        drop(lock);
    }

    #[test]
    fn second_acquire_is_rejected_while_first_is_alive() {
        let dir = TestDir::new();
        let first = PidLock::acquire(&dir.0).unwrap();

        let error = match PidLock::acquire(&dir.0) {
            Ok(_) => panic!("повторный захват PID-lock неожиданно успешен"),
            Err(error) => error.to_string(),
        };

        assert!(error.contains("Сервис agents-mcp уже запущен"));
        assert!(
            error.contains(&format!("PID {}", std::process::id())),
            "{error}"
        );
        drop(first);
    }

    #[test]
    fn acquire_succeeds_after_first_is_dropped() {
        let dir = TestDir::new();
        let first = PidLock::acquire(&dir.0).unwrap();
        drop(first);
        assert_eq!(
            std::fs::read_to_string(dir.0.join(PID_FILE_NAME)).unwrap(),
            ""
        );

        let second = PidLock::acquire(&dir.0).unwrap();

        drop(second);
    }

    #[test]
    fn drop_keeps_file_rewritten_with_foreign_pid() {
        let dir = TestDir::new();
        let mut lock = PidLock::acquire(&dir.0).unwrap();
        let foreign_pid = lock.pid.saturating_add(1);
        let file = lock.file.as_mut().unwrap();
        file.set_len(0).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(foreign_pid.to_string().as_bytes()).unwrap();
        file.flush().unwrap();

        drop(lock);

        assert_eq!(
            std::fs::read_to_string(dir.0.join(PID_FILE_NAME)).unwrap(),
            foreign_pid.to_string()
        );
    }
}
