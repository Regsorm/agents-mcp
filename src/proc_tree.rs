//! Привязка запускаемого процесса к дереву, которое целиком снимается при Drop.

use std::io;

use tokio::process::{Child, Command};

/// Запустить процесс в отдельном дереве и вернуть guard его времени жизни.
pub(crate) fn spawn(cmd: &mut Command) -> io::Result<(Child, ProcessTree)> {
    configure(cmd);
    let mut child = cmd.spawn()?;
    match ProcessTree::attach(&child) {
        Ok(tree) => Ok((child, tree)),
        Err(error) => {
            // Без дерева нельзя обещать снятие потомков. Прямой процесс всё же
            // останавливаем, чтобы ошибка назначения не оставила его работать.
            let _ = child.start_kill();
            Err(error)
        }
    }
}

#[cfg(unix)]
fn configure(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;

    cmd.as_std_mut().process_group(0);
}

#[cfg(windows)]
fn configure(_cmd: &mut Command) {}

#[cfg(unix)]
pub(crate) struct ProcessTree {
    process_group: i32,
}

#[cfg(unix)]
impl ProcessTree {
    fn attach(child: &Child) -> io::Result<Self> {
        let pid = child
            .id()
            .ok_or_else(|| io::Error::other("у дочернего процесса нет PID"))?;
        let process_group = i32::try_from(pid)
            .map_err(|_| io::Error::other(format!("PID {pid} не помещается в i32")))?;
        Ok(Self { process_group })
    }
}

#[cfg(unix)]
impl Drop for ProcessTree {
    fn drop(&mut self) {
        const SIGKILL: i32 = 9;

        unsafe extern "C" {
            fn killpg(process_group: i32, signal: i32) -> i32;
        }

        // ESRCH означает, что группа уже завершилась; при Drop сообщить ошибку
        // вызывающему коду всё равно невозможно.
        let _ = unsafe { killpg(self.process_group, SIGKILL) };
    }
}

#[cfg(windows)]
pub(crate) struct ProcessTree {
    job: isize,
}

#[cfg(windows)]
impl ProcessTree {
    fn attach(child: &Child) -> io::Result<Self> {
        use std::ffi::c_void;

        type Handle = *mut c_void;

        #[repr(C)]
        struct BasicLimitInformation {
            per_process_user_time_limit: i64,
            per_job_user_time_limit: i64,
            limit_flags: u32,
            minimum_working_set_size: usize,
            maximum_working_set_size: usize,
            active_process_limit: u32,
            affinity: usize,
            priority_class: u32,
            scheduling_class: u32,
        }

        #[repr(C)]
        struct IoCounters {
            read_operation_count: u64,
            write_operation_count: u64,
            other_operation_count: u64,
            read_transfer_count: u64,
            write_transfer_count: u64,
            other_transfer_count: u64,
        }

        #[repr(C)]
        struct ExtendedLimitInformation {
            basic_limit_information: BasicLimitInformation,
            io_info: IoCounters,
            process_memory_limit: usize,
            job_memory_limit: usize,
            peak_process_memory_used: usize,
            peak_job_memory_used: usize,
        }

        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn CreateJobObjectW(attributes: *mut c_void, name: *const u16) -> Handle;
            fn SetInformationJobObject(
                job: Handle,
                info_class: i32,
                info: *const c_void,
                info_len: u32,
            ) -> i32;
            fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
        }

        const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: i32 = 9;
        const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x0000_2000;

        let job = unsafe { CreateJobObjectW(std::ptr::null_mut(), std::ptr::null()) };
        if job.is_null() {
            return Err(io::Error::last_os_error());
        }
        let tree = Self { job: job as isize };

        let mut info: ExtendedLimitInformation = unsafe { std::mem::zeroed() };
        info.basic_limit_information.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let info_len = u32::try_from(std::mem::size_of::<ExtendedLimitInformation>())
            .expect("размер структуры WinAPI помещается в u32");
        let configured = unsafe {
            SetInformationJobObject(
                job,
                JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                &info as *const ExtendedLimitInformation as *const c_void,
                info_len,
            )
        };
        if configured == 0 {
            return Err(io::Error::last_os_error());
        }

        let process = child
            .raw_handle()
            .ok_or_else(|| io::Error::other("у дочернего процесса нет HANDLE"))?
            as Handle;
        if unsafe { AssignProcessToJobObject(job, process) } == 0 {
            return Err(io::Error::last_os_error());
        }

        Ok(tree)
    }
}

#[cfg(windows)]
impl Drop for ProcessTree {
    fn drop(&mut self) {
        use std::ffi::c_void;

        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn CloseHandle(handle: *mut c_void) -> i32;
        }

        let _ = unsafe { CloseHandle(self.job as *mut c_void) };
    }
}

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    use std::ffi::c_void;
    use std::path::Path;
    use std::process::Stdio;
    use std::time::{Duration, Instant};

    use tokio::process::Command;

    use super::spawn;

    #[cfg(windows)]
    type Handle = *mut c_void;

    #[cfg(windows)]
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn OpenProcess(access: u32, inherit_handle: i32, process_id: u32) -> Handle;
        fn WaitForSingleObject(handle: Handle, milliseconds: u32) -> u32;
        fn TerminateProcess(process: Handle, exit_code: u32) -> i32;
        fn CloseHandle(handle: Handle) -> i32;
    }

    #[cfg(windows)]
    const SYNCHRONIZE: u32 = 0x0010_0000;
    #[cfg(windows)]
    const PROCESS_TERMINATE: u32 = 0x0001;
    #[cfg(windows)]
    const WAIT_TIMEOUT: u32 = 0x0000_0102;

    #[cfg(windows)]
    fn process_is_alive(pid: u32) -> bool {
        let process = unsafe { OpenProcess(SYNCHRONIZE, 0, pid) };
        if process.is_null() {
            return false;
        }
        let status = unsafe { WaitForSingleObject(process, 0) };
        let _ = unsafe { CloseHandle(process) };
        status == WAIT_TIMEOUT
    }

    #[cfg(windows)]
    fn terminate_process(pid: u32) {
        let process = unsafe { OpenProcess(PROCESS_TERMINATE, 0, pid) };
        if !process.is_null() {
            let _ = unsafe { TerminateProcess(process, 1) };
            let _ = unsafe { CloseHandle(process) };
        }
    }

    #[cfg(unix)]
    fn process_is_alive(pid: u32) -> bool {
        unsafe extern "C" {
            fn kill(pid: i32, signal: i32) -> i32;
        }

        let Ok(pid) = i32::try_from(pid) else {
            return false;
        };
        unsafe { kill(pid, 0) == 0 }
    }

    #[cfg(unix)]
    fn terminate_process(pid: u32) {
        unsafe extern "C" {
            fn kill(pid: i32, signal: i32) -> i32;
        }

        if let Ok(pid) = i32::try_from(pid) {
            let _ = unsafe { kill(pid, 9) };
        }
    }

    #[cfg(windows)]
    fn grandchild_command(pid_file: &Path) -> Command {
        let path = pid_file.to_string_lossy().replace('\'', "''");
        let script = format!(
            "$p = Start-Process -FilePath ping.exe -ArgumentList '-n','999','127.0.0.1' \
             -WindowStyle Hidden -PassThru; \
             [IO.File]::WriteAllText('{path}', $p.Id.ToString()); Wait-Process -Id $p.Id"
        );
        let mut cmd = Command::new("powershell.exe");
        cmd.args(["-NoProfile", "-NonInteractive", "-Command", &script]);
        cmd
    }

    #[cfg(unix)]
    fn grandchild_command(pid_file: &Path) -> Command {
        let path = pid_file.to_string_lossy().replace('\'', "'\\''");
        let script = format!("sleep 999 & echo $! > '{path}'; wait");
        let mut cmd = Command::new("sh");
        cmd.args(["-c", &script]);
        cmd
    }

    #[cfg(windows)]
    fn exit_seven_command() -> Command {
        let mut cmd = Command::new("cmd.exe");
        cmd.args(["/D", "/C", "exit 7"]);
        cmd
    }

    #[cfg(unix)]
    fn exit_seven_command() -> Command {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "exit 7"]);
        cmd
    }

    #[tokio::test]
    async fn timeout_drop_kills_grandchild() {
        let pid_file =
            std::env::temp_dir().join(format!("agents-mcp-proc-tree-{}.pid", uuid::Uuid::new_v4()));
        let mut cmd = grandchild_command(&pid_file);
        cmd.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let (mut child, tree) = spawn(&mut cmd).expect("запуск стенда");

        let deadline = Instant::now() + Duration::from_secs(10);
        let grandchild_pid = loop {
            if let Ok(text) = std::fs::read_to_string(&pid_file) {
                if let Ok(pid) = text.trim().parse::<u32>() {
                    break pid;
                }
            }
            assert!(Instant::now() < deadline, "внук не записал PID");
            tokio::time::sleep(Duration::from_millis(25)).await;
        };
        assert!(process_is_alive(grandchild_pid), "внук не запустился");

        assert!(
            tokio::time::timeout(Duration::from_millis(50), child.wait())
                .await
                .is_err(),
            "стенд неожиданно завершился до таймаута"
        );
        drop(tree);
        let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;

        let deadline = Instant::now() + Duration::from_secs(5);
        while process_is_alive(grandchild_pid) && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let grandchild_alive = process_is_alive(grandchild_pid);
        if grandchild_alive {
            terminate_process(grandchild_pid);
        }
        let _ = std::fs::remove_file(&pid_file);
        assert!(!grandchild_alive, "внук пережил снятие дерева процессов");
    }

    #[tokio::test]
    async fn normal_exit_keeps_status_code() {
        let mut cmd = exit_seven_command();
        cmd.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let (mut child, _tree) = spawn(&mut cmd).expect("запуск cmd");
        let status = child.wait().await.expect("ожидание cmd");
        assert_eq!(status.code(), Some(7));
    }
}
