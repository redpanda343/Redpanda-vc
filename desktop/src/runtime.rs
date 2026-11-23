use std::{
    env, fs, io,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
};

use serde::Serialize;
use tauri::{AppHandle, Manager};

include!(concat!(env!("OUT_DIR"), "/backend.rs"));

const BOOTSTRAP_MARKER: &str = ".bootstrap_complete";
const CUDA_INDEX_URL: &str = "https://download.pytorch.org/whl/cu130";
const CPU_INDEX_URL: &str = "https://download.pytorch.org/whl/cpu";

#[derive(Clone, Debug, Serialize)]
pub struct Workspace {
    pub data_root: PathBuf,
    pub code_root: PathBuf,
    pub standalone: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Runtime {
    pub workspace: Workspace,
    pub python: PathBuf,
}

#[derive(Clone, Debug)]
pub enum BootstrapEvent {
    Step { step: u32, total: u32, message: String },
    Log(String),
}

pub fn resolve_workspace(app: &AppHandle) -> Result<Workspace, String> {
    if let Some(root) = env::var_os("REDPANDA_HOME") {
        let target = PathBuf::from(root);
        return prepare_portable_workspace(&target).map_err(|error| error.to_string());
    }
    // A binary inside a source checkout reuses that checkout directly for development.
    if let Ok(exe) = env::current_exe() {
        if let Some(checkout) = exe.ancestors().skip(1).find(|path| is_dev_checkout(path)) {
            let checkout = checkout.to_path_buf();
            ensure_data_dirs(&checkout).map_err(|error| error.to_string())?;
            return Ok(Workspace {
                data_root: checkout.clone(),
                code_root: checkout,
                standalone: false,
            });
        }
        // Distributed standalone build: keep runtime/, assets/, datasets/, models/, logs/ beside Redpanda.exe.
        if let Some(exe_dir) = exe.parent() {
            if is_writable_dir(exe_dir) {
                return prepare_portable_workspace(exe_dir).map_err(|error| error.to_string());
            }
        }
    }
    let fallback = app
        .path()
        .app_local_data_dir()
        .map_err(|error| error.to_string())?;
    prepare_portable_workspace(&fallback).map_err(|error| error.to_string())
}

fn is_dev_checkout(path: &Path) -> bool {
    path.join("core.py").is_file()
        && path.join("shared/desktop/bridge.py").is_file()
        && path.join("desktop/Cargo.toml").is_file()
}

fn is_writable_dir(dir: &Path) -> bool {
    if fs::create_dir_all(dir).is_err() {
        return false;
    }
    let probe = dir.join(format!(".redpanda-write-test-{}", std::process::id()));
    match fs::write(&probe, b"") {
        Ok(()) => {
            let _ = fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

fn ensure_data_dirs(root: &Path) -> io::Result<()> {
    for folder in ["assets", "datasets", "models", "logs"] {
        fs::create_dir_all(root.join(folder))?;
    }
    Ok(())
}

pub fn prepare_portable_workspace(path: &Path) -> io::Result<Workspace> {
    fs::create_dir_all(path)?;
    let data_root = std::path::absolute(path)?;
    ensure_data_dirs(&data_root)?;
    let runtime_root = data_root.join("runtime");
    let code_root = runtime_root.join("app");
    fs::create_dir_all(&code_root)?;

    for (relative, bytes) in BACKEND {
        let is_data_asset = relative.starts_with("assets/")
            || relative.starts_with("logs/")
            || relative.starts_with("models/");
        let file = if is_data_asset {
            data_root.join(relative)
        } else {
            code_root.join(relative)
        };
        if let Some(parent) = file.parent() {
            fs::create_dir_all(parent)?;
        }
        if fs::read(&file).is_ok_and(|existing| existing == *bytes) {
            continue;
        }
        let temporary = file.with_extension(format!("redpanda-{}.tmp", std::process::id()));
        fs::write(&temporary, bytes)?;
        fs::rename(temporary, file)?;
    }

    Ok(Workspace {
        data_root,
        code_root,
        standalone: true,
    })
}

fn standalone_python(python_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        python_dir.join("python.exe")
    } else {
        python_dir.join("bin/python3")
    }
}

fn venv_python(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir.join("Scripts/python.exe")
    } else {
        venv_dir.join("bin/python")
    }
}

fn env_has_torch(env_dir: &Path) -> bool {
    if cfg!(windows) {
        return env_dir.join("Lib/site-packages/torch").is_dir();
    }
    let lib = env_dir.join("lib");
    if let Ok(entries) = fs::read_dir(lib) {
        for entry in entries.flatten() {
            if entry.path().join("site-packages/torch").is_dir() {
                return true;
            }
        }
    }
    false
}

pub fn find_existing_python(ws: &Workspace) -> Option<PathBuf> {
    if let Some(explicit) = env::var_os("REDPANDA_PYTHON") {
        let path = PathBuf::from(explicit);
        if path.is_file() {
            return Some(path);
        }
    }

    let runtime_dir = ws.data_root.join("runtime");
    let portable_python_dir = runtime_dir.join("python");
    let standalone_py = standalone_python(&portable_python_dir);
    if standalone_py.is_file()
        && (runtime_dir.join(BOOTSTRAP_MARKER).is_file() || env_has_torch(&portable_python_dir))
    {
        return Some(standalone_py);
    }

    let portable_venv = runtime_dir.join("venv");
    let portable_py = venv_python(&portable_venv);
    if portable_py.is_file()
        && (runtime_dir.join(BOOTSTRAP_MARKER).is_file() || env_has_torch(&portable_venv))
    {
        return Some(portable_py);
    }

    if !ws.standalone {
        let candidates = if cfg!(windows) {
            [
                "env/python.exe",
                "env/Scripts/python.exe",
                ".venv/Scripts/python.exe",
            ]
        } else {
            ["env/bin/python", ".venv/bin/python", "venv/bin/python"]
        };
        for relative in candidates {
            let candidate = ws.data_root.join(relative);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    None
}

fn no_window(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    #[cfg(not(windows))]
    {
        let _ = command;
    }
}

fn run_logged<F>(mut command: Command, label: &str, on_event: &mut F) -> Result<(), String>
where
    F: FnMut(BootstrapEvent),
{
    no_window(&mut command);
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("{label} failed to start: {error}"))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let (tx, rx) = std::sync::mpsc::channel::<String>();

    let t_out = stdout.map(|out| {
        let tx = tx.clone();
        thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    let _ = tx.send(trimmed.to_string());
                }
            }
        })
    });
    let t_err = stderr.map(|err| {
        let tx = tx.clone();
        thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    let _ = tx.send(trimmed.to_string());
                }
            }
        })
    });
    drop(tx);

    for line in rx {
        on_event(BootstrapEvent::Log(line));
    }
    if let Some(handle) = t_out {
        let _ = handle.join();
    }
    if let Some(handle) = t_err {
        let _ = handle.join();
    }
    let status = child
        .wait()
        .map_err(|error| format!("{label} failed while waiting: {error}"))?;
    if !status.success() {
        return Err(format!("{label} exited with status {status}. Check Logs for details."));
    }
    Ok(())
}

fn has_nvidia_gpu() -> bool {
    #[cfg(windows)]
    {
        if let Some(sys_root) = env::var_os("SystemRoot") {
            if PathBuf::from(sys_root).join("System32/nvcuda.dll").is_file() {
                return true;
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if Path::new("/dev/nvidia0").exists() || Path::new("/dev/nvidiactl").exists() {
            return true;
        }
    }
    let mut cmd = Command::new(if cfg!(windows) { "nvidia-smi.exe" } else { "nvidia-smi" });
    no_window(&mut cmd);
    cmd.arg("-L")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn find_in_path(binary: &str) -> Option<PathBuf> {
    let paths = env::var_os("PATH")?;
    for dir in env::split_paths(&paths) {
        let candidate = dir.join(binary);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn ensure_uv<F>(runtime_dir: &Path, on_event: &mut F) -> Result<PathBuf, String>
where
    F: FnMut(BootstrapEvent),
{
    let bin_name = if cfg!(windows) { "uv.exe" } else { "uv" };
    let bin_dir = runtime_dir.join("bin");
    let local_uv = bin_dir.join(bin_name);
    if local_uv.is_file() {
        return Ok(local_uv);
    }
    if let Some(system_uv) = find_in_path(bin_name) {
        return Ok(system_uv);
    }

    fs::create_dir_all(&bin_dir).map_err(|error| error.to_string())?;

    #[cfg(windows)]
    {
        let archive_name = if cfg!(target_arch = "aarch64") {
            "uv-aarch64-pc-windows-msvc.zip"
        } else {
            "uv-x86_64-pc-windows-msvc.zip"
        };
        let url = format!("https://github.com/astral-sh/uv/releases/latest/download/{archive_name}");
        let zip_path = bin_dir.join("uv.zip");
        let script = format!(
            "$ProgressPreference='SilentlyContinue'; \
             Invoke-WebRequest -UseBasicParsing -Uri '{url}' -OutFile '{zip}'; \
             Expand-Archive -Force -Path '{zip}' -DestinationPath '{dest}'; \
             Remove-Item -Force '{zip}'",
            url = url,
            zip = zip_path.display(),
            dest = bin_dir.display()
        );
        let mut cmd = Command::new("powershell.exe");
        cmd.args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", &script]);
        run_logged(cmd, "Download portable uv", on_event)?;
    }

    #[cfg(not(windows))]
    {
        let archive_name = if cfg!(target_os = "macos") {
            if cfg!(target_arch = "aarch64") {
                "uv-aarch64-apple-darwin.tar.gz"
            } else {
                "uv-x86_64-apple-darwin.tar.gz"
            }
        } else if cfg!(target_arch = "aarch64") {
            "uv-aarch64-unknown-linux-gnu.tar.gz"
        } else {
            "uv-x86_64-unknown-linux-gnu.tar.gz"
        };
        let url = format!("https://github.com/astral-sh/uv/releases/latest/download/{archive_name}");
        let script = format!(
            "curl -fsSL '{url}' | tar -xz --strip-components=1 -C '{dest}'",
            url = url,
            dest = bin_dir.display()
        );
        let mut cmd = Command::new("sh");
        cmd.args(["-c", &script]);
        run_logged(cmd, "Download portable uv", on_event)?;
    }

    if local_uv.is_file() {
        return Ok(local_uv);
    }
    // Some archive layouts nest uv inside a subfolder; locate it if needed.
    if let Ok(entries) = fs::read_dir(&bin_dir) {
        for entry in entries.flatten() {
            let nested = entry.path().join(bin_name);
            if nested.is_file() {
                let _ = fs::rename(&nested, &local_uv);
                if local_uv.is_file() {
                    return Ok(local_uv);
                }
            }
        }
    }
    Err(format!("Downloaded uv archive did not contain {bin_name}."))
}

pub fn bootstrap_runtime<F>(ws: &Workspace, mut on_event: F) -> Result<PathBuf, String>
where
    F: FnMut(BootstrapEvent),
{
    let runtime_dir = ws.data_root.join("runtime");
    let python_dir = runtime_dir.join("python");
    let venv_dir = runtime_dir.join("venv");
    let cache_dir = runtime_dir.join("cache");
    fs::create_dir_all(&runtime_dir).map_err(|error| error.to_string())?;

    on_event(BootstrapEvent::Step {
        step: 1,
        total: 4,
        message: "Preparing portable package manager (uv)…".into(),
    });
    let uv = ensure_uv(&runtime_dir, &mut on_event)?;

    on_event(BootstrapEvent::Step {
        step: 2,
        total: 4,
        message: "Downloading portable Python 3.12 into runtime/…".into(),
    });
    let mut venv_cmd = Command::new(&uv);
    venv_cmd
        .args(["venv", "--python", "3.12", "--managed-python", "--clear"])
        .arg(&venv_dir)
        .env("UV_PYTHON_INSTALL_DIR", &python_dir)
        .env("UV_CACHE_DIR", &cache_dir);
    run_logged(venv_cmd, "Create portable Python environment", &mut on_event)?;

    let py = venv_python(&venv_dir);
    if !py.is_file() {
        return Err(format!("Portable Python executable missing after setup: {}", py.display()));
    }

    let use_cuda = has_nvidia_gpu();
    let index_url = if use_cuda { CUDA_INDEX_URL } else { CPU_INDEX_URL };
    on_event(BootstrapEvent::Step {
        step: 3,
        total: 4,
        message: if use_cuda {
            "Installing PyTorch (CUDA) and audio dependencies into runtime/…".into()
        } else {
            "Installing PyTorch (CPU) and audio dependencies into runtime/…".into()
        },
    });
    let requirements = ws.code_root.join("requirements.txt");
    let mut deps_cmd = Command::new(&uv);
    deps_cmd
        .args(["pip", "install", "--python"])
        .arg(&py)
        .arg("-r")
        .arg(&requirements)
        .args([
            "--extra-index-url",
            index_url,
            "--index-strategy",
            "unsafe-best-match",
        ])
        .env("UV_PYTHON_INSTALL_DIR", &python_dir)
        .env("UV_CACHE_DIR", &cache_dir);
    run_logged(deps_cmd, "Install Python dependencies", &mut on_event)?;

    on_event(BootstrapEvent::Step {
        step: 4,
        total: 4,
        message: "Installing SwiftF0 and RNNoise into runtime/…".into(),
    });
    let mut extras_cmd = Command::new(&uv);
    extras_cmd
        .args(["pip", "install", "--python"])
        .arg(&py)
        .args(["--no-deps", "swift-f0==0.3.0", "pyrnnoise==0.4.6"])
        .env("UV_PYTHON_INSTALL_DIR", &python_dir)
        .env("UV_CACHE_DIR", &cache_dir);
    run_logged(extras_cmd, "Install SwiftF0 and RNNoise", &mut on_event)?;

    let _ = fs::remove_dir_all(&cache_dir);
    fs::write(runtime_dir.join(BOOTSTRAP_MARKER), b"ok\n").map_err(|error| error.to_string())?;
    Ok(py)
}

pub fn python_command(runtime: &Runtime) -> Command {
    let mut command = if let Some(zluda) = env::var_os("REDPANDA_ZLUDA") {
        let mut command = Command::new(zluda);
        command.arg("--").arg(&runtime.python);
        command
    } else {
        Command::new(&runtime.python)
    };
    command
        .current_dir(&runtime.workspace.data_root)
        .env("PYTHONUTF8", "1")
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONPATH", &runtime.workspace.code_root)
        .env("REDPANDA_CODE_ROOT", &runtime.workspace.code_root)
        .env("REDPANDA_DATA_ROOT", &runtime.workspace.data_root);
    if runtime.workspace.standalone {
        if env::var_os("CAUSALVC_WEIGHTS").is_none() {
            command.env(
                "CAUSALVC_WEIGHTS",
                runtime.workspace.data_root.join("models/causalvc"),
            );
        }
        if env::var_os("CAUSALVC_RUNS").is_none() {
            command.env(
                "CAUSALVC_RUNS",
                runtime.workspace.data_root.join("logs/causalvc"),
            );
        }
    }
    if let Some(parent) = runtime.python.parent().filter(|p| p.is_absolute()) {
        let scripts = parent.join("Scripts");
        let dlls = parent.join("DLLs");
        let conda_libraries = parent.join("Library/bin");
        let paths = env::var_os("PATH").unwrap_or_default();
        let search = [parent.to_path_buf(), scripts, dlls, conda_libraries]
            .into_iter()
            .filter(|dir| dir.is_dir())
            .chain(env::split_paths(&paths));
        if let Ok(paths) = env::join_paths(search) {
            command.env("PATH", paths);
        }
    }
    no_window(&mut command);
    command
}

pub fn open_path(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("This path does not exist: {}", path.display()));
    }
    let program = if cfg!(windows) {
        "explorer.exe"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    Command::new(program)
        .arg(path)
        .spawn()
        .map(|_| ())
        .map_err(|error| error.to_string())
}

pub fn validate_local_url(url: &str) -> Result<&str, String> {
    let trimmed = url.trim();
    if trimmed.is_empty()
        || trimmed.chars().any(|ch| ch.is_whitespace() || ch.is_control())
        || !(trimmed.starts_with("http://127.0.0.1:") || trimmed.starts_with("http://localhost:"))
    {
        return Err("Only local HTTP URLs can be opened.".into());
    }
    Ok(trimmed)
}

pub fn open_url(url: &str) -> Result<(), String> {
    let safe_url = validate_local_url(url)?;
    #[cfg(windows)]
    {
        open_url_windows(safe_url)
    }
    #[cfg(not(windows))]
    {
        let program = if cfg!(target_os = "macos") {
            "open"
        } else {
            "xdg-open"
        };
        Command::new(program)
            .arg(safe_url)
            .spawn()
            .map(|_| ())
            .map_err(|error| error.to_string())
    }
}

#[cfg(windows)]
fn open_url_windows(url: &str) -> Result<(), String> {
    use std::{ffi::OsStr, os::windows::ffi::OsStrExt, ptr};

    #[link(name = "shell32")]
    unsafe extern "system" {
        fn ShellExecuteW(
            hwnd: *mut std::ffi::c_void,
            operation: *const u16,
            file: *const u16,
            parameters: *const u16,
            directory: *const u16,
            show_cmd: i32,
        ) -> isize;
    }

    let verb: Vec<u16> = OsStr::new("open").encode_wide().chain(Some(0)).collect();
    let target: Vec<u16> = OsStr::new(url).encode_wide().chain(Some(0)).collect();
    let result = unsafe {
        ShellExecuteW(
            ptr::null_mut(),
            verb.as_ptr(),
            target.as_ptr(),
            ptr::null(),
            ptr::null(),
            1, // SW_SHOWNORMAL
        )
    };
    if result <= 32 {
        return Err(format!("Could not open browser (Windows error code {result})."));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_only_loopback_http_urls() {
        assert!(validate_local_url("http://127.0.0.1:6006/tensorboard/?pinnedCards=%5B%5D").is_ok());
        assert!(validate_local_url("http://localhost:6006/tensorboard/").is_ok());
        assert!(validate_local_url("https://example.com").is_err());
        assert!(validate_local_url("file:///C:/Windows/System32/calc.exe").is_err());
        assert!(validate_local_url("http://127.0.0.1:6006/a b").is_err());
        assert!(validate_local_url("").is_err());
    }

    #[test]
    fn prepares_clean_five_folder_portable_workspace() {
        let dir = env::temp_dir().join(format!("redpanda-portable-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let ws = prepare_portable_workspace(&dir).unwrap();
        assert!(ws.standalone);
        assert_eq!(ws.code_root, ws.data_root.join("runtime/app"));

        let mut top_level: Vec<String> = fs::read_dir(&ws.data_root)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        top_level.sort();
        assert_eq!(top_level, vec!["assets", "datasets", "logs", "models", "runtime"]);
        assert!(ws.code_root.join("core.py").is_file());
        assert!(ws.code_root.join("shared/desktop/bridge.py").is_file());
        assert!(ws.data_root.join("assets/config_template.json").is_file());
        assert!(ws.data_root.join("logs/mute").is_dir());

        // Without runtime/venv, no python should be reported in standalone mode
        assert!(find_existing_python(&ws).is_none());

        // Pre-populated runtime/venv with marker should be detected without bootstrap
        let fake_py = venv_python(&ws.data_root.join("runtime/venv"));
        fs::create_dir_all(fake_py.parent().unwrap()).unwrap();
        fs::write(&fake_py, b"").unwrap();
        fs::write(ws.data_root.join("runtime").join(BOOTSTRAP_MARKER), b"ok\n").unwrap();
        assert_eq!(find_existing_python(&ws), Some(fake_py));

        // Relocatable standalone runtime/python should take priority when present
        let standalone_py = standalone_python(&ws.data_root.join("runtime/python"));
        fs::create_dir_all(standalone_py.parent().unwrap()).unwrap();
        fs::write(&standalone_py, b"").unwrap();
        assert_eq!(find_existing_python(&ws), Some(standalone_py));

        fs::remove_dir_all(&dir).unwrap();
    }
}
