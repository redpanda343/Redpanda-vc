use std::{
    io::{BufRead, BufReader, Read, Write},
    process::{ChildStdin, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use serde_json::{Value, json};
use tauri::{AppHandle, Emitter, Manager, State};

use crate::runtime::{self, BootstrapEvent, Runtime, Workspace};

const MAX_LOG_CHUNK: usize = 8192;

#[derive(Default)]
pub struct Bridge {
    connection: Mutex<Option<Connection>>,
    bootstrapping: AtomicBool,
}

struct Connection {
    stdin: ChildStdin,
    closed: Arc<AtomicBool>,
    replaced: Arc<AtomicBool>,
    runtime: Runtime,
}

pub fn close(bridge: &Bridge) {
    if let Ok(mut connection) = bridge.connection.lock() {
        if let Some(old) = connection.take() {
            old.replaced.store(true, Ordering::Release);
        }
    }
}

fn stream_logs<R: Read, F: FnMut(String, bool)>(reader: R, mut emit: F) {
    let mut reader = BufReader::new(reader);
    let mut buffer = Vec::new();
    let mut byte = [0u8; 1];
    let mut pending_cr = false;

    loop {
        match reader.read(&mut byte) {
            Ok(0) => break,
            Ok(_) => match byte[0] {
                b'\r' => {
                    if pending_cr {
                        let text = String::from_utf8_lossy(&buffer).into_owned();
                        buffer.clear();
                        if !text.trim().is_empty() {
                            emit(text, true);
                        }
                    }
                    pending_cr = true;
                }
                b'\n' => {
                    pending_cr = false;
                    let text = String::from_utf8_lossy(&buffer).into_owned();
                    buffer.clear();
                    if !text.trim().is_empty() {
                        emit(text, false);
                    }
                }
                ch => {
                    if pending_cr {
                        pending_cr = false;
                        let text = String::from_utf8_lossy(&buffer).into_owned();
                        buffer.clear();
                        if !text.trim().is_empty() {
                            emit(text, true);
                        }
                    }
                    if buffer.len() < MAX_LOG_CHUNK {
                        buffer.push(ch);
                    }
                }
            },
            Err(_) => break,
        }
    }
    if !buffer.is_empty() {
        let text = String::from_utf8_lossy(&buffer).into_owned();
        if !text.trim().is_empty() {
            emit(text, pending_cr);
        }
    }
}

fn spawn_worker(app: &AppHandle, bridge: &Bridge, runtime: Runtime) -> Result<(), String> {
    let mut connection = bridge.connection.lock().map_err(|error| error.to_string())?;
    if let Some(current) = connection.as_ref() {
        if !current.closed.load(Ordering::Acquire) {
            return Ok(());
        }
    }

    let mut child = runtime::python_command(&runtime)
        .args(["-u", "-m", "shared.desktop.bridge"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("Could not start Python: {error}"))?;
    let stdin = child.stdin.take().expect("piped worker input");
    let stdout = child.stdout.take().expect("piped worker output");
    let stderr = child.stderr.take().expect("piped worker logs");
    let closed = Arc::new(AtomicBool::new(false));
    let done = Arc::clone(&closed);
    let replaced = Arc::new(AtomicBool::new(false));
    let log_replaced = Arc::clone(&replaced);
    let out_replaced = Arc::clone(&replaced);
    let logger = app.clone();
    thread::spawn(move || {
        stream_logs(stderr, |line, replace| {
            if !log_replaced.load(Ordering::Acquire) {
                let _ = logger.emit(
                    "engine-message",
                    json!({"event": "log", "data": line, "replace": replace}),
                );
            }
        });
    });
    let app_handle = app.clone();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if out_replaced.load(Ordering::Acquire) {
                continue;
            }
            if let Ok(value) = serde_json::from_str::<Value>(&line) {
                let _ = app_handle.emit("engine-message", value);
            }
        }
        let code = child.wait().ok().and_then(|status| status.code());
        done.store(true, Ordering::Release);
        if !out_replaced.load(Ordering::Acquire) {
            let _ = app_handle.emit("engine-message", json!({"event": "closed", "data": code}));
        }
    });
    *connection = Some(Connection {
        stdin,
        closed,
        replaced,
        runtime,
    });
    Ok(())
}

fn launch_bootstrap(app: AppHandle, workspace: Workspace) {
    thread::spawn(move || {
        let emitter = app.clone();
        let result = runtime::bootstrap_runtime(&workspace, |event| match event {
            BootstrapEvent::Step {
                step,
                total,
                message,
            } => {
                let _ = emitter.emit(
                    "engine-message",
                    json!({
                        "event": "bootstrap",
                        "data": {
                            "state": "installing",
                            "step": step,
                            "total": total,
                            "message": message,
                        }
                    }),
                );
            }
            BootstrapEvent::Log(line) => {
                let _ = emitter.emit(
                    "engine-message",
                    json!({"event": "log", "data": line, "replace": false}),
                );
            }
        });

        let bridge = app.state::<Bridge>();
        match result {
            Ok(python) => {
                let rt = Runtime { workspace, python };
                if let Err(error) = spawn_worker(&app, &bridge, rt) {
                    let _ = app.emit(
                        "engine-message",
                        json!({"event": "bootstrap", "data": {"state": "error", "message": error}}),
                    );
                }
            }
            Err(error) => {
                let _ = app.emit(
                    "engine-message",
                    json!({"event": "bootstrap", "data": {"state": "error", "message": error}}),
                );
            }
        }
        bridge.bootstrapping.store(false, Ordering::Release);
    });
}

#[tauri::command]
pub async fn connect(app: AppHandle, bridge: State<'_, Bridge>) -> Result<Value, String> {
    let workspace = runtime::resolve_workspace(&app)?;

    {
        let mut connection = bridge.connection.lock().map_err(|error| error.to_string())?;
        if let Some(current) = connection.as_ref() {
            if !current.closed.load(Ordering::Acquire) {
                return Ok(json!({
                    "root": current.runtime.workspace.data_root,
                    "python": current.runtime.python,
                    "connected": true,
                    "bootstrapping": false,
                }));
            }
            connection.take();
        }
    }

    if let Some(python) = runtime::find_existing_python(&workspace) {
        let rt = Runtime {
            workspace: workspace.clone(),
            python: python.clone(),
        };
        spawn_worker(&app, &bridge, rt)?;
        return Ok(json!({
            "root": workspace.data_root,
            "python": python,
            "connected": false,
            "bootstrapping": false,
        }));
    }

    if !bridge.bootstrapping.swap(true, Ordering::AcqRel) {
        launch_bootstrap(app, workspace.clone());
    }
    Ok(json!({
        "root": workspace.data_root,
        "python": Value::Null,
        "connected": false,
        "bootstrapping": true,
    }))
}

#[tauri::command]
pub fn send(bridge: State<'_, Bridge>, request: Value) -> Result<(), String> {
    let mut client = bridge.connection.lock().map_err(|error| error.to_string())?;
    let connection = client.as_mut().ok_or("Python worker is unavailable.")?;
    let mut line = serde_json::to_vec(&request).map_err(|error| error.to_string())?;
    line.push(b'\n');
    connection
        .stdin
        .write_all(&line)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::stream_logs;

    #[test]
    fn splits_crlf_lf_and_carriage_return_progress() {
        let input = b"line one\r\nTraining: 1/10\rTraining: 2/10\rTraining: 10/10\nDone";
        let mut lines = Vec::new();
        stream_logs(&input[..], |text, replace| lines.push((text, replace)));
        assert_eq!(
            lines,
            vec![
                ("line one".to_string(), false),
                ("Training: 1/10".to_string(), true),
                ("Training: 2/10".to_string(), true),
                ("Training: 10/10".to_string(), false),
                ("Done".to_string(), false),
            ]
        );
    }
}
