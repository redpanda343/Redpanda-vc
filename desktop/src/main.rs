#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod bridge;
mod runtime;

use std::path::PathBuf;

use tauri::{AppHandle, LogicalSize, Manager};

const DEFAULT_WIDTH: f64 = 1160.0;
const DEFAULT_HEIGHT: f64 = 780.0;
const MIN_WIDTH: f64 = 640.0;
const MIN_HEIGHT: f64 = 460.0;

fn target_window_size(work_width: f64, work_height: f64) -> (f64, f64) {
    let max_w = (work_width * 0.94).floor();
    let max_h = (work_height * 0.92).floor();
    (
        DEFAULT_WIDTH.min(max_w).max(320.0),
        DEFAULT_HEIGHT.min(max_h).max(240.0),
    )
}

fn fit_window(window: &tauri::WebviewWindow) {
    let monitor = window
        .current_monitor()
        .ok()
        .flatten()
        .or_else(|| window.primary_monitor().ok().flatten());
    if let Some(monitor) = monitor {
        let scale = monitor.scale_factor().max(1.0);
        let area = monitor.work_area();
        let work_w = area.size.width as f64 / scale;
        let work_h = area.size.height as f64 / scale;
        let (width, height) = target_window_size(work_w, work_h);
        let _ = window.set_min_size(Some(LogicalSize::new(
            MIN_WIDTH.min(width),
            MIN_HEIGHT.min(height),
        )));
        let _ = window.set_size(LogicalSize::new(width, height));
    }
    let _ = window.center();
    let _ = window.show();
}

#[tauri::command]
async fn workspace(app: AppHandle) -> Result<PathBuf, String> {
    Ok(runtime::resolve_workspace(&app)?.data_root)
}

#[tauri::command]
fn show_file(path: PathBuf) -> Result<(), String> {
    runtime::open_path(&path)
}

#[tauri::command]
fn open_url(url: String) -> Result<(), String> {
    runtime::open_url(&url)
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(bridge::Bridge::default())
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                fit_window(&window);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            bridge::connect,
            bridge::send,
            workspace,
            show_file,
            open_url
        ])
        .build(tauri::generate_context!())
        .expect("could not start Redpanda")
        .run(|app, event| {
            if matches!(event, tauri::RunEvent::Exit) {
                bridge::close(&app.state::<bridge::Bridge>());
            }
        });
}

#[cfg(test)]
mod tests {
    use super::target_window_size;

    #[test]
    fn clamps_window_size_to_low_resolution_monitors() {
        assert_eq!(target_window_size(1920.0, 1040.0), (1160.0, 780.0));
        assert_eq!(target_window_size(1366.0, 728.0), (1160.0, 669.0));
        assert_eq!(target_window_size(1024.0, 560.0), (962.0, 515.0));
    }
}
