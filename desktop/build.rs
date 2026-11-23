use std::{env, fs, path::Path};

fn add_files(root: &Path, directory: &str, entries: &mut Vec<String>) {
    println!("cargo:rerun-if-changed=../{directory}");
    for file in fs::read_dir(root.join(directory)).expect("read backend folder") {
        let path = file.expect("read backend entry").path();
        let name = path.file_name().unwrap().to_str().expect("UTF-8 file name");
        if name.starts_with('.') || name == "__pycache__" {
            continue;
        }
        let relative = format!("{directory}/{name}");
        if path.is_dir() {
            add_files(root, &relative, entries);
        } else if name.starts_with("LICENSE")
            || matches!(
                path.extension().and_then(|ext| ext.to_str()),
                Some("py" | "json" | "wav" | "npy" | "html" | "css" | "js" | "png" | "ico")
            )
        {
            entries.push(relative);
        }
    }
}

fn main() {
    tauri_build::build();

    let root = Path::new("..")
        .canonicalize()
        .expect("locate Redpanda sources");
    let mut entries: Vec<String> = [
        "core.py",
        "requirements.txt",
        "LICENSE",
        "TERMS_OF_USE.md",
        "assets/config_template.json",
        "assets/red_panda_favicon.ico",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    for dir in [
        "causalvc",
        "rvc",
        "shared",
        "desktop/ui",
        "logs/mute",
        "logs/mute_spin-v2",
    ] {
        add_files(&root, dir, &mut entries);
    }
    entries.sort();
    let mut data = String::from("static BACKEND: &[(&str, &[u8])] = &[\n");
    for relative in entries {
        println!("cargo:rerun-if-changed=../{relative}");
        let path = root.join(&relative);
        data.push_str(&format!("    ({relative:?}, include_bytes!({path:?})),\n"));
    }
    data.push_str("];\n");
    let output = Path::new(&env::var_os("OUT_DIR").unwrap()).join("backend.rs");
    fs::write(output, data).expect("write embedded backend manifest");
}
