const native = window.__TAURI__;
export const isDesktop = Boolean(native);

const pending = new Map();
let sequence = 0;
let eventSource = null;

function formatBytes(bytes) {
  if (bytes == null) return "";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let idx = 0;
  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024;
    idx += 1;
  }
  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} ${units[idx]}`;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(text.slice(0, 300));
    }
  }
  if (!response.ok) {
    throw new Error(data.error || `Request failed (${response.status})`);
  }
  if (data.error) {
    throw new Error(data.error);
  }
  return data;
}

function joinPath(dir, name) {
  if (!dir || dir === "__drives__") return name;
  const sep = dir.includes("\\") && !dir.includes("/") ? "\\" : "/";
  return dir.endsWith(sep) ? `${dir}${name}` : `${dir}${sep}${name}`;
}

function splitPath(raw) {
  const cleaned = (raw || "").trim().replace(/^"|"$/g, "");
  if (!cleaned) return { dir: "", name: "" };
  const idx = Math.max(cleaned.lastIndexOf("/"), cleaned.lastIndexOf("\\"));
  if (idx < 0) return { dir: "", name: cleaned };
  return { dir: cleaned.slice(0, idx) || cleaned.slice(0, idx + 1), name: cleaned.slice(idx + 1) };
}

function openWebPicker(mode, opts = {}) {
  const dialog = document.getElementById("web-picker-dialog");
  const titleEl = document.getElementById("picker-title");
  const pathInput = document.getElementById("picker-path");
  const upBtn = document.getElementById("picker-up");
  const goBtn = document.getElementById("picker-go");
  const listEl = document.getElementById("picker-list");
  const saveRow = document.getElementById("picker-save-row");
  const nameInput = document.getElementById("picker-filename");
  const uploadBtn = document.getElementById("picker-upload");
  const uploadInput = document.getElementById("picker-upload-input");
  const statusEl = document.getElementById("picker-status");
  const confirmBtn = document.getElementById("picker-confirm");
  const cancelBtn = document.getElementById("picker-cancel");
  const closeBtn = document.getElementById("picker-close");

  const isFolder = mode === "folder";
  const isSave = mode === "save";
  const filters = Array.isArray(opts.filters) ? opts.filters : [];
  const exts = [...new Set(filters.flatMap(f => f.extensions || []).map(e => String(e).toLowerCase()))];

  titleEl.textContent = opts.title || (isSave ? "Save file" : isFolder ? "Choose a folder" : "Choose a file");
  saveRow.hidden = !isSave;
  uploadBtn.hidden = isSave;
  uploadBtn.textContent = isFolder ? "Upload files…" : "Upload file…";
  uploadInput.multiple = isFolder;
  uploadInput.accept = exts.length ? exts.map(e => `.${e}`).join(",") : "";
  confirmBtn.textContent = isSave ? "Save here" : isFolder ? "Select folder" : "Select file";
  statusEl.textContent = "";
  statusEl.classList.remove("error");

  let currentDir = "";
  let parentDir = null;
  let selectedPath = "";

  const initial = splitPath(opts.defaultPath || "");
  nameInput.value = isSave ? (initial.name || (exts[0] ? `output.${exts[0]}` : "output.wav")) : "";

  function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
  }

  function updateConfirm() {
    if (isSave) {
      confirmBtn.disabled = !currentDir || currentDir === "__drives__" || !nameInput.value.trim();
    } else if (isFolder) {
      confirmBtn.disabled = !(selectedPath || (currentDir && currentDir !== "__drives__"));
    } else {
      confirmBtn.disabled = !selectedPath;
    }
  }

  async function loadDir(targetPath) {
    setStatus("Loading…");
    selectedPath = "";
    listEl.replaceChildren();
    const params = new URLSearchParams();
    if (targetPath) params.set("path", targetPath);
    if (!isFolder && exts.length) params.set("ext", exts.join(","));
    try {
      const data = await fetchJson(`/api/fs?${params.toString()}`);
      if (data.is_file) {
        return loadDir(data.parent || "");
      }
      currentDir = data.path;
      parentDir = data.parent;
      pathInput.value = data.display_path || data.path;
      upBtn.disabled = !parentDir;
      setStatus("");

      if (!data.entries.length) {
        const empty = document.createElement("li");
        empty.className = "picker-empty";
        empty.textContent = "No matching items in this folder.";
        listEl.appendChild(empty);
      } else {
        for (const entry of data.entries) {
          if (isFolder && !entry.is_dir) continue;
          const li = document.createElement("li");
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "picker-item";
          btn.dataset.path = entry.path;
          btn.dataset.dir = entry.is_dir ? "1" : "0";
          const icon = document.createElement("span");
          icon.className = "picker-icon";
          icon.textContent = entry.is_dir ? "▸" : "•";
          const name = document.createElement("span");
          name.className = "picker-name";
          name.textContent = entry.name;
          const meta = document.createElement("span");
          meta.className = "picker-size";
          meta.textContent = entry.is_dir ? "" : formatBytes(entry.size);
          btn.append(icon, name, meta);
          btn.addEventListener("click", () => {
            listEl.querySelectorAll(".picker-item.selected").forEach(el => el.classList.remove("selected"));
            if (entry.is_dir) {
              if (isFolder) {
                btn.classList.add("selected");
                selectedPath = entry.path;
                updateConfirm();
              } else {
                loadDir(entry.path);
              }
            } else if (isSave) {
              btn.classList.add("selected");
              nameInput.value = entry.name;
              updateConfirm();
            } else {
              btn.classList.add("selected");
              selectedPath = entry.path;
              updateConfirm();
            }
          });
          btn.addEventListener("dblclick", () => {
            if (entry.is_dir) {
              loadDir(entry.path);
            } else if (!isFolder) {
              finish(isSave ? joinPath(currentDir, entry.name) : entry.path);
            }
          });
          li.appendChild(btn);
          listEl.appendChild(li);
        }
      }
      updateConfirm();
    } catch (error) {
      setStatus(error.message || String(error), true);
      updateConfirm();
    }
  }

  let done = false;
  let resolvePromise;
  function cleanup() {
    upBtn.onclick = null;
    goBtn.onclick = null;
    pathInput.onkeydown = null;
    nameInput.oninput = null;
    uploadBtn.onclick = null;
    uploadInput.onchange = null;
    confirmBtn.onclick = null;
    cancelBtn.onclick = null;
    closeBtn.onclick = null;
    dialog.oncancel = null;
  }

  function finish(value) {
    if (done) return;
    done = true;
    cleanup();
    if (dialog.open) dialog.close();
    resolvePromise(value);
  }

  return new Promise(resolve => {
    resolvePromise = resolve;
    upBtn.onclick = () => { if (parentDir) loadDir(parentDir); };
    goBtn.onclick = () => loadDir(pathInput.value.trim());
    pathInput.onkeydown = event => {
      if (event.key === "Enter") {
        event.preventDefault();
        loadDir(pathInput.value.trim());
      }
    };
    nameInput.oninput = updateConfirm;
    uploadBtn.onclick = () => {
      uploadInput.value = "";
      uploadInput.click();
    };
    uploadInput.onchange = async () => {
      const files = Array.from(uploadInput.files || []);
      if (!files.length) return;
      const folderTag = `${isFolder ? "dataset" : "file"}-${Date.now()}`;
      uploadBtn.disabled = true;
      confirmBtn.disabled = true;
      try {
        let last = null;
        for (let i = 0; i < files.length; i += 1) {
          const file = files[i];
          setStatus(files.length > 1 ? `Uploading ${i + 1} / ${files.length}: ${file.name}…` : `Uploading ${file.name}…`);
          const url = `/api/upload?filename=${encodeURIComponent(file.name)}&folder=${encodeURIComponent(folderTag)}`;
          last = await fetchJson(url, { method: "POST", body: file });
        }
        finish(isFolder ? last.folder : last.path);
      } catch (error) {
        setStatus(error.message || String(error), true);
        updateConfirm();
      } finally {
        uploadBtn.disabled = false;
      }
    };
    confirmBtn.onclick = () => {
      if (isSave) {
        const filename = nameInput.value.trim();
        if (filename && currentDir && currentDir !== "__drives__") {
          finish(joinPath(currentDir, filename));
        }
      } else if (isFolder) {
        finish(selectedPath || currentDir);
      } else if (selectedPath) {
        finish(selectedPath);
      }
    };
    cancelBtn.onclick = () => finish(null);
    closeBtn.onclick = () => finish(null);
    dialog.oncancel = event => {
      event.preventDefault();
      finish(null);
    };

    dialog.showModal();
    loadDir(isSave ? initial.dir : (opts.defaultPath || ""));
  });
}

async function openWebOutput(targetPath) {
  const dialog = document.getElementById("web-output-dialog");
  const titleEl = document.getElementById("output-title");
  const pathEl = document.getElementById("output-path");
  const upBtn = document.getElementById("output-up");
  const audioWrap = document.getElementById("output-audio-wrap");
  const audioEl = document.getElementById("output-audio");
  const downloadBtn = document.getElementById("output-download");
  const listEl = document.getElementById("output-list");
  const statusEl = document.getElementById("output-status");
  const closeBtn = document.getElementById("output-close");

  let parentDir = null;

  function setPreview(filePath, isAudio) {
    downloadBtn.href = `/api/download?path=${encodeURIComponent(filePath)}`;
    downloadBtn.hidden = false;
    if (isAudio) {
      audioEl.src = `/api/download?path=${encodeURIComponent(filePath)}&inline=1`;
      audioWrap.hidden = false;
    } else {
      audioEl.pause();
      audioEl.removeAttribute("src");
      audioWrap.hidden = true;
    }
  }

  async function load(path) {
    statusEl.textContent = "Loading…";
    statusEl.classList.remove("error");
    listEl.replaceChildren();
    audioEl.pause();
    audioEl.removeAttribute("src");
    audioWrap.hidden = true;
    downloadBtn.hidden = true;
    try {
      const data = await fetchJson(`/api/fs?path=${encodeURIComponent(path || "")}`);
      pathEl.textContent = data.display_path || data.path;
      parentDir = data.parent;
      upBtn.hidden = !parentDir || data.is_file;
      statusEl.textContent = "";
      if (data.is_file) {
        titleEl.textContent = data.name || "Output file";
        setPreview(data.path, Boolean(data.audio));
        listEl.hidden = true;
        return;
      }
      titleEl.textContent = "Files & output";
      listEl.hidden = false;
      if (!data.entries.length) {
        const empty = document.createElement("li");
        empty.className = "picker-empty";
        empty.textContent = "This folder is empty.";
        listEl.appendChild(empty);
        return;
      }
      for (const entry of data.entries) {
        const li = document.createElement("li");
        li.className = "output-row";
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "picker-item";
        const icon = document.createElement("span");
        icon.className = "picker-icon";
        icon.textContent = entry.is_dir ? "▸" : entry.audio ? "♪" : "•";
        const name = document.createElement("span");
        name.className = "picker-name";
        name.textContent = entry.name;
        const size = document.createElement("span");
        size.className = "picker-size";
        size.textContent = entry.is_dir ? "" : formatBytes(entry.size);
        btn.append(icon, name, size);
        btn.addEventListener("click", () => {
          if (entry.is_dir) {
            load(entry.path);
          } else {
            listEl.querySelectorAll(".picker-item.selected").forEach(el => el.classList.remove("selected"));
            btn.classList.add("selected");
            setPreview(entry.path, Boolean(entry.audio));
          }
        });
        li.appendChild(btn);
        if (!entry.is_dir) {
          const dl = document.createElement("a");
          dl.className = "mini-link";
          dl.href = `/api/download?path=${encodeURIComponent(entry.path)}`;
          dl.textContent = "Download";
          li.appendChild(dl);
        }
        listEl.appendChild(li);
      }
    } catch (error) {
      statusEl.textContent = error.message || String(error);
      statusEl.classList.add("error");
    }
  }

  upBtn.onclick = () => { if (parentDir) load(parentDir); };
  closeBtn.onclick = () => {
    audioEl.pause();
    dialog.close();
  };
  dialog.oncancel = () => audioEl.pause();

  if (!dialog.open) dialog.showModal();
  await load(targetPath);
}

export async function invoke(command, args = {}) {
  if (isDesktop) {
    return native.core.invoke(command, args);
  }
  switch (command) {
    case "connect":
      return fetchJson("/api/connect", { method: "POST" });
    case "workspace": {
      const data = await fetchJson("/api/workspace");
      return data.root;
    }
    case "open_url":
      window.open(args.url, "_blank", "noopener");
      return null;
    case "show_file":
      return openWebOutput(args.path);
    case "plugin:dialog|open": {
      const opts = args.options || {};
      return openWebPicker(opts.directory ? "folder" : "open", opts);
    }
    case "plugin:dialog|save":
      return openWebPicker("save", args.options || {});
    default:
      throw new Error(`Unsupported command in browser WebUI: ${command}`);
  }
}

export function attach(receive) {
  if (isDesktop) {
    return native.event.listen("engine-message", ({ payload }) => {
      if (payload.id != null) {
        const request = pending.get(payload.id);
        if (!request) return;
        pending.delete(payload.id);
        clearTimeout(request.timer);
        if (payload.error) request.reject(new Error(payload.error));
        else request.resolve(payload.result);
        return;
      }
      if (payload.event === "closed") {
        for (const request of pending.values()) {
          clearTimeout(request.timer);
          request.reject(new Error("The Python worker closed."));
        }
        pending.clear();
      }
      receive(payload.event, payload.data, payload);
    });
  }

  if (eventSource) {
    eventSource.close();
  }
  eventSource = new EventSource("/api/events");
  eventSource.onmessage = event => {
    try {
      const payload = JSON.parse(event.data);
      receive(payload.event, payload.data, payload);
    } catch {
      // Ignore malformed keepalive frames
    }
  };
  return Promise.resolve();
}

export function call(action, args = {}) {
  if (!isDesktop) {
    return fetchJson("/api/call", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, args }),
    }).then(data => data.result);
  }
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error("The Python worker did not respond."));
    }, action === "audio_start" ? 900_000 : 60_000);
    pending.set(id, { resolve, reject, timer });
    invoke("send", { request: { id, action, args } }).catch(error => {
      pending.delete(id);
      clearTimeout(timer);
      reject(new Error(String(error)));
    });
  });
}
