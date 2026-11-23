import { attach, call, invoke } from "./ipc.js";
import { $, browse, causalVoice, choices, field, list, options } from "./forms.js";

const infer = $("infer-form");
const rvc = $("rvc-form");
const causal = $("causal-form");
const realtime = $("realtime-form");
const state = { ready: false, root: "", page: "inference", job: null, realtime: "idle", scanned: false, tensorboardUrl: "" };
const pages = {
  inference: ["Inference", "Convert an audio file or a folder with a trained voice."],
  training: ["Training", "Build or resume an RVC or Casual Voice Changer voice."],
  realtime: ["Realtime", "Route converted microphone audio to an output device."],
  tensorboard: ["TensorBoard", "Inspect live training metrics in the app or in your browser."],
};
const active = new Set(["working", "running", "paused", "stopping", "finalizing"]);
const stageNames = {
  preprocess: "Preprocess",
  extract: "Extract",
  train: "Train",
  index: "Index",
};
const badgeLabels = {
  idle: "Ready",
  connecting: "Connecting",
  working: "Working",
  running: "Running",
  paused: "Paused",
  stopping: "Stopping",
  finalizing: "Finalizing",
  finished: "Done",
  stopped: "Stopped",
  error: "Error",
};
const logs = [];
let lastLogReplace = false;
let noticeTimer = null;

function isBusy() {
  return Boolean(state.job && active.has(state.job.state));
}

function setBadge(phase, label) {
  const badge = $("activity-badge");
  badge.dataset.state = phase;
  badge.textContent = label || badgeLabels[phase] || phase;
}

function report(text, error = false) {
  const el = $("activity-status");
  const message = (text || "").trim();
  const badgeText = $("activity-badge").textContent.trim().toLowerCase();
  const normalized = message.replace(/[.…]+$/, "").toLowerCase();
  const redundant = !message || (!error && normalized === badgeText);
  el.textContent = redundant ? "" : message;
  el.hidden = redundant;
  el.classList.toggle("error", Boolean(error && !redundant));
  if (!isBusy() && error) setBadge("error");
}

function notify(text, error = false) {
  const box = $("activity-notice");
  if (noticeTimer) {
    clearTimeout(noticeTimer);
    noticeTimer = null;
  }
  if (!text) {
    box.hidden = true;
    return;
  }
  if (!isBusy()) {
    box.hidden = true;
    report(text, error);
    return;
  }
  box.textContent = text;
  box.classList.toggle("error", error);
  box.hidden = false;
  if (!error) {
    noticeTimer = setTimeout(() => { box.hidden = true; }, 6000);
  }
}

function formatPercent(step, total) {
  if (!total || total <= 0) return "0% of 100%";
  const pct = Math.min(100, Math.max(0, (step * 100) / total));
  const formatted = pct === 0 ? "0" : pct >= 10 || Number.isInteger(pct) ? pct.toFixed(0) : pct >= 1 ? pct.toFixed(1) : pct.toFixed(2);
  return `${formatted}% of 100%`;
}

function renderStages(stages, current, finished) {
  const container = $("job-stages");
  if (!stages || stages.length <= 1) {
    container.hidden = true;
    container.replaceChildren();
    return;
  }
  const activeIdx = finished ? stages.length : stages.indexOf(current);
  const items = stages.map((key, idx) => {
    const li = document.createElement("li");
    const done = finished || (activeIdx >= 0 && idx < activeIdx);
    const isCurrent = !finished && idx === activeIdx;
    li.className = done ? "done" : isCurrent ? "active" : "";
    li.textContent = `${done ? "✓ " : isCurrent ? "● " : ""}${stageNames[key] || key}`;
    return li;
  });
  container.replaceChildren(...items);
  container.hidden = false;
}

function renderMeter({ step, total, unit = "steps", meta = "", indeterminate = false, visible = false }) {
  const meter = $("job-meter");
  const progress = $("job-progress");
  const header = meter.querySelector(".meter-header");
  meter.hidden = !visible;
  if (!visible) return;
  if (indeterminate || !total || total <= 0) {
    header.hidden = true;
    progress.removeAttribute("value");
    progress.removeAttribute("max");
  } else {
    header.hidden = false;
    const clamped = Math.min(step ?? 0, total);
    $("job-percent").textContent = formatPercent(clamped, total);
    $("job-steps").textContent = `${clamped.toLocaleString()} / ${total.toLocaleString()} ${unit}`;
    progress.max = total;
    progress.value = clamped;
  }
  const metaEl = $("job-meta");
  metaEl.textContent = meta;
  metaEl.hidden = !meta;
}

function parseStageProgress(text) {
  if (state.job?.kind !== "rvc" || !isBusy() || state.job.state === "paused" || state.job.state === "stopping") {
    return;
  }
  if (state.job.stage === "train") return;
  const stage = text.match(/^(Preprocessing|Extracting F0[^:]*|Extracting features[^:]*):\s*\d+%\|[^|]*\|\s*(\d+)\/(\d+)/);
  if (stage) {
    const done = Number(stage[2]);
    const total = Number(stage[3]);
    if (total > 0) {
      const unit = stage[1] === "Preprocessing" ? "files" : "clips";
      renderMeter({ step: done, total, unit, meta: stage[1], visible: true });
      report("");
    }
  }
}

function log(text, replace = false) {
  if (!text) return;
  const line = String(text).slice(0, 3500);
  if (replace && lastLogReplace && logs.length) {
    logs[logs.length - 1] = line;
  } else {
    logs.push(line);
    if (logs.length > 250) logs.shift();
  }
  lastLogReplace = Boolean(replace);
  parseStageProgress(line);
  if ($("log-details").open) showLog();
}

function showLog(forceBottom = false) {
  const pre = $("log");
  const atBottom = forceBottom || (pre.scrollHeight - pre.scrollTop - pre.clientHeight) < 28;
  pre.textContent = logs.join("\n");
  if (atBottom) pre.scrollTop = pre.scrollHeight;
}

function runtimeStatus(label, phase) {
  const element = $("runtime-status");
  element.lastChild.textContent = label;
  element.dataset.state = phase;
}

function reportError(error) {
  notify(error.message || String(error), true);
}

function levelMeter(linear) {
  if (!linear || linear <= 0) return 0;
  return Math.max(0, Math.min(1, (20 * Math.log10(linear) + 60) / 60));
}

function syncRvcControls() {
  const isV1 = field(rvc, "version").value === "v1";
  const isSimple = field(rvc, "cut_preprocess").value === "Simple";
  const usePretrained = field(rvc, "pretrained").checked;
  if (isV1) {
    field(rvc, "vocoder").value = "HiFi-GAN";
    field(rvc, "embedder_model").value = "contentvec";
  }
  field(rvc, "vocoder").disabled = isV1;
  field(rvc, "embedder_model").disabled = isV1;
  field(rvc, "chunk_len").disabled = !isSimple;
  field(rvc, "overlap_len").disabled = !isSimple;
  for (const name of ["g_pretrained_path", "d_pretrained_path"]) {
    const input = field(rvc, name);
    input.disabled = !usePretrained;
    const btn = input.nextElementSibling;
    if (btn) btn.disabled = !usePretrained;
  }
}

function controls() {
  const busy = isBusy();
  $("infer-fields").disabled = !state.ready || busy;
  $("rvc-fields").disabled = !state.ready || busy;
  $("causal-fields").disabled = !state.ready || busy;
  $("realtime-fields").disabled = !state.ready;
  $("reload-tensorboard").disabled = !state.ready || Boolean(tensorboardPromise);
  $("open-tensorboard-browser").disabled = !state.ready || Boolean(tensorboardPromise);
  $("refresh-catalog").disabled = !state.ready || busy;
  if (state.ready && !busy) syncRvcControls();
  const streaming = state.realtime !== "idle";
  realtime.querySelectorAll("[data-locked]").forEach(control => { control.disabled = streaming; });
  $("start-realtime").disabled = streaming || !state.scanned;
  $("stop-realtime").disabled = !streaming || state.realtime === "stopping";
  $("pause-job").hidden = !busy || !state.job.pausable;
  $("pause-job").disabled = false;
  $("pause-job").textContent = state.job?.state === "paused" ? "Resume" : "Pause";
  $("stop-job").hidden = !busy || !state.job.stoppable;
  $("stop-job").disabled = state.job?.kind === "rvc" && state.job?.state === "stopping";
  $("stop-job").textContent = state.job?.kind === "causal" && state.job?.state === "stopping" ? "Force stop" : "Stop training";
}

async function refreshCatalog() {
  const items = await call("catalog");
  list("models", [...items.rvc, ...items.causal]);
  list("indexes", items.indexes);
  list("runs", items.runs);
  list("causal-runs", items.causal_runs || []);
}

let listedDevices = false;
async function ensureDevices() {
  if (!state.ready || listedDevices) return;
  const result = await call("devices");
  choices(field(causal, "device"), result.map(({ id, label }) => ({ id, label: label || "CPU" })), result[0]?.id);
  if (result.length && result.every(({ id }) => id === "cpu") && field(rvc, "gpu").value === "0") {
    field(rvc, "gpu").value = "-";
  }
  listedDevices = true;
}

async function scanDevices(rescan = false) {
  if (!state.ready) return;
  $("refresh-devices").disabled = true;
  try {
    const data = await call("audio_devices", { rescan, all_drivers: field(realtime, "all_drivers").checked });
    choices(field(realtime, "input_label"), data.inputs, data.default_inputs, data.inputs.length ? null : "No microphone");
    choices(field(realtime, "output_label"), data.outputs, data.default_outputs, data.outputs.length ? null : "No output device");
    choices(field(realtime, "monitor_label"), data.outputs, "", "Off");
    state.scanned = data.inputs.length > 0 && data.outputs.length > 0;
    if (!state.scanned) notify("Connect an audio input and output device, then refresh.", true);
  } finally {
    controls();
  }
}

let tensorboardPromise = null;
function setTensorBoardStatus(text, error = false) {
  const el = $("tensorboard-status");
  el.textContent = text;
  el.classList.toggle("error", error);
}

function resetTensorBoard(message, error = false) {
  state.tensorboardUrl = "";
  const frame = $("tensorboard-frame");
  frame.src = "about:blank";
  frame.hidden = true;
  setTensorBoardStatus(message, error);
}

async function ensureTensorBoard() {
  if (!state.ready) {
    setTensorBoardStatus("Waiting for the Python runtime…");
    return null;
  }
  if (state.tensorboardUrl) {
    const frame = $("tensorboard-frame");
    if (frame.hidden || frame.getAttribute("src") !== state.tensorboardUrl) {
      frame.src = state.tensorboardUrl;
      frame.hidden = false;
    }
    return state.tensorboardUrl;
  }
  if (!tensorboardPromise) {
    setTensorBoardStatus("Starting TensorBoard…");
    tensorboardPromise = call("tensorboard_start")
      .then(({ url }) => {
        state.tensorboardUrl = url;
        const frame = $("tensorboard-frame");
        frame.src = url;
        frame.hidden = false;
        setTensorBoardStatus(url.split("?")[0]);
        return url;
      })
      .catch(error => {
        setTensorBoardStatus(error.message || String(error), true);
        return null;
      })
      .finally(() => {
        tensorboardPromise = null;
        controls();
      });
    controls();
  }
  return tensorboardPromise;
}

async function reloadTensorBoard() {
  if (!state.ready) return;
  if (!state.tensorboardUrl) {
    await ensureTensorBoard();
    return;
  }
  const url = state.tensorboardUrl;
  const frame = $("tensorboard-frame");
  frame.src = "about:blank";
  setTimeout(() => {
    if (state.tensorboardUrl === url) {
      frame.src = url;
      frame.hidden = false;
    }
  }, 0);
}

async function openTensorBoardBrowser() {
  const url = await ensureTensorBoard();
  if (!url) return;
  try {
    await invoke("open_url", { url });
  } catch (error) {
    setTensorBoardStatus(error.message || String(error), true);
  }
}

async function page(name) {
  state.page = name;
  document.querySelectorAll("[data-page]").forEach(button => {
    const selected = button.dataset.page === name;
    button.classList.toggle("selected", selected);
    if (selected) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  for (const key of Object.keys(pages)) $(key).hidden = key !== name;
  [$("page-title").textContent, $("page-description").textContent] = pages[name];
  const isTensorBoard = name === "tensorboard";
  const content = document.querySelector(".content");
  content.classList.toggle("is-tensorboard", isTensorBoard);
  $("activity-panel").hidden = isTensorBoard;
  if (!isTensorBoard) content.scrollTo(0, 0);
  if (state.ready && !isTensorBoard) refreshCatalog().catch(error => log(error.message || String(error)));
  if (name === "training" && state.ready) ensureDevices().catch(error => log(error.message || String(error)));
  if (name === "realtime" && state.ready && !state.scanned) await scanDevices();
  if (isTensorBoard) await ensureTensorBoard();
}

async function initRuntime() {
  $("retry-runtime").hidden = true;
  if (!state.ready) {
    runtimeStatus("Starting Python…", "starting");
    setBadge("connecting");
    report("Starting the Python runtime…");
  }
  try {
    const metadata = await invoke("connect", {});
    state.root = metadata.root;
    if (metadata.connected) {
      runtimeStatus("Local engine ready", "ready");
      if (!isBusy()) {
        setBadge("idle");
        report("");
      }
    } else {
      state.ready = false;
      state.scanned = false;
      listedDevices = false;
      resetTensorBoard("Waiting for the Python runtime…");
      runtimeStatus(metadata.bootstrapping ? "Setting up runtime…" : "Starting Python…", "starting");
      setBadge(metadata.bootstrapping ? "working" : "connecting", metadata.bootstrapping ? "Setting up" : "Connecting");
      controls();
    }
  } catch (error) {
    renderMeter({ visible: false });
    runtimeStatus("Runtime setup failed", "error");
    setBadge("error", "Setup error");
    report(String(error), true);
    log(String(error));
    $("retry-runtime").hidden = false;
  }
}

function applySettings(settings) {
  if (!settings) return;
  if (settings.precision) field(rvc, "precision").value = settings.precision;
  field(rvc, "model_author").value = settings.model_author || "";
}

function persistRvcSettings() {
  if (!state.ready) return;
  call("settings_set", {
    precision: field(rvc, "precision").value,
    model_author: field(rvc, "model_author").value.trim(),
  }).then(applySettings).catch(error => log(error.message || String(error)));
}

function jobStatus(data) {
  state.job = data;
  const busy = active.has(data.state);
  const finished = data.state === "finished";
  const hasMeterStats = busy && Boolean(data.total && (data.step > 0 || data.epoch || data.rate || data.eta || data.losses));
  setBadge(data.state);
  report(hasMeterStats ? "" : data.message, data.state === "error");
  if (data.state === "error") log(data.message);
  renderStages(data.stages, data.stage, finished);

  if (busy) {
    const metaParts = [];
    if (data.epoch) metaParts.push(`Epoch ${data.epoch}`);
    if (data.rate) metaParts.push(`${data.rate.toFixed(1)} step/s`);
    if (data.eta) metaParts.push(`ETA ${data.eta}`);
    if (data.losses) {
      metaParts.push(`g ${data.losses.g.toFixed(2)} · d ${data.losses.d.toFixed(2)} · mel ${data.losses.mel.toFixed(2)}`);
    }
    renderMeter({
      step: data.step,
      total: data.total,
      unit: "steps",
      meta: metaParts.join(" · "),
      indeterminate: !data.total,
      visible: true,
    });
  } else if (finished && data.total) {
    renderMeter({ step: data.step ?? data.total, total: data.total, unit: "steps", visible: true });
  } else {
    renderMeter({ visible: false });
  }

  if (data.output) {
    const jobBtn = $("job-output");
    jobBtn.hidden = false;
    jobBtn.dataset.path = data.output;
    if (data.kind === "infer") {
      const inferBtn = $("infer-result");
      inferBtn.hidden = false;
      inferBtn.dataset.path = data.output;
    } else {
      refreshCatalog().catch(error => log(error.message || String(error)));
    }
  }
  controls();
}

function realtimeStatus(data) {
  const changed = state.realtime !== data.state;
  state.realtime = data.state;
  $("input-level").value = levelMeter(data.input);
  $("output-level").value = levelMeter(data.output);
  $("latency").value = data.latency == null ? "—" : data.latency + " ms";
  $("dropouts").value = data.dropouts ?? "—";
  $("sample-rate").value = data.sampleRate ? data.sampleRate / 1000 + " kHz" : "—";
  if (data.error) {
    notify(data.error, true);
    log(data.error);
  } else if (data.notice && state.page === "realtime") {
    notify(data.notice);
  } else if (changed && state.page === "realtime") {
    if (data.state === "running") notify("Realtime conversion is running.");
    else if (data.state === "starting") notify("Loading voice and opening audio devices…");
    else if (data.state === "idle") notify("Realtime conversion stopped.");
  }
  if (changed) controls();
}

function receive(event, data, payload) {
  switch (event) {
    case "bootstrap": {
      const installing = data?.state === "installing";
      const failed = data?.state === "error";
      $("retry-runtime").hidden = !failed;
      if (installing) {
        runtimeStatus("Setting up runtime…", "starting");
        setBadge("working", "Setting up");
        report(data.message || "Setting up the Python runtime…");
        renderMeter({
          step: data.step,
          total: data.total,
          unit: "steps",
          meta: "First-time portable runtime setup",
          indeterminate: !data.total,
          visible: true,
        });
      } else if (failed) {
        renderMeter({ visible: false });
        runtimeStatus("Runtime setup failed", "error");
        setBadge("error", "Setup error");
        report(data.message || "Runtime setup failed. Check Logs and click Retry.", true);
        log(data.message || "Runtime setup failed.");
      }
      break;
    }
    case "runtime_error":
      log(data);
      renderMeter({ visible: false });
      runtimeStatus("Runtime error", "error");
      setBadge("error", "Runtime error");
      report(data || "Runtime error. Check Logs for details.", true);
      $("retry-runtime").hidden = false;
      break;
    case "ready":
      state.ready = true;
      state.root = data.root;
      listedDevices = false;
      $("retry-runtime").hidden = true;
      renderMeter({ visible: false });
      field(rvc, "cpu_cores").value = data.cpu;
      applySettings(data.settings);
      runtimeStatus("Local engine ready", "ready");
      setBadge("idle");
      report("");
      controls();
      refreshCatalog().catch(error => log(error.message || String(error)));
      if (state.page === "training") ensureDevices().catch(error => log(error.message || String(error)));
      if (state.page === "realtime") scanDevices().catch(reportError);
      if (state.page === "tensorboard") ensureTensorBoard();
      break;
    case "closed":
      state.ready = false;
      state.scanned = false;
      listedDevices = false;
      state.job = null;
      renderStages(null);
      renderMeter({ visible: false });
      resetTensorBoard("Python runtime disconnected.", true);
      realtimeStatus({ state: "idle" });
      runtimeStatus("Python worker closed", "error");
      setBadge("error", "Disconnected");
      report("The Python worker closed. Click Retry to reconnect, or check Logs.", true);
      $("retry-runtime").hidden = false;
      controls();
      break;
    case "log":
      log(data, Boolean(payload?.replace));
      break;
    case "job":
      jobStatus(data);
      break;
    case "realtime":
      realtimeStatus(data);
      break;
  }
}

const inferRvc = {};
const rvcRealtime = {};
const autoIndex = new WeakMap();
let inferModel = "";
let inferCausal = false;
let realtimeModel = "";
let realtimeCausal = false;
const inferVoiceKeys = ["pitch", "index_rate"];
const voiceKeys = ["pitch", "index_rate", "threshold", "input_gain_db", "output_gain_db", "monitor_gain_db"];

function toggleVoiceFields(form, isCausal) {
  field(form, "pitch").step = isCausal ? "0.1" : "1";
  if (form === infer) {
    form.querySelectorAll("[data-infer-rvc]").forEach(element => { element.hidden = isCausal; });
  } else {
    form.querySelectorAll("[data-rt-rvc]").forEach(element => { element.hidden = isCausal; });
    form.querySelectorAll("[data-rt-causal]").forEach(element => { element.hidden = !isCausal; });
  }
}

async function modelChanged(form) {
  const path = field(form, "model_path").value.trim();
  const isCausal = causalVoice(path);
  toggleVoiceFields(form, isCausal);
  if (form === infer) {
    if (inferModel === path) return;
    if (isCausal && !inferCausal) {
      for (const key of inferVoiceKeys) inferRvc[key] = field(form, key).value;
    } else if (!isCausal && inferCausal) {
      for (const [key, value] of Object.entries(inferRvc)) field(form, key).value = value;
      const pitch = field(form, "pitch");
      if (Number.isFinite(pitch.valueAsNumber)) pitch.value = String(Math.round(pitch.valueAsNumber));
    }
    inferModel = path;
    inferCausal = isCausal;
  } else {
    if (realtimeModel === path) return;
    if (isCausal && !realtimeCausal) {
      for (const key of [...voiceKeys, "block_time"]) rvcRealtime[key] = field(form, key).value;
    } else if (!isCausal && realtimeCausal) {
      for (const [key, value] of Object.entries(rvcRealtime)) field(form, key).value = value;
      const pitch = field(form, "pitch");
      if (Number.isFinite(pitch.valueAsNumber)) pitch.value = String(Math.round(pitch.valueAsNumber));
    }
    realtimeModel = path;
    realtimeCausal = isCausal;
    if (isCausal) field(form, "block_time").value = "0.04";
  }
  if (!state.ready || !path) return;
  if (isCausal) {
    const preset = await call("voice_settings", { path });
    if (field(form, "model_path").value.trim() !== path) return;
    for (const [key, value] of Object.entries(preset)) {
      const input = field(form, key);
      if (input) input.value = value;
    }
  } else {
    const indexInput = field(form, "index_path");
    const currentIndex = indexInput.value.trim();
    if (!currentIndex || currentIndex === autoIndex.get(form)) {
      const matched = await call("match_index", { path });
      if (field(form, "model_path").value.trim() !== path) return;
      indexInput.value = matched.index_path || "";
      autoIndex.set(form, matched.index_path || "");
    }
  }
}

async function trainingMode(useCausal) {
  $("choose-causal").setAttribute("aria-pressed", String(useCausal));
  $("choose-rvc").setAttribute("aria-pressed", String(!useCausal));
  rvc.hidden = useCausal;
  causal.hidden = !useCausal;
  if (state.ready) await ensureDevices();
}

function syncSamples() {
  const version = field(rvc, "version").value;
  const vocoder = field(rvc, "vocoder");
  if (version === "v1") vocoder.value = "HiFi-GAN";
  const rates = version === "v1" ? ["40000"] : vocoder.value === "RefineGAN" ? ["24000", "32000"] : ["32000", "40000", "48000"];
  choices(field(rvc, "sample_rate"), rates.map(rate => ({ id: rate, label: Number(rate) / 1000 + " kHz" })), rates[0]);
  syncRvcControls();
}

async function runJob(kind, args) {
  $("job-output").hidden = true;
  if (kind === "infer") $("infer-result").hidden = true;
  $("activity-notice").hidden = true;
  jobStatus({ kind, state: "working", message: "Preparing…" });
  try {
    await call("job_start", { kind, options: args });
  } catch (error) {
    jobStatus({ kind, state: "error", message: error.message });
  }
}

function applyTheme(theme, persist = false) {
  const dark = theme === "dark";
  if (dark) document.documentElement.dataset.theme = "dark";
  else delete document.documentElement.dataset.theme;
  const button = $("toggle-theme");
  if (button) button.setAttribute("aria-pressed", String(dark));
  const label = $("theme-label");
  if (label) label.textContent = dark ? "Light mode" : "Dark mode";
  if (persist) {
    try { localStorage.setItem("redpanda-theme", dark ? "dark" : "light"); } catch {}
  }
}

function bind() {
  $("log-details").addEventListener("toggle", () => {
    if ($("log-details").open) showLog(true);
  });
  $("copy-log").addEventListener("click", event => {
    event.preventDefault();
    event.stopPropagation();
    navigator.clipboard?.writeText(logs.join("\n")).then(() => notify("Logs copied to clipboard.")).catch(reportError);
  });
  $("clear-log").addEventListener("click", event => {
    event.preventDefault();
    event.stopPropagation();
    logs.length = 0;
    lastLogReplace = false;
    $("log").textContent = "";
  });
  document.querySelectorAll("[data-page]").forEach(button => button.addEventListener("click", () => page(button.dataset.page).catch(reportError)));
  document.querySelectorAll("[data-pick]").forEach(button => button.addEventListener("click", () => browse(button, state.root).catch(reportError)));
  document.querySelectorAll("[data-windows]").forEach(element => { element.hidden = !/Windows/i.test(navigator.userAgent); });
  $("retry-runtime").addEventListener("click", () => initRuntime());
  $("toggle-theme").addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
  });
  applyTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light");
  $("open-data").addEventListener("click", async () => {
    try { await invoke("show_file", { path: state.root || await invoke("workspace") }); }
    catch (error) { reportError(error); }
  });
  $("reload-tensorboard").addEventListener("click", () => reloadTensorBoard());
  $("open-tensorboard-browser").addEventListener("click", () => openTensorBoardBrowser());
  $("refresh-catalog").addEventListener("click", () => {
    refreshCatalog().then(() => notify("Model list refreshed.")).catch(reportError);
  });
  for (const key of ["precision", "model_author"]) field(rvc, key).addEventListener("change", persistRvcSettings);
  for (const button of [$("job-output"), $("infer-result")]) {
    button.addEventListener("click", () => invoke("show_file", { path: button.dataset.path }).catch(reportError));
  }
  for (const form of [infer, realtime]) {
    const modelInput = field(form, "model_path");
    modelInput.addEventListener("input", () => toggleVoiceFields(form, causalVoice(modelInput.value)));
    modelInput.addEventListener("change", () => modelChanged(form).catch(reportError));
    field(form, "index_path").addEventListener("input", () => { autoIndex.delete(form); });
  }

  field(infer, "batch").addEventListener("change", event => {
    const batch = event.target.checked;
    $("input-caption").textContent = batch ? "Input folder" : "Input audio";
    $("output-caption").textContent = batch ? "Output folder" : "Save as";
    $("infer-input-picker").dataset.pick = batch ? "folder" : "audio";
    $("infer-output-picker").dataset.pick = batch ? "folder" : "save";
    field(infer, "input").placeholder = batch ? "Choose a folder" : "Choose an audio file";
    field(infer, "output").placeholder = batch ? "Next to the input folder" : "Next to the input";
    for (const key of ["input", "output"]) field(infer, key).value = "";
  });
  infer.addEventListener("submit", event => {
    event.preventDefault();
    runJob("infer", options(infer));
  });

  $("choose-rvc").addEventListener("click", () => trainingMode(false).catch(reportError));
  $("choose-causal").addEventListener("click", () => trainingMode(true).catch(reportError));
  for (const key of ["vocoder", "version"]) field(rvc, key).addEventListener("change", syncSamples);
  for (const key of ["cut_preprocess", "pretrained"]) field(rvc, key).addEventListener("change", syncRvcControls);
  syncSamples();
  rvc.addEventListener("submit", event => {
    event.preventDefault();
    runJob("rvc", { ...options(rvc), stage: "all" });
  });
  causal.addEventListener("submit", event => {
    event.preventDefault();
    runJob("causal", options(causal));
  });
  $("stop-job").addEventListener("click", () => call("job_control", { command: "stop" }).catch(reportError));
  $("pause-job").addEventListener("click", () => {
    if (!state.job) return;
    $("pause-job").disabled = true;
    call("job_control", { command: state.job.state === "paused" ? "resume" : "pause" }).catch(error => {
      $("pause-job").disabled = false;
      reportError(error);
    });
  });

  $("refresh-devices").addEventListener("click", () => scanDevices(true).catch(reportError));
  field(realtime, "all_drivers").addEventListener("change", () => scanDevices().catch(reportError));
  realtime.addEventListener("submit", async event => {
    event.preventDefault();
    const args = options(realtime);
    realtimeStatus({ state: "starting" });
    try { await call("audio_start", args); }
    catch (error) {
      realtimeStatus({ state: "idle" });
      reportError(error);
    }
  });
  $("stop-realtime").addEventListener("click", () => {
    realtimeStatus({ state: "stopping" });
    call("audio_stop").catch(reportError);
  });
  realtime.addEventListener("change", event => {
    const input = event.target;
    if (state.realtime !== "running" || !input.matches("[data-live]") || !input.checkValidity()) return;
    const value = input.type === "checkbox" ? input.checked : input.type === "number" ? input.valueAsNumber : input.value;
    if (typeof value !== "number" || Number.isFinite(value)) {
      call("audio_update", { [input.name]: value }).catch(reportError);
    }
  });
}

bind();
controls();
attach(receive).then(() => initRuntime()).catch(reportError);
