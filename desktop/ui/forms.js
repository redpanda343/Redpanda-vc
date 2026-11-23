import { invoke } from "./ipc.js";

export const $ = id => document.getElementById(id);
export const field = (form, name) => form.elements.namedItem(name);
export const causalVoice = value => /\.(cvc|causalvc\.json|nextvc\.json)$/i.test(value.trim().replace(/"$/, ""));

export function options(form) {
  const result = {};
  for (const control of form.elements) {
    if (!control.name || control.matches(":disabled") || control.closest("[hidden]")) continue;
    if (control.type === "checkbox") result[control.name] = control.checked;
    else if (control.type === "number") {
      if (Number.isFinite(control.valueAsNumber)) result[control.name] = control.valueAsNumber;
    } else {
      result[control.name] = control.value.trim();
    }
  }
  return result;
}

export function list(id, values) {
  $(id).replaceChildren(...values.map(value => new Option(value.split(/[\\/]/).slice(-2).join(" / "), value)));
}

export function choices(select, values, preferred, empty = null) {
  const previous = select.value;
  const items = values.map(value => typeof value === "string" ? new Option(value, value) : new Option(value.label, value.id));
  if (empty !== null) items.unshift(new Option(empty, ""));
  select.replaceChildren(...items);
  const choices = Array.from(select.options, option => option.value);
  select.value = choices.includes(previous) ? previous : choices.includes(preferred) ? preferred : choices[0] || "";
}

const AUDIO = { name: "Audio", extensions: ["wav", "mp3", "flac", "ogg", "opus", "m4a", "aac", "aiff"] };
const FILTER = {
  audio: AUDIO,
  voice: { name: "Voice models", extensions: ["pth", "cvc", "json"] },
  rvc: { name: "RVC checkpoint", extensions: ["pth"] },
  index: { name: "Feature index", extensions: ["index"] },
};

export async function browse(button, root = "") {
  const form = button.closest("form");
  const target = field(form, button.dataset.field);
  const kind = button.dataset.pick;
  const current = target.value.trim();
  const sep = root.includes("\\") && !root.includes("/") ? "\\" : "/";
  const fallbackDir = !current && root && button.dataset.field === "dataset_path"
    ? `${root.replace(/[\\/]+$/, "")}${sep}datasets`
    : "";
  let path;
  if (kind === "save") {
    const ext = field(form, "format").value;
    const input = field(form, "input").value.trim().replace(/^"|"$/g, "");
    const suggested = input ? input.replace(/\.[^./\\]+$/, "") + "_converted." + ext : "output." + ext;
    path = await invoke("plugin:dialog|save", {
      options: { title: "Save converted audio", defaultPath: current || suggested, filters: [{ name: ext.toUpperCase(), extensions: [ext] }] },
    });
  } else {
    const custom = FILTER[kind];
    const options = {
      directory: kind === "folder",
      multiple: false,
      ...(custom && { filters: [custom] }),
    };
    if (current && current !== "default") options.defaultPath = current;
    else if (fallbackDir) options.defaultPath = fallbackDir;
    path = await invoke("plugin:dialog|open", { options });
  }
  if (path) {
    target.value = path;
    target.dispatchEvent(new Event("change", { bubbles: true }));
  }
}
