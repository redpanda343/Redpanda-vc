"""Paths, training settings, and catalogs for the local desktop & web app."""

import json
import os
import re
import tempfile
import threading
from pathlib import Path

from shared.tools.causal_voices import VOICE_RUNS, available_voices, is_causal_voice

CODE_ROOT = Path(os.environ.get("REDPANDA_CODE_ROOT") or Path(__file__).resolve().parents[2]).resolve()
ROOT = Path(os.environ.get("REDPANDA_DATA_ROOT") or CODE_ROOT).resolve()
LOGS = ROOT / "logs"
MODELS = ROOT / "models"
DATASETS = ROOT / "datasets"
CONFIG_PATH = ROOT / "assets" / "config.json"
PRECISIONS = ("fp16", "bf16", "fp32")
AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".aiff", ".wma"})
BATCH_AUDIO_SUFFIXES = AUDIO_SUFFIXES | frozenset({".mp4", ".alac", ".webm", ".ac3"})
EXCLUDED_RUNS = frozenset({"mute", "mute_spin-v2", "zips", "reference", "causalvc"})
EXCLUDED_MODEL_DIRS = frozenset({"pretraineds", "embedders", "predictors", "causalvc"})

_zluda_lock = threading.Lock()
_zluda_loaded = False


def ensure_zluda():
    global _zluda_loaded
    if _zluda_loaded:
        return
    with _zluda_lock:
        if not _zluda_loaded:
            import shared.zluda  # noqa: F401

            _zluda_loaded = True


def _read_config():
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def read_settings():
    config = _read_config()
    precision = config.get("precision", "fp16")
    if precision not in PRECISIONS:
        precision = "fp16"
    author = str(config.get("model_author") or "").strip()
    if author.lower() == "none":
        author = ""
    return {"precision": precision, "model_author": author}


def write_settings(args):
    config = _read_config()
    if "precision" in args and args["precision"] is not None:
        precision = str(args["precision"]).strip().lower()
        if precision not in PRECISIONS:
            raise ValueError("Precision must be fp16, bf16, or fp32.")
        config["precision"] = precision
    if "model_author" in args and args["model_author"] is not None:
        author = str(args["model_author"] or "").strip()
        config["model_author"] = author or "None"
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=CONFIG_PATH.parent, prefix=".config-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_path, CONFIG_PATH)
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
    return read_settings()


def absolute(path):
    return Path(str(path or "").strip().strip('"')).expanduser().resolve()


def require_path(path, label, *, directory=False):
    if not str(path or "").strip().strip('"'):
        raise ValueError(f"Select {label}.")
    resolved = absolute(path)
    if not (resolved.is_dir() if directory else resolved.is_file()):
        raise FileNotFoundError(f"{label.capitalize()} could not be found: {resolved}")
    return resolved


def model_name(name):
    name = str(name or "").strip()
    if not re.fullmatch(r"[\w.-]+", name) or name.startswith(".") or name.endswith("."):
        raise ValueError("Use letters, numbers, dashes or underscores in the model name.")
    return name


def match_index(model_path):
    """Find the most likely RVC .index file accompanying a .pth voice model."""
    if not model_path or is_causal_voice(model_path):
        return ""
    model = absolute(model_path)
    if not model.is_file():
        return ""
    folder = model.parent
    candidates = sorted(
        path for path in folder.glob("*.index")
        if path.is_file() and not path.name.startswith("trained_")
    )
    if not candidates:
        return ""
    if len(candidates) == 1:
        return str(candidates[0])
    stem = model.stem.lower()
    base = re.sub(r"_\d+e_\d+s$", "", stem)
    for target in (stem, base):
        for candidate in candidates:
            if candidate.stem.lower() == target:
                return str(candidate)
        for candidate in candidates:
            if target and target in candidate.stem.lower():
                return str(candidate)
    return str(candidates[0])


def _iter_user_model_files(pattern):
    if not MODELS.is_dir():
        return
    for path in MODELS.rglob(pattern):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(MODELS).parts
        if rel_parts and rel_parts[0].lower() in EXCLUDED_MODEL_DIRS:
            continue
        yield path


def catalog():
    rvc_paths = set()
    index_paths = set()
    if LOGS.is_dir():
        for path in LOGS.rglob("*.pth"):
            if path.is_file() and not path.name.startswith(("G_", "D_")):
                rvc_paths.add(str(path))
        for path in LOGS.rglob("*.index"):
            if path.is_file() and not path.name.startswith("trained_"):
                index_paths.add(str(path))
    for path in _iter_user_model_files("*.pth"):
        if not path.name.startswith(("G_", "D_")):
            rvc_paths.add(str(path))
    for path in _iter_user_model_files("*.index"):
        if not path.name.startswith("trained_"):
            index_paths.add(str(path))

    runs = sorted(
        folder.name for folder in (LOGS.iterdir() if LOGS.is_dir() else ())
        if folder.is_dir() and folder.name not in EXCLUDED_RUNS
    )
    causal_runs = sorted(
        folder.name.removeprefix("voice-")
        for folder in (VOICE_RUNS.iterdir() if VOICE_RUNS.is_dir() else ())
        if folder.is_dir() and folder.name.startswith("voice-")
    )
    causal_voices = {str(ROOT / path) for path in available_voices()}
    for path in _iter_user_model_files("*"):
        if is_causal_voice(path):
            causal_voices.add(str(path))
    return {
        "rvc": sorted(rvc_paths),
        "causal": sorted(causal_voices),
        "indexes": sorted(index_paths),
        "runs": runs,
        "causal_runs": causal_runs,
    }
