import json
import math
import os
from pathlib import Path

from causalvc.voice import CVC_SETTINGS, CVC_SUFFIX, SUFFIXES, is_cvc, settings_in


ROOT = Path(os.environ.get("REDPANDA_DATA_ROOT") or Path(__file__).resolve().parents[2]).resolve()
VOICE_RUNS = Path(os.environ.get("CAUSALVC_RUNS") or (ROOT / "runs" / "models")).resolve()
SAMPLE_RATE = 48000
VOICE_SUFFIXES = (CVC_SUFFIX, *SUFFIXES)

# The order matches CausalVC.change_pitch_map.
PITCH_MAP_DEFAULTS = {
    "pitch_offset_hz": 0.0,
    "pitch_ceiling_hz": 0.0,
    "pitch_ceiling_keep": 0.25,
    "pitch_floor_hz": 0.0,
    "pitch_floor_keep": 0.25,
}
PITCH_MAP_CONTROLS = {
    "pitch_offset_hz": ("Pitch offset (Hz)", -300, 300, 1),
    "pitch_ceiling_hz": ("Pitch ceiling (Hz)", 0, 1000, 1),
    "pitch_ceiling_keep": ("Share kept above ceiling", 0, 1, 0.01),
    "pitch_floor_hz": ("Pitch floor (Hz)", 0, 500, 1),
    "pitch_floor_keep": ("Share kept below floor", 0, 1, 0.01),
}
VOICE_DEFAULTS = {
    "pitch": 0.0,
    "index_rate": 0.0,
    "threshold": -60.0,
    "input_gain_db": 0.0,
    "output_gain_db": 0.0,
    "monitor_gain_db": 0.0,
    **PITCH_MAP_DEFAULTS,
}


def is_causal_voice(path):
    return str(path).lower().endswith(VOICE_SUFFIXES)


def voice_settings(path):
    values = VOICE_DEFAULTS.copy()
    if not is_causal_voice(path):
        return values
    try:
        if is_cvc(path):
            data = settings_in(path)
        else:
            with Path(path).with_name(CVC_SETTINGS).open("rb") as settings:
                data = settings.read((1 << 20) + 1)
        saved = json.loads(data) if data and len(data) <= (1 << 20) else None
    except (OSError, ValueError):
        return values
    if not isinstance(saved, dict):
        return values

    for key in values:
        # Upstream spells the noise gate key "threhold" in its voice files.
        value = saved.get(key)
        if key == "threshold":
            value = saved.get("threhold", saved.get("noise_gate_db", value))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        try:
            number = float(value)
        except OverflowError:
            continue
        if math.isfinite(number):
            values[key] = number
    return values


def available_voices():
    folders = (ROOT / "logs", *VOICE_RUNS.glob("voice-*/voices"))
    return sorted(
        path.relative_to(ROOT).as_posix()
        for folder in folders
        for path in folder.rglob("*")
        if is_causal_voice(path) and path.is_file()
    )
