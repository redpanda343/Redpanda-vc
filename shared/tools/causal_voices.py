from pathlib import Path

from causalvc.voice import is_cvc


ROOT = Path(__file__).resolve().parents[2]
VOICE_RUNS = ROOT / "runs" / "models"
SAMPLE_RATE = 48000


def available_voices():
    voices = set(VOICE_RUNS.glob("voice-*/voices/*.cvc"))
    voices.update((ROOT / "logs").rglob("*.cvc"))
    return sorted(path.relative_to(ROOT).as_posix() for path in voices if path.is_file())
