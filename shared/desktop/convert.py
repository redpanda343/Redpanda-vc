"""Render CVC files through the same streaming model as microphone conversion."""

import gc

from shared.desktop.paths import AUDIO_SUFFIXES
from shared.tools.causal_voices import VOICE_DEFAULTS


def convert_causal(model, source, target, options):
    import numpy as np
    import soundfile as sf
    import torch

    from rvc.configs.config import Config
    from rvc.infer.realtime_engine import db_to_linear
    from shared.tools.causal_realtime import CausalRealtime, SAMPLE_RATE
    from shared.utils import load_audio_infer

    settings = {"model_path": str(model), "block_time": 0.04}
    settings.update({key: float(options[key]) for key in VOICE_DEFAULTS if key in options})
    files = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES) if source.is_dir() else [source]
    if not files:
        raise ValueError("No audio files were found in the input folder.")
    stream = None
    try:
        with torch.inference_mode():
            stream = CausalRealtime(settings, torch.device(Config().device))
            delay = round(stream.delay_seconds * SAMPLE_RATE)
            for index, path in enumerate(files, 1):
                out = target / f"{path.stem}_converted.{options.get('format', 'wav')}" if source.is_dir() else target
                print(f"[{index}/{len(files)}] {path.name}", flush=True)
                samples = load_audio_infer(str(path), SAMPLE_RATE).astype(np.float32, copy=False)
                if len(samples) == 0:
                    raise ValueError(f"{path.name} is empty.")
                frames = stream.prepare(SAMPLE_RATE)
                blocks = (len(samples) + delay + frames - 1) // frames
                audio = np.pad(samples, (0, blocks * frames - len(samples)))
                np.multiply(audio, np.float32(db_to_linear(settings["input_gain_db"])), out=audio)
                np.clip(audio, -1, 1, out=audio)
                output = np.empty_like(audio)
                gain = db_to_linear(settings["output_gain_db"])
                for start in range(0, len(audio), frames):
                    span = slice(start, start + frames)
                    output[span] = stream.output_gain(stream.process(audio[span]), gain)
                sf.write(out, output[delay:delay + len(samples)], SAMPLE_RATE)
    finally:
        del stream
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
