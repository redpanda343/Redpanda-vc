import numpy as np

from rvc.train.preprocess.audio import SequentialAudioReader
from rvc.train.preprocess.constants import (
    SIMPLE_BLEND_FRAMES,
    SIMPLE_MIN_SILENCE_SECONDS,
    SIMPLE_SILENCE_COMPRESS_PERCENT,
    SIMPLE_SILENCE_THRESHOLD_DB,
    SIMPLE_TRUNCATE_TO_SECONDS,
)


def truncate_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = SIMPLE_SILENCE_THRESHOLD_DB,
    minimum_silence: float = SIMPLE_MIN_SILENCE_SECONDS,
    truncate_to: float = SIMPLE_TRUNCATE_TO_SECONDS,
    blend_frames: int = SIMPLE_BLEND_FRAMES,
    action: str = "truncate",
    compress_percent: float = SIMPLE_SILENCE_COMPRESS_PERCENT,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio

    action = normalize_silence_action(action)
    compress_percent = validate_silence_compress_percent(compress_percent)
    threshold = 10.0 ** (threshold_db / 20.0)
    silent = np.abs(audio) < threshold
    boundaries = np.flatnonzero(
        np.diff(np.pad(silent.astype(np.int8), (1, 1)))
    )
    if boundaries.size == 0:
        return audio

    minimum_frames = max(1, int(round(minimum_silence * sample_rate)))
    truncate_frames = max(0, int(round(truncate_to * sample_rate)))
    cuts = []
    if action == "truncate":
        for start, end in boundaries.reshape(-1, 2):
            silence_frames = int(end - start)
            if silence_frames < minimum_frames:
                continue
            output_frames = min(truncate_frames, silence_frames)
            cut_frames = silence_frames - output_frames
            if cut_frames <= 0:
                continue
            cut_start = int(start + output_frames // 2)
            cuts.append((cut_start, cut_start + cut_frames))
    else:
        for start, end in boundaries.reshape(-1, 2):
            silence_frames = int(end - start)
            if silence_frames < minimum_frames:
                continue
            excess_frames = silence_frames - minimum_frames
            output_frames = minimum_frames + int(
                round(excess_frames * compress_percent / 100.0)
            )
            cut_frames = silence_frames - output_frames
            if cut_frames <= 0:
                continue
            cut_start = int(start + output_frames // 2)
            cuts.append((cut_start, cut_start + cut_frames))

    if not cuts:
        return audio

    parts = []
    cursor = 0
    for cut_start, cut_end in cuts:
        splice_frames = min(
            blend_frames,
            cut_start * 2,
            (len(audio) - cut_end) * 2,
        )
        if splice_frames > 0:
            half_blend = splice_frames // 2
            blend_start = cut_start - half_blend
            right_start = cut_end - half_blend
            left = audio[blend_start : blend_start + splice_frames]
            right = audio[right_start : right_start + splice_frames]
            weights = np.arange(splice_frames, dtype=np.float32) / splice_frames
            blended = left * (1.0 - weights) + right * weights
            parts.append(audio[cursor:blend_start])
            parts.append(blended)
            cursor = right_start + splice_frames
        else:
            parts.append(audio[cursor:cut_start])
            cursor = cut_end

    parts.append(audio[cursor:])
    return np.concatenate(parts)


def normalize_silence_action(action: str) -> str:
    normalized = str(action).strip().lower()
    if normalized not in {"truncate", "compress"}:
        raise ValueError(f"Unsupported silence action: {action}")
    return normalized


def validate_silence_compress_percent(compress_percent: float) -> float:
    compress_percent = float(compress_percent)
    if not 0.0 <= compress_percent <= 99.9:
        raise ValueError("Silence compression must be between 0.0 and 99.9 percent")
    return compress_percent


def silence_cut(
    start: int,
    end: int,
    minimum_frames: int,
    truncate_frames: int,
    action: str,
    compress_percent: float,
):
    silence_frames = end - start
    if silence_frames < minimum_frames:
        return None
    if action == "compress":
        excess_frames = silence_frames - minimum_frames
        output_frames = minimum_frames + int(
            round(excess_frames * compress_percent / 100.0)
        )
    else:
        output_frames = min(truncate_frames, silence_frames)
    output_frames = min(output_frames, silence_frames)
    cut_frames = silence_frames - output_frames
    if cut_frames <= 0:
        return None
    cut_start = start + output_frames // 2
    return cut_start, cut_start + cut_frames


def find_streaming_silence_cuts(
    blocks,
    sample_rate: int,
    threshold_db: float,
    minimum_silence: float,
    truncate_to: float,
    action: str,
    compress_percent: float,
):
    action = normalize_silence_action(action)
    compress_percent = validate_silence_compress_percent(compress_percent)
    threshold = 10.0 ** (threshold_db / 20.0)
    minimum_frames = max(1, int(round(minimum_silence * sample_rate)))
    truncate_frames = max(0, int(round(truncate_to * sample_rate)))
    previous_silent = False
    silence_start = None
    total_frames = 0
    cuts = []
    for block in blocks:
        block = np.asarray(block, dtype=np.float32)
        if block.size == 0:
            continue
        silent = np.abs(block) < threshold
        transitions = np.flatnonzero(
            np.diff(silent.astype(np.int8), prepend=np.int8(previous_silent))
        )
        for index in transitions:
            position = total_frames + int(index)
            if silent[index]:
                silence_start = position
            elif silence_start is not None:
                if position - silence_start >= minimum_frames:
                    cut = silence_cut(
                        silence_start,
                        position,
                        minimum_frames,
                        truncate_frames,
                        action,
                        compress_percent,
                    )
                    if cut is not None:
                        cuts.append(cut)
                silence_start = None
        previous_silent = bool(silent[-1])
        total_frames += len(block)
    if previous_silent and silence_start is not None:
        if total_frames - silence_start >= minimum_frames:
            cut = silence_cut(
                silence_start,
                total_frames,
                minimum_frames,
                truncate_frames,
                action,
                compress_percent,
            )
            if cut is not None:
                cuts.append(cut)
    return cuts, total_frames


def iter_audio_with_silence_cuts(
    blocks, cuts, total_frames: int, blend_frames: int = SIMPLE_BLEND_FRAMES
):
    reader = SequentialAudioReader(blocks)
    for cut_start, cut_end in cuts:
        splice_frames = min(
            blend_frames,
            cut_start * 2,
            (total_frames - cut_end) * 2,
        )
        half_blend = splice_frames // 2
        blend_start = cut_start - half_blend
        yield from reader.iter_read(blend_start - reader.position)
        if splice_frames > 0:
            left = reader.read(splice_frames)
            right_start = cut_end - half_blend
            if right_start < reader.position:
                overlap_start = right_start - blend_start
                right_prefix = left[overlap_start:]
                right_suffix = reader.read(splice_frames - len(right_prefix))
                right = np.concatenate((right_prefix, right_suffix))
            else:
                reader.discard_to(right_start)
                right = reader.read(splice_frames)
            weights = (
                np.arange(splice_frames, dtype=np.float32) / splice_frames
            )
            yield left * (1.0 - weights) + right * weights
        else:
            reader.discard_to(cut_end)
    yield from reader.iter_read(total_frames - reader.position)
