import math

import numpy as np

from shared.preprocess.audio import SequentialAudioReader
from shared.preprocess.constants import (
    SIMPLE_BLEND_FRAMES,
    SIMPLE_MIN_SILENCE_SECONDS,
    SIMPLE_SILENCE_COMPRESS_PERCENT,
    SIMPLE_SILENCE_THRESHOLD_DB,
    SIMPLE_TRUNCATE_TO_SECONDS,
)

MIN_DETECTED_SILENCE_SECONDS = 0.001


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
    cuts, total_frames = find_streaming_silence_cuts(
        (audio,),
        sample_rate,
        threshold_db,
        minimum_silence,
        truncate_to,
        action,
        compress_percent,
        blend_frames,
    )
    if not cuts:
        return audio
    parts = list(iter_audio_with_silence_cuts((audio,), cuts, total_frames))
    if not parts:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


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


def silence_threshold(threshold_db: float) -> np.float32:
    threshold = 10.0 ** (float(threshold_db) / 20.0)
    threshold32 = np.float32(threshold)
    if float(threshold32) >= threshold:
        return threshold32
    return np.nextafter(threshold32, np.float32(np.inf))


def minimum_silence_frames(minimum_silence: float, sample_rate: int) -> int:
    return max(1, int(max(float(minimum_silence), MIN_DETECTED_SILENCE_SECONDS) * sample_rate))


def time_to_samples(seconds: float, sample_rate: int) -> int:
    return int(math.floor(seconds * sample_rate + 0.5))


def silence_cut(
    start: int,
    end: int,
    sample_rate: int,
    minimum_silence: float,
    truncate_to: float,
    action: str,
    compress_percent: float,
    blend_frames: int = SIMPLE_BLEND_FRAMES,
):
    region_start = start / sample_rate
    region_end = end / sample_rate
    in_length = region_end - region_start
    if in_length < minimum_silence - 0.000000001:
        return None
    if action == "compress":
        out_length = minimum_silence + (in_length - minimum_silence) * compress_percent / 100.0
    else:
        out_length = min(truncate_to, in_length)
    cut_length = max(0.0, in_length - out_length)
    if cut_length == 0.0:
        return None
    cut_start = (region_start + region_end - cut_length) / 2
    cut_end = cut_start + cut_length
    blend = int(blend_frames)
    if blend / sample_rate > in_length:
        blend = time_to_samples(in_length, sample_rate)
    return (
        time_to_samples(cut_start, sample_rate),
        time_to_samples(cut_end, sample_rate),
        max(0, blend),
    )


def find_streaming_silence_cuts(
    blocks,
    sample_rate: int,
    threshold_db: float,
    minimum_silence: float,
    truncate_to: float,
    action: str,
    compress_percent: float,
    blend_frames: int = SIMPLE_BLEND_FRAMES,
):
    action = normalize_silence_action(action)
    compress_percent = validate_silence_compress_percent(compress_percent)
    minimum_silence = float(minimum_silence)
    truncate_to = float(truncate_to)
    threshold = silence_threshold(threshold_db)
    minimum_frames = minimum_silence_frames(minimum_silence, sample_rate)

    def add_cut(start, end):
        if end - start < minimum_frames:
            return
        cut = silence_cut(
            start,
            end,
            sample_rate,
            minimum_silence,
            truncate_to,
            action,
            compress_percent,
            blend_frames,
        )
        if cut is not None:
            cuts.append(cut)

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
                add_cut(silence_start, position)
                silence_start = None
        previous_silent = bool(silent[-1])
        total_frames += len(block)
    if previous_silent and silence_start is not None:
        add_cut(silence_start, total_frames)
    return cuts, total_frames


def _silence_cut_clusters(cuts, total_frames: int):
    max_blend = max(blend for _, _, blend in cuts)
    max_half = max_blend // 2
    clusters = []
    index = 0
    while index < len(cuts):
        stop = index + 1
        reach = cuts[index][1] + max_blend
        while stop < len(cuts) and cuts[stop][0] - max_half < reach:
            reach = max(reach, cuts[stop][1] + max_blend)
            stop += 1
        ops = cuts[index:stop]
        start = max(0, min(s0 - blend // 2 for s0, _, blend in ops))
        clusters.append((ops, start, min(reach, total_frames)))
        index = stop
    later_cut = 0
    result = []
    for ops, start, end in reversed(clusters):
        result.append((ops, start, end, later_cut))
        later_cut += sum(s1 - s0 for s0, s1, _ in ops)
    return result[::-1], max_blend


class _SparseAudio:
    def __init__(self, reader, intervals, total_frames: int):
        self.total_frames = total_frames
        self.pieces = []
        for start, end in intervals:
            reader.discard_to(start)
            self.pieces.append((start, reader.read(end - start)))

    def read(self, start: int, end: int) -> np.ndarray:
        out = np.zeros(max(0, end - start), dtype=np.float32)
        low, high = max(start, 0), min(end, self.total_frames)
        if high <= low:
            return out
        for piece_start, data in self.pieces:
            if piece_start <= low and high <= piece_start + len(data):
                out[low - start : high - start] = data[low - piece_start : high - piece_start]
                return out
        raise RuntimeError("Truncate silence needed audio outside the buffered splice windows")


def _splice_windows(ops, start: int, end: int, margin: int):
    intervals = []
    for s0, s1, _ in ops:
        intervals.append((max(start, s0 - margin), min(end, s0 + margin)))
        intervals.append((max(start, s1 - margin), min(end, s1 + margin)))
    intervals.sort()
    merged = []
    for low, high in intervals:
        if high <= low:
            continue
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return merged


def _crossfade(left: np.ndarray, right: np.ndarray, blend: int) -> np.ndarray:
    index = np.arange(blend)
    mixed = (blend - index).astype(np.float32) * left + index.astype(np.float32) * right
    return (mixed.astype(np.float64) / blend).astype(np.float32)


def _iter_cluster(ops, start: int, end: int, later_cut: int, source, total_frames: int):
    count = len(ops)
    t1 = [s0 - blend // 2 for s0, _, blend in ops]
    t2 = [s1 - blend // 2 for _, s1, blend in ops]
    blends = [blend for _, _, blend in ops]
    cut = [s1 - s0 for s0, s1, _ in ops]
    lengths = [0] * count
    floors = [math.inf] * (count + 1)
    removed = later_cut
    for index in range(count - 1, -1, -1):
        removed += cut[index]
        lengths[index] = total_frames - removed
        floors[index] = min(floors[index + 1], t1[index])
    mixes = [None] * count

    def walk(level, position, stop, out):
        while position < stop:
            if position < 0:
                step = min(stop, 0)
                out.append(np.zeros(step - position, dtype=np.float32))
                position = step
                continue
            if level == count:
                out.append(source.read(position, stop))
                return
            if position < floors[level]:
                step = min(stop, floors[level])
                out.append(source.read(position, step))
                position = step
                continue
            if position < t1[level]:
                step = min(stop, t1[level])
                walk(level + 1, position, step, out)
                position = step
                continue
            window_end = t1[level] + blends[level]
            if position < window_end:
                step = min(stop, window_end)
                written = max(position, min(step, lengths[level]))
                if written > position:
                    out.append(mixes[level][position - t1[level] : written - t1[level]])
                if step > written:
                    out.append(np.zeros(step - written, dtype=np.float32))
                position = step
                continue
            position += cut[level]
            stop += cut[level]
            level += 1

    def read_level(level, position, frames):
        out = []
        walk(level, position, position + frames, out)
        if not out:
            return np.zeros(frames, dtype=np.float32)
        return np.concatenate(out)

    for index in range(count - 1, -1, -1):
        blend = blends[index]
        if blend == 0:
            mixes[index] = np.empty(0, dtype=np.float32)
            continue
        mixes[index] = _crossfade(
            read_level(index + 1, t1[index], blend),
            read_level(index + 1, t2[index], blend),
            blend,
        )
    out = []
    walk(0, start, end - sum(cut), out)
    return out


def iter_audio_with_silence_cuts(blocks, cuts, total_frames: int):
    reader = SequentialAudioReader(blocks)
    if cuts:
        clusters, max_blend = _silence_cut_clusters(cuts, total_frames)
        for ops, start, end, later_cut in clusters:
            yield from reader.iter_read(start - reader.position)
            source = _SparseAudio(
                reader, _splice_windows(ops, start, end, 2 * max_blend), total_frames
            )
            yield from _iter_cluster(ops, start, end, later_cut, source, total_frames)
            reader.discard_to(end)
    yield from reader.iter_read(total_frames - reader.position)
