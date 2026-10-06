from concurrent.futures import ProcessPoolExecutor

import numpy as np

PARSELMOUTH_F0_MIN = 65.0
PARSELMOUTH_F0_MAX = 1100.0


def uses_parselmouth(data):
    return data.get('pitch_extractor') == 'parselmouth'


def interpolate_f0(f0):
    f0 = np.asarray(f0, dtype=np.float32).copy()
    if f0.ndim != 1 or not len(f0) or not np.isfinite(f0).all() or (f0 < 0).any():
        raise ValueError('Pitch extraction returned invalid frequencies.')
    voiced = f0 > 0
    if voiced.any():
        positions = np.arange(len(f0))
        f0 = np.exp2(np.interp(positions, positions[voiced], np.log2(f0[voiced]))).astype(np.float32)
    return f0


def parselmouth_f0(waveform, sample_rate, hop, frames, f0_min=PARSELMOUTH_F0_MIN, f0_max=PARSELMOUTH_F0_MAX):
    import parselmouth

    waveform = np.asarray(waveform)
    left = int(np.ceil(1.5 / f0_min * sample_rate))
    right = hop * ((len(waveform) - 1) // hop + 1) - len(waveform) + left + 1
    waveform = np.pad(waveform, (left, right))
    contour = parselmouth.Sound(waveform, sampling_frequency=sample_rate).to_pitch_ac(
        time_step=hop / sample_rate, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max,
    ).selected_array['frequency'].astype(np.float32)
    contour = np.pad(contour, (0, max(0, frames - len(contour))))[:frames]
    return interpolate_f0(contour)


def has_voiced_frames(path, hop):
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(-1)
    return bool(parselmouth_f0(audio, sample_rate, hop, max(1, len(audio) // hop)).any())


def drop_unvoiced_clips(filelist, hop, workers=1):
    with open(filelist, encoding='utf-8') as handle:
        rows = [row for row in handle.read().splitlines() if row.strip()]
    paths = [row.split('|')[0] for row in rows]
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        voiced = list(executor.map(has_voiced_frames, paths, [hop] * len(paths), chunksize=16))
    for path, keep in zip(paths, voiced):
        if not keep:
            print(f"Skipped '{path}': empty gt f0")
    kept = [row for row, keep in zip(rows, voiced) if keep]
    if rows and not kept:
        raise RuntimeError('Parselmouth found no voiced frames in any training clip.')
    print(f'Parselmouth F0: kept {len(kept):,} of {len(rows):,} clip(s).')
    with open(filelist, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(kept))
