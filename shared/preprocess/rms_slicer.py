import numpy as np


RMS_FRAMES_PER_CHUNK = 4096


class Slicer:
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError("min_length >= min_interval >= hop_size is required")
        if not max_sil_kept >= hop_size:
            raise ValueError("max_sil_kept >= hop_size is required")

        min_interval_samples = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval_samples), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval_samples / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        start_idx = begin * self.hop_size
        if waveform.ndim > 1:
            end_idx = min(waveform.shape[1], end * self.hop_size)
            return waveform[:, start_idx:end_idx]
        end_idx = min(waveform.shape[0], end * self.hop_size)
        return waveform[start_idx:end_idx]

    def slice(self, waveform):
        samples = waveform.mean(axis=0) if waveform.ndim > 1 else waveform
        if (samples.shape[0] + self.hop_size - 1) // self.hop_size <= self.min_length:
            return [waveform]

        rms_list = get_rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(0)
        sil_tags = []
        silence_start, clip_start = None, 0
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue

            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[
                    i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()
                pos += i - self.max_sil_kept
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None

        total_frames = rms_list.shape[0]
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))

        if not sil_tags:
            return [waveform]

        chunks = []
        if sil_tags[0][0] > 0:
            chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
        for i in range(len(sil_tags) - 1):
            chunks.append(
                self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0])
            )
        if sil_tags[-1][1] < total_frames:
            chunks.append(
                self._apply_slice(waveform, sil_tags[-1][1], total_frames)
            )
        return chunks


def _legacy_get_rms(y, frame_length, hop_length, pad_mode):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)
    axis = -1
    out_strides = y.strides + (y.strides[axis],)
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + (frame_length,)
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    target_axis = axis - 1 if axis < 0 else axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)


def get_rms(
    y,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
    frames_per_chunk=RMS_FRAMES_PER_CHUNK,
):
    y = np.asarray(y)
    if y.ndim != 1:
        return _legacy_get_rms(y, frame_length, hop_length, pad_mode)

    half_frame = frame_length // 2
    window_count = y.shape[0] + 2 * half_frame - frame_length + 1
    if window_count <= 0:
        return _legacy_get_rms(y, frame_length, hop_length, pad_mode)
    total_frames = (window_count - 1) // hop_length + 1
    padded = None
    if pad_mode != "constant":
        padded = np.pad(y, (half_frame, half_frame), mode=pad_mode)
    power_chunks = []
    for first_frame in range(0, total_frames, frames_per_chunk):
        frame_count = min(frames_per_chunk, total_frames - first_frame)
        if padded is not None:
            padded_start = first_frame * hop_length
            padded_end = (
                (first_frame + frame_count - 1) * hop_length
                + frame_length
            )
            block = padded[padded_start:padded_end]
        else:
            block_start = first_frame * hop_length - half_frame
            block_end = (
                (first_frame + frame_count - 1) * hop_length
                - half_frame
                + frame_length
            )
            source_start = max(0, block_start)
            source_end = min(y.shape[0], block_end)
            block = y[source_start:source_end]
            left_padding = source_start - block_start
            right_padding = block_end - source_end
            if left_padding or right_padding:
                block = np.pad(
                    block,
                    (left_padding, right_padding),
                    mode="constant",
                )
        frames = np.lib.stride_tricks.sliding_window_view(
            block, frame_length
        )[::hop_length]
        power_chunks.append(np.mean(np.square(frames), axis=-1))

    power = np.concatenate(power_chunks)
    return np.sqrt(power)[np.newaxis, :]
