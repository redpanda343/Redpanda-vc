import os

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
from scipy import signal

from rvc.train.preprocess.slicer import Slicer
from rvc.train.preprocess.rms_slicer import Slicer as AutomaticSlicer
from rvc.train.preprocess.audio import (
    _clean_audio_path,
    iter_high_pass_audio,
    iter_resampled_audio_ffmpeg,
    load_audio_ffmpeg,
)
from rvc.train.preprocess.constants import (
    ALPHA,
    HIGH_PASS_CUTOFF,
    MAX_AMPLITUDE,
    MINIMUM_AUTOMATIC_SOURCE_AUDIO_SECONDS,
    MINIMUM_OUTPUT_AUDIO_SECONDS,
    OVERLAP,
    PERCENTAGE,
    POST_NORMALIZATION_MAX_GAIN,
    SIMPLE_MIN_SILENCE_SECONDS,
    SIMPLE_SILENCE_COMPRESS_PERCENT,
    SIMPLE_SILENCE_THRESHOLD_DB,
    SIMPLE_STREAM_THRESHOLD_BYTES,
    SIMPLE_TRUNCATE_TO_SECONDS,
)
from rvc.train.preprocess.dataset import (
    BoundedAudioWriter,
    normalize_dataset_format,
    write_training_audio,
)
from rvc.train.preprocess.silence import (
    find_streaming_silence_cuts,
    iter_audio_with_silence_cuts,
    truncate_silence,
)


class PreProcess:
    def __init__(
        self,
        sr: int,
        exp_dir: str,
        dataset_format: str = "wav",
        use_fireredvad_gpu: bool = False,
    ):
        self.post_normalization_slicer = Slicer(
            sr=sr,
            use_gpu=use_fireredvad_gpu,
        )
        self.automatic_slicer = AutomaticSlicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.b_high, self.a_high = signal.butter(
            N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=self.sr
        )
        self.exp_dir = exp_dir
        self.dataset_format = normalize_dataset_format(dataset_format)
        self.audio_write_workers = 1
        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)

    def _normalize_audio(self, audio: np.ndarray):
        tmp_max = np.abs(audio).max()
        if tmp_max > 2.5:
            return None
        return (audio / tmp_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio

    @staticmethod
    def _post_normalization_gain(voice_peak: float):
        if not np.isfinite(voice_peak) or voice_peak <= 0:
            return 1.0
        return min(MAX_AMPLITUDE / voice_peak, POST_NORMALIZATION_MAX_GAIN)

    def _detect_post_normalization_gain(self, audio: np.ndarray):
        voiced_chunks = self.post_normalization_slicer.slice(audio)
        if not voiced_chunks:
            return 1.0
        voice_peak = max(float(np.max(np.abs(chunk))) for chunk in voiced_chunks)
        return self._post_normalization_gain(voice_peak)

    def _peak_normalize_audio(self, audio: np.ndarray, source_gain: float):
        if audio.size == 0:
            return audio
        peak = np.abs(audio).max()
        if not np.isfinite(peak) or peak > 2.5:
            return None
        if peak == 0:
            return audio
        gain = min(source_gain, MAX_AMPLITUDE / peak)
        return audio * gain

    def process_audio_segment(
        self,
        normalized_audio: np.ndarray,
        sid: int,
        idx0: int,
        idx1: int,
        normalization_mode: str,
        normalization_gain: float = 1.0,
        writer: BoundedAudioWriter | None = None,
    ):
        if normalized_audio is None:
            print(f"{sid}-{idx0}-{idx1}-filtered")
            return
        if normalization_mode == "post":
            normalized_audio = self._peak_normalize_audio(
                normalized_audio, normalization_gain
            )
        args = (
            self.gt_wavs_dir,
            f"{sid}_{idx0}_{idx1}",
            self.sr,
            normalized_audio,
            self.dataset_format,
        )
        if writer is None:
            if len(normalized_audio) < round(
                self.sr * MINIMUM_OUTPUT_AUDIO_SECONDS
            ):
                return 1
            write_training_audio(*args)
            return 0
        return writer.submit(*args)

    def simple_cut(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
        normalization_gain: float = 1.0,
    ):
        chunk_length = int(self.sr * chunk_len)
        overlap_length = int(self.sr * overlap_len)
        with BoundedAudioWriter(self.audio_write_workers) as writer:
            i = 0
            while i < len(audio):
                chunk = audio[i : i + chunk_length]
                if normalization_mode == "post":
                    chunk = self._peak_normalize_audio(chunk, normalization_gain)
                writer.submit(
                    self.gt_wavs_dir,
                    f"{sid}_{idx0}_{i // (chunk_length - overlap_length)}",
                    self.sr,
                    chunk,
                    self.dataset_format,
                )
                i += chunk_length - overlap_length
        return writer.skipped_short

    def simple_cut_stream(
        self,
        blocks,
        sid: int,
        idx0: int,
        chunk_len: float,
        overlap_len: float,
    ):
        chunk_length = int(self.sr * chunk_len)
        overlap_length = int(self.sr * overlap_len)
        step_length = chunk_length - overlap_length
        buffer = np.empty(0, dtype=np.float32)
        buffer_start = 0
        next_start = 0
        with BoundedAudioWriter(self.audio_write_workers) as writer:
            for block in blocks:
                block = np.asarray(block)
                if block.size == 0:
                    continue
                buffer = (
                    np.concatenate((buffer, block))
                    if buffer.size
                    else block.copy()
                )
                buffer_end = buffer_start + len(buffer)
                while next_start + chunk_length <= buffer_end:
                    local_start = next_start - buffer_start
                    chunk = buffer[local_start : local_start + chunk_length].copy()
                    writer.submit(
                        self.gt_wavs_dir,
                        f"{sid}_{idx0}_{next_start // step_length}",
                        self.sr,
                        chunk,
                        self.dataset_format,
                    )
                    next_start += step_length
                    drop_frames = next_start - buffer_start
                    buffer = buffer[drop_frames:]
                    buffer_start = next_start
                    buffer_end = buffer_start + len(buffer)
            if next_start < buffer_start + len(buffer):
                local_start = next_start - buffer_start
                writer.submit(
                    self.gt_wavs_dir,
                    f"{sid}_{idx0}_{next_start // step_length}",
                    self.sr,
                    buffer[local_start:].copy(),
                    self.dataset_format,
                )
        return writer.skipped_short

    def should_stream_simple_audio(
        self,
        path: str,
        noise_reduction: bool,
        normalization_mode: str,
    ) -> bool:
        if noise_reduction or normalization_mode != "none":
            return False
        try:
            duration = float(sf.info(_clean_audio_path(path)).duration)
        except (RuntimeError, TypeError, ValueError, OSError):
            return False
        return duration * self.sr * np.dtype(np.float32).itemsize >= (
            SIMPLE_STREAM_THRESHOLD_BYTES
        )

    def process_simple_audio_streaming(
        self,
        path: str,
        idx0: int,
        sid: int,
        process_effects: bool,
        chunk_len: float,
        overlap_len: float,
        truncate_silence_threshold_db: float,
        truncate_silence_to_seconds: float,
        truncate_silence_minimum_seconds: float,
        truncate_silence_action: str,
        truncate_silence_compress_percent: float,
    ):
        cuts, total_frames = find_streaming_silence_cuts(
            iter_resampled_audio_ffmpeg(path, self.sr),
            self.sr,
            truncate_silence_threshold_db,
            truncate_silence_minimum_seconds,
            truncate_silence_to_seconds,
            truncate_silence_action,
            truncate_silence_compress_percent,
        )
        blocks = iter_audio_with_silence_cuts(
            iter_resampled_audio_ffmpeg(path, self.sr),
            cuts,
            total_frames,
        )
        if process_effects:
            blocks = iter_high_pass_audio(blocks, self.b_high, self.a_high)
        skipped_short = self.simple_cut_stream(
            blocks,
            sid,
            idx0,
            chunk_len,
            overlap_len,
        )
        return total_frames / self.sr, skipped_short

    def process_simple_audio(
        self,
        path: str,
        idx0: int,
        sid: int,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
        truncate_silence_enabled: bool = False,
        truncate_silence_threshold_db: float = SIMPLE_SILENCE_THRESHOLD_DB,
        truncate_silence_to_seconds: float = SIMPLE_TRUNCATE_TO_SECONDS,
        truncate_silence_minimum_seconds: float = SIMPLE_MIN_SILENCE_SECONDS,
        truncate_silence_action: str = "truncate",
        truncate_silence_compress_percent: float = SIMPLE_SILENCE_COMPRESS_PERCENT,
    ):
        if truncate_silence_enabled and self.should_stream_simple_audio(
            path,
            noise_reduction,
            normalization_mode,
        ):
            return self.process_simple_audio_streaming(
                path,
                idx0,
                sid,
                process_effects,
                chunk_len,
                overlap_len,
                truncate_silence_threshold_db,
                truncate_silence_to_seconds,
                truncate_silence_minimum_seconds,
                truncate_silence_action,
                truncate_silence_compress_percent,
            )
        audio = load_audio_ffmpeg(path, self.sr)
        audio_length = len(audio) / self.sr
        if truncate_silence_enabled:
            audio = truncate_silence(
                audio,
                self.sr,
                threshold_db=truncate_silence_threshold_db,
                minimum_silence=truncate_silence_minimum_seconds,
                truncate_to=truncate_silence_to_seconds,
                action=truncate_silence_action,
                compress_percent=truncate_silence_compress_percent,
            )
        audio = self._prepare_audio(
            audio,
            process_effects,
            noise_reduction,
            reduction_strength,
            normalization_mode,
        )
        normalization_gain = 1.0
        if normalization_mode == "post" and audio is not None:
            normalization_gain = self._detect_post_normalization_gain(audio)
        skipped_short = self.simple_cut(
            audio,
            sid,
            idx0,
            chunk_len,
            overlap_len,
            normalization_mode,
            normalization_gain,
        )
        return audio_length, skipped_short

    def _prepare_audio(
        self,
        audio: np.ndarray,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        normalization_mode: str,
    ):
        if process_effects:
            audio = signal.lfilter(self.b_high, self.a_high, audio)
        if normalization_mode == "pre":
            audio = self._normalize_audio(audio)
        if noise_reduction and audio is not None and audio.size:
            audio = nr.reduce_noise(
                y=audio, sr=self.sr, prop_decrease=reduction_strength
            )
        return audio

    def _process_automatic(
        self,
        path: str,
        idx0: int,
        sid: int,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        normalization_mode: str,
    ):
        audio = load_audio_ffmpeg(path, self.sr)
        duration_s = librosa.get_duration(y=audio, sr=self.sr)
        if duration_s < MINIMUM_AUTOMATIC_SOURCE_AUDIO_SECONDS:
            return 0.0, 0
        audio = self._prepare_audio(
            audio,
            process_effects,
            noise_reduction,
            reduction_strength,
            normalization_mode,
        )
        if audio is None:
            return duration_s, 0
        normalization_gain = 1.0
        if normalization_mode == "post":
            normalization_gain = self._detect_post_normalization_gain(audio)

        segments = self.automatic_slicer.slice(audio)
        idx1 = 0
        step_samples = int(self.sr * (PERCENTAGE - OVERLAP))
        chunk_samples = int(self.sr * PERCENTAGE)
        long_tail_samples = int(self.sr * (PERCENTAGE + OVERLAP))
        with BoundedAudioWriter(self.audio_write_workers) as writer:
            for segment in segments:
                start = 0
                while start < len(segment):
                    if len(segment) - start > long_tail_samples:
                        chunk = segment[start : start + chunk_samples]
                    else:
                        chunk = segment[start:]
                    self.process_audio_segment(
                        chunk,
                        sid,
                        idx0,
                        idx1,
                        normalization_mode,
                        normalization_gain,
                        writer,
                    )
                    idx1 += 1
                    if len(segment) - start <= long_tail_samples:
                        break
                    start += step_samples
        return duration_s, writer.skipped_short

    def process_audio(
        self,
        path: str,
        idx0: int,
        sid: int,
        cut_preprocess: str,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
    ):
        audio_length = 0
        skipped_short = 0
        try:
            if cut_preprocess == "Automatic":
                return self._process_automatic(
                    path,
                    idx0,
                    sid,
                    process_effects,
                    noise_reduction,
                    reduction_strength,
                    normalization_mode,
                )

            audio = load_audio_ffmpeg(path, self.sr)
            audio_length = librosa.get_duration(y=audio, sr=self.sr)
            audio = self._prepare_audio(
                audio,
                process_effects,
                noise_reduction,
                reduction_strength,
                normalization_mode,
            )
            normalization_gain = 1.0
            if normalization_mode == "post" and audio is not None:
                normalization_gain = self._detect_post_normalization_gain(audio)
            if cut_preprocess == "Skip":

                skipped_short = self.process_audio_segment(
                    audio,
                    sid,
                    idx0,
                    0,
                    normalization_mode,
                    normalization_gain,
                )
            elif cut_preprocess == "Simple":

                skipped_short = self.simple_cut(
                    audio,
                    sid,
                    idx0,
                    chunk_len,
                    overlap_len,
                    normalization_mode,
                    normalization_gain,
                )
        except Exception as error:
            print(f"Error processing audio: {error}")
            if cut_preprocess == "Automatic" or self.dataset_format in {
                "flac",
                "wav_float32",
            }:
                raise
        return audio_length, skipped_short
