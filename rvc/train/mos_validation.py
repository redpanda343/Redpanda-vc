import gc
import importlib.util
import os
import random
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

from rvc.train.validation_data import MAXIMUM_VALIDATION_SECONDS


@contextmanager
def deterministic_validation_scope(seed, cuda_devices=None):
    seed = int(seed) % (2**63 - 1)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    previous_settings = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.fp32_precision,
        torch.backends.cudnn.fp32_precision,
    )
    devices = list(cuda_devices or [])
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
            try:
                yield
            finally:
                (
                    deterministic_algorithms,
                    deterministic_warn_only,
                    cudnn_benchmark,
                    cudnn_deterministic,
                    matmul_precision,
                    cudnn_precision,
                ) = previous_settings
                torch.use_deterministic_algorithms(
                    deterministic_algorithms,
                    warn_only=deterministic_warn_only,
                )
                torch.backends.cudnn.benchmark = cudnn_benchmark
                torch.backends.cudnn.deterministic = cudnn_deterministic
                torch.backends.cuda.matmul.fp32_precision = matmul_precision
                torch.backends.cudnn.fp32_precision = cudnn_precision
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class UTMOSv2Validator:
    def __init__(self, model_paths, device):
        missing_paths = [path for path in model_paths if not os.path.isfile(path)]
        if missing_paths:
            raise FileNotFoundError(
                f"UTMOSv2 checkpoint not found: {missing_paths[0]}"
            )
        if importlib.util.find_spec("utmosv2") is None:
            raise ModuleNotFoundError("UTMOSv2 is not installed")
        self.model_paths = list(model_paths)
        self.device = torch.device(device)
        self.repetitions = 5

    def _load_model(self, model_path, fold):
        import utmosv2

        model = utmosv2.create_model(
            pretrained=True,
            fold=fold,
            checkpoint_path=model_path,
            device="cpu",
        )
        model.eval().float().to("cpu")
        floating_dtypes = {
            parameter.dtype
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
        if floating_dtypes != {torch.float32}:
            raise RuntimeError(
                f"UTMOSv2 must use FP32 parameters, got {sorted(map(str, floating_dtypes))}"
            )
        return model

    @staticmethod
    def _prepare_clips(generated, lengths, speaker_ids, sample_rate):
        generated = generated.detach().to(device="cpu", dtype=torch.float32)
        length_values = lengths.detach().cpu().tolist()
        speaker_values = speaker_ids.detach().cpu().tolist()
        minimum_length = int(2 * sample_rate)
        maximum_length = int(MAXIMUM_VALIDATION_SECONDS * sample_rate)
        clips = []
        speakers = []
        for index, requested_length in enumerate(length_values):
            clip = generated[index]
            while clip.dim() > 1:
                clip = clip.mean(dim=0)
            valid_length = min(max(0, int(requested_length)), clip.numel())
            clip = clip[:valid_length].contiguous()
            if clip.numel() < minimum_length or not torch.isfinite(clip).all():
                continue
            if clip.abs().mean().item() <= 1e-6:
                continue
            if clip.numel() > maximum_length:
                start = (clip.numel() - maximum_length) // 2
                clip = clip[start : start + maximum_length].contiguous()
            clips.append(clip.numpy())
            speakers.append(int(speaker_values[index]))
        if not clips:
            raise ValueError("No valid generated audio is available for UTMOSv2")
        return clips, speakers

    @staticmethod
    def _resample_clip(model, clip, sample_rate):
        import torchaudio

        audio = torch.from_numpy(clip)
        target_sample_rate = int(model._cfg.sr)
        if int(sample_rate) != target_sample_rate:
            audio = torchaudio.functional.resample(
                audio,
                int(sample_rate),
                target_sample_rate,
            )
        return audio.numpy().astype(np.float32, copy=False)

    @staticmethod
    def _prepare_tta_inputs(model, clip, repetitions):
        from utmosv2.dataset._schema import InMemoryData
        from utmosv2.utils import get_dataset

        data = InMemoryData(
            data=clip,
            dataset_name="sarulab",
        )
        initial_state = getattr(model._cfg.dataset, "remove_silent_section", None)
        model._cfg.dataset.remove_silent_section = False
        try:
            dataset = get_dataset(model._cfg, data, model._cfg.phase)
            samples = [dataset[0] for _ in range(repetitions)]
        finally:
            model._cfg.dataset.remove_silent_section = initial_state
        return [
            torch.stack([sample[index] for sample in samples])
            for index in range(len(samples[0]) - 1)
        ]

    @staticmethod
    def _score_input_batches(model, inputs, device, batch_size):
        predictions = []
        with torch.inference_mode():
            for start in range(0, inputs[0].shape[0], batch_size):
                batch = [
                    value[start : start + batch_size].to(device) for value in inputs
                ]
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    prediction = model(*batch).reshape(-1).float().cpu()
                predictions.append(prediction)
                del prediction, batch
        return torch.cat(predictions)

    def _score_tta(self, model, inputs, device):
        repetitions = int(inputs[0].shape[0])
        batch_sizes = [repetitions]
        if device.type == "cuda":
            batch_sizes.extend([max(1, repetitions // 2), 1])
        batch_sizes = list(dict.fromkeys(batch_sizes))
        for batch_size in batch_sizes:
            try:
                predictions = self._score_input_batches(
                    model, inputs, device, batch_size
                )
            except RuntimeError as error:
                if device.type != "cuda" or "out of memory" not in str(error).lower():
                    raise
                if batch_size == 1:
                    raise
                gc.collect()
                torch.cuda.empty_cache()
                next_batch_size = batch_sizes[batch_sizes.index(batch_size) + 1]
                print(
                    f"UTMOSv2 TTA batch {batch_size} ran out of VRAM; retrying "
                    f"with batch {next_batch_size}."
                )
                continue
            if not torch.isfinite(predictions).all():
                raise RuntimeError("UTMOSv2 returned invalid TTA predictions")
            result = float(predictions.mean().item())
            del predictions
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return result
        raise RuntimeError("UTMOSv2 TTA inference failed")

    def _score_clips(self, model, clips, sample_rate, device):
        model.eval().float().to(device)
        predictions = []
        for clip in clips:
            prepared_clip = self._resample_clip(model, clip, sample_rate)
            inputs = self._prepare_tta_inputs(
                model, prepared_clip, self.repetitions
            )
            prediction = self._score_tta(model, inputs, device)
            del inputs
            predictions.append(prediction)
        return predictions

    def _score_fold(self, model, clips, sample_rate):
        target_device = self.device
        try:
            return self._score_clips(
                model, clips, sample_rate, target_device
            )
        except RuntimeError as error:
            if target_device.type != "cuda" or "out of memory" not in str(error).lower():
                raise
            model.to("cpu")
            torch.cuda.empty_cache()
            print("UTMOSv2 GPU validation ran out of VRAM; retrying safely on CPU.")
            return self._score_clips(
                model, clips, sample_rate, torch.device("cpu")
            )

    def score_batch(self, generated, lengths, speaker_ids, sample_rate):
        clips, speakers = self._prepare_clips(
            generated, lengths, speaker_ids, sample_rate
        )
        fold_predictions = []
        for fold, model_path in enumerate(self.model_paths):
            print(f"UTMOSv2 validating fold {fold + 1}/{len(self.model_paths)}.")
            model = self._load_model(model_path, fold)
            try:
                fold_predictions.append(
                    self._score_fold(model, clips, sample_rate)
                )
            finally:
                model.to("cpu")
                del model
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        predictions = np.asarray(fold_predictions, dtype=np.float32).mean(axis=0)
        predictions = predictions.reshape(-1)
        if predictions.size != len(speakers) or not np.isfinite(predictions).all():
            raise RuntimeError("UTMOSv2 returned invalid predictions")
        by_speaker = defaultdict(list)
        for speaker_id, prediction in zip(speakers, predictions):
            by_speaker[speaker_id].append(float(prediction))
        speaker_means = [sum(values) / len(values) for values in by_speaker.values()]
        return sum(speaker_means) / len(speaker_means)
