from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as audio_functional
from sklearn import metrics

from rvc.algorithm.ecapa_tdnn import load_ecapa_tdnn


def _error_rates(scores, labels):
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_total = sum(labels)
    negative_total = len(labels) - positive_total
    false_negatives = 0
    false_positives = negative_total
    fnrs = []
    fprs = []
    thresholds = []
    for score, label in ordered:
        if label == 1:
            false_negatives += 1
        else:
            false_positives -= 1
        fnrs.append(false_negatives / positive_total)
        fprs.append(false_positives / negative_total)
        thresholds.append(score)
    return fnrs, fprs, thresholds


def _eer(scores, labels):
    fprs, tprs, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnrs = 1 - tprs
    index = np.nanargmin(np.absolute(fnrs - fprs))
    return max(fprs[index], fnrs[index]) * 100.0, thresholds[index]


def _min_dcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1.0, c_fa=1.0):
    costs = [
        c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
        for fnr, fpr in zip(fnrs, fprs)
    ]
    index = min(range(len(costs)), key=costs.__getitem__)
    default_cost = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return costs[index] / default_cost, thresholds[index]


class ECAPATimbreValidator:
    sample_rate = 16000
    segment_samples = 300 * 160 + 240
    segment_count = 5
    cpu_segment_batch_size = 20
    cuda_segment_batch_size = 10
    cuda_memory_headroom = 512 * 1024 * 1024

    def __init__(self, model_path):
        self.model = load_ecapa_tdnn(model_path, device="cpu")
        self.device = torch.device("cpu")
        self.references = {}

    def to(self, device):
        device = torch.device(device)
        self.model.to(device)
        self.references = {
            speaker_id: (full.to(device), segments.to(device))
            for speaker_id, (full, segments) in self.references.items()
        }
        self.device = device
        return self

    def preferred_device(self, device):
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            free_memory, _ = torch.cuda.mem_get_info(device)
            reusable_memory = max(
                0,
                torch.cuda.memory_reserved(device)
                - torch.cuda.memory_allocated(device),
            )
            if free_memory + reusable_memory >= self.cuda_memory_headroom:
                return device
        except RuntimeError:
            pass
        return torch.device("cpu")

    def _prepare_audio(self, audio, sample_rate):
        audio = audio.detach().to(device=self.device, dtype=torch.float32)
        while audio.dim() > 1:
            audio = audio.mean(dim=0)
        if audio.numel() < 2:
            audio = F.pad(audio, (0, 2 - audio.numel()))
        if sample_rate != self.sample_rate:
            audio = audio_functional.resample(audio, sample_rate, self.sample_rate)
        return audio.contiguous()

    def _segments(self, audio):
        if audio.numel() < self.segment_samples:
            repeats = (self.segment_samples + audio.numel() - 1) // audio.numel()
            audio = audio.repeat(repeats)[: self.segment_samples]
        starts = torch.linspace(
            0,
            audio.numel() - self.segment_samples,
            steps=self.segment_count,
        ).long()
        return torch.stack(
            [audio[start : start + self.segment_samples] for start in starts]
        )

    @torch.inference_mode()
    def embeddings(self, audio, sample_rate):
        audio = self._prepare_audio(audio, sample_rate)
        full = F.normalize(self.model(audio.unsqueeze(0), aug=False), p=2, dim=1)
        segments = F.normalize(self.model(self._segments(audio), aug=False), p=2, dim=1)
        return full, segments

    @torch.inference_mode()
    def batch_embeddings(self, audio, lengths, sample_rate):
        audio = audio.detach().to(device=self.device, dtype=torch.float32)
        length_values = lengths.detach().cpu().tolist()
        prepared = [
            self._prepare_audio(audio[index, :, : int(length_values[index])], sample_rate)
            for index in range(audio.size(0))
        ]
        full = torch.cat(
            [
                F.normalize(self.model(item.unsqueeze(0), aug=False), p=2, dim=1)
                for item in prepared
            ],
            dim=0,
        )
        segment_audio = torch.cat([self._segments(item) for item in prepared], dim=0)
        segment_embeddings = []
        segment_batch_size = (
            self.cuda_segment_batch_size
            if self.device.type == "cuda"
            else self.cpu_segment_batch_size
        )
        for start in range(0, segment_audio.size(0), segment_batch_size):
            segment_embeddings.append(
                F.normalize(
                    self.model(
                        segment_audio[start : start + segment_batch_size], aug=False
                    ),
                    p=2,
                    dim=1,
                )
            )
        segments = torch.cat(segment_embeddings, dim=0).view(
            len(prepared), self.segment_count, -1
        )
        return full, segments

    @torch.inference_mode()
    def similarity(self, first, second, sample_rate):
        first_full, first_segments = self.embeddings(first, sample_rate)
        second_full, second_segments = self.embeddings(second, sample_rate)
        full_score = torch.mean(first_full @ second_full.T)
        segment_score = torch.mean(first_segments @ second_segments.T)
        return float(((full_score + segment_score) / 2).item())

    def set_references(self, natural, natural_lengths, speaker_ids, sample_rate):
        references = defaultdict(lambda: {"full": [], "segments": []})
        for index in range(natural.size(0)):
            natural_audio = natural[index, :, : int(natural_lengths[index])]
            full, segments = self.embeddings(natural_audio, sample_rate)
            speaker_id = int(speaker_ids[index])
            references[speaker_id]["full"].append(full)
            references[speaker_id]["segments"].append(segments)
        self.references = {
            speaker_id: (
                torch.cat(values["full"], dim=0),
                torch.cat(values["segments"], dim=0),
            )
            for speaker_id, values in sorted(references.items())
        }

    def score_batch(
        self,
        generated,
        generated_lengths,
        speaker_ids,
        sample_rate,
    ):
        if not self.references:
            raise ValueError("No ECAPA speaker references are available")
        scores = []
        negative_scores = []
        closest_negative_scores = []
        margins = []
        correct_predictions = []
        scores_by_speaker = defaultdict(list)
        margins_by_speaker = defaultdict(list)
        accuracy_by_speaker = defaultdict(list)
        generated_full, generated_segments = self.batch_embeddings(
            generated, generated_lengths, sample_rate
        )
        for index in range(generated.size(0)):
            speaker_id = int(speaker_ids[index])
            if speaker_id not in self.references:
                raise ValueError(f"No ECAPA reference exists for speaker {speaker_id}")
            probe_full = generated_full[index : index + 1]
            probe_segments = generated_segments[index]
            trial_scores = {}
            for reference_speaker_id, references in self.references.items():
                reference_full, reference_segments = references
                full_score = torch.mean(probe_full @ reference_full.T)
                segment_score = torch.mean(
                    probe_segments @ reference_segments.T
                )
                trial_scores[reference_speaker_id] = float(
                    ((full_score + segment_score) / 2).item()
                )
            positive_score = trial_scores[speaker_id]
            predicted_speaker = max(trial_scores, key=trial_scores.get)
            correct = float(predicted_speaker == speaker_id)
            scores.append(positive_score)
            scores_by_speaker[speaker_id].append(positive_score)
            correct_predictions.append(correct)
            accuracy_by_speaker[speaker_id].append(correct)
            probe_negatives = [
                score
                for reference_speaker_id, score in trial_scores.items()
                if reference_speaker_id != speaker_id
            ]
            if probe_negatives:
                closest_negative = max(probe_negatives)
                margin = positive_score - closest_negative
                negative_scores.extend(probe_negatives)
                closest_negative_scores.append(closest_negative)
                margins.append(margin)
                margins_by_speaker[speaker_id].append(margin)
        tensor_scores = torch.tensor(scores, dtype=torch.float32)
        speaker_scores = {
            speaker_id: sum(values) / len(values)
            for speaker_id, values in sorted(scores_by_speaker.items())
        }
        result = {
            "mean": float(tensor_scores.mean().item()),
            "max": float(tensor_scores.max().item()),
            "speaker_mean": sum(speaker_scores.values()) / len(speaker_scores),
            "speakers": speaker_scores,
            "reference_speaker_count": len(self.references),
            "probe_count": len(scores),
        }
        if not negative_scores:
            result["multi_speaker"] = False
            return result

        labels = [1] * len(scores) + [0] * len(negative_scores)
        trial_scores = scores + negative_scores
        eer, eer_threshold = _eer(trial_scores, labels)
        fnrs, fprs, thresholds = _error_rates(trial_scores, labels)
        min_dcf, min_dcf_threshold = _min_dcf(fnrs, fprs, thresholds)
        result.update(
            {
                "multi_speaker": True,
                "negative_mean": sum(negative_scores) / len(negative_scores),
                "negative_max": max(negative_scores),
                "closest_negative_mean": sum(closest_negative_scores)
                / len(closest_negative_scores),
                "margin_mean": sum(margins) / len(margins),
                "margin_min": min(margins),
                "top1_accuracy_percent": sum(correct_predictions)
                / len(correct_predictions)
                * 100.0,
                "eer_percent": eer,
                "eer_threshold": eer_threshold,
                "min_dcf": min_dcf,
                "min_dcf_threshold": min_dcf_threshold,
                "positive_trial_count": len(scores),
                "negative_trial_count": len(negative_scores),
                "speaker_margins": {
                    speaker_id: sum(values) / len(values)
                    for speaker_id, values in sorted(margins_by_speaker.items())
                },
                "speaker_top1_accuracy_percent": {
                    speaker_id: sum(values) / len(values) * 100.0
                    for speaker_id, values in sorted(accuracy_by_speaker.items())
                },
            }
        )
        return result

    def score_batch_accelerated(
        self,
        generated,
        generated_lengths,
        speaker_ids,
        sample_rate,
        preferred_device,
    ):
        target_device = self.preferred_device(preferred_device)
        try:
            try:
                self.to(target_device)
                return self.score_batch(
                    generated,
                    generated_lengths,
                    speaker_ids,
                    sample_rate,
                )
            except RuntimeError as error:
                if target_device.type != "cuda" or "out of memory" not in str(
                    error
                ).lower():
                    raise
                self.to("cpu")
                return self.score_batch(
                    generated,
                    generated_lengths,
                    speaker_ids,
                    sample_rate,
                )
        finally:
            self.to("cpu")
