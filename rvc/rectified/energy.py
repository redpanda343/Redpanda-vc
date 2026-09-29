import torch
from torch.nn import functional as F


FRAME_RATE = 100

WINDOW_SECONDS = 0.032

FLOOR_DB = -70.0


def frame_energy(audio: torch.Tensor, sample_rate: int, frames: int) -> torch.Tensor:
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
    hop = int(sample_rate) // FRAME_RATE
    window = int(round(WINDOW_SECONDS * sample_rate)) | 1
    power = F.avg_pool1d(
        audio.float().pow(2),
        kernel_size=window,
        stride=hop,
        padding=window // 2,
        count_include_pad=False,
    )
    if power.shape[-1] < frames:
        power = F.pad(power, (0, frames - power.shape[-1]), mode="replicate")
    db = (10.0 * torch.log10(power[..., :frames].clamp_min(1e-10))).clamp(FLOOR_DB, 0.0)
    return (db / (-FLOOR_DB / 2.0) + 1.0).squeeze(1)
