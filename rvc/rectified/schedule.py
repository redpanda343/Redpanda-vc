import math

def learning_rate(base, step, warmup, total, final_ratio, schedule="cosine", decay_step=4000, gamma=0.9, step_offset=1, min_lr=0.0, step_warmup=False):
    if not math.isfinite(min_lr) or min_lr < 0:
        raise ValueError("Minimum learning rate must be finite and nonnegative.")
    floor = min(base, min_lr)
    if schedule == "step":
        if decay_step <= 0 or not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
            raise ValueError("Step LR requires a positive decay_step and gamma between 0 and 1.")
        warmup = warmup if step_warmup else 0
        if warmup and step < warmup:
            return base * (step + 1) / warmup
        return max(floor, base * gamma ** max((step - warmup - step_offset) // decay_step, 0))
    if schedule != "cosine":
        raise ValueError(f"Unsupported LR schedule: {schedule}")
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return max(floor, base * (final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))
