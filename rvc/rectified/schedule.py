import math

def learning_rate(base, step, warmup, total, final_ratio):
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return base * (final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))

def freeze_voice(model) -> int:
    modules = [model.backbone.time_mlp, model.encoder.speaker_proj, model.backbone.voice]
    modules += [layer.modulation for layer in model.backbone.layers]
    frozen = 0
    for module in filter(None, modules):
        for param in module.parameters():
            param.requires_grad_(False)
            frozen += 1
    return frozen
