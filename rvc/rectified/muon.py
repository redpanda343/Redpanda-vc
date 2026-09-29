import torch
from torch import nn
from torch.nn import functional as F


MIN_FAN_IN = 16


def orthogonalize(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = F.normalize(g.float().flatten(), dim=0, eps=1e-7).view_as(g)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x.T if transposed else x).to(g.dtype)


def muon_parameters(model: nn.Module) -> set:
    chosen = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding) or getattr(module, "use_adamw", False):
            continue
        for param in module.parameters(recurse=False):
            if param.requires_grad and param.dim() >= 2 and param[0].numel() >= MIN_FAN_IN:
                chosen.add(id(param))
    return chosen


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, model, lr, muon_weight_decay=0.1, adamw_weight_decay=0.0,
                 momentum=0.95, betas=(0.9, 0.98), eps=1e-8):
        chosen = muon_parameters(model)
        params = [p for p in model.parameters() if p.requires_grad]
        groups = [
            dict(params=[p for p in params if id(p) in chosen], muon=True,
                 weight_decay=muon_weight_decay),
            dict(params=[p for p in params if id(p) not in chosen], muon=False,
                 weight_decay=adamw_weight_decay),
        ]
        super().__init__(groups, dict(lr=lr, momentum=momentum, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, decay = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if decay > 0:
                    p.mul_(1.0 - lr * decay)
                if group["muon"]:
                    buffer = state.setdefault("momentum_buffer", torch.zeros_like(p))
                    buffer.lerp_(p.grad, 1.0 - group["momentum"])
                    update = p.grad.lerp(buffer, group["momentum"]).reshape(p.shape[0], -1)
                    update = orthogonalize(update)
                    p.add_(update.view_as(p), alpha=-lr * max(update.shape) ** 0.5)
                else:
                    beta1, beta2 = group["betas"]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    state["exp_avg"].lerp_(p.grad, 1.0 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(p.grad, p.grad, value=1.0 - beta2)
                    corrected = state["exp_avg_sq"] / (1.0 - beta2 ** state["step"])
                    step_size = lr / (1.0 - beta1 ** state["step"])
                    p.addcdiv_(state["exp_avg"], corrected.sqrt().add_(group["eps"]), value=-step_size)
