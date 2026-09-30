import torch
from torch import nn


MIN_FAN_IN = 16


def _iteration_dtype(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.dtype:
    return dtype if device.type == "cuda" else torch.float32


def orthogonalize(g: torch.Tensor, steps: int = 5, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    x = x / x.flatten(-2).norm(dim=-1).clamp_min(1e-7)[..., None, None]
    x = x.to(_iteration_dtype(g.device, dtype))
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return (x.mT if transposed else x).to(g.dtype)


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
                 momentum=0.95, betas=(0.9, 0.98), eps=1e-8, iteration_dtype=torch.float32):
        if iteration_dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError(f"Unsupported Muon iteration dtype: {iteration_dtype}")
        self.iteration_dtype = iteration_dtype
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
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            lr, decay = group["lr"], group["weight_decay"]
            if decay > 0:
                torch._foreach_mul_(params, 1.0 - lr * decay)
            if group["muon"]:
                self._muon(group, params, lr)
            else:
                self._adamw(group, params, lr)

    def _muon(self, group, params, lr):
        grads = [p.grad for p in params]
        buffers = [self.state[p].setdefault("momentum_buffer", torch.zeros_like(p)) for p in params]
        torch._foreach_lerp_(buffers, grads, 1.0 - group["momentum"])
        updates = torch._foreach_lerp(grads, buffers, group["momentum"])

        shapes = {}
        for p, update in zip(params, updates):
            update = update.reshape(p.shape[0], -1)
            tall = update.shape[0] > update.shape[1]
            shapes.setdefault(tuple(sorted(update.shape)), []).append((p, update.mT if tall else update, tall))
        for shape, members in shapes.items():
            orthogonal = orthogonalize(torch.stack([update for _, update, _ in members]), dtype=self.iteration_dtype).unbind(0)
            torch._foreach_add_(
                [p for p, _, _ in members],
                [(u.mT if tall else u).reshape(p.shape) for (p, _, tall), u in zip(members, orthogonal)],
                alpha=-lr * max(shape) ** 0.5,
            )

    def _adamw(self, group, params, lr):
        beta1, beta2 = group["betas"]
        by_step = {}
        for p in params:
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            by_step.setdefault(state["step"], []).append(p)
        for step, members in by_step.items():
            grads = [p.grad for p in members]
            exp_avg = [self.state[p]["exp_avg"] for p in members]
            exp_avg_sq = [self.state[p]["exp_avg_sq"] for p in members]
            torch._foreach_lerp_(exp_avg, grads, 1.0 - beta1)
            torch._foreach_mul_(exp_avg_sq, beta2)
            torch._foreach_addcmul_(exp_avg_sq, grads, grads, value=1.0 - beta2)
            denom = torch._foreach_div(exp_avg_sq, 1.0 - beta2 ** step)
            torch._foreach_sqrt_(denom)
            torch._foreach_add_(denom, group["eps"])
            torch._foreach_addcdiv_(members, exp_avg, denom, value=-lr / (1.0 - beta1 ** step))
