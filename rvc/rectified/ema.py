from __future__ import annotations

import contextlib

import torch


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


class WeightEMA:
    DEFAULT_DECAY = 0.999

    def __init__(self, model, decay: float = DEFAULT_DECAY, warmup: bool = True):
        self.decay = min(1.0, max(0.0, float(decay)))
        self.warmup = bool(warmup)
        self.updates = 0

        state = _unwrap(model).state_dict()
        self.shadow = {key: value.detach().clone() for key, value in state.items()}


        self._live_float = []
        self._shadow_float = []
        self._other_keys = []
        for key, value in state.items():
            if value.is_floating_point():
                self._live_float.append(value)
                self._shadow_float.append(self.shadow[key])
            else:
                self._other_keys.append(key)

    def _decay_at(self, update: int) -> float:
        if not self.warmup or update < 1:
            return self.decay
        return min(self.decay, 1.0 - 1.0 / update)

    def current_decay(self) -> float:
        return self._decay_at(self.updates)

    @torch.no_grad()
    def update(self, model, steps: int = 1) -> None:
        steps = int(steps)
        if steps < 1:
            return
        effective_decay = 1.0
        for update in range(self.updates + 1, self.updates + steps + 1):
            effective_decay *= self._decay_at(update)
        self.updates += steps

        torch._foreach_lerp_(self._shadow_float, self._live_float, 1.0 - effective_decay)
        if self._other_keys:
            state = _unwrap(model).state_dict()
            for key in self._other_keys:
                self.shadow[key].copy_(state[key])

    @contextlib.contextmanager
    def applied(self, model):
        module = _unwrap(model)
        backup = {
            key: value.detach().to("cpu", copy=True)
            for key, value in module.state_dict().items()
        }
        module.load_state_dict(self.shadow, strict=True)
        try:
            yield module
        finally:
            module.load_state_dict(backup, strict=True)

    def cpu_state_dict(self) -> dict:
        return {key: value.detach().to("cpu", copy=True) for key, value in self.shadow.items()}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    @torch.no_grad()
    def reseed(self, model) -> None:
        state = _unwrap(model).state_dict()
        for key, value in self.shadow.items():
            value.copy_(state[key])
        self.updates = 0

    @torch.no_grad()
    def load_state_dict(self, data, model) -> bool:
        shadow = (data or {}).get("shadow")
        if not shadow or any(key not in shadow for key in self.shadow):
            self.reseed(model)
            return False
        for key, value in self.shadow.items():
            value.copy_(shadow[key])
        self.updates = int(data.get("updates", 0))
        return True
