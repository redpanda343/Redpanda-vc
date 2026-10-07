import logging
from collections import OrderedDict

import torch


logger = logging.getLogger(__name__)


def _tensor_signature(value):
    if value is None:
        return None
    return tuple(value.shape), value.dtype, value.device


class _CapturedCall:
    def __init__(self, function, inputs):
        self.inputs = tuple(value.clone() for value in inputs)
        device = self.inputs[0].device
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            current = torch.cuda.current_stream(device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(3):
                    function(*self.inputs)
            current.wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream), torch.inference_mode():
                self.output = function(*self.inputs)
            current.wait_stream(stream)

    def replay(self, inputs):
        for target, value in zip(self.inputs, inputs):
            target.copy_(value)
        self.graph.replay()
        return self.output.clone()


class RealtimeGraph:
    def __init__(self, function, name, enabled=True, max_entries=4):
        self.function = function
        self.name = name
        self.enabled = bool(enabled)
        self.max_entries = int(max_entries)
        self.entries = OrderedDict()
        self.failed = False

    def _fail(self, message, error=None):
        self.failed = True
        self.entries.clear()
        logger.warning("Realtime %s CUDA Graph %s; using eager inference%s", self.name, message,
                       f": {error}" if error is not None else ".")

    @torch.inference_mode()
    def __call__(self, *inputs):
        if not self.enabled or self.failed or inputs[0].device.type != "cuda":
            return self.function(*inputs)
        signature = tuple(_tensor_signature(value) for value in inputs)
        entry = self.entries.get(signature)
        if entry is None:
            free_memory, _ = torch.cuda.mem_get_info(inputs[0].device)
            if free_memory < 512 * 1024 * 1024:
                self._fail("disabled: insufficient free GPU memory")
                return self.function(*inputs)
            try:
                entry = _CapturedCall(self.function, inputs)
            except (RuntimeError, NotImplementedError, MemoryError) as error:
                self._fail("unavailable", error)
                return self.function(*inputs)
            self.entries[signature] = entry
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
        else:
            self.entries.move_to_end(signature)
        try:
            return entry.replay(inputs)
        except (RuntimeError, NotImplementedError) as error:
            self._fail("replay failed", error)
            return self.function(*inputs)


class _FlowSampleGraph:
    def __init__(self, model, args, kwargs):
        self.args = tuple(value.clone() for value in args)
        self.kwargs = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in kwargs.items()
        }
        content, mask = args[0], args[-1]
        self.noise = torch.zeros(
            content.shape[0], model.n_mels, mask.shape[-1] - kwargs.get("start", 0), device=content.device
        )
        self.kwargs["noise"] = self.noise
        with torch.cuda.device(content.device):
            stream = torch.cuda.Stream(device=content.device)
            current = torch.cuda.current_stream(content.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(3):
                    model.sample(*self.args, **self.kwargs)
            current.wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream), torch.inference_mode():
                self.output = model.sample(*self.args, **self.kwargs)
            current.wait_stream(stream)

    def replay(self, args, kwargs, noise):
        for target, value in zip(self.args, args):
            target.copy_(value)
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                self.kwargs[key].copy_(value)
        self.noise.copy_(noise)
        self.graph.replay()
        return self.output


class RealtimeFlowSampler:
    def __init__(self, model, enabled=True):
        self.model = model
        self.enabled = bool(enabled)
        self.failed = False
        self.entry = None
        self.signature = None

    @torch.inference_mode()
    def __call__(self, content, f0, speaker, mask, steps=None, variances=None, start=0):
        args = (content, f0, speaker, mask)
        kwargs = {
            "steps": self.model.sampling_steps if steps is None else int(steps),
            "method": self.model.sampling_method,
        }
        if variances is not None:
            kwargs["variances"] = variances
        if start:
            kwargs["start"] = int(start)
        if not self.enabled or self.failed or content.device.type != "cuda":
            return self.model.sample(*args, **kwargs)
        signature = (
            tuple(_tensor_signature(value) for value in args), _tensor_signature(variances),
            kwargs["steps"], kwargs["method"], self.model.t_start_infer, int(start),
        )
        if signature != self.signature:
            self.entry = None
            self.signature = None
        if self.entry is None:
            free_memory, _ = torch.cuda.mem_get_info(content.device)
            if free_memory < 512 * 1024 * 1024:
                self.failed = True
                logger.warning("Realtime flow CUDA Graph disabled: insufficient free GPU memory.")
                return self.model.sample(*args, **kwargs)
            try:
                self.entry = _FlowSampleGraph(self.model, args, kwargs)
                self.signature = signature
            except (RuntimeError, NotImplementedError, MemoryError) as error:
                self.failed = True
                self.entry = None
                logger.warning("Realtime flow CUDA Graph unavailable; using eager inference: %s", error)
                return self.model.sample(*args, **kwargs)
        noise = torch.randn(
            content.shape[0], self.model.n_mels, mask.shape[-1] - int(start), device=content.device
        )
        try:
            return self.entry.replay(args, kwargs, noise)
        except (RuntimeError, NotImplementedError) as error:
            self.failed = True
            self.entry = None
            self.signature = None
            logger.warning("Realtime flow CUDA Graph replay failed; using eager inference: %s", error)
            return self.model.sample(*args, noise=noise, **kwargs)
