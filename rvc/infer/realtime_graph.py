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

    def reset(self):
        self.failed = False
        self.entries.clear()

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
