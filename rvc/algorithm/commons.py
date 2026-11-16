import math
import torch
from typing import Optional

_fourier_bases = {}


def init_weights(m, mean=0.0, std=0.01):
    """
    Initialize the weights of a module.

    Args:
        m: The module to initialize.
        mean: The mean of the normal distribution.
        std: The standard deviation of the normal distribution.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class _LeakyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, negative_slope):
        y = torch.nn.functional.leaky_relu(x, negative_slope)
        ctx.negative_slope = negative_slope
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        return (
            torch.ops.aten.leaky_relu_backward(grad, y, ctx.negative_slope, True),
            None,
        )


def leaky_relu(x: torch.Tensor, negative_slope: float):
    if torch.is_grad_enabled() and x.requires_grad:
        return _LeakyReLU.apply(x, negative_slope)
    return torch.nn.functional.leaky_relu(x, negative_slope)


def get_padding(kernel_size, dilation=1):
    """
    Calculate the padding needed for a convolution.

    Args:
        kernel_size: The size of the kernel.
        dilation: The dilation of the convolution.
    """
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    """
    Convert the pad shape to a list of integers.

    Args:
        pad_shape: The pad shape..
    """
    if isinstance(pad_shape, torch.Tensor):
        return pad_shape.flip(0).reshape(-1).int().tolist()
    return [int(value) for pair in reversed(pad_shape) for value in pair]


def slice_segments(
    x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4, dim: int = 3
):
    """
    Slice segments from a tensor, handling tensors with different numbers of dimensions.

    Args:
        x (torch.Tensor): The tensor to slice.
        ids_str (torch.Tensor): The starting indices of the segments.
        segment_size (int, optional): The size of each segment. Defaults to 4.
        dim (int, optional): The dimension to slice across (2D or 3D tensors). Defaults to 3.
    """
    offsets = torch.arange(segment_size, device=x.device)
    if dim == 2:
        return x.gather(1, ids_str.view(-1, 1) + offsets)
    index = ids_str.view(-1, 1, 1) + offsets
    return x.gather(2, index.expand(x.size(0), x.size(1), segment_size))


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """
    Randomly slice segments from a tensor.

    Args:
        x: The tensor to slice.
        x_lengths: The lengths of the sequences.
        segment_size: The size of each segment.
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    rand_vals = (
        torch.rand([b], device=x.device)
        if x.device.type in ("xla", "tpu")
        else torch.rand([b]).to(device=x.device, non_blocking=True)
    )
    ids_str = (rand_vals * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size, dim=3)
    return ret, ids_str


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """
    Fused add tanh sigmoid multiply operation.

    Args:
        input_a: The first input tensor.
        input_b: The second input tensor.
        n_channels: The number of channels.
    """
    # Avoid reading a device scalar (.item()) on XLA/TPU for every layer.
    n_channels_int = n_channels if isinstance(n_channels, int) else int(n_channels[0])
    in_act = input_a + input_b if input_b is not None else input_a
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None):
    """
    Generate a sequence mask.

    Args:
        length: The lengths of the sequences.
        max_length: The maximum length of the sequences.
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def _fourier_basis(n_fft: int, device):
    key = (n_fft, str(device))
    if key not in _fourier_bases:
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
        angle = 2 * math.pi * k.unsqueeze(1) * n.unsqueeze(0) / n_fft
        basis = torch.cat((torch.cos(angle), -torch.sin(angle)), dim=0)
        _fourier_bases[key] = basis.to(dtype=torch.float32).to(device)
    return _fourier_bases[key]


def stft_real_imag(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor,
    center: bool = False,
):
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        win_length = window.shape[-1]
        window = window.float()
        if win_length < n_fft:
            left = (n_fft - win_length) // 2
            window = torch.nn.functional.pad(window, (left, n_fft - win_length - left))
        x = x.float().unsqueeze(1)
        if center:
            x = torch.nn.functional.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
        basis = _fourier_basis(n_fft, x.device) * window
        frames = torch.nn.functional.conv1d(x, basis.unsqueeze(1), stride=hop_length)
        return frames.chunk(2, dim=1)


def upsample_nearest_1d(x: torch.Tensor, factor: int) -> torch.Tensor:
    if x.device.type in ("xla", "tpu"):
        return (
            x.unsqueeze(-1)
            .expand(*x.shape, factor)
            .reshape(*x.shape[:-1], x.shape[-1] * factor)
        )
    return torch.nn.functional.interpolate(
        x, scale_factor=float(factor), mode="nearest"
    )


_TPU_LINEAR1D_BACKWARD_SUPPORTED = None


def _needs_linear1d_bilinear_fallback(device_type: str) -> bool:
    if device_type == "xla":
        return True
    if device_type != "tpu":
        return False
    global _TPU_LINEAR1D_BACKWARD_SUPPORTED
    if _TPU_LINEAR1D_BACKWARD_SUPPORTED is None:
        checker = getattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", None)
        try:
            _TPU_LINEAR1D_BACKWARD_SUPPORTED = bool(
                checker is not None
                and checker("aten::upsample_linear1d_backward.grad_input", "PrivateUse1")
            )
        except RuntimeError:
            _TPU_LINEAR1D_BACKWARD_SUPPORTED = False
    return not _TPU_LINEAR1D_BACKWARD_SUPPORTED


def upsample_linear_1d(
    x: torch.Tensor,
    size: int | None = None,
    scale_factor: int | float | None = None,
) -> torch.Tensor:
    if _needs_linear1d_bilinear_fallback(x.device.type):
        target = int(size if size is not None else x.shape[-1] * scale_factor)
        return torch.nn.functional.interpolate(
            x.unsqueeze(2),
            size=(1, target),
            mode="bilinear",
            align_corners=False,
        ).squeeze(2)
    return torch.nn.functional.interpolate(
        x, size=size, scale_factor=scale_factor, mode="linear"
    )


class UpsampleLinear1d(torch.nn.Module):
    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return upsample_linear_1d(x, scale_factor=self.scale_factor)


_XLA_CHECKPOINT_FN = None


def _get_xla_checkpoint():
    global _XLA_CHECKPOINT_FN
    if _XLA_CHECKPOINT_FN is None:
        from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint

        _XLA_CHECKPOINT_FN = xla_checkpoint
    return _XLA_CHECKPOINT_FN


def _get_xla_autocast_state() -> tuple[bool, torch.dtype]:
    try:
        return (
            bool(torch.is_autocast_enabled("xla")),
            torch.get_autocast_dtype("xla"),
        )
    except TypeError:
        return (
            bool(torch.is_autocast_xla_enabled()),
            torch.get_autocast_xla_dtype(),
        )


def activation_checkpoint(
    module: torch.nn.Module,
    x: torch.Tensor,
    *,
    tuple_with_list: bool = False,
):
    if not isinstance(x, torch.Tensor) or x.device.type != "xla":
        return torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False)

    needs_grad = torch.is_grad_enabled() and (
        x.requires_grad or any(p.requires_grad for p in module.parameters())
    )
    if not needs_grad:
        return module(x)

    xla_checkpoint = _get_xla_checkpoint()
    autocast_enabled, autocast_dtype = _get_xla_autocast_state()

    if tuple_with_list:

        def _run(inp, _dummy=None):
            with torch.amp.autocast(
                device_type="xla", enabled=autocast_enabled, dtype=autocast_dtype
            ):
                score, fmap = module(inp)
                return (score, *fmap)

    else:

        def _run(inp, _dummy=None):
            with torch.amp.autocast(
                device_type="xla", enabled=autocast_enabled, dtype=autocast_dtype
            ):
                return module(inp)

    if x.requires_grad:
        out = xla_checkpoint(_run, x, use_reentrant=True, preserve_rng_state=True)
    else:
        dummy = torch.zeros((), device=x.device, dtype=x.dtype, requires_grad=True)
        out = xla_checkpoint(
            _run, x, dummy, use_reentrant=True, preserve_rng_state=True
        )

    if tuple_with_list:
        return out[0], list(out[1:])
    return out


def grad_norm(parameters, norm_type: float = 2.0):
    """
    Calculates norm of parameter gradients

    Args:
        parameters: The list of parameters to clip.
        norm_type: The type of norm to use for clipping.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    gradients = [p.grad.detach().float() for p in parameters if p.grad is not None]
    if not gradients:
        return 0.0

    with torch.amp.autocast(device_type=gradients[0].device.type, enabled=False):
        total_norm = torch.nn.utils.get_total_norm(
            gradients,
            norm_type=float(norm_type),
            foreach=None,
        )
    return total_norm
