import glob
import importlib.util
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

INSTALL_HINT = (
    "TPU training needs PyTorch/XLA. On a TPU VM, install the TPU requirements:\n"
    "    pip install -r requirements-tpu.txt\n"
    "See https://github.com/pytorch/xla for the current release."
)


def torch_xla_installed():
    return importlib.util.find_spec("torch_xla") is not None


def tpu_available():
    if not torch_xla_installed():
        return False
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        return True
    if any(
        os.environ.get(name)
        for name in (
            "TPU_NAME",
            "COLAB_TPU_ADDR",
            "TPU_ACCELERATOR_TYPE",
            "TPU_CHIPS_PER_HOST_BOUNDS",
            "TPU_WORKER_ID",
            "KAGGLE_TPU_NAME",
        )
    ):
        return True
    return bool(glob.glob("/dev/accel*") or glob.glob("/dev/vfio/[0-9]*"))


_PRECISION_CONFIGURED_PID = None


def require_torch_xla():
    global _PRECISION_CONFIGURED_PID
    if not torch_xla_installed():
        raise RuntimeError(INSTALL_HINT)
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.pop("XLA_USE_BF16", None)
    os.environ.pop("XLA_DOWNCAST_BF16", None)
    import torch_xla

    pid = os.getpid()
    if _PRECISION_CONFIGURED_PID != pid:
        set_matmul_precision("highest")
        patch_xla_ops()
        _PRECISION_CONFIGURED_PID = pid

    return torch_xla


def _xm():
    import torch_xla.core.xla_model as xm

    return xm


def _xr():
    import torch_xla.runtime as xr

    return xr


def spawn(fn, args, single_core=False):
    torch_xla = require_torch_xla()
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    if hasattr(torch_xla, "launch"):
        torch_xla.launch(fn, args=args, debug_single_process=single_core)
        return
    import torch_xla.distributed.xla_multiprocessing as xmp

    xmp.spawn(fn, args=args, nprocs=1 if single_core else None)


_XLA_OPS_PATCHED = False


def _compute_weight_norm_xla(weight_v, weight_g, dim):
    with torch.amp.autocast(device_type="xla", enabled=False):
        v_fp32 = weight_v.float()
        g_fp32 = weight_g.float()
        if dim == -1 or dim is None:
            norm = torch.linalg.vector_norm(v_fp32)
        else:
            target_dim = dim if dim >= 0 else v_fp32.ndim + dim
            dims = tuple(i for i in range(v_fp32.ndim) if i != target_dim)
            norm = (
                torch.linalg.vector_norm(v_fp32, dim=dims, keepdim=True)
                if dims
                else v_fp32.abs()
            )
        return (v_fp32 * (g_fp32 / norm.clamp_min(1e-12))).to(weight_v.dtype)


def patch_xla_ops():
    global _XLA_OPS_PATCHED
    if _XLA_OPS_PATCHED:
        return
    import torch
    from torch.nn.utils.parametrizations import _WeightNorm

    orig_forward = _WeightNorm.forward

    def _xla_weight_norm_forward(self, weight_g, weight_v):
        if weight_v.device.type != "xla":
            return orig_forward(self, weight_g, weight_v)
        return _compute_weight_norm_xla(weight_v, weight_g, self.dim)

    _WeightNorm.forward = _xla_weight_norm_forward

    orig_weight_norm_fn = getattr(torch, "_weight_norm", None)
    if orig_weight_norm_fn is not None:

        def _xla_weight_norm_fn(weight_v, weight_g, dim=0):
            if weight_v.device.type != "xla":
                return orig_weight_norm_fn(weight_v, weight_g, dim)
            return _compute_weight_norm_xla(weight_v, weight_g, dim)

        torch._weight_norm = _xla_weight_norm_fn

    _XLA_OPS_PATCHED = True


def device():
    torch_xla = require_torch_xla()

    if hasattr(torch_xla, "device"):
        return torch_xla.device()
    return _xm().xla_device()


def rank():
    return _xr().global_ordinal()


def world_size():
    return _xr().world_size()


def sync():
    import torch_xla

    if hasattr(torch_xla, "sync"):
        torch_xla.sync()
    else:
        _xm().mark_step()


def initialize_cache(path, index):
    try:
        os.makedirs(path, exist_ok=True)
        _xr().initialize_cache(path, readonly=index != 0)
    except Exception as error:
        print(f"Could not enable the XLA compilation cache at {path}: {error}")


def set_matmul_precision(precision="highest"):
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(precision)
    try:
        import torch_xla.backends

        torch_xla.backends.set_mat_mul_precision(precision)
    except (ImportError, AttributeError):
        pass
    try:
        import torch_xla

        xlac = getattr(torch_xla, "_XLAC", None)
        if xlac is not None and hasattr(xlac, "_xla_set_mat_mul_precision"):
            xlac._xla_set_mat_mul_precision(precision)
    except Exception as error:
        print(f"Could not set the TPU matmul precision to {precision}: {error}")


def manual_seed(seed):
    _xm().set_rng_state(int(seed))


def get_rng_state():
    return int(_xm().get_rng_state())


def set_rng_state(state):
    _xm().set_rng_state(int(state))


def reduce_gradients(optimizer):
    _xm().reduce_gradients(optimizer)


def broadcast_master_params(module):
    xm = _xm()
    if hasattr(xm, "broadcast_master_param"):
        xm.broadcast_master_param(module)
    else:
        is_master = rank() == 0
        with torch.no_grad():
            for param in module.parameters():
                masked = param if is_master else torch.zeros_like(param)
                param.copy_(xm.all_reduce(xm.REDUCE_SUM, masked))
    sync()


def gather_objects(tag, obj):
    return _xm().mesh_reduce(tag, obj, list)


def sum_tensors(tag, tensor):
    return _xm().mesh_reduce(tag, tensor, lambda tensors: torch.stack(tensors).sum(0))


def rendezvous(tag, payload=b""):
    return _xm().rendezvous(tag, payload)


def iter_batches(loader, xla_device):
    import torch_xla.distributed.parallel_loader as pl

    parallel = pl.ParallelLoader(loader, [xla_device])
    try:
        yield from parallel.per_device_loader(xla_device)
    finally:
        parallel.close()


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return type(obj)((key, to_cpu(value)) for key, value in obj.items())
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(value) for value in obj)
    return obj
