import glob
import importlib.util
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TPU_CACHE_DIR = os.environ.get(
    "RVC_XLA_CACHE_DIR", os.path.join(_REPO_ROOT, "assets", "xla_cache")
)


def torch_tpu_installed():
    return importlib.util.find_spec("torch_tpu") is not None


def torch_xla_installed():
    return importlib.util.find_spec("torch_xla") is not None


def tpu_installed():
    return torch_tpu_installed() or torch_xla_installed()


def backend():
    """Return the active TPU backend name: 'torch_tpu', 'xla', or None."""
    preferred = os.environ.get("RVC_TPU_BACKEND", "").strip().lower()
    has_torch_tpu = torch_tpu_installed()
    has_xla = torch_xla_installed()
    if preferred in ("xla", "torch_xla") and has_xla:
        return "xla"
    if preferred in ("torch_tpu", "tpu") and has_torch_tpu:
        return "torch_tpu"
    if has_torch_tpu:
        return "torch_tpu"
    if has_xla:
        return "xla"
    return None


def configure_env(cache_dir=None):
    """Configure TorchTPU or PyTorch/XLA env vars before importing torch."""
    active = backend()
    if os.environ.get("RVC_TPU_DISABLE_AUTOLOAD") == "1":
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        return active
    if active == "torch_tpu":
        # Override legacy TORCH_DEVICE_BACKEND_AUTOLOAD=0 so preinstalled torch-tpu
        # registers the native "tpu" PrivateUse1 backend when torch loads.
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
        mode = os.environ.get("RVC_TPU_MODE", "defer_and_fuse").strip().lower()
        if mode == "defer_and_fuse":
            os.environ.setdefault("TORCH_TPU_DEFER_AND_FUSE", "1")
        elif mode == "eager":
            os.environ.setdefault("TORCH_TPU_DEFER_AND_FUSE", "0")
        os.environ.setdefault(
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
            cache_dir or DEFAULT_TPU_CACHE_DIR,
        )
    else:
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    return active


def disable_child_autoload():
    """Prevent CPU child processes (e.g. DataLoader workers) from loading TPU/PJRT."""
    os.environ["RVC_TPU_DISABLE_AUTOLOAD"] = "1"
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"


configure_env()

import torch
import torch.distributed as dist

INSTALL_HINT = (
    "TPU training needs TorchTPU (torch-tpu) or PyTorch/XLA (torch_xla). "
    "On a TPU VM, install the TPU requirements:\n"
    "    pip install -r requirements-tpu.txt"
)


def is_tpu_device(dev):
    if dev is None:
        return False
    if hasattr(dev, "type"):
        dev_type = dev.type
    elif isinstance(dev, str):
        dev_type = dev.split(":", 1)[0].strip().lower()
    else:
        dev_type = torch.device(dev).type
    return dev_type in ("tpu", "xla")


def mode_description():
    active = backend()
    if active == "torch_tpu":
        defer = os.environ.get("TORCH_TPU_DEFER_AND_FUSE", "0").strip().lower()
        mode = "DeferAndFuse" if defer in ("1", "true", "yes") else "eager"
        return f"TorchTPU ({mode})"
    if active == "xla":
        return "PyTorch/XLA"
    return "TPU"


def tpu_available():
    if not tpu_installed():
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
            "ACCELERATOR_TYPE",
        )
    ):
        return True
    if glob.glob("/dev/accel*") or glob.glob("/dev/vfio/[0-9]*"):
        return True
    if backend() == "torch_tpu" and hasattr(torch, "accelerator"):
        try:
            acc = torch.accelerator.current_accelerator(check_available=True)
            return acc is not None and acc.type == "tpu"
        except Exception:
            return False
    return False


_PRECISION_CONFIGURED_PID = None
_MATMUL_PRECISION_STATE = None
_ORIG_NN_GRU = torch.nn.GRU


def require_tpu():
    """Ensure the active TPU backend is loaded and configured for this process."""
    global _PRECISION_CONFIGURED_PID
    active = backend()
    if active is None:
        raise RuntimeError(INSTALL_HINT)

    if getattr(torch.nn, "GRU", None) is not _ORIG_NN_GRU:
        torch.nn.GRU = _ORIG_NN_GRU

    if active == "torch_tpu":
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
        import torch_tpu

        loader = getattr(torch_tpu, "_loader", None)
        if loader is not None and not getattr(loader, "_LOADED", False):
            load_fn = getattr(loader, "load", None)
            if load_fn is not None:
                load_fn()
        pid = os.getpid()
        if _PRECISION_CONFIGURED_PID != pid:
            set_matmul_precision("highest")
            patch_xla_ops()
            _PRECISION_CONFIGURED_PID = pid
        return torch_tpu

    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.pop("XLA_USE_BF16", None)
    os.environ.pop("XLA_DOWNCAST_BF16", None)
    import logging

    xla_logger = logging.getLogger("torch_xla.backends")
    prev_level = xla_logger.level
    xla_logger.setLevel(logging.ERROR)
    try:
        import torch_xla

        pid = os.getpid()
        if _PRECISION_CONFIGURED_PID != pid:
            set_matmul_precision("highest")
            patch_xla_ops()
            _PRECISION_CONFIGURED_PID = pid
    finally:
        xla_logger.setLevel(prev_level)

    return torch_xla


require_torch_xla = require_tpu


def _xm():
    import torch_xla.core.xla_model as xm

    return xm


def _xr():
    import torch_xla.runtime as xr

    return xr


_TOPOLOGY_ENV_VARS = (
    "TPU_PROCESS_ADDRESSES",
    "TPU_PROCESS_PORT",
    "TPU_VISIBLE_CHIPS",
    "TPU_PROCESS_BOUNDS",
    "TPU_CHIPS_PER_PROCESS_BOUNDS",
    "CLOUD_TPU_TASK_ID",
    "PJRT_LOCAL_PROCESS_COUNT",
    "PJRT_LOCAL_PROCESS_RANK",
)


def _mesh_size(spec):
    """Return the chip/host count represented by a comma-separated TPU mesh shape."""
    if not spec:
        return None
    total = 1
    for part in spec.split(","):
        try:
            dim = int(part.strip())
        except ValueError:
            return None
        if dim <= 0:
            return None
        total *= dim
    return total


def _probe_single_chip_topology():
    """Return (is_single_chip, local_devices, num_workers) without initializing XLA."""
    try:
        from torch_xla._internal import tpu as internal_tpu
    except ImportError:
        return False, None, None
    try:
        if internal_tpu.num_available_chips() != 1:
            return False, None, None
    except Exception:
        return False, None, None

    num_workers = None
    local_devices = None
    try:
        num_workers = internal_tpu.num_tpu_workers()
    except Exception:
        pass
    try:
        local_devices = internal_tpu.num_available_devices()
    except Exception:
        pass
    return True, local_devices, num_workers


def _configure_spawn_topology(single_core):
    """Scrub stale topology env vars and normalize single-chip host bounds."""
    for name in _TOPOLOGY_ENV_VARS:
        if single_core and name == "TPU_VISIBLE_CHIPS":
            visible = os.environ.get(name, "").strip()
            if visible and "," not in visible:
                continue
        os.environ.pop(name, None)

    is_single_chip, local_devices, num_workers = _probe_single_chip_topology()

    # Single-chip containers (e.g. Colab v5e-1) can inherit 4-chip host bounds
    # such as TPU_CHIPS_PER_HOST_BOUNDS='2,2,1', causing PJRT multiprocess
    # init to expect 4 worker addresses when only 1 process is spawned.
    if is_single_chip:
        chips_bounds = os.environ.get("TPU_CHIPS_PER_HOST_BOUNDS")
        if chips_bounds is not None and _mesh_size(chips_bounds) != 1:
            os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
        if num_workers == 1:
            host_bounds = os.environ.get("TPU_HOST_BOUNDS")
            if host_bounds is not None and _mesh_size(host_bounds) != 1:
                os.environ["TPU_HOST_BOUNDS"] = "1,1,1"

    if single_core:
        return True
    return is_single_chip and num_workers == 1 and (local_devices is None or local_devices == 1)


def _torch_tpu_worker(local_rank, fn, args, nproc_per_node):
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    nnodes = int(os.environ.get("NNODES", "1"))
    global_rank = node_rank * nproc_per_node + local_rank
    world_size = nnodes * nproc_per_node
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(nproc_per_node)
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    require_tpu()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="tpu_dist",
            init_method="env://",
            world_size=world_size,
            rank=global_rank,
        )
    try:
        fn(global_rank, *args)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def spawn(fn, args, single_core=False):
    active = backend()
    if active is None:
        raise RuntimeError(INSTALL_HINT)

    if active == "torch_tpu":
        configure_env()
        count = 1
        if not single_core and hasattr(torch, "accelerator"):
            try:
                acc = torch.accelerator.current_accelerator()
                if acc is None or acc.type != "tpu":
                    require_tpu()
                count = max(1, int(torch.accelerator.device_count()))
            except Exception:
                count = 1
        if single_core or count <= 1:
            if single_core:
                os.environ.setdefault("TPU_VISIBLE_DEVICES", "0")
            for key in ("RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "WORLD_SIZE"):
                os.environ.pop(key, None)
            fn(0, *args)
            return

        require_tpu()
        tpu_dist = getattr(getattr(torch, "tpu", None), "distributed", None)
        if tpu_dist is not None and hasattr(tpu_dist, "set_tpu_launch_env"):
            try:
                tpu_dist.set_tpu_launch_env(nproc_per_node=count)
            except TypeError:
                tpu_dist.set_tpu_launch_env()
        if torch.multiprocessing.get_start_method(allow_none=True) != "spawn":
            torch.multiprocessing.set_start_method("spawn", force=True)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
        torch.multiprocessing.spawn(
            _torch_tpu_worker,
            args=(fn, args, count),
            nprocs=count,
            join=True,
        )
        return

    # Don't initialize XLA in the parent: each child configures its own topology.
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.pop("XLA_USE_BF16", None)
    os.environ.pop("XLA_DOWNCAST_BF16", None)
    effective_single_core = _configure_spawn_topology(single_core)
    import torch_xla

    if hasattr(torch_xla, "launch"):
        torch_xla.launch(fn, args=args, debug_single_process=effective_single_core)
        return
    import torch_xla.distributed.xla_multiprocessing as xmp

    xmp.spawn(fn, args=args, nprocs=1 if effective_single_core else None)


_XLA_OPS_PATCHED = False


def _has_privateuse1_kernel(op_name):
    checker = getattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", None)
    if checker is None:
        return False
    try:
        return bool(checker(op_name, "PrivateUse1"))
    except RuntimeError:
        return False


def _compute_weight_norm_xla(weight_v, weight_g, dim):
    with torch.amp.autocast(device_type=weight_v.device.type, enabled=False):
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
    active = backend()
    if active == "torch_tpu":
        if _has_privateuse1_kernel("aten::_weight_norm_interface_backward"):
            _XLA_OPS_PATCHED = True
            return
    elif active != "xla":
        return
    from torch.nn.utils.parametrizations import _WeightNorm

    orig_forward = _WeightNorm.forward

    def _tpu_weight_norm_forward(self, weight_g, weight_v):
        if weight_v.device.type not in ("xla", "tpu"):
            return orig_forward(self, weight_g, weight_v)
        return _compute_weight_norm_xla(weight_v, weight_g, self.dim)

    _WeightNorm.forward = _tpu_weight_norm_forward

    orig_weight_norm_fn = getattr(torch, "_weight_norm", None)
    if orig_weight_norm_fn is not None:

        def _tpu_weight_norm_fn(weight_v, weight_g, dim=0):
            if weight_v.device.type not in ("xla", "tpu"):
                return orig_weight_norm_fn(weight_v, weight_g, dim)
            return _compute_weight_norm_xla(weight_v, weight_g, dim)

        torch._weight_norm = _tpu_weight_norm_fn

    _XLA_OPS_PATCHED = True


patch_tpu_ops = patch_xla_ops


def device():
    tpu_mod = require_tpu()
    if backend() == "torch_tpu":
        if hasattr(torch, "accelerator"):
            try:
                return torch.device("tpu", int(torch.accelerator.current_device_index()))
            except Exception:
                pass
        return torch.device("tpu")

    if hasattr(tpu_mod, "device"):
        return tpu_mod.device()
    return _xm().xla_device()


def rank():
    if backend() == "torch_tpu":
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0
    return _xr().global_ordinal()


def world_size():
    if backend() == "torch_tpu":
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1
    return _xr().world_size()


def sync():
    if backend() == "torch_tpu":
        if hasattr(torch, "accelerator"):
            torch.accelerator.synchronize()
        return
    import torch_xla

    if hasattr(torch_xla, "sync"):
        torch_xla.sync()
    else:
        _xm().mark_step()


def compile_stats():
    """Return (compile_count, uncached_count) from PyTorch/XLA metrics when active."""
    if backend() != "xla":
        return 0, 0
    try:
        import torch_xla.debug.metrics as met
    except ImportError:
        return 0, 0
    compile_data = met.metric_data("CompileTime")
    compiles = int(compile_data[0]) if compile_data else 0
    uncached = met.counter_value("UncachedCompile")
    return compiles, int(uncached or 0)


def initialize_cache(path, index):
    try:
        os.makedirs(path, exist_ok=True)
        if backend() == "torch_tpu":
            os.environ.setdefault("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", path)
            return
        _xr().initialize_cache(path, readonly=index != 0)
    except Exception as error:
        print(f"Could not enable the TPU compilation cache at {path}: {error}")


def set_matmul_precision(precision="highest"):
    global _MATMUL_PRECISION_STATE
    state = (os.getpid(), backend(), precision)
    if _MATMUL_PRECISION_STATE == state:
        return
    if backend() == "torch_tpu":
        try:
            torch.set_float32_matmul_precision(precision)
        except Exception as error:
            print(f"Could not set the TPU matmul precision to {precision}: {error}")
        _MATMUL_PRECISION_STATE = state
        return

    import logging

    xla_logger = logging.getLogger("torch_xla.backends")
    prev_level = xla_logger.level
    xla_logger.setLevel(logging.ERROR)
    try:
        try:
            import torch_xla.backends

            torch_xla.backends.set_mat_mul_precision(precision)
        except (ImportError, AttributeError):
            try:
                import torch_xla

                xlac = getattr(torch_xla, "_XLAC", None)
                if xlac is not None and hasattr(xlac, "_xla_set_use_full_mat_mul_precision"):
                    xlac._xla_set_use_full_mat_mul_precision(precision == "highest")
            except Exception as error:
                print(f"Could not set the TPU matmul precision to {precision}: {error}")
    finally:
        xla_logger.setLevel(prev_level)
    _MATMUL_PRECISION_STATE = state


def manual_seed(seed):
    if backend() == "torch_tpu":
        tpu_mod = getattr(torch, "tpu", None)
        if tpu_mod is not None and hasattr(tpu_mod, "manual_seed"):
            tpu_mod.manual_seed(int(seed))
        return
    _xm().set_rng_state(int(seed))


def get_rng_state():
    if backend() == "torch_tpu":
        acc_random = getattr(getattr(torch, "accelerator", None), "random", None)
        if acc_random is not None and hasattr(acc_random, "get_rng_state"):
            return acc_random.get_rng_state().cpu()
        return torch.tpu.get_rng_state().cpu()
    return int(_xm().get_rng_state())


def set_rng_state(state):
    if backend() == "torch_tpu":
        if isinstance(state, int):
            manual_seed(state)
            return
        if torch.is_tensor(state):
            state = state.detach().to(device="cpu", dtype=torch.uint8)
        acc_random = getattr(getattr(torch, "accelerator", None), "random", None)
        if acc_random is not None and hasattr(acc_random, "set_rng_state"):
            acc_random.set_rng_state(state)
            return
        torch.tpu.set_rng_state(state)
        return
    if torch.is_tensor(state):
        raw = bytes(state.detach().cpu().to(torch.uint8).reshape(-1)[:8].tolist())
        state = int.from_bytes(raw or b"\x00", "little")
    _xm().set_rng_state(int(state))


def reduce_gradients(optimizer):
    if backend() == "torch_tpu":
        ws = world_size()
        if ws <= 1:
            return
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(ws)
        return
    _xm().reduce_gradients(optimizer)


def broadcast_master_params(module):
    if backend() == "torch_tpu":
        if world_size() > 1:
            with torch.no_grad():
                for param in module.parameters():
                    dist.broadcast(param.data, src=0)
            sync()
        return

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
    if backend() == "torch_tpu":
        ws = world_size()
        if ws <= 1:
            return [obj]
        gathered = [None] * ws if rank() == 0 else None
        dist.gather_object(obj, gathered, dst=0)
        return gathered
    return _xm().mesh_reduce(tag, obj, list)


def sum_tensors(tag, tensor):
    if backend() == "torch_tpu":
        if world_size() > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor
    return _xm().mesh_reduce(tag, tensor, lambda tensors: torch.stack(tensors).sum(0))


def rendezvous(tag, payload=b""):
    if backend() == "torch_tpu":
        if world_size() > 1:
            dist.barrier()
        return
    return _xm().rendezvous(tag, payload)


def iter_batches(loader, tpu_device):
    if backend() == "torch_tpu":
        for batch in loader:
            yield [tensor.to(tpu_device, non_blocking=True) for tensor in batch]
        return

    import torch_xla.distributed.parallel_loader as pl

    parallel = pl.ParallelLoader(loader, [tpu_device])
    try:
        yield from parallel.per_device_loader(tpu_device)
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
