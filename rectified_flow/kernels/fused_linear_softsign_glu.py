import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def is_triton_available():
    return _TRITON_AVAILABLE


def fused_supported(device, dtype):
    device = torch.device(device)
    if not _TRITON_AVAILABLE or device.type != 'cuda' or torch.version.hip:
        return False
    version = tuple(int(part) for part in triton.__version__.split('.')[:2])
    capability = torch.cuda.get_device_capability(device)
    minimum = (8, 0) if version >= (3, 3) else (7, 0)
    return capability >= minimum and (dtype == torch.float16 or dtype == torch.bfloat16 and capability >= (8, 0))


if _TRITON_AVAILABLE:

    @triton.autotune(
        configs=[

            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),

            triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        ],
        key=['M_BUCKET', 'N', 'K'],
    )
    @triton.jit
    def _fused_linear_softsign_glu_fwd_kernel(
        x_ptr, w_left_ptr, w_right_ptr, b_left_ptr, b_right_ptr,
        y_ptr, left_ptr, gate_ptr,
        M, N, K,
        M_BUCKET,
        stride_x_b, stride_x_k,
        stride_wl_n, stride_wl_k,
        stride_wr_n, stride_wr_k,
        stride_y_b, stride_y_n,
        stride_l_b, stride_l_n,
        stride_g_b, stride_g_n,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)

        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        m_mask_2d = offs_m[:, None] < M
        n_mask_nk = offs_n[:, None] < N
        n_mask_mn = offs_n[None, :] < N
        n_mask_1d = offs_n < N

        acc_left = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        acc_gate = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            k_mask_2d = k_offs[None, :] < K

            x = tl.load(
                x_ptr + offs_m[:, None] * stride_x_b + k_offs[None, :] * stride_x_k,
                mask=m_mask_2d & k_mask_2d, other=0.0,
            )
            wl = tl.load(
                w_left_ptr + offs_n[:, None] * stride_wl_n + k_offs[None, :] * stride_wl_k,
                mask=n_mask_nk & k_mask_2d, other=0.0,
            )
            acc_left += tl.dot(x, wl.T)

            wr = tl.load(
                w_right_ptr + offs_n[:, None] * stride_wr_n + k_offs[None, :] * stride_wr_k,
                mask=n_mask_nk & k_mask_2d, other=0.0,
            )
            acc_gate += tl.dot(x, wr.T)


        b_left = tl.load(b_left_ptr + offs_n, mask=n_mask_1d, other=0.0)
        b_right = tl.load(b_right_ptr + offs_n, mask=n_mask_1d, other=0.0)
        acc_left += b_left
        acc_gate += b_right


        gate_f32 = acc_gate.to(tl.float32)
        ss_gate = gate_f32 / (1.0 + tl.abs(gate_f32))
        gated = acc_left * ss_gate


        tl.store(
            y_ptr + offs_m[:, None] * stride_y_b + offs_n[None, :] * stride_y_n,
            gated, mask=m_mask_2d & n_mask_mn,
        )


        tl.store(
            left_ptr + offs_m[:, None] * stride_l_b + offs_n[None, :] * stride_l_n,
            acc_left, mask=m_mask_2d & n_mask_mn,
        )
        tl.store(
            gate_ptr + offs_m[:, None] * stride_g_b + offs_n[None, :] * stride_g_n,
            acc_gate, mask=m_mask_2d & n_mask_mn,
        )


    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        ],
        key=['N'],
    )
    @triton.jit
    def _softsign_glu_bwd_elem_kernel(
        left_ptr, gate_ptr, grad_y_ptr,
        glp_ptr, gg_ptr,
        M, N,
        stride_l_b, stride_l_n,
        stride_g_b, stride_g_n,
        stride_gy_b, stride_gy_n,
        stride_glp_b, stride_glp_n,
        stride_gg_b, stride_gg_n,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N

        left = tl.load(left_ptr + offs_m[:, None] * stride_l_b + offs_n[None, :] * stride_l_n,
                       mask=m_mask & n_mask, other=0.0)
        gate = tl.load(gate_ptr + offs_m[:, None] * stride_g_b + offs_n[None, :] * stride_g_n,
                       mask=m_mask & n_mask, other=0.0)
        gy = tl.load(grad_y_ptr + offs_m[:, None] * stride_gy_b + offs_n[None, :] * stride_gy_n,
                     mask=m_mask & n_mask, other=0.0)

        gate_f32 = gate.to(tl.float32)
        left_f32 = left.to(tl.float32)
        denom_g = 1.0 / (1.0 + tl.abs(gate_f32))
        denom_g2 = denom_g * denom_g
        ss_gate = gate_f32 * denom_g
        grad_left_pre = gy * ss_gate
        grad_gate = gy * (left_f32 * denom_g2)

        tl.store(glp_ptr + offs_m[:, None] * stride_glp_b + offs_n[None, :] * stride_glp_n,
                 grad_left_pre, mask=m_mask & n_mask)
        tl.store(gg_ptr + offs_m[:, None] * stride_gg_b + offs_n[None, :] * stride_gg_n,
                 grad_gate, mask=m_mask & n_mask)


    class FusedLinearSoftSignGLUFn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, weight, bias):
            orig_shape = x.shape
            K = weight.shape[1]
            N = weight.shape[0] // 2
            x_2d = x.reshape(-1, K)
            M = x_2d.shape[0]

            w_left, w_right = weight.split(N, dim=0)
            if bias is not None:
                b_left, b_right = bias.split(N, dim=0)
            else:
                b_left = b_right = None

            out = torch.empty(M, N, device=x.device, dtype=x.dtype)
            left = torch.empty(M, N, device=x.device, dtype=x.dtype)
            gate = torch.empty(M, N, device=x.device, dtype=x.dtype)

            def grid(meta):
                return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

            _fused_linear_softsign_glu_fwd_kernel[grid](
                x_2d, w_left, w_right, b_left, b_right,
                out, left, gate,
                M, N, K,
                triton.next_power_of_2(M),
                x_2d.stride(0), x_2d.stride(1),
                w_left.stride(0), w_left.stride(1),
                w_right.stride(0), w_right.stride(1),
                out.stride(0), out.stride(1),
                left.stride(0), left.stride(1),
                gate.stride(0), gate.stride(1),
            )

            if x.dim() != 2:
                out = out.view(*orig_shape[:-1], N)

            ctx.save_for_backward(x_2d, weight, left, gate)
            ctx.orig_x_shape = orig_shape
            ctx.N = N
            return out

        @staticmethod
        def backward(ctx, grad_y):
            x, weight, left, gate = ctx.saved_tensors
            M, K = x.shape
            N = ctx.N
            w_left, w_right = weight.split(N, dim=0)

            if grad_y.dim() != 2:
                grad_y = grad_y.reshape(-1, N)
            if not grad_y.is_contiguous():
                grad_y = grad_y.contiguous()


            grad_left_pre = torch.empty(M, N, device=x.device, dtype=x.dtype)
            grad_gate = torch.empty(M, N, device=x.device, dtype=x.dtype)

            def elem_grid(meta):
                return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

            _softsign_glu_bwd_elem_kernel[elem_grid](
                left, gate, grad_y,
                grad_left_pre, grad_gate,
                M, N,
                left.stride(0), left.stride(1),
                gate.stride(0), gate.stride(1),
                grad_y.stride(0), grad_y.stride(1),
                grad_left_pre.stride(0), grad_left_pre.stride(1),
                grad_gate.stride(0), grad_gate.stride(1),
            )


            grad_weight = torch.empty(2 * N, K, device=x.device, dtype=x.dtype)
            torch.mm(grad_left_pre.T, x, out=grad_weight[:N])
            torch.mm(grad_gate.T, x, out=grad_weight[N:])
            grad_bias = torch.cat([grad_left_pre.sum(0), grad_gate.sum(0)], dim=0)


            grad_x = torch.mm(grad_left_pre, w_left)
            grad_x.addmm_(grad_gate, w_right)

            if len(ctx.orig_x_shape) != 2:
                grad_x = grad_x.view(*ctx.orig_x_shape)

            return grad_x, grad_weight, grad_bias


def _eager_linear_softsign_glu(x, weight, bias):
    linear = F.linear(x, weight, bias)
    left, gate = torch.split(linear, linear.shape[-1] // 2, dim=-1)
    return left * F.softsign(gate)


def fused_linear_softsign_glu(x, weight, bias):
    if bias is None or x.numel() == 0 or not fused_supported(x.device, x.dtype):
        return _eager_linear_softsign_glu(x, weight, bias)
    if weight.dtype != x.dtype:
        weight = weight.to(x.dtype)
    if bias.dtype != x.dtype:
        bias = bias.to(x.dtype)
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()
    with torch.cuda.device(x.device):
        return FusedLinearSoftSignGLUFn.apply(x, weight, bias)
