"""Weight-only per-row symmetric INT8 GEMV / small-M GEMM in Triton for GB10 (SM121).
Quantizes weights W [N, K] -> W_q [N, K] int8 + scale [N] fp32 at load time.
Cuts LPDDR5X weight bandwidth in half for DSpark draft LM head & Markov fold
while keeping the target model 100% untouched (lossless verification).
"""
import torch
import triton
import triton.language as tl


@torch.no_grad()
def quantize_weight_int8_per_row(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row INT8 quantization: w ≈ w_q * scale[:, None]."""
    assert w.ndim == 2
    w_f = w.float()
    max_abs = w_f.abs().amax(dim=1).clamp(min=1e-8)
    scale = (max_abs / 127.0).to(torch.float32).contiguous()
    w_q = torch.round(w_f / scale[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
    return w_q, scale


@triton.jit
def _int8_gemv_m1_kernel(
    x_ptr,          # [1, K] bf16
    wq_ptr,         # [N, K] int8, row-major
    scale_ptr,      # [N] fp32
    bias_ptr,       # [N] bf16 or dummy
    out_ptr,        # [1, N] bf16
    N,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        wq = tl.load(
            wq_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.float32)
        acc += tl.sum(wq * x[None, :], axis=1)

    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0)
    res = acc * scale
    if HAS_BIAS:
        b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        res += b
    tl.store(out_ptr + offs_n, res.to(tl.bfloat16), mask=mask_n)


@triton.jit
def _int8_gemm_small_m_kernel(
    x_ptr,          # [M, K] bf16
    wq_ptr,         # [N, K] int8, row-major
    scale_ptr,      # [N] fp32
    bias_ptr,       # [N] bf16 or dummy
    out_ptr,        # [M, N] bf16
    M,
    N,
    K: tl.constexpr,
    stride_xm,
    stride_om,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        wq = tl.load(
            wq_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.bfloat16)
        # [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] on BF16 tensor cores
        acc = tl.dot(x, tl.trans(wq), acc=acc)

    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0)
    res = acc * scale[None, :]
    if HAS_BIAS:
        b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        res += b[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
        res.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def int8_linear(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute x @ (w_q * scale).T + bias for 2D x [M, K] bf16 -> [M, N] bf16."""
    assert x.ndim == 2
    M, K = x.shape
    N, K_w = w_q.shape
    assert K == K_w
    x_c = x.contiguous()
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    has_bias = bias is not None
    b_ptr = bias if has_bias else scale  # dummy valid pointer when unused

    if M == 1:
        BLOCK_N = 64 if K <= 512 else 32
        BLOCK_K = 256 if K <= 512 else 256
        grid = (triton.cdiv(N, BLOCK_N),)
        _int8_gemv_m1_kernel[grid](
            x_c, w_q, scale, b_ptr, out,
            N, K,
            HAS_BIAS=has_bias,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )
    else:
        BLOCK_M = 16 if M <= 16 else 32
        BLOCK_N = 64
        BLOCK_K = 128
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
        _int8_gemm_small_m_kernel[grid](
            x_c, w_q, scale, b_ptr, out,
            M, N, K,
            x_c.stride(0), out.stride(0),
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=3,
        )
    return out


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    for name, M, N, K, has_bias in [
        ("Markov fold (bs=1, 201Kx256)", 1, 201024, 256, True),
        ("Markov fold (bs=4, 201Kx256)", 4, 201024, 256, True),
        ("Draft LM head (bs=1, 7x201Kx2304)", 7, 201024, 2304, False),
        ("Draft LM head (bs=4, 28x201Kx2304)", 28, 201024, 2304, False),
    ]:
        x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda") * 0.1
        w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda") * 0.02
        b = torch.randn((N,), dtype=torch.bfloat16, device="cuda") * 0.01 if has_bias else None
        wq, sc = quantize_weight_int8_per_row(w)

        ref = torch.addmm(b, x, w.T) if has_bias else torch.mm(x, w.T)
        got = int8_linear(x, wq, sc, b)
        rel_err = (ref.float() - got.float()).norm() / ref.float().norm()

        # Warmup
        for _ in range(10):
            _ = torch.addmm(b, x, w.T) if has_bias else torch.mm(x, w.T)
            _ = int8_linear(x, wq, sc, b)
        torch.cuda.synchronize()

        def bench(fn, iters=50):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) * 1000.0 / iters  # us

        t_bf16 = bench(lambda: torch.addmm(b, x, w.T) if has_bias else torch.mm(x, w.T))
        t_int8 = bench(lambda: int8_linear(x, wq, sc, b))
        print(f"{name:36s} | BF16: {t_bf16:7.1f} us | INT8: {t_int8:7.1f} us | speedup: {t_bf16/t_int8:.2f}x | rel_err: {rel_err:.4f}")
