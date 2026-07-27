import math
import torch
import triton
import triton.language as tl


class MyFlashAttnAutogradFunctionClass_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        out, lse = flash_fwd_triton(q, k, v, is_causal)

        ctx.save_for_backward(q, k, v, lse)
        ctx.is_causal = is_causal

        return out
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, lse = ctx.saved_tensors

        dq, dk, dv = flash_bwd_triton(
            q, k, v, do, lse, ctx.is_causal
        )

        return dq, dk, dv, None


def flash_fwd_triton(q, k, v, is_causal):
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    B, T, D = q.shape
    o = torch.empty_like(q)
    lse = torch.empty((B, T), device=q.device, dtype=torch.float32)

    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)

    grid = (triton.cdiv(T, BLOCK_M),
            B,
    )

    sm_scale = 1.0 / math.sqrt(D)

    flash_fwd_kernel[grid](
        q,k,v,
        o, lse,
        T, D,
        sm_scale,
        is_causal,
        BLOCK_M = BLOCK_M,
        BLOCK_N = BLOCK_N,
        BLOCK_D = BLOCK_D,
    )

    return o, lse

@triton.jit
def flash_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    sm_scale: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_offset = pid_bh * T * D + offs_m[:, None] * D + offs_d[None, :]
    q_mask = (offs_m[:, None] < T) & (offs_d[None, :] < D)

    q = tl.load(q_ptr + q_offset, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, T, BLOCK_N):
        curr_n = start_n + offs_n

        k_offsets = pid_bh * T * D + curr_n[:,None] * D + offs_d[None,:]
        v_offsets = pid_bh * T * D + curr_n[:,None] * D + offs_d[None,:]

        kv_mask = (curr_n[:, None] < T) & (offs_d[None, :] < D)

        k = tl.load(k_ptr + k_offsets, mask= kv_mask, other=0.0)
        v = tl.load(v_ptr + v_offsets, mask= kv_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k)) * sm_scale
        scores = tl.where(curr_n[None, :] < T, scores, -float("inf"))

        if IS_CAUSAL:
            causal_mask = offs_m[:,None] >= curr_n[None, :]
            scores = tl.where(causal_mask, scores, -float("inf"))
        
        m_ij = tl.max(scores, axis= 1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:,None])

        l_new = alpha * l_i  + tl.sum(p, axis=1) #当前加旧的
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        
        m_i = m_new
        l_i = l_new
    
    acc = acc / l_i[:, None]
    o_offsets = pid_bh * T * D + offs_m[:, None] * D + offs_d[None, :]
    o_mask = (offs_m[:, None] < T) & (offs_d[None, :] < D)
    
    tl.store(o_ptr + o_offsets, acc, mask=o_mask)
    lse_offsets = pid_bh * T + offs_m
    lse = m_i + tl.log(l_i)

    tl.store(lse_ptr + lse_offsets, lse, mask=offs_m < T)


def flash_bwd_triton(q, k, v, do, lse, is_causal):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    do = do.contiguous()
    lse = lse.contiguous()

    B, NQ, D = q.shape
    NK = k.shape[1]

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    dk.zero_()
    dv.zero_()

    BLOCK_M = 16
    BLOCK_N = triton.next_power_of_2(NK)
    BLOCK_D = triton.next_power_of_2(D)

    grid = (triton.cdiv(NQ, BLOCK_M), B)

    _flash_bwd_kernel[grid](
        q, k, v, do, lse,
        dq, dk, dv,
        NQ, NK, D,
        1.0 / math.sqrt(D),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=is_causal,
    )

    return dq, dk, dv


@triton.jit
def _flash_bwd_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    offsets_d = tl.arange(0, BLOCK_D)

    q_offsets = pid_b * NQ * D + offsets_m[:, None] * D + offsets_d[None,:]
    k_offsets = pid_b * NQ * D + offsets_n[:, None] * D + offsets_d[None,:]
    v_offsets = pid_b * NQ * D + offsets_n[:, None] * D + offsets_d[None,:]
    do_offsets = pid_b * NQ * D + offsets_m[:, None] * D + offsets_d[None,:]

    q_mask = (offsets_m[:, None] < NQ) & (offsets_d[None, :] < D)
    k_mask = (offsets_n[:, None] < NK) & (offsets_d[None, :] < D)
    do_mask = q_mask

    q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
    k = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
    v = tl.load(v_ptr + v_offsets, mask=k_mask, other=0.0)
    do = tl.load(do_ptr + do_offsets, mask=do_mask, other=0.0)

    q = q.to(tl.float32)
    k = k.to(tl.float32)
    v = v.to(tl.float32)
    do = do.to(tl.float32)

    scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
    valid_scores = (offsets_m[:, None] < NQ) & (offsets_n[None, :] < NK)

    if IS_CAUSAL:
        causal_mask = offsets_n[None,:] <= offsets_m[:,None]
        valid_scores = valid_scores & causal_mask

    scores = tl.where(valid_scores, scores, -float("inf"))
    lse = tl.load(lse_ptr + pid_b * NQ + offsets_m, mask=offsets_m<NQ, other=0.0)

    p = tl.exp(scores-lse[:,None])
    p = tl.where(valid_scores, p, 0.0)

    # O = P @ V
    # do = dL / dO
    # dv contribution: P.T @ dO
    dv_contrib = tl.dot(tl.trans(p), do, input_precision="ieee")

    # dp = dO @ V.T
    dp = tl.dot(do, tl.trans(v), input_precision="ieee")

    # softmax backward
    delta = tl.sum(p * dp, axis=1)
    ds = p * (dp - delta[:, None])
    ds = tl.where(valid_scores, ds, 0.0)

    # dq = ds @ K * scale
    dq = tl.dot(ds, k) * scale

    # dk contribution = ds.T @ Q * scale
    dk_contrib = tl.dot(tl.trans(ds), q, input_precision="ieee") * scale

    dq_offsets = pid_b * NQ * D + offsets_m[:, None] * D + offsets_d[None, :]
    dk_offsets = pid_b * NK * D + offsets_n[:, None] * D + offsets_d[None, :]
    dv_offsets = pid_b * NK * D + offsets_n[:, None] * D + offsets_d[None, :]

    dq_mask = (offsets_m[:, None] < NQ) & (offsets_d[None, :] < D)
    dkv_mask = (offsets_n[:, None] < NK) & (offsets_d[None, :] < D)

    tl.store(dq_ptr + dq_offsets, dq, mask=dq_mask)

    tl.atomic_add(dk_ptr + dk_offsets, dk_contrib, mask=dkv_mask)
    tl.atomic_add(dv_ptr + dv_offsets, dv_contrib, mask=dkv_mask)

