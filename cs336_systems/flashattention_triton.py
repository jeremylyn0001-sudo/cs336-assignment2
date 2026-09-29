import math

import torch
import triton
import triton.language as tl

from cs336_systems.flashattention_torch import compiled_backward

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_lb, stride_ln,
    N_Q: tl.constexpr,
    N_K: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,):
    #初始化当前块
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    #计算Query_tile的地址
    offs_q = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = (Q_ptr + pid_b*stride_qb + offs_q[:, None]*stride_qn +  offs_d[None, :]*stride_qd)
    #取出Query_tile
    q = tl.load(q_ptrs, mask = (offs_q[:, None]<N_Q)&(offs_d[None, :]<D), other=0.0)
    #初始化当前BLOCK的m,l,O
    m = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    #取出当前计算的K,V tile
    for start_k in range(0, N_K, BLOCK_N):
        cur_k = start_k + tl.arange(0, BLOCK_N)
        k_ptrs = (K_ptr + pid_b*stride_kb + cur_k[:, None]*stride_kn + offs_d[None, :]*stride_kd)
        v_ptrs = (V_ptr + pid_b*stride_vb + cur_k[:, None]*stride_vn + offs_d[None, :]*stride_vd)
        kv_mask = (cur_k[:, None]<N_K)&(offs_d[None, :]<D)
        k = tl.load(k_ptrs, mask = kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask = kv_mask, other=0.0)
        #先计算这一小块的s
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        valid = cur_k[None, :] < N_K
        if IS_CAUSAL:
            valid = valid & (offs_q[:, None]>=cur_k[None, :])
        s = tl.where(valid, s, float("-inf"))
        #计算这一块的m,l,O
        m_new = tl.maximum(m, tl.max(s, axis=-1))
        p = tl.exp(s - m_new[:, None])
        l = tl.sum(p, axis=-1) + l*tl.exp(m  - m_new)
        O_acc = tl.dot(p.to(v.dtype), v, out_dtype=tl.float32,input_precision="ieee") + O_acc*(tl.exp(m - m_new)[:, None])
        m = m_new
    #计算出最终的O，l
    O_acc /= l[:, None]
    lse = m + tl.log(l)
    #返回O，l
    o_ptrs = (O_ptr + pid_b*stride_ob + offs_q[:, None]*stride_on + offs_d[None, :]*stride_od)
    l_ptrs = (L_ptr + pid_b*stride_lb + offs_q*stride_ln)
    tl.store(o_ptrs, O_acc.to(q.dtype), mask=(offs_q[:, None] < N_Q) & (offs_d[None, :] < D),)
    tl.store(l_ptrs, lse, mask = offs_q < N_Q)

class FlashAttention_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N_Q, D = Q.shape
        _, N_K, D_K = K.shape

        assert D == D_K == V.shape[-1]
        assert K.shape[-2] == V.shape[-2]

        O = torch.empty_like(Q)
        L = torch.empty((B, N_Q), device=Q.device, dtype=torch.float32)

        BLOCK_M = 32
        needs_small_k_tile = Q.dtype == torch.float32 and D >= 128
        BLOCK_N = 32 if needs_small_k_tile else 64
        num_stages = 1 if needs_small_k_tile else 3
        BLOCK_D = triton.next_power_of_2(D)

        grid = (triton.cdiv(N_Q, BLOCK_M), B)
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_Q, N_K, D, 1.0 / math.sqrt(D),
            BLOCK_M, BLOCK_N, BLOCK_D, is_causal,
            num_warps=4,
            num_stages=num_stages,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = compiled_backward(Q, K, V, O, L, dO, ctx.is_causal)
        return dQ, dK, dV, None
