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
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,):
    #初始化当前块
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    #计算Query_tile的地址
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = (Q_ptr + pid_b*stride_qb + offs_q[:, None]*stride_qn +  offs_d[None, :]*stride_qd)
    #取出Query_tile
    q = tl.load(q_ptrs, mask = (offs_q[:, None]<N_Q)&(offs_d[None, :]<D), other=0.0)
    #初始化当前BLOCK的m,l,O
    m = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    #取出当前计算的K,V tile
    # Causal rows never use K tiles to the right of this Q tile. Keep D=128 on the static full loop; runtime bounds regressed there on H800.
    if IS_CAUSAL and D < 128:
        end_k = tl.minimum(N_K, ((pid_q + 1) * BLOCK_Q + BLOCK_K - 1) // BLOCK_K * BLOCK_K)
    else:
        end_k = N_K
    for start_k in tl.range(0, end_k, BLOCK_K):
        cur_k = start_k + tl.arange(0, BLOCK_K)
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
    tl.store(o_ptrs, O_acc.to(q.dtype), mask=(offs_q[:, None] < N_Q) & (offs_d[None, :] < D))
    tl.store(l_ptrs, lse, mask = offs_q < N_Q)


@triton.jit
def flash_bwd_q_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_dob, stride_don, stride_dod,
    stride_lb, stride_ln,
    stride_dqb, stride_dqn, stride_dqd,
    N_Q: tl.constexpr,
    N_K: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,):
    #初始化，计算当前处理的行i
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    dq = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    #计算Q_tile的地址
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)#计算i，为Q,O,l使用，为共同的行下标,先取出Q_tile
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = (Q_ptr + pid_b*stride_qb + offs_q[:, None]*stride_qn +  offs_d[None, :]*stride_qd)
    l_ptrs = (L_ptr + pid_b*stride_lb + offs_q*stride_ln)
    dO_ptrs = (dO_ptr + pid_b*stride_dob + offs_q[:, None]*stride_don +  offs_d[None, :]*stride_dod)
    O_ptrs = (O_ptr + pid_b*stride_ob + offs_q[:, None]*stride_on +  offs_d[None, :]*stride_od)
    dQ_ptrs = (dQ_ptr + pid_b*stride_dqb + offs_q[:, None]*stride_dqn +  offs_d[None, :]*stride_dqd)
    #取出Q_tile,L_tile,dO_tile,O_tile
    q = tl.load(q_ptrs, mask = (offs_q[:, None]<N_Q)&(offs_d[None, :]<D), other=0.0)
    l = tl.load(l_ptrs, mask = offs_q < N_Q)
    do = tl.load(dO_ptrs, mask = (offs_q[:, None]<N_Q)&(offs_d[None, :]<D), other=0.0)
    o = tl.load(O_ptrs, mask = (offs_q[:, None]<N_Q)&(offs_d[None, :]<D), other=0.0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=-1)
    #对K,V_tile进行运算，求P,S的梯度
    # Causal rows never use K tiles to the right of this Q tile. Keep D=128 on the static full loop; runtime bounds regressed there on H800.
    if IS_CAUSAL and D < 128:
        end_k = tl.minimum(N_K, ((pid_q + 1) * BLOCK_Q + BLOCK_K - 1) // BLOCK_K * BLOCK_K)
    else:
        end_k = N_K
    for start_k in tl.range(0, end_k, BLOCK_K):
        cur_k = start_k + offs_k
        k_ptrs = (K_ptr + pid_b*stride_kb + cur_k[:, None]*stride_kn + offs_d[None, :]*stride_kd)
        v_ptrs = (V_ptr + pid_b*stride_vb + cur_k[:, None]*stride_vn + offs_d[None, :]*stride_vd)
        kv_mask = (cur_k[:, None]<N_K)&(offs_d[None, :]<D)
        k = tl.load(k_ptrs, mask = kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask = kv_mask, other=0.0)
        #先计算S,P
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        valid = cur_k[None, :] < N_K
        if IS_CAUSAL:
            valid = valid & (offs_q[:, None]>=cur_k[None, :])
        s = tl.where(valid, s, float("-inf"))
        p = tl.exp(s - l[:, None])
        #计算dP,dS，dq
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p*(dp - delta[:, None])
        dq += tl.dot(ds, k.to(tl.float32), out_dtype=tl.float32, input_precision="ieee") * SCALE
    #写回dQ
    tl.store(dQ_ptrs, dq.to(q.dtype), mask=(offs_q[:, None] < N_Q) & (offs_d[None, :] < D))

@triton.jit
def flash_bwd_kv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_dob, stride_don, stride_dod,
    stride_lb, stride_ln,
    stride_dkb, stride_dkn, stride_dkd,
    stride_dvb, stride_dvn, stride_dvd,
    N_Q: tl.constexpr,
    N_K: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,):
    #初始化当前的K,V，行为j
    pid_k = tl.program_id(0)
    pid_b = tl.program_id(1)
    dk = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
    #计算K,V_tile的地址
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_q = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    #取出K_tile,V_tile
    k_ptrs = (K_ptr + pid_b*stride_kb + offs_k[:, None]*stride_kn +  offs_d[None, :]*stride_kd)
    v_ptrs = (V_ptr + pid_b*stride_vb + offs_k[:, None]*stride_vn +  offs_d[None, :]*stride_vd)
    dK_ptrs = (dK_ptr + pid_b*stride_dkb + offs_k[:, None]*stride_dkn +  offs_d[None, :]*stride_dkd)
    dV_ptrs = (dV_ptr + pid_b*stride_dvb + offs_k[:, None]*stride_dvn +  offs_d[None, :]*stride_dvd)
    k = tl.load(k_ptrs, mask = (offs_k[:, None]<N_K)&(offs_d[None, :]<D), other=0.0)
    v = tl.load(v_ptrs, mask = (offs_k[:, None]<N_K)&(offs_d[None, :]<D), other=0.0)
    #对Q_tile进行运算，求V梯度
    # Query tiles before the one overlapping this K tile have q < k everywhere, so their dK/dV contributions are exactly zero.
    if IS_CAUSAL and D < 128:
        start_q0 = (pid_k * BLOCK_K // BLOCK_Q) * BLOCK_Q
    else:
        start_q0 = 0
    for start_q in tl.range(start_q0, N_Q, BLOCK_Q):
        cur_q = start_q + offs_q
        q_ptrs = (Q_ptr + pid_b*stride_qb + cur_q[:, None]*stride_qn +  offs_d[None, :]*stride_qd)
        l_ptrs = (L_ptr + pid_b*stride_lb + cur_q*stride_ln)
        do_ptrs = (dO_ptr + pid_b*stride_dob + cur_q[:, None]*stride_don +  offs_d[None, :]*stride_dod)
        o_ptrs = (O_ptr + pid_b*stride_ob + cur_q[:, None]*stride_on +  offs_d[None, :]*stride_od)
        #取出Q_tile,L_tile,dO_tile,O_tile
        q_mask = (cur_q[:, None]<N_Q)&(offs_d[None, :]<D)
        q = tl.load(q_ptrs, mask = q_mask, other=0.0)
        l = tl.load(l_ptrs, mask = cur_q < N_Q, other=0.0)
        do = tl.load(do_ptrs, mask = q_mask, other=0.0)
        o = tl.load(o_ptrs, mask = q_mask, other=0.0)
        #先计算S,P
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        valid = offs_k[None, :] < N_K
        if IS_CAUSAL:
            valid = valid & (cur_q[:, None]>=offs_k[None, :])
        s = tl.where(valid, s, float("-inf"))
        p = tl.exp(s - l[:, None])
        #计算dP,dS，dK,dV
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=-1)
        ds = p*(dp - delta[:, None])
        dk += tl.dot(tl.trans(ds), q.to(tl.float32), out_dtype=tl.float32, input_precision="ieee") * SCALE
        dv += tl.dot(tl.trans(p), do.to(tl.float32), out_dtype=tl.float32, input_precision="ieee")
    #写回dK, dV
    tl.store(dK_ptrs, dk.to(k.dtype), mask=(offs_k[:, None] < N_K) & (offs_d[None, :] < D))
    tl.store(dV_ptrs, dv.to(v.dtype), mask=(offs_k[:, None] < N_K) & (offs_d[None, :] < D))

class FlashAttention_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N_Q, D = Q.shape
        _, N_K, D_K = K.shape

        assert D == D_K == V.shape[-1]
        assert K.shape[-2] == V.shape[-2]

        O = torch.empty_like(Q)
        L = torch.empty((B, N_Q), device=Q.device, dtype=torch.float32)

        BLOCK_Q = 32
        needs_small_k_tile = Q.dtype == torch.float32 and D >= 128
        BLOCK_K = 32 if needs_small_k_tile else 64
        num_stages = 1 if needs_small_k_tile else 3
        BLOCK_D = triton.next_power_of_2(D)

        grid = (triton.cdiv(N_Q, BLOCK_Q), B)
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_Q, N_K, D, 1.0 / math.sqrt(D),
            BLOCK_Q, BLOCK_K, BLOCK_D, is_causal,
            num_warps=4,
            num_stages=num_stages,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        B, N_Q, D = Q.shape
        N_K = K.shape[1]

        BLOCK_Q = 32
        needs_small_k_tile = Q.dtype == torch.float32 and D >= 128
        BLOCK_K = 32 if needs_small_k_tile else 64
        num_stages = 1 if needs_small_k_tile else 3
        BLOCK_D = triton.next_power_of_2(D)

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        grid_q = (triton.cdiv(N_Q, BLOCK_Q), B)
        grid_kv = (triton.cdiv(N_K, BLOCK_K), B)
        #针对dQ的算子
        flash_bwd_q_kernel[grid_q](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            N_Q, N_K, D, 1.0 / math.sqrt(D),
            BLOCK_Q, BLOCK_K, BLOCK_D, ctx.is_causal,
            num_warps=4,
            num_stages=num_stages,
        )
        #针对dK,dV的算子
        flash_bwd_kv_kernel[grid_kv](
            Q, K, V, O, dO, L, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            N_Q, N_K, D, 1.0 / math.sqrt(D),
            BLOCK_Q, BLOCK_K, BLOCK_D, ctx.is_causal,
            num_warps=4,
            num_stages=num_stages,
        )

        return dQ, dK, dV, None


class FlashAttention_Triton_CompiledBackward(torch.autograd.Function):
    """Required hybrid path: Triton forward with torch.compile backward."""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Reuse the tested Triton forward implementation and its saved tensors.
        return FlashAttention_Triton.forward(ctx, Q, K, V, is_causal)

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = compiled_backward(
            Q, K, V, O, L, dO, ctx.is_causal
        )
        return dQ, dK, dV, None
