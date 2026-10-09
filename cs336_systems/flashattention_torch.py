
import math
import torch


def flash_backward_impl(Q, K, V, O, L ,dO, is_causal):

        B, Nq, D = Q.shape
        _, Nk, Dk = K.shape
        _, Nv, Dv = V.shape

        BLOCK_Q = 16
        BLOCK_K = 16

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        

        for q_start in range(0, Nq, BLOCK_Q):
            q_end = min(Nq, q_start + BLOCK_Q)
            Q_blk = Q[:, q_start:q_end, :]
            dO_blk = dO[:, q_start:q_end, :]
            O_blk = O[:, q_start:q_end, :]
            delta_i = (dO_blk * O_blk).sum(dim=-1)
            L_blk = L[:, q_start:q_end]
            for k_start in range(0, Nk, BLOCK_K):
                k_end = min(k_start + BLOCK_K, Nk)
                K_blk = K[:, k_start:k_end, :]
                V_blk = V[:, k_start:k_end, :]
                q_pos = torch.arange(q_start, q_end, device=Q.device)[:, None]
                k_pos = torch.arange(k_start, k_end, device=Q.device)[None, :]
                mask = ~(k_pos <= q_pos) & is_causal
                #计算出分块S，然后得到分块P
                S_blk = Q_blk @ K_blk.transpose(-2, -1)
                S_blk = S_blk / math.sqrt(D)
                S_blk = S_blk.masked_fill(mask, -torch.inf)
                P_blk = torch.exp(S_blk - L_blk.unsqueeze(-1))
                #计算当前的Q对dV_blk的贡献，这里V的下标j和K的下标j对应
                dV_blk = P_blk.transpose(-2, -1) @ dO_blk
                dV[:, k_start:k_end, :] += dV_blk
                dP_blk = dO_blk @ V_blk.transpose(-2, -1)
                #先计算P_blk的梯度，为后续运算Q,K的梯度做准备
                dS_blk = P_blk * (dP_blk - delta_i.unsqueeze(-1))
                dQ_blk = (dS_blk @ K_blk)/math.sqrt(D)
                dK_blk = (dS_blk.transpose(-2, -1) @ Q_blk)/math.sqrt(D)
                dQ[:, q_start:q_end, :] += dQ_blk
                dK[:, k_start:k_end, :] += dK_blk 
                
        return dQ, dK, dV

compiled_backward = torch.compile(flash_backward_impl)

class FlashAttention_Pytorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 实现 forward
        B, Nq, D = Q.shape
        _, Nk, Dk = K.shape
        _, Nv, Dv = V.shape

        O = torch.zeros_like(Q)
        L = torch.zeros(Q.shape[:-1],device=Q.device,dtype=torch.float32,)
        
        BLOCK_Q = 16
        BLOCK_K = 16

        for q_start in range(0, Nq, BLOCK_Q):
            q_end = min(q_start + BLOCK_Q, Nq)
            Q_blk = Q[:, q_start:q_end, :]

            m = torch.full(Q_blk.shape[:-1], -torch.inf, device  = Q_blk.device, dtype = torch.float32,)
            l = torch.zeros_like(m)
            O_acc = torch.zeros_like(Q_blk)
            
            for k_start in range(0, Nk, BLOCK_K):
                k_end = min(k_start + BLOCK_K, Nk)
                K_blk = K[:, k_start:k_end, :]
                V_blk = V[:, k_start:k_end, :]
                S_blk = Q_blk @ K_blk.transpose(-2, -1)
                S_blk = S_blk / math.sqrt(D)
                q_pos = torch.arange(q_start, q_end, device=Q.device)[:, None]
                k_pos = torch.arange(k_start, k_end, device=Q.device)[None, :]
                mask = ~(k_pos <= q_pos) & is_causal
                S_blk = S_blk.masked_fill(mask, -torch.inf)
                m_blk = S_blk.max(dim=-1).values
                #计算当前的m
                m_new = torch.maximum(m,m_blk)
                m_new_col = m_new.unsqueeze(-1)
                #计算l，O，作为online softmax的分子分母
                l = torch.exp(S_blk - m_new_col).sum(dim=-1) + l * torch.exp(m - m_new)
                O_acc = O_acc * torch.exp(m - m_new).unsqueeze(-1) + torch.exp(S_blk - m_new_col) @ V_blk
                #更新这一块的m
                m = m_new
            #结果写回O，L用于反向传播
            O[:, q_start:q_end, :] = O_acc / l.unsqueeze(-1)
            L[:, q_start:q_end] = torch.log(l) + m
        #保存反向传播所需矩阵，并且返回正向传播的最终结果O
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_backward_impl(
            Q, K, V, O, L, dO, ctx.is_causal
        )
        return dQ, dK, dV, None