import torch
import math
import torch.nn as nn
from .Linear import Linear
from .rope import RotaryPositionalEmbedding

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 纯数学逻辑，不需要 self，不需要 nn.Module
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    seq_len = Q.size(-2)
    
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores/(d_k ** 0.5)
    
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    
    attn_weights = softmax(scores, dim=-1)
    
    return torch.matmul(attn_weights, V)

def cross_entropy(logits, targets):
    # 1. 处理数值稳定性（数值平移）
    # logits shape: (batch, seq, vocab)
    m, _ = torch.max(logits, dim=-1, keepdim=True)
    
    # 2. 计算 Log-Sum-Exp 部分
    lse = m + torch.log(torch.sum(torch.exp(logits - m), dim=-1, keepdim=True))
    
    # 3. 抓取目标类别的得分
    # targets shape: (batch, seq) -> unsqueeze 为 (batch, seq, 1)
    correct_logits = torch.gather(logits, -1, targets.unsqueeze(-1))
    
    # 4. 根据数学公式：loss = log(sum(exp)) - x_y
    # 这里合并了 log 和 exp，实现了 PDF 要求的 "Cancel out log and exp" 优化
    loss = lse - correct_logits
    
    return loss.mean()


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, theta, max_seq_len, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        #q, k, v
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        #rope
        self.rope = RotaryPositionalEmbedding(self.d_k, theta, max_seq_len)
        
    def forward(self, x, token_positions):
        #(B, T, D)
        B, T, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        #(B, T, D) -> (B, T, h, d)
        q = q.view(B, T, self.num_heads, self.d_k)
        k = k.view(B, T, self.num_heads, self.d_k)
        v = v.view(B, T, self.num_heads, self.d_k)
        # (B, H, T, d_k)
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        #apply rope
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        #casual mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()                             
        #compute attention
         # --- 这里是关键：保险丝逻辑 ---
        
        context = scaled_dot_product_attention(q, k, v, mask=mask)
        #(B,T,h,d)
        context = context.transpose(1, 2)
        context = context.contiguous().view(B, T, self.d_model)
        
        return self.output_proj(context)
    
def gradient_clipping(params, max_norm):
    total_norm_sq = 0.0
    
    for p in params:
        if p.grad is not None:
            param_norm_sq = p.grad.detach().pow(2).sum()
            total_norm_sq += param_norm_sq
        
    total_norm = torch.sqrt(total_norm_sq)
    if total_norm > max_norm:
        clip_coeff = max_norm / (total_norm + 1e-6)
    else:
        clip_coeff = 1.0
        
    if clip_coeff < 1.0:
        for p in params:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coeff)
                

import torch

@torch.no_grad() # 关键：生成时不需要计算梯度，省显存并提速
def generate(
    model, 
    idx: torch.Tensor, 
    max_new_tokens: int, 
    temperature: float = 1.0, 
    top_p: float = 1.0,
    eos_token_id: int = None
) -> torch.Tensor:
    """
    根据给定的提示词 (prompt) 自动续写文本。
    
    Args:
        model: 训练好的 TransformerLM 模型
        idx: 初始提示词的 Token ID 序列，形状 (batch_size, sequence_length)
        max_new_tokens: 最多生成多少个新字
        temperature: 温度 (默认 1.0，越低越保守，越高越有创意)
        top_p: 截断概率阈值 (默认 1.0，即不截断；比如 0.9 表示只在累加概率前 90% 的词里选)
        eos_token_id: 结束符的 ID (比如 <|endoftext|> 的 ID)，遇到了就提前退出
    """
    model.eval() # 切换到推理模式
    
    for _ in range(max_new_tokens):
        # 1. 上下文裁剪：Transformer 的注意力长度是有限的 (context_length)
        # 如果输入的句子太长，只截取最后 context_length 个词
        idx_cond = idx if idx.size(1) <= model.token_embeddings.weight.size(0) else idx[:, -model.token_embeddings.weight.size(0):]
        
        # 2. 前向传播：拿到模型预测的所有分值
        logits = model(idx_cond) # 形状: (batch_size, seq_len, vocab_size)
        
        # 3. 聚焦当前：我们只关心【最后一个字】对下一个字的预测
        logits = logits[:, -1, :] # 形状变回: (batch_size, vocab_size)
        
        # 4. 温度缩放 (Temperature Scaling)
        if temperature > 0:
            logits = logits / temperature
            
        # 5. 核采样 (Top-p / Nucleus Sampling)
        if top_p < 1.0:
            # a. 把所有分值从大到小排序
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            
            # b. 算这些分值对应的概率
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            
            # c. 计算累加概率 (Cumulative Sum)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # d. 找到超过 top_p 的边界，把后面的“小鱼小虾”标记为需要删除
            # 技巧：将掩码右移一位，确保【刚刚好超过 top_p 的那第一个词】也被包含进来
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # e. 把要删掉的位置的分数强行设为负无穷 -inf (Softmax 后就会变成 0 概率)
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # 6. 将加工后的得分转为最终概率分布
        probs = torch.softmax(logits, dim=-1)
        
        # 7. 根据概率分配，进行“抽签/抓号” (Multinomial Sampling)
        # 如果 temperature 很低，这里概率几乎全部集中在第一名身上
        next_token = torch.multinomial(probs, num_samples=1) # 形状: (batch_size, 1)
        
        # 8. 将新预测出来的字接在句子的末尾
        idx = torch.cat((idx, next_token), dim=1)
        
        # 9. 提前结束判断
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
            
    return idx