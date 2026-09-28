# cs336_basics/model.py
import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k, theta, max_seq_len):
        super().__init__()
        # 1. 计算频率 (inv_freq)
        # j = [0, 2, 4, ..., d_k-2]
        j = torch.arange(0, d_k, 2).float()
        inv_freq = 1.0 / (theta ** (j / d_k))
        
        # 2. 生成所有位置的角度矩阵
        t = torch.arange(max_seq_len).float()
        # freqs 形状: (max_seq_len, d_k/2)
        freqs = torch.outer(t, inv_freq)
        
        # 3. 关键一步：维度扩充 (从 d_k/2 变回 d_k)
        # 把 [θ1, θ2] 变成 [θ1, θ1, θ2, θ2]，为了和输入向量 x 对齐
        # 形状变为: (max_seq_len, d_k)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        
        # 4. 预计算 cos 和 sin 并存入缓存
        # register_buffer 保证了它们不是参数（不更新），但会随模型移动到 GPU
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, token_positions):
        # x 形状: (..., seq_len, d_k)
        # token_positions 形状: (..., seq_len)
        
        # 1. 根据当前 token 的位置，从预计算的表中抠出对应的 cos 和 sin
        # 形状变为: (..., seq_len, d_k)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        # 2. 实现旋转公式的关键：构造旋转后的向量
        # 原向量 x: [x1, x2, x3, x4, ...]
        # 旋转助手 x_rotated: [-x2, x1, -x4, x3, ...]
        x_even = x[..., 0::2] # 取 [x1, x3, ...]
        x_odd = x[..., 1::2]  # 取 [x2, x4, ...]
        
        # 用 stack 和 flatten 配合，交替组合成 [-x2, x1, -x4, x3]
        x_rotated = torch.stack([-x_odd, x_even], dim=-1).flatten(-2)
        
        # 3. 应用最终公式：x_new = x * cos + x_rotated * sin
        return x * cos + x_rotated * sin