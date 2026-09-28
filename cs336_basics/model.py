import torch
from .nn_utils import MultiHeadSelfAttention
from .RMSNorm import RMSNorm
import torch.nn as nn
from .swiglu import SwiGLU
from .Embedding import Embedding
from .Linear import Linear


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device = device, dtype = dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, 
                                           device=device,dtype=dtype)
        self.ln2 = RMSNorm(d_model, device = device, dtype = dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype = dtype)
    
    def forward(self, x, token_positions = None):
        x1 = self.attn(self.ln1(x), token_positions) + x
        return  self.ffn(self.ln2(x1)) + x1

class Transformerlm(nn.Module):
    def __init__(self, vocab_size, context_length, d_model,num_layers,
                 num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        # token_embedding,输入 vocab_size, 输出 d_model
        self.token_embeddings = Embedding(vocab_size, d_model,device=device,dtype=dtype)
        # 2. 核心楼层：N 个相同的 TransformerBlock
        # 必须使用 nn.ModuleList，把刚才写好的 Block 重复 num_layers 次
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, 
                num_heads=num_heads, 
                d_ff=d_ff, 
                max_seq_len=context_length, 
                theta=rope_theta, 
                device=device, 
                dtype=dtype
            ) for _ in range(num_layers)
            ])
        #最终的 RMSNorm (洗脸后才能出厂预测)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        #映射回词表
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, in_indices):
        x = self.token_embeddings(in_indices)
        T = x.size(1)
        token_positions = torch.arange(T, device=x.device)
        for layer in self.layers:
            x = layer(x, token_positions)
        output = self.lm_head(self.ln_final(x))
        return output