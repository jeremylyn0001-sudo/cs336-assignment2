# cs336_basics/RMSNorm.py
import torch.nn as nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        nn.init.ones_(self.gain)
    
    def forward(self, x):
        orig_type = x.dtype
        x_f32 = x.to(torch.float32)
        ms = x_f32.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        x_norm = x_f32 / rms
        output = x_norm * self.gain
        
        return output.to(orig_type)