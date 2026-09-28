# cs336_basics/swiglu.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device = None, dtype = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        
    def forward(self, x):
        gate_raw = self.w1(x)
        value = self.w3(x)
        gate = F.silu(gate_raw)
        output_raw = gate * value
        output = self.w2(output_raw)
        return output