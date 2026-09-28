# cs336_basics/Linear.py
import torch.nn as nn
import torch
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features,
                                               device=device, dtype=dtype))
        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, std=sigma, 
                              a=-3.0*sigma, b=3.0*sigma)
    
    def forward(self, x):
        x = x @ self.weight.T
        return x