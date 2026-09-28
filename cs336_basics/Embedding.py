# cs336_basics/Embedding.py
import torch
import torch.nn as nn
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim),**kwargs))
        nn.init.trunc_normal_(self.weight, mean = 0.0, std = 1.0, a = -3.0, b = 3.0)
    
    def forward(self, token_ids):
        return self.weight[token_ids]