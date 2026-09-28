import torch
import numpy as np

def get_batch(dataset, batch_size, context_length, device):
    # 1. 产生随机点
    ix = torch.randint(0, len(dataset) - context_length, (batch_size,))
    
    # 2. 切割数据集，生成张量
    all_chunks = [dataset[i: i+context_length+1] for i in ix]
    tensor_data = torch.from_numpy(np.stack(all_chunks)).to(torch.int64).to(device)
    
    # 3.生成模型训练集
    x = tensor_data[: ,:-1]
    y = tensor_data[: ,1: ]   
     
    return x, y