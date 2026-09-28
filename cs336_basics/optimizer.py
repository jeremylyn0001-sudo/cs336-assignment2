import torch 
import math 
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']    
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)     # 一阶动量 m
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # 二阶动量 v
                
                m = state['exp_avg']
                v = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                
                if weight_decay != 0:
                    p.data = p.data - (lr * weight_decay * p.data)
                
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad **2)
                state['exp_avg'] = m
                state['exp_avg_sq'] = v
                
                bias_correction1 = 1 - (beta1 ** t)
                bias_correction2 = 1 - (beta2 ** t)
                step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)
                
                denominator = torch.sqrt(v) + eps
                p.data = p.data - (step_size * (m / denominator))
                
        return loss
    
def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)

def save_checkpoint(model, optimizer, iteration, out):
    # 1. 建立打包箱
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    
    # 2. 执行封存操作
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    # 1. 从硬盘拆包
    checkpoint = torch.load(src, map_location='cpu') # 稳健的做法，先载到内存
    
    # 2. 精准灌回 (调用 PyTorch 模块自带的 load 方法)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 3. 拿到那个当时记下来的时间
    iteration = checkpoint['iteration']
    
    # 4. 根据测试脚本要求，必须返回这个步数
    return iteration