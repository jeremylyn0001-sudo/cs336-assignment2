import torch.nn as nn
import torch
import torch.distributed as dist

class DDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.hook_handles = []
        self.pending_works = []
        # 1. 把 rank 0 的初始参数广播给其他 rank
        # 2. 给每个需要梯度的参数注册 hook
        #    梯度就绪时，异步 all_reduce 并保存 work handle
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)

        for param in self.module.parameters():
            if param.requires_grad:
                handle = param.register_post_accumulate_grad_hook(
                    self._sync_gradient
                )
                self.hook_handles.append(handle)

    def _sync_gradient(self, param):
        if param.grad is None:
            return
        # all_reduce 默认求和；先除以进程数，求和后就是平均梯度。
        param.grad.div_(self.world_size)
        work = dist.all_reduce(param.grad, async_op=True)
        self.pending_works.append(work)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for work in self.pending_works:
            work.wait()
        self.pending_works.clear()