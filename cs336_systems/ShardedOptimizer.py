import torch
from torch.optim import Optimizer
import torch.distributed as dist
import copy

class SharedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self._optim_defaults = kwargs.copy()
        self.initialized = False
        self._partition_parameters_cache = []
        super().__init__(params, self._optim_defaults)
        local_param_groups = self._partition_parameters()[self.rank]
        self.optim = self.optimizer_cls(local_param_groups,**self._optim_defaults,)
        self.initialized = True

    def _partition_parameters(self):
        if len(self._partition_parameters_cache) == 0:
            self._partition_parameters_cache = [[] for _ in range(self.world_size)]
            sizes = [0] * self.world_size
            #为cache添加参数组
            for param_group in self.param_groups:
                params_per_rank_in_group = [
                [] for _ in range(self.world_size)
                ]
                #从大到小排参数量然后分配给rank
                params_sorted = sorted(
                param_group["params"],
                key=lambda param: param.numel(),
                reverse=True,
            )
                for param in params_sorted:
                    rank = self._get_min_index(sizes)
                    params_per_rank_in_group[rank].append(param)
                    sizes[rank] += param.numel()
                 #rank得到对应的参数后打超参数标签
                self._partition_param_group(param_group, params_per_rank_in_group)
        return self._partition_parameters_cache
            
                
    def _get_min_index(self, values):
        min_index = -1
        min_value = float("inf")
        for i, value in enumerate(values):
            if value < min_value:
                min_value = value
                min_index = i
        return min_index

    def _partition_param_group(self, param_group, params_per_rank_in_group):
        for rank, params in enumerate(params_per_rank_in_group):
            rank_param_group = copy.copy(param_group)
            rank_param_group["params"] = params
            self._partition_parameters_cache[rank].append(rank_param_group)

    def step(self, closure=None, **kwargs):
        result = self.optim.step(closure=closure, **kwargs)
        self._sync_params()
        return result
    #广播梯度
    def _sync_params(self):
        handles = []
        param_groups = self._partition_parameters()
        for src_rank in range(self.world_size):
            for group in param_groups[src_rank]:
                handles.extend(
                    dist.broadcast(
                        tensor = param.data,
                        src = src_rank,
                        async_op = True,)
                    for param in group["params"]
                )
        for handle in handles:
            handle.wait()