from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from tests.adapters import get_sharded_optimizer
from tests.common import _cleanup_process_group, _setup_process_group


class TwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(10, 10)
        self.second = nn.Linear(10, 10)

    def forward(self, x):
        return self.second(torch.tanh(self.first(x)))


def _worker(rank, world_size):
    _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    try:
        torch.manual_seed(42)
        reference = TwoLayerModel()
        sharded = deepcopy(reference)
        reference.second.requires_grad_(False)
        sharded.second.requires_grad_(False)

        optimizer_cls = torch.optim.AdamW
        reference_optimizer = optimizer_cls(
            [{"params": list(reference.first.parameters()), "lr": 1e-2}],
            weight_decay=0.1,
        )
        sharded_optimizer = get_sharded_optimizer(
            [{"params": list(sharded.first.parameters()), "lr": 1e-2}],
            optimizer_cls,
            lr=1e-2,
            weight_decay=0.1,
        )

        torch.manual_seed(99)
        batches = [(torch.randn(8, 10), torch.randn(8, 10)) for _ in range(5)]

        def train_one_step(model, optimizer, x, target):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x) - target).square().mean()
            loss.backward()
            optimizer.step()

        for x, target in batches[:2]:
            train_one_step(reference, reference_optimizer, x, target)
            train_one_step(sharded, sharded_optimizer, x, target)

        reference.second.requires_grad_(True)
        sharded.second.requires_grad_(True)
        reference_optimizer.add_param_group(
            {"params": list(reference.second.parameters()), "lr": 2e-3}
        )
        sharded_optimizer.add_param_group(
            {"params": list(sharded.second.parameters()), "lr": 2e-3}
        )

        for x, target in batches[2:]:
            train_one_step(reference, reference_optimizer, x, target)
            train_one_step(sharded, sharded_optimizer, x, target)

        for expected, actual in zip(reference.parameters(), sharded.parameters()):
            np.testing.assert_allclose(
                expected.detach().numpy(), actual.detach().numpy(), rtol=1e-6, atol=1e-7
            )
    finally:
        if dist.is_initialized():
            _cleanup_process_group()


def test_add_param_group_after_initialization():
    mp.spawn(_worker, args=(2,), nprocs=2, join=True)
