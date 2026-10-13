"""A correctness-first FSDP implementation for CS336 Assignment 2.

This implementation shards the assignment's Linear/Embedding weights. Other
parameters (for example RMSNorm gains) stay replicated and have their gradients
averaged across ranks. It intentionally keeps the implementation small: it
does not implement production features such as nested auto-wrap, CPU offload,
checkpoint state-dict conversion, or communication prefetching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from cs336_basics.model import Embedding, Linear


@dataclass
class _ParamHandle:
    """Metadata needed to shard, gather, and reconstruct one parameter."""

    name: str
    module: nn.Module
    parameter: nn.Parameter
    shape: torch.Size
    numel: int
    shard_numel: int
    padding: int


class _ShardedLinearFunction(torch.autograd.Function):
    """Linear forward/backward using a locally stored weight shard."""

    @staticmethod
    def forward(ctx, inputs, local_shard, fsdp, handle):
        compute_dtype = fsdp.compute_dtype or local_shard.dtype
        weight = fsdp._all_gather_param(handle, dtype=compute_dtype)
        ctx.fsdp = fsdp
        ctx.handle = handle
        ctx.compute_dtype = compute_dtype
        ctx.input_dtype = inputs.dtype
        ctx.save_for_backward(inputs)
        return torch.matmul(inputs.to(compute_dtype), weight.transpose(-2, -1))

    @staticmethod
    def backward(ctx, grad_output):
        (inputs,) = ctx.saved_tensors
        handle = ctx.handle
        fsdp = ctx.fsdp
        dtype = ctx.compute_dtype

        # Re-gather for dInput; the full weight was freed after forward.
        weight = fsdp._all_gather_param(handle, dtype=dtype)
        grad_output = grad_output.to(dtype)
        grad_input = torch.matmul(grad_output, weight).to(ctx.input_dtype)

        in_features = handle.shape[-1]
        out_features = handle.shape[0]
        grad_weight = torch.matmul(
            grad_output.reshape(-1, out_features).transpose(0, 1),
            inputs.to(dtype).reshape(-1, in_features),
        )
        fsdp._reduce_scatter_param_grad(handle, grad_weight)
        # FSDP assigns the shard gradient after its asynchronous collective has
        # completed; returning it here would let autograd consume it too early.
        return grad_input, None, None, None


class _ShardedEmbeddingFunction(torch.autograd.Function):
    """Embedding forward/backward using a locally stored weight shard."""

    @staticmethod
    def forward(ctx, indices, local_shard, fsdp, handle):
        compute_dtype = fsdp.compute_dtype or local_shard.dtype
        weight = fsdp._all_gather_param(handle, dtype=compute_dtype)
        ctx.fsdp = fsdp
        ctx.handle = handle
        ctx.compute_dtype = compute_dtype
        ctx.save_for_backward(indices)
        return F.embedding(indices, weight)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        handle = ctx.handle
        grad_full = torch.ops.aten.embedding_dense_backward.default(
            grad_output.to(ctx.compute_dtype),
            indices,
            handle.shape[0],
            -1,  # no padding index
            False,  # scale_grad_by_freq
        )
        ctx.fsdp._reduce_scatter_param_grad(handle, grad_full)
        return None, None, None, None


class FSDP(nn.Module):
    """Simplified fully-sharded wrapper for the CS336 A2 model components."""

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Initialize torch.distributed before constructing FSDP")

        self.module = module
        self.compute_dtype = compute_dtype
        self.process_group = None  # use the default process group
        self.rank = dist.get_rank(self.process_group)
        self.world_size = dist.get_world_size(self.process_group)

        params = list(module.parameters())
        if not params:
            raise ValueError("FSDP module must contain at least one parameter")
        devices = {param.device for param in params}
        if len(devices) != 1:
            raise ValueError(f"FSDP expects parameters on one device, got {devices}")
        self.device = params[0].device

        self._param_handles: list[_ParamHandle] = []
        self._handles_by_param_id: dict[int, _ParamHandle] = {}
        self._handles_by_module_id: dict[int, _ParamHandle] = {}
        # Each entry is (Work, parameter, input buffer, result buffer,
        # assign_result_as_parameter_grad). Keep buffers alive until Work ends.
        self._pending_works: list[
            tuple[object | None, nn.Parameter, torch.Tensor, torch.Tensor, bool]
        ] = []
        self._hook_handles = []

        self._init_param_handles()
        self._register_hooks()

    def _init_param_handles(self) -> None:
        """Replace each supported weight with this rank's flat parameter shard."""
        seen_param_ids: set[int] = set()
        for module_name, submodule in self.module.named_modules():
            if not isinstance(submodule, (Linear, Embedding)):
                continue

            direct_params = list(submodule.named_parameters(recurse=False))
            for param_name, param in direct_params:
                # The A2 Linear and Embedding modules each have only a weight.
                # Keeping this explicit prevents silently mishandling a bias.
                if param_name != "weight":
                    raise NotImplementedError(
                        f"A2 FSDP only shards weight parameters; got {module_name}.{param_name}"
                    )
                if id(param) in seen_param_ids:
                    raise NotImplementedError("Tied FSDP parameters are outside this A2 implementation")
                seen_param_ids.add(id(param))

                numel = param.numel()
                shard_numel = math.ceil(numel / self.world_size)
                padded_numel = shard_numel * self.world_size
                padding = padded_numel - numel
                flat = param.detach().contiguous().view(-1)
                if padding:
                    flat = F.pad(flat, (0, padding))
                local_shard = flat.narrow(
                    0, self.rank * shard_numel, shard_numel
                ).clone()

                handle = _ParamHandle(
                    name=f"{module_name}.{param_name}" if module_name else param_name,
                    module=submodule,
                    parameter=param,
                    shape=param.shape,
                    numel=numel,
                    shard_numel=shard_numel,
                    padding=padding,
                )
                # Keep the Parameter object (and requires_grad flag) registered
                # on the module, but make its storage the local shard.
                param.data = local_shard
                self._param_handles.append(handle)
                self._handles_by_param_id[id(param)] = handle
                self._handles_by_module_id[id(submodule)] = handle

    def _register_hooks(self) -> None:
        """Route supported module forwards through sharded autograd functions."""
        for submodule in self.module.modules():
            handle = self._handles_by_module_id.get(id(submodule))
            if handle is None:
                continue
            if isinstance(submodule, Linear):
                def linear_forward(inputs, _module=submodule, _handle=handle, _fsdp=self):
                    return _ShardedLinearFunction.apply(
                        inputs, _module.weight, _fsdp, _handle
                    )

                submodule.forward = linear_forward
            elif isinstance(submodule, Embedding):
                def embedding_forward(indices, _module=submodule, _handle=handle, _fsdp=self):
                    return _ShardedEmbeddingFunction.apply(
                        indices, _module.weight, _fsdp, _handle
                    )

                submodule.forward = embedding_forward

        sharded_ids = set(self._handles_by_param_id)
        seen_replicated: set[int] = set()
        for param in self.module.parameters():
            if id(param) in sharded_ids or id(param) in seen_replicated:
                continue
            seen_replicated.add(id(param))
            if param.requires_grad:
                hook = param.register_post_accumulate_grad_hook(
                    self._sync_replicated_gradient
                )
                self._hook_handles.append(hook)

    def _all_gather_param(
        self, handle: _ParamHandle, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Reconstruct one full parameter temporarily on every rank."""
        shard = handle.parameter.detach().contiguous()
        if dtype is not None and shard.dtype != dtype:
            shard = shard.to(dtype)
        if self.world_size == 1:
            flat = shard[: handle.numel]
        else:
            shards = [torch.empty_like(shard) for _ in range(self.world_size)]
            dist.all_gather(shards, shard, group=self.process_group)
            flat = torch.cat(shards, dim=0)[: handle.numel]
        return flat.view(handle.shape)

    def _reduce_scatter_param_grad(
        self, handle: _ParamHandle, grad_full: torch.Tensor
    ) -> None:
        """Reduce a full gradient and queue this rank's averaged shard."""
        grad_flat = grad_full.detach().to(torch.float32).contiguous().view(-1)
        if handle.padding:
            grad_flat = F.pad(grad_flat, (0, handle.padding))
        if self.world_size == 1:
            local_grad = grad_flat[: handle.numel].clone()
            self._pending_works.append(
                (None, handle.parameter, grad_flat, local_grad, True)
            )
            return

        # Gloo is used by the official correctness tests. Use all-reduce plus
        # local slicing there; NCCL uses the actual reduce-scatter collective.
        backend = str(dist.get_backend(self.process_group)).lower()
        if "gloo" in backend:
            dist.all_reduce(grad_flat, op=dist.ReduceOp.SUM, group=self.process_group)
            start = self.rank * handle.shard_numel
            local_grad = grad_flat.narrow(0, start, handle.shard_numel).clone()
            self._pending_works.append(
                (None, handle.parameter, grad_flat, local_grad, True)
            )
            return

        local_grad = torch.empty(
            handle.shard_numel, device=grad_flat.device, dtype=grad_flat.dtype
        )
        work = dist.reduce_scatter_tensor(
            local_grad,
            grad_flat,
            op=dist.ReduceOp.SUM,
            group=self.process_group,
            async_op=True,
        )
        self._pending_works.append(
            (work, handle.parameter, grad_flat, local_grad, True)
        )

    def _sync_replicated_gradient(self, param: nn.Parameter) -> None:
        """Average gradients for non-sharded parameters such as RMSNorm gains."""
        if param.grad is None or self.world_size == 1:
            return
        backend = str(dist.get_backend(self.process_group)).lower()
        if "gloo" in backend:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.process_group)
            param.grad.div_(self.world_size)
            return
        work = dist.all_reduce(
            param.grad,
            op=dist.ReduceOp.SUM,
            group=self.process_group,
            async_op=True,
        )
        self._pending_works.append((work, param, param.grad, param.grad, False))

    def finish_gradient_synchronization(self) -> None:
        """Wait for asynchronous NCCL gradient collectives before optimizer.step()."""
        pending, self._pending_works = self._pending_works, []
        for work, param, _input_buffer, output_buffer, assign_result_as_grad in pending:
            if work is not None:
                work.wait()
            if assign_result_as_grad:
                output_buffer.div_(self.world_size)
                if param.grad is None:
                    param.grad = output_buffer
                else:
                    param.grad.add_(output_buffer)
                continue
            if param.grad is None:
                raise RuntimeError(f"Missing synchronized gradient for {param.shape}")
            param.grad.div_(self.world_size)

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """Testing helper: return full parameters, gathering sharded weights."""
        full_params = {}
        for name, param in self.module.named_parameters():
            handle = self._handles_by_param_id.get(id(param))
            if handle is None:
                full_params[name] = param.detach().clone()
            else:
                full_params[name] = self._all_gather_param(handle).detach().clone()
        return full_params

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
