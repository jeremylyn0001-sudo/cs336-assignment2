"""Measure XL-model memory and step time for DDP and assignment FSDP."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW

from benchmark import build_model
from cs336_systems.FSDP import FSDP


def _train_step(model, optimizer, input_ids, labels, use_fsdp: bool) -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.nvtx.range("forward"):
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
    with torch.cuda.nvtx.range("backward"):
        loss.backward()
    if use_fsdp:
        model.finish_gradient_synchronization()
    with torch.cuda.nvtx.range("optimizer_step"):
        optimizer.step()
    del logits, loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ddp", "fsdp"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-nsys", action="store_true")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)

    module = build_model(
        "xl", args.context_length, args.vocab_size, device=str(device)
    )
    if args.mode == "ddp":
        model = DistributedDataParallel(module, device_ids=[local_rank])
        use_fsdp = False
    else:
        model = FSDP(module)
        use_fsdp = True
    optimizer = AdamW(model.parameters())

    cpu_generator = torch.Generator().manual_seed(123)
    input_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        generator=cpu_generator,
        dtype=torch.long,
    )
    labels = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        generator=cpu_generator,
        dtype=torch.long,
    )
    local_batch = args.batch_size // world_size
    if args.batch_size % world_size:
        raise ValueError("Global batch size must be divisible by world size")
    start = rank * local_batch
    stop = start + local_batch
    input_ids = input_ids[start:stop].to(device)
    labels = labels[start:stop].to(device)

    torch.cuda.synchronize(device)
    memory_after_init = torch.cuda.memory_allocated(device)

    # The first optimizer step includes lazy AdamW-state allocation.
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    if args.profile_nsys:
        torch.cuda.cudart().cudaProfilerStart()
    with torch.cuda.nvtx.range("first_iteration"):
        with torch.cuda.nvtx.range("forward"):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )
        with torch.cuda.nvtx.range("backward"):
            loss.backward()
        if use_fsdp:
            model.finish_gradient_synchronization()
    torch.cuda.synchronize(device)
    if args.profile_nsys:
        torch.cuda.cudart().cudaProfilerStop()
    peak_forward_backward = torch.cuda.max_memory_allocated(device)
    allocated_before_optimizer_step = torch.cuda.memory_allocated(device)
    del logits, loss

    torch.cuda.reset_peak_memory_stats(device)
    with torch.cuda.nvtx.range("optimizer_step"):
        optimizer.step()
    torch.cuda.synchronize(device)
    peak_optimizer_step = torch.cuda.max_memory_allocated(device)
    allocated_after_step = torch.cuda.memory_allocated(device)

    timings_ms = []
    for _ in range(args.warmups):
        _train_step(model, optimizer, input_ids, labels, use_fsdp)
    torch.cuda.synchronize(device)
    for _ in range(args.trials):
        dist.barrier()
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        _train_step(model, optimizer, input_ids, labels, use_fsdp)
        torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - begin) * 1000)

    local_record = {
        "rank": rank,
        "gpu": torch.cuda.get_device_name(device),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        ),
        "gradient_bytes": sum(
            parameter.grad.numel() * parameter.grad.element_size()
            for parameter in model.parameters()
            if parameter.grad is not None
        ),
        "optimizer_state_tensor_bytes": sum(
            tensor.numel() * tensor.element_size()
            for state in optimizer.state.values()
            for tensor in state.values()
            if isinstance(tensor, torch.Tensor)
        ),
        "memory_after_init_bytes": memory_after_init,
        "peak_forward_backward_bytes": peak_forward_backward,
        "allocated_before_optimizer_step_bytes": allocated_before_optimizer_step,
        "peak_optimizer_step_bytes": peak_optimizer_step,
        "allocated_after_step_bytes": allocated_after_step,
        "step_timings_ms": timings_ms,
        "mean_step_ms": statistics.mean(timings_ms),
    }
    records = [None for _ in range(world_size)]
    dist.all_gather_object(records, local_record)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(records, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
