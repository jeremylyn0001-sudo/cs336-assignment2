"""A2 optimizer-state-sharding memory and step-time comparison (2 GPUs)."""
import argparse
import gc
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW

from cs336_basics.model import Transformerlm
from cs336_systems.DDP import DDP
from cs336_systems.ShardedOptimizer import SharedOptimizer

VOCAB_SIZE = 10_000
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}
MIB = 1024**2


def build_model(size, context_length, device):
    cfg = MODEL_CONFIGS[size]
    model = Transformerlm(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10_000,
        device="cpu",
        dtype=torch.float32,
    )
    return model.to(device).train()


def optimizer_state_bytes(optimizer):
    inner = optimizer.optim if isinstance(optimizer, SharedOptimizer) else optimizer
    return sum(
        value.numel() * value.element_size()
        for state in inner.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def parameter_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def gradient_bytes(model):
    return sum(
        p.grad.numel() * p.grad.element_size()
        for p in model.parameters()
        if p.grad is not None
    )


def train_step(model, optimizer, x, targets):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
    )
    loss.backward()
    model.finish_gradient_synchronization()
    optimizer.step()
    torch.cuda.synchronize()


def run_mode(mode, rank, world_size, args, device, x, targets):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(1000 + rank)
    base_model = build_model(args.model_size, args.sequence_length, device)
    model = DDP(base_model)
    if mode == "baseline":
        optimizer = AdamW(model.parameters())
    else:
        optimizer = SharedOptimizer(model.parameters(), AdamW)

    torch.cuda.synchronize(device)
    init_current = torch.cuda.memory_allocated(device)
    init_peak = torch.cuda.max_memory_allocated(device)
    param_bytes = parameter_bytes(model)

    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
    )
    loss.backward()
    model.finish_gradient_synchronization()
    torch.cuda.synchronize(device)
    before_step_current = torch.cuda.memory_allocated(device)
    before_step_peak = torch.cuda.max_memory_allocated(device)
    grad_bytes = gradient_bytes(model)

    torch.cuda.reset_peak_memory_stats(device)
    optimizer.step()
    torch.cuda.synchronize(device)
    after_step_current = torch.cuda.memory_allocated(device)
    after_step_peak = torch.cuda.max_memory_allocated(device)
    state_bytes = optimizer_state_bytes(optimizer)
    del logits, loss

    def ordinary_step():
        train_step(model, optimizer, x, targets)

    for _ in range(args.warmups):
        ordinary_step()
    torch.cuda.synchronize(device)

    samples = []
    for _ in range(args.trials):
        dist.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        ordinary_step()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)

    inner = optimizer.optim if isinstance(optimizer, SharedOptimizer) else optimizer
    result = {
        "rank": rank,
        "mode": mode,
        "init_current_mib": init_current / MIB,
        "init_peak_mib": init_peak / MIB,
        "before_step_current_mib": before_step_current / MIB,
        "before_step_peak_mib": before_step_peak / MIB,
        "after_step_current_mib": after_step_current / MIB,
        "after_step_peak_mib": after_step_peak / MIB,
        "parameter_bytes": param_bytes,
        "gradient_bytes": grad_bytes,
        "optimizer_state_bytes": state_bytes,
        "post_step_unaccounted_bytes": max(
            0, after_step_current - param_bytes - grad_bytes - state_bytes
        ),
        "local_optimizer_parameter_count": sum(
            p.numel() for group in inner.param_groups for p in group["params"]
        ),
        "step_times_ms": samples,
        "mean_step_ms": statistics.mean(samples),
        "stdev_step_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }

    inner.state.clear()
    for handle in model.hook_handles:
        handle.remove()
    model.hook_handles.clear()
    del inner, ordinary_step, optimizer, model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Global batch size, split evenly across the two ranks.")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("The official optimizer-sharding accounting uses 2 GPUs.")
    if args.batch_size % world_size:
        raise ValueError("Global batch size must be divisible by 2.")

    generator = torch.Generator().manual_seed(2026)
    all_x = torch.randint(
        VOCAB_SIZE, (args.batch_size, args.sequence_length),
        generator=generator, dtype=torch.long,
    )
    all_y = torch.randint(
        VOCAB_SIZE, (args.batch_size, args.sequence_length),
        generator=generator, dtype=torch.long,
    )
    local_batch = args.batch_size // world_size
    start = rank * local_batch
    x = all_x[start:start + local_batch].to(device)
    targets = all_y[start:start + local_batch].to(device)

    results = {}
    try:
        for mode in ("baseline", "sharded"):
            local = run_mode(mode, rank, world_size, args, device, x, targets)
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local)
            if rank == 0:
                results[mode] = gathered
                worst_step = max(item["mean_step_ms"] for item in gathered)
                max_init = max(item["init_peak_mib"] for item in gathered)
                max_before = max(item["before_step_peak_mib"] for item in gathered)
                max_after = max(item["after_step_peak_mib"] for item in gathered)
                print(
                    f"{mode}: slowest-rank mean step={worst_step:.2f} ms; "
                    f"peak MiB init/before-step/step={max_init:.1f}/"
                    f"{max_before:.1f}/{max_after:.1f}", flush=True,
                )
            dist.barrier()

        if rank == 0:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "gpu": torch.cuda.get_device_name(local_rank),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "world_size": world_size,
                "model_size": args.model_size,
                "model_config": MODEL_CONFIGS[args.model_size],
                "global_batch": args.batch_size,
                "local_batch": local_batch,
                "sequence_length": args.sequence_length,
                "dtype": "float32",
                "optimizer": "AdamW defaults",
                "warmups": args.warmups,
                "trials": args.trials,
                "memory_metric": "torch.cuda.max_memory_allocated/current memory_allocated",
            }
            output.write_text(
                json.dumps({"metadata": metadata, "results": results}, indent=2),
                encoding="utf-8",
            )
            print(f"Saved {output.resolve()}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
