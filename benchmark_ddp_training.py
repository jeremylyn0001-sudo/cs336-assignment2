#!/usr/bin/env python3
"""Quick XL DDP benchmark: per-parameter, flat-gradient, and overlapped sync."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from torch.optim import AdamW

from benchmark import MODEL_CONFIGS, build_model, VOCAB_SIZE
from cs336_systems.DDP import DDP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Global batch, split evenly across ranks.")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--output", default="ddp_training_results.csv")
    parser.add_argument(
        "--modes", nargs="+", choices=["naive", "flat", "overlap"],
        default=["naive", "flat", "overlap"],
    )
    return parser.parse_args()


def broadcast_parameters(module):
    with torch.no_grad():
        for parameter in module.parameters():
            dist.broadcast(parameter, src=0)


def train_step(model, optimizer, x, targets, mode, world_size, device):
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    step_start = time.perf_counter()

    with nvtx.range(f"{mode}_forward"):
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
        )
    with nvtx.range(f"{mode}_backward"):
        loss.backward()

    comm_metric_ms = 0.0
    if mode == "naive":
        # Communicate each ready gradient after backward has completed.
        torch.cuda.synchronize(device)
        comm_start = time.perf_counter()
        with nvtx.range("gradient_sync_individual"):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        comm_metric_ms = (time.perf_counter() - comm_start) * 1000.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(world_size)

    elif mode == "flat":
        # Flatten all gradients and communicate them in one collective.
        parameters = [p for p in model.parameters() if p.grad is not None]
        flat = torch.cat([p.grad.contiguous().view(-1) for p in parameters])
        torch.cuda.synchronize(device)
        comm_start = time.perf_counter()
        with nvtx.range("gradient_sync_flat"):
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        comm_metric_ms = (time.perf_counter() - comm_start) * 1000.0
        flat.div_(world_size)
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.grad.copy_(flat[offset:offset + count].view_as(parameter))
            offset += count

    elif mode == "overlap":
        # DDP hooks launched per-parameter asynchronous all-reduces during backward.
        wait_start = time.perf_counter()
        with nvtx.range("overlap_wait_tail"):
            model.finish_gradient_synchronization()
        torch.cuda.synchronize(device)
        # This is the remaining wait after backward, not total comm duration.
        comm_metric_ms = (time.perf_counter() - wait_start) * 1000.0

    with nvtx.range(f"{mode}_optimizer_step"):
        optimizer.step()
    torch.cuda.synchronize(device)
    step_ms = (time.perf_counter() - step_start) * 1000.0
    return step_ms, comm_metric_ms


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise ValueError("The XL training benchmark is specified for 2 GPUs.")
    if args.batch_size % world_size:
        raise ValueError("--batch-size must be divisible by world_size.")

    local_batch = args.batch_size // world_size
    model_cfg = MODEL_CONFIGS["xl"]
    print(
        f"rank={rank}/{world_size} GPU={torch.cuda.get_device_name(local_rank)} "
        f"XL={model_cfg} global_batch={args.batch_size} "
        f"local_batch={local_batch} seq={args.sequence_length}",
        flush=True,
    )

    # Identical global synthetic batch on all ranks, then take disjoint shards.
    torch.manual_seed(2026)
    all_x = torch.randint(
        0, VOCAB_SIZE, (args.batch_size, args.sequence_length), dtype=torch.long
    )
    all_y = torch.randint(
        0, VOCAB_SIZE, (args.batch_size, args.sequence_length), dtype=torch.long
    )
    start = rank * local_batch
    x = all_x[start:start + local_batch].to(device)
    targets = all_y[start:start + local_batch].to(device)

    rows = []
    try:
        for mode in args.modes:
            # Make independent replicas; DDP/manual broadcast syncs from rank 0.
            torch.manual_seed(1000 + rank)
            base_model = build_model(
                "xl",
                context_length=args.sequence_length,
                vocab_size=VOCAB_SIZE,
                device=str(device),
            )
            if mode == "overlap":
                model = DDP(base_model)
            else:
                broadcast_parameters(base_model)
                model = base_model

            optimizer = AdamW(model.parameters())
            step_fn = lambda: train_step(
                model, optimizer, x, targets, mode, world_size, device
            )

            for _ in range(args.warmups):
                step_fn()
            torch.cuda.synchronize(device)

            step_samples = []
            comm_samples = []
            for _ in range(args.trials):
                dist.barrier()
                step_ms, comm_ms = step_fn()
                step_samples.append(step_ms)
                comm_samples.append(comm_ms)

            rank_result = (
                statistics.mean(step_samples),
                statistics.mean(comm_samples),
            )
            gathered = [None] * world_size
            dist.all_gather_object(gathered, rank_result)

            if rank == 0:
                step_mean = statistics.mean(x[0] for x in gathered)
                step_max = max(x[0] for x in gathered)
                comm_mean = statistics.mean(x[1] for x in gathered)
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "gpu": torch.cuda.get_device_name(local_rank),
                    "world_size": world_size,
                    "model": "xl",
                    "global_batch": args.batch_size,
                    "local_batch": local_batch,
                    "sequence_length": args.sequence_length,
                    "mode": mode,
                    "warmups": args.warmups,
                    "trials": args.trials,
                    "mean_step_ms": step_mean,
                    "max_rank_step_ms": step_max,
                    "mean_communication_or_wait_ms": comm_mean,
                }
                rows.append(row)
                print(
                    f"mode={mode} mean_step={step_mean:.2f} ms "
                    f"max_rank_step={step_max:.2f} ms "
                    f"comm_or_wait={comm_mean:.2f} ms",
                    flush=True,
                )

            del step_fn, optimizer, model, base_model
            gc.collect()
            torch.cuda.empty_cache()

        if rank == 0:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            fields = list(rows[0].keys())
            write_header = not output.exists() or output.stat().st_size == 0
            with output.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)
            print(f"Saved results to {output.resolve()}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
