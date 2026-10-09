#!/usr/bin/env python3
"""Single-node NCCL all-reduce latency sweep for the CS336 DDP assignment."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes-mib", nargs="+", type=int, default=[1, 10, 100, 1024]
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--large-iterations", type=int, default=5)
    parser.add_argument(
        "--output",
        default="ddp_allreduce_results.csv",
        help="CSV output path; only rank 0 writes.",
    )
    return parser.parse_args()


def benchmark_one(payload_mib: int, warmup: int, iterations: int, device):
    numel = payload_mib * 1024 * 1024 // torch.tensor([], dtype=torch.float32).element_size()
    tensor = torch.zeros(numel, dtype=torch.float32, device=device)

    # Warm up NCCL and allocator before timing.
    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)

    # Align ranks before each measured collective. Synchronize CUDA afterwards
    # because NCCL work is asynchronous with respect to the host.
    samples_ms = []
    for _ in range(iterations):
        dist.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    return statistics.mean(samples_ms)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the NCCL communication benchmark.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gpu_name = torch.cuda.get_device_name(local_rank)

    rows = []
    try:
        for size_mib in args.sizes_mib:
            iterations = (
                args.large_iterations if size_mib >= 1024 else args.iterations
            )
            local_ms = benchmark_one(
                size_mib, args.warmup, iterations, device
            )

            rank_times = [None] * world_size
            dist.all_gather_object(rank_times, local_ms)

            if rank == 0:
                mean_rank_ms = statistics.mean(rank_times)
                max_rank_ms = max(rank_times)
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "gpu": gpu_name,
                    "world_size": world_size,
                    "payload_mib": size_mib,
                    "warmup_iterations": args.warmup,
                    "measured_iterations": iterations,
                    "mean_rank_latency_ms": mean_rank_ms,
                    "max_rank_latency_ms": max_rank_ms,
                }
                rows.append(row)
                print(
                    f"world_size={world_size} payload={size_mib} MiB "
                    f"mean_rank={mean_rank_ms:.3f} ms "
                    f"max_rank={max_rank_ms:.3f} ms",
                    flush=True,
                )

        if rank == 0:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            fields = [
                "timestamp_utc",
                "gpu",
                "world_size",
                "payload_mib",
                "warmup_iterations",
                "measured_iterations",
                "mean_rank_latency_ms",
                "max_rank_latency_ms",
            ]
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
