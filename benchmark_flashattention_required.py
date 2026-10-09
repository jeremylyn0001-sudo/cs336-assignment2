#!/usr/bin/env python3
"""Required FA2 benchmark path: Triton forward + torch.compile backward."""

from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path

import torch

from benchmark_flashattention import (
    DTYPES,
    HEAD_DIMS,
    SEQ_LENS,
    dense_causal_attention,
    run_implementation,
)
from cs336_systems.flashattention_triton import FlashAttention_Triton_CompiledBackward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="flashattention_benchmark_h800_compiled_backward.csv",
    )
    parser.add_argument("--seq-lens", default=",".join(map(str, SEQ_LENS)))
    parser.add_argument("--head-dims", default=",".join(map(str, HEAD_DIMS)))
    parser.add_argument("--dtypes", default="bf16,fp32")
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Resume a prior sweep and skip completed rows")
    args = parser.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",") if x]
    head_dims = [int(x) for x in args.head_dims.split(",") if x]
    dtype_names = [x.strip() for x in args.dtypes.split(",") if x.strip()]

    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Configs: {len(seq_lens) * len(head_dims) * len(dtype_names)}; "
        f"warmup={args.warmup_ms}ms rep={args.rep_ms}ms"
    )

    output = Path(args.output)
    fields = [
        "gpu", "seq_len", "head_dim", "dtype", "implementation",
        "warmup_ms", "rep_ms", "forward_ms", "backward_ms", "e2e_ms", "status", "error",
    ]
    implementations = ("triton_fwd_compiled_bwd", "pytorch_dense")
    total = len(seq_lens) * len(head_dims) * len(dtype_names) * len(implementations)
    index = 0

    existing = {}
    if args.resume and output.exists():
        with output.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != fields:
                raise ValueError(f"Cannot resume {output}: CSV columns do not match this script")
            for row in reader:
                key = (row["seq_len"], row["head_dim"], row["dtype"], row["implementation"])
                existing[key] = row
    completed = {
        key for key, row in existing.items() if row["status"] in {"ok", "partial"}
    }
    mode = "a" if args.resume and output.exists() else "w"
    write_header = mode == "w" or output.stat().st_size == 0
    with output.open(mode, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        if write_header:
            writer.writeheader()
            stream.flush()

        for n in seq_lens:
            for d in head_dims:
                for dtype_name in dtype_names:
                    dtype = DTYPES[dtype_name]
                    q0 = torch.randn((1, n, d), device="cuda", dtype=dtype)
                    k0 = torch.randn_like(q0)
                    v0 = torch.randn_like(q0)
                    do = torch.randn_like(q0)
                    scale = 1.0 / math.sqrt(d)

                    positions = torch.arange(n, device="cuda")
                    mask = positions[None, :] > positions[:, None]
                    impls = {
                        "triton_fwd_compiled_bwd": (
                            lambda q, k, v:
                            FlashAttention_Triton_CompiledBackward.apply(
                                q, k, v, True
                            )
                        ),
                        "pytorch_dense": dense_causal_attention(mask, scale),
                    }

                    for implementation in implementations:
                        index += 1
                        key = (str(n), str(d), dtype_name, implementation)
                        if key in completed:
                            print(f"[{index}/{total}] {implementation} N={n} D={d} {dtype_name} already complete; skipping", flush=True)
                            continue
                        impl = impls[implementation]
                        print(
                            f"[{index}/{total}] {implementation} "
                            f"N={n} D={d} {dtype_name}",
                            flush=True,
                        )
                        try:
                            result = run_implementation(
                                impl, q0, k0, v0, do,
                                args.warmup_ms, args.rep_ms
                            )
                        except Exception as exc:
                            if isinstance(exc, torch.cuda.OutOfMemoryError):
                                torch.cuda.empty_cache()
                            result = {
                                "forward_ms": "",
                                "backward_ms": "",
                                "e2e_ms": "",
                                "status": "error",
                                "error": f"{type(exc).__name__}:{str(exc).splitlines()[0]}",
                            }
                        writer.writerow({
                            "gpu": torch.cuda.get_device_name(0),
                            "seq_len": n,
                            "head_dim": d,
                            "dtype": dtype_name,
                            "implementation": implementation,
                            "warmup_ms": args.warmup_ms,
                            "rep_ms": args.rep_ms,
                            **result,
                        })
                        stream.flush()
                        print(
                            f"    fwd={result['forward_ms']} "
                            f"bwd={result['backward_ms']} "
                            f"e2e={result['e2e_ms']} status={result['status']}",
                            flush=True,
                        )
                        gc.collect()
                        torch.cuda.empty_cache()

                    del impls, mask, positions, q0, k0, v0, do
                    gc.collect()
                    torch.cuda.empty_cache()

    if args.resume:
        latest = {}
        with output.open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["seq_len"], row["head_dim"], row["dtype"], row["implementation"])
                latest[key] = row
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(latest.values())
        temporary.replace(output)
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
