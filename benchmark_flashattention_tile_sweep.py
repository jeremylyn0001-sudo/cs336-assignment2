#!/usr/bin/env python3
"""Sweep backward tile shapes for D=128 FlashAttention on one CUDA GPU."""
from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import torch
import triton
import triton.testing

import cs336_systems.flashattention_triton as fa


DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
CONFIGS = {
    "bf16": [
        ("q32k64w4s3", 32, 64, 4, 3),
        ("q32k64w8s3", 32, 64, 8, 3),
        ("q32k64w8s2", 32, 64, 8, 2),
        ("q64k64w8s2", 64, 64, 8, 2),
        ("q64k32w8s3", 64, 32, 8, 3),
    ],
    "fp32": [
        ("q32k32w4s1", 32, 32, 4, 1),
        ("q32k32w8s1", 32, 32, 8, 1),
        ("q64k32w8s1", 64, 32, 8, 1),
        ("q32k64w8s1", 32, 64, 8, 1),
        ("q32k32w4s2", 32, 32, 4, 2),
    ],
}


def measure(fn, warmup_ms: int, rep_ms: int) -> float:
    torch.cuda.synchronize()
    value = triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms)
    torch.cuda.synchronize()
    return float(value)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", default="128,256,512,1024,2048,4096,8192,16384,32768,65536")
    parser.add_argument("--dtypes", default="bf16,fp32")
    parser.add_argument("--output", default="flashattention_d128_tile_sweep_h800.csv")
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    args = parser.parse_args()
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x]
    dtype_names = [x.strip() for x in args.dtypes.split(",") if x.strip()]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    output = Path(args.output)
    fields = [
        "gpu", "seq_len", "head_dim", "dtype", "candidate", "block_q", "block_k",
        "num_warps", "num_stages", "warmup_ms", "rep_ms", "backward_ms", "e2e_ms", "status", "error",
    ]
    original_selector = fa._select_backward_tile_config
    total = sum(len(CONFIGS[name]) for name in dtype_names) * len(seq_lens)
    index = 0
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        stream.flush()
        for dtype_name in dtype_names:
            dtype = DTYPES[dtype_name]
            for n in seq_lens:
                q0 = torch.randn((1, n, 128), device="cuda", dtype=dtype)
                k0 = torch.randn_like(q0)
                v0 = torch.randn_like(q0)
                do = torch.randn_like(q0)
                for name, block_q, block_k, warps, stages in CONFIGS[dtype_name]:
                    index += 1
                    config = (block_q, block_k, warps, stages)
                    fa._select_backward_tile_config = lambda *unused, _config=config: _config
                    result = {"backward_ms": "", "e2e_ms": "", "status": "ok", "error": ""}
                    print(f"[{index}/{total}] N={n} D=128 {dtype_name} {name}", flush=True)
                    q = q0.detach().requires_grad_(True)
                    k = k0.detach().requires_grad_(True)
                    v = v0.detach().requires_grad_(True)
                    try:
                        out = fa.FlashAttention_Triton.apply(q, k, v, True)
                        result["backward_ms"] = measure(
                            lambda: torch.autograd.grad(
                                out, (q, k, v), grad_outputs=do, retain_graph=True
                            ),
                            args.warmup_ms,
                            args.rep_ms,
                        )
                    except Exception as exc:
                        if isinstance(exc, torch.cuda.OutOfMemoryError):
                            torch.cuda.empty_cache()
                        result["status"] = "error"
                        result["error"] = f"backward:{type(exc).__name__}:{str(exc).splitlines()[0]}"
                    finally:
                        del q, k, v
                    def e2e():
                        q1 = q0.detach().requires_grad_(True)
                        k1 = k0.detach().requires_grad_(True)
                        v1 = v0.detach().requires_grad_(True)
                        out1 = fa.FlashAttention_Triton.apply(q1, k1, v1, True)
                        torch.autograd.grad(out1, (q1, k1, v1), grad_outputs=do)
                    if result["status"] == "ok":
                        try:
                            result["e2e_ms"] = measure(e2e, args.warmup_ms, args.rep_ms)
                        except Exception as exc:
                            if isinstance(exc, torch.cuda.OutOfMemoryError):
                                torch.cuda.empty_cache()
                            result["status"] = "partial"
                            result["error"] = f"e2e:{type(exc).__name__}:{str(exc).splitlines()[0]}"
                    writer.writerow({
                        "gpu": torch.cuda.get_device_name(0), "seq_len": n, "head_dim": 128,
                        "dtype": dtype_name, "candidate": name, "block_q": block_q,
                        "block_k": block_k, "num_warps": warps, "num_stages": stages,
                        "warmup_ms": args.warmup_ms, "rep_ms": args.rep_ms,
                        **result,
                    })
                    stream.flush()
                    print(f"    bwd={result['backward_ms']} e2e={result['e2e_ms']} status={result['status']}", flush=True)
                    del e2e
                    gc.collect()
                    torch.cuda.empty_cache()
                del q0, k0, v0, do
                gc.collect()
                torch.cuda.empty_cache()
    fa._select_backward_tile_config = original_selector
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    run()
