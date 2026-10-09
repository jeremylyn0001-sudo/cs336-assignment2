from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path

import torch
import triton

from cs336_systems.flashattention_triton import FlashAttention_Triton


SEQ_LENS = [2**p for p in range(7, 17)]
HEAD_DIMS = [16, 32, 64, 128]
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def dense_causal_attention(mask: torch.Tensor, scale: float):
    """Regular dense PyTorch attention; materializes the Nq x Nk scores."""
    def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores.mul_(scale)
        scores.masked_fill_(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)
    return run


def measure(fn, warmup_ms: int, rep_ms: int) -> float:
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms)
    torch.cuda.synchronize()
    return float(ms)


def run_implementation(impl, q0, k0, v0, do, warmup_ms, rep_ms):
    result = {"forward_ms": "", "backward_ms": "", "e2e_ms": "", "status": "ok", "error": ""}
    errors = []

    def try_measure(label, fn):
        try:
            return measure(fn, warmup_ms, rep_ms)
        except Exception as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            errors.append(f"{label}:{type(exc).__name__}:{str(exc).splitlines()[0]}")
            return ""

    def forward_only():
        with torch.no_grad():
            impl(q0, k0, v0)

    result["forward_ms"] = try_measure("forward", forward_only)

    q = q0.detach().requires_grad_(True)
    k = k0.detach().requires_grad_(True)
    v = v0.detach().requires_grad_(True)
    try:
        out = impl(q, k, v)
        torch.cuda.synchronize()
        result["backward_ms"] = try_measure(
            "backward",
            lambda: torch.autograd.grad(out, (q, k, v), grad_outputs=do, retain_graph=True),
        )
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        errors.append(f"backward_setup:{type(exc).__name__}:{str(exc).splitlines()[0]}")
    del q, k, v

    def end_to_end():
        q = q0.detach().requires_grad_(True)
        k = k0.detach().requires_grad_(True)
        v = v0.detach().requires_grad_(True)
        out = impl(q, k, v)
        torch.autograd.grad(out, (q, k, v), grad_outputs=do)

    result["e2e_ms"] = try_measure("e2e", end_to_end)
    if errors:
        result["status"] = "partial"
        result["error"] = " | ".join(errors)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="flashattention_benchmark_h800.csv")
    parser.add_argument("--seq-lens", default=",".join(map(str, SEQ_LENS)))
    parser.add_argument("--head-dims", default=",".join(map(str, HEAD_DIMS)))
    parser.add_argument("--dtypes", default="bf16,fp32")
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    args = parser.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",") if x]
    head_dims = [int(x) for x in args.head_dims.split(",") if x]
    dtype_names = [x.strip() for x in args.dtypes.split(",") if x.strip()]
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Configs: {len(seq_lens) * len(head_dims) * len(dtype_names)}; warmup={args.warmup_ms}ms rep={args.rep_ms}ms")

    output = Path(args.output)
    fields = ["gpu", "seq_len", "head_dim", "dtype", "implementation", "forward_ms", "backward_ms", "e2e_ms", "status", "error"]
    implementations = ("triton_fa2", "pytorch_dense")
    total = len(seq_lens) * len(head_dims) * len(dtype_names) * len(implementations)
    index = 0
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
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

                    for implementation in implementations:
                        index += 1
                        positions = None
                        mask = None
                        if implementation == "triton_fa2":
                            impl = lambda q, k, v: FlashAttention_Triton.apply(q, k, v, True)
                        else:
                            positions = torch.arange(n, device="cuda")
                            mask = positions[None, :] > positions[:, None]
                            impl = dense_causal_attention(mask, scale)
                        print(f"[{index}/{total}] {implementation} N={n} D={d} {dtype_name}", flush=True)
                        try:
                            result = run_implementation(impl, q0, k0, v0, do, args.warmup_ms, args.rep_ms)
                        except Exception as exc:
                            if isinstance(exc, torch.cuda.OutOfMemoryError):
                                torch.cuda.empty_cache()
                            result = {"forward_ms": "", "backward_ms": "", "e2e_ms": "", "status": "error", "error": f"{type(exc).__name__}:{str(exc).splitlines()[0]}"}
                        writer.writerow({
                            "gpu": torch.cuda.get_device_name(0),
                            "seq_len": n,
                            "head_dim": d,
                            "dtype": dtype_name,
                            "implementation": implementation,
                            **result,
                        })
                        stream.flush()
                        print(f"    fwd={result['forward_ms']} bwd={result['backward_ms']} e2e={result['e2e_ms']} status={result['status']}", flush=True)
                        gc.collect()
                        del impl, mask, positions
                        torch.cuda.empty_cache()
                    del q0, k0, v0, do
                    gc.collect()
                    torch.cuda.empty_cache()
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
