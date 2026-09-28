"""Collect the two missing measurements for the memory-profiling report.

Examples
--------
python benchmark_memory_requirements.py --mode block_backward --model_size xl \
    --context_length 128 --batch_size 4 --record_memory

The Nsight run should use --mode full (or block_backward) and the CUDA profiler
range.  See analyze_nsys_ops.py for the operation top-five report.
"""

import argparse
import contextlib
import gc
import json
import statistics
import timeit
import weakref

import torch
import torch.cuda.nvtx as nvtx
import torch.nn.functional as F
from torch.optim import AdamW

from cs336_basics.model import Transformerlm


MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


def build_model(size, context_length, vocab_size, device):
    c = MODEL_CONFIGS[size]
    model = Transformerlm(
        vocab_size=vocab_size, context_length=context_length,
        d_model=c["d_model"], num_layers=c["num_layers"],
        num_heads=c["num_heads"], d_ff=c["d_ff"], rope_theta=10_000,
        device="cpu", dtype=torch.float32,
    ).to(device)
    return model.train()


def mib(n):
    return n / (1024 ** 2)


class SavedTensorReleaseTracker:
    """Count each autograd saved-tensor token until autograd drops it.

    This is the exact logical size of saved tensors released by autograd. It
    intentionally does not claim to be the same as CUDA ``reserved`` memory:
    the caching allocator can keep released blocks in its pool.
    """
    def __init__(self):
        self.saved_bytes = 0
        self.released_bytes = 0
        self.live_tokens = 0

    def pack(self, tensor):
        nbytes = tensor.numel() * tensor.element_size()
        self.saved_bytes += nbytes
        self.live_tokens += 1
        token = _SavedTensorToken(tensor, self, nbytes)
        weakref.finalize(token, self._released, nbytes)
        return token

    @staticmethod
    def unpack(token):
        return token.tensor

    def _released(self, nbytes):
        self.released_bytes += nbytes
        self.live_tokens -= 1


class _SavedTensorToken:
    __slots__ = ("tensor", "tracker", "nbytes", "__weakref__")

    def __init__(self, tensor, tracker, nbytes):
        self.tensor = tensor
        self.tracker = tracker
        self.nbytes = nbytes


def collect_operation_memory(step_fn, output_json=None):
    """Aggregate per-operation CUDA allocator deltas using torch.profiler.

    ``self_cuda_memory_usage`` is the operation-attributed tensor allocation /
    free metric. It is the appropriate operation-level companion to the
    Nsight timeline, while Nsight Systems' CUDA memory-operation report alone
    only contains memcpy/memset categories.
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.profiler.record_function("profiled_full_step"):
            step_fn()
    rows = []
    for event in prof.key_averages():
        value = getattr(event, "self_cuda_memory_usage", None)
        if value is None:
            value = getattr(event, "self_device_memory_usage", 0)
        if value and value > 0:
            rows.append({"operation": event.key, "bytes": int(value)})
    rows.sort(key=lambda r: r["bytes"], reverse=True)
    total = sum(r["bytes"] for r in rows)
    for rank, row in enumerate(rows[:5], 1):
        row["rank"] = rank
        row["mib"] = mib(row["bytes"])
        row["share_percent"] = 100.0 * row["bytes"] / total if total else 0.0
    result = {"metric": "self_cuda_memory_usage", "total_bytes": total,
              "top5": rows[:5]}
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


def block_backward_measure(block, x, warmups, autocast_ctx, output_json=None):
    """Measure net active-byte change across exactly one block backward.

    The tensor x and the block output remain alive at both samples. Therefore
    ``saved_tensor_release`` is the allocator-visible net release caused by
    backward, not the release caused by deleting Python references afterwards.
    The full-backward hook records the end point immediately after this block's
    backward callback, and NVTX makes the interval visible in Nsight Systems.
    """
    for _ in range(warmups):
        block.zero_grad(set_to_none=True)
        warm_x = x.detach().clone().requires_grad_(True)
        with autocast_ctx:
            warm_loss = block(warm_x).float().square().mean()
        warm_loss.backward()
    torch.cuda.synchronize()

    hook_sample = {}

    def after_block_backward(_module, _grad_input, _grad_output):
        torch.cuda.synchronize()
        hook_sample.update(
            allocated_after=int(torch.cuda.memory_allocated()),
            reserved_after=int(torch.cuda.memory_reserved()),
        )

    handle = block.register_full_backward_hook(after_block_backward)
    block.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    tracker = SavedTensorReleaseTracker()
    with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
        with autocast_ctx, nvtx.range("single_transformer_block_forward"):
            y = block(x)
            loss = y.float().square().mean()
    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated())
    reserved_before = int(torch.cuda.memory_reserved())

    with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
        with nvtx.range("single_transformer_block_backward"):
            loss.backward()
    torch.cuda.synchronize()
    del loss, y
    gc.collect()
    torch.cuda.synchronize()
    handle.remove()

    if not hook_sample:
        raise RuntimeError("TransformerBlock backward hook did not fire")
    result = {
        "allocated_before_backward_bytes": allocated_before,
        "allocated_after_backward_bytes": hook_sample["allocated_after"],
        "reserved_before_backward_bytes": reserved_before,
        "reserved_after_backward_bytes": hook_sample["reserved_after"],
        "saved_tensor_release_bytes": allocated_before - hook_sample["allocated_after"],
        "saved_tensor_release_mib": mib(allocated_before - hook_sample["allocated_after"]),
        "reserved_delta_bytes": hook_sample["reserved_after"] - reserved_before,
        "backward_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "backward_peak_delta_bytes": int(torch.cuda.max_memory_allocated()) - allocated_before,
        "autograd_saved_tensor_bytes": tracker.saved_bytes,
        "autograd_saved_tensor_released_bytes": tracker.released_bytes,
        "autograd_saved_tensor_released_mib": mib(tracker.released_bytes),
        "autograd_saved_tensor_live_tokens": tracker.live_tokens,
    }
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_size", choices=MODEL_CONFIGS, default="small")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--vocab_size", type=int, default=10_000)
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--mode", choices=("block_backward", "full"), default="block_backward")
    p.add_argument("--block_index", type=int, default=0)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--output_json", default=None)
    p.add_argument("--operation_memory_json", default=None)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = "cuda"
    model = build_model(args.model_size, args.context_length, args.vocab_size, device)
    autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if args.dtype == "bf16" else contextlib.nullcontext())

    if args.mode == "block_backward":
        if not 0 <= args.block_index < len(model.layers):
            raise ValueError(f"block_index must be in [0, {len(model.layers) - 1}]")
        c = MODEL_CONFIGS[args.model_size]
        x = torch.randn(args.batch_size, args.context_length, c["d_model"],
                        device=device, dtype=torch.float32)
        block_backward_measure(model.layers[args.block_index], x, args.warmups,
                               autocast_ctx, args.output_json)
        return

    ids = torch.randint(args.vocab_size, (args.batch_size, args.context_length),
                        device=device, dtype=torch.long)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.context_length),
                            device=device, dtype=torch.long)
    optimizer = AdamW(model.parameters())

    def step():
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx, nvtx.range("forward_step"):
            logits = model(ids)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), targets.reshape(-1))
        with nvtx.range("backward_step"):
            loss.backward()
        with nvtx.range("optimizer_step"):
            optimizer.step()

    # One profiled iteration gives operation-level memory attribution. Keep it
    # separate from the Nsight run because profiler instrumentation changes
    # timing and can slightly change allocator behavior.
    if args.operation_memory_json:
        for _ in range(args.warmups):
            step()
        torch.cuda.synchronize()
        collect_operation_memory(step, args.operation_memory_json)

    for _ in range(args.warmups):
        step()
    torch.cuda.synchronize()
    timings = []
    torch.cuda.cudart().cudaProfilerStart()
    with nvtx.range("nsys_capture_full_step"):
        for _ in range(args.trials):
            torch.cuda.synchronize()
            start = timeit.default_timer()
            step()
            torch.cuda.synchronize()
            timings.append((timeit.default_timer() - start) * 1000)
    torch.cuda.cudart().cudaProfilerStop()
    print(f"full step: {statistics.mean(timings):.2f} ms")


if __name__ == "__main__":
    main()
