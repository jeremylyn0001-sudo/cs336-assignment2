import argparse
import contextlib
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
import timeit
import statistics
from torch.optim import AdamW
from cs336_basics.model import Transformerlm
import torch.cuda.nvtx as nvtx

VOCAB_SIZE = 10_000
BATCH_SIZE = 4
CONTEXT_LENGTH = 512
DEVICE = "cuda"
DTYPE = torch.float32

NUM_WARMUPS = 2
NUM_TRIALS = 1

MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B":    {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


def build_model(model_size: str, context_length: int, vocab_size: int = 10_000, device: str = "cuda"):
    cfg = MODEL_CONFIGS[model_size]
    
    torch.cuda.empty_cache()
    
    # 1. 在 CPU 上构建并完成权重初始化（不会触发 CUDA NVRTC 报错）
    model = Transformerlm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10_000,
        device="cpu",          # 👈 改为 "cpu"
        dtype=torch.float32,
    )
    
    # 2. 统一将完整模型和 Buffer 转移到 GPU
    model = model.to(device)
    return model.train()


def verify_correctness(f1, f2, dim = 16384, atol = 1e-5, rtol = 1e-5):
    x = torch.randn(dim, device='cuda', dtype=torch.float32)
    y1 = f1(x)
    y2 = f2(x)
    is_close = torch.allclose(y1, y2, atol=1e-5, rtol=1e-5)
    if is_close:
        print(f"Correctness verified: tensors are close within atol = {atol:.6e} and rtol = {rtol:.6e}")
    else:
        print(f"Correctness failed: tensors are not close within atol = {atol:.6e} and rtol = {rtol:.6e}")


def run_benchmark(step_fn, warmups: int, trials: int):
    # Warmup
    for _ in range(warmups):
        step_fn()
    torch.cuda.synchronize()

    # 正式测量
    timings_ms = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = timeit.default_timer()
        step_fn()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        timings_ms.append((end - start) * 1000)

    mean_time = statistics.mean(timings_ms)
    std_time = statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0
    return mean_time, std_time, timings_ms


def profile_once(step_fn, warmups=2):
    # warmup：Nsight 不采集
    for _ in range(warmups):
        step_fn()
        torch.cuda.synchronize()
    # 开始采集
    torch.cuda.cudart().cudaProfilerStart()
    # 保留 NVTX 名称，方便之后在 GUI 中识别
    with nvtx.range("profile_forward"):
        step_fn()
        torch.cuda.synchronize()
    # 停止采集
    torch.cuda.cudart().cudaProfilerStop()
    
    
def main():
    parser = argparse.ArgumentParser(description="CS336 Profiling & Benchmarking Harness")
    parser.add_argument("--model_size", type=str, default="small", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--mode", type=str, default="full", choices=["forward", "backward", "full"])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--record_memory", action="store_true", help="Dump PyTorch memory snapshot")
    parser.add_argument("--snapshot_prefix", type=str, default="memory_snapshot")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到 CUDA GPU。")

    print("GPU:", torch.cuda.get_device_name(0))
    
    device = "cuda"
    model = build_model(args.model_size, args.context_length, args.vocab_size, device)
    optimizer = AdamW(model.parameters())

    # 输入 token 和训练目标都必须是 long/int64
    input_ids = torch.randint(
        0, args.vocab_size,
        (args.batch_size, args.context_length),
        device=DEVICE,
        dtype=torch.long,
    )
    targets = torch.randint(
        0, args.vocab_size,
        (args.batch_size, args.context_length),
        device=DEVICE,
        dtype=torch.long,
    )
    
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bf16"
        else contextlib.nullcontext()
    )
    # 1. 只测 forward
    def forward_fn():
        with torch.inference_mode(), autocast_ctx, nvtx.range("forward_pass"):
            _ = model(input_ids)

    def backward_fn():
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx, nvtx.range("forward_for_backward"):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), targets.reshape(-1))
        with nvtx.range("backward_pass"):
            loss.backward()

    def full_step_fn():
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx, nvtx.range("forward_step"):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), targets.reshape(-1))
        with nvtx.range("backward_step"):
            loss.backward()
        with nvtx.range("optimizer_step"):
            optimizer.step()
    
    step_fn_map = {
        "forward": forward_fn,
        "backward": backward_fn,
        "full": full_step_fn,
    }
    step_fn = step_fn_map[args.mode]
    
    if args.record_memory:
        # 先做 warmup
        for _ in range(args.warmups):
            step_fn()
        torch.cuda.synchronize()

        # 开始记录显存历史
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)
        step_fn()
        torch.cuda.synchronize()
        filename = f"{args.snapshot_prefix}_{args.model_size}_ctx{args.context_length}_{args.mode}_{args.dtype}.pickle"
        torch.cuda.memory._dump_snapshot(filename)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"[Memory Snapshot] Saved to: {filename}")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MiB")
        return

    # ---------------- 正常 Benchmark 计时 ----------------
    mean_ms, std_ms, _ = run_benchmark(step_fn, warmups=args.warmups, trials=args.trials)
    print(f"[{args.model_size.upper()}] dtype={args.dtype} mode={args.mode:8s} | "
          f"Time: {mean_ms:.2f} ± {std_ms:.2f} ms (warmups={args.warmups}, trials={args.trials})")

if __name__ == "__main__":
    main()