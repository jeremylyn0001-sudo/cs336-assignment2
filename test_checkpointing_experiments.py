import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from cs336_basics.model import Transformerlm
import gc

# 使用 32 层模型对比“不加检查点”与“加检查点”的显存差异
D_MODEL = 2560
D_FF = 10240
NUM_LAYERS = 32
NUM_HEADS = 32
VOCAB_SIZE = 10_000
BATCH_SIZE = 4
CONTEXT_LENGTH = 2048
DEVICE = "cuda"

def build_model():
    torch.cuda.empty_cache()
    model = Transformerlm(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=10_000,
        device="cpu",
        dtype=torch.float32,
    )
    return model.to(DEVICE).train()

# ----------------- 策略 1: 二分递归检查点 -----------------
def recursive_checkpoint_forward(layers, x, token_positions):
    if len(layers) == 1:
        return layers[0](x, token_positions)
    
    mid = len(layers) // 2
    left_layers = layers[:mid]
    right_layers = layers[mid:]
    
    left_fn = lambda act: recursive_checkpoint_forward(left_layers, act, token_positions)
    right_fn = lambda act: recursive_checkpoint_forward(right_layers, act, token_positions)
    
    x = checkpoint(left_fn, x, use_reentrant=False)
    x = checkpoint(right_fn, x, use_reentrant=False)
    return x

# ----------------- 策略 2: 分块检查点 -----------------
def chunked_checkpoint_forward(layers, x, token_positions, chunk_size=1):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    def make_run_chunk(chunk_layers):
        def run_chunk(act):
            for layer in chunk_layers:
                act = layer(act, token_positions)
            return act
        return run_chunk

    # range 的 stop 使用 len(layers)，确保最后一个不足 chunk_size 的尾部
    # 也会执行；不能使用 len(layers) // chunk_size 计算 chunk 数量。
    for start in range(0, len(layers), chunk_size):
        chunk_layers = layers[start : start + chunk_size]
        run_fn = make_run_chunk(chunk_layers)
        x = checkpoint(run_fn, x, use_reentrant=False)
    return x


def compare_gradients(base_grad, test_grad, atol=1e-4, rtol=1e-4):
    """Compare gradients for every parameter, not only the first parameter."""
    if base_grad is None:
        return "not measured: baseline OOM"
    if test_grad is None:
        return "not measured: checkpoint gradient unavailable"
    if len(base_grad) != len(test_grad):
        return "mismatch: parameter count differs"

    max_abs_error = 0.0
    max_rel_error = 0.0
    for index, (base, test) in enumerate(zip(base_grad, test_grad)):
        if base is None or test is None:
            if base is not None or test is not None:
                return f"mismatch: missing gradient at parameter {index}"
            continue
        if base.shape != test.shape or base.dtype != test.dtype:
            return f"mismatch: shape/dtype differs at parameter {index}"
        difference = (base - test).abs()
        abs_error = difference.max().item()
        rel_error = (difference / base.abs().clamp_min(1e-12)).max().item()
        max_abs_error = max(max_abs_error, abs_error)
        max_rel_error = max(max_rel_error, rel_error)
        if not torch.allclose(base, test, atol=atol, rtol=rtol):
            return (f"mismatch: parameter {index}, max_abs={abs_error:.3e}, "
                    f"max_rel={rel_error:.3e}")
    return f"match: max_abs={max_abs_error:.3e}, max_rel={max_rel_error:.3e}"


def cleanup_after_strategy(model):
    """Release references and cached blocks after a run/OOM before the next run."""
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ----------------- 统一测速与显存测量函数 -----------------
def measure_strategy(model, forward_type="none", chunk_size=1):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.zero_grad(set_to_none=True)
    
    # 构造固定的输入
    torch.manual_seed(42)
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), device=DEVICE)
    token_positions = torch.arange(CONTEXT_LENGTH, device=DEVICE)
    
    h = model.token_embeddings(x)
    
    if forward_type == "none":
        # 1. 默认无检查点（全部存激活值）
        for layer in model.layers:
            h = layer(h, token_positions)
    elif forward_type == "recursive":
        # 2. 二分递归检查点
        h = recursive_checkpoint_forward(model.layers, h, token_positions)
    elif forward_type == "chunked":
        # 3. 固定分块检查点
        h = chunked_checkpoint_forward(model.layers, h, token_positions, chunk_size=chunk_size)
        
    out = model.lm_head(model.ln_final(h))
    loss = out.sum()
    loss.backward()
    
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    # 将全部参数梯度复制到 CPU，避免下一次 measure_strategy 的
    # zero_grad 覆盖 baseline；compare_gradients 因此执行全参数比较。
    grad_snapshot = [
        parameter.grad.detach().cpu().clone()
        if parameter.grad is not None else None
        for parameter in model.parameters()
    ]
    return peak_mem_gb, grad_snapshot

def main():
    print("正在初始化测试模型 (32层 Transformer)...")
    model = build_model()
    
    print("\n" + "="*70)
    print(f"{'检查点策略 (Strategy)':<35} | {'显存峰值 (Peak Memory)':<20} | {'梯度校验'}")
    print("="*70)
    
    # 1. 基准线：不加检查点
    try:
        base_mem, base_grad = measure_strategy(model, forward_type="none")
        print(f"{'0. 无检查点 (No Checkpoint, O(N))':<35} | {base_mem:6.2f} GiB           | 基准值 (Baseline)")
    except torch.OutOfMemoryError:
        print(f"{'0. 无检查点 (No Checkpoint, O(N))':<35} | ❌ OOM               | -")
        base_grad = None
        cleanup_after_strategy(model)

    # 2. 二分递归检查点
    rec_mem, rec_grad = measure_strategy(model, forward_type="recursive")
    rec_status = compare_gradients(base_grad, rec_grad)
    print(f"{'1. 二分递归检查点 (Recursive, O(logN))':<35} | {rec_mem:6.2f} GiB           | {rec_status}")

    # 3. 分块检查点 (k=1, 2, 4, 8)
    for k in [1, 2, 4, 8]:
        try:
            chk_mem, chk_grad = measure_strategy(model, forward_type="chunked", chunk_size=k)
            chk_status = compare_gradients(base_grad, chk_grad)
            print(f"{f'2. 分块检查点 (Chunked k={k})':<35} | {chk_mem:6.2f} GiB           | {chk_status}")
        except torch.OutOfMemoryError:
            print(f"{f'2. 分块检查点 (Chunked k={k})':<35} | ❌ OOM               | -")
            cleanup_after_strategy(model)

    print("="*70)

if __name__ == "__main__":
    main()
