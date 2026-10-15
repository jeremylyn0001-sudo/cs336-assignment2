# CS336 A2 leaderboard workload on 2×H20

This is a substitute-hardware measurement of the official two-B200 leaderboard workload. It is **not a B200 leaderboard score**.

## Result

| Measurement | Result |
|---|---:|
| First training step, including Triton JIT | 122.65 s |
| Mean of 3 steady-state steps | 121.18 s |
| Peak PyTorch allocated memory, each rank | 81,895,039,488 bytes (76.27 GiB) |
| Post-initialization PyTorch allocation, each rank | 16,822,345,728 bytes (15.67 GiB) |
| First-step loss, rank 0 / rank 1 | 11.9575 / 11.9556 |

The first measured step in this process completed in about 123 s. The official handout's 600 s requirement is specifically for a run starting with empty PyTorch/Triton caches. We did not clear and verify both caches before this H20 run, so this measurement does **not** establish a formal cold-cache pass. It also uses H20 rather than B200. The fastest public B200 entry reports 3.837 s/step, so the raw ratio is about 31.6×; this mixes hardware and implementation differences and is not a direct comparison of kernel quality.

## Workload and implementation

- Hardware: 2×NVIDIA H20, 95 GiB visible memory per GPU; driver 580.82.07.
- Software: Python 3.12.3, PyTorch 2.12.1+cu130, CUDA runtime 13.0, Triton 3.7.1.
- Repository revision: `7aaf7babeb14c7144984c429b7034de2788fa8d9` (`main`).
- Model: vocabulary 151,936; `d_model=4096`; `d_ff=11008`; 34 layers; 32 heads; sequence 32,768; BF16; global batch 2.
- Training step: causal Triton FlashAttention, custom asynchronous DDP gradient averaging, token-mean cross-entropy, and `torch.optim.AdamW`.
- Memory controls: activation checkpointing on each transformer layer and a chunked 1,024-token vocabulary projection/cross-entropy. The chunked calculation sums token losses and divides by the total token count, preserving the global token mean while limiting live logits to one chunk.
- Timing: one warmup and three timed steps. The first-step timer starts after model/device initialization and includes Triton JIT. Each rank uses the same three timing trials; reported times differ by less than 1 ms.

Reproduce from the repository root with:

```bash
torchrun --standalone --nproc_per_node=2 writeups/leaderboard_h20_20261015/leaderboard_h20_benchmark.py \
  --checkpoint --loss-chunk-tokens 1024 \
  --output writeups/leaderboard_h20_20261015/leaderboard_h20_2gpu.json
```

The initial attempt with a full 32K×151,936 logits tensor ran out of memory during backward (the GPU had about 1.2 GiB free and backward requested another 9.27 GiB). The successful run used chunked projection/cross-entropy. A reduced-size two-rank smoke test also completed forward, backward, gradient synchronization, AdamW update, and JSON output before the full run.

## Files

- `leaderboard_h20_2gpu.json`: per-rank measurements and exact benchmark configuration.
- `leaderboard_h20_2gpu.log`: captured run output, including both rank records.
- Runner: [`leaderboard_h20_benchmark.py`](leaderboard_h20_benchmark.py).

The full-model run completed with finite loss and no CUDA errors. It was not submitted to or verified by the B200 leaderboard, and it did not validate the official empty-cache timing condition.
