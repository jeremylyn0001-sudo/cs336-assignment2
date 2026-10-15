# CS336 Assignment 2 Leaderboard Snapshot

Source: [official Stanford-CS336 Assignment 2 leaderboard](https://github.com/stanford-cs336/assignment2-systems-leaderboard), checked 2026-10-15.

## Official workload

The leaderboard measures one complete training step for the assignment's 8B-class Transformer on **two B200 GPUs**: BF16, causal attention, global batch 2, sequence length 32,768, vocabulary 151,936, `d_model=4096`, `d_ff=11008`, 34 layers and 32 heads. The handout requires a run from empty PyTorch/Triton caches to finish within 10 minutes and expects submissions to beat the naive 10-second baseline. A leaderboard submission also has to preserve model behavior and pass correctness checks.

The fastest entries displayed when this snapshot was taken were between 3.837 s and 5.745 s per step, with the naive baseline at 10 s. Treat student entries as reported timings; leaderboard verification status is separate from the published time.

## Substitute-hardware experiment

Because B200 was unavailable, the same model dimensions and global batch were measured on **2×H20**. The first step in this process took 122.65 s and the three steady steps averaged 121.18 s. Peak PyTorch allocation was 81.90 GB per rank. This run uses activation checkpointing and 1,024-token chunked vocabulary projection/cross-entropy to fit H20 memory. We did not clear and verify both PyTorch and Triton caches before the run, so it does not establish the handout's formal empty-cache timing condition. It is useful as a workload experiment but is **not a B200 score and was not submitted to the official leaderboard**.

Full configuration, raw measurements, log, and runner: [`leaderboard_h20_20261015/`](../writeups/leaderboard_h20_20261015/README.md). The JSON records both ranks. The run passed a reduced two-rank smoke test before the full workload.

## What remains

To claim completion of the official leaderboard experiment, run the official configuration on two B200s from empty PyTorch/Triton caches, validate correctness against the assignment model, record the official timing, and submit it to the leaderboard. The H20 measurement does not satisfy those hardware-specific steps.
