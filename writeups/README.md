# Assignment 2 experiment notes

These notes collect the measurements and writeups completed for the systems lab. They are personal experiment records and are separate from Stanford's official leaderboard results.

| Area | Record |
|---|---|
| End-to-end and Nsight benchmarking | [`benchmark_writeup.md`](benchmark_writeup.md), [`nsight_benchmark_writeup.md`](nsight_benchmark_writeup.md) |
| Mixed precision and memory | [`mixed_precision_writeup.md`](mixed_precision_writeup.md), [`memory_profiling_writeup.md`](memory_profiling_writeup.md) |
| Activation checkpointing | [`checkpointing_writeup.md`](checkpointing_writeup.md) |
| FlashAttention-2 | [`flashattention_h800_writeup.md`](flashattention_h800_writeup.md), [`nsight_backward_quantitative.md`](nsight_backward_quantitative.md) |
| DDP and communication | [`ddp_writeup.md`](ddp_writeup.md) |
| ZeRO-1 optimizer-state sharding | [`optimizer_state_sharding_a800/writeup.md`](optimizer_state_sharding_a800/writeup.md) |
| FSDP | [`fsdp_rtx6000d/writeup.md`](fsdp_rtx6000d/writeup.md) |
| Parallelism calculations | [`parallelism_calculations.md`](parallelism_calculations.md) |
| Leaderboard workload on substitute GPUs | [`leaderboard_h20_20261015/README.md`](leaderboard_h20_20261015/README.md) |
| Remaining experiments and deliverables | [`remaining_a2_experiments.md`](remaining_a2_experiments.md) |

The leaderboard row is explicitly a 2×H20 substitute run. The official timing target is 2×B200, so it must not be presented as an official B200 result. The interactive math exercise index is under [`study_guides/`](../study_guides/a1_a2_math_practice.md).

The detailed closeout checklist separates required runs, missing screenshots, optional scale-out follow-up, and the B200-only leaderboard condition: [`remaining_a2_experiments.md`](remaining_a2_experiments.md).
