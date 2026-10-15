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

The leaderboard row is explicitly a 2×H20 substitute run. The official timing target is 2×B200, so it must not be presented as an official B200 result. The interactive math exercise index is under [`study_guides/`](../study_guides/a1_a2_math_practice.md).

## Remaining gaps before calling the full lab complete

- The official leaderboard measurement still needs two B200 GPUs and a verified empty PyTorch/Triton cache; the H20 run is a substitute only.
- The DDP report itself notes that full-model training scaling was measured on two ranks, not across 2/4/6 ranks.
- The FA2 substitute report has four dense FP32 OOM cases and no direct peak-memory measurements; it reports these as partial rather than filling in unsupported results.
- For the large checkpointing configuration, the no-checkpoint baseline OOMed before gradient equivalence could be measured. Correctness was verified separately on a reduced configuration.
- The A2 §8.3–§8.5 FSDP, TP and 2D parallelism derivations remain to be solved in the math-practice page; §8.1–§8.2 already have a written calculation note.

So the major experiment collection is in place, but it is more accurate to say **the lab is close, not completely closed**. Keep these scope limits with any score or portfolio summary.
