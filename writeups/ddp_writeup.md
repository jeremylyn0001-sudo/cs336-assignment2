# Distributed Data Parallel: Implementation and Benchmark

## Scope

This writeup records the custom DDP wrapper, its correctness check, and the available single-node communication and XL training benchmarks. The implementation and existing GPU measurements are now integrated with the assignment adapters. Communication was measured at 2, 4, and 6 ranks; the XL training comparison was measured at 2 ranks only.

## Wrapper and correctness

`cs336_systems/DDP.py` broadcasts each parameter from rank 0 when the wrapper is created. For every trainable parameter, it registers a post-accumulation gradient hook. The hook divides the local gradient by `world_size`, launches an asynchronous sum all-reduce, and stores the returned work handle. `ddp_on_after_backward` waits for those collectives before the optimizer step. Dividing each rank's gradient before the sum produces the mean gradient while allowing communication for earlier-ready parameters to overlap with the rest of backward.

The official `tests/test_ddp.py` passed both model cases (`ToyModel` and `ToyModelWithTiedWeights`) on CPU/Gloo in two consecutive integration runs: **2 passed in 29.78 seconds**, then **2 passed in 31.60 seconds**. The check compares the distributed model's parameters with the non-parallel reference after training on rank-local data.

## Hardware and measurement procedure

| Experiment | Configuration |
|---|---|
| GPU node | 6 × NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB each |
| Driver / CUDA | 595.71.05 / 13.2 |
| Communication backend | NCCL, one node, FP32 all-reduce |
| Payloads / ranks | 1, 10, 100, and 1,024 MiB per rank; 2, 4, and 6 ranks |
| Communication timing | 5 warmups; 20 measured iterations, except 5 for 1,024 MiB |
| XL training | 2 ranks; global batch 4 (local batch 2); sequence length 512; FP32 + AdamW |
| XL model | `d_model=2560`, `d_ff=10240`, 32 layers, 32 heads |
| Training timing | 1 warmup and 2 measured trials per synchronization mode |

Each all-reduce latency is the mean across ranks. Raw per-rank means and slowest-rank values are in [`ddp_allreduce_results.csv`](../ddp_allreduce_results.csv). The training CSV is [`ddp_training_results.csv`](../ddp_training_results.csv).

## Results

Mean all-reduce latency per rank, in milliseconds:

| Payload per rank | 2 ranks | 4 ranks | 6 ranks |
|---:|---:|---:|---:|
| 1 MiB | 0.077 | 0.118 | 0.128 |
| 10 MiB | 0.379 | 0.629 | 0.670 |
| 100 MiB | 3.501 | 5.963 | 6.553 |
| 1,024 MiB (1 GiB) | 34.291 | 62.363 | 68.966 |

For these fixed per-rank payloads, collective latency rose with both message size and rank count. For example, the 1 GiB per-rank all-reduce rose from 34.29 ms with 2 ranks to 68.97 ms with 6 ranks.

The XL training step comparison was:

| Synchronization mode | Mean step | Communication or wait metric |
|---|---:|---:|
| Per-parameter all-reduce after backward | 1,067.51 ms | 451.59 ms communication |
| Flatten all gradients, then one all-reduce | 1,078.82 ms | 425.35 ms collective |
| Asynchronous per-parameter overlap | 862.02 ms | 298.70 ms wait tail after backward |

For this workload, overlapping communication with backward reduced the measured step time by about **19%** versus synchronizing after backward. The 298.70 ms is the remaining wait after backward, not the total communication time; most communication was issued while backward continued. Flattening slightly reduced the collective measurement but did not improve step time, likely because packing and copying the gradients offset that benefit. The Nsight traces for the post-backward and overlap modes are included as [`ddp_nsys_naive.nsys-rep`](../ddp_nsys_naive.nsys-rep) and [`ddp_nsys_overlap.nsys-rep`](../ddp_nsys_overlap.nsys-rep); profiler-instrumented times include overhead and are not directly comparable to the unprofiled averages.

## Conclusion and limitations

The correctness check passes, and the measurements show that asynchronous per-parameter all-reduce can improve step time by hiding part of the communication behind backward computation. Flattening every gradient into one collective was not faster in this short run. These results support the value of communication overlap for the tested XL workload, but they do not yet establish DDP training scaling across 2, 4, and 6 GPUs: only the all-reduce microbenchmark used all three world sizes, while XL training was measured on 2 GPUs.

The training benchmark has only two timed trials, and it measures a short step rather than full training throughput or convergence. Treat the result as preliminary. A stronger scaling study would repeat the XL training workload at 2, 4, and 6 ranks with more warmups and trials, then report throughput and scaling efficiency alongside step latency.

The recorded GPU measurements were collected while the six-card node exposed all devices. On a later SSH check, the instance reported no visible GPU devices (`torch.cuda.is_available() == False`, device count 0), so no new GPU run was attempted. The existing measurements and CPU/Gloo correctness test are sufficient for this writeup; reopening the GPU allocation is only necessary for the additional 4- and 6-rank XL training runs or a repeat with more trials.

## Reproduction files

- [`benchmark_ddp_communication.py`](../benchmark_ddp_communication.py): NCCL all-reduce sweep.
- [`benchmark_ddp_training.py`](../benchmark_ddp_training.py): XL training comparison for `naive`, `flat`, and `overlap` synchronization.
- [`cs336_systems/DDP.py`](../cs336_systems/DDP.py): custom DDP wrapper.
