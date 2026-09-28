# Nsight Profiling Benchmark

## 1. Experimental setup

| Item | Value |
|---|---|
| GPU | NVIDIA vGPU-48GB, 49,140 MiB |
| Profiler | NVIDIA Nsight Systems 2026.5.1 |
| Batch size | 4 |
| Vocabulary size | 10,000 |
| Precision | FP32 |
| Context lengths | 256, 512, 1,024 |
| Models | Small: 128.6M; Medium: 423.2M parameters |
| Optimizer | AdamW for full-step profiles |
| Profile procedure | 2 warmups + 1 measured execution |

Three timing domains are reported separately:

- Python benchmark time: end-to-end wall-clock time, including launch and synchronization overhead.
- NVTX range time: CPU-side duration of an annotated code range.
- Nsight CUDA kernel time: the sum of GPU kernel durations from `cuda_gpu_kern_sum`.

These metrics are related but are not interchangeable.

## 2. Raw evidence

The raw reports are stored under:

`C:\Users\jeremy.lin\nsight-artifacts\reports\`

The workspace copy is available at [nsight-artifacts/reports](C:\Users\jeremy.lin\Documents\ChatGPT\new code\nsight-artifacts\reports).

For every Small/Medium × 256/512/1,024 configuration, the directory contains one forward report, one standalone backward report, one full-step report, and the corresponding `cuda_gpu_kern_sum.csv` exports. There are 18 `.nsys-rep` files and 18 kernel-summary CSV files in total.

The standalone backward table is generated directly from those CSV files by [summarize_nsight_backward.py](C:\Users\jeremy.lin\Documents\ChatGPT\new code\summarize_nsight_backward.py).

## 3. Python forward baseline

| Model | Context 256 | Context 512 | Context 1,024 |
|---|---:|---:|---:|
| Small | 11.73 ± 1.21 ms | 24.93 ± 0.08 ms | 77.14 ± 0.06 ms |
| Medium | 32.24 ± 0.03 ms | 77.23 ± 0.08 ms | 222.46 ± 0.13 ms |

## 4. Forward kernel analysis

The following table uses `cuda_gpu_kern_sum` exports. Raw calls and time contain three executions; the per-forward estimate divides raw values by three.

| Model / context | Python forward | Dominant kernel | Dominant share | Raw calls / time | Calls / time per forward | All matmul |
|---|---:|---|---:|---:|---:|---:|
| Small / 256 | 11.73 ± 1.21 ms | `ampere_sgemm_128x64_tn` | 69.6% | 255 / 21.85 ms | ≈85 / 7.28 ms | 74.2% |
| Small / 512 | 24.93 ± 0.08 ms | `ampere_sgemm_128x64_tn` | 55.6% | 252 / 39.44 ms | ≈84 / 13.15 ms | 65.6% |
| Small / 1,024 | 77.14 ± 0.06 ms | `ampere_sgemm_128x64_tn` | 34.6% | 252 / 74.20 ms | ≈84 / 24.73 ms | 45.7% |
| Medium / 256 | 32.24 ± 0.03 ms | `ampere_sgemm_128x64_tn` | 74.9% | 507 / 69.11 ms | ≈169 / 23.04 ms | 78.3% |
| Medium / 512 | 77.23 ± 0.08 ms | `ampere_sgemm_128x64_tn` | 56.4% | 504 / 123.00 ms | ≈168 / 41.00 ms | 63.2% |
| Medium / 1,024 | 222.46 ± 0.13 ms | `ampere_sgemm_128x64_tn` | 40.6% | 504 / 252.18 ms | ≈168 / 84.06 ms | 49.9% |

The dominant forward kernel is the same across all six configurations. As context grows, attention-related reduction, masking, and elementwise work occupies a larger fraction of GPU time, so the dominant SGEMM share decreases.

## 5. Standalone backward analysis

This table uses the six standalone backward reports, rather than inferring backward values from full-step reports. The CSV contains two warmups and one measured execution; “per-run GPU kernel time” divides the raw accumulated kernel time by three.

| Model / context | Raw GPU kernel time | Per-run GPU kernel time | Dominant backward kernel | Calls/run | Share |
|---|---:|---:|---|---:|---:|
| Small / 256 | 95.31 ms | 31.77 ms | `ampere_sgemm_128x64_tn` | 85.0 | 23.20% |
| Small / 512 | 218.41 ms | 72.80 ms | `ampere_sgemm_128x64_tn` | 84.0 | 18.00% |
| Small / 1,024 | 651.41 ms | 217.14 ms | `ampere_sgemm_128x64_tn` | 84.0 | 11.60% |
| Medium / 256 | 249.08 ms | 83.03 ms | `ampere_sgemm_128x64_tn` | 169.0 | 27.70% |
| Medium / 512 | 646.13 ms | 215.38 ms | `ampere_sgemm_128x64_tn` | 168.0 | 19.20% |
| Medium / 1,024 | 1,847.11 ms | 615.70 ms | `ampere_sgemm_128x64_tn` | 168.0 | 13.80% |

The displayed kernel name may be reused across different matrix shapes and call sites. Kernel-name equality alone does not prove that forward and backward perform the same semantic operation; the profile mode and NVTX phase provide the attribution.

## 6. Full-step matmul share

| Model / context | Forward: all matmul | Full step: all matmul | Change |
|---|---:|---:|---:|
| Small / 256 | 74.2% | 53.2% | −21.0 percentage points |
| Small / 512 | 65.6% | 51.3% | −14.3 percentage points |
| Small / 1,024 | 45.7% | 39.1% | −6.6 percentage points |
| Medium / 256 | 78.3% | 51.0% | −27.3 percentage points |
| Medium / 512 | 63.2% | 48.7% | −14.5 percentage points |
| Medium / 1,024 | 49.9% | 42.0% | −7.9 percentage points |

Full-step matmul share is lower in every configuration because backward and AdamW introduce gradient, reduction, elementwise, and optimizer kernels.

## 7. Small/context-512 attention timing

The following values are from the Small/context-512 forward kernel summary. QKᵀ and AV use the same kernel family in the summary, so they are reported together.

| Attention component | Time per forward | Calls per forward | GPU-time share |
|---|---:|---:|---:|
| QKᵀ + AV matmul | 1.66 ms | 24 | 7.0% |
| Exponential | 0.91 ms | 12 | 3.8% |
| Max reduction | 0.25 ms | 12 | 1.0% |
| Sum reduction | 0.19 ms | 12 | 0.8% |
| Masked fill | 0.50 ms | 12 | 2.1% |
| Softmax core | **1.35 ms** | — | 5.6% |
| Softmax-related: core + masked fill | **1.85 ms** | — | **7.7%** |

The softmax-related value is consistently reported as 1.85 ms and 7.7%. The previously used 1.94 ms and 8.2% values are removed because they used a different attribution denominator.

The division kernel is also used by RMSNorm and cannot be cleanly attributed to softmax from a kernel-only summary. Therefore, 1.35 ms is the directly attributable softmax-core estimate, while 1.85 ms includes the separately identified masked-fill component.

## 8. FLOPs interpretation

Let `h` be the number of attention heads, `d_model` the hidden size, `d_head = d_model / h` the dimension of one head, and `L` the context length.

For one attention-score element, QKᵀ and AV together require approximately:

```text
4 × d_head FLOPs
```

The two attention matmuls have total complexity `O(B × h × L² × d_head) = O(B × L² × d_model)`. Softmax has complexity `O(B × h × L²)`, but consists mainly of elementwise operations and reductions.

For Small, `d_model = 768`, `h = 12`, and `d_head = 64`. The relevant per-score comparison is therefore `4 × 64`; `d_model / h` should not be introduced as a second head-dimension derivation.

## 9. Screenshots and reproducibility

The existing summary images are statistical evidence:

![Small context 512 forward kernel summary](nsight-artifacts/small_ctx512_forward_kernel_summary.png)

![Small context 512 full-step summary](nsight-artifacts/small_ctx512_full_timeline_summary.png)

They are exported summary images, not native Nsight GUI timeline screenshots. To capture a native timeline, open this absolute path in Nsight Systems:

```text
C:\Users\jeremy.lin\nsight-artifacts\reports\nsight26_small_ctx512_full.nsys-rep
```

Use `File -> Open`; do not open the report through the relative path `code\nsight-artifacts\...`.

## 10. Answers to required questions

| Question | Answer |
|---|---|
| Does Nsight forward time match the Python benchmark? | They are close in scale but not equal. Python is end-to-end wall-clock time; `cuda_gpu_kern_sum` is summed GPU kernel duration and excludes CPU launch and synchronization effects. |
| Which kernel is most time-consuming? | `ampere_sgemm_128x64_tn` is the dominant forward kernel in all six configurations and is also the largest standalone-backward kernel in the supplied backward summaries. |
| Is forward the same as backward? | A displayed kernel name may be reused, but phase attribution comes from the standalone profile/NVTX range. Kernel-name equality alone does not establish identical semantics. |
| What important non-matmul kernels appear? | Exponential, max/sum reductions, masked fill, elementwise operations, SiLU, and optimizer kernels. |
| How does full-step matmul share change? | It decreases in all six configurations because backward and AdamW add substantial non-matmul work. |
| How do softmax and matmul compare? | Small/context-512 QKᵀ+AV is about 1.66 ms; softmax core is about 1.35 ms; softmax plus masked fill is 1.85 ms or 7.7% under the unified attribution denominator. |

## 11. Measurement caveats

- A summed GPU kernel duration can exceed a CPU NVTX range when kernels overlap across streams.
- The backward table reports kernel-time metrics, not memory-allocation metrics.
- Nsight Systems CUDA memory-operation reports do not automatically attribute PyTorch caching-allocator tensor allocations to model operations.
- The raw `.nsys-rep` and `cuda_gpu_kern_sum.csv` files are the authoritative evidence for the quantitative tables; summary images are visualization aids.
