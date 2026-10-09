# FlashAttention-2 H800 Substitute Benchmark

## Status and scope

The H800 substitute benchmark is now complete for the required hybrid path and the optional full-Triton backward experiments. The assignment specifies one B200; H800 results are a hardware substitute and must not be presented as B200 measurements. Requirements and point values are from the [assignment handout](../cs336_assignment2_systems.pdf).

The required `flash_backward` problem (5 points) uses PyTorch and `torch.compile`; `flash_benchmarking` (5 points) compares Triton forward plus that compiled PyTorch backward against dense PyTorch. The full H800 hybrid grid is included below. The full Triton backward and D=128 tile sweep are additional experiments for the optional Section 4.2.3 path.

## Experimental setup

| Item | Setting |
|---|---|
| GPU | NVIDIA H800 PCIe, 81,559 MiB |
| Driver / CUDA | 580.82.07 / 13.0 |
| Python / PyTorch / Triton | 3.12.3 / 2.12.1+cu130 / 3.7.1 |
| Batch size | 1 |
| Masking | Causal |
| Sequence lengths | 128 through 65,536, powers of two |
| Head dimensions / dtypes | 16, 32, 64, 128 / BF16, FP32 |
| Timer | `triton.testing.do_bench`, 25 ms warmup and 100 ms repetition |

Random inputs were generated before timing. The hybrid path uses Triton for forward and a compiled, blockwise PyTorch recomputation for backward. BF16 tensors are cast to FP32 inside the hybrid wrapper for the compiled backward, then the gradients are cast back to the input dtype. The standalone pure PyTorch implementation and dense baseline are unchanged.

## Required hybrid path: Triton forward + compiled PyTorch backward

The full data is in [`flashattention_benchmark_h800_compiled_backward.csv`](../flashattention_benchmark_h800_compiled_backward.csv), with summary metadata in the accompanying JSON file. The table reports the geometric mean of dense-PyTorch latency divided by hybrid latency, using fully successful pairs; values above 1 mean the hybrid path was faster.

| Region and metric | Paired configs | Dense / hybrid geometric mean | Hybrid faster |
|---|---:|---:|---:|
| All: forward | 76 | 1.66× | 58 / 76 |
| All: backward | 76 | 0.41× | 0 / 76 |
| All: end-to-end | 76 | 0.58× | 2 / 76 |
| Sequence length ≥8,192: forward | 28 | 3.00× | 25 / 28 |
| Sequence length ≥8,192: backward | 28 | 0.42× | 0 / 28 |
| Sequence length ≥8,192: end-to-end | 28 | 0.56× | 2 / 28 |
| Sequence length ≥8,192, D≤64: end-to-end | 21 | 0.63× | 2 / 21 |

The Triton forward is substantially faster on long inputs, but the compiled PyTorch backward is the bottleneck: its geometric-mean latency is about 2.44× dense PyTorch across all successful pairs. As a result, the end-to-end hybrid path is about 1.71× slower overall and 1.77× slower at sequence lengths ≥8,192. The H800 experiment therefore completes the required comparison grid as a substitute run, but it does not show an end-to-end speedup over dense PyTorch.

Four dense FP32 cases at N=65,536 (D=16, 32, 64, 128) ran out of memory during backward and end-to-end timing. All 80 hybrid rows completed; the CSV marks the four dense rows as partial. No peak-memory measurements were collected.

## Full Triton backward and causal-tile pruning

The separate full-Triton forward/backward comparison is in [`flashattention_benchmark_h800_causal_pruned.csv`](../flashattention_benchmark_h800_causal_pruned.csv), with metadata and aggregate statistics in its JSON summary. It is the optional tiled-Triton backward path, not the required hybrid path above.

Compared with the earlier fixed-loop Triton version, causal pruning improves the full Triton implementation's end-to-end geometric mean by about 1.10× across all 80 Triton configurations and 1.56× for the 32 configurations with N≥8,192. For causal attention, Q tile i can only attend to K positions j≤i; whole K tiles beyond that boundary have zero probability and gradient contribution. Skipping them reduces redundant matrix products and memory loads.

The pruning is enabled for D<128. Dynamic loop bounds regressed on H800 for D=128, so that dimension retains the static full loop and is tuned separately below.

## D=128 backward tile sweep

The backward-only sweep varies `BLOCK_Q`, `BLOCK_K`, warps, and stages. Results cover all ten sequence lengths and both dtypes; short and long sweeps are stored separately because their warmup/repetition settings differ. See [`flashattention_d128_tile_sweep_h800.csv`](../flashattention_d128_tile_sweep_h800.csv), [`flashattention_d128_tile_sweep_h800_short.csv`](../flashattention_d128_tile_sweep_h800_short.csv), and the [summary JSON](../flashattention_d128_tile_sweep_h800.json).

| Dtype | Original tile | Selected tile |
|---|---|---|
| BF16 | Q32 × K64, 4 warps, 3 stages | Q64 × K32, 8 warps, 3 stages |
| FP32 | Q32 × K32, 4 warps, 1 stage | Q32 × K64, 8 warps, 1 stage |

The selected configurations were best across the measured sequence lengths, so dispatch depends on D and dtype rather than a separate sequence-length threshold. Relative to the original Triton backward tile, end-to-end latency improves geometrically by 1.47× for N=128–4,096 and 1.56× for N=8,192–65,536. At N=65,536, BF16 backward falls from about 480 ms to 250 ms and end-to-end from 496 ms to 267 ms. FP32 backward falls from about 763 ms to 523 ms and end-to-end from 1,074 ms to 810 ms.

These changes improve the custom Triton D=128 path, but it remains slower than dense PyTorch end-to-end on the successful long D=128 comparisons. The tile results show why tuning must measure combinations: more warps and a different Q/K tile balance helped, while some larger K tiles were much slower or resource-limited.

## Correctness and limitations

- `tests/test_attention.py`: 6 passed. The supplied attention tests use D=64; supplemental D=128 hybrid checks passed against dense PyTorch at 1e-2 for BF16 and FP32.
- Full `pytest`: 8 passed and 6 failed. The failures are the existing FSDP and Sharded Optimizer tests, whose adapter implementations are not complete; FA2 and DDP tests passed.
- All timing rows are single `do_bench` estimates. Small differences should be treated as indicative rather than statistically conclusive.
- H800 is not B200. These are substitute-hardware results, and no peak-memory measurement was collected.

## Conclusion

The H800 hybrid experiment is complete as a substitute-hardware benchmark: it covers the required batch size, causal setting, shape/dtype grid, and forward/backward/end-to-end timings. It shows that Triton forward is faster on most shapes, but the `torch.compile` backward dominates and makes end-to-end slower than dense PyTorch on most configurations. The optional full-Triton backward and D=128 tile sweep provide additional data and tuning evidence. The data are now suitable for a writeup, provided the H800/B200 distinction and the backward bottleneck remain explicit.
