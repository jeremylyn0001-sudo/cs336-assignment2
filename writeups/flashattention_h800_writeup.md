# FlashAttention-2 H800 Substitute Benchmark

## Status and scope

The requirements and point values below are from the [assignment handout](../cs336_assignment2_systems.pdf).

This report records the completed H800 substitute experiments for the custom Triton FlashAttention implementation, including causal-tile pruning and a D=128 backward tile sweep. The GPU was an NVIDIA H800 PCIe with 81,559 MiB of memory. This is substitute hardware: the assignment benchmarking problem specifies a single B200.

The handout's required `flash_backward` (5 points) uses regular PyTorch backward compiled with `torch.compile`; the required `flash_benchmarking` problem (5 points) compares the partially Triton implementation against dense PyTorch on one B200. The hybrid implementation and `benchmark_flashattention_required.py` are in the repository, and a small correctness smoke test passed, but the full configuration sweep has not been collected. The H800 CSVs below measure the full Triton backward path instead. That path corresponds to the optional tiled Triton backward extension in Section 4.2.3. Therefore, the H800 full-Triton substitute experiment is complete, but the required hybrid benchmark deliverable is still open.

## Experimental setup

| Item | Setting |
|---|---|
| GPU | NVIDIA H800 PCIe, 81,559 MiB |
| Driver / CUDA | 580.82.07 / 13.0 |
| Python / PyTorch / Triton | 3.12.3 / 2.12.1+cu130 / 3.7.1 |
| Batch size | 1 |
| Masking | Causal |
| Sequence lengths | 128, 256, 512, 1,024, 2,048, 4,096, 8,192, 16,384, 32,768, 65,536 |
| Head dimensions / dtypes | 16, 32, 64, 128 / BF16, FP32 |
| Timer | `triton.testing.do_bench`; random inputs generated before timing |

The full grid contains 80 shape/dtype configurations for each implementation. Each CSV stores forward, backward, and end-to-end latency in milliseconds. Four dense FP32 cases at sequence length 65,536 ran out of memory during backward and end-to-end timing; those rows are marked partial and are excluded from paired geometric means.

## Full Triton implementation versus dense PyTorch

The optimized full-Triton results are in [`flashattention_benchmark_h800_causal_pruned.csv`](../flashattention_benchmark_h800_causal_pruned.csv), with summary metadata in the corresponding JSON file. The table reports the geometric mean of dense-PyTorch latency divided by Triton latency, using fully successful paired configurations. Values above 1 mean Triton was faster.

| Region and metric | Paired configs | Dense / Triton geometric mean | Triton faster |
|---|---:|---:|---:|
| All: forward | 76 | 1.69× | 58 / 76 |
| All: backward | 76 | 0.75× | 33 / 76 |
| All: end-to-end | 76 | 0.94× | 46 / 76 |
| Sequence length ≥8,192: forward | 28 | 2.99× | 25 / 28 |
| Sequence length ≥8,192: backward | 28 | 0.93× | 17 / 28 |
| Sequence length ≥8,192: end-to-end | 28 | 1.19× | 19 / 28 |
| Sequence length ≥8,192, D≤64: end-to-end | 21 | 2.02× | 19 / 21 |

Across all paired shapes, Triton forward is faster on average, but its slower backward pulls end-to-end performance to about 6% behind dense PyTorch. For long sequences, the end-to-end average reverses to about 1.19× faster, with the clearest gains at D≤64. These latency measurements do not include peak-memory measurements; the memory advantage of avoiding an N×N score tensor is an algorithmic property, not a measured result in this experiment.

## Causal tile pruning

The causal rule permits only key positions j≤i. For a query tile, K tiles strictly to the right of its last query row are fully masked and contribute zero. For a K tile, query tiles strictly to its left are fully masked and contribute zero to dK and dV. The implementation bounds the loops to skip those regions where the Triton code generator benefits from a bounded loop.

Compared with the previous fixed-loop Triton version, the causal-pruned version has an end-to-end geometric-mean speedup of about 1.10× over all 80 Triton configurations and 1.56× over the 32 configurations with sequence length at least 8,192. D=128 keeps the static full loop because a runtime-bounded loop regressed substantially on H800. The data and comparison are in [`flashattention_benchmark_h800_causal_pruned.json`](../flashattention_benchmark_h800_causal_pruned.json).

## D=128 backward tile sweep

A separate sweep varied `BLOCK_Q`, `BLOCK_K`, warps, and stages for both dtypes. Results cover all ten sequence lengths; the short- and long-sequence sweeps are stored separately because their timer warmup/repetition settings differed. Candidate data is in [`flashattention_d128_tile_sweep_h800.csv`](../flashattention_d128_tile_sweep_h800.csv) and [`flashattention_d128_tile_sweep_h800_short.csv`](../flashattention_d128_tile_sweep_h800_short.csv); the [`JSON summary`](../flashattention_d128_tile_sweep_h800.json) records each candidate, shape, timing settings, and baseline comparison.

| Dtype | Original backward tile | Selected backward tile |
|---|---|---|
| BF16, D=128 | Q32 × K64, 4 warps, 3 stages | Q64 × K32, 8 warps, 3 stages |
| FP32, D=128 | Q32 × K32, 4 warps, 1 stage | Q32 × K64, 8 warps, 1 stage |

The selected configurations were best across the tested sequence lengths, so the runtime dispatch is based on D and dtype rather than adding a separate threshold for every sequence length. The end-to-end geometric-mean speedup over the original Triton tile configuration is 1.47× for lengths 128–4,096 and 1.56× for lengths 8,192–65,536. At N=65,536, BF16 backward decreases from about 480 ms to 250 ms and end-to-end from about 496 ms to 267 ms. FP32 backward decreases from about 763 ms to 523 ms and end-to-end from about 1,074 ms to 810 ms.

These gains come from changing the work assigned to each CTA and the degree of parallelism within each tile. BF16 benefits from more query rows per CTA with a smaller K tile; FP32 benefits from a larger K tile and more warps. Increasing a tile is not automatically faster: larger tested tiles were slower or resource-limited. The sweep selects configurations empirically rather than assuming fewer loop iterations always wins.

The D=128 optimization improves the custom Triton implementation but does not make it faster than dense PyTorch in the paired long-sequence D=128 cases. Dense PyTorch remains about 2.4× faster for BF16 and 2.7× faster for FP32 over the successful long cases; dense FP32 at N=65,536 is OOM. This result should be reported alongside the gains over the original Triton configuration.

## Correctness and limitations

- `tests/test_attention.py`: 6 passed, including both causal settings for the tested D=64 Triton cases.
- Full `pytest`: 8 passed and 6 failed. The failures are the existing FSDP and Sharded Optimizer tests, whose adapter functions are unimplemented; the FA2 and DDP tests passed.
- A supplemental D=128 FP32 causal comparison against dense PyTorch passed with a 1e-2 tolerance. BF16 compared with a same-dtype dense reference has a small mismatch above 1e-2; the same discrepancy occurs with the original tile configuration. BF16 compared with an FP32 dense reference rounded to BF16 passed at 1e-2 in a small check. The supplied official attention tests use D=64, so they do not establish strict D=128 BF16 agreement.
- Each candidate/shape in the tile sweep has one `do_bench` estimate. Treat small differences as indicative; repeat measurements before making a fine-grained claim.

## Conclusion

The H800 substitute run is complete for the full-Triton path and the D=128 tile-tuning extension. It shows clear forward gains and meaningful D=128 backward gains over the original Triton tile configuration, especially for long sequences. It does not establish that D=128 Triton is faster than dense PyTorch, and it does not replace the required hybrid Triton-forward/`torch.compile`-backward sweep on the assignment's specified B200 hardware. The hybrid benchmark script is present, but its full-grid result file is not yet available.


