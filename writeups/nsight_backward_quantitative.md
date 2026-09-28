# Nsight backward kernel quantitative table

Source: `cuda_gpu_kern_sum` exported from the six raw backward `.nsys-rep` reports.
The total is the sum of GPU kernel durations; it is not CPU wall-clock time.

The CSV contains 3 executions; per-run values divide the raw totals by 3.

| Model / context | Raw GPU kernel time | Per-run GPU kernel time | Dominant backward kernel | Calls/run | Share |
| --- | ---: | ---: | --- | ---: | ---: |
| Medium / 256 | 249.08 ms | 83.03 ms | `ampere_sgemm_128x64_tn` | 169.0 | 27.70% |
| Medium / 512 | 646.13 ms | 215.38 ms | `ampere_sgemm_128x64_tn` | 168.0 | 19.20% |
| Medium / 1,024 | 1847.11 ms | 615.70 ms | `ampere_sgemm_128x64_tn` | 168.0 | 13.80% |
| Small / 256 | 95.31 ms | 31.77 ms | `ampere_sgemm_128x64_tn` | 85.0 | 23.20% |
| Small / 512 | 218.41 ms | 72.80 ms | `ampere_sgemm_128x64_tn` | 84.0 | 18.00% |
| Small / 1,024 | 651.41 ms | 217.14 ms | `ampere_sgemm_128x64_tn` | 84.0 | 11.60% |
