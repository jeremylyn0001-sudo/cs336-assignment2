## Benchmarking

### Experimental Setup

- GPU: NVIDIA vGPU-48GB（49,140 MiB）
- Batch size: 4
- Context length: 512
- Vocabulary size: 10,000
- Trials: 10
- Precision: FP32、BF16 autocast
- Timed modes: Forward、Forward + Backward、Full training step
- Optimizer: AdamW
- RoPE theta: 10,000
- Warmups: 标准精度对照采用 5 次 warmup；另对 FP32 的 0、1、2、5 次 warmup 进行消融实验
- Reported metric: 平均耗时 ± 标准差（ms）

本次 benchmark 使用的模型超参数如下。“10B”是该档配置的简称；按当前实现计算，其实际参数量约为 12.83B。

| Model | `d_model` | `d_ff` | Layers | Attention heads | Approx. parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small | 768 | 3,072 | 12 | 12 | 128.6M |
| Medium | 1,024 | 4,096 | 24 | 16 | 423.2M |
| Large | 1,280 | 5,120 | 36 | 20 | 969.4M |
| XL | 2,560 | 10,240 | 32 | 32 | 3.41B |
| 10B | 4,608 | 12,288 | 50 | 36 | ≈12.83B |

### Standard Benchmark Results

| Model | Precision | Forward | Forward + Backward | Full step |
| --- | --- | ---: | ---: | ---: |
| Small | FP32 | 25.29 ± 0.02 | 77.57 ± 0.30 | 89.28 ± 0.37 |
| Small | BF16 | 13.20 ± 0.07 | 51.15 ± 2.59 | 62.56 ± 3.66 |
| Medium | FP32 | 78.30 ± 0.03 | 229.42 ± 0.23 | 266.36 ± 0.20 |
| Medium | BF16 | 38.72 ± 0.29 | 121.39 ± 0.22 | 158.49 ± 0.17 |
| Large | FP32 | 180.74 ± 0.62 | 515.50 ± 0.64 | 600.33 ± 0.57 |
| Large | BF16 | 81.78 ± 0.05 | 256.19 ± 1.18 | 340.99 ± 0.26 |
| XL | FP32 | 454.34 ± 1.48 | 1255.84 ± 3.38 | OOM |
| XL | BF16 | 194.39 ± 0.07 | 609.96 ± 0.66 | OOM |
| 10B | FP32 | OOM（模型加载阶段） | — | — |
| 10B | BF16 | 未单独运行；autocast 不减少 FP32 参数加载内存 | — | — |

Forward、Forward + Backward 和 Full step 的耗时均随模型规模增大而显著增长，同时 BF16 在所有可运行配置中都快于 FP32。完成 warmup 后，大多数结果的标准差很小；例外是 Small 的 BF16 Forward + Backward 和 Full step，其相对标准差分别约为 5.1% 和 5.9%。

### Warmup Analysis

| Model | Warmups | Forward | Forward + Backward | Full |
| --- | ---: | ---: | ---: | ---: |
| Small | 0 | 50.14 ± 78.64 | 109.09 ± 100.51 | 124.95 ± 115.41 |
| Small | 1 | 25.33 ± 0.06 | 77.50 ± 0.43 | 88.53 ± 0.19 |
| Small | 2 | 25.32 ± 0.06 | 77.34 ± 0.14 | 89.06 ± 0.25 |
| Small | 5 | 25.29 ± 0.02 | 77.57 ± 0.30 | 89.28 ± 0.37 |
| Medium | 0 | 104.77 ± 83.54 | 260.47 ± 99.98 | 301.66 ± 112.43 |
| Medium | 1 | 78.32 ± 0.06 | 229.04 ± 0.23 | 265.96 ± 0.29 |
| Medium | 2 | 78.36 ± 0.03 | 229.55 ± 0.86 | 266.27 ± 0.56 |
| Medium | 5 | 78.30 ± 0.03 | 229.42 ± 0.23 | 266.36 ± 0.20 |
| Large | 0 | 205.04 ± 82.07 | 542.48 ± 94.17 | 630.65 ± 103.54 |
| Large | 1 | 180.39 ± 1.09 | 513.96 ± 0.63 | 599.09 ± 0.74 |
| Large | 2 | 180.41 ± 0.05 | 514.45 ± 0.68 | 599.29 ± 0.45 |
| Large | 5 | 180.74 ± 0.62 | 515.50 ± 0.64 | 600.33 ± 0.57 |
| XL | 0 | 479.93 ± 86.52 | 1276.43 ± 86.73 | OOM |
| XL | 1 | 454.01 ± 3.00 | 1249.53 ± 5.07 | OOM |
| XL | 2 | 454.56 ± 1.93 | 1248.33 ± 6.55 | OOM |
| XL | 5 | 454.34 ± 1.48 | 1255.84 ± 3.38 | OOM |

不进行 warmup 会显著抬高平均耗时和方差。该现象来自首次运行时的 CUDA 上下文建立、内核与库的延迟加载、显存分配以及缓存或算法初始化等一次性开销。一次 warmup 后结果已基本稳定，而 2 次 warmup 并非在所有模型和计时模式下都最优。
