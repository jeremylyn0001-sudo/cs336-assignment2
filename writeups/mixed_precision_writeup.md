# Mixed Precision Benchmark and Analysis

## Experimental Setup

- GPU: NVIDIA vGPU-48GB（49,140 MiB）
- Driver: 580.76.05
- PyTorch: 2.11.0+cu130
- Nsight Systems: 2026.5.1
- Batch size: 4
- Context length: 512
- Vocabulary size: 10,000
- Trials: 10
- Standard benchmark warmups: 5

## FP32 and BF16 Benchmark

单位为毫秒，结果格式为 `mean ± std`。

| Model | Precision | Forward | Forward + Backward | Full step |
| --- | --- | ---: | ---: | ---: |
| Small | FP32 | 25.29 ± 0.02 | 77.57 ± 0.30 | 89.28 ± 0.37 |
| Small | BF16 | 14.19 ± 0.14 | 52.09 ± 3.57 | 62.56 ± 3.66 |
| Medium | FP32 | 78.30 ± 0.03 | 229.42 ± 0.23 | 266.36 ± 0.20 |
| Medium | BF16 | 38.26 ± 0.50 | 119.59 ± 0.31 | 158.49 ± 0.17 |
| Large | FP32 | 180.74 ± 0.62 | 515.50 ± 0.64 | 600.33 ± 0.57 |
| Large | BF16 | 80.54 ± 0.06 | 252.39 ± 0.09 | 340.99 ± 0.26 |
| XL | FP32 | 454.34 ± 1.48 | 1255.84 ± 3.38 | OOM |
| XL | BF16 | 193.53 ± 0.14 | 607.64 ± 0.86 | OOM |
| 10B | FP32 | 模型加载阶段 OOM | — | — |

BF16 在全部可运行配置中都快于 FP32。Forward 加速约为 1.78×–2.35×，Forward + Backward 加速约为 1.49×–2.07×，Full step 加速约为 1.43×–1.76×；模型越大，低精度矩阵乘法通常越能充分利用 GPU 的 Tensor Core 吞吐量。Full step 的加速较低，因为优化器更新、归约和部分数值敏感操作仍使用 FP32，并不会随矩阵乘法一起等比例加速。

## Warmup Analysis

Small / FP32：

| Warmups | Forward | Forward + Backward | Full step |
| ---: | ---: | ---: | ---: |
| 0 | 50.14 ± 78.64 | 109.09 ± 100.51 | 124.95 ± 115.41 |
| 1 | 25.33 ± 0.06 | 77.50 ± 0.43 | 88.53 ± 0.19 |
| 2 | 25.32 ± 0.06 | 77.34 ± 0.14 | 89.06 ± 0.25 |
| 5 | 25.29 ± 0.02 | 77.57 ± 0.30 | 89.28 ± 0.37 |

0 warmup 会把 CUDA 上下文建立、kernel/库延迟加载、显存分配和缓存初始化等一次性开销混入正式计时，从而显著抬高均值和方差。1 次 warmup 后结果已经基本稳定，2 次 warmup 并非在所有计时模式下都最优。

## Nsight Forward Kernel Analysis

全部六组 forward profile 的主 kernel 均为 `ampere_sgemm_128x64_tn`。每个 profile 包含 2 次 warmup 和 1 次正式执行，下表的调用次数已经除以约 3，表示单次 forward。

| Model | Context | Dominant-kernel share | Calls per forward | All-matmul share |
| --- | ---: | ---: | ---: | ---: |
| Small | 256 | 69.6% | ≈85 | 74.2% |
| Small | 512 | 55.6% | ≈84 | 65.6% |
| Small | 1,024 | 34.6% | ≈84 | 45.7% |
| Medium | 256 | 74.9% | ≈169 | 78.3% |
| Medium | 512 | 56.4% | ≈168 | 63.2% |
| Medium | 1,024 | 40.6% | ≈168 | 49.9% |

在 Small/context 512 中，全部 matmul kernel 约占 GPU 时间的 65.6%，softmax 相关的 `exp`、reduce 和 masked-fill kernel 约占 8.2%。随着 context 增长，主 SGEMM 和全部 matmul 的占比下降，说明 softmax、归约、mask 等随序列长度增长的开销变得更加显著。

## Nsight Full-step Analysis

| Model | Context 256 | Context 512 | Context 1024 |
| --- | ---: | ---: | ---: |
| Small | 53.2% | 51.3% | 39.1% |
| Medium | 51.0% | 48.7% | 42.0% |

Full step 中 matmul 的总占比低于 forward-only，因为 backward 和 AdamW optimizer step 引入了大量梯度、elementwise、reduce 和参数更新 kernel。

context 512 的 NVTX 阶段中位时长如下。NVTX 是 CPU 侧 range 时间，用于分析阶段组成，不等同于 GPU kernel 累计执行时间。

| Model | Forward | Backward | Optimizer |
| --- | ---: | ---: | ---: |
| Small | 31.24 ms | 38.36 ms | 3.01 ms |
| Medium | 39.96 ms | 144.72 ms | 9.97 ms |

## ToyModel FP16 Autocast

ToyModel 的计算路径为：

```text
input → Linear(fc1) → ReLU → LayerNorm → Linear(fc2) → logits → loss → backward
```

实际 dtype 记录如下：

| Value | dtype |
| --- | --- |
| Model parameters | FP32 |
| `fc1` output | FP16 |
| ReLU output | FP16 |
| LayerNorm output | FP32 |
| `fc2` output / logits | FP16 |
| Loss | FP32 |
| All parameter gradients | FP32 |

Autocast 只为每个算子选择合适的计算 dtype，并不会永久改变模型参数的存储 dtype。因此 Linear 使用 FP16 以提高吞吐量，LayerNorm 和 loss 回到 FP32 以保证数值稳定性，而 backward 最终累积到 FP32 参数上的 gradient 也保持 FP32。

## FP16 and BF16

FP16（IEEE 754 half precision）和 BF16（bfloat16）都使用 16 bit，但它们把位数分配给指数和有效数字的方式不同。

| Format | Sign | Exponent | Fraction | Approximate maximum | Main characteristic |
| --- | ---: | ---: | ---: | ---: | --- |
| FP16 | 1 bit | 5 bits | 10 bits | 65,504 | 有效数字更多，但动态范围较小 |
| BF16 | 1 bit | 8 bits | 7 bits | ≈3.39 × 10³⁸ | 动态范围接近 FP32，但有效数字更少 |
| FP32 | 1 bit | 8 bits | 23 bits | ≈3.40 × 10³⁸ | 动态范围和精度都更高，但存储与计算成本更大 |

FP16 大约提供 3–4 位十进制有效数字，但较小的指数范围使大值容易 overflow、小值容易 underflow，因此 FP16 训练经常需要 loss scaling。BF16 保留了与 FP32 相同的 8 位指数，通常不需要像 FP16 那样依赖 loss scaling，但它只有 7 位 fraction，局部舍入误差实际上可能比 FP16 更大。

## LayerNorm Precision Analysis

LayerNorm 中最敏感的计算包括：

1. 对隐藏维度求均值。大量元素的 reduction 会累积舍入误差。
2. 计算 `x - mean`。当两个相近数相减时会发生消减误差，使有效数字丢失。
3. 计算并累加平方偏差。FP16 的小动态范围可能造成平方值 overflow 或较小偏差 underflow。
4. 计算 `1 / sqrt(variance + eps)`。当方差较小时，方差和 epsilon 的误差会被倒数平方根放大。
5. 执行归一化及 affine 变换。前面均值和方差中的误差会传播到整层输出和后续梯度。

LayerNorm 通常需要更高精度，是因为它同时包含 reduction、相近数相减和倒数平方根，这些操作比矩阵乘法更容易放大低精度误差。若均值或方差估计不准，整个隐藏向量都会使用错误的中心和尺度进行归一化；误差还会沿网络层数累积。因此常见 mixed-precision 实现会让输入和矩阵乘法保持低精度，但使用 FP32 计算或至少累加 LayerNorm 的均值与方差，并以 FP32 执行关键归一化步骤。

使用 BF16 时仍建议让 LayerNorm 的 reduction 和归一化关键步骤使用 FP32。BF16 的 8 位指数显著降低了 FP16 常见的 overflow/underflow 风险，通常也减少了对 loss scaling 的需求；但 BF16 只有 7 位 fraction，均值、方差累加以及 `x - mean` 的舍入与消减误差仍然存在。因此，BF16 改善的是动态范围，不是归一化统计量所需的有效精度；将 LayerNorm 保持为 FP32 仍是稳健且常见的策略。

本次 ToyModel 的实际结果与这一策略一致：Linear、ReLU 和 logits 使用 FP16，而 LayerNorm 输出、loss、参数及 gradient 保持 FP32。这说明 autocast 在高吞吐算子上采用低精度，同时为数值敏感操作保留更高精度。
