# Memory Profiling

## (a) Memory Snapshots

| Mode | Context | Precision | Snapshot |
| --- | ---: | --- | --- |
| Forward | 128 | FP32 | `memoryR2_xl_ctx128_fp32_xl_ctx128_forward_fp32.pickle` |
| Forward | 128 | BF16 autocast | `memoryR2_xl_ctx128_bf16_xl_ctx128_forward_bf16.pickle` |
| Forward | 2,048 | FP32 | `memoryR2_xl_ctx2048_fp32_xl_ctx2048_forward_fp32.pickle` |
| Forward | 2,048 | BF16 autocast | `memoryR2_xl_ctx2048_bf16_xl_ctx2048_forward_bf16.pickle` |
| Full step | 128 | FP32 | `memoryR2_xl_ctx128_fp32_xl_ctx128_full_fp32.pickle` |
| Full step | 128 | BF16 autocast | `memoryR2_xl_ctx128_bf16_xl_ctx128_full_bf16.pickle` |
| Block backward detail | 128 | FP32 | `memfinal_block_xl_ctx128.json` |
| Operation memory summary | 128 | FP32 | `memfinal_operation_memory.json` |

Context 2,048 的 full step 在 forward 阶段已经 OOM，因此没有生成 snapshot。`memory90_*` 文件与 context-128 FP32 的 R2 文件是旧命名副本，不作为新的独立实验计数。

## Memory-viz Screenshots

![XL context 128 FP32 forward，Detail=1](memory-artifacts/xl_ctx128_forward_fp32_detail1.png)

*Figure 1. Forward Active Memory Timeline，Detail 降至 1 后选中最大的 100 MiB allocation。该条目是 ghost block，因此没有 stack frame。*

![XL context 128 FP32 full-step Active Memory Timeline](memory-artifacts/xl_ctx128_full_fp32_timeline.png)

*Figure 2. Full-step Active Memory Timeline。相比 forward，backward 生成的 gradients 及 AdamW 状态使 active memory 持续上升至约 51 GiB。*

## (b) Peak Memory

### FP32

| Context | Forward peak | Full-step peak |
| ---: | ---: | ---: |
| 128 | 13,259.14 MiB | 65,709.68 MiB |
| 2,048 | 19,746.78 MiB | OOM |

### BF16 autocast

| Context | Forward peak | Full-step peak |
| ---: | ---: | ---: |
| 128 | 13,266.64 MiB | 65,503.45 MiB |
| 2,048 | 19,705.28 MiB | OOM |

Context 从 128 增至 2,048 后，forward peak 明显上升；full step 在 context 2,048 下尚未完成 forward 就已 OOM，因此无法生成对应 snapshot。

## (c) Does Mixed Precision Save Significant Memory?

BF16 autocast 没有显著降低本实验的峰值显存：context 128 forward 反而比 FP32 高 7.50 MiB，context 2,048 forward 只低 41.50 MiB，而 context 128 full step 也只低 206.23 MiB（约 0.31%）。Context 2,048 full step 在两种精度下仍然 OOM，因为 autocast 只降低部分算子输入和临时 activation 的精度，模型主参数、gradients 和 AdamW optimizer states 仍保留为 FP32。

## (d) Residual Tensor Size

XL 模型的 `d_model` 为 2,560；一个 residual tensor 包含 `batch × context × d_model` 个元素。

| Context | Elements | FP32 residual | BF16 residual |
| ---: | ---: | ---: | ---: |
| 128 | 4 × 128 × 2,560 = 1,310,720 | 5 MiB | 2.5 MiB |
| 2,048 | 4 × 2,048 × 2,560 = 20,971,520 | 80 MiB | 40 MiB |

官方 XL/context 2,048 FP32 参考配置中，单个 residual tensor 的主要答案是 **80 MiB**。这只是一个 residual tensor 的理论大小；TransformerBlock 可能同时保留多个 residual、归一化输入和 autograd saved tensors，因此它不等于 backward 完成一个 block 时实际释放的全部显存。

## (e) Memory-viz Maximum Allocation

将 XL/context 128/FP32 forward snapshot 加载到 Memory Viz，并把 **Detail** 降至 1 后，显示的最大 active allocation 为 **100.0 MiB（104,857,600 bytes）**。该条目是 ghost block，没有可用 stack frame：它在 `_record_memory_history()` 启用之前已经存在，或相应 allocation event 已从 trace ring buffer 中被淘汰。

100 MiB 与 XL 模型一个 FP32 MLP 权重矩阵的大小完全一致：

```text
10,240 × 2,560 × 4 bytes
= 104,857,600 bytes
= 100 MiB
```

因此，该 ghost allocation 最可能来自 SwiGLU/MLP 中形状为 `[d_ff, d_model]` 的 FP32 Linear 权重。由于模型参数在开始记录 memory history 前已经创建，Memory Viz 无法为它提供直接调用栈；这里的来源判断是根据精确大小和模型结构作出的推断。

最大且具有完整 trace 的 allocation 为 **20.0 MiB（20,971,520 bytes）**，其用户代码调用栈为：

```text
cs336_basics/Linear.py:16       in forward
cs336_basics/swiglu.py:15      in forward
cs336_basics/model.py:21       in forward
cs336_basics/model.py:52       in forward
benchmark.py:148               in forward_fn
benchmark.py:183               in main
benchmark.py:198               in <module>
```

该大小对应一个 FP32 SwiGLU projection 输出：

```text
batch × context × d_ff × 4 bytes
= 4 × 128 × 10,240 × 4 bytes
= 20,971,520 bytes
= 20 MiB
```

因此，最大可追踪的运行时 allocation 来自 `Linear.forward` 中的矩阵乘法输出，具体位于 SwiGLU 的 MLP projection。

### Cross-snapshot comparison

| Snapshot | Reserved at snapshot | Active at snapshot | Largest active block | Largest traced allocation | Traced source |
| --- | ---: | ---: | ---: | ---: | --- |
| Context 128 / Forward / FP32 | 12.97 GiB | 12.85 GiB | 100 MiB, ghost | 20 MiB | SwiGLU `Linear.forward` |
| Context 128 / Forward / BF16 | 12.97 GiB | 12.85 GiB | 100 MiB, ghost | 50 MiB | `Linear.forward` |
| Context 2,048 / Forward / FP32 | 19.63 GiB | 12.89 GiB | 100 MiB, ghost | 2 GiB | `scaled_dot_product_attention` |
| Context 2,048 / Forward / BF16 | 19.56 GiB | 12.89 GiB | 100 MiB, ghost | 2 GiB | `softmax` |
| Context 128 / Full / FP32 | 68.95 GiB | 51.33 GiB | 100 MiB | 100 MiB | no user frame |
| Context 128 / Full / BF16 | 68.79 GiB | 51.27 GiB | 100 MiB | 100 MiB | no user frame |

Context 2,048 的 2 GiB allocation 与 attention score 张量相符：

```text
batch × attention_heads × context² × 4 bytes
= 4 × 32 × 2,048² × 4 bytes
= 2 GiB
```

BF16 autocast 下该 allocation 仍为 2 GiB，说明 softmax/相关数值敏感路径仍使用 FP32。Context 从 128 增至 2,048 后，attention 的 `context²` 张量取代 MLP 输出成为最大运行时 allocation，也解释了长 context 的显存压力。

## Snapshot-derived Allocation Sources

以下是 Memory Viz trace 中按累计 allocation traffic 排名的前五个用户代码来源。它们描述“记录期间累计分配了多少字节”，不是某一时刻的峰值占用，也不能替代官方要求的 Nsight memory-operation 占比。

### XL / context 128 / forward / FP32

| Rank | Source | Cumulative allocation | Share of traced allocation traffic |
| ---: | --- | ---: | ---: |
| 1 | `Linear.py:16` — Linear matmul output | 2,099.53 MiB | 24.78% |
| 2 | `rope.py:47` — RoPE operation | 960.00 MiB | 11.33% |
| 3 | `swiglu.py:17` — SwiGLU intermediate | 640.00 MiB | 7.55% |
| 4 | `swiglu.py:18` — SwiGLU intermediate | 640.00 MiB | 7.55% |
| 5 | `nn_utils.py:17` — scaled dot-product attention | 576.00 MiB | 6.80% |

### XL / context 2,048 / forward / FP32

| Rank | Source | Cumulative allocation | Share of traced allocation traffic |
| ---: | --- | ---: | ---: |
| 1 | `nn_utils.py:10` — softmax | 128.00 GiB | 25.93% |
| 2 | `nn_utils.py:17` — scaled dot-product attention | 69.00 GiB | 13.98% |
| 3 | `nn_utils.py:21` — scaled dot-product attention | 64.13 GiB | 12.99% |
| 4 | `nn_utils.py:11` — softmax | 64.03 GiB | 12.97% |
| 5 | `nn_utils.py:18` — scaled dot-product attention | 64.00 GiB | 12.97% |

这些是整个 trace 中重复 allocation 的累计流量，所以数值可以远高于实际 GPU 容量。它们表明短 context 下 MLP/Linear 是主要分配来源，而长 context 下 `context²` attention 和 softmax 张量占据主导。

## (f) Nsight Memory Profiling

新增的 `memnsys_xl_ctx128_forward.nsys-rep` 和 `memnsys_xl_ctx128_full.nsys-rep` 使用了 `--cuda-memory-usage=true`，并同时记录 CUDA、NVTX、cuBLAS、cuDNN、OS runtime 和 PyTorch operation ranges。它们包含 GPU memory-usage track；但当前导出的 `cuda_gpu_mem_size_sum` 只汇总 CUDA memcpy/memset，并不会把 PyTorch caching allocator 内部的 tensor allocation 自动归因到五个模型 operation。

![XL context 128 FP32 full-step Nsight memory timeline](memory-artifacts/nsight_xl_ctx128_full_memory_timeline.png)

*Figure 3. Nsight Systems 2026.5.1 原生 Full-step timeline，包含 CUDA HW Memory、PyTorch operation ranges 与 NVTX ranges。*

### CUDA memory-operation summary

| Mode | Operation | Total transferred | Count | Maximum single operation |
| --- | --- | ---: | ---: | ---: |
| Forward | Host-to-Device memcpy | 13,629.860 MB | 355 | 104.858 MB |
| Forward | Device-to-Device memcpy | 805.306 MB | 96 | 8.389 MB |
| Forward | CUDA memset | 0.202 MB | 675 | 0.001 MB |
| Full step | Host-to-Device memcpy | 13,629.860 MB | 355 | 104.858 MB |
| Full step | Device-to-Device memcpy | 2,632.974 MB | 387 | 8.389 MB |
| Full step | CUDA memset | 0.540 MB | 1,350 | 0.001 MB |

该表描述 CUDA memory operations，而不是模型 tensor 的峰值归因；因此它只有三种 operation，不能人为拆成官方要求的五个模型 operation。

### TransformerBlock residual release

`memfinal_block_xl_ctx128.json` 给出了单个 TransformerBlock backward 的实测结果：

| Metric | Value |
| --- | ---: |
| Allocated before block backward | 13,357.34 MiB |
| Allocated after block backward | 13,630.16 MiB |
| Net allocator change | +272.81 MiB |
| Peak allocated during block backward | 13,655.16 MiB |
| Peak increase over entry | 297.81 MiB |
| Autograd saved tensors present | 584.23 MiB |
| Autograd saved tensors released | **584.23 MiB** |
| Saved tensors remaining after block backward | 0 |
| Reserved before / after | 13,816.00 / 13,816.00 MiB |

因此，单个 TransformerBlock backward 精确释放的 autograd saved-tensor memory 为 **584.23 MiB**。`allocated` 前后值不能直接用于计算该释放量：backward 在释放 584.23 MiB saved tensors 的同时创建了参数 gradients 和临时 buffer，所以整体 allocated memory 净增加了 272.81 MiB；CUDA allocator 的 reserved memory 则保持不变。

Full-step trace 中也存在大量 **5 MiB** 的 `free_requested` 事件，与单个 XL/context-128 FP32 residual tensor 的理论大小一致。5 MiB 代表一个 residual tensor，而 584.23 MiB 是该 TransformerBlock backward 释放的全部 autograd saved tensors，两者统计范围不同。

### Five largest allocation contributors

`memfinal_operation_memory.json` 使用 `self_cuda_memory_usage` 统计 operation 自身产生的 CUDA memory，全部 operation 合计为 **58,709,833,728 bytes（约 54.68 GiB）**。排名前五的 operation 如下：

| Rank | Operation | Self CUDA memory | Share of total |
| ---: | --- | ---: | ---: |
| 1 | `aten::mm` | 16,731.00 MiB | 29.88% |
| 2 | `aten::empty_strided` | 13,125.14 MiB | 23.44% |
| 3 | Autograd engine `run_backward` | 8,406.01 MiB | 15.01% |
| 4 | `aten::mul` | 5,662.13 MiB | 10.11% |
| 5 | `aten::div` | 3,161.13 MiB | 5.65% |

前五项合计约占 **84.10%**。其中 `aten::mm` 是线性层和 attention 矩阵乘法的主要内存贡献者；`aten::empty_strided` 负责创建具有指定 shape/stride 的输出或临时 tensor；Autograd engine 的 `run_backward` 项反映无法进一步归入单个 ATen 子操作的 backward 自身分配；`aten::mul` 和 `aten::div` 主要来自 elementwise、归一化及梯度计算。

### Gradient-memory difference

XL 模型约含 3.41B 参数。若参数 gradient 使用 FP32，则仅 gradients 理论上需要：

```text
3.41 billion parameters × 4 bytes
≈ 13.64 GB
≈ 12.70 GiB
```

Full-step snapshot 在一次记录窗口中的最大动态 live-allocation 增量为 **13,015.48 MiB（约 12.71 GiB）**，而一个参数规模的 traced allocation 为 **12,995.95 MiB（约 12.69 GiB）**。二者与 FP32 gradient 的理论值 12.70 GiB 高度一致，因此实际 gradient memory difference 可报告为 **约 12.7 GiB**；其余约 19.5 MiB 来自该时刻同时存活的临时 allocation。BF16 autocast 不会把 FP32 参数的 `.grad` storage 自动改成 BF16，因此 BF16 的 gradient memory 仍应接近同一数值。

## Conclusion

- Memory Viz 中最大的 active allocation 为 100 MiB，但它是无 stack 的 ghost block，最可能是记录开始前创建的 FP32 MLP weight。
- 最大带有效 stack 的 context-128 FP32 forward allocation 为 20 MiB，来自 SwiGLU 内的 `Linear.forward` 输出。
- Context 2,048 的最大 traced allocation 为 2 GiB，来自 attention/softmax 的 `context²` 张量。
- BF16 autocast 对峰值显存影响很小，因为参数、gradients 和 optimizer states 仍为 FP32。
- 单个 XL/context 2,048 FP32 residual tensor 为 80 MiB；这是理论值，不等于完整 TransformerBlock 的实际释放量。
- XL 的实际动态 gradient memory 增量约为 12.71 GiB，与理论估算一致。
- Full-step trace 中观察到 5 MiB 的 residual-sized free event；独立 block 记录显示该 TransformerBlock 共释放 584.23 MiB autograd saved tensors。
- `self_cuda_memory_usage` 的前五大 operation 为 `aten::mm`、`aten::empty_strided`、Autograd `run_backward`、`aten::mul` 和 `aten::div`，合计约占 84.10%。
- Nsight 报告已启用 CUDA memory usage，并附有 GPU memory timeline；单个 residual 的 5 MiB 释放由 allocation/free trace 与理论张量大小共同验证。

## Completion Status

| Part | Status |
| --- | --- |
| (a) Memory snapshots | Complete |
| (b) Peak-memory table | Complete |
| (c) Mixed-precision memory conclusion | Complete |
| (d) Residual tensor estimate | Complete |
| (e) Memory-viz maximum allocation and stack | Complete, with ghost-block limitation documented |
| (f) Nsight memory profiling | Complete, with CUDA memory-operation scope documented |
