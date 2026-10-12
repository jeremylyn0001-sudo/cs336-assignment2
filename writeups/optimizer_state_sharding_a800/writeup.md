# A2 Optimizer State Sharding 实验记录

## 官方要求核对

依据 [CS336 Spring 2026 Assignment 2 handout](https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf) 第 6 节：

- `optimizer_state_sharding`：实现包装任意 `torch.optim.Optimizer` 的分片优化器，支持参数组、`step(closure, **kwargs)`，并在本地更新后同步参数；通过 `tests/test_sharded_optimizer.py`。
- `optimizer_state_sharding_accounting`：单节点、双 GPU、XL 模型；对比分片和普通优化器的显存与每步时间。显存需要在模型初始化后、optimizer step 前和 step 后采样，并解释参数、梯度和 optimizer state 的显存组成；另需简述与 ZeRO Stage 1 的差别。

本次没有使用 B200。该 handout 的 optimizer-state-sharding accounting 配置指定 2 张 GPU 和 XL 模型；本次用同机 2 张 A800-SXM4-80GB 完成。

## 实现和正确性

- 官方 `tests/test_sharded_optimizer.py` 连续运行 5 次，每次 2 个用例（`ToyModel`、`ToyModelWithTiedWeights`），全部通过。
- Handout 明确要求运行期间可以调用 `add_param_group`。额外增加了一个渐进解冻回归测试：先只优化第一层，训练两步后加入第二层参数组，再比较普通 AdamW 和分片优化器的最终参数；测试通过。
- 分片优化器测试采用 2 个进程、Gloo process group；GPU 性能和显存实验另用 NCCL。

## 双卡 XL 配置

| 项目 | 配置 |
|---|---|
| GPU | 2 × NVIDIA A800-SXM4-80GB，单节点 |
| PyTorch / CUDA | 2.12.1+cu130 / 13.0 |
| 模型 | XL：`d_model=2560`、`d_ff=10240`、32 层、32 heads、词表 10,000 |
| 精度 | FP32，TF32 关闭 |
| Batch / 序列长度 | global batch 4、每 rank batch 2、sequence length 512 |
| 优化器 | 两种模式都用 AdamW 默认参数 |
| step 时间统计 | 2 次 warmup，5 次测量；报告两 rank 中较慢 rank 的平均值 |
| 数据 | 固定随机 token / target，用于同形状、可重复的显存和 step 时间对照 |

环境使用预装 PyTorch 2.12.1+cu130；仓库 `pyproject.toml` 声明的是 `torch~=2.11.0`。本次结果是在同一环境中做的相对对照，需在正式提交时注明此版本差异。

## 峰值显存

以下为每个 rank 的 `torch.cuda.max_memory_allocated()`，MiB。模型初始化和前反向阶段的两种模式基本相同；AdamW 第一次 `step()` 初始化状态时，分片模式的峰值更低。

| 配置 | 模型初始化后 | optimizer step 前（前反向峰值） | optimizer step 阶段峰值 |
|---|---:|---:|---:|
| DDP + 普通 AdamW | 13,134.7 | 26,469.8 | 65,392.4 |
| DDP + SharedOptimizer | 13,149.9 | 26,468.8 | 45,704.5 |

optimizer step 后的峰值降低 **19,687.9 MiB（约 19.23 GiB，30.1%）**。step 后当前已分配显存分别为 52,267.8 MiB 和 39,183.2 MiB。

每个 rank 上模型参数约 **12.69 GiB**，反向后的梯度约 **12.69 GiB**，两者在分片模式下仍然完整复制。AdamW 的两个状态张量从普通模式约 **25.38 GiB/rank** 降至分片模式约 **12.69 GiB/rank**，接近按 2 个 rank 均分；这符合“只切 optimizer state”的预期。step 阶段峰值还包含 optimizer 更新时的临时分配，因此高于 step 后的当前显存。

## 每步时间

| 配置 | 较慢 rank 平均 step 时间 |
|---|---:|
| DDP + 普通 AdamW | 1,537.98 ms |
| DDP + SharedOptimizer | 1,546.03 ms |

本次 5 次测量中，分片版约慢 **0.52%**，step 时间基本接近。结果只代表这台 A800 上的 XL、FP32、global batch 4 配置；不要外推到 B200 或更大规模集群。

## 与 ZeRO Stage 1 的关系

两者都只分片 optimizer state；每个 rank 仍持有完整参数和完整梯度，所以参数与梯度显存没有按 world size 缩小。本实现先沿用 DDP 同步完整梯度，再由负责该参数的 rank 更新参数并逐参数广播更新结果，因而在梯度同步之外增加了参数广播，也产生较多小 collective。实际 ZeRO Stage 1 实现会更积极地使用 bucket、reduce-scatter / all-gather 等通信安排来控制通信量和延迟；本作业实现是教学用简化方案。

## 产物

- `xl_fp32.json`：逐 rank 的原始显存、optimizer state 字节数和 step 时间数据。
- `benchmark_sharded_optimizer.py`：本次使用的双卡显存 / 时间采集脚本。
- `test_sharded_optimizer_add_param_group.py`：渐进解冻场景的额外回归测试。
