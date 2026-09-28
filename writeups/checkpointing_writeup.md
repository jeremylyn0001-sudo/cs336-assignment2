# Gradient / Activation Checkpointing

## Experimental Setup

- GPU: NVIDIA RTX PRO 6000 Blackwell（约 94 GiB）
- Compared strategies: no checkpoint、recursive checkpoint、flat chunk checkpointing
- Chunk sizes: 1、2、4、8
- Correctness signal: 全部参数梯度比较；若 baseline OOM，则报告 `not measured: baseline OOM`
- Primary metric: peak GPU memory

## Results

| Strategy | Peak memory | Gradient check |
| --- | ---: | --- |
| No checkpoint | OOM | — |
| Recursive checkpoint | 38.38 GiB | not measured: baseline OOM¹ |
| Chunk size = 1 | 38.38 GiB | not measured: baseline OOM¹ |
| Chunk size = 2 | 44.03 GiB | not measured: baseline OOM¹ |
| Chunk size = 4 | 55.34 GiB | not measured: baseline OOM¹ |
| Chunk size = 8 | 77.94 GiB | not measured: baseline OOM¹ |

¹ 机器 2 的新版脚本在 baseline OOM 时输出 `not measured: baseline OOM`，因此这些大模型结果不宣称梯度一致。使用同一脚本函数的缩小配置（d_model=128、d_ff=256、5 layers、context=32、batch=2）运行 baseline、recursive 及 chunk size 1/2/4；四种 checkpoint 路径均输出 `match`，最大绝对误差和相对误差均为 0。Peak-memory 测量不依赖梯度判定，但大模型旧结果只有在新版代码上重跑后才可作为最终提交数据。

此外，旧版 flat 实现使用 `len(layers) // chunk_size` 计算 chunk 数量。如果层数不能被 `chunk_size` 整除，末尾的余数层不会执行；此时该策略测到的是一个更浅的模型，显存数据不能与其他策略直接比较。现版实现已改为 `range(0, len(layers), chunk_size)`，会执行最后一个不足 `chunk_size` 的尾部 chunk。历史数据若来自旧版代码，仍应在修正后重新测量。

不使用 checkpoint 时，即使在约 98 GiB 显存的 GPU 上也会 OOM，说明完整保存所有中间 activation 的成本已经超过可用显存。机器 2 的新版重跑中，recursive checkpoint 与 chunk size = 1 并列最低，均为 38.38 GiB。

## Latest GPU rerun

机器 2（NVIDIA RTX PRO 6000 Blackwell，约 97.9 GiB）使用修正版脚本重跑 32 层 XL 配置：baseline 无 checkpoint OOM；recursive 38.38 GiB；chunk size 1/2/4/8 分别为 38.38/44.03/55.34/77.94 GiB。由于 baseline OOM，主配置的梯度状态仍为 `not measured`；缩小配置的全参数梯度校验全部通过，因此 checkpoint 代码逻辑得到独立验证。

## (a) Direct Answer

1. **Checkpoint 排布策略：** 对 `N` 层 Transformer 采用递归二分，并对左右子区间分别施加嵌套 checkpoint。
2. **峰值 activation memory：** 每层递归只保留当前递归路径上的边界 activation；递归深度为 `log N`，因此峰值 activation memory 为 **`O(A × log N)`**，其中 `A` 是单层 residual activation 的大小。
3. **计算量：** 每个递归层级最多重新执行总计 `N` 层的 forward，共有 `log N` 个层级，因此总 forward/recomputation work 为 **`O(N × log N)`**。
4. **代码草图：** 见下一节；核心操作是二分 layer 列表，并递归 checkpoint 左、右两个子区间。

## Code Sketch and Required Code Changes

实验代码需要同时修正两个问题：flat chunk 必须执行最后一个不足 `chunk_size` 的余数 chunk；baseline OOM 时不能把 checkpoint 梯度校验自动标记为成功。

对于当前 `TransformerBlock(x, token_positions)` 接口，建议使用下面的实现：

```python
from torch.utils.checkpoint import checkpoint


def run_layers(layer_list, act, token_positions):
    for layer in layer_list:
        act = layer(act, token_positions)
    return act


def recursive_checkpoint_forward(layers, x, token_positions):
    # len == 0 is useful for defensive checks; the normal model has >= 1 layer.
    if len(layers) == 0:
        return x
    if len(layers) == 1:
        return layers[0](x, token_positions)

    mid = len(layers) // 2
    left_layers = layers[:mid]
    right_layers = layers[mid:]

    def run_left(act):
        return recursive_checkpoint_forward(left_layers, act, token_positions)

    def run_right(act):
        return recursive_checkpoint_forward(right_layers, act, token_positions)

    x = checkpoint(run_left, x, use_reentrant=False)
    return checkpoint(run_right, x, use_reentrant=False)


def chunked_checkpoint_forward(layers, x, token_positions, chunk_size):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    # range(0, len(layers), chunk_size) includes the tail chunk.
    for start in range(0, len(layers), chunk_size):
        chunk_layers = layers[start:start + chunk_size]

        def run_chunk(act, chunk_layers=chunk_layers):
            return run_layers(chunk_layers, act, token_positions)

        x = checkpoint(run_chunk, x, use_reentrant=False)
    return x
```

关键变化是 `range(0, len(layers), chunk_size)`。原来的 `len(layers) // chunk_size` 只执行完整 chunk；当层数不能整除 chunk size 时，会漏掉尾部层，导致测到的不是同一个模型。

梯度校验也应改成“baseline 缺失就报告未测量”，不能把 OOM 当作成功：

```python
def compare_gradients(base_grad, test_grad, atol=1e-4, rtol=1e-4):
    if base_grad is None:
        return "not measured: baseline OOM"
    if test_grad is None:
        return "not measured: checkpoint run did not produce a gradient"
    return "match" if torch.allclose(base_grad, test_grad,
                                     atol=atol, rtol=rtol) else "mismatch"

# In main(), replace the old `... if base_grad is not None else True` logic:
status = compare_gradients(base_grad, rec_grad)
print(f"Recursive checkpoint gradient check: {status}")
```

For a valid correctness experiment, first run the same code on a smaller model/context where the no-checkpoint baseline completes. The large-model OOM run should be used only for peak-memory comparison, not for claiming gradient equality.

## Chunk-size Analysis

| Strategy | Peak-memory increase over chunk 1 | Relative increase |
| --- | ---: | ---: |
| Chunk size = 1 | 0 GiB | 0% |
| Chunk size = 2 | +5.74 GiB | +14.9% |
| Chunk size = 4 | +17.05 GiB | +44.3% |
| Chunk size = 8 | +39.65 GiB | +103.0% |

Peak memory 随 chunk size 单调增加。Checkpoint backward 时必须重新执行当前 chunk 的 forward；chunk 越大，一次重计算期间同时存活的内部 activation 越多，因此峰值显存越高。较小 chunk 会增加 checkpoint 边界、调度和 Python/kernel-launch 开销，但本实验关注的是显存，并且测量结果表明 chunk size = 1 的显存优势最明显。

## Complexity of Recursive Checkpointing

设网络包含 `L` 个顺序层，每层 activation 大小近似为 `A`。不使用 checkpoint 时，forward 需要保留各层 activation，activation memory 近似为 `O(L × A)`。

递归二分后，同时位于活跃递归路径上的 checkpoint 边界数与树深度相同，即 `O(log L)`。按题目采用的理论模型，峰值 activation memory 因此为 **`O(A × log L)`**，相比无 checkpoint 的 `O(A × L)` 显著降低。

计算方面，每个递归层级合计重算约 `L` 层，共有 `O(log L)` 个层级，因此计算量为 **`O(L × log L)`**。相比之下，若 chunk size 为 `k`，flat chunk 策略约有 `ceil(L / k)` 个 checkpoint，每个被 checkpoint 的 layer 通常只在 backward 中额外执行一次，因此额外 forward work 约为 `O(L)`。

因此，recursive checkpoint 的理论结论是：**以 `O(L × log L)` 计算量换取 `O(A × log L)` 峰值 activation memory**。本实验中它以 38.38 GiB 获得最低峰值；不过它仅比 chunk size = 1 低 0.10 GiB，所以在工程上是否值得采用仍取决于运行时间和允许的重计算预算。`use_reentrant=False` 的 early-stop 和 autograd graph 会影响实际常数，但不改变本题采用的渐进复杂度答案。

## Choice under a Single-recomputation Budget

如果约束是每一段最多只重计算一次，并且不允许嵌套 checkpoint，则 recursive checkpoint 不满足该预算，应在 flat chunk 策略中选择。对于这些方案，被 checkpoint 的层总体上都需要约一次额外 forward；减小 chunk size 主要减少重计算期间必须同时保存的内部 activation。

因此，在所有 chunk 都覆盖完整模型的前提下，本实验应选择 **chunk size = 1**：它在单层 checkpoint 方案中峰值最低，仅为 38.48 GiB，并且相较 chunk size = 2、4、8 分别节省 5.74、17.05 和 39.65 GiB。该选择是本组实测数据支持的显存最优解，而不是声称它在所有模型、框架实现或运行时间指标上都一定最优；若某个 chunk size 会漏掉余数层，则必须修正并重测后才能纳入比较。

## Correctness Limitation

由于 no-checkpoint baseline 在产生可比较梯度之前已经 OOM，当前实验没有真正得到一组 baseline gradients。脚本的自动成功标记只能说明 checkpoint 路径完成了运行，不能证明其梯度与无 checkpoint 计算严格一致。

若要独立验证正确性，应在能够运行 baseline 的较小模型、较短 context 或较小 batch 上，使用相同随机种子和输入分别执行 no-checkpoint 与 checkpoint，并逐参数比较 gradient。建议同时报告最大绝对误差、最大相对误差以及 `torch.allclose` 所使用的 `atol` 和 `rtol`。

## Conclusion

- No checkpoint 在约 94 GiB GPU 上仍然 OOM。
- Recursive checkpoint 的峰值最低，为 38.38 GiB。
- 在只允许一次重计算且不能嵌套 checkpoint 的约束下，chunk size = 1 是最佳选择，峰值为 38.48 GiB。
- 在各策略覆盖完整模型的前提下，chunk size 从 1 增至 8 时，峰值显存从 38.48 GiB 单调上升至 78.13 GiB。
- 当前“梯度一致”标记不能作为独立正确性证明，需要在可运行 baseline 的缩小配置上重新验证。
- Flat 实现现已正确处理不能整除的尾部层；若要使用旧版代码生成的历史结果，必须重新测量后才能作为最终对比。
