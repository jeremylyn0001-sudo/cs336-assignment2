# CS336 A1/A2 数学题练习

这里整理了 Assignment 1 和 Assignment 2 作业手册中适合推导、计算或解释的题目。本文是练习清单，不提供答案。你先在前端尝试作答，我会逐题检查推理；订正后再整理成最终 writeup。

打开交互页面 [`math_practice.html`](math_practice.html) 可以逐题作答、保存草稿和标记完成；公式通过 MathJax 排版，需要浏览器能访问 jsDelivr。若公式库加载失败，页面会保留原始 LaTeX。内容保存在当前浏览器的本地存储；换浏览器或设备前，请用“导出我的答案”保存 Markdown 文件。

前端每题的“官方给定 / 已知条件”区只列作业手册中已经给出的关系、配置或假设；要求计算的总式和结果留给你推导。A1 的交叉熵手算与 AdamW 手算卡是补充练习，不是作业手册里的独立数值题。

## Assignment 1

1. **Transformer 参数量与权重存储**（Transformer accounting，§3.5）：GPT-2 XL 风格配置为词表 50,257、上下文长度 1,024、48 层、宽度 1,600、25 个注意力头、FFN 宽度 4,288。计算可训练参数量和 FP32 权重占用，并说明是否假设输入 embedding 与输出投影共享权重。
2. **按模块计算前向 FLOPs**（§3.5）：列出该配置前向过程中的主要矩阵乘法并求总 FLOPs，分别统计 Q/K/V/输出投影、注意力分数与加权求和、SwiGLU 投影和词表投影。
3. **模型规模与上下文长度对 FLOPs 的影响**（§3.5）：比较 GPT-2 Small/Medium/Large/XL 各模块的 FLOPs 占比；再把 XL 上下文从 1,024 增至 16,384，解释哪些项随上下文线性增长，哪些项呈二次增长。
4. **AdamW 训练显存核算**（§4.3）：假设所有张量都是 FP32，分别推导参数、激活、梯度和 AdamW 状态的显存表达式。激活项按题目要求统计 norm、注意力、SwiGLU、最终 norm、词表输出和交叉熵。代入 GPT-2 XL，写成 `a × batch_size + b`，求 80 GB 下允许的最大 batch，并说明假设和单位换算。
5. **训练 FLOPs 和时长估算**（§4.3）：推导一次训练步的 FLOPs 表达式，写明反向 FLOPs 如何相对前向计算、是否计入优化器算术。再估算 H100（495 TFLOP/s 峰值、MFU 50%）上以 batch 1,024 训练 GPT-2 XL 共 400K 步所需的时间，明确 token、FLOPs、秒和天的换算。
6. **稳定交叉熵与困惑度**（§4.1）：自选一组短 logits 和目标类别，用减最大值后的 log-sum-exp 计算交叉熵；再从若干 token loss 计算序列 perplexity。解释整体减去最大 logit 为什么不改变 softmax 概率。
7. **手算一次 AdamW 更新**（§4.3）：自选参数 `θ`、梯度 `g`、初始 `m/v`、学习率、`β₁/β₂`、`ε` 和 weight decay，手算一到两步。分别列出偏差修正学习率、解耦 weight decay、一阶矩、二阶矩和最终参数更新。

## Assignment 2

8. **混合精度累加**（§2.1.5）：比较用 FP32、FP16 反复加一个小增量，以及 FP16 输入但 FP32 累加时的结果。用尾数精度解释何时增量会消失；再区分 FP16 与 BF16 的指数范围，并说明哪种更容易溢出。
9. **激活检查点与重计算**（§3.3）：对顺序堆叠的 `N` 个 block，比较不做 checkpoint、单层 checkpoint 和递归嵌套 checkpoint 的峰值激活内存与重计算量。忽略计算代价时，给出降低峰值内存的策略及渐进复杂度；如果只能重算一次，说明怎样选择 checkpoint 粒度。
10. **注意力显存复杂度**（§4.1）：对 batch `B`、头数 `H`、序列长度 `T` 和每元素 `s` 字节，推导 dense attention score/probability 张量的形状和字节数。计算 `T` 翻倍时的影响，并与 FlashAttention 保存的 Q/K/V/O/L 张量比较。
11. **FlashAttention 在线 softmax**（§4.2.2）：已处理的 key tiles 保存行状态 `(m_old, l_old, o_old)`，其中 `m` 是运行最大值，`l` 是缩放后的分母，`o` 是未归一化的加权 value 累加器。给定新 tile 的分数 `S_tile` 和 values `V_tile`，推导稳定更新式，并写出全部 tile 处理完后如何恢复归一化输出。
12. **FlashAttention 反向推导**（§4.2.2）：从 `O=PV` 和上游梯度 `dO` 出发，推导 `D = rowsum(O ⊙ dO) = rowsum(P ⊙ dP)`。然后写出由 Q/K/L 重算 P，以及 dV、dP、dS、dQ、dK 的表达式；说明哪个大张量在反向重算而不保存。
13. **Ring All-Reduce 通信量**（§8.1）：有 `N` 张卡、张量大小 `S` 字节、每卡出口带宽 `W`。推导标准 reduce-scatter + all-gather Ring 的每卡传输量与理想时间，再推导每轮传完整张量的替代算法；用 `N=4, S=120 MB` 比较两种算法。
14. **数据并行 FFN 核算**（§8.2）：令 `x:(B,D)`、`W₁/W₂:(D,Dff)`、`W₃:(Dff,D)`。列出反向的所有矩阵乘法，推导 `N_DP` 下每卡 FLOPs；假设 FP16，计算梯度字节数和 Ring All-Reduce 时间；比较通信与计算时间并求仍保持 compute-bound 的 `N_DP` 上限。
15. **FSDP FLOPs 与通信**（§8.3）：在相同 FFN 设定下，推导 `N_FSDP` 下每卡前向和反向 FLOPs；统计两阶段的权重 all-gather、梯度 reduce-scatter 通信量与时间；将通信时间和计算时间比较，分别求前向/反向通信瓶颈阈值。
16. **Tensor Parallel 反向与扩展上限**（§8.4）：`W₁/W₂` 按输出维切分，`W₃` 按输入维切分。写出每卡从 `dY` 到 `dX` 和局部权重梯度的反向公式，标出 collective；推导前向/反向 FLOPs、通信时间和 `N_TP` 的 compute-bound 上限。
17. **FSDP + TP 二维并行**（§8.5）：总卡数 `N=N_FSDP×N_TP`。推导每卡前向 FLOPs 与两个 mesh 方向通信时间。先假设两个方向可重叠，用 max 表示通信代价并优化切分；再假设网络资源共享、通信无法重叠，重新推导最大 compute-bound 卡数。
18. **DDP、ZeRO-1、FSDP 显存**（§§6–7）：对 `P` 个参数、`N` 张卡、FP32 权重与梯度、每个参数两个 FP32 Adam moment，推导 DDP、ZeRO-1、FSDP 的每卡持久显存。另计 FSDP 临时 all-gather 单元，并说明为什么 `16P/N` 不能直接视为训练峰值显存。

## 建议顺序

先做 A1 参数量和 FLOPs，再做 A2 Ring All-Reduce 与 DP；之后依次做 FSDP、TP、二维并行，最后做激活/注意力显存和 FlashAttention 推导。这样从基本量纲逐步走到系统级公式。

## 实验完成度与数学练习分开

本练习页只整理数学与概念推理，不替代 A2 的编码、测试、性能分析、profile 截图和基准数据。H20 Leaderboard 运行是替代硬件实验，不能当成官方 B200 排名成绩。
