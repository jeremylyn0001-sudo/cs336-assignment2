# A2 实验收尾清单

本清单按 Assignment 2 Systems 官方手册逐项对照当前仓库记录，区分实际还要跑的实验、只缺交付证据的项目，以及硬件受限的官方 Leaderboard。它只讨论实验交付；§8.3–§8.5 的并行计算推导见 [A1/A2 数学练习页](../study_guides/a1_a2_math_practice.md)，不需要 GPU。

## 还需要实际跑的实验

| 优先级 | 手册项目 | 分值 | 当前记录 | 需要补的内容 | 完成标准 |
|---|---|---:|---|---|---|
| 1 | §4.1.1 PyTorch Attention Benchmarking | 2 | 未找到对应的独立 batch-8 benchmark 表；现有 FA2 网格是 batch 1 + causal，负载不同 | batch=8、不使用 multi-head；`d_model∈{16,32,64,128}`，`seq_len∈{256,1024,4096,8192,16384}`。记录 forward、backward 前显存和 backward 耗时；报告 OOM 的配置 | 表格覆盖完整 4×5 网格，含 warmup、同步方法和 OOM 记录 |
| 2 | §4.1.2 Torch Compile | 2 | 仓库里有 FA2 compiled backward，但没找到本题要求的对照表 | 对上面的 attention 网格比较普通 PyTorch 与 `torch.compile`；另比较未编译/编译的完整 Transformer 前向、前向+反向、含 optimizer step | 两张对照表；将编译/JIT 首次开销与稳态耗时分开记录，并写出完整模型是否成功编译 |

这两项可先用可用 GPU 做课程实验；如果仍使用 H800/H20 等替代卡，报告里写清型号，不把结果描述成 B200 测量。

## 只缺交付证据，不必重跑训练

| 手册项目 | 现有证据 | 需要补的交付 |
|---|---|---|
| §5.3.2 DDP overlap profiler 对比 | 仓库已有 naïve 与 overlap 的 `.nsys-rep` 原始 trace，writeup 有耗时比较 | 用 Nsight Systems 打开两份 trace，各截一张能看出“通信等待 backward”与“通信和 backward 重叠”的 GUI 截图，并在 writeup 里标注对应区间 |
| §7 FSDP accounting | 已有 FSDP raw `.nsys-rep`、SQLite、统计 JSON 和从 trace 生成的 timeline 图 | 手册要求 Nsight 截图；补一张 Nsight GUI 中 all-gather 与 forward compute 的截图，并简短说明 all-gather 是否赶在消费权重前完成。当前 PNG 是绘图导出，不应称作 GUI 截图 |

## 有条件才能完成的官方 Leaderboard

§9 Leaderboard（10 分）需要两张 B200、batch 2、序列长度 32,768、BF16 causal 的完整训练步（forward、loss、backward、AdamW）；测试从空 PyTorch/Triton cache 开始，要求在 10 分钟内结束，并提交官方排行榜。

目前 [2×H20 记录](leaderboard_h20_20261015/README.md)测得约 121.18 s/step，是有用的替代实验，但不是官方 B200 结果，也没有验证空 cache 条件。没有 B200 时不要重复跑 H20 来“补成”官方成绩；将此项明确留作硬件受限即可。若以后拿到 B200，再按手册从干净环境运行、检查 correctness、记录耗时并提交。

## 已有实验，不建议为收尾重复跑

- End-to-end benchmarking、Nsight profiler、mixed precision、memory profiling 和 activation checkpointing 均有脚本、结果与 writeup。
- FA2 的 H800 substitute 网格、backward 比较及 tile sweep 已记录。4 个 dense FP32 配置 OOM 已如实标为 partial；除非要补 B200 对照或你想专门研究该 OOM，否则无需再次做相同 H800 扫描。
- DDP/collective、ZeRO-1 optimizer-state sharding 和 FSDP 均有多卡正确性与性能/显存记录，硬件是替代卡并已注明。
- Checkpointing 大配置的 no-checkpoint baseline OOM，无法比较大配置梯度；缩小配置已有梯度一致性检查。它是已记录的实验限制，不需要再用同样会 OOM 的配置重复跑。

**DDP 的 4/6-rank XL 全训练 scaling** 是可选扩展，不是手册当前要求的 DDP XL benchmark（该项指定 2 GPUs）；手册要求 2/4/6 ranks 的是 all-reduce 通信基准，该表已有记录。

## 建议收尾顺序

1. 补跑 §4.1.1 attention 基准和 §4.1.2 compile 对照。
2. 补拍 DDP overlap 与 FSDP all-gather 的两类 Nsight GUI 截图。
3. 若没有 B200，就把 Leaderboard 保留为硬件限制；不要把 H20 结果标成官方完成。
4. 单独完成数学练习页中尚未作答的 §8.3–§8.5 推导，然后再写最终 A2 总结。
