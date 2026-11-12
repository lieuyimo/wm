# PR36：修复世界模型在线 H20 的严格预算接纳门禁

研究日期：2026-11-12。

## 一、实际问题与受影响的已知结果

并行独立 [PR #33](https://github.com/lieuyimo/wm/pull/33) 的源码与数据归档审计证实：原 `OnlineHorizonPolicy.__call__` 只在各候选预测的**开始前**检查0.5秒CPU预算。如果最后一个候选的 H20/P3 预测在预算耗尽后才返回，旧代码仍可能直接采用其世界模型评分并改变派单。PR33发现 [在线真实预算审计](https://github.com/lieuyimo/wm/actions/runs/43146849683) 中已有超出真实CPU预算的改选实例。

因此此前PR30～PR35仍是原代码下的真实仿真结果，但**不构成严格0.5s模型决策预算下的性能证明**。事件预测Brier、完整真实轨迹和模型权重不因为本次源码问题发生变化，不重写或覆盖历史实验产物。

## 二、最小修复：无需新算法或模型训练

本 PR 基于现有在线策略作最小修复，所有世界模型、方法与场景均保持原样：

1. 各候选世界模型推演完成后，**立即**检查当前 CPU process time；超预算即清空候选评分，记录超预算并回退至合法 CP-SAT 首选方案。
2. 所有候选评分完成后，在准备使用候选完整排序时再次检查预算。
3. 候选排序应用前记录准确的 `model_selection_cpu_seconds`；**任何接受的模型选案都必须在0.5秒以内**。超预算最终结果不能改变本次真实动作。
4. 利用可控 CPU时钟的确定性单元测试覆盖「最后一个候选超时」「评分动作恰好越过预算」「预算内合法改选」「不使用模型时继续采用CP-SAT」四种情况。不以少量人工睡眠时间代替测试。
5. 运行时单个模型 rollout 和 CP-SAT 续规划不可强制中断，因此本实现**不是硬实时0.5秒墙钟保证**。它保证的是超时的模型结果不会被用于派单，并将超时请求和完整计算开销记录供后续优化。

## 三、严格独立的在线复测

场景、预算和对照在 `experiments/world_model_integration/trajectory_cost/pr36_protocol.json` 事先冻结：

- **24组全新独立场景**，light / medium / heavy / shift 各6组，与PR24～PR35已有Train/Val/Test随机种子均分离；
- 六条事实在线策略：CP-SAT、Rule-progress、Rule-cost、原 V4-progress、原 V4-cost、PR31 Validation事先选中的 Scratch-cost；
- 相同原始 BusinessSimulator 动态、同场景 keyed service duration、同一 CP-SAT 候选集与后续计划；H20/P3和单次0.5秒CPU门槛完全相同；
- 不训练模型、不引入新随机机制、不修改企业工程演示PR19；
- 每次实际使用模型方案时必须记录 `model_selection_cpu_seconds < 0.5`。如出现任何未记录但实际改变了原CP-SAT首选方案的行为，整次实验证据失效；
- 主要比较真实最终 makespan、超时任务、资源等待、受限预算回退次数、候选改变次数、模型加载及全场景CPU开销，并同时报告Rule参考。

**原始 GitHub Actions 运行**：[CI #43148877448](https://github.com/lieuyimo/wm/actions/runs/43148877448)。正式结果以该次 CI 原始独立测试 artifact 为准。任何结果都不用于修改已封存的 PR36 协议。

## 四、方法定位与后续研究约束

本 PR 解决的是世界模型的**在线预算与决策执行正确性**，不是新网络架构或价值模型算法。长期研究仍应优先分析：现有 V4 事件预测能否在多步骤真实动态场景下产生有别于 Rule/CP-SAT 的可靠动作价值，尤其是任务最终交付与工作站资源排队的因果路径。若要后训练，必须继续采用PR31已经完成的完整pip环境、源文件哈希、Train/Val/Test独立性、best/latest optimizer及RNG可恢复检查点归档要求。
