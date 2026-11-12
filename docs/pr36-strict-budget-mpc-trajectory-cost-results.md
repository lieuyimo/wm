# PR #36：已完成的 PETS/MPC 累计轨迹代价严格预算在线实证

日期：2026-11-12
状态：**代码、独立实际实验、7臂CPU门禁和原始实验日志均已完成并通过。**
本PR是基于 Draft PR #32 的独立严格预算补充验证，与仓库已有 [PR #35](https://github.com/lieuyimo/wm/pull/35) 的累计代价迁移研究相邻；它**不是新的预测网络、训练算法或原创累计损失**。

## 1. 已有方法、具体实现及差异

- Chua et al. (2018), *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models*, NeurIPS PETS，[原论文](https://proceedings.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)：参照概率粒子轨迹采样、候选动作的预计累计代价规划，不复制其连续控制、模型集成及 CEM。
- Reiter et al. (2026), *Synthesis of model predictive control and reinforcement learning: Survey and classification*, [DOI](https://doi.org/10.1016/j.arcontrol.2026.101045)：参考有限H步累计代价、执行首步然后在线重规划。
- **仓库直接调用** `experiments/world_model_integration/decision_value/pr27_evaluate.py::_trace_cost`，它已复用原 `lab_transport/decision_value.py::trace_score` 的未完成任务累计时间成本、逾期固定惩罚4和预测终点剩余运输距离，按RHWM三粒子取期望并对两种成本按既定字典序排序。所有模型不训练、不改结构、不新造 cost/奖励。
- 与 PR30/32 的既有 `expected_completed_samples` 最大化作为固定对照，所有结果在全新未见场景计算。
- 独立于 PR #35 的明确补充：本PR在每个候选预测**开始前和完成后**都核验0.5s CPU，任意超限不使用部分或全部模型重排，而退回CP-SAT首方案。这直接回应 [PR #33 CPU预算源码审计](https://github.com/lieuyimo/wm/pull/33) 已证实的旧策略只在开始前查时限问题。单次 `rollout` 本身不可中断，**超过时限后安全回退并不等于硬实时最坏延迟保证**。

冻结文件 `experiments/world_model_integration/mpc_cost/pr33_protocol.json`（历史分支内部版本号PR33，外部正式PR号为36）：24组预设新seed/4种负载各6，独立于PR24～PR32全部已见场景；原始V4 SHA `60c8a6c3d832da3e6519072688ea099f64ae120ba19d3bd2ec705f7f1a11ac0c`；新Scratch权重来自PR31 validation选出的checkpoint及归档hash。每组均实际完成原始 BusinessSimulator 业务轨迹，CP-SAT候选、H20/P3、公开状态和服务时长哈希一致。唯一干预是固定的完成进展与累计代价目标。无GPU、无新模型训练。

## 2. 专用GitHub Actions事实验证

[Actions #43147122344](https://github.com/lieuyimo/wm/actions/runs/43147122344) **SUCCESS**；[7臂×24组全部实际决策与事实JSON、验收日志、SHA256 artifact #13912451562](https://github.com/lieuyimo/wm/actions/runs/43147122344/artifacts/13912451562)。2项源代码/模拟器合约测试通过，**168条真实在线策略场景轨迹**，全部完成，0安全违规，训练0，GPU0。主要实验 CPU进程约235.79秒。

| 6组各负载，平均最终完成tick（低者好）| CP-SAT | Rule进展 | Rule累计代价 | 原V4进展 | 原V4累计代价 | Scratch进展 | Scratch累计代价 |
|---|---:|---:|---:|---:|---:|---:|---:|
| light | 20.167 | 20.167 | 20.167 | 20.167 | 20.167 | 20.167 | 20.167 |
| medium | 48.000 | 47.667 | **46.667** | **47.000** | 47.333 | 47.167 | 48.167 |
| heavy | 75.333 | 75.000 | 75.500 | **74.000** | 74.833 | 75.000 | 75.000 |
| shift | 109.833 | 109.833 | **109.000** | 109.833 | 109.333 | 109.833 | 109.167 |

真实每层均只有6组，不能因某一均值选出部署默认策略。所有完成时间均以实际仿真为准，不把模型认为的预计成本当成真实收益。

### 累计代价相对于同模型进展评分的配对结果

| 条件 | Rule累计代价−进展 | 原V4累计代价−进展 | Scratch累计代价−进展 |
|---|---:|---:|---:|
| light | 0 | 0 | 0 |
| medium | −1.000 tick (2胜3平1负) | +0.333 (1胜3平2负) | **+1.000 (0胜4平2负)** |
| heavy | +0.500 (1胜4平1负) | **+0.833 (0胜4平2负)** | 0 (3胜1平2负) |
| shift | −0.833 (1胜5平0负) | −0.500 (1胜5平0负) | −0.667 (1胜5平0负) |

按24组均匀平均，相较同模型原进展评分：Rule累计代价约改善0.333tick；原V4累计代价约**退化0.167tick**；Scratch累计代价约**退化0.083tick**。仅为预注册合成场景描述性结果，方向并不稳定；**不存在“轨迹累计代价普遍优于H20工序进展”或“新V4必然改善交付”的证据**。

### 严格CPU预算：原研究里最应保留的结果

每次多候选评分的总CPU预算为0.5s，超预算时整批候选作废并回退原CP-SAT首方案。预算回退/超时计数（6组合计）：

| 条件 | Rule进展 / 累计 | 原V4进展 / 累计 | Scratch进展 / 累计 |
|---|---:|---:|---:|
| light | 0 / 0 | 0 / 0 | 0 / 0 |
| medium | 2 / 3 | 1 / 3 | 2 / 4 |
| heavy | 13 / 15 | 18 / 18 | 18 / 18 |
| shift | 1 / 6 | 16 / 17 | 15 / 15 |

这表明真正执行候选级后置时限检查后，**高负载和服务参数变化下的H20模型预测评分仍过于昂贵**。和旧PR30/32的“开始前检查”的预算统计不能不加说明直接作数字比较。当前最多可验证“超限后不会执行超预算模型建议动作”，不能将之写为硬实时达标（单候选内部rollout仍可能耗时超过0.5s）。

## 3. 阶段裁决与下一步骤

1. **已取得工程结果**：在合法的事件驱动多机器人样本转运系统中，既有RHWM V4、Rule均能以PETS/MPC方式计算累计预测业务代价，已完成实际在线实验且不修改旧系统核心约束。
2. **仅替换预测评分不构成稳定增益**：原V4的累计代价在medium/heavy比原H20进展评分变差；Scratch同样无稳定改进。之前PR31事件预测Brier的改善没有自动转换成在线交付价值。
3. **紧迫瓶颈是规划计算预算和动作选择信号**：24组中，多候选H20预测在heavy/shift经常超0.5s；Rule也受影响，说明内层CP-SAT重复求解是影响因素之一。后续方法优先参考已有 **event-triggered MPC** / 规划器的计算复用策略并严格保留合法业务状态转移，不应在已见数据上调整该累计代价系数。
4. **不自动继续训练**：原PR31完整环境与Adam+RNG状态已经保存于 [GitHub Draft Release](https://github.com/lieuyimo/wm/releases/tag/untagged-871b60f7bf013400cb45)。下一次仅在有新独立训练数据或新的事件模型适用问题时再训练，仍强制完整环境快照与可恢复优化器状态保存。
5. **不晋升新默认调度策略**。本阶段足以说明：需要研究有限计算资源下如何利用世界模型，而非仅提升静态事件概率或任意加长预测时域。
