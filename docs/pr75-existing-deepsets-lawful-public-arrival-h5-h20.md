# PR75：真实 PR74 已训练 Deep Sets 服务事件接入合法 H5/H20 世界递推

**核心问题**：PR73 证实原服务时长合成数据的确定性阻止了学习服务事件价值检验；PR74 在原模拟器 J2 服务抖动上真实训练已有 Deep Sets 服务事件模型，DEV Brier 0.18894（比 Rule 0.28886 好，但比简单 TRAIN-only hazard 0.17665 差）。本 PR 将**真实 CPU 训练权重**接入已有业务规则，验证样本任务状态、服务事件与后继阶段能否依法递推到 H5/H20，使用相同公开到达与设备条件，而不虚构事实多步准确度。

## 1. 模型、输入与数据来源

- 源模型：**PR74 的原版 `BusinessSessionEventModel(query_width=48)`**，既有 Deep Sets 40d 实体 token + 48d 公共活动服务查询。真实 PR74 DEV 选出的 CPU best.pt SHA256 `a08b257c7f8b8a6191b954e9adb117ef8076f3ff93d5373c347d4735e1e45f02`，来自 Actions Run [#43452830808](https://github.com/lieuyimo/wm/actions/runs/43452830808)。本 PR 的 CI 使用 `actions/download-artifact` 获取原 checkpoint，验证完整 PR74 `SHA256SUMS` 和 checkpoint 单文件 SHA，**没有任何随机权重代替**。
- 执行适配器：仓库原有 `business_transport.session_dynamics.SessionRuleDynamics(experimental=True,feature_version='v2')`，再交给原 `WorkflowDynamics(..., rules)` 负责业务规则、设备容量、样本阶段后继释放，原 `check_business_transition` 不关闭也不弱化。Rule 对照沿用原 `WorkflowDynamics(ProgressAwareRule)`。
- 外生设备：原 PR66 `PublicHazardRuleRollout._probabilities` + `lawful_rule_then_event`，风险参数**仅在 PR64 TRAIN** 公共设备事件拟合，两臂使用相同公开状态起点、同源外生随机数种子、同原始工作站集合，无事实未来设备时刻或私有 fault 表作为输入。由于递推状态不同，条件设备风险概率也可能不同。
- 公开任务到达：从原 J2 BusinessSimulator 真实 *current public tick t* 观察，复用 PR71 的 `apply_current_public_stage1` 仅注入当前时刻公开的 stage1；stage2 必须由上一阶段实际在**预测世界内部**完成后经合法 workflow transition 生成。这里的未来 task arrival 序列是“随时间到达时被观察”的 **外生 oracle-conditioned 评估边界**，不是预测任务到达的模型。
- 两种动力学均使用 `nearest_batch(state, candidates_for(state), geometry)` 从**自己当前预测状态**选合法动作；没有读取事实未来 CP-SAT 动作。原 PR64 TEST/PR59 BLIND 未打开，没有新训练/优化参数。

## 2. 真正 CPU 实验与可重复证据

最终 [GitHub Actions **#43454277172**](https://github.com/lieuyimo/wm/actions/runs/43454277172) **SUCCESS**，12/12 单元与接口/历史约束测试通过；[Artifact **#14052103791**](https://github.com/lieuyimo/wm/actions/runs/43454277172/artifacts/14052103791) 含完整 144 次逐时刻运行记录、环境与源码哈希、日志、SHA256SUMS。

12 个原 DEV 场景、三组独立 sampler seeds、两个窗口 H5/H20、两臂 Rule vs **原 PR74 已训练神经模型**：共 **144/144 条真实 CPU H5/H20 均达到完整窗口并通过原业务合法性校验**；真实 Deep Sets 服务查询共 **660** 次，内部 `sampled_finished` **270** 次。没有偷偷将神经模型恢复成 Rule 或改成随机模型。

| 每个 arm × horizon = 36 runs | Rule H5 | Deep Sets H5 | Rule H20 | Deep Sets H20 |
|---|---:|---:|---:|---:|
| 合法递推完成 | 36/36 | 36/36 | 36/36 | 36/36 |
| 终态平均 completed stages | 0 | 0 | **3.8333** | **2.7500** |
| 终态平均 waiting stages | 7.4167 | 7.4167 | 9.9167 | 9.9167 |
| 终态平均 visible stage1 | 10.25 | 10.25 | 13.0833 | 13.0833 |
| 终态平均 visible stage2 | 0 | 0 | **2.5833** | **2.0000** |
| 终态平均 active services | 0.5833 | 0.4444 | 0.5833 | 0.4444 |
| arm 内部 sampled-finish 次数 | 0 | 37 | 0 | 233 |

- 对每个原场景先聚合三组服务采样种子，再按 **12 个独立场景** bootstrap，H20 terminal completed delta `DeepSets - Rule = −1.083333`，95% CI **[−1.527778, −0.638889]**。
- **24/36** 个对应场景/种子在 H20 的完成阶段数上不同；H5 完成量差为 0（0/36 不同），符合 H5 无最终交付的原始观测窗口局限。
- 双臂当前公开 stage1 到达次数严格相同（H20 36 次运行各臂共 123 条注入记录；H5 各 21 条），均没有复制事实 stage2。
- 法律约束检查只证明**预测世界内部**任务和工作站的合法性，并不等同外部真实资源占用、路径冲突已经全部验证。行动规则是 nearest，而不是 CP-SAT。
- **重要：这个“完成阶段数”是各自预测模型生成的状态，而不是给定相同动作和真实随机服务实现后的事实真值。** Deep Sets 预测更少的完成阶段，不能据此宣称其长预测“更差”或调度“减少 1.08 个任务”。只有在实际反事实环境给出同动作、同外生任务/服务实现的 reference 后才可评估严格 H20 accuracy。

## 3. 本轮裁决

**PASS：** 现有公开 Deep Sets 模型对随机服务完成概率的真实 trained best.pt 可以接入已有合法的任务/站点世界转移和任务生命周期，完成 H5/H20；神经事件在 H20 产生了显著不同的预测演化。完全 CPU，原 SHA、模拟器与参考算法均可回溯。

**NO-GO（预测准确率与决策价值）：** 这些递归状态仍未与“相同实际动作、相同服务扰动和设备扰动”的真实模拟器对齐；不能由终态差异断言学习模型更准、调度更好。PR74 的条件一步 Brier 虽优于 Rule，但仍弱于简单 TRAIN-only 公开 hazard，**暂不推荐扩大模型参数/GPU训练**。

**后续可验证工作：** 在真正 action/arrival/exogenous 条件匹配的反事实环境下，比较 Rule/神经/传统服务 hazard 的逐任务分相位 `t→t+1` 事件校准、相同动作多步状态误差、工作站等待/后继阶段释放误差；只有出现超越简单模型的稳健预测及调度目标差异后，才解锁后续大规模训练与独立 TEST。PR75 Draft，PR59 BLIND 未打开，不合并历史。
