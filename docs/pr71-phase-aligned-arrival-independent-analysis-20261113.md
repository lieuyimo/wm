# PR71：自行复核 PR70 事件相位、公开任务到达和服务预测价值

**范围**：直接读取原始 PR64 BusinessSimulator DEV 源数据及 PR70 已归档概率结果；新增可执行 CPU-only 数据审计、合法规则递推及严格单元测试。未读取原始 PR59 BLIND、未训练新网络、未使用 GPU、未调用不可访问的 PR61 受控 checkpoint；不把 Rule-only 测试冒充神经模型长期精度验证。

## 1. 独立数据审计：PR70 时间差比较不成立

**[PR71 实际 CPU Actions #43405635888](https://github.com/lieuyimo/wm/actions/runs/43405635888)** 成功运行。原始 PR64 developer-validation `data.json` SHA256 `31a36e0940db7afe015fe35689cb629102dea45e8d61123ed89250abc01d0b86`。PR71 `pr71_phase_audit.py` 对每个原始公开 `t→t+1` 任务节点标签直接检查：PR52 的 one-hot 状态顺序为 `waiting, assigned, in_transit, completed`。这里的取样完成定义为已公开任务 `assigned→in_transit`，交付完成定义为 `in_transit→completed`。

| 12 个独立开发验证场景的真实公开标签 | 结果 |
|---|---:|
| `assigned→in_transit` 取样完成事件 | **320** |
| `in_transit→completed` 交付完成事件 | **320** |
| 每场首次取样完成 public observation tick | **3，12/12** |
| 每场首次交付完成 public observation tick | **9 或 11，12/12** |

这两个事件都来自真实公开后继状态标签、均非神经模型输出。更完整的逐场景第一时刻和原 PR70 时间字段在 CPU Artifact 的 `phase_audit.json`。

**对 PR70 原结论的纠正**：
- `pr70_service_timing.py` 的 `first_model_sampled_completion_tick` 来自 `NeuralServiceEvent.sampled_complete`，未区分 `to_pickup` 与 `to_dropoff`。第 3 步内部抽样不等于“模型预测样本最终交付第 3 步”。
- PR70 对照的 `first_completed_observation_tick` 是事实样本最终 `completed`，属于**交付事件**，首次发生于 tick9/11。
- 原始真实数据 **首次取样完成就发生在 tick3**，所以 `model sampled at t3 vs actual delivery t9/11` 的“模型提前 6–8 步完成”结论在事件定义上不成立。
- PR70 内部 `first_model_completion_tick` 在 `y`（事实标签）首次为 1 时被赋值，并不是网络预测完成时间；该变量名不应作为模型时序证据。
- 由于 PR70 汇总数据**没有保存每次神经抽样事件的 phase 和每个 phase 的预测概率**，不能事后把 39/93 次服务提议精确分配到 pickup/dropoff，也不能声称“第 3 步恰好预测对了取样”或推导 phase-specific Brier。

这些结论完全通过当前可读取的 PR70 和 PR64 原始证据计算，而非凭直觉推测。

## 2. PR70 的服务风险 Brier 仍有效，但解读需精确

PR70 原实际 `service_timing.json` 保持原样：12 场景、751 个 **active eligible session** 条目，508 个真实命中标签。每个模型 seed 的 Brier：map_off **0.0244 / 0.0325 / 0.0279**，map_on **0.0215 / 0.0326 / 0.0309**；Rule **0**。以 12 场景作为独立单位的平均配对差（model−Rule）map_off **+0.0283**，map_on **+0.0284**，原 bootstrap CI 均大于 0。

**这不因修正首次完成时间定义而自动失效**：该 Brier 是在**相同已开始、公开可观测的活动服务风险集**上对 `next=in_transit`（pickup）或 `next=completed`（dropoff）的真实同相位目标评估。模型在此样本集未超过 Rule，map_on 也无稳定胜出。这里的 Rule 使用与生成真实数据接近的**已公开确定性服务业务规则**，其 0 错误显示该评估近乎由已知规则决定。因此不能由此反推所有随机服务场景/网络容量都无价值。

原始数据的 320+320 个状态变化发生在全部公开状态步中；PR70 的 508 个是更窄的 active eligible service 风险集命中，**两种计数的样本条件不同，不可混用**。

## 3. 真正修正公开 stage1 到达的业务关系

PR70 的 `inject_public_stage1` 只把新任务加进 `samples`，**没有随到达重新计算站点 `waiting_pickups` 和资源容量等业务衍生字段**。PR71 新的 `pr71_public_arrivals.py`：

- 只允许在预测状态的 `tick=t`、真实 **当前公开** `observation_t` 中注入 **首次公开且处于 waiting 的 stage1**；校验前一公开观察 `observation_{t-1}` 的时间顺序；拒绝恢复已经应该存在但丢失的旧 stage1；**不读取未来 `observation_{t+1}`、不复制真实 stage2、不从隐藏工作负载调度表生成未来任务**。
- 复用原 `rebuild`、`attach_prediction`、`capacity_view` 重新推导工作站关系和容量状态；**在 `rebuild` 丢失业务扩展字段前保留当前已观测的 `device_available`**，再恢复业务设备位。
- 从先前世界出发，仍使用原 `WorkflowDynamics` / `ProgressAwareRule` / `check_business_transition` 严格检查合法任务分配与工作流后继生成。

**当前公开到达流本身是外生 oracle-conditioned 输入，不是训练好的自主 arrival predictor**。与 PR70 所谓完全 autonomous 的 H20 必须分开报告。

## 4. 实际 Rule-only CPU 验证（无真实 PR61 权重依赖）

实际 GitHub Actions [#43405635888](https://github.com/lieuyimo/wm/actions/runs/43405635888)，完整日志与环境、`phase_audit.json` / `arrival_rule.json` / `SHA256SUMS` 归档于 [Artifact #14026294967](https://github.com/lieuyimo/wm/actions/runs/43405635888/artifacts/14026294967)。

使用原 PR64 TRAIN/DEV/quality SHA、原 CP-SAT 事实轨迹的公开观察与动作、原 Rule 与 PR66 TRAIN 拟合的设备风险模型；实验均为纯 CPU，原业务合法检查不修改。

| 实际 CPU 实验 | 结果 |
|---|---:|
| PR71 新增 unit + PR70 arrival + PR66 legality 测试 | **13/13** |
| 原 12 DEV 场景事实动作/真实下一设备条件 Rule H20 | **12/12** 完整且合法 |
| 严格当前公开 stage1 注入总计（12 场景 × H20 一次） | **41** |
| 每种 12 场景 × 3 seeds × H5/H20，fresh nearest legal rule 和 TRAIN-only hazard | **72/72** 全部完整且合法 |
| 新 arrival-conditioned Rule H20 平均完成阶段 | **3.5833** |
| 新 arrival-conditioned Rule H20 平均可见 stage1 数 | **13.0833** |
| 原 PR70 **未注入后续公开到达**的 Rule H20 平均完成阶段 | **3.5833** |

这表明 **PR70 的 H20 completed count 恰好没有体现公开新到达任务造成的分布变化**：在此窗口和这些起点中，补入 41 个发布任务后，完成阶段数仍为 3.5833。以后必须同时报告阶段到达覆盖、任务状态、等待数、服务时序、对任务集的匹配程度，而不能单靠完成阶段数证明预测正确。

注意：72 条 nearest Rule 实验使用了**在每个 t 实际公开的外生 stage1 到达流**，且外生设备来自 TRAIN-only hazard；它不是完整真实 H20 精度，也没有执行任何神经模型对比。Rule 这里的可执行性只检验业务约束和公开事件输入的时序，不代表模型预测质量。

## 5. 决策与下一步

**已完成独立验证的判断**：

1. PR70 的“首次完成时序提前 6–8 步”的原论证应撤回或标注 **phase mismatch / not comparable**；不可由该时序对比判定训练模型过早。
2. PR70 的 **active-session 事件 Brier** 对比依然是一个有限、有效的信号：在这套确定性服务场景下，没有观测到比规则更好的概率预测。
3. 新公开任务必须在合规观察边界注入并重算资源状态；该修正后的 Rule H20 真正测试已通过。
4. 尚未建立**相同条件下 map_off/map_on 与 Rule 的真正 phase-specific 预训练模型闭环误差**，因为 PR70 归档只有聚合服务分数，而原 PR61 6 个 best.pt 没有提供给当前 CI。本轮不加载其他权重、不重跑 GPU、不假报模型 H20 性能。

**下一步的具体研究判断**：在现有源证据中无需增加参数或盲目训练。若要证明“面向决策的世界模型学习”，优先给已训练模型的每一条 *pickup/dropoff* service proposal 附带 phase、预测概率、是否实际产生对应合法状态变化，按 stage/service-active 风险集做 paired prediction calibration；再使用**同一个当前公开到达事件边界**对 Rule/map_off/map_on 的预测轨迹做合法性及因果误差对照。只有在同一随机事件/相同候选动作的真实反事实源上观测到稳定差异，才进入决策价值结论。

PR71 保持 Draft；无新 GPU、无 BLIND、无历史 PR 合并。
