# PR74：复用原 Deep Sets 条件服务事件世界模型的真实 CPU 训练

继承 [PR73](https://github.com/lieuyimo/wm/pull/73) 已验证的 **原始 BusinessSimulator keyed 0–2 ticks** 服务抖动机制。不是创建神经算法，不增加原 PR61 参数，不将旧图网络 checkpoint 当作新输入。直接复用 `lab_transport.business_session_learning.BusinessSessionEventModel`、`encode_session_batch`、`metrics`、标准 BCEWithLogits+Adam、当前 `BusinessObserver` 的公开动态，以及原 `SessionRuleDynamics` 的离线加载接口。

## 1. 数据和训练目标

- 输入：`business_transport.state_features.build_business_features` 的当时公开实体、已执行合法 CP-SAT action 与 `session_events.eligible_sessions(require_active=True)` 的服务查询；原始 **48d** `encode_query` 和现有 Deep Sets **40d/实体 token**。包含已公开的 wall/active/paused elapsed、已完成服务的有效时长统计、站点设备状态与公开资源关系；未知未来故障、实际服务剩余倒计时、私有哈希 jitter 不在模型输入中。
- next-tick 标签：`to_pickup → in_transit` 为取样服务完成；`to_dropoff → completed` 为交付服务完成。标签仅从真实 `t+1` 公开状态读取。
- 真实来源：PR64 原始 SHA 已校验 TRAIN 24 独立场景、DEV 12 独立场景，三类业务 regime 同时覆盖；固定原始 simulator 内置 jitter=2 的可复现实验，原 CP-SAT 与随机设备故障源不修改。
- 监督架构：完全沿用现有 `BusinessSessionEventModel(query_width=48)`；单 CPU seed 1810、20 epochs、batch64、lr=0.001；标准 BCE/Adam；用 DEV Brier 选择最佳 epoch。此为**开发验证的训练烟测**，没有独立未见 TEST，也没有优化线上调度器。
- 保存 best model、真实优化器状态、torch RNG、训练/验证源码 SHA、完整 epoch loss 与 metric、pip freeze、git SHA；旧 PR61 checkpoint 不加载，输出 `online_verified=False`。新 checkpoint 使用现有 `lab_transport.business_session_learning.load_session_model(experimental=True)` 完成离线格式 round-trip，不能直接上线。

## 2. 首次真实训练验证（原代码、未做模型搜索）

[首轮 CPU CI #43451874810](https://github.com/lieuyimo/wm/actions/runs/43451874810) SUCCESS，9/9 单元测试（包括 PR73 基础机制测试），24 TRAIN / 12 DEV 真实仿真源全量生成，单 seed 训练 20 epochs、选出最佳第 18 epoch。 [原首轮 checkpoint/artifact #14051980301](https://github.com/lieuyimo/wm/actions/runs/43451874810/artifacts/14051980301)。

| 同一原 PR64 DEV 12 场景，J2 | 服务事件 Brier |
|---|---:|
| Original Rule (deterministic progress) | 0.28886 |
| TRAIN-only phase constant event rate | 0.24199 |
| **现有 Deep Sets 服务事件模型（实际 CPU 训练）** | **0.18896** |
| TRAIN-only station×phase×active elapsed discrete hazard | **0.17665** |

- 真正 DEV 风险集 1,409 个，其中 594 个下一步事件；TRAIN 同设计 2,593 个风险集条目。不是将 1409 个相关时刻当独立场景。
- Deep Sets 取样 Brier **0.17209**（831 risk rows），交付 Brier **0.21321**（578 risk rows）；ECE(5 bins) **0.04638**，DEV threshold=0.5 accuracy **0.70121**。
- Deep Sets 优于原 deterministic Rule，但**尚未优于更简单的 TRAIN-only 公开条件离散 hazard**。本轮更重要的结果是原有 Deep Sets 模型在随机服务环境下学到了明显非零的条件事件动态，而不是证明神经网络结构必要。
- 无 PR61 本体图网络训练/续训，无 BLIND test，无合法 H20 learned-model rollout、候选动作改选或端到端真实调度收益证据。

## 3. 最终真实 CPU 复测与逐场景显著性

[最终 CI Run #43452830808](https://github.com/lieuyimo/wm/actions/runs/43452830808) **SUCCESS**，9/9 测试， [原始模型权重及全部结果 Artifact #14053230686](https://github.com/lieuyimo/wm/actions/runs/43452830808/artifacts/14053230686)。

- 验证集来源仍为相同 12 个 PR64 原始场景，1,409 个实际活动服务风险，594 个事件；没有换 seed、模型、损失或选择更有利的数据集。
- Best epoch=18，完整 20 epochs；Deep Sets **DEV Brier=0.18893717**；Rule=**0.28885735**；TRAIN-only station×phase×active_elapsed 传统风险估计=**0.17665487**。
- 相位：Deep Sets pickup=**0.17207059**（n831），delivery=**0.21318654**（n578）；DEV ECE5=**0.04743214**。
- **12 独立场景等权宏平均 Brier**：Deep Sets **0.19121147**，Rule **0.29511460**，传统公开条件 hazard **0.17659304**。这些宏平均与按 1,409 事件行加权的 Brier 不是同一种统计单位。
- 按 12 原场景做 2,000 次 paired bootstrap，Deep Sets−Rule 场景平均 Brier 差 **−0.10390313**、95% CI **[−0.12198012, −0.08728231]**；**12/12 场景** Deep Sets 更低。
- Deep Sets−传统公开条件 hazard 差 **+0.01461842**、95% CI **[+0.00317812, +0.02477636]**；Deep Sets **仅在 3/12 场景** Brier 更低。
- 新最佳 checkpoint SHA256 **`a08b257c7f8b8a6191b954e9adb117ef8076f3ff93d5373c347d4735e1e45f02`**，文件 `model/existing_deepsets_public_service_cpu_best.pt`。保存原 v2 `BusinessSessionEventModel` 元数据、best optimizer deep-copy 状态、torch RNG、source SHA、offline-only `online_verified=False`；CI 已实际运行 `load_session_model(checkpoint, experimental=True)` 安全加载往返检查。

**解释**：场景配对结果支持“基于公开服务查询的原 Deep Sets 已学得足以优于原确定性 Rule 的条件事件概率”，但同组更简单的 TRAIN-only 传统风险模型更准确。**PR74 不支持立刻扩大模型参数、声称世界模型长期预测更好或在线调度收益**。这仅是定向服务事件组件的一步条件预测能力。

## 4. 后续 CPU 代码审计和接受标准

- `experiments/world_model_integration/service_pr74/pr74_cpu_training.py`：复用真实业务仿真、PR73 事件数据源、原 Deep Sets 40d 实体 token+48d 查询，原 Adam/BCE，存储正确 best optimizer / RNG 以及 original `load_session_model` 可接受的离线元数据。
- `experiments/world_model_integration/service_pr74/test_pr74.py`：冻结 input width、模型复用、开发场景为独立统计单位。
- `.github/workflows/pr74-existing-deepsets-service-cpu-train.yml`：验原 SHA、下载原 PR64 公开源、真正生成 TRAIN/DEV 事件、训练、复核 checkpoint、归档环境与原始结果，明确 0GPU/0BLIND。
- 后续训练规模和模型选择应以开发集场景级配对分数与同一风险集公开条件传统 hazard 为最低门槛。若神经模型未优于简单基线，先对原服务特征/校准进行诊断，不盲目加参数或上 GPU。

## 5. 研究裁决

**训练可行性：GO**。真实随机服务数据、已存在的服务观测输入和现有 Deep Sets 服务网络可以完整运行监督训练并达到高于固定 Rule 的条件预测能力。**模型选择/调度价值：NO-GO**，因为尚未超过传统条件风险估计，也尚未完成合法 H5/H20、跨场景泛化或可执行 CP-SAT 调度收益验证。

下一轮优先复用已有 `SessionRuleDynamics` 将训练好的真实事件概率介入合法 `WorkflowDynamics`，再结合 PR71 的当前公开任务到达边界与 PR66 的设备事件，验证在**同信息、同外生条件**下能否改善完整未来业务状态；不得根据 DEV 结果调整新 BLIND 训练方案。无线上模型晋升。

最终状态：PR74 与 PR73 保留 Draft stacked PR；原 PR61 checkpoint、PR59 BLIND、原 CP-SAT 和原模拟器实现均未覆盖。
