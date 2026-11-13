# PR72：关系世界模型的服务事件预测、输入可识别性与数据目标审计

**研究起点**：[PR70](https://github.com/lieuyimo/wm/pull/70) 的真实 PR61 pretrained map_on/off active-service 事件结果、[PR71](https://github.com/lieuyimo/wm/pull/71) 的已执行 pickup/delivery 相位纠正。本文以 PR71 HEAD 为基线开展真实 CPU 源分析，保持 stacked Draft；0 GPU、0 新训练、0 BLIND、0 checkpoint 替代。

## 1. 动机：从“加参数”转向世界模型输入是否可识别

原 PR61 `PublicMapTemporalRelationalWorld` 是 **4 帧 GRU + 业务实体 RGCN + 静态地图 PyG SAGEConv** 的 next-public-state 模型，任务状态头是 4 分类 `softmax`：`waiting / assigned / in_transit / completed`。原 `supervised_losses` 的训练目标是 masked node MSE + 0.3 new-stage event MSE + 0.1 task-status CE；**它不是专门拟合“开始服务后的下一步完成概率”的二分类 hazard 模型**。PR67 将既有 `P(next=in_transit)` / `P(next=completed)` 用作相应公开活动服务会话的事件采样，最后由规则执行合法下一状态。

输入信息存在具体差别：

| 公开服务信息 | ProgressAwareRule | 现有 PR61 原始节点编码 |
|---|---|---|
| 当前机器人 wall-clock `service_elapsed` | 使用进度校正 | **存在**，robot x[9] |
| 当前机器人 `service_active_elapsed` | **使用**，扣除暂停时间 | **缺失** |
| `service_paused_elapsed` | 由 active/wall 区分 | **缺失** |
| 工作站 `observed_effective_service_mean` | **使用**，已知时覆盖名义预计时长 | **缺失** |
| 站点 `device_available` | 使用 | **存在** |
| 静态地图拓扑 | 运动规则使用 | **存在**（map_on） |

上表已在 `lab_transport/torch_structured_adapter.py:ProgressAwareRule`、`lab_transport/state_v2.py:progress_adjusted`、`experiments/world_model_integration/decision_learning_pr51/pr51_train.py:node_features`、`pr59_dual_model.py` 和 PR67 `PublicGraphHistory` 的实际代码中检查。因此，PR70 **Rule Brier=0 对神经模型不是信息完全对等的对照**。这不否定 Rule 基线或服务 Brier 的正确性，只否定据此直接归因“仅仅是模型容量不够”的解释。

## 2. 原始 PR64 数据上的完整同相位活动服务风险集

原 PR64 TRAIN SHA256：`4c1157777117f82f7deb44f98e1d8255e95611dfcb41c6dfff9e14dd23aee183`；原 PR64 DEV SHA256：`31a36e0940db7afe015fe35689cb629102dea45e8d61123ed89250abc01d0b86`。

通过 `factual_trace` 原样重新运行 12 个不同 DEV BusinessSimulator 场景、原 CP-SAT 动作和原 `eligible_sessions(require_active=True)`，对同一公开 observation`_t` / 合法 action`_t` / 真实下一公开状态核对：

| 指标 | 取样 `to_pickup` | 交付 `to_dropoff` | 合计 |
|---|---:|---:|---:|
| 活跃且可服务的 public risk rows | **506** | **245** | **751** |
| 真实下一时刻目标状态事件 | **320** | **188** | **508** |
| 事件率 | 63.24% | 76.73% | 67.64% |
| Rule 对同风险集二值事件错误 | **0** | **0** | **0** |
| wall elapsed ≠ effective active elapsed | 0 | **16** | 16 |
| 已有公开 effective service mean | 458 | 210 | **668** |
| 公开 effective mean ≠ nominal service ticks | 18 | 53 | **71** |

*这 751/508 个标签与 PR70 已归档的风险集计数一致，且分开了阶段。不是把 PR71 的全部任务转换 320/320 混到 narrower active-session 风险集。*

Rule 完全匹配原真实模拟器生成的公开事件，印证这批 DEV 活动服务在现有可观察信息和公开规则下已接近可确定。这不意味着在所有场景里真实服务时长必然可知。

## 3. 控制变量证明：同一 PR61 输入，合法下一状态不同

对 PR64 真实 DEV 公开活动服务起点（tick1，pickup，公开 active elapsed=1、wall elapsed=1）建立**两个受控反事实当前状态**，除“当前已经公开的站点有效服务时长”外，其他公开实体、机器人、动作、地图和上一步历史保持相同：

- 场景 A 的公开有效服务预计时长设为 2，原 `WorkflowDynamics(ProgressAwareRule)` 合法预测 `in_transit`；
- 场景 B 设为 8，同一规则合法预测 `assigned`；
- PR51/PR54 原始 graph 与 16 维 node features **逐字段相同**；
- PR61 `PublicGraphHistory` 以相同 t−1 公开 observation/action 为历史，比较其输出的原始 **PyG entity.x、history、mask、relation edges、map features/edges、entity-map links**；结果应全部相同（由本 PR CPU CI 强制检查）。

上述对照是**受控反事实输入别名**，不是声称自然 DEV 存在同一个观测标签冲突，也没有实际运行 pretrained PR61 权重。但任何在这种同输入下确定性返回分布的现有模型都无法同时区分 A/B 规则结果，构成明确的**表达输入信息瓶颈**。加大网络参数无法恢复被特征编码丢弃的服务时长。

## 4. 监督目标的状态不变主导

直接在原 PR64 source `input.graph.nodes` 与 `target.node_target + node_mask` 上计数，每个 task node 使用 PR61 完全相同的 4 状态编码：

| 相同标签 schema 的 PR64 source | TRAIN（24 场景） | DEV（12 场景） |
|---|---:|---:|
| 有效 next-task-state 节点标签 | **64,551** | **30,402** |
| 下一状态不变 | **62,770（97.24%）** | **29,442（96.84%）** |
| 下一状态变化 | **1,781（2.76%）** | **960（3.16%）** |
| `assigned→in_transit` | 595 | 320 |
| `in_transit→completed` | 590 | 320 |
| `completed→completed` | 36,416 | 18,894 |

因此，节点级总体 MSE/CE 在这个 source schema 下可被大量状态不变事件主导，不能直接等同于关键活动服务事件风险的预测收益。**注意：这些是 PR64 数据的标签密度；真正原 PR61 使用的是 PR59+PR60 源，不能把这些数字冒称“PR61 原训练比例”**，只能据此证明后续 DEV 评估空间内的目标结构，并指导检查原训练集标签。

## 5. 直接复核真正 pretrained map_on/off 的已归档预测

读取原 PR70 `service_timing.json`（记录明确使用 SHA 锁定的原 PR61 `best.pt`）原样按 12 个开发场景、原三个 seeds 1810/1811/1812 对 map_on−map_off 进行 scene-paired Brier 对照：

- **mean map_on minus map_off Brier = +0.00006526**；
- **95% CI [−0.00263083, +0.00214560]**，5,000 次按独立**场景** bootstrap，包含零；
- **12 场景中只有 5 场** map_on 的 3-seed 平均服务 Brier 更低；
- 原 751 risk rows/508 event labels 完全一致；Rule 同组 Brier 0。

这是**对 PR70 原真实 pretrained 预测结果的重复分析**，没有在 PR72 运行一次新权重推理。因此，结论严格限定为**混合 pickup/dropoff 的活动服务 Brier**：没有发现稳定 map_on 优于 map_off；不能由此否定地图在全节点状态 MSE 中的原 PR61 收益，也不能代替相位内校准。

## 6. 已执行的代码与可复现证据

- `experiments/world_model_integration/identifiability_pr72/pr72_risk_and_alias.py`：直接从 PR64 SHA 锁定 TRAIN/DEV 源和原 12 场景 BusinessSimulator 动作观察，抽取活动服务事件，统计图信息缺失与监督标签，并生成受控同输入异结果见证。
- `experiments/world_model_integration/identifiability_pr72/pr72_frozen_trained_arms.py`：原 PR70 已训练权重输出的场景/seed 配对统计，不加载模型。
- `experiments/world_model_integration/identifiability_pr72/test_pr72.py`：验证 phase 对应、mask 约束、原 graph 无服务 effective mean/active progress。
- `.github/workflows/pr72-phase-risk-input-information-cpu.yml`：自动下载和 SHA 核对原 PR64 全部 artifacts，执行 12 个真实 DEV 场景及 PR70 frozen-result 统计，归档日志、JSON、source HEAD、pip freeze 和 SHA256SUMS。
- **GitHub CPU 成功 Run [#43410871772](https://github.com/lieuyimo/wm/actions/runs/43410871772)**：10/10 单元测试；原始 PR64 12/12 场景，751/508 活动服务风险集，受控 graph 相同/Rule 输出不同；真实 pretrained 存档 map_on/off 配对。归档 [Artifact #14030205096](https://github.com/lieuyimo/wm/actions/runs/43410871772/artifacts/14030205096)。
- 已在最新 PR72 Run #43410871772 **真实验证完整 PR61 PyG history / entity / relationship / public map tensors 相等**，而原合法规则下一状态不同；无原数据修改。

## 7. 研究裁决与下一阶段的具体研究门槛

**GO**：明确解决模型“看不到重要公开服务进度与有效时长”的输入合同问题；保留当前 PR61 架构和已训练检查点供原模型可追溯性复核。下一次真正模型训练前优先验证能否复用既有 `business_transport.session_events.encode_query`、`state_v2` 公开 effective progress 特征与 PR54/PR59 关系图输入；不为获得指标而造新算法，不把新字段硬塞到旧 checkpoint 里冒称原模型。

**NO-GO**：仅靠扩增参数、继续盲目增加确定性场景、用无相位/混合事件标签训练、重新使用过早完成误判、按 751 条时间点假设统计独立、或把一组“合法 H20 rollout”当作预测改进和调度收益。

**下一步是否需要 GPU？当前不需要。** 必须先对照原 PR59+60 实际训练标签、既有公开服务进度语义、以及模型 head 的任务状态与 conditional active-session 风险集合，冻结信息对齐与评估协议，再决定是否开展定向再训练。没有盲测/线上调度收益声明，保持 Draft。
