# PR73：公开服务输入对齐与原始合成服务时长抖动机制 CPU 验证

**研究对象**：真实 `BusinessSimulator` 离散步服务动态与原 PR61 世界模型的下一步服务事件可预测性。继承 PR72 输入不可识别性审计；不修改旧模拟器、不覆盖任何 PR59/61/64/70/72 结果。

## 1. 原合成环境为何 Rule 对服务事件 Brier=0

真实服务时长由 `lab_transport/simulation.py:Simulator._service_duration` 定义：`station.service_ticks + SHA256(seed, sample_id, phase, station_id) % (jitter_limit+1)`。该机制**已存在**，而 `experiments/business_constraints/tests/test_business.py:fixture` 给基础站点设置 `jitter_ticks=0`、全局 `duration_jitter=0`。PR48 → PR55 → PR56 → PR59 原始来源均复用这个基础业务场景；PR55 slow_service 修改确定性的名义时长，不将该抖动调为正值。PR64 额外随机化的是设备故障起始时间，并未将服务时长 jitter 改为正值。

因此原 PR70 采用公开服务进度与可观测业务规则时，确定性服务事件集上的 Rule Brier 可以达到零；这是合成机制与输入信息共同造成的可预测性上限，不证明神经模型参数太小。

## 2. 只使用现有仿真机制的受控试验

`experiments/world_model_integration/synthesis_pr73/pr73_pilot.py` 调用原 `build_topology_business + private_outage_shift + BusinessSimulator + TrueCPsatBranch`，每个原始 PR64 TRAIN/DEV 场景重复两次：

- 对照 J0：仅设置每个 station 的 `jitter_ticks=0`。
- 扰动 J2：仅设置 `jitter_ticks=2`，即相同 `service_ticks` 加不公开的均匀整数扰动 0/1/2；同一 seed、任务、服务阶段、站点的哈希抽样可复现。
- 两臂保留原站点名义预计/先验时长、地图、机器人、容量、任务与公布时间、CP-SAT 规则，以及同一已预注册设备故障机制；不把服务私有真实剩余时间、随机数、未来异常时间或任务表作为模型输入。
- 源严格 SHA 锁定：PR64 TRAIN `4c1157777117f82f7deb44f98e1d8255e95611dfcb41c6dfff9e14dd23aee183`，DEV `31a36e0940db7afe015fe35689cb629102dea45e8d61123ed89250abc01d0b86`；每组 3 类业务场景等量覆盖。
- 仅在真实 `eligible_sessions(require_active=True)`、公开状态 `observation_t` 和当时真实合法 `action_t` 上建立样本；pickup 标签是 `next=in_transit`，delivery 标签是 `next=completed`，next label **不**在 t 时刻输入。
- 复用项目既有 `business_transport.state_features.build_business_features` 与 `business_transport.session_events.encode_query` 输出完整 48 维带 known-mask 的公开服务输入，保持 PR61 原 16d 网络及其旧检查点**完全不动**。新 48d 语义为将来针对性训练提供独立数据合同，不是声称旧权重已学过此输入。
- 除全量评分外，归档按真实场景分离的 `event_risk_dataset.train / developer_validation`，每条包括 48d 公共输入、phase、公开 tick、仅作分析的 station、下一步服务完成事件标签；**没有私有服务 jitter 抽样或未来故障日程输入**。
- 复用 PR63 已采用的 **TRAIN-only 离散时间风险集频率 / Laplace 平滑** 估计作为透明基线：按服务 phase 常量估计，以及按公开 station+phase+active service elapsed 分桶；在 DEV 上比较 Brier，按独立**场景**而不是相关时间步 bootstrap。

## 3. 首轮真实 CPU 数据

[PR73 实际已完成 CPU pilot #43449652606](https://github.com/lieuyimo/wm/actions/runs/43449652606)；[Source data + CI artifact #14048519834](https://github.com/lieuyimo/wm/actions/runs/43449652606/artifacts/14048519834)。该轮含原三个 regime 等量选取的 **12 TRAIN、6 DEV** 场景，各运行 jitter=0/2 两臂，10/10 测试通过：

| DEV (6 scenes) | deterministic jitter0 | keyed jitter0–2 |
|---|---:|---:|
| active public service risk rows | 370 | **681** |
| factual next service events | 252 | **293** |
| original Rule Brier | **0.00000** | **0.29809** |
| TRAIN phase-frequency Laplace hazard Brier | 0.21319 | **0.24451** |
| TRAIN station+phase+active-elapsed hazard Brier | 0.03105 | **0.18574** |

TRAIN keyed jitter0–2 记录 **1,260** 个合法公开服务会话及其下一步事件标签，DEV 记录 **681** 个；48d 输入宽度与 t+1 标签时钟均已在 CI 实际验证。三类场景中，PR70-style Rule 完美服务预测在扰动后失效，而使用当前公开服务进度的**非神经、非新算法**风险估计显著降低 Brier。该现象说明存在有用的公开条件预测信息，**并非噪声越大越好**。不过，这一结果仍属于小规模合成开发试验，不是模型已训练或在线调度增益证明。

## 4. 扩展样本与研究门禁

同一已冻结机制正在原 PR64 **全部 24 TRAIN + 12 DEV** 场景上执行，CI 产物有源校验、完整合法轨迹、各相位样本分布、Rule 与 TRAIN-only hazard Brier、按 DEV 场景配对 bootstrap CI，以及完整公开 48d TRAIN/DEV 事件记录。最终判读以该轮真实输出为准，不能根据 DEV 指标改变抖动、分桶或训练参数。

**可科学主张**：原确定性合成服务机制不足以检验“学习世界模型能否预测不可确定服务事件”；启用已有服务 jitter 后，事实事件出现条件性不确定性，公开服务时长/有效进度能够提供预测信息。

**不能主张**：现有 PR61 checkpoint 已经预测得更好、对真实企业服务过程泛化、H20 递推准确、CP-SAT 调度收益、BLIND 改进，或新服务头已训练。

下一模型阶段应优先在原 PR59/61 架构外添加已公开验证的 48d `service-query-v2` 条件输入或复用现有 `SessionRuleDynamics` 事件编码路径，明确检查所有输入 known mask、训练按场景独立划分，并分别评估取样/交付事件 Brier、校准、合法递推及选择差异。原 16d PR61 检查点与新输入维度不兼容，不能直接当成“续训”。

PR73 保持 Draft，0 GPU、0 新神经训练、0 BLIND/TEST 访问。
