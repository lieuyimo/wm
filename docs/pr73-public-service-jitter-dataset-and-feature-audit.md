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

同一已冻结机制已经在原 PR64 **全部 24 TRAIN + 12 DEV** 场景上完成 CPU 验证。CI 产物含源校验、完整合法轨迹、各相位样本分布、Rule 与 TRAIN-only hazard Brier、按 DEV 场景配对 bootstrap CI，以及完整公开 48d TRAIN/DEV 事件记录。完整数值见第 5 节。

**可科学主张**：原确定性合成服务机制不足以检验“学习世界模型能否预测不可确定服务事件”；启用已有服务 jitter 后，事实事件出现条件性不确定性，公开服务时长/有效进度能够提供预测信息。

**不能主张**：现有 PR61 checkpoint 已经预测得更好、对真实企业服务过程泛化、H20 递推准确、CP-SAT 调度收益、BLIND 改进，或新服务头已训练。

下一模型阶段应优先在原 PR59/61 架构外添加已公开验证的 48d `service-query-v2` 条件输入或复用现有 `SessionRuleDynamics` 事件编码路径，明确检查所有输入 known mask、训练按场景独立划分，并分别评估取样/交付事件 Brier、校准、合法递推及选择差异。原 16d PR61 检查点与新输入维度不兼容，不能直接当成“续训”。

PR73 保持 Draft，0 GPU、0 新神经训练、0 BLIND/TEST 访问。\n
## 5. 全部预注册 PR64 TRAIN/DEV 场景的 CPU 实测结果（正式数据质量门禁）

**最终成功运行**：[GitHub Actions #43450302135](https://github.com/lieuyimo/wm/actions/runs/43450302135)，含原 PR64 全部 24 TRAIN / 12 DEV 场景的配对两臂、SHA 锁源、每个活动服务的 48d 公开输入以及下一状态标签；[Artifact #14050126388](https://github.com/lieuyimo/wm/actions/runs/43450302135/artifacts/14050126388) 内含 `pilot.json`、完整原始 CPU 日志、10/10 测试、source HEAD、pip freeze、SHA256SUMS。**36 个场景都按原三个业务 regime 等量覆盖，未选择 DEV 高收益样本。**

| 原 PR64 全量 DEV (12 independent scenarios) | J0 固定服务 | J2 原生 keyed 随机服务 0–2 步 |
|---|---:|---:|
| 有效活动服务观察 (n) | 751 | **1,409** |
| 下一步真实事件 | 508 | **594** |
| 原 ProgressAwareRule Brier | **0.00000** | **0.28886** |
| TRAIN phase constant（Laplace）Brier | 0.21518 | 0.24199 |
| TRAIN station+phase+active elapsed hazard Brier | 0.03187 | **0.17665** |

TRAIN J2 共 **2,593 条**有质量和可见性约束的 public 48d 事件样本（其中下一步事件 1,106）；DEV J2 **1,409 条**（事件 594）。所有场景的源 train/validation 任务 seed 不重叠；数据分割严格按原 scene，不将一个场景的时间步分散到两组。

**相位计数：**

- J2 DEV pickup：**831** 条活动服务，**320** 个下一步事件；Rule Brier **0.28279**；TRAIN-public-hazard Brier **0.15795**。
- J2 DEV delivery：**578** 条活动服务，**274** 个下一步事件；Rule Brier **0.29758**；TRAIN-public-hazard Brier **0.20355**。
- 同机制 TRAIN pickup 1,553 条、601 事件；TRAIN delivery 1,040 条、505 事件。
- J2 DEV 统计有 **20** 组按当前公开 station/phase/effective active elapsed 分桶同时出现事件与非事件，表明随机服务时长在当前公开信息下不能由固定 0/1 规则精确预测。这个 coarse risk 混合是事件随机性诊断，**不等于可预测信息量精确数值**。

**独立场景配对统计**：与原 Rule 相比，TRAIN-fitted station+phase+public-progress hazard 在 DEV 的 **12/12 场景**具有更低的服务事件 Brier；按 12 个**场景**均等的模型−Rule Brier 均值 **−0.11852**，2,000 次独立场景 bootstrap 95% CI **[−0.14003, −0.09553]**。这个结果与逐活动服务样本加权的整体 Brier **0.17665 vs 0.28886** 是两个不同汇总权重，不应混用。

**对照解释**：当 J0 服务时长固定且先验可见时，Rule Brier=0，传统概率估计会稍差；当 J2 由既有 `_service_duration` 的 task/phase/station keyed jitter 引入条件服务不确定性时，原 Rule 不再完美。此时可观测 active progress 具有统计预测价值，TRAIN-only 风险估计的优势在本批 12 DEV 场景内一致。**这不是 neural model 提升证据**；这里只验证新的合成过程与公开事件监督确实产生了可利用的预测任务。

### 本阶段最终裁决

- **GO，数据质量与条件预测价值**：原内置 jitter 机制足够构成第一批有监督的服务事件随机动态；CPU 实测与概率风险 baseline 支持使用其生成有针对性的训练数据。
- **GO，训练接口前置**：既有 `service-query-v2` 48d 是可执行公开输入合同，已对原 24 TRAIN+12 DEV 逐会话抽取；可复用现有 `session_events.encode_query`，无需造新模型算法或修改原 PR61 旧 checkpoint。
- **NO-GO，立即扩大量或改 PR61 权重**：当前只完成数据与基线验证；尚无新模型参数、loss head、phase-specific 训练/校准或完整 H5/H20 评价。原 PR61 16d 图节点模型与新的服务事件输入不兼容；不能将其当作已支持这些字段。
- **NO-GO，BLIND 与调度收益**：开发验证没有对应真实外部测试、没有生成 BLIND、没有证明模型泛化到企业服务时间分布或多机器人部署。

PR73 CPU 验证通过；保持 Draft；后续训练应先冻结服务事件 head 与明确的公开信息/标签划分，执行 CPU 梯度 smoke 后再按用户既定工作流交付真实 GPU 训练。不将测试集用于挑选模型。

