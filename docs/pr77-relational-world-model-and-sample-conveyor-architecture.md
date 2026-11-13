# PR77 更正：面向实验室样本传送带/轨道输送线的关系世界模型扩展

## 背景纠偏

这里的“轨道交通”指 **laboratory sample conveyor / sample transport track**，即自动化检测流水线的样本载具（puck）、输送带/轨道段、分流装置、缓冲位、转接机械装置和分析仪工位，**不是铁路列车、火车站、信号联锁或列车调度**。本 PR 不采用 Flatland、铁路晚点预测或列车 head 作为样本运输研究主线。

已经存在的公开工程参考：
- Siemens Healthineers Aptio operator/academy：main lane、secondary/analyzer buffer track、T/L/U turn、stop/divert gates、sample pucks、analyzer/interface module；可核验结构为 <https://academy.siemens-healthineers.com/_/en-us/aptio-automation-siemens-system-overview/> 及 <https://doclib.siemens-healthineers.com/rest/v1/view?document-id=187086>。
- *Modeling, Analysis, Simulation and Control of Laboratory Automation Systems Using Petri Nets: Part 1. Modeling*（JALA 2005，DOI 10.1016/j.jala.2005.04.005）：实验室线性样本传送带、工作站与机器人资源，以及 Petri net 事件/资源建模；<https://journals.sagepub.com/doi/10.1016/j.jala.2005.04.005>。
- Roche 的实验室样本传输与布局案例 <https://diagnostics.roche.com/us/en/products/systems/cobas-connection-modules-sys-364.html>。这类产品和论文是系统需求与实体/事件建模参考，不意味着我们复现某一厂家的内部传输策略。
- 样本载具沿带段/区段运输、不同区段限制载具数目，可参考公开专利 <https://patents.google.com/patent/WO2025174543A1/en>；专利是结构证据，不是已经获得的模型源代码。

## 1. 我的世界模型在哪里？

**主干**：原 PR61 的 `PublicMapTemporalRelationalWorld`，4 帧公开历史 `GRU` + `RGCNConv` + 公开二维网格 `SAGEConv`，共 52,953 参数，真实 PR61 map_on/map_off × 3 seeds A100 权重由 PR68 SHA/manifest 独立归档；没有新训练。原 PR61 是**样本转运 robot/task/station/resource 的单步关系状态预测网络**，不能错误解释成经过验证的传送带预测。

**世界状态演化**：原 PR67 `NeuralTaskServiceRule` 将图模型当前活动任务的下一状态概率用于服务事件，复用 `WorkflowDynamics` 和 `check_business_transition` 执行原有合法机器人物流及任务工作流状态转移。PR77 的 `RelationalGNNSystemWorldModel` 使用 RHWM 已有的 `ForecastStep / ForecastBatch / ForecastParticle / StepDistribution` 公开全系统状态和多粒子未来分布，而不是只报单个服务完成事件概率。

**独立基线**：PR74 Deep Sets 只是事件模型 baselines，不是未来唯一主世界模型。PR72 已证实 PR61 图输入未完整编码公开有效服务进度/有效时长；PR73-76 又验证了随机服务与多步概率预测的挑战。应先改输入/标签，之后做真正 PR61 新图训练，与 Rule / TRAIN-only hazard / Deep Sets 对比。

## 2. 机器人与传送带是同一个实验室样本转运世界

原先错误地将 rail_train/rail_switch/rail_signal 视为另一个 `rail_transit` domain。**现在纠正为统一 `lab_sample_transport` domain**，共享样本任务、工位、资源和工作流。不能用两个相互割裂的世界模型处理同一个样本。

### 公开系统图 schema（非已经生效的模拟器）

| 层 | 公开图实体与关系 | 预测与调度要关注的问题 |
|---|---|---|
| 样本任务 | `task`（workflow、stage、deadline 等）| 样本状态、可达性、后续工位交付、等待与超时 |
| AMR/AGV | `robot` + `assign_robot_task` | 移动资源占用、任务分配、冲突与到达 |
| 传输载具 | `sample_carrier` + `carries_sample` | 哪个载具承载哪个样本，物理跟踪 |
| 输送段 | `conveyor_section` + `feeds_into` | 段占用、容量、滞留时间、输送速度 |
| 分流节点 | `track_junction`、`diverter` + `branches_to` | 转向/分流动作与共享冲突资源 |
| 等待缓存 | `buffer_slot` + `buffered_by` | 上游阻塞、积压、等待时长 |
| 换乘/对接 | `transfer_port` + `transfers_at` | 机器人与传送带交换载具/样本的同步与缓冲 |
| 业务工位 | `station`、`resource` + `serves_station` | 设备服务完成、空闲/故障、工位吞吐 |

候选动作包括已有的机器人任务分配，加上**拟支持的** `route_carrier`、`hold_carrier`、`set_diverter`、`release_buffer`、`transfer_to_robot`、`transfer_to_conveyor`。这些动作**目前只是类型化候选行动**，还没有任何已经实现或验证的带段控制器，不能将它们解释为真实可执行 API。

统一查询可以同时表示机器人的机器人分配动作和同一公开世界中载具在传送带上的分流动作：`ActionConditionedWorldQuery(history, action, horizon, particles)`。每个字段必须记录其公开时刻；提前公开的工位计划是合法输入，未来**实际**堵塞、停线、载具到达时刻不能当作现在已知。

## 3. 哪些部分能够沿用，哪些需要实际扩展

**保持不变**：论文的 sample/task/workflow/资源语义、候选行动→未来系统状态→候选评分的决策世界模型研究目标、多粒子概率递推范式、状态因果与知情边界、单步/多步评估协议、旧 PR61 A100 检查点的 SHA 身份。

**必须扩展并做真实实验**：
1. `PublicGraphHistory` 的当前 16d/4-node-type lab 编码与 `public_topology_for_graph` 的 3d 网格节点，不能表示 conveyor section/diverter/buffer/transfer_port。需增加**模式/实体类型识别、轨道节点和有向边**。优先参考已有 PyG `RGCNConv/SAGEConv` 的不同实体/关系类型设计，禁止自行造新的神经消息传递算法。
2. 预测 head 不能只输出 task 四态。需复用现有状态监督方式，增加输送段**未来占用、载具到达/等待、缓存积压、分流成功/堵塞事件、机器人-输送线交接完成**等目标。模型应预测统一完整样本系统状态，不是只预测传送带速度。
3. 合法转移需接入明确的**段容量、单向边、冲突/互斥、分流器切换、载具不丢失、转接工作站容量和任务后继释放**的独立离散事件仿真参考；可复用既有 `WorkflowDynamics` 任务生命周期，不得把现有机器人的二维运动合法性冒充输送线真实规则。先参考公开厂商结构与 Petri net 建模，并对照开放实现选出仿真方法。
4. 需要带真实输送线机制的 TRAIN/DEV、独立测试，再比较 Rule、Deep Sets、map_off/map_on GNN 的同目标预测，以及统一合法多步与调度性能。无此数据时只能做**接口验证**，不能宣称模型已能预测输送带堵塞/拥堵或具有跨运输模式泛化。

## 4. 本 PR 的实际交付、阶段门禁

- `world_model_core/contracts.py` 现扩展 **同一个 `lab_sample_transport`** 图域（不再注册 `rail_transit`），包含机器人+载具+带段+分流+缓冲+转接节点/边/动作。严格校验公开信息时钟、实体身份、图拓扑关系、同一预测历史时间单位、候选动作在当前已知世界中的引用。
- `CONVEYOR_READINESS` 明确：**公开图契约已建立，但**带段编码器、专项 head、真实离散事件/容量转移、已训练带段权重和事实预测验证尚未完成，因此 `require_trained_forecasting(CONVEYOR_READINESS)` 必须拒绝。
- `LAB_READINESS` 对原机器人-only 训练基线作历史证据记录，不得借此声称新出现的 conveyor entity 可直接进入原 PR61 模型。
- `RelationalGNNSystemWorldModel.from_original_archive` 才能载入原 PR61 的真正训练权重。缺 SHA 文件即硬失败；仅带 **UNTRAINED** 标识的可控测试 fixture 用于验证现有 RGCN/SAGEConv→合法实验室业务 ForecastBatch 接线，无真实 PR61 训练效果结果。
- 新的 CPU 单元测试验证在**同一张实验室图**中含样本任务、AGV、载具、轨道段、分流器、缓存位、工位交接，以及两种候选行动；拒绝事实未来信息、未知区段或非法外部 rail 行动。继承 PR67/PR68 原有 GNN/权重硬门禁，旧模型保持不变。

**决策**：GO（同一世界模型图语义可兼容两类样本运输设施）；NO-GO（只做接口不等于已做传送带仿真/预测、真实轨道兼容权重或调度收益）。未来必须由实际输送线/检测流程选择带段模型与合法离散事件参考，再判断采用统一图网络训练还是以原图主干迁移。

## 5. 修订后的真实 CI 验证

[最终 GitHub Actions #43470529697](https://github.com/lieuyimo/wm/actions/runs/43470529697) **SUCCESS，18/18 测试通过**；[原始日志、schema readiness 与环境 Artifact #14061255794](https://github.com/lieuyimo/wm/actions/runs/43470529697/artifacts/14061255794)。

本轮不再测试铁路列车，而是测试**同一实验室样本世界**中机器人、样本、puck/载具、输送段、分流器、缓冲位、交接端口、分析工位同时存在并由公开实体/关系图表示；分别能表达机器人任务分配和载具分流，严格拒绝未观测的未来堵塞或事后事件输入、未知节点、错误外部行动与混合时钟。

原 PR61 实际图骨架仍是 **52,953** 参数，原 PR67 业务合法转移与 PR68 原训练归档 SHA admission 单元测试全部通过。CI `contract.json` 明确 `conveyor_trained_inference_supported=false`、`conveyor_physical_legality_verified=false`、`actual_pr61_checkpoint_loaded=false`、`BLIND_TEST_read=false`。

这是完成了**兼容接口**的实际验证，不是完成了传送带动态预测训练。下阶段仍需现实可复现的 conveyer carrier 区段和工位对接动力学仿真，以及 TRAIN/DEV 事件序列和专门的 GNN 实体编码。
