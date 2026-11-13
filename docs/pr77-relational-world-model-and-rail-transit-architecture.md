# PR77：世界模型主线在哪里，以及向轨道交通预测扩展的架构边界

## 最终定位

研究主线是 **action-conditioned relational world dynamics**，即给定当前公开系统状态、过去已观测历史、公开运输拓扑与候选行动，估计**下一时刻系统公共状态的条件分布**，在合法状态转移下进行 H5/H20 多粒子 rollout，为现有调度器提供候选动作的未来状态及风险读出。

\[
P_\\theta(S_{t+1}, E_{t+1}\\mid S_t,A_t,G,H_{\\le t}),\quad
\\hat{S}_{t+1} = \\text{LegalProject}_{\\mathrm{domain}}(S_t,A_t,\\hat E_{t+1})
\]

**世界模型不是单独的 Deep Sets、单个服务事件头、单个预测评分，更不是 CP-SAT 优化器**。本仓库已经有完整的系统状态世界模型合同和合法业务推演，但关系图主干与通用世界模型接口并非同一条稳定线上调用：

| 层 | 已存在的真实代码 | 事实与责任 |
|---|---|---|
| 公开关系图历史 | `PR54 graph_public_features` + `PR67 PublicGraphHistory` | 时序 4 帧；节点实体、任务、机器人、工作站；候选动作关系和公开地图；单 tick 公开因果边界 |
| **主模型**：图卷积世界状态预测 | `experiments/world_model_integration/sealed_pr59/pr59_dual_model.py:PublicMapTemporalRelationalWorld` | 原版 **GRU + RGCNConv + SAGEConv**；PR61 真实 A100 的 map_on/map_off × 3 seeds、52,953 参数；任务公共 next-status 和新任务计数预测 |
| 神经状态到合法世界动态 | `PR67 NeuralTaskServiceRule` → `ProgressAwareRule` + `WorkflowDynamics` | 使用任务状态分布决定可观测的活动服务完成；保留工作站容量、机器人运动、后继阶段释放与合法检查；不是全神经端到端仿真器 |
| **世界模型规范** | `business_transport/world_model/contracts.py` | `ForecastStep`、`ForecastParticle`、`ForecastBatch`、`StepDistribution`，粒子输出整个可观测状态，不只是服务事件概率 |
| 原世界模型运行接口 | `business_transport/world_model/hybrid.py:HybridSystemWorldModel` | 原有 RHWM-v1 多粒子 rollout 接口，但支持 Rule/V4/传统 survival；之前**未把 PR61 GNN 接成首要 backbone** |
| Deep Sets | `lab_transport/business_session_learning.py` | 独立的业务会话完成事件概率基线；PR74 CPU 训练，PR75/76 在 J2 随机服务场景验证，不取代图关系主模型 |
| 具体业务调度 | CP-SAT、nearest | 提出候选行动/后续策略，不属于学习世界动态本身 |

### PR77 的实际代码整合

`business_transport/world_model/relational_gnn.py:RelationalGNNSystemWorldModel` **直接复用**原 PR59/61 `PublicMapTemporalRelationalWorld` → PR67 `NeuralTaskServiceRule` → 原业务合法校验，并用**既有 RHWM** `ForecastStep / ForecastBatch / ForecastParticle / _distribution / _seed` 输出整个可观测世界与粒子分布。

- `predict_next(public_state,legal_assignments)` 返回完整合法下一业务公共状态与有分相位语义的服务事件，PR61 GNN 是预测主干。
- `rollout(...,continuation_policy,horizon_ticks,particles)` 每个粒子维护自己实际逐步递推的原 PR54 **4 帧历史**；第 1 个行动由调用者提供，未来行动必须由**本粒子当前预测状态**的公开策略生成，而不能偷用真实世界后续行动。
- 可用于真实研究推理的唯一入口 `from_original_archive`：加载并检查 PR68 原始 **6,661,520 bytes / SHA256 `919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14`** 的 PR61 训练包、完整 manifest、map_on/off 与 seed 的具体 best.pt；没有文件即硬失败。**本 CI 不存在这个权重文件**，所以 PR77 的真实模型性能没有被重新测量。
- `_for_untrained_contract_test_only` 仅用于 CPU 测试真正的**图卷积网络结构**到既有合法世界 ForecastBatch 的维度、路径与状态合法性；输出显式带 `UNTRAINED`，不当作训练权重结果或 H20 预测改进。
- 不改 PR61 的 16 维输入合同、不变更其 52,953 个模型参数或历史 checkpoint，不合并旧 PR、不打开 PR59 BLIND/TEST。

## 轨道交通到底兼容到哪里

**可兼容的是世界模型范式和可复用的神经图结构，不是同一份已经训练的权重、相同业务 head 或相同状态约束。**

`world_model_core/contracts.py` 已添加跨领域的公共状态与候选动作规范（没有训练或声称新算法）：

| 共性 | 试验室样本转运 | 轨道交通列车运行预测 |
|---|---|---|
| 实体图 | robot/task/station/resource | train/track_section/station/switch/signal/platform |
| 拓扑图 | 可通行网格、工作站与资源关联 | 有向区间连接、道岔、站台、冲突关系 |
| 公开动态观测 | 分配、任务状态、有效服务进度、占用、设备状态 | 列车位置/速度、公开时刻表、延误、区间占用、公开信号状态 |
| 候选行动 | robot→task 分配、保留 | hold_train / route_train / set_switch / adjust_departure |
| 输出 head | 任务状态、服务完成、资源状态 | 区间占用概率、到发时刻/延误分布、堵塞/冲突风险 |
| 合法物理/业务转移 | `WorkflowDynamics`、`check_business_transition`、PIBT 等 | **尚需接入真实轨道仿真/联锁约束**；不得沿用网格机器人碰撞规则代替列车行车规则 |
| 多步状态与评价 | 取样、交付、等待、任务超时、后继阶段 | 预测未来列车/区间/车站状态、晚点传播、资源冲突、调度响应 |

新接口支持公开时间的逐属性审计：**未来时刻表可以在现在已经发布时作为公开规划输入，但未来真实晚点、隐含故障日程、私有仿真计时器不能提前泄漏到 t 时刻。** 同一个公共图 schema 允许不同 `tick_seconds`，但不允许在一条预测历史中偷偷混用不一致时间尺度。

### 当前不能复用的部分：必须明确

1. PR61 `node_features` 只有 robot/task/station/resource 四种硬编码节点，16d；坐标依据当前网格尺寸归一化；`RGCNConv` 的关系类型仅为样本转运六种语义；`SAGEConv` 的 `map_x` 固定 3d 网格节点。**当前 52,953 参数的训练权重不能直接读取 rail train/track/signal 节点**。
2. PR61 `node_next_head` 的任务四分类只适用于样本工作流，并不能表示列车延误秒数、站台 ETA 或占用分布；未来轨道数据需要自己的观察编码器/目标解码 head（是否共用图关系主干，需要做迁移学习实验才能确认）。
3. `HybridSystemWorldModel` 与 PR77 的实际关系图业务桥仍要求 `ObservationV2`，原 `check_business_transition` 是实验室机器人业务安全约束，不能直接当作铁路联锁、信号和区间冲突安全验证。
4. 轨道交通领域没有训练集、合法状态适配器或真实测试证据。本 PR 只提供可以表达线路/列车/行动/公开历史的跨域**contract**，并显式拒绝把这个合同宣称为已具备训练预测的轨道世界模型。

### 后续接入轨道任务的最短技术路线（引用已有方法，不发明新算法）

- **首先定义轨道任务**。若仅是逐车站平均晚点或小时级流量预测，直接参考 <https://arxiv.org/abs/2510.01262> **RSTGCN**（IEEE T-ITS，2026；有[公开铁路数据](https://github.com/KoyenaChowdhury/RSTGCN)）及已发表的 *event-activity network + spatiotemporal GCN* 研究（Computers & Industrial Engineering，2026）；这类单纯交通预测本身未必需要 action-conditioned world model。
- 若是 **候选行车调度动作下的未来列车/区间状态**，可优先接入现有开源 **[Flatland Railway Environment](https://flatland-association.github.io/flatland-book/)** 的可复现轨道、道岔、列车调度与故障机制，将`PublicWorldFrame`通过 **RailAdapter** 翻译为图卷积模型的标准张量，并由 Flatland/公开行车规则验证合法状态，保持原世界模型的单步→粒子 rollout→候选动作评估模式。Flatland 是研究仿真，不等于实际铁路联锁安全认证。
- 轨道预测与实验室样本转运首先分开训练，**共享架构形式，分别训练/校准领域 head**。之后有跨领域数据和实验时才比较共享主干、迁移训练、独立训练的有效性。不能预先宣称 PR61 已训练权重具有真实列车预测泛化能力。

### 研究门禁

当前阶段优先继续以**PR61 关系图世界模型为主线**；先修复 PR72 证明的公开服务信息缺失，再使用 PR73 真正随机服务数据与 PR74 Deep Sets/传统 hazard 做公平对照。任何新模型训练需要保留原 PR61 checkpoint，不将扩维后的不同输入解释为同一 checkpoint 续训。

轨道领域保持**接口已预留、训练和验证尚未开展**；不能把未来轨道业务改成一套虚构的固定速度规则来制造看似完成的测试指标。
