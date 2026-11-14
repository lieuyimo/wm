# CDU-SCIFN AI4Opt.Prj：四方向文献核对与 `wm` 研究转向方案

**日期**：2026-11-15　**读取基线**：公开仓库 [PR78](https://github.com/lieuyimo/wm/pull/78)，当时 HEAD `2404395f2c40583a66501808ef3d65dcc5635bec`，Draft；main 分支仍明显早于研究 Draft 分支。研究方向包括问题定义、神经策略、世界模型、优化 Agent。**本报告是迁移研究与实施协议，不是新算法、已实现 FJSP/LLM 代理或 Sim2Real 效果的声明。**

## 1. 给定文献逐条核验

| 用户提供 | 核定书目与贡献 | 方法与可迁移性 | 公开源码情况 |
|---|---|---|---|
| Wang & Liu, `A Syntactic Problem Solver Learning Landscape Structures for Clinical Scheduling`, TEVC 2024 | DOI [10.1109/TEVC.2024.3378911](https://doi.org/10.1109/TEVC.2024.3378911)。2024 年 online，正式卷期是 **IEEE TEC 29(4):1313–1327 (2025)**。将临床检验任务解释为异构柔性作业车间调度，MIP 优化平均 TAT，分析不同局部搜索 move 下的 landscape ruggedness、plateau、optima connectivity，再选择适合搜索策略。 | #1 问题建模；#2 实例特征→求解器/局部启发式选择参考。不应误称它是标准可直接复制的现有 `wm` 模块。 | 本轮未查到经核实的作者官方可复现仓库。算法可作建模与实验方法参考，正式迁移前另核实现有公开实现。 |
| Archetti, Speranza & Garrafa, `Managing an automated clinical laboratory: optimization challenges and opportunities` | [10.1007/s40070-019-00097-2](https://doi.org/10.1007/s40070-019-00097-2)，**EURO Journal on Decision Processes 8:41–60 (2020)**。按战略、战术、运行层面分析真实 TLA 工位、设备、人员、检验、TAT。 | #1 域内约束/目标的可信业务依据，明确这是临床检验生产与转运共同构成的运营问题。 | 描述/分类论文，**不是神经求解算法代码库**。 |
| Clinical Chemistry 2026 Supplement `B-020 Efficiency Analysis ... Specimen Transport Logics and Track Layouts Using a Digital Twin` | [10.1093/clinchem/hvag086.353](https://doi.org/10.1093/clinchem/hvag086.353)，**2026-10-04 会议增刊摘要**。公开材料比较 random、greedy、look-ahead、critical-path 运输控制，离散事件数字孪生，平均 TAT/最小 TAT 比、busy/queue index。 | #1 问题与指标验证；#3 数字孪生实验设计。**此文的 track 是实验室传送轨道，不是铁道交通**。但用户已明确暂停传送带实现，此处只保留外部业务参考。 | 会议摘要，不视为完整可复现算法和正式实测全证据。 |
| Gebreyesus et al., Evolutionary Intelligence (2025) | `Deep reinforcement learning-based spatio-temporal graph neural network for solving job shop scheduling problem`，[10.1007/s12065-024-00989-6](https://doi.org/10.1007/s12065-024-00989-6)，2024 online，2025 vol.18, article 6。**动态图、Attentive-GIN 空间块、LSTM 时间块、门控融合、PPO**。 | #2 神经调度策略与时空编码；**本论文不是 Hyper-heuristic，也不是 `p(s'|s,a)` 世界模型**。其主问题是 JSSP，不可直接当作柔性设备+机器人转运的完整实现。 | 本轮未查到可核验的官方训练仓库；代码移植优先有官方源码的 Song 等 FJSP-DRL。 |
| Song et al. `Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning`（补充直接参考） | IEEE TII 19(2):1600–1610 (2023)，[10.1109/TII.2022.3189725](https://doi.org/10.1109/TII.2022.3189725)。操作选择与机器分配的联合动作，异构图神经策略。 | #2 更接近 FJSP 和可选设备，适合作第一版可复用神经策略。 | 官方 [songwenas12/fjsp-drl](https://github.com/songwenas12/fjsp-drl)，Apache-2.0，包含 env、PPO/HGNN、训练与测试；需要约束映射和协议审查。 |
| Chen et al. `Deep Reinforcement Learning Assisted Genetic Programming Ensemble Hyper-Heuristics for Dynamic Scheduling of Container Port Trucks`（补充） | IEEE TEC 29(4):1371–1385 (2025)，[10.1109/TEVC.2024.3381042](https://doi.org/10.1109/TEVC.2024.3381042)。DRL 选择/编排 GP 低层启发式，并报告 GP 表达式可解释。 | #2 **Hyper-heuristic** 正确方法来源，与 Gebreyesus 的 STGNN 是两项不同研究，必须分别复现/评价再决定结合。 | 可获取论文与方法描述；未在本轮确证完整作者官方代码，不得把“论文发表”当“代码可复用”。 |
| Liao et al. `Addressing corner cases in autonomous driving: A world model-based approach with mixture of experts and LLMs` | [10.1016/j.trc.2025.105456](https://doi.org/10.1016/j.trc.2025.105456)，2025 年预印本，**Transportation Research Part C 183 (2026)**。WM-MoE 处理稀有场景/缺失数据、专家路由、LLM 时间上下文。 | #3 罕见故障、OOD 评估与专家分解的**概念迁移**；不能直接把自动驾驶 MoE/LLM 预测器当成实验室服务时长模型，当前也没有需要 MoE 的实测证据。 | 本轮未确认作者可复用训练源码。 |
| Hafner et al. Nature 2025 | `Mastering diverse control tasks through world models`，[10.1038/s41586-025-08744-2](https://doi.org/10.1038/s41586-025-08744-2)，**DreamerV3**。状态/奖励/继续性预测、想象轨迹、基于模型训练 actor/critic。 | #3 世界模型必须具备 `(state, action)→future` 及长期规划/策略价值的**方法依据**。不是直接要求把 5 万参数 `wm` 改写为完整 JAX DreamerV3 或声称已经支持 RL 想象训练。 | 官方 [danijar/dreamerv3](https://github.com/danijar/dreamerv3)，MIT；框架依赖 JAX、完整 RL 工作流，若移植需严格划分研究复用与改造。 |
| `AgenticBB0-Bench(2026年1)`, arXiv:2610.12183 | 实际为 **AgenticBBO-Bench**，`A Closer Look at Agentic BBO: Benchmarking LLM Agents for Black-Box Optimization`，**2026-10-08** 首发，并非 2026-01。五类黑盒优化中 agentic 方法在原作者协议下四类胜过最强数值基线。 | #4 **有限客观评估次数、工具/语义消融、LLM 与数值算法公平成本比较**；不能将跨领域 BBO 成绩直接说成优于临床 CP-SAT。 | 官方 [lamda-bbo/agentic-bbo](https://github.com/lamda-bbo/agentic-bbo)，MIT，77 tasks/5 domains，host-controlled evaluation budget，isolated workspace，append-only trials，token/cost。 |
| `OPT-BENCH`、`OPT-Agent` | [Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1417/)，[10.18653/v1/2026.findings-acl.1417](https://doi.org/10.18653/v1/2026.findings-acl.1417)。20 个 ML + 10 个 NP-hard 任务，感知-记忆-推理迭代优化，原作者指出先进 LLM **仍未达到人类专家表现**。 | #4 feedback/memory/iteration、约束合法性检查和 solver 调用循环；不证明 LLM end-to-end 优于专门的 CP-SAT/JSSP solver。 | 官方 [OliverLeeXZ/OPT-BENCH](https://github.com/OliverLeeXZ/OPT-BENCH)，OPTAgent 目录包含 agent、journal、interpreter、validation。 |
| Sun et al. `Structural properties and an exact formulation for flexible Job-Shop Scheduling with AGVs`（补充最直接问题模型） | [10.1016/j.cie.2026.112496](https://doi.org/10.1016/j.cie.2026.112496)，**Computers & Industrial Engineering，2026-11-03 online**。将 machine operations 与 AGV transport 一并建模，含 AGV 对称性与下界。 | #1 **FJSP-AGV** 比单纯 FJSP 与实验室 AGV 转运现有业务更接近；作为领域联合建模参考，先验算哪些假设（例如三角不等式）符合我方有向/堵塞地图。 | 未确认官方实现，不能未经证明直接应用其对称性剪枝于动态场景。 |
| Reijnen et al. `Job shop scheduling benchmark: environments and instances for learning and non-learning methods`（补充评测底座） | [10.1007/s10472-026-10007-3](https://doi.org/10.1007/s10472-026-10007-3)，Annals of Mathematics and Artificial Intelligence 2026，统一 JSP/FJSP/online jobs 及 CP-SAT、GA、规则、FJSP-DRL 等对照。 | #1、#2 的**标准外部基准**。仅将标准 benchmark 用于外部基准，不把其简化物理机制强称为现有 `wm` 事实仿真。 | 官方 [ai-for-decision-making-tue/Job_Shop_Scheduling_Benchmark_Environments_and_Instances](https://github.com/ai-for-decision-making-tue/Job_Shop_Scheduling_Benchmark_Environments_and_Instances)，MIT。 |

## 2. 实际 `wm` 最新 PR 审计：可以直接复用什么、尚缺什么

**读取基线**：[PR78](https://github.com/lieuyimo/wm/pull/78) 是截至审计最新更新的研究 PR，完整继承 PR76 的公开合法 H5/H20 研究；PR77 的实验室传送带原型是**另一条暂停中的 Draft 分支**，不要混入 PR78。main 仍停留在早期已合并内容，不能用 main 来判断当前研究进度。

| 现有代码/证据 | 已有能力 | 四方向迁移的缺口 |
|---|---|---|
| `business_transport/dispatch.py:BusinessCPSAT` | OR-Tools CP-SAT 对当前可见任务生成候选，保持多阶段 pickup/dropoff、站点/资源 `AddCumulative`、机器人 `AddNoOverlap`、多方案解，原目标**先最大化当前选择数量，再最小化 makespan** | 尚不是完整 FJSP：**可选择的临床检验加工设备、每工序可选设备/加工时长及加工工序与 AGV 联合调度**尚未建立，不可仅改名为 FJSP |
| `business_transport/policy.py:BusinessPolicy` | 最近调度、CP-SAT、粒子合法 rollout、候选 scoring，`solver_seconds`、`particles`、`horizon`、候选预算已有显式边界 | 缺通用 `operator policy`/neural solver/Agent 在**完全相同候选集与预算**下的统一插口与代价核算 |
| `business_transport/workflow_scoring.py` | 任务剩余距离、服务工作量、资源 contention proxy、unfinished workflow 等成本分解 | 现 proxy 不等于真实 FIFO 队列；尚未统一临床全链路 TAT、急诊超时与完整加工工作流的业务指标 |
| `business_transport/world_model/contracts.py` | `ForecastBatch`、`ForecastStep`、`ForecastParticle`、`StepDistribution`，保持预测的**完整公开系统状态**、合法行动、任务/资源/机器人预测结构 | 预测粒子不是已校准的联合轨迹分布；需明确评估多时间尺度任务/资源/拥堵变化与动作反事实的决策价值 |
| `PR61 → PR78`，`gnn_pr78/pr78_model.py` | 原 **52,953 参数** GRU/RGCN/SAGEConv 预测整个公开关系节点下一状态；PR78 加新2,689参数服务事件 head。GPU 六种子 head-only 已完成，**权重私有**。 | 冻结原图、将48d关系嵌入压成6d绝对状态输出，才与48d service query 拼接，因此 H1 活动服务完成没有表现出优于简单 hazard 的图结构增益 |
| `PR73–76` | 同一真实业务仿真 J2 TRAIN24/DEV12 活动服务、服务事件 Brier、H5/H20 物理合法递推、事实动作失配审计 | H20 的单事实动作分叉不等于概率世界预测失败，需用**同一公开合法后续策略**比较预测分布，而不是幸存样本偏差 |
| `PR77` | 同一领域 AGV+实验室样本传送带实体/关系的未来公共契约 | 用户已暂停传送带研究；本次只参考 ClinChem 会议摘要，不启用传送带训练/模拟 |

**最后一次真实训练（PR78 执行方报告）**：DEV 12场景、1409 active risk，map_on 3 seeds mean Brier **0.19149335**，map_off **0.19251220**；Deep Sets **0.18893717**、TRAIN-only hazard **0.17665487**，地图分支不稳定。**这是一个条件服务事件目标下、冻结原6d输出读出的结论**；不能把它包装成“全关系世界模型无效”，也不能说已建立高级 AI4Opt 基准。

## 3. 统一问题定义：动态临床实验室 FJSP-AGV（拟建立，而非既有全实现）

令实验室样本/检验单为 `job j`，多阶段加工/预处理/分析为 `operation o_jk`，可用分析仪/工位集合为 `eligible machines M_jk`，有 AGV/AMR 运输任务 `transport u_jk`，状态包含当前任务与阶段、站点容量、机器人位置和设备可用性、拥堵/队列、已公开急诊与延期信息。

明确区分**临床检验加工工序**、**样本的 AGV 转运作业**、**工位的取样/交付服务**：当前 `wm` 主要建模的是后两者，尚未显式实现可替代分析仪加工的完整作业车间处理流程。先通过 #1 确认一个任务哪些分析流程有可选机台后，才可严谨地宣称 FJSP，而非套名字。

参考 Archetti/Wang/Sun 的建模方法，定义以下基础约束，不声称当前均已编码：
1. 同一 job 工序前驱/后继、释放时刻、截止时刻；可选设备资格与工序加工时长；
2. 单台设备容量、同一 robot 不重叠、到达/装卸/输送时间，合法地图运动路径与冲突；临床工作站状态和故障/停机；
3. 当前公开可观测信息与未来私有外生变量严格隔离；调度滚动决策避免窥看未来故障；
4. 固定**第一业务优化目标**：全流程样本 TAT `C_j - r_j`，以及急诊逾期、资源排队、机器利用率。不同权重在实验开始前固定，不混淆现有 CP-SAT 的 makespan 与平均 TAT。

**范围控制**：先 FJSP-AGV 与现有多阶段 AGV 转运，传送带/轨道布局作为后续扩展，不在目前强制实现。

## 4. 四方向如何协作（统一端口、不是四套仿真器）

```text
#1 公共 Job/Operation/Station/Robot/Resource/Topology + 业务目标/约束
                      |
       真实 BusinessSimulator + 当前可观察状态 s_t
                      |
      CP-SAT 合法候选集合 A(s_t) + 安全约束检查
         |                         |
#2 神经策略/可解释超启发式     #3 小型关系世界模型
  选择/排列算子、候选动作        p_theta(s_{t+1},事件 | s_t,a_t,history)
         |                         |
         +----> 同成本候选比较 <--+ H5/H20 多粒子合法状态递推
                      |
                 CP-SAT/业务转移最终校验
                      |
         #4 OPT-Agent 上层任务/工具/预算编排
     （在线或离线可选；不能绕过真实约束与统一预算）
```

**先统一公共状态和动作契约**：`ObservationV2 + Assignment`、候选解列表、`ForecastBatch`、真实任务成本、运行预算/模型调用计数。规划/策略都依赖同一事实仿真器，任何学习模型不能作为真值裁判或生成未校验物理动作。

### #1 柔性实验室调度建模（@all）

- 原 Archetti + Wang 分类业务任务、设备资格、前驱工序、站点缓冲、AGV 运输、设备异常，建正式可行性合同。
- 先在**现有 `BusinessCPSAT` 的真实转运子问题上**实现严格目标对齐（平均 TAT / 急诊延误），为原 CP-SAT 和新问题进行等价回归，禁止直接替换旧基线。
- 复用 [Reijnen 2026 统一 JSSP benchmark](https://github.com/ai-for-decision-making-tue/Job_Shop_Scheduling_Benchmark_Environments_and_Instances) 中 FJSP/online arrival/CP-SAT 作为**独立基准适配**。新加入加工设备选择后用 Wang 与 Sun 文献对照可行性/效率，并与原有 AGV 转运部分分层比较。
- 输出：数学约束、合法实例、目标计算器、与真实状态观测的 schema crosswalk、独立 benchmark 与现有多机器人案例一致性测试。

### #2 神经策略 / 时空图 / Hyper-heuristic（@鲨鱼辣椒）

- **第一个有代码的 baseline**：[Song FJSP-DRL](https://github.com/songwenas12/fjsp-drl)（Apache-2.0）HGNN+PPO：先在标准 FJSP 实例上对齐，之后适配 #1 的操作-机器可行选择。不同于原 `wm` 的 GNN 世界状态预测，策略直接输出合法 action 或优先级。
- **时空模型研究参考** Gebreyesus 2025：Attentive-GIN、动态 disjunctive graph、LSTM、门控融合+PPO。其代码未证实可直接复现，必须明确哪些层使用现有 PyG/PPO 公开实现、哪些是领域接线。
- **Hyper-heuristic 研究参考** Chen 等 IEEE TEC 2025 的 DRL/GP-GPHH：在**固定一组已存在**的最近/短剩余时间/急诊优先/CP-SAT 优先级规则上做算子选择，保留规则选择记录、被选择规则与局部约束/资源原因。**原 Gebreyesus 本身没有 hyperheuristic 模块**。
- 成本与安全：同候选解、同真实目标、同 wall-clock/solver 预算，比较 CP-SAT、传统调度规则、FJSP-DRL 与 operator chooser；对动作可行性做绝对门禁。
- 输出：可信基线复现（标准 FJSP）、适配到样本运输候选动作、可审计 operator traces、规则基线成本/质量曲线；不据注意力热力图宣称天然“可解释”。

### #3 世界模型（@葳蕤，本组的主要方法学优先级）

- **保留**既有 PR61 `RGCNConv/SAGEConv` 与合法公开全状态合同，采用 DreamerV3 的 **动作条件状态递推→想象后果→改善决策** 方法原则；如果借用 DreamerV3 实际模块须标注 MIT / JAX 技术边界，不直接声称同一个具体 Dreamer 网络被移植。
- **变更 H1 核心评价对象**：下一步服务完成 Brier 为辅助事件头，主要检测任务阶段释放、机器人到站/阻塞、工作站积压、设备占用、逾期/最终 TAT 的**完整系统公开状态**分布，以及候选动作改选后的真实成本/后悔值。
- 先解决 #85 的 6d 原任务状态读出瓶颈：区分 PR61 原 RGCN 高维节点/关系 embedding、原 6d 状态概率和 48d 当前服务 query。对同一业务任务/动作，做纯公开的 `query-only / frozen 6d / hidden graph / map on-off` 监督对照。**不能先增加模型参数或新造网络，再称为世界模型创新**。
- 对 H5/H20 使用现有 `ForecastBatch/StepDistribution` 粒子、`WorkflowDynamics` 物理合法化及 PR76 的“**每个概率世界使用同一套合法后续策略**”协议；反事实从相同已公开状态、共同种子/外生机制和 CP-SAT 候选池开始，评价分布评分/校准与真实调度候选后悔值。
- **“小参数 / 低成本”是需检验的主张**：原网络约5.3万、PR78新增约2.7k，至少比较 brier / multistep score / decision regret 相对于 **模型参数、训练 GPU-hours、端到端 wall-clock P50/P95、推理成本**，与公开条件 hazard、Rule、PETS、Deep Sets 对比，在每种方案完整的 CPU/GPU预算下报告质量-成本 Pareto。简单模型更优时不回避。
- 稀有设备故障、停机恢复与信息缺失场景可以借用 Liao WM-MoE **corner-case stress auditing / expert gating** 原则，但要先证明单一小世界模型在哪一类真实公开 OOD 工况不稳，再考虑采用已有专家/集成机制。**LLM/MoE 和 Sim2Real 都非第一阶段训练必要条件**。
- Sim2Real 要求**真实实验室数据来源**（到站/服务时长/设备故障/队列/传感公开延迟），目前只有合成仿真时仅能报告 Sim-to-Sim/OOD 泛化，不得声称已完成真实迁移。
- 输出：全状态+分布评分合同、固定场景分组/合法再规划、多步/候选价值实验与质量-成本统计。

### #4 智能体决策 / LLM Cascading / 神经符号融合（@智能体方向同学）

- **首选定位**：优化 `BusinessCPSAT`/可解释启发式及世界模型的**高层工具/预算/算法编排器**（例如在给定问题规模、工位可用信息与预算时选择 solver 配置或候选策略），不让 LLM 直接越过 CP-SAT 可行性或承诺毫秒级每 tick 决策。
- 复用 [AgenticBBO-Bench](https://github.com/lamda-bbo/agentic-bbo) 的 **host-controlled finite evaluation budget、隔离工作区、append-only objective journal、同初始试验、工具与先验消融、质量-美元开销**；复用 [OPT-Agent](https://github.com/OliverLeeXZ/OPT-BENCH) 的反馈/记忆/迭代循环。具体代码适配需保留 MIT/原许可证条件。
- 独立研究对照 `CP-SAT only`、`同预算固定启发式选择`、`LLM-direct`、`LLM+可行性工具`、`小模型优先/大模型兜底 cascading`。Cascading 是**拟验证工程设计**，不是这两篇原文已经在临床调度中证明的实现。
- 任务评价必须包括：解可行率（100%作为目标但报告真实失败）、最好可行目标值、评估次数、搜索速度、API token/费用、调用次数、P50/P95 决策墙钟，以及无效提议/幻觉约束率；对于在线超时或约束失败，回退经过验证的固定 CP-SAT/规则。
- 输出：API/tool contract 和安全审计、Agent benchmark 与传统/神经基线质量-成本 Pareto。不得引用 AgenticBBO 在其他五领域的 4/5 优势，推断其已打败我们的 CP-SAT。

## 5. 里程碑和门禁（优先 #1+#3，后 #2，再 #4）

| 门禁 | 本轮最小工作 | 必须保留的负对照和报告 | 放行下阶段条件 |
|---|---|---|---|
| **G0. Problem contract** | #1 确认现有 AGV 多阶段 transport 与将来可选分析仪加工的边界；列 1 份数学模型、可行性审计与 KPI schema | 原 `BusinessCPSAT`、nearest、原真实 BusinessSimulator；不更换旧种子/约束 | 对已有样本场景 replay 可合法等价；公开不声称现有系统已是完整 FJSP |
| **G1. Full-system WM audit** | #3 审阅 #85 表示瓶颈，构建 task/robot/station/resource 全状态 h1/h5/h20 预测评分协议，保留原物理合法器 | Rule / TRAIN-only hazard / Deep Sets / PR61 map_off/on，真实仿真环境；按场景拆分 | 有可评价的动作条件**全系统未来分布**、统计单位和预测质量/运行成本证据 |
| **G2. Neural policy** | #2 在 Song 官方 FJSP 数据+公开实现复现，再尝试共用我方 CP-SAT 候选动作 / 可解释规则 | 传统 dispatching、CP-SAT 固定预算、消融无时间/无图 | 调度解合法且与传统规则相比有报告的实际质量/成本差异 |
| **G3. World-model-aided planning** | #2+#3 在同候选、同 wall-clock 预算下比较世界模型辅助排序对真实模拟 TAT / 急诊迟延 / 资源拥堵的贡献 | 原 CP-SAT 原分数 vs 同世界模型候选重排 vs 近邻与 heuristic | 预测→候选排序→真正执行收益因果链在独立场景上可检查，不能只比较模型分数 |
| **G4. Agentic optimizer** | #4 复用两个公开 agent benchmark 的**有限预算**协议，选择 solver / 参数 / heuristic；保证所有行动 CP-SAT 校验 | no LLM、直接 LLM、LLM+tool、小大模型 cascading；成本与 token | 在相同预算下有可验证目标价值或成本 Pareto，若无则仅保留高层解释/实验工具 |
| **G5. Sim-to-Sim/OOD；可选 Sim2Real** | #3 综合规模、到达率、设备异常、拓扑和服务时长分布外测试；有真实日志时才开展 Sim2Real | 测试从未用于训练/选型，未知未来事件不作为输入 | 真实数据/权限/量纲/隐私充分时才宣称真实实验室迁移 |

**统一评估**：全工作流平均 TAT 和急诊超时/任务完成率是业务优化指标；次要指标包括设备利用率/工位排队/机器人运动冲突、H1 事件校准、H5/H20 系统状态 proper scoring 和候选决策 regret；另报告训练成本、模型大小、P50/P95 决策延迟、调用/token费用。比较要有统一求解器候选集与合法性、相同随机外生条件/统计单位/软硬时间预算，防止用不同 objective、不同场景或删失失配样本做虚假改进。

## 6. GitHub Project 与代码管理

- **保留原 PR78 Draft 与 #79–#85 历史证据**，不得合并过去未评审分支或重写旧研究结果。此文档是研究方向调整，不是在 PR78 偷偷改 solver 行为。
- 建立 AI4Opt 一级路线 Issue 并链接 #79，#1/#2/#3/#4 各建独立 scoped issue，依赖用清晰的 G0→G1/G2→G3→G4 规则；按照现有 GitHub Projects 板字段管理（当前连接尚不支持直接编辑 Projects v2）。
- 训练的原 PR61 和新增世界模型、神经策略、Agent adapter 的权重/优化器状态都**只放受控私有存储**。不能放公开 Git、Release、PR 附件或公开 Actions Artifact；GitHub 可以存源码、公开标准 benchmark、审查后的脱敏指标和哈希摘要。
- 现有 **PR77 实验室样本传送带**仍暂停；ClinChem 2026 的“track layout”仅作为外部业务参考。学术论文、答辩、申报若要求 Sim2Real，需要独立真实数据，不可用纯合成报告代替。

**核心研究分工**：
- #1 @all：临床业务和 FJSP-AGV 的数学模型、目标、可行性与标准基准；
- #2 @鲨鱼辣椒：有公开代码的 FJSP-DRL/时空图神经调度策略、可解释 operator selection；
- #3 @葳蕤：**主线**小型动作条件关系世界模型、H5/H20 全系统状态预测、候选后果/泛化/成本；
- #4 智能体方向：有限预算 agentic solver 编排、神经符号安全工具、LLM cascading 成本评估。

**本阶段最终裁决**：**GO** 改研究框架和任务定义，**GO** 复用现有 CP-SAT/BusinessSimulator/PR61 关系世界模型，**GO** 各方向独立的有公开依据基线；**暂不 GO** 无实测依据的强制传送带、整套 DreamerV3/MoE/LLM 端到端移植、大规模重训或 Sim2Real 性能声明。
