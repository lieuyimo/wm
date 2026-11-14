# PR78：仅使用现有公开汇总的统计与代码级结构诊断

**研究范围**：本报告不使用任何私有 PR61/PR78 checkpoint、不读取 PR59 BLIND/TEST、不执行新的模型推理或训练，不改样本物理机制。原始六支数据来自 [PR78 GPU 执行方评论](https://github.com/lieuyimo/wm/pull/78#issuecomment-6879536110)，传统 hazard 与样本风险来自 [PR73 数据质量审计](pr73-public-service-jitter-dataset-and-feature-audit.md)，Deep Sets 和 12 场景成对统计来自 [PR74 实验报告](pr74-existing-deepsets-conditional-service-cpu-training.md)。本分析独立重新计算**仅基于已公布数值**的平均值、配对差、相位差和敏感性，不把缺失的逐场景数据虚构成新实验结果。

## 1. 同一公开 DEV 风险集

- **独立统计单位**：12 个原 PR64 DEV 场景（3 个 regime 各 4）；总共 1,409 个当前活动服务风险/594 个下一步真实事件。取样风险 **831/320**，交付风险 **578/274**。这些是场景内相关的逐时间步，不是 1,409 个独立泛化样本。
- 训练：24 个不同原始场景，2,593 风险/1,106 事件；训练、DEV 片段按原场景划分，不混用未来事实和 TEST。
- **模型选择**：全部六支 PR78 GNN 仅有原 PR61 冻结主干＋新2,689参数条件事件头训练，20 epochs，最佳 epoch 按同一个 DEV 场景宏 Brier 选择。因此以下 DEV 统计是**诊断**，并非独立泛化报告。

## 2. 六支原始数据及可重算统计

| arm | seed | best epoch | risk Brier | 12-scene macro Brier | pickup Brier | dropoff Brier |
|---|---:|---:|---:|---:|---:|---:|
| map_off | 1810 | 19 | 0.19218723 | 0.19343601 | 0.17650680 | 0.21473123 |
| map_off | 1811 | 18 | 0.19720440 | 0.19816063 | 0.17908821 | 0.22325039 |
| map_off | 1812 | 20 | 0.18814497 | 0.18934325 | 0.17146103 | 0.21213172 |
| map_on | 1810 | 19 | 0.19246271 | 0.19390402 | 0.17566261 | 0.21661648 |
| map_on | 1811 | 18 | 0.19305506 | 0.19387416 | 0.17462695 | 0.21954948 |
| map_on | 1812 | 20 | 0.18896227 | 0.19064675 | 0.17437671 | 0.20993212 |

**三种子样本均值与样本标准差**（风险加权 Brier）：
- `map_on`：**0.19149335 ± 0.00221190**。
- `map_off`：**0.19251220 ± 0.00453845**。
- 配对 `map_on - map_off`（按原种子同号匹配）：seed 1810 = **+0.00027548**，1811 = **−0.00414934**，1812 = **+0.00081730**。配对均值 **−0.00101885**，配对样本标准差 **0.00272458**；两个种子结果表明地图更差，一个种子显示地图更好。
- **留一原种子敏感性**：移除 seed1811 后 map_on−map_off 均值 **+0.00054639**（方向反转）；移除 1810 或 1812 后分别为 **−0.00166602**、**−0.00193693**。当前小幅均值地图优势**不稳定**。三次种子不等于三个独立场景，更不能做 36 个独立样本的置信区间或宣称有显著地图增益。
- 三组 map_on/off best epoch 都按原 seed 分别为 **19、18、20**。这一观察提示需要查训练损失/早停曲线，但**不证明必然欠拟合**；20 epochs 附近选优也可能是小数据持续收敛或 DEV 选择现象。

**对应已公布简单基线**：

| 方法 | risk Brier | macro Brier | pickup Brier | dropoff Brier |
|---|---:|---:|---:|---:|
| 原 Rule | 0.28885735 | 0.29511460 | 0.28279* | 0.29758* |
| PR73 TRAIN-only phase×station×public-active-elapsed hazard | **0.17665487** | **0.17659304** | **0.15795*** | **0.20355*** |
| PR74 已训练 Deep Sets | 0.18893717 | 0.19121147 | 0.17207059 | 0.21318654 |
| PR78 map_on 3-seed mean | 0.19149335 | 0.19280831 | 0.17488876 | 0.21536603 |
| PR78 map_off 3-seed mean | 0.19251220 | 0.19364663 | 0.17568535 | 0.21670445 |

* PR73 取样/交付 Brier 在原文仅披露五位小数，相位差的末位为**近似值**；PR78/PR74 使用更高精度公开数值。

**在同一风险集内的可确认效应量**：
- `map_on - Rule` 风险 Brier **−0.09736400**；图事件头能优于没有正确随机条件服务概率的原 Rule。
- `map_on - Deep Sets` 风险 Brier **+0.00255618**（约 **1.35%** 相对劣化）；`map_on - hazard` **+0.01483848**（约 **8.40%** 相对劣化）。
- `map_off - Deep Sets` **+0.00357503**；`map_off - hazard` **+0.01585733**。
- **12 场景等权宏 Brier**同样未改变排序：hazard 0.17659304、Deep Sets 0.19121147、map_on 0.19280831、map_off 0.19364663。注意这不是同一条数据的 risk-weighted metric。
- **取样/交付都落后**：map_on 相对 hazard 取样 **+0.01693876**，交付约 **+0.01181603**；相对 Deep Sets 取样 **+0.00281817**，交付 **+0.00217949**。按 831/578 风险样本加权，map_on 相对 hazard 总差约 67% 来自取样阶段、33% 来自交付阶段（因 PR73 相位数字四舍五入，这是近似份额），但**绝对不是证明某单一 phase 特征发生 bug**。
- PR74 已公布的 **Deep Sets vs hazard** 12-scene paired bootstrap：平均 Brier 差 **+0.01461842**，95% CI **[+0.00317812, +0.02477636]**，Deep Sets 仅在 **3/12** 场景更好；这是既有模型对照已经做过的场景级分析，不是本轮新计算的 PR78 CI。PR78 vs hazard / Deep Sets 的场景级 CI **无从计算**（缺六支 PR78 scene-wise 原始统计）。

## 3. 源码事实：当前 PR78 根本没直接监督高维关系特征

原 [PR59/61 backbone](../experiments/world_model_integration/sealed_pr59/pr59_dual_model.py)：
- 每节点 16d public feature + 四帧历史 `GRU`，两层 `RGCNConv`、可选 `SAGEConv` 公共地图；`node` **48d 隐表征**经 `core.node_next_head` 映射成 6d 节点输出。
- 对 task，原 6d 输出前四项是 four-state softmax，后两项是 sigmoid。地图分支位于节点状态头**之前**，对 PR61 本来的全状态 node loss 有独立价值检验，但不保证对服务完成 hazard 额外有用。
- 原 PR61 训练 `masked node MSE + 0.3 new-stage-event MSE + 0.1 task-status CE`，不是已经见过 J2 keyed jitter 的活动服务下一步 hazard。

新 [PR78 event model](../experiments/world_model_integration/gnn_pr78/pr78_model.py)：`logits = MLP([ original_next_node_6d , current_public_service_query_48d ])`。训练脚本 [PR78 frozen-head](../experiments/world_model_integration/gnn_pr78/pr78_pretrained_train.py) 将 `graph_world.parameters()` 全部 `requires_grad_(False)`、在 eval 模式固定所有 GNN 输出，并只训练 `service_event_head`。所以其实际检测对象是：

> 在 J2 的公开 48d 条件服务信息之外，**冻结的原 PR61 绝对 next-node 状态 6d 读出**是否为条件 H1 服务事件增加预测价值。

这**不是**“PR61 原 RGCN 中间 48d 关系表征，在新随机服务条件下端到端学习并被证明无效”，也**不是**证明世界模型整体资源/拥堵/任务状态预测无效。另有 [PR61 GPU 全状态评估](https://github.com/lieuyimo/wm/pull/61) 在其原目标的 map_on/off node-MSE 上显示过地图收益，不能把不同预测目标的证据混为一谈。

## 4. 源码机制诊断：当前 H1 服务完成事件不以地图几何为直接生成因子

本次继续核实 [原始仿真器 `lab_transport/simulation.py`](../lab_transport/simulation.py) `Simulator._service_duration` 及 `_handle_station`，并追踪 PR73 的 `BusinessSimulator`（继承这个 Simulator）和 `original_factual_episode`，确认 **PR73 的 J2 动态直接使用原物理引擎，而非另造新的服务时长预测任务**：

```text
private draw = SHA256(seed, sample_id, phase, station_id)
jitter = draw % (jitter_limit + 1)     # J2: jitter_limit = 2
service_duration = station.service_ticks + jitter
```

服务开始时固定抽出 `service_remaining`；之后在工位可用时每个 tick 减一，不可用时暂停；完成归因于**私有随机剩余时长、当前已服务进度和设备可用状态**。在原实现中，**二维/有向地图、路径长度、全局拥堵及其他机器人的空间关系没有直接进入已经开始服务的 `_service_duration` 或每 tick `service_remaining` 更新公式**。

这不是说地图和机器人关系对整个世界状态无价值，而是说：

- 路径拓扑、机器人阻塞、调度动作可以影响**什么时候到站、能不能开始服务、候选动作选择与资源争用**；一旦已对 `require_active=True` 的“正在服务”会话筛选，H1 取样/交付完成时间主要由**服务进度、工作站类别、相位、当前有效性和不可观测的既定随机扰动**决定。
- 传统 `station × phase × public-active-elapsed` hazard 已经针对这一生成机制直接建模，因此它在这个目标上比一个原本专门学习绝对节点状态/地图关系的冻结 GNN 拥有更强的**目标对齐**，是合理且可追踪的解释。
- PR78 的原 6d 绝对下一状态输出经过状态 softmax/sigmoid 及多任务训练，又在 J2 任务下冻结，并不一定携带对随机服务完成额外有用的信息。添加 `map_on` 也不能凭空提供 hash 私有随机源。这为当前 map_on/off 不稳定、两 phase 都落后于 hazard 提供了**业务机制解释**。
- 但 `sample_id` 与 scene seed 的 hash 在整个有限合成数据集中是固定伪随机而非统计意义上的严格独立随机变量，且动态任务进入活动服务的条件、工作站故障、容量与系统历史可能引入选择偏差；**不能据代码直接断言“地图与事件条件独立”已得到数据显著性验证，也不能计算理论贝叶斯误差下界**。本次是**生成因子代码审计+模型目标诊断**，不是对所有交互的形式证明。

**直接研究建议**：不要只拿 `H1 active service completed` 作为“关系图世界模型具有未来动态建模能力”的决定性指标。继续将 H1 事件概率用于合法状态递推，但若要检验 GNN/地图价值，应在**原始全系统公开状态预测**（任务运行状态/实际到达、工作站排队和资源占用、机器人阻塞/路径冲突、候选调度动作后的后继工作流释放）上分别做配对比较，并在 H5/H20 设计中将事件预测误差与运动/拥堵状态误差区分。复用 PR61 原状态 head 和 PR76 已有公开策略粒子协议，不自行造控制/调度算法；任何新评估仍须保证同已知公开外生条件、无未来隐私泄漏及相同候选动作预算。

这形成了比“图卷积参数不够”更明确的两个**独立待验证假设**：
1. **任务信息限制**：H1 活动服务完成目标自身对地图全局关系不敏感，地图优势应在运动与资源交互目标上检验。
2. **读出与迁移限制**：原 48d RGCN 关系表示先变为 6d 绝对状态输出、J2 训练时被冻结，所以新的服务事件 head 无法充分利用关系表征。

两个假设均有源码支持的动机，尚需要同公共场景的控制实验才能判断各自贡献；**当前证据没有给任何一个假设分配一个数值百分比责任**。

## 5. 现有数据的限制与下游门禁

### A. 诊断优先级

1. **真实统计已支持**：不应马上用同一种 6d frozen head 放大 GPU 轮数、参数或多次种子并宣称会追上基线；简单 public-progress hazard 优势在总体、相位、宏平均几个口径上均出现。
2. **输入-读出匹配是首个代码级嫌疑**：6d 原 absolute-state 输出已经经过 softmax/sigmoid 和 4-class 分配约束，可能压缩了能供服务完成事件直接使用的高维关系与拓扑信息。与 48d query 拼接未显示 map_on 稳定增益，但**尚无法从这些数值识别 6d 是否有任何正/负条件信息价值**。
3. **下一最小严谨控制**：在相同 TRAIN24/DEV12 数据和公开服务风险集训练一个**48d query-only 同系列标准 MLP**，对照当前 54d PR78 head，比较训练参数与输出复杂度并报告差别；二者都须用相同 seeds/epoch-selection 规则。其结果还未存在，不能推测。
4. **保留世界模型主干目标**：若 48d query-only 与54d结论表明原 6d 不增益，再用**现有 RGCN 48d 中间关系表征**作为标准任务 head 的输入做有出处的 readout/迁移学习控制，并审计原图下一步全状态损失、任务/机器人/站点/路径输出与调度敏感性。只在上述有证据时有限解冻 GNN，不新增消息传递算法或盲目扩大网络。
5. **H5/H20 是不同问题**：长窗世界状态分布有工作站/等待/路径相互作用，不能凭服务事件 H1 一个数完全否定，也不能凭 PR78 当前 H1 成绩证明已能预测或改善调度。PR76 单事实动作强制回放的合法性错配须在任何长窗评估中修复。

### B. 无法从当前公开数据作出的统计声明

- **不能**得出 PR78 map_on vs map_off 或 vs hazard 的 12 场景 paired bootstrap CI、p 值、统计显著性，缺 **6 × 12 同场景 Brier**；种子不是独立 scene。
- **不能**重新计算 PR78 ECE、NLL、概率区间校准或反事实占用/流转差异，缺公开的可审核预测摘要或在受控环境执行的统计计算；汇总 Brier 无法倒推出概率分布。
- **不能**识别 PR78 6d 图输入对 48d query 的边际信息量，缺同预算 query-only 的重新训练配对与可重复条件消融。
- **不能**证明模型欠拟合、数据合成有错误、真实 H20 稳定提升或样本调度收益；仅已知 best epochs 在 18–20、J2 公开服务风险集存在随机性。

**结论**：现有公开汇总+源码足以完成**配对种子稳定性、阶段误差、基线效应量与实际瓶颈位置**的可核查初步诊断；还不支持 PR78 的场景级显著性、图特征增量信息或 H5/H20 分布质量结论。后续先完成 [#82 12-scene H1 统计](https://github.com/lieuyimo/wm/issues/82) 和 [#85 表征诊断](https://github.com/lieuyimo/wm/issues/85)，不需要公开任何模型权重。
