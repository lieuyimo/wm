# PR78：回到原始样本转运关系世界模型，修复公开信息与事件训练目标

## 阶段范围与分支

在原 [PR76](https://github.com/lieuyimo/wm/pull/76) HEAD `e79205e3441eee43dba3142189042d47a47d9a76` 上新建 `research/pr78-original-gnn-public-service-event-alignment-20261114`，**故意不继承 PR77** 的传送带/固定轨道扩展。传送带研发已明确暂停，本轮只修复原 AGV/AMR 实验室样本转运世界模型遗留问题。

不修改/替换原 PR61 A100 的 map_on/map_off 六份 SHA 存档；不打开 PR59 的 sealed BLIND TEST；不修改原来的多机器人 CP-SAT、PIBT、仿真服务时长实现；不新造世界模型算法；GPU 训练由独立获授权的执行环境负责。

## 1. 经过事实审计的遗留问题

1. **PR72：输入信息不可识别。** 现有原 PR61 `node_features` 有 16 维机器人/任务/工作站/资源输入，但没有编码已经公开的 `service_active_elapsed`、`service_paused_elapsed`、`observed_effective_service_mean`。PR72 构造的受控反事实在原 PR61 当前图、关系图、4 帧历史与地图特征完全相同的情况下，因服务有效时长不同而合法下一状态不同。**单纯加大原模型参数无助于恢复未输入的信息。**
2. **训练目标不对齐。** PR59/PR61 采用原 `supervised_losses` 的 next-node mask MSE、next-visible event MSE、task status CE：输出对所有任务下一**绝对状态**的概率，而 PR67 真正调度事件采样需要的是**已开始且正在有效服务**的 `P(本步完成 | 当前公开业务进度, 状态, 行动)`。原 PR64 同标签格式的 TRAIN 样本中 97.24% 相邻任务状态不变，但**这是 PR64 同格式诊断，不是原 PR61 PR59/60 训练集的分布**。
3. **随机服务条件下 H1 预测与长期任务状态。** PR73 直接启用原有 keyed jitter=2 后，原 Rule 在 DEV 的服务事件 Brier 0.28886、TRAIN-only 离散风险模型 Brier 0.17665、实际 CPU 训练的 Deep Sets Brier 0.18894。PR76 在事实动作/同外生边界下显示 H5 活跃服务起点的小幅改善，但 H20 因随机服务分叉经常失去事实动作合法性，因此**尚不能从幸存轨迹宣称任一模型 H20 事实状态预测领先**。
4. **PR77 的关系世界模型调用链只是结构接线。** 52,953 参数原 GNN 的预训练 A100 权重只在外部 SHA 归档，GitHub CI 未实际加载；原 PR77 工程基线不证明训练后的图模型在新的随机服务场景下有收益。应回到关系图预测主干做可验证的监督目标与分布实验。

## 2. PR78 做了什么，没做什么

### 现有主干完整保留

`PublicMapTemporalRelationalWorld` 保留原 **GRU(4帧×16d) + PyG RGCNConv(原6种业务关系) + SAGEConv(3d public grid cells) + next-public-node/event head**，原 52,953 个参数、原 map_on/off 等参结构和已训练权重身份不变。

### 修复信息别名和事件标签

- `experiments/world_model_integration/gnn_pr78/pr78_model.py:PublicServiceGraphHistory` 直接调用原 PR67 `PublicGraphHistory`，完整继承原 4 帧实体历史、关系边和地图张量；**只另外**为当前正在有效服务的 task 节点绑定来自原 `business_transport.session_events.encode_query` 的 **48 维公开条件服务查询**。
- 48d 已包含并用显式 known-mask 表示：服务墙钟/有效/暂停进度、样本任务优先级/截止时间、工作站公开有效服务历史、设备/容量/占用与竞争信息。任何未观测的未来故障、实际服务剩余倒计时、未发布外部任务、私有 jitter 哈希都不能出现。
- 其他非活跃任务的 `service_public_query` 严格置零、`service_risk_mask=0`；**不将等待中尚未开始服务的任务作为事件负例**。
- `attach_factual_next_service_labels` 在输出当前图**之后**读取真实 `t+1` 的原模拟器公共状态，只对当前有效服务打标签：pickup→`in_transit`，delivery→`completed`，非活动任务完全遮蔽。严格要求相邻公共 tick、任务 ID 与 phase 对齐。
- `PublicMapRelationalServiceWorld` 复用**原 PR61 整个 GNN**先获得下一节点状态分布，再用一个**普通 48d public-query + 6d GNN-node output → MLP → BCEWithLogits** 条件服务完成 head（新增 2,689 个参数）。这是标准监督分类头，不声称新的世界模型算法。新 head 有自己的权重，原 PR61 权重**从未学习过新增 48d 输入**，不能把原 checkpoint 直接冒充这个模型的训练结果。

### 真实数据与梯度回归

- 使用真实、SHA 锁定的 PR64 TRAIN/DEV 原场景，调用现有 PR73 `original_factual_episode(..., jitter=2)` 重建业务公开状态及真实 CP-SAT 行动；不生成新模拟物理机制。
- 原 3/3 TRAIN/DEV、三类业务 regime 等量的 CPU 验证中：**map_off 和 map_on 各执行 12 次真实 BCE 梯度更新**，完整 PR61 GNN 主干 52,953 参数原样保留，新增 head 2,689 参数，map_on/off 参数量相同。所有该轮数据是**随机初始化的测试模型**；相关 DEV Brier 只用于观察训练和评分程序能跑通，**不能当成 PR61 已训练后的实验成绩**。
- 原 PR72 信息碰撞的单元回归证明：不同当前公开服务预计时长仍导致旧 16d 图及 GRU 历史和地图张量**完全相同**，但现在新 48d service sidecar **确实不同**，Rule 的合法未来状态也不同。
- 服务 event BCE 的梯度传入新增 head、原 RGCNConv **和 map_on 的 SAGEConv**；原 PR67 业务合法转移、PR59 PyG 双图 batching、PR68 缺少权重时明确硬失败的旧测试继续保留。
- `pr78_data_gate.py` 额外验证全部原 PR64 **24 TRAIN/12 DEV** 的有效服务事件是否与 PR73/PR74 原始 J2 审计**完全一致**：应为 TRAIN 2,593 条、1,106 次完成事件；DEV 1,409 条、594 次完成事件，并核实 pickup/delivery 分相位、完整图张量、标签下一 tick。只有这个 CPU gate PASS，才能授权后续 GPU 阶段训练。

## 3. 以原 PR61 **训练权重**为基础的正规训练门禁（未执行）

`gnn_pr78/pr78_pretrained_train.py` 是**待执行入口**，不是完成实验。强制提供原 PR68 `original-pr61 archive`、`checkpoint_manifest.csv` 及 PR64 TRAIN/DEV/quality；利用现有 `load_original_models` 确认6份原 PR61 权重均为已归档 A100 best.pt、训练参数/来源和 SHA 全匹配。错误、缺文件或 CUDA 不存在则**直接报错，不回退到随机初始化**。

预注册的第一训练阶段：map_off / map_on × seeds 1810/1811/1812 共6支，原 52,953 参数 GNN 主干**完全冻结**，只训练新 2,689 参数的条件服务 head，20 epochs，Adam 1e−3，batch 16，选择开发验证的 **12 场景 macro Brier** 最优 epoch。保存每个 best.pt 的原主干 state_dict、训练服务 head、优化器、随机状态、开发结果、训练历史、源文件 SHA、完整 Python/torch/CUDA/pip freeze 环境，不覆盖原 A100 检查点。

这一阶段能直接检验“在相同随机服务源、同等公开信息与监督目标下，原训练关系 GNN map_on/off 是否能胜过既有 Deep Sets（Brier 0.18894）及 TRAIN-only 公开条件风险模型（Brier 0.17665）”。如果达不到这个必要门槛，继续对关系图加参数或直接跑 20 步不会自动增加研究价值；应先看相位校准、语义映射及是否需要有依据地解冻原 GNN 训练。

## 4. 恢复真正的多步概率世界模型验证

训练后不能只比较一条神经采样轨迹与唯一一条随机事实轨迹，也不能删掉 H20 分叉后动作不合法的轨迹再宣称准确。PR76 已确认这种评估协议会导致严重的**未来动作支持不一致**。

应先复用现有 PR66/67/75/76、PETS 粒子 rollout、公开规则与同一合法继续策略，在**同一公开初始状态和外生条件协议**下比较：

- H1：pickup/delivery 分相位服务事件 Brier、log loss、ECE、12 场景配对不确定性，Rule/传统 hazard/PR74 Deep Sets/原始 PR61 map_on/off/新条件 GNN。
- H5/H20：粒子分布预测的样本状态、完成阶段/等待阶段/工作站积压/后继任务释放时刻的校准和 proper scoring；不同预测世界必须用**相同公开合法后续策略**，不要强制同一条已实现事实后续动作在所有随机粒子世界合法。
- 任务与设备外生到达必须按当前公开可观察边界提供，不泄漏未来故障、未来服务私有计时器或事实 stage2；原 PR71 的阶段到达条件必须保留。
- 最后才验证 CP-SAT 候选动作选择是否发生可解释的改进和真实环境决策收益。**只有预测分布改善且带来稳定决策差异，才能支持决策相关的世界模型学习**。不以任何指标异常触发未经记录的无限制调参。

## 5. 研究裁决

**GO（CPU 验证与工程修复）**：原 GNN 已拥有针对当前公开服务状态的完整 48d 数据入口、下一步真实活动服务事件标签、梯度可达路径，保留原关系/地图模型与权重身份。下一阶段进入原 PR61 **真权重**的目标 head 训练（GPU 由另一个环境执行）。

**NO-GO（尚无证据的主张）**：本轮没有完成新 GNN 训练、没有训练后 map_on/off 性能结论、没有证明 H20 事实分布预测准确或调度收益，亦没有样本传送带任何实现。本分支刻意继承 PR76、**不继承 PR77** 的传送带代码，全部历史 PR 继续 Draft。
