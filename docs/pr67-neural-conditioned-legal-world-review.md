# PR67：PR61 关系神经世界模型输出驱动合法业务状态递推

## 研究目的

PR64/65 已确认新故障时间随机化的来源和离散 hazard 概率接口；PR66 完成**规则世界**的合法随机外生事件递推，但世界状态主体仍由 `ProgressAwareRule` 产生。PR67 将 **PR61 PublicMapTemporalRelationalWorld 的任务节点下一状态分布**直接用作已开始服务的完成概率，允许神经预测改变业务任务状态，而不是对规则轨迹做纯打分。

方法不新增神经网络结构、不自创优化器。沿用仓库中：
- PR54 `graph_public_features` 和 t−3…t 四帧按真实公共实体 ID 对齐的因果历史编码；
- PR56 真实静态网格的 map-cell / entity-to-map graph；
- PR59/61 原 PyG SAGEConv + GRU/RGCN 关系世界网络；
- `business_transport/session_dynamics.py` 中已经存在的“神经服务完成概率 → 抽样 → 设定服务剩余门槛 → `ProgressAwareRule` 执行”的机制；
- PR66 原 `check_business_transition` + `capacity_view` + `rebuild` 外生设备边界。

参考已有思路：Chua et al., NeurIPS 2018, [PETS](https://papers.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html) 中概率动力学轨迹传播；Suresh et al., 2022, DOI 10.1186/s12874-022-01679-6 的条件离散风险建模。**不声称复现或发明 PETS；不把简单神经服务预测作为新算法。**

## 原理与输入输出

给定可观察 `o_t` 和实际候选动作 `a_t`：
1. `PublicGraphHistory.encode(o_t, a_t)` 构建原 PR61 形状的双图张量；图节点只含真实公开的机器人、工作站、资源、任务信息；历史 t−3...t；过去任务通过**内部对齐键**连接，真实标识符不会成为网络数值特征；从不使用隐藏 outage timetable 或真实未来标签。
2. PR61 现有 `forward` 输出每个已知任务节点 6 维下一状态估计。对**已在工作站、已开始服务、已具备可观察有效进度、设备当前可用且容量允许**的查询：
   - pickup（当前 assigned）：取 task head 的 `p(next=in_transit)`;
   - dropoff（当前 in_transit）：取 task head 的 `p(next=completed)`.
3. 参考现有 `SessionRuleDynamics` 的事件执行机制，从预测概率独立抽样完成事件，将当前工作站的规则服务门槛临时设为 `observed_active_elapsed + (sampled?1:2)`。完整 `WorkflowDynamics` 按原运动、会话、容量、任务阶段规则执行一次。随后恢复临时服务门槛的公开字段。
4. **严格调用现有 `check_business_transition`** 审查任务状态、运动、资源和后续阶段释放。外生设备风险仍由 PR66 在 *t+1 下一时刻边界* 抽样与重建。下一步使用预测完成的 `o_{t+1}` 重新编码图和历史，形成**真实多步、动作条件下的神经任务事件 + 规则运动/业务 + 随机设备事件混合世界递推**。

注：本机制由 PR61 **任务输出头**驱动服务状态变化，不代表 PR61 的机器人物理位置坐标、资源数量等全部六维节点量已经可被解码为合法完整世界。未观测有效服务进度时仅用既有规则，不会用神经概率替代隐藏服务变量。匿名新任务事件仅做诊断，后继任务始终由现有 `release_successors` 合法产生。多候选动作要使用**独立重置的**神经历史分支。

## 可验证实验与证据门禁

- `stochastic_pr66/pr66_rule_rollout.py` 以 `world_dynamics` 注入既有 `lawful_rule_then_event`，默认纯规则行为保持原样。
- `neural_pr67/pr67_neural_world.py` 实现真实现有 PR61 PyG input/forward → observable service completion → 原规则世界。
- `neural_pr67/test_pr67.py` 使用原 BusinessSimulator 的长服务会话进行**控制因果对照**：同一公开状态、同一动作、同一资源环境下，把单一任务完成概率分别设为 0/1；要求原规则检查器两边都通过，且最终任务状态真的不同。此对照仅证明接口敏感性，不是神经预测精度。
- `neural_pr67/pr67_cpu_admission.py` 直接 SHA 验证 PR64 TRAIN 24 / developer VAL 12 原始归档；每个来源复算一例真实业务 t0→t1 原输入/标签；用实际未改变的 PR61 PyG 网络在两种固定初始化随机种子下完成 H5/H20 **每一步 forward**、神经服务会话风险和 PR66 外生事件叠加、业务合法性验证。
- `.github/workflows/pr67-neural-to-legal-world-cpu.yml` Python3.10 / 原 lock 依赖 / PyG2.6.1 CPU-only，存储日志、依赖环境、HEAD 和原始 SHA。

## 不能宣称的内容

PR61 GPU 训练所得 6 组 `best.pt` / optimizer/RNG 权重归档在先前 PR61 报告中注明为 **仅本地**，当前 GitHub 仓库无法直接读取。PR67 的真实 PyG forward 只能用随机初始化权重进行 API/合法性 smoke。这虽然确实允许神经输出 *改变规则执行的任务完成状态*，但不能拿随机权重计算世界模型 Brier、长期状态误差、样本泛化或在线调度收益。

PR64 developer VAL 已参与开发，**禁止将其作为新模型最终盲测**。PR59 真正 OOD BLIND 不读取。PR67 不训练、不微调、不使用 GPU、不覆盖已有 checkpoint，不新建方法类别。

## 下一次必须进行的训练/权重任务（交付 GPU Agent 前先审查）

1. **权重取回与归档**：从 PR61 原运行环境恢复 `pr60-gpu-checkpoints-20261112.tar.gz`；对照 PR61 记录的 SHA256 `919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14`；验明训练代码 HEAD、map_on/off arm、3 seeds、state_dict、模型结构和 optimizer/RNG。必须保留整个实验 Python/pip freeze、CUDA/Torch/PyG 版本与参数；不得将其他 checkpoint 充作 PR61。
2. **预训练模型真实 CPU 评测**：不需要 GPU 即可先加载保存权重，固定等量 `map_on` / `map_off` 的原模型，按 TRAIN/DEV 原 PR59/60 数据重新核查教师强制的一阶节点误差与 device Brier；随后使原权重通过 PR67 合法 H5/H20 的完整 graph recurrence。报告预测服务事件的数量、频率、校准和状态距离，使用场景级配对 CI，明确区分数据分布。
3. **协议预冻结**：新一批独立时间/地图拆分不能依赖 PR64 developer VAL 结果手调风险分桶。定死原 PR61 单步 network vs task-head decoder 的控制组，服务完成 event interpretation，事件种子，H5/H20 状态量（任务完成、工作站负载、机器人位置、拥堵/等待、资源、设备可用性）、每个事件/状态的准确性与合法率；单独新建未使用测试 cohort。
4. **GPU 训练 GO/STOP**：如果恢复 checkpoint 后显示模型对递推服务状态不可校准，先检验是训练数据、输出头语义还是递推分布漂移问题；仅在独立训练数据和预注册对照足以识别改进后，才升级训练。保持 PR61 原网络作为复现基线，不能简单用更大参数量冒充世界模型研究创新。

## 科学裁决

**预期 PR67 通过门禁后仅能声称“PR61 架构的输出可通过已存在的合法服务状态执行机制，在 CPU 上形成神经条件 H5/H20 的真实闭环架构”。** 不能声称经过训练的 PR61 模型长期预测更好。PR67 保持 Draft、stacked 在 PR66 上，不合并旧 PR，等待独立评审。
