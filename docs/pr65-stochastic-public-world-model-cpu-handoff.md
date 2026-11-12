# PR65：关系世界模型的公开历史条件随机设备转移接口（CPU 阶段）

## 目标与研究边界

研究主线仍为 **动作条件下合法公开系统状态的关系世界模型**。PR61 的 PyG public-map / GRU-RGCN 网络在一阶 t→t+1 实体状态预测上有已验证收益，但 PR63/64 证明当前神经网络不能据此宣称随机设备故障发生/恢复预测已经校准。

本 PR 保留 `PublicMapTemporalRelationalWorld` 的任务、机器人、工作站其他量和匿名新任务事件输出，通过**独立的、TRAIN-only 离散时间条件 hazard**覆盖下一步 station `device_available` 的概率通道。具体参考：

- Suresh, Severn & Ghosh (2022), *Survival prediction models: an introduction to discrete-time modeling*, DOI [10.1186/s12874-022-01679-6](https://doi.org/10.1186/s12874-022-01679-6)。UP→DOWN 与 DOWN→UP 使用各自处于风险集的公开状态历史条件概率。
- Chua et al. (NeurIPS 2018), *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models* (PETS) 的轨迹采样思想，**当前 PR 仅提供 Bernoulli 事件候选采样接口，不声称实现 PETS、粒子 H20 或完整模拟器**。

没有引入新 Q head、新调度 ranker 或自创 hazard 学习算法。

## 代码与原始数据源

- `experiments/world_model_integration/stochastic_pr65/pr65_public_device.py`：公开 4 帧可见历史下的 station 状态年龄、TRAIN-fitted `constant` / `age_h4` 故障/恢复概率，以及 `overlay_one_step` 与确定种子 Bernoulli **事件提案**。
- `pr65_cpu_admission.py`：只从 PR64 实际原始 TRAIN 24 与 DEV 12 场景归档读取 t→t+1 数据；对所有 DEV 有效 station 风险时刻逐项核对 PR64 已冻结 `fit()` 与 `_p()` 输出；用随机初始化的**真实原 PR61 PyG 网络**做最小 forward 与覆盖范围 smoke，保留其他节点/事件输出。
- `test_pr65.py`：当前输入与过去历史一致性、UP/DOWN 概率、TRAIN-only 表验证、禁用 clock/exact/future、受影响通道限制、seeded event proposals。
- `.github/workflows/pr65-public-hazard-world-cpu.yml`：从原 PR64 **Run #43264954353** 直接取 TRAIN/DEV/quality 三个原始 artifacts，并校验原 SHA256SUMS；CPU-only、0 GPU、0 新仿真、0 模型训练、0 原 PR59 BLIND。

**特别注意**：PR64 新 DEV 已经用于模型/方法开发，不能再称为独立未知测试。不能用这份开发集反复试出专用 clock buckets。PR61 原始 A100 六个 `best.pt` 权重没有加入 Git 源，CPU smoke 是随机初始化模型**接口检查**，不意味着已获得新的状态精度或学习模型效果。不要把复现 PR64 `constant` / `age_h4` Brier 当作 PR65 性能提升。

## 当前可检验门禁

1. 所有原始源码、源数据与 quality SHA 完整，24 TRAIN/12 DEV + 全部有效 UP/DOWN 风险时刻保留；原 BLIND 未打开。
2. PR65 adapter 在每个风险时刻的预测概率与原 PR64 **TRAIN-fitted** 基线精确一致，Brier 重算也一致。
3. 原 PR61 一步 forward 的非 station-device 字段与任务事件输出不变，station-device 概率在 0..1 且仅由 t 时公开历史提供。
4. 设备事件候选采样可由种子复现。**该二值事件并非经业务合法投影的下一世界**，因此不能据此声称 H5/H20 世界状态可执行。

如果 CI 或任一门禁失败，不提交“模型已经改进”或“可以扩大训练”结论，保留失败输出后修改当前 PR 的实现缺陷。

## 下一阶段的独立训练门槛（需要独立评审，先不使用 GPU）

目前随机故障时刻由不进入模型的私有 SHA/seed 生成，若公开历史无因果信息，未来真实故障时间只能以频率风险描述，不可能要求学习器神奇地提前猜中具体 tick。应把 `P(UP→DOWN|public history)` 与 `P(DOWN→UP|public history)` 的 **calibration、sharpness 与多步 uncertainty** 区分报告。

下一批研究必须优先：

1. **合法递推接口**：以原 `BusinessSimulator` 规则和资源/安全约束为准，定义下一时刻 device 事件抽样进入状态转移的 API，检查对服务、队列、任务与机器人可观察状态的影响。输入为当前公开状态+候选调度动作+外生事件分布；输出为合法下一世界。不可简单在预测向量中改 device bit 就声称合法 world rollout。
2. **严格独立的时间机制对照**：预先固定新的同任务、同地图、同业务配置和基础随机性配对，每对只变**私有起始时间**。必要时分层增加真实故障/恢复事件，使罕见事件统计有意义。保持 PR64 开发集、PR59 真正 BLIND 以及旧模型权重全部不变；另留完全未参与方法选择的测试 cohort。场景是独立性单位。
3. **训练决定**：若独立开发场景上，公开历史条件方法相对 TRAIN 常数基线在场景配对 Brier 和 log-loss 提供稳定增益，才考虑训练一个**已有 GRU/RGCN 关系表征共享的二元 hazard 输出头**，与原 PR61 网络对照。先在 CPU 用逻辑回归/离散 hazard 验证信息含量；无信号时保留常数概率，不盲目扩模型或训练集。
4. **GPU Agent handoff**：训练前冻结特征、模型结构、参数量/对照组、训练/开发/最终测试拆分、种子、损失权重及停止准则；可复用 PR59 CUDA 训练/Checkpoint best/last/Adam/RNG 归档机制；确保 loadable 权重单独安全归档。训练后先单步外生事件概率与实体误差，再合法 H5/H20 rollout；真正通过后才比较在线 CP-SAT 候选动作决策效果。所有不可见真实未来只能用于计算测试指标，不可用于预测输入。

## 执行结论

**本 PR 的通过仅授权“概率事件接口已接入世界模型一步输出”的工程结论。** 这不等于设备故障可被预测、改进 PR61 神经网络、合法递推到 H20 或在线调度改善。保留 Draft，与 PR64 stacked，未合并任何历史 PR。
