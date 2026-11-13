# PR76：同动作、同真实外生边界下的事实 H5/H20 世界模型预测审计

## 研究目标

PR73→PR75 已完成原仿真器随机服务时长 0–2 步、公开服务输入、原 Deep Sets 模型实际训练及合法多步递推。PR75 比较的是不同模型**自己生成的合法行动和轨迹**，不是将预测与真实事实进行对照，因此需要本 PR 的严格事实匹配。

本轮沿用原始 PR64 24 TRAIN / 12 DEV 与 PR73 `Simulator._service_duration` **不变的 keyed 服务时长 jitter=2**。比较三个方法：

1. 现有 `WorkflowDynamics(ProgressAwareRule)`：公开的确定性 Rule。
2. PR74 **实际 CPU 训练并已 SHA 核对**的原 `BusinessSessionEventModel(query_width=48)`，经既有 `SessionRuleDynamics` + `WorkflowDynamics` 执行；检查点 SHA256 `a08b257c7f8b8a6191b954e9adb117ef8076f3ff93d5373c347d4735e1e45f02`。
3. 原 PR73 严格在 TRAIN 服务事件风险集拟合的 station×phase×公开 active elapsed 离散概率 / Laplace 平滑，通过现有 `SurvivalEventRuleDynamics` 执行；这不是新算法、没有 DEV 拟合。

## 两层互补证据

### A. 同一真实公开 t→t+1 服务事件的概率误差

- 直接复用 PR74 `build_source`，严格按原 24 TRAIN、12 DEV 的**独立场景**生成公开状态和合法 CP-SAT 行动、next-status 服务标签；每个 target 都是取样 `next=in_transit` 或交付 `next=completed` 的真实下一观察事件。
- 三臂均在同一 J2 开发风险集上算 Brier、按取样/交付分层及 12 场景配对 bootstrap。原神经权重只加载、不再训练。DEV 被 PR74 用于 best epoch 选择，因此这里是**开发数据重核，不是新的独立测试**。

### B. 同一真实动作、同一外生观察边界下的 H5/H20

- 从两种**预先规定**的真实公开状态起点出发：`initial_t0` 与该事实场景**最早的 active-service risk**。不能按模型预测结果选择更容易的起点。
- 对每个当前公开时刻 `t`，使用相同原始事实 CP-SAT action `a_t`、并在合法规则状态转移**之后**注入相同实际 `device_available(t+1)`；相同初始场景服务 jitter 实现是唯一的事实目标，模型不读取私有 service_remaining。
- 复用 PR71 `apply_current_public_stage1`：当前时刻真实已经可见的 stage1 任务输入一致，**绝不**将事实 stage2 后继样本强塞进预测世界；后继阶段由现有合法 `WorkflowDynamics` 自己产生。
- 真实下一时刻 `observation(t+1)` 只作为评分对象。对**预测与事实任务 ID 的并集**计算任务状态误差：预测没有按时出现的 stage2 和过早出现的 stage2 都计错；同时计完成阶段数绝对误差、活动服务误差、机器人位置与阶段误差。
- 任何一臂因自己的预测服务事件偏离事实，导致真实 CP-SAT 后续行动引用其**预测世界中不存在的任务**，严格记录 `factual_action_not_executable_in_predicted_world` 并终止该轨迹；不偷换为最近策略，不把截断轨迹伪装成完整 H20。
- 三臂、三个固定 service particle seeds `1810/1811/1812`、H5/H20，12 DEV 原始场景、两种固定起点，共 **432 条原始事实对照运行**。每个起点/窗口的分母固定 **12 场景 × 3 seed × 3 arms = 108**。
- 统计按 12 场景而非相关时刻作 bootstrap。**只有当所有模型在全部 36 个 scene/seed 完整执行固定动作与设备边界时，才允许给出 H5/H20 的公平配对置信区间**；否则报告覆盖率、失败原因与三臂公共合法前缀误差。不能根据留下来的少数成功轨迹宣布模型准确度领先。

## 数据身份和部署边界

- 来源：原 PR64 TRAIN SHA `4c1157777117f82f7deb44f98e1d8255e95611dfcb41c6dfff9e14dd23aee183`、DEV SHA `31a36e0940db7afe015fe35689cb629102dea45e8d61123ed89250abc01d0b86`，原 quality 的 SHA256SUMS 校验。
- 模型：原 PR74 权重、优化器与随机状态由原 Actions #43452830808 下载，模型文件完整 SHA 核查；无随机权重回退，`online_verified=False`。
- 外生设备：真实公开 `t+1` 状态**仅**用作评估中的 oracle-conditioned 外生设备边界，**绝不**喂给当前 `t` 神经模型服务预测。这是条件预测误差研究，不能推广成自部署的未知未来设备预测精度。
- 行动：给定的是事实轨迹的 CP-SAT 动作序列，**不是两臂自己的政策执行**。只检验模型在固定真实行动条件下预测是否准确，不报告闭环调度增益。
- 不使用 PR59 BLIND/TEST，不修改历史权重、旧仿真器、原 CP-SAT 或其他 PR。不使用 GPU 或新训练。三项比较的统计单位是场景。
- CPU 证据存放于 `experiments/world_model_integration/matched_pr76/` 和 GitHub Actions 工作流 `.github/workflows/pr76-factual-action-exogenous-matched-cpu.yml`。

## 解释原则

**完整 H20 预测准确率**只有在指定的事实动作序列能够在该模型全部 20 个预测步骤里合法执行时才有定义。若模型提前走向不一致的任务生命周期，导致真实动作无法执行，那个问题本身就是重要的世界模型长期预测失配，而不是应当删除的异常数据。若大量长窗无法公平对齐，下一步应比较 matched common-prefix 长度、服务阶段事件偏差和后继样本释放时机，再决定是否有必要修改原服务网络的训练监督或重新设计概率模型输入。

## 最终 CPU 执行证据（真实结果）

- **通过** [PR76 原始 CPU Actions #43459048514](https://github.com/lieuyimo/wm/actions/runs/43459048514)，**17/17** 测试，严格使用原 PR64 24 TRAIN/12 DEV J2 真实业务仿真，原 PR74 best.pt SHA 锁定，不使用 GPU/BLIND；[原始 432 条动作/外生事实对照、全部前缀与环境 Artifacts #14056035859](https://github.com/lieuyimo/wm/actions/runs/43459048514/artifacts/14056035859) 包括完整 `results.json`、原始运行 log、源 HEAD、pip freeze、SHA256SUMS。
- **432 条原始对照**：12 DEV 独立场景 × 2 个先验起点 × 2 个窗口 × 3 随机粒子 seeds × 3 个原始基线；每个起点/窗口/arm 的分母严格固定 **36**。全部运行条件化同一条真实事实 CP-SAT 动作序列、同一原始工作站设备下一边界；预测模型不读取真实服务私有剩余时长或未来 stage2 状态。
- **真正一步事实服务事件**：从真实 J2 DEV 1,409 个 eligible risk rows、594 个 event 标签直接重新计算得：Rule Brier **0.28885735**；**原始实际训练** Deep Sets **0.18893717**；仅 TRAIN 拟合的传统 station×phase×public active elapsed hazard **0.17665487**。Deep Sets pickup (n=831) **0.17207059**、delivery (n=578) **0.21318654**。复核了 PR73/74 评分，仍然是开发集不是新 BLIND。

### 固定事实动作的 H5/H20 完整覆盖率

| 同一模型每组 36 次原始 DEV 轨迹 | Rule | 训练的 Deep Sets | TRAIN 条件 hazard |
|---|---:|---:|---:|
| 初始 t0，H5 | **36/36** | **36/36** | **36/36** |
| 最早 active-service，H5 | **36/36** | **36/36** | **36/36** |
| 初始 t0，H20 | **33/36** | **17/36** | **11/36** |
| 最早 active-service，H20 | **24/36** | **12/36** | **10/36** |

- H20 同一场景/随机粒子组合中**三臂同时合法执行原动作 20 步**的数量仅 t0 **9/36**，active-service **7/36**。按预注册门槛，**不能对 H20 全体场景宣布任何方法在事实状态误差上更优**；完整 H20 条件样本覆盖不足。
- t0 H20 中：Rule **3** 次原合法转移验证拒绝；Deep Sets **19** 次原动作引用预测世界中未出现的任务；传统 hazard **25** 次原动作引用缺失任务。
- active-service H20 中：Rule **9** 次原合法转移验证拒绝 + **3** 次原动作引用缺失任务；Deep Sets **2** 次验证拒绝 + **22** 次缺失任务；传统 hazard **26** 次原动作引用缺失任务。
- 前缀截断并非“世界模型软件崩溃”或证明随机性模型概率错误：在**单一抽样的随机服务世界**中，模型生成了与事实不同的合法服务完成分支，事实未来 CP-SAT 后续动作自然可能不再适用。它证明的是**固定事实动作序列的延续支持率较低**，而不是统计意义上的 H20 预测分布本身一定失校准。已同时保存所有失败前的事实状态误差；不删失失败场景来算平均。
- 完成 H20 的单臂子集分别为 Rule 33、神经 17、hazard 11 等，**样本筛选条件不同，直接比较这些子集的终态 MAE 会产生选择偏差**，本 PR 不据此排名 H20 forecast accuracy。

### 完全匹配的 H5 事实状态评分

同一观测 H5 的**全部 36/36** 场景/种子三臂同时完成。终态对事实全部可见任务 ID **并集**上的 `status mismatch fraction`（任务错误或缺失、过早生成后继都计错，越低越好）：

| H5 起点 | Rule | 已训练 Deep Sets | TRAIN 条件 hazard |
|---|---:|---:|---:|
| 初始 t0 | **0.00000** | 0.01072 | **0.00000** |
| 最早 active service | 0.15716 | 0.13572 | **0.12500** |

- **t0 H5**：事实服务尚未充分驱动未来阶段，原 Rule 和传统 hazard 零错误，随机神经采样反而引入少量状态偏差。不可说“模型提高了 H5 预测准确率”。
- **active-service H5**：Deep Sets 相对 Rule 的状态误差有改善（场景/种子均值 `−0.02144`），但仍不如 TRAIN-only 传统 hazard（差 `+0.01072`）。这里是原开发集、单 seed 训练出的模型、三个预测粒子的配对描述，不是独立测试提升主张。
- 原代码另按原场景（而非相关的 36 个粒子）在有充分配对时保存 `scene_bootstrap`；本轮 H20 三臂全程配对不足，必须遵守“不报告有偏 H20 CI”的冻结规则。

## 实验判断和下一步

1. **GO**：现有真实训练的服务事件头能参与业务真实事实条件化的 H5/H20，对活跃服务状态 H5 的预测精度优于原确定性 Rule，且原合法状态校验全程保持开启。
2. **NO-GO**：Deep Sets 不优于简单公开条件 hazard；没有可靠的跨全部原 DEV 场景 H20 action-matched 预测误差优势；更不能推断 CP-SAT 在线调度效率。仅提高模型参数数、重复新 GPU 训练没有证据支持。
3. **下个研究问题**：对不确定服务时长，**个别粒子与事实轨迹分叉是合法概率预测的常态**。应先用已公开的原 PETS 粒子 rollout/标准概率预测评分方法，研究按统一合法策略、相同外生来源得到的**预测分布**和事实结果分布之间的校准、后继阶段释放时间分布和完成/等待量分布，而不是强制把单条事实动作序列硬施加于所有抽样世界。必须采用有依据的已发表方法/公开实现，且只在开发数据确定门槛，不打开 BLIND。
4. PR76、PR75、PR74、PR73 全部保持 Draft，0 GPU、0 BLIND、0 新模型训练。
