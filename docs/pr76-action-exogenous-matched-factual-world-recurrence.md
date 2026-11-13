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

## 最终 CPU 执行证据

最终 CI Run 和真实 Brier、H5/H20 全体分母、截断覆盖、场景配对结果在该分支审计结果确认后归档；以机器产物为准，不先写假实验数字。
