# PR66：关系世界模型的合法随机外生事件递推 CPU 验证

## 研究目的

PR61 已学习关系图上的公共一阶状态，PR65 实现不依赖隐藏故障时间的设备概率输出接口，但概率比特不能直接作为合法完整业务状态。PR66 的工作是最小化修正**状态转移语义**，而不是新造训练方法或增加 GNN 参数。

### 外生事件的因果时序（t→t+1）

1. 使用当前可观测业务状态 `x_t` 和合法候选动作 `a_t` 调用原始 `WorkflowDynamics(ProgressAwareRule)`，得到不借助未来随机事件的 `x^-_{t+1}`。
2. 使用 **原 `check_business_transition(x_t, x^-_{t+1})`** 对运动、资源、任务阶段释放、身份、时间顺序等进行合法性审查。这一函数明确拒绝未经解释的未来设备变化，**禁止修改或弱化该保护**。
3. 使用 PR64 TRAIN-only 全风险集拟合的离散时间 UP→DOWN / DOWN→UP hazard，独立抽样得到下一边界的公开设备状态 `d_{t+1}`，不允许读取私有故障表和下一时刻真值。
4. 仅在 `t+1` 观测边界通过 `capacity_view` 与 `rebuild` 更新设备可用性、资源入场/占用和服务会话关系。保持所有机器人状态、任务生命期、固定站点字段完全不变。此后 `x_{t+1}` 成为下一次合法规则转移的输入。
5. 使用与候选独立的确定性粒子随机种子递推 H5/H20；当前只证明 **CPU 规则状态递推的合法性**，并非 PR61 GNN 的真实 H5/H20 预测、设备风险校准或在线调度收益。

对应实现：`stochastic_pr66/pr66_rule_rollout.py`。参考条件离散风险模型：Suresh et al., 2022, DOI 10.1186/s12874-022-01679-6；概率模型轨迹传播参考 Chua et al., NeurIPS 2018 PETS，不声称复现或新发明这些算法。

## 原有真实数据和验证范围

- PR64 **原始** CPU Run [#43264954353](https://github.com/lieuyimo/wm/actions/runs/43264954353) 的 24 TRAIN、12 developer VAL 和质量审计归档，逐包 SHA256 检查，不重写任何已归档数据。
- `pr66_cpu_admission.py`：使用原 PR64 的公开实验构造/CP-SAT，在 TRAIN 与 DEV 各选一个既有场景，从原 BusinessSimulator 重新获得真正 t=0 的公开 ObservationV2，与原数据首个公开 t→t+1 输入和标签逐字段比对；再以 TRAIN-fitted 风险模型进行两种方法（constant/age_h4）×H5/H20 的 CPU 规则随机递推。**这些 DEV 全部属于已观察开发样本，不作为新的未知 BLIND。**
- `test_pr66.py`：原有多阶段业务 fixture；设备事件只在下一时刻生效，避免追溯改变上一时刻服务；设备 down 时正确暂停，服务占用按原规则保留；严格对照不允许直接通过旧 checker 注入未来事件；所有容量、工作流和机器人约束需逐步满足。
- 完整 CI `.github/workflows/pr66-lawful-stochastic-world-cpu.yml` 保留程序日志、源 HEAD、依赖 freeze 和 SHA256SUMS；只使用 CPU，0 GPU，不加载 PR61 checkpoint，不触碰 PR59 12 个水平货架 BLIND TEST。

## 边界、判断与下一步

**当前模块不是直接让 PR61 神经网络进行 H20 rollout。** PR61 预测的是节点 t+1 向量，尚缺从预测连续变量/新任务事件到符合原 BusinessSimulator 工作流的、可重新编码的完整公开世界 `ObservationV2` 的闭环映射。PR66 因此选择已有合法规则模型负责世界递推，PR65 hazard 负责外生事件不确定性。这是最小可核查的状态转移接口演示，不等于证明 learned dynamics 的多步价值。

在实际 CI 通过后，才允许声称**规则世界在随机事件下完成 H5/H20 的合法递推接口验证**，但不允许声称神经 H20、故障预测改善或调度收益。下一研发目标应是：以现有公开任务/实体目标作为约束，设计可复现的 PR61 action-conditioned node/event distribution → 合法业务世界下一状态的映射，严格与 `check_business_transition` 协作；经过独立新开发场景条件对照后，再决定是否训练新 neural head。原 PR64 数据小、罕见事件不稳，禁止依据一次 CPU smoke 就启动大量 GPU 训练。

PR66 与 PR65/64 stacked，保持 Draft，不合并，不覆盖历史结果。
