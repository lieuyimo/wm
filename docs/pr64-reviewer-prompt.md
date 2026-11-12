# PR64 Review Agent 指令

仓库：<https://github.com/lieuyimo/wm>，目标：[Draft PR #64](https://github.com/lieuyimo/wm/pull/64)。

请在 PR #64 当前最新HEAD（不是历史固定SHA）执行**只读独立研究代码审查**。这是GPU训练前的**外生事件概率建模与生成机制可观测性审查**，不是新模型训练委托。禁止合并PR、改原PR48/55/56/59/61/62/63/64历史实验结果、运行任何BLIND TEST，不触碰GPU、不打开原PR59 12个测试场景。

## 阅读内容与核实数据

- `docs/pr64-random-outage-shift-review.md`；完整现实仿真和CIs。
- `experiments/world_model_integration/outage_pr64/protocol.json`；**实验前固定** 24 TRAIN/12 DEV VAL、隐藏起始时间支持、数据质量门禁。
- `pr64_source.py`，对照原`stress_pr48/pr48_scenarios.py`、`quality_pr55/pr55_generate.py`、`topology_pr56/pr56_public_map.py`、`temporal_pr54/pr54_public_sequence.py`，独立确认**仅修改私有故障起始时刻**，不影响原资源规则、任务、机器人、工序、PR48故障持续时间采样和合法调度。核查图/历史输入绝无私有故障开始、结束和未发布任务未来信息。
- `pr64_gate.py`，逐项检查完整UP/DOWN风险集、常数风险TRAIN-only拟合、年龄/时刻信息来源、uniform-4-tick基线没有偷用PR63固定tick6/12特殊分桶。独立确认恢复指标分组和实际可观察年龄的定义。
- [真实CPU来源Run #43264954353](https://github.com/lieuyimo/wm/actions/runs/43264954353)：原24+12真实场景、全时序源数据及SHA256、0 GPU、0 NN训练、0 BLIND。
- [原源SHA只读bootstrap Run #43265617968](https://github.com/lieuyimo/wm/actions/runs/43265617968)：审计数据无需再运行任何仿真，场景等权差值与CIs都已归档。任何结果应以**实际归档数据**为准。
- 相关既有研究：Suresh et al. 2022 离散时间生存/风险模型（DOI 10.1186/s12874-022-01679-6）；Chua et al. 2018 PETS概率轨迹传播（NeurIPS）。**不要把既有离散hazard或粒子rollout写成论文创新点。**

## 六项必须给出明确审查裁决的问题

1. **身份和完整性**：所有真实world数据是否真的来自原BusinessSimulator/CP-SAT/原Grid，地图与先验随机种子隔离是否严格？未来schedule是否只在真实仿真器内部，且进入模型输入的字段与真实可观测状态相符？
2. **识别性**：这次训练和验证的地图家族与故障时间支持同时不同，因而并非只控制时间的因果配对；模型从该结果究竟可以声称什么、不能声称什么？如果因果归因需要下一轮，设计最少的同seed同地图同服务/任务、仅改私有事件起始时间的独立配对。
3. **概率评估**：24 TRAIN有32个发生/32个恢复，12新DEV有16/16。完整UP/DOWN风险集模型与分组Brier、log-loss、scene-level bootstrap有没有错误？是否存在tick级伪独立或事后修改阈值/分桶？
4. **推断边界**：`age_h4`的轻微好处、`clock4`变差、`exact_age4`数值更优但95% CI跨0。是否支持“设备事件可预测”？应明确**不是**，也不能断言时间特征必然有害；原PR61 NN没有在新PR64源上运行。
5. **模型迁移**：能否基于PR61真实关系图状态模型，单独加一个**共享公共历史条件的离散时间故障/恢复分布**并与约束合法的世界转移合成？哪些外生信息本来就不可预测，应使用概率/粒子而不是点预测？参照已有方法而非发明新的Q/ranker。首次CPU原型需要满足哪些统计门禁，之后才允许GPU？
6. **继续还是停止**：如果尚缺可靠真实外生分布、故障时间无法从公开状态预测、独立事件太少或结果不稳，给出清晰BLOCKED/NO-GO证据与最小高质量新数据需求。**不允许因当前数据不支持方法而无约束扩大数据或参数。**

## 输出要求

提供在PR #64 Conversation中的**结构化review**，标记`PASS`、`BLOCKED`、`NO-GO`与缺陷严重性，引用代码路径、具体样本数/检验和源SHA，列出最小后续实验。保持PR Draft；**未经额外授权不得新增GPU训练、打开BLIND、覆盖旧PR实验或合并**。

学术主线始终是**学到动作条件下合法公开系统动态的关系世界模型**；随机外生设备事件是其中需要处理的不确定性，而不是把事件hazard基线包装成独立世界模型。
