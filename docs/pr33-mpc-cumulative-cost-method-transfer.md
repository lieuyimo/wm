# PR33：PETS / MPC 累计轨迹代价的已有方法迁移

## 研究动机与实际实现

PR31 扩充服务事件训练数据后，冻结V4、Warmstart和Scratch的独立事件预测Brier改善，但PR32全程在线的Scratch V4相比原V4在24个新场景只有2胜19平3负。PR32已经归档且不再用于选择 PR33 评分形式。因此，本 PR 将研究焦点从“服务事件概率精度”移到**候选调度方案的长期业务结果评分**。

**方法参考**：
- Chua et al., *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models*, NeurIPS 2018, https://proceedings.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html。原论文PETS采用概率动力学和轨迹采样规划。这里仅迁移既有H20/P3概率滚动预测下的**期望累计代价**思想，并非其CEM连续动作求解器或网络ensemble的完整复现。
- Reiter et al., *Synthesis of model predictive control and reinforcement learning: Survey and classification*, Annual Reviews in Control, 2026, https://doi.org/10.1016/j.arcontrol.2026.101045。MPC在有限未来轨迹上最小化累计代价，并仅执行当前首步，在下一时刻重规划。
- 本仓库早已实现的 `lab_transport/decision_value.py::trace_score` 与 `experiments/world_model_integration/decision_value/pr27_evaluate.py::_trace_cost`。PR33 **直接调用原有适配器**，逐粒子统计：已知任务各预测tick未完成记1、逾期未完成额外记4，预测终点剩余运输路径距离作为确定性第二排序项。代价权重、对象与规则都来自**现有代码**，本PR没有重新定义或调参。

两种固定评分与同一 CP-SAT 候选（最多4）、同一H20/P3世界模型预测、同一0.5s决策预算做严格配对：

1. `progress`：PR30/PR32的固定模式，选预期H20完成工序数最多的方案；平分优先原CP-SAT第一方案。
2. `cost`：按原 `_trace_cost` 计算每粒子累计未完成/逾期时间成本期望，再以终点路径距离为第二键，仍平分优先原CP-SAT第一方案。**这个目标不是另训的决策网络，而是已有模型预测用途的迁移。**

三种动态模式来源固定：Rule、原冻结 V4、PR31 验证集选中的 Scratch V4。完整七臂为 CP-SAT、Rule-progress、Rule-cost、Original-progress、Original-cost、Scratch-progress、Scratch-cost。

## 冻结独立性、计算与安全

所有开发场景在PR33首次运行前写入 `experiments/world_model_integration/mpc_cost/pr33_protocol.json`，24个从未使用的场景seed（轻、中、高、服务参数偏移各6组），独立于PR24～PR32先前全部已见数据和PR31 Train/Val/Test。参数沿用已有合成业务场景机制，无新增设备故障、私有未来信息或人工调节随机数以制造收益。

真实仿真结果均来自原 `BusinessSimulator`、原业务规则、原合法 CP-SAT 候选、同一scene seed的 keyed service duration。各臂对自己的当前公开状态在线计算候选。每次完整评估所有候选之前或之后超过0.5s预算，即丢弃候选评分并退回原CP-SAT第一方案，不进行不完整比较；报告所有超预算及回退。

本阶段**0 GPU、0 新训练**。原V4权重SHA256固定，Scratch来源与PR31的Validator选择和不可变artifact SHA核对；PR31完整环境、锁文件、数据、训练日志、完整Adam状态和RNG断点继续留在其GitHub Draft Release，不覆盖任何已有文件。

## 分析规则

只以预先确定的Makespan、逾期作业数量、工作站资源等待、完成作业、安全违规及全流程CPU成本评价业务结果，按场景seed配对统计。任务预测阶段误差和训练Brier改善不直接当作业务价值；不得事后用本PR场景调累计损失的1/4权重，也不以某个场景最优模型直接晋升正式策略。

若总体表现无稳定改善、回退过多或Rule同样有效，则保留现有架构与已恢复训练环境，后续应审查业务场景观测条件和预算，而不是重复对测试组后训练。
