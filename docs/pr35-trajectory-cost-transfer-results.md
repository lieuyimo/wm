# PR #35：参考 PETS 的轨迹累计代价评分是否改善世界模型在线调度？

日期：2026-11-12。**代码、预注册、真实在线仿真和 CPU 测试已完成。综合判断：评分方法迁移已实现，但不支持将其晋升为默认策略。**

## 方法与相邻工作来源

- Chua et al., *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models*, NeurIPS 2018, [论文](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)：PETS 使用概率模型与多个采样轨迹，以候选序列的预测累计回报选择行动。**本项目只迁移多粒子轨迹上积分/聚合代价的 MPC 思想，未复现 PETS ensemble 网络、CEM 搜索、连续运动控制或其损失函数。**
- 本项目不存在新自造评分损失，直接复用已有 `experiments/world_model_integration/decision_value/pr27_evaluate._trace_cost`，其定义来源于既有 `lab_transport.decision_value.trace_score`：
  1. 每个未来 tick，已有任务/工序尚未完成，加1单位等待成本；
  2. 超过公开任务 deadline 仍未完成，每tick额外+4单位迟交惩罚；
  3. 用预测终点的剩余几何运输距离作为平分时的二级排序成本；
  4. 各粒子先计算真实预测路径成本后取平均，候选评分按 **(累计成本, 终点距离)** 字典序最小化，平分选原 CP-SAT 第一提案。
- 原 PR30/32 的默认模型评分未变：H20 末端预计完成工序数最大化，相同预计值原 CP-SAT 优先。不修改已有网络训练、不更改 `WorkflowDynamics`、地图、服务随机机制、任务派发求解器与运动规则。
- 在 PR30 的 `OnlineHorizonPolicy` 中只抽取两个可以覆写的评分/候选选择方法作为接口，默认类没有行为变化，新类 `TrajectoryCostPolicy` 通过相同的原版 `pr30_run.one()` 执行真实仿真。因此**同模型进度评分 vs 同模型轨迹代价评分**具有相同 H20/P3、CP-SAT 候选、seed和0.5秒单次决策CPU预算。

## 冻结实验与完整审计

协议 `experiments/world_model_integration/trajectory_cost/pr33_protocol.json` 在实验前固定，共24组完全新scene seed（light/medium/heavy/shift各6组）。6臂：
- CP-SAT；
- Rule-H20 endpoint progress；
- Rule-H20 trajectory cost；
- 原 V4-H20 endpoint progress；
- 原 V4-H20 trajectory cost；
- PR31 Validation选中的 Scratch V4-H20 trajectory cost。

模型权重从已归档原V4和 PR31 [全部可恢复训练环境](https://github.com/lieuyimo/wm/releases/tag/untagged-871b60f7bf013400cb45) 恢复，SHA256与验证选型匹配，**不根据PR32/33测试结果再次训练或选择模型**。

[GitHub Actions #43147330962](https://github.com/lieuyimo/wm/actions/runs/43147330962) **SUCCESS**，[6策略 × 24组真实在线原始数据、日志和SHA256 artifact #13912985342](https://github.com/lieuyimo/wm/actions/runs/43147330962/artifacts/13912985342)。

- 3项新专用测试 + 3项 PR30 回归测试全部 PASS。
- 24个场景 × 6个真正执行的策略，**144条真实仿真轨迹均完成且0安全违规**。
- 主CPU约**148.856秒**、0 GPU、0新训练，没有模型/业务不合法事件。

### 最终完成时间（每组场景等权，tick越小越好）

| 场景（各6组） | CP-SAT | Rule-progress | Rule-cost | V4-progress | V4-cost | Scratch-cost |
|---|---:|---:|---:|---:|---:|---:|
| light | 20.167 | 20.167 | 20.167 | 20.167 | 20.167 | 20.167 |
| medium | 49.833 | 49.500 | 48.500 | **48.500** | **48.500** | **48.500** |
| heavy | 76.667 | **74.000** | 76.500 | 74.167 | 76.500 | 76.500 |
| shift | 109.333 | 109.333 | 108.500 | 108.500 | 108.500 | **106.667** |

### 同模型、相同 H20 的价值函数对照

- **V4 cost vs V4 progress：** medium 6/6持平；heavy **1胜0平5负**，完成时间平均**+2.333 tick（变差）**，平均逾期作业数 **+0.500**；shift平均完成时间相同，1胜3平2负，平均逾期作业减少0.333。
- **V4 cost vs Rule cost：** **所有24组的最终makespan全部持平**，即在这个指标与这些场景中**没有证据显示学习服务事件动态独立创造收益**。注意这不意味着所有时刻派单行为、资源等待和预测轨迹都完全一致：shift中 Rule-cost 与 V4-cost 的CP-SAT首选翻转次数分别8与9。
- **Scratch cost vs 原 V4 cost：** shift 平均 makespan改善**1.833 tick**（1胜5平0负），但平均迟交作业数**从3.167升到3.667**（变差0.5）。其它三类完成时间全持平，因此不能只挑 shift 的均值宣称新模型可靠。
- **V4 cost vs CP-SAT：** medium 平均−1.333 tick，3胜3平；heavy平均−0.167 tick，1胜4平1负；shift平均−0.833 tick，1胜5平。Rule-cost亦有相同最终makespan表现，所以不能将这些相对CP-SAT的变化全部算作学习模型贡献。
- **H20成本和安全**：heavy V4-progress和V4-cost各有1次预算回退；Rule-cost 0，Scratch-cost 0。相同预算策略会因真实CPU运行时与候选选择变化出现回退差异；未修改预算来美化结果。所有策略0违规/0错误。

## 阶段判断和决策

**本 PR 方法复用与在线实验的工程实施成功，但“轨迹累计业务代价优于旧H20工序进展”和“新训练模型具有跨场景决策优势”均不成立。**

这是一个清晰的研究诊断：世界模型从一阶服务完成概率到长时序未来状态预测的链路已存在，但其对“关键业务动作哪条更值钱”的稳定增量仍不足。继续调节阈值、deadline系数或不断更换 Score，并在本次24组上反复评价，会把测试集变成开发集。

下轮优先直接研究**世界模型对动作条件下完整任务交付/资源约束的可识别性**：何种公开状态确实预示未来候选动作优劣，哪些长时差异不是简单已知几何/调度规则就能推出来，判断是否需要带有实际条件动态的新增观察数据。只有新的训练标签或目标能被相邻成熟方法正确处理时，再启动新的有完整环境和optimizer/RNG保存的训练，不做无据加epoch。

保持所有PR为Draft，不改正式 CP-SAT 的默认运行策略，也不修改PR19 工程演示。原V4、Scratch-V4及Rule各自作为清晰实验基线。
