# PR78 GPU H1: 六组关系图世界模型条件服务事件预测的初步研究评审

**事实来源**：[PR78 GPU 执行方训练报告（2026-11-14）](https://github.com/lieuyimo/wm/pull/78#issuecomment-6879536110)  
**研究 Project**：[入口 #79](https://github.com/lieuyimo/wm/issues/79)、[H1 评审 #82](https://github.com/lieuyimo/wm/issues/82)、[表示与监督目标诊断 #85](https://github.com/lieuyimo/wm/issues/85)  
**隐私边界**：这里只记录已披露的汇总指标与训练事实。原 PR61 和新 PR78 的所有检查点均在受控私有服务器，本评审不请求或下载权重。**本报告未在本地独立重跑私有模型**，引用的 GPU 数字为训练执行方提供。

## 1. 模型来源、训练合同

- 执行方报告原 PR61 archive **6,661,520 bytes** / SHA256 `919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14`，12/12 manifest 的 best/last 校验，6/6 原 `best.pt` 安全加载，原 PR59 验证 SHA 匹配。原 GNN 为 **52,953** 参数 GRU/RGCN/SAGEConv，map_on/off × seeds 1810/1811/1812。
- 新服务完成事件头 **2,689** 参数，原 PR61 图主干完全冻结；使用 **48d** 已公开的有效服务特征和原网络经过 `node_next_head` 压缩得到的 **6d next-node 状态输出**，标准 BCE/Adam，**20 epochs**、batch16、lr1e−3、按照 DEV 12 场景 macro Brier 选择 epoch。
- A100 GPU 0 执行，环境 Python3.11.15 / torch2.5.1+cu124 / PyG2.6.1；六支约 2m17s，缓存与真实原 GNN 的最终概率预测一致（执行方报告最大差异小于 1e−7）。
- 原 PR64 J2 TRAIN24：**2,593** 风险；DEV12：**1,409** 风险 / **594** 事件。按原训练执行方报告与 PR73/74 的风险定义一致；原 PR59 BLIND/TEST 保持封存。**12 场景而非 1,409 个事件是配对泛化置信区间的统计单位**。

## 2. 原始报告结果（DEV，Brier 越小越好）

| arm | seed | best epoch | 风险加权 Brier | 12 场景宏 Brier | pickup Brier | dropoff Brier |
|---|---:|---:|---:|---:|---:|---:|
| map_off | 1810 | 19 | 0.19218723 | 0.19343601 | 0.17650680 | 0.21473123 |
| map_off | 1811 | 18 | 0.19720440 | 0.19816063 | 0.17908821 | 0.22325039 |
| map_off | 1812 | 20 | 0.18814497 | 0.18934325 | 0.17146103 | 0.21213172 |
| map_on | 1810 | 19 | 0.19246271 | 0.19390402 | 0.17566261 | 0.21661648 |
| map_on | 1811 | 18 | 0.19305506 | 0.19387416 | 0.17462695 | 0.21954948 |
| map_on | 1812 | 20 | 0.18896227 | 0.19064675 | 0.17437671 | 0.20993212 |

| 方法 | 同一风险目标的 Brier |
|---|---:|
| Rule | 0.28885735 |
| PR74 已训练 Deep Sets | 0.18893717 |
| PR73 TRAIN-only phase/station/elapsed hazard | **0.17665487** |
| PR78 map_off 3 seeds 均值 | 0.19251220 |
| PR78 map_on 3 seeds 均值 | 0.19149335 |

计算校核：`map_on - map_off` 的 seed 配对差为 **+0.00027548、−0.00414934、+0.00081730**。方向不是持续为负：3 seeds 中 2 次 map_on 较差。map_on 均值比 hazard Brier 高约 **0.01483848**（约 **8.4%** 相对劣化），比 Deep Sets 高约 **0.00255618**。此处只是观察到的开发集差距，**不是显著性检验**。

单支最低的 `map_off/1812` 风险 Brier 0.18814497，确实略低于 Deep Sets 0.18893717，但这属于**事后选择有利种子**，不能将它当成预注册的三种子方法平均结论；仍明显不如 TRAIN-only hazard。

## 3. 模型架构必须正确解读

源码路径 `gnn_pr78/pr78_model.py:PublicMapRelationalServiceWorld`：

```text
original PR61 public 4-frame GNN state predictor
   GRU + RGCNConv (+ optional SAGEConv map)
   → original 6d next-node PUBLIC status head output
                          + current 48d public service query
   → ordinary MLP / BCE conditional service completion probability
   → original PR67 lawful WorkflowDynamics world-state projection
```

**重要区别**：新 head 不是直接从 **RGCN 中间高维关系 embedding** 读取，而是从原来的**6d 已压缩状态输出**读取；同时原图主干被冻结，J2 下一步服务事件监督不会修改原图的关系表示。因此本轮只能严谨判断：**冻结原 PR61 状态输出 + 48d 服务查询，在这个特定 H1 条件事件任务上尚未增加优于简单已训练/经验模型的价值**。不能据此断言原 RGCN 本身对**全系统任务、设备、资源与路径状态预测**没有价值，更不能说扩增参数必然有用。

PR61 原预训练任务针对绝对 next-state，PR73 J2 才引入 keyed 0–2 tick 服务随机性；当前目标迁移有实质差异。地图信息也可能主要对交通拥堵和未来资源状态有价值，而非受当前公开服务进度主导的单步 completion hazard。需要严格按研究目标判断。

## 4. 截止此报告的 GO / NO-GO

- **PASS（工程）**：原权重身份在受控服务器上由 GPU 执行方报告核验、六支 GPU head-only 训练完成、训练结果已提交脱敏汇总、旧权重不公开。
- **NO-GO（当前 H1 结论）**：在“冻结原图 + 6d next-node 输出 + 当前 48d 服务查询”这个固定训练设置下，**没有显示超越 Deep Sets 和 TRAIN-only hazard 的平均 H1 优势**；也没有稳定的 map_on 优于 map_off 证据。
- **GO（科学诊断）**：执行 [#82](https://github.com/lieuyimo/wm/issues/82) 的 12 场景配对统计、NLL/ECE/分相位/分 regime 与错误归因；执行 [#85](https://github.com/lieuyimo/wm/issues/85) 的 `48d public query-only` vs `54d frozen GNN` 公平输入信息消融，审计原 GNN 6d 瓶颈与是否有必要迁移中间表示。优先审计已有结果，不增加 GPU 训练规模。
- **暂缓** [#83 H5/H20](https://github.com/lieuyimo/wm/issues/83) 和 [#84 调度收益](https://github.com/lieuyimo/wm/issues/84) 作为主研究结论；待模型输出、分布估计和世界模型真实预测目标得到合理解释后才放行。不能用 H1 事件损失失败直接否定所有多步世界状态性能，但也不能越过检验门槛盲目推演。

**缺失的可独立核验研究证据**：执行方公开报告仅有每支 aggregate 风险与宏 Brier、取样/交付 Brier，**没有**6×12逐场景 paired row、每场景 NLL/ECE、配对 bootstrap 95% CI。未经受控执行方提供审查过的这些纯统计结果，不能宣称地图优劣或 GNN vs baseline 差异在统计上显著。**权重文件完全不需要也不应上传 GitHub 进行该评审**。
