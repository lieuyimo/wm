# AI4Opt #2：Song et al. FJSP-DRL 原始 HGNN+PPO 公开基准复现

**状态**：第一轮原 HGNN+PPO 真代码 smoke 通过；第二轮 [GitHub Actions #43640036522](https://github.com/lieuyimo/wm/actions/runs/43640036522) **SUCCESS，三个种子各120 PPO updates/100 official tests 均已完成**。本实验新训练权重只在运行时内存出现，未保存或发布。另有第三阶段“原作者已公开 pretrained 10x5 官方模型”在一次性外部参考实验中独立评测，仅用于检查与论文发布性能的关联，不向本公开仓库提交第三方或我方权重。

## 来源与版本锁定

- 文献：Song, Chen, Li, Cao，*Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning*, IEEE Transactions on Industrial Informatics 19(2):1600–1610 (2023)；DOI [10.1109/TII.2022.3189725](https://doi.org/10.1109/TII.2022.3189725)。
- 原作者 Apache-2.0 项目 [songwenas12/fjsp-drl](https://github.com/songwenas12/fjsp-drl)，**原始固定 commit `d7637b70029be6cb8fe78e450dcb07767876daea`**。没有把自造模型当作者论文模型。
- 通过 sparse checkout 仅获取上游源代码 `PPO_model.py` / `graph/` / `env/` / `mlp.py`、公开 `data_test/1005/` 测试实例，**排除上游公开 `model/*.pt`、`results/*.pt`**。
- 环境：Ubuntu GitHub Actions 临时 CPU；Python 3.11.18，PyTorch **2.5.1+cpu**，NumPy **1.26.4**，Gym **0.26.2**，Matplotlib **3.9.2**。不调用任何私有 PR61/PR78 训练权重或模型推理。

## 方法与实验设计

- **严格使用上游原始** `CaseGenerator`、`FJSPEnv`、`PPO_model.PPO`、HGNN forward、PPO 原 loss/optimizer 和 `validate_gantt`。
- 兼容旧环境的差异**只在我们独立的 harness**：直接调用原 `FJSPEnv`，不使用新版 Gym wrapper，绕开原 `train.py` 已过时的 Pandas ExcelWriter / Visdom 存盘接口。并未改上游网络架构、损失、梯度、状态/动作空间或真实原 FJSP 规则。
- 任务是**原作者标准 flexible job shop processing**：可选加工机台、固定工序前驱、机器资源；当前**没有**我方 AGV 样本运输、设备停机、临床流程等附加约束，不能把这次基准当作完整 FJSP-AGV 效果。
- 固定 10 jobs × 5 machines FJSP 论文公开测试数据，按原仓库 `data_test/1005/` 文件名排序，所有方法使用**同一测试集**；比较标准最短加工时间优先 **SPT**、随机初始化的 HGNN+PPO、指定训练步数后的 HGNN+PPO。原 `FJSPEnv.validate_gantt` 必须在**每个训练回合和测试实例**通过。
- 训练真实 PPO 参数更新，权重**只存在内存中**，每轮检查 `state_dict` 张量变化，禁止伪造训练或随机权重代替已训练模型报告。
- 训练配置从原作者 `config.json` 复用：学习率 2e-4、PPO `K_epochs=3`、`eps_clip=0.2`、原网络宽度与损失；第一轮缩短为 8 updates/batch2，第二轮为 120 updates/batch8。**相比原配置的1,000 updates/batch20，这是有限预算兼容性/初步效果复现，不是论文全量收敛复现**。
- CI 不使用 `torch.save`、不上传任何 `*.pt` 或模型权重 Artifact，输出仅原公开 benchmark 的 makespan、合法性、公开源哈希与非敏感运行耗时。仓库其他历史 evidence 中已有旧 `*.pt`，这次只检查**新的实验目录和外部 source sparse checkout**，不处理无关历史文件。

## 第一轮实际执行（seed=1810，8 PPO updates，12 官方 10×5 测试）

[CI #43639716729](https://github.com/lieuyimo/wm/actions/runs/43639716729)：复现脚本在日志输出 `AI4OPT_FJSP_REPRO_OK`，真 PPO 训练 **8 次迭代**、16 个合法训练回合，累计 806 条环境决策步；每次优化器更新改变 **82 个模型 state_dict 张量**。后续 CI **失败属于误把旧 repo `examples/evidence/*.pt` 也计算为本次新增权重**，不是模型运行/调度失败。此误报已被固定后的 CI 严格限定为当前实验目录。

| 模型（同12官方测试例） | Mean makespan（越低越好） | 全部 Gantt 合法 |
|---|---:|---:|
| SPT（最短加工时长合法派工） | **134.66667** | 12/12 |
| 原 HGNN+PPO 随机初始化前 | 149.25000 | 12/12 |
| 原 HGNN+PPO 8 个 optimizer updates 后 | 152.41667 | 12/12 |

**阶段解释**：原算法神经网络/环境/损失及状态合法性能够正确运行，但 8次更新不足以建立策略质量；此时未超过 SPT 是观测数据，不能当成复现论文最终效率或论证神经调度一定较差。

## 第二轮：3 个 seed、120 updates、全部100个官方10×5测试实例（正式成功）

**证据**：[GitHub Actions #43640036522 SUCCESS](https://github.com/lieuyimo/wm/actions/runs/43640036522)，固定原源 SHA `d7637b70029be6cb8fe78e450dcb07767876daea`，PyTorch2.5.1+cpu、Gym0.26.2、NumPy1.26.4；`seed=1810/1811/1812` 三个独立训练过程各执行120次 `PPO.update`、batch8，`K_epochs=3`、Adam lr2e−4、eps_clip0.2；**每一次更新的82个 model state_dict 张量实际变化**。三个 seed 合计 **2,880 个合法训练回合 / 143,624 环境决策步骤**，全部使用原 `FJSPEnv.validate_gantt` 校验。

| seed | SPT | HGNN+PPO 随机初始化 | HGNN+PPO 120 updates | 训练后 − SPT | CPU 训练 wall 秒 | 测试解合法 |
|---|---:|---:|---:|---:|---:|---|
| 1810 | 130.09 | 145.53 | **138.40** | +8.31 | 351.68 | 100/100 × 3 方法 |
| 1811 | 130.09 | 126.79 | **140.62** | +10.53 | 200.96 | 100/100 × 3 方法 |
| 1812 | 130.09 | 118.39 | **128.90** | −1.19 | 199.84 | 100/100 × 3 方法 |
| **三种子等权平均** | **130.09** | **130.23667** | **135.97333** | **+5.88333 (+4.52%)** | 总计 752.49 | **900/900 测试解合法** |

- **训练效果并不稳定**：相对于自己未训练的原网络，seed1810 改善 makespan −7.13；seed1811 劣化 +13.83；seed1812 劣化 +10.51。三种子中，训练后仅 1812 优于 SPT。
- 同 100 个 test case、先对三个固定模型 seed 求每场景平均，训练模型 vs SPT 的 per-case gap 为 **+5.8833**，在 100 个 case 中只有 **32 个**的三种子平均 makespan 更低，**1 个打平**。固定三种子、按 100 case 有放回 bootstrap 4000次，条件性95%区间 **[+2.81,+8.82]**。这是在**这三个短训seed条件下、这100个既有公开实例**的场景 bootstrap，**不能代替不同训练种子、不同任务分布或完整论文训练的置信区间**。
- 测试前、测试后、SPT 全部使用作者原始 FJSP Env 的相同工作量和合法性验证；不会因为非法解而虚假降低makespan。
- **NO-GO** 用此120-update短训练模型直接迁移现有 AGV 闭环或宣称超过 CP-SAT；**GO** 认为原论文的 HGNN/PPO 算法执行和评估链路可复现。后续须以**作者已公开 pretrained 参考**或论文完整训练预算独立评估来决定最终性能，而不是在短训练上反复更改算法。

## 第三轮：作者已公开 10×5 pretrained 模型参考（隔离运行）

只有来自原作者官方 GitHub 的`model/save_10_5.pt`、作者在外部仓库已经公开（**不是我方PR61/78私人权重**），会在隔离 CI 临时目录中按固定代码 SHA/原文件大小校验，用 `torch.load(weights_only=True)` 仅运行同100官方测试案例；执行后从临时目录删除，不缓存、不在用户仓库提交、不上传 Artifacts。这个对照用于区分“有限预算短训练未收敛”与“作者公开训练模型的实际性能”，不能解释为我们从头复现了作者千轮训练。参考 [Actions #43641048583](https://github.com/lieuyimo/wm/actions/runs/43641048583)，最终数值待该独立CI完成后填写。

**研究边界**：上游原实现是一个**神经策略**，不是 `p(s_{t+1}|s_t,a_t)` 世界模型；其性能不构成对 PR78 世界模型预测/决策能力的直接验证。后续若在自有临床仿真中使用 FJSP-DRL 必须先完成 AI4Opt #87 FJSP-AGV 可选加工/搬运任务动作合同，再适配到相同 CP-SAT/BusinessSimulator 合法性检查与时间预算。

## 源码与复现入口

- 我方复现程序：[`experiments/ai4opt_reproduction/fjsp_drl_repro.py`](../experiments/ai4opt_reproduction/fjsp_drl_repro.py)
- 无权重 Actions：[workflow](../.github/workflows/ai4opt-fjsp-drl-public-benchmark-reproduction.yml)
- Draft 实验 PR：[PR #91](https://github.com/lieuyimo/wm/pull/91)
- 关联研究任务：[AI4Opt #88](https://github.com/lieuyimo/wm/issues/88)

绝不将公开作者的预训练模型权重转存到本公开仓库，亦绝不上传本实验的新策略权重。
