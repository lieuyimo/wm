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
- 训练配置从原作者 `config.json` 复用：学习率 2e-4、PPO `K_epochs=3`、`eps_clip=0.2`、原网络宽度与损失；第一轮缩短为 8 updates/batch2，第二轮为 120 updates/batch8。**相比原配置的1,000 updates/batch20，这是有限预算兼容性/初步效果复现，不是论文全量收敛复现**。论文预训练参考由第三轮仅调用作者已经公开的原本权重推理得到。
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

## 第三轮：原作者已公开10×5预训练模型（原始权重在隔离环境只读）

**实际运行 SUCCESS**：[官方权重参考评测 Actions #43641048583](https://github.com/lieuyimo/wm/actions/runs/43641048583)，原始作者源码 commit `d7637b70029be6cb8fe78e450dcb07767876daea`；仅从作者本人已公开的官方 GitHub 获取官方**第三方公开权重** `model/save_10_5.pt`（大小 **849,961 bytes**；SHA256 `594f93a9f9838f89bf508297b3019ebaae8a0482a6f3abf91a3be7412198f0ce`），使用 `torch.load(weights_only=True)` 与原 `PPO_model.PPO` 严格一致性加载。

**隐私与归档边界**：原作者权重仅暂时存在 CI 运行器 `$RUNNER_TEMP`，加载推理后由工作流删除，未复制到本用户仓库、没有 upload-artifact、没有训练任何新 checkpoint。此原作者公开权重**不是**用户私有的 PR61/PR78 模型；本实验中所有我方权重从未公开上传。

同样 **100 个原作者 `data_test/1005` 测试实例**、同一 `FJSPEnv.validate_gantt`：

| 预训练发布模型与规则 | 100实例平均 Makespan | Gantt 合法 | CPU end-to-end 100实例 wall time |
|---|---:|---:|---:|
| SPT 公开经典最短加工时长优先 | **130.09** | 100/100 | **7.01 s** |
| **原作者公开的 pretrained HGNN+PPO** | **111.67** | **100/100** | **18.94 s** |

- 对 SPT 平均 Makespan **−18.42（−14.159%）**；在100个相同任务中，作者 pretrained **93 更优、5 更差、2 打平**。按测试任务配对有放回 bootstrap 6000次，以固定的模型权重/种子为条件，原作者模型−SPT 95% CI 约 **[−21.18,−15.61]**。这个CI代表这100个公开样本的**案例抽样差异**，不代表不同训练种子或真实医院现场的泛化置信区间。
- 作者 pretrained 的 CPU 推理时间约是 SPT 的 **2.70 倍**（18.94 / 7.01），但Makespan更低；这是可核验的方案质量与额外计算时间取舍。两者不应被解读为线上 AGV 滚动调度 P95 时限。
- 本研究现在已实际复现**原始发表算法代码训练可执行性、原环境合法性、官方公开参考权重的推理表现**；并已明确短训练预算120 updates无法复现论文的最终质量。这比只看一个随机初始化HGNN好坏更完整、且没有假造收敛训练结果。

**最终复现判断**：
1. **PASS**：原作者 HGNN/PPO 的原始实现可在当前 PyTorch2.5.1 CPU 依赖下成功训练与推理，标准FJSP操作-机器合法性检查全部通过；作者发布的原始预训练模型在100个标准10×5实例上平均 Makespan 优于 SPT。
2. **NO-GO**：用仅120 PPO updates、batch8、三个种子的短训表现替代原作者1,000次updates、batch20的完整训练结果。短训三种子平均135.97比SPT130.09差4.52%，不能声称“复现论文训练性能”或“证明模型无效”。
3. **NEXT**：在 #87 严格定义 FJSP加工 + AGV搬运的合法动作和全TAT问题后，才进行与我方CP-SAT/BusinessSimulator的业务迁移；这需要新的有对照研究，不是直接把官方FJSP策略接到现有样本运输后即可认定收益。

**研究边界**：上游原实现是一个**神经策略**，不是 `p(s_{t+1}|s_t,a_t)` 世界模型；其性能不构成对 PR78 世界模型预测/决策能力的直接验证。后续若在自有临床仿真中使用 FJSP-DRL 必须先完成 AI4Opt #87 FJSP-AGV 可选加工/搬运任务动作合同，再适配到相同 CP-SAT/BusinessSimulator 合法性检查与时间预算。

## 源码与复现入口

- 我方复现程序：[`experiments/ai4opt_reproduction/fjsp_drl_repro.py`](../experiments/ai4opt_reproduction/fjsp_drl_repro.py)
- 无权重 Actions：[workflow](../.github/workflows/ai4opt-fjsp-drl-public-benchmark-reproduction.yml)
- Draft 实验 PR：[PR #91](https://github.com/lieuyimo/wm/pull/91)
- 关联研究任务：[AI4Opt #88](https://github.com/lieuyimo/wm/issues/88)

绝不将公开作者的预训练模型权重转存到本公开仓库，亦绝不上传本实验的新策略权重。
