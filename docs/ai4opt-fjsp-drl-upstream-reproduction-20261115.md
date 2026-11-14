# AI4Opt #2：Song et al. FJSP-DRL 原始 HGNN+PPO 公开基准复现

**状态**：实际第一轮实现链路 PASS；第二轮 **3 seeds × 120 PPO updates × 100 official tests** 运行于 [GitHub Actions #43640036522](https://github.com/lieuyimo/wm/actions/runs/43640036522)，**本节第二轮数值尚未写入**。任何失败 CI 应区分“模型计算已成功、后续路径/产物保护断言误报”和“训练本身失败”。所有模型权重不保存、不上传、不发布。

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

## 第二轮验证

[Actions #43640036522](https://github.com/lieuyimo/wm/actions/runs/43640036522)，3 个原始训练 seed **1810/1811/1812** 各 **120 updates × 8 同步训练回合**，各独立复测全部100个上游官方10×5文件，输出 per-case / mean makespan、PPO参数实际更新与所有Gantt合法率、CPU端到端wall seconds。不保存权重，使用同一公开代码SHA，seed之间不共享训练状态；待 CI 完成后填入结果。

**研究边界**：上游原实现是一个**神经策略**，不是 `p(s_{t+1}|s_t,a_t)` 世界模型；其性能不构成对 PR78 世界模型预测/决策能力的直接验证。后续若在自有临床仿真中使用 FJSP-DRL 必须先完成 AI4Opt #87 FJSP-AGV 可选加工/搬运任务动作合同，再适配到相同 CP-SAT/BusinessSimulator 合法性检查与时间预算。

## 源码与复现入口

- 我方复现程序：[`experiments/ai4opt_reproduction/fjsp_drl_repro.py`](../experiments/ai4opt_reproduction/fjsp_drl_repro.py)
- 无权重 Actions：[workflow](../.github/workflows/ai4opt-fjsp-drl-public-benchmark-reproduction.yml)
- Draft 实验 PR：[PR #91](https://github.com/lieuyimo/wm/pull/91)
- 关联研究任务：[AI4Opt #88](https://github.com/lieuyimo/wm/issues/88)

绝不将公开作者的预训练模型权重转存到本公开仓库，亦绝不上传本实验的新策略权重。
