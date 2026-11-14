# AI4Opt 首个论文复现：Song et al. FJSP-DRL HGNN + PPO

**论文**：Wen Song, Xinyang Chen, Qiqiang Li, Zhiguang Cao, *Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning*, IEEE Transactions on Industrial Informatics 19(2):1600–1610 (2023), [DOI:10.1109/TII.2022.3189725](https://doi.org/10.1109/TII.2022.3189725)。

**复现结论**：**PASS 作者原公开 HGNN+PPO 预训练模型的正式推理测试**，在作者原版数据 `data_test/1005` 全量 **100/100** 实例中输出合法 FJSP 调度；另与固定 SPT/LPT 调度规则及首10实例 OR-Tools CP-SAT 作了完全可核对的模型质量/时间对照。**这不是从头复现论文的 PPO 训练，也不是 Gebreyesus STGNN、DreamerV3 或 OPT-Agent 的实验复现。**

## 1. 可检验的源数据和运行身份

| 项目 | 本轮执行 |
|---|---|
| 官方仓库 | [songwenas12/fjsp-drl](https://github.com/songwenas12/fjsp-drl)，Apache-2.0 |
| 冻结上游 Git Commit | `d7637b70029be6cb8fe78e450dcb07767876daea` |
| 已公开原作者预训练模型 | `model/save_10_5.pt`，文件身份 SHA256 `594f93a9f9838f89bf508297b3019ebaae8a0482a6f3abf91a3be7412198f0ce` |
| 测试集 | 原仓库 `data_test/1005/*.fjs`，**100** 实例，每例 10 jobs × 5 machines × 52 operations；按源文件名称排序全量测试，没有根据结果挑选 |
| 随机种子 | 1810；作者 `PPO_model.PPO` 的 `policy_old.act(..., flag_sample=False, flag_train=False)` 贪心推理 |
| 环境 | GitHub Actions Ubuntu24.04, Python3.10, PyTorch2.5.1+cpu, NumPy1.26.4, Gym0.26.2, CPU torch threads=1 |
| 无侵入复用 | 直接调用**原 `FJSPEnv`、`PPO_model`、公开 state_dict**。独立适配入口跳过原 `test.py` CPU 上也初始化 NVML 的逻辑与新版 pandas 已删除的 `ExcelWriter.save`，不修改原HGNN/PPO参数/求解算法 |
| 实际运行 | [Actions #43639821299：SUCCESS](https://github.com/lieuyimo/wm/actions/runs/43639821299) |
| 原始逐例统计 | [100×3 +10 CP-SAT 公开指标 JSON](ai4opt-fjsp-drl-paper-repro-public-results.json)，不含任何模型张量、用户私有数据或恢复 checkpoint |
| 权重保护 | 仅在临时 CI 执行容器从作者**原本已公开的仓库**读取第三方权重；原 PR61/PR78 私有模型从未加载/上传；**没有任何 upload-artifact/Release/权重保存步骤** |

## 2. 作者公开预训练模型的全量 100 实例复现

100 个原作者 FJSP 测试实例，三种策略全部按**同一原 FJSPEnv**执行并验证 `validate_gantt()[0]==True`；各调度均完成 52 道工序。

| 方法 | 100例平均 makespan (↓) | 单例平均决策总时长 | 合法率 |
|---|---:|---:|---:|
| **HGNN + PPO，原作者 pretrained greedy** | **111.67** | **0.19067 秒** | **100/100** |
| SPT，合法操作-机器对的最短加工时间规则 | 130.09 | 0.06427 秒 | 100/100 |
| LPT，合法操作-机器对的最长加工时间规则 | 139.84 | 0.06477 秒 | 100/100 |

- **HGNN-PPO 比 SPT**：平均 makespan 差 **−18.42**，相对 SPT 的平均 makespan 降低 **14.16%**；逐例 **93 更好、2 相同、5 更差**。基于100个原始实例、固定LCG bootstrap seed=20261115 进行10,000次成对重采样，**平均差95% bootstrap CI [−21.17, −15.63]**。此区间表示从这100个案例的经验样本重复抽样的稳定性，**不能替代其他规模/OOD/真实实验室中的模型验证**。
- **HGNN-PPO 比 LPT**：平均 makespan 低 28.17。平均CPU推理耗时约为 SPT 的 **2.97×**，不能表述成“任何算力上都更快”。
- SPT/LPT 是经原 `FJSPEnv` 的**已合法动作候选**选择加工时间最短/最长机器-工序对的标准规则。它们是本次额外控制组，**不宣称是论文表格中原作者全部启发式配置的精确重跑**。

## 3. CP-SAT 约束求解参考（同前10例、每例2秒）

在同一原作者 FJSP 文件上复用标准 [OR-Tools CP-SAT optional-interval flexible job shop](https://developers.google.com/optimization/scheduling/job_shop) 方法：每个工序选恰好一个可选机器、固定机器时长、同job工序前驱、每机器 `AddNoOverlap`、最小化 `C_max`，单个 solver 单 worker、seed1810、最多2秒/实例。所有 returned schedule 由独立代码再次检查**机器无重叠、工序顺序、makespan等于返回的调度完工时刻**。本代码是透明 FJSP CP-SAT 基线，并非作者原 PPO 或临床 AGV 系统。

| 前10相同实例的指标 | 原 HGNN-PPO | SPT | LPT | CP-SAT (2秒限制) |
|---|---:|---:|---:|---:|
| 平均 makespan (↓) | 114.5 | 136.2 | 142.6 | **102.9** |
| 合法率 | 10/10 | 10/10 | 10/10 | 10/10 |
| CPU平均时间 | 约0.198秒（全量run前10） | 约0.06秒 | 约0.06秒 | **2.002秒** |

这10个例子里 CP-SAT 结果对 HGNN-PPO **10/10** 较小，平均差 **−11.6**。但所有 CP-SAT `status=FEASIBLE`，**0/10 证明最优**；求解器给出的最优下界与已有解之间平均相对剩余 gap 约30.17%（此指标非论文普遍最优 gap），不能把得到的102.9说成该规模真实 optimum。不同计算时间与预算下，结论应写作质量-时间折中，不应宣称公平时间预算下 CP-SAT 支配 HGNN。

## 4. 可重跑代码

```bash
git clone https://github.com/songwenas12/fjsp-drl.git /tmp/paper-fjsp-drl
git -C /tmp/paper-fjsp-drl checkout --detach d7637b70029be6cb8fe78e450dcb07767876daea

# 安装与 GitHub Actions workflow 相同的固定版本 CPU 依赖后：
python experiments/ai4opt_reproduction/fjsp_drl_pretrained_eval.py \
  --upstream /tmp/paper-fjsp-drl --count 100 --seed 1810

python experiments/ai4opt_reproduction/fjsp_cp_sat_baseline.py \
  --upstream /tmp/paper-fjsp-drl --count 10 --seconds 2 --seed 1810
```

实际版本与安装方法固定于 [AI4Opt Actions workflow](../.github/workflows/ai4opt-paper-fjsp-drl-public-baseline.yml)。由于作者已有公开预训练 checkpoint，本轮**没有再训练**也**未将原权重复制到我方公开仓库**。执行源代码在 `experiments/ai4opt_reproduction/`；所有实例数值见上一节 JSON。

## 5. 不能夸大的结论与论文迁移边界

1. **完成的是公开 pretrained HGNN/PPO 推理复现和标准规则/CP-SAT对照，不是端到端论文全表训练复现**。原作者论文大量训练与多规模/多基准实验，本次只覆盖作者公开 `10 jobs×5 machines` 的完整单组 test，使用公开权重。现有实验不证明重新训练得到完全相同模型。
2. **当前 `wm` 的 CP-SAT/BusinessSimulator 不是完整 FJSP**。它已有多机器人、多阶段取送和容量，但仍无完整可选临床分析机台加工操作；因此不能把标准 FJSP 的111.67 makespan迁移为临床实验室平均 TAT/流转时间或称模型已在我方业务上有效。
3. **标准 FJSP 不包含实验室 AGV 的路径冲突、运动时间、故障与动态任务释放**。必须先通过 [#87 数学建模](https://github.com/lieuyimo/wm/issues/87) 明确这些约束后，才可以把 [#88 HGNN策略](https://github.com/lieuyimo/wm/issues/88) 接到 `wm` 的合法动作/评价接口。
4. **不以外部成功代替 PR78世界模型结果**：HGNN-PPO 是直接调度策略，PR61/78 是未来状态预测器，两者目标不同；下一步需要在 #89/PR78 独立验证世界模型 H5/H20 真正决策价值。
5. 原作者公开checkpoint来自第三方已有公开发布；**你们 PR61/78、新训练的任意模型权重一律继续保留在授权私有存储**。没有创建公开模型 checkpoint Artifact。

**实证阶段判断**：在作者原始FJSP实例上可行、可复现，并已有学习策略相对简单规则的质量优势；进入与临床动态调度系统的接口适配前，#87数学约束是必需的。后续 #88 优先从可复用实现开始，不重复发明 HGNN/PPO；所有新业务训练需单独受控环境与私有checkpoint。
