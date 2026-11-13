# PR67 下一阶段：PR61 训练检查点恢复与真实神经递推验证任务清单

> 当前状态：PR67 CPU 合法神经服务状态接口已经验证；**本任务尚未执行**。先恢复原模型权重并开展 CPU 推理，只有证据充分时才按独立授权进入 GPU 训练。不得将该任务清单当作已经完成的实验记录。

## 一、必须先拿到的真实输入

仓库：`https://github.com/lieuyimo/wm`

从 PR67 当前 Draft HEAD 开始，执行前核对最新 SHA，不能覆盖后续提交，不合并 PR64/65/66/67，不改 PR19。

原 PR61 报告记录 **6,661,520 字节**的 `pr60-gpu-checkpoints-20261112.tar.gz`（best/last/Adam/RNG），SHA256：
`919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14`。

此归档为先前 A100 agent 的**本地归档而非 Git 源仓库文件**。禁止在没有文件的情况下猜路径、伪造模型、用随机初始化权重冒充已训练模型或将不同 PR 的检查点混用。若无法获取该归档，停止 checkpoint 验证，并在 PR67 留下具体缺失输入与恢复方法；不得擅自用 GPU 补训作为替代。

已有已验证原始 CPU 数据：PR64 GitHub Actions #43264954353 的 TRAIN 24 / DEV 12 与 risk-set quality 三个原始 SHA 归档。DEV 已经用于多轮方法开发，**不是未见过的最终测试集**。PR59 原始未打开的 OOD BLIND 测试集必须保持未读。

## 二、恢复后的 CPU 测试

1. 检查 SHA、tar 条目目录、模型结构和训练配置、checkpoint 中的 `architecture`、`arm`、`seed`、`hidden`、`source_sha256`、`git_head_source`、`model_state_dict`、`optimizer_state_dict`、CPU/CUDA RNG，并记录归档路径与副本 SHA。不得覆盖已有 best/last。
2. 复用原 `experiments/world_model_integration/sealed_pr59/pr59_dual_model.py` 中的 `PublicMapTemporalRelationalWorld(hidden=48)` 构造，在 CPU `load_state_dict(strict=True)`，分别验证 map_on/off 3 个 seed，尽可能复算 PR61 TRAIN/VAL 的一阶 node MSE、device Brier 和新任务 MAE，核对已发表数值；必须区分 teacher-forcing 与 rollout。
3. 检查 PR67 `neural_pr67/pr67_neural_world.py` 的公开节点/任务会话映射和 PR66 `stochastic_pr66/pr66_rule_rollout.py` 的合法状态转移：每步都执行真实训练好 PR61 的 forward，不得替换为随机网络。冻结模型权重，比较相同场景和随机条件下：纯规则；原 PR61 map_off；原 PR61 map_on；所有三组采用相同外生 hazard 和合法约束。针对 H5/H20，报告任务阶段完成、服务等待、资源拥堵、实体状态差距、设备事件、合法率、event Brier/log-loss、种子重复性和每个候选评分变化。
4. 先在未用于最终测试的 developer cohort 做代码 debug。以**场景**为独立样本，配对 bootstrap 或明确的场景级置信区间；稀有故障事件要同时报告事件计数和分组表现。不因模型在少量轨迹上的表现临时调阈值、预测窗口、分桶或训练损失。
5. 保存：原 tar 与 SHA256、pip freeze、Python/Torch/PyG/CUDA 环境、显卡、具体源 HEAD、训练配置及锁文件、随机种子、调用命令、模型权重与优化器/RNG、未裁剪完整评测轨迹、汇总 JSON/CSV 和审计日志。用新的带时间戳 evidence 子目录；不要覆盖 PR61 或 PR64 原有证据。

## 三、是否需要 GPU 继续训练

恢复并验证模型之前，**不需要 GPU**。如果实际预训练 PR61 模型的任务服务事件校准明显不佳，需首先判明是否来自任务状态概率与服务事件解释的语义不一致、错误归一化、rollout distribution shift、缺少服务历史或事件样本稀少。禁止凭单次随机网络 H20 就宣布欠拟合或扩大模型参数。

只有在明确了所需可学习信号、训练/验证独立拆分、主指标与对照组，且 reviewer 明确 GO 后，再启动 GPU agent。GPU 训练应尽量复用原 PR59/61 训练器、完整 best/last/RNG 归档和 3-seed、map_on/off 对照；若需要新的 service hazard head，须说明其对应公开实现、输入标签、作用范围、损失和消融，不将新头加入运动路径生成或动作选择。

## 四、给执行 agent 的可直接复制 Prompt

```text
任务：PR67 已通过的关系世界模型神经条件合法递推，在真实 PR61 训练权重下完成可复现 CPU 验证；只有独立评审 GO 后才进行 GPU 训练。

仓库 https://github.com/lieuyimo/wm
基线 PR #67（stacked PR #66）；开始前检查最新 HEAD，绝不覆盖已有后续提交。所有 PR 保持 Draft，不合并 PR，不触碰 PR #19。禁止改旧训练检查点、改 PR59 BLIND 或将 PR64 developer VAL 当作独立未知测试。

第一优先级：从上一 GPU 环境获取 pr60-gpu-checkpoints-20261112.tar.gz，校验文件大小 6661520 bytes 和 SHA256 919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14，核对所有 best/last/Adam/RNG 与 PR61 的 source/arm/seed/环境。若文件不存在，明确记录 BLOCKED 并停止，不要用随机模型或新训练冒充旧权重。

第二优先级：复用原 PR59/61 的未修改模型结构，CPU 加载 map_on/map_off 三个训练 seed，复算独立 VALIDATION 单步指标；之后接入 PR67 PublicGraphHistory + NeuralTaskServiceRule + PR66 PublicHazardRuleRollout，在每步真正执行已训练模型 forward，并经过原 check_business_transition。用相同的场景、动作候选、预测时域、外生随机种子，比较纯规则、map_off 和 map_on 的 H5/H20 合法完整公开系统状态，报告每个场景的任务状态、服务阶段事件、资源/等待、运动位置与设备可用性偏差；报告稀有事件数、场景配对 CI 和模型运行时间。不要把 teacher forcing 指标误当闭环结果。

第三优先级：记录全部输入 SHA、运行 HEAD、完整 CPU/GPU 环境、训练权重、RNG、命令及原始/聚合结果，输出文档和 PR67 评论。不要开启 BLIND 或 GPU 训练，除非已完成明确审查并获得独立授权。若有研究性性能瓶颈，请明确归因并提出引用公开实现的最小后续训练方案，不自行造算法，不写防御性叙述。
```

结论：当前要验证的是**训练好 PR61 世界模型的实际闭环预测价值**。未拿到原训练权重之前，继续添加参数或启动 GPU 新训练没有必要。
