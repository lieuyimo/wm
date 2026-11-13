# PR69：事实动作/设备条件下的多步预测评估

日期：2026-11-13  
PR69 head：`f95c16e`  
来源：PR68 已验证的原始 PR61 `best.pt`、PR59 validation、PR64 developer validation。**CPU inference only；无训练/微调/参数更新；无 BLIND/TEST。**

## 1. 评估设计

- 12 个 PR64 developer-validation 场景，每个场景从 `t=0` 开始。
- H5 / H20 两档；3 个原始 PR61 seeds（1810/1811/1812）；arms：Rule / map_off / map_on。
- 每一步使用**原始事实 CP-SAT 动作**和**原始下一时刻公开设备状态**作为条件；模型不使用自己的动作策略，也不预测设备外生事件。
- 对比当前原始的 rule transition 基线；无重训练。
- 这是**条件多步预测**，不是自主策略、不是 factual H20 ground-truth accuracy，也不是在线调度收益。

## 2. 结果

### H5：全部跑满 5 步

| arm | full 5/5 | legal steps mean | task status acc | robot position acc | completed count MAE |
|---|---:|---:|---:|---:|---:|
| rule | 36/36 | 5.00 | 0.9583 | 1.0000 | 0 |
| map_off | 36/36 | 5.00 | 0.9583 | 1.0000 | 0 |
| map_on | 36/36 | 5.00 | 0.9583 | 1.0000 | 0 |

- 三臂在 H5 的终态/逐步指标完全相同；没有看到神经模型相对 rule 的增量。
- 0.9583 的 status accuracy 来自 2 个场景中的新阶段/新样本未出现在 rule/neural 的递推状态中；三臂同样缺失（每臂 51 个缺失 sample-step），不是神经模型独有的错误。
- Stepwise status accuracy：step 0–1 = 1.0，step 2 = 0.9583，step 3–4 = 0.9167；robot position 始终 1.0，completed count 误差 0。

### H20：递推覆盖受限

| arm | full 20/20 | mean legal steps | task status acc* | robot position acc* | completed MAE* | errors |
|---|---:|---:|---:|---:|---:|---|
| rule | 30/36 | 19.17 | 0.8353 | 1.0000 | 0 | 6 × action missing sample |
| map_off | 9/36 | 14.31 | 0.8296 | 0.9994 | 0.0466 | 25 × action missing sample, 2 × illegal assignment |
| map_on | 9/36 | 14.28 | 0.8301 | 0.9981 | 0.0506 | 25 × action missing sample, 2 × illegal assignment |

\\* 只在已走通的前缀上计算；不能当作完整 H20 指标。

- 神经模型在 H20 只有 9/36 个场景走满；rule 基线也只有 30/36。两者共享“新阶段/新 sample 未在当前递推状态中生成”的限制。
- 失败模式主要是事实动作引用了预测状态中不存在的 sample（`action_references_missing_sample`），另有 2 个 `assignment must select a distinct idle robot and currently waiting sample`。
- map_off 与 map_on 在 coverage、误差和失败模式上几乎相同；没有 map-side 多步优势。
- 在已走通的 prefix 上，模型 status accuracy 与 rule 接近；没有证据支持神经模型改善多步状态预测。

## 3. 判读

1. **H5 条件多步 rollout 可执行**：36/36 三臂都完成 5 步，机器人和完成计数完全对齐；但神经模型与 rule 完全一致，没有增量。
2. **H20 不能作为完整预测指标**：多数神经 run 在约第 9–17 步因事实动作引用缺失 sample 而停止；rule 也在 6 个 run 上遇到同类问题。
3. 主要瓶颈不是训练不足，而是**多步状态递推中的新任务/新阶段生成和状态对齐**；当前 `NeuralTaskServiceRule` 使用 rule transition，无法凭空生成事实 future 的新 stage。
4. 不据此重训、扩参数或宣称 H20 预测/调度收益。下一步应先补齐合法的 new-stage/entity lifecycle 递归，再做完整 H20 误差与校准。
5. 本次结果不替代 PR68 的旧 PR59 teacher-forced 复现；PR68 的 6/6 权重验证仍是本评估的前置条件。

## 4. 产物

`examples/evidence/pr69-factual-multistep-evaluation-20261113/`

- `evaluation.json`：216 个 run（12 场景 × 3 seeds × 2 horizons × 3 arms）及逐步指标。
- `environment.json`：PR69 head、archive/PR59/PR64 SHA、CPU policy；引用 PR68 已验证环境。
- `command.txt`、`source_head.txt`、`SHA256SUMS`。

原始权重和训练数据未提交；无 GPU、无训练、无 BLIND。PR69 保持 Draft，不合并。
