# PR63：TRAIN+VAL 风险集与事件时间可观测性审计

范围：只读 PR59/PR60 原始 TRAIN 与 PR59 冻结 validation；**不读取/生成 BLIND/TEST，不做训练或参数更新，不使用 H20/在线调度**。本 PR 回应 PR62 review：不再把“真实事件子群上的条件 Brier”称为校准，而是在完整公开风险集上重算，并区分模型可观测与不可观测的事件时间信息。

## 1. 结论

**当前神经模型没有在完整风险集上超过 TRAIN-only 常数风险基线；也没有证据支持 onset/recovery 判别或事件驱动 H20/在线调度。**

- 完整风险集 Brier（tick 加权）：TRAIN-fitted constant `0.0040652`；`age_h4` `0.0040417`；当前 map_off `0.0042344`、map_on `0.0042009`。
- 按 12 个独立场景等权（scene-macro）并做场景 bootstrap：constant `0.0042338`；map_on `0.0043768`，配对差 `+0.0001430`，95% CI `[+0.0000521, +0.0002236]`，**map_on 反而更差**；map_off 差 `+0.0001572`，CI 跨 0。
- 模型可观测的 4 帧年龄基线 `age_h4` 与 constant 的 scene-macro 配对差为 `−0.0000300`，CI `[−0.0000650, +0.0000061]`，**没有稳定改善**。
- 精确 up/down 年龄 `age_exact` 和绝对 public tick `abs_tick` 分别改善 `−0.0003214`、`−0.0003821`（scene-macro，CI 均排除 0），说明事件时间信号存在；但这两个基线**不是当前模型输入特征**。
- 真实 onset/recovery 子群上的 Brier 对所有方法都很高（onset 约 0.69–0.996，recovery 约 0.79–0.87）；这只能说明「已知稀有事件子群的误差」，**不是校准检验**。

## 2. 数据与风险集

| 项目 | TRAIN | VALIDATION |
|---|---:|---:|
| 独立场景 | 60 | 12 |
| 真实 public t→t+1 transitions | 9,046 | 1,517 |
| 当前 UP tick / onset | 44,180 / 80 | 7,360 / 16 |
| 当前 DOWN tick / recovery | 1,050 / 80 | 225 / 16 |
| onset 的下一 public tick | 7 (60), 13 (20) | 7 (12), 13 (4) |
| recovery 的下一 public tick | 10–43 | 11–43 |

- 风险集：`x=1` 的所有 station tick（onset 风险集）、`x=0` 的所有 station tick（recovery 风险集）。
- 四相位：`healthy_hold / fault_onset / continuing_down / recovery`；相位按真实下一时刻标签划分，**只用于评估，不作模型特征或在线阈值**。
- 16 个 onset、16 个 recovery 是相关 station tick 中的稀有事件；不是 1,517 个独立样本。
- 地图 0/1 各 8 个 onset、8 个 recovery；regime 为 congestion 4、multi_fault 8、staggered_arrivals 4。

## 3. 基线（只在 TRAIN 上拟合）

| 基线 | 特征 | 权重 |
|---|---|---|
| `constant` | 无 | TRAIN 全局 onset `80/44180=0.0018108`，recovery `80/1050=0.0761905` |
| `age_h4` | 最近 4 个公开快照中的连续状态年龄 | 模型实际可见的历史窗口；最多 4 |
| `age_exact` | 从 episode 开始累计的连续 up/down 年龄 | 仅诊断；模型只看 4 帧 |
| `abs_tick` | 当前 `public_tick` | 仅诊断；输入 JSON 有 tick，但网络不使用 |

`age_h4`、`age_exact`、`abs_tick` 的 bucket 风险率只用 TRAIN 拟合，并使用 Laplace 平滑；constant 使用 TRAIN 经验风险率。**没有任何 VALIDATION 标签参与拟合。**

## 4. 结果

### 4.1 完整风险集

`all tick` 是全部 7,585 个 station tick 的加权 Brier；`scene-macro` 是 12 个独立场景等权后的均值。二者不同，因为各场景 tick 数不同。最后两列是 scene-macro 配对 bootstrap（12 场景，2,000 次，95% CI）。

| 方法 | all-tick Brier | scene-macro Brier | Δ scene-macro vs constant | 95% CI |
|---|---:|---:|---:|---:|
| constant（TRAIN） | 0.0040652 | 0.0042338 | 0 | — |
| age_h4（模型可见） | 0.0040417 | 0.0042038 | −0.0000300 | [−0.0000650, +0.0000061] |
| age_exact（诊断） | 0.0037619 | 0.0039124 | −0.0003214 | [−0.0004285, −0.0002215] |
| abs_tick（诊断） | 0.0037054 | 0.0038517 | −0.0003821 | [−0.0004888, −0.0002848] |
| map_off mean | 0.0042344 | 0.0043910 | +0.0001572 | [−0.0000468, +0.0003109] |
| map_on mean | 0.0042009 | 0.0043768 | +0.0001430 | [+0.0000521, +0.0002236] |

- map_on 在 scene-macro 上**明确差于 constant**（CI 不跨 0）；map_off 数值更差但 CI 跨 0。
- `age_h4` 是唯一模型可观测的年龄基线，和 constant 无可分辨差异；这说明 **4 帧历史没有提供稳定的 onset/recovery 风险信息**。
- `age_exact`/`abs_tick` 的改善是信息可观测性的证据，不是当前模型的成绩。

### 4.2 风险集拆分

| 方法 | UP Brier | DOWN Brier | UP log-loss | DOWN log-loss |
|---|---:|---:|---:|---:|
| constant | 0.0021693 | 0.0660801 | 0.0155346 | 0.2566908 |
| age_h4 | 0.0021694 | 0.0652876 | 0.0155448 | 0.2472085 |
| age_exact | 0.0018831 | 0.0652193 | 0.0067760 | 0.2460266 |
| abs_tick | 0.0018045 | 0.0658872 | 0.0060772 | 0.2542933 |
| map_off mean | 0.0022016 | 0.0707296 | 0.0152449 | 0.2814432 |
| map_on mean | 0.0022053 | 0.0694799 | 0.0141055 | 0.2895754 |

### 4.3 相位描述

相位条件 Brier 只描述“在已知真实事件子群上的误差”，不是校准：

| 方法 | healthy_hold | fault_onset | continuing_down | recovery |
|---|---:|---:|---:|---:|
| constant | 0.0000033 | 0.9963817 | 0.0058050 | 0.8534240 |
| age_h4 | 0.0000036 | 0.9962398 | 0.0071585 | 0.8245992 |
| age_exact | 0.0002574 | 0.7480505 | 0.0095002 | 0.7930495 |
| abs_tick | 0.0003095 | 0.6879892 | 0.0073724 | 0.8302373 |
| map_off mean | 0.0000485 | 0.9904447 | 0.0122182 | 0.8350346 |
| map_on mean | 0.0000518 | 0.9906278 | 0.0083271 | 0.8682890 |

- 所有方法的 onset/recovery 相位 Brier 都很高；一个**完全校准的极低风险预测器**也会在这些稀有事件上得到接近 1 的平方损失。因此不能从该表直接得出“校准失败”。
- 模型在 onset 上预测概率约 0.99，在 recovery 上约 0.07–0.09，基本仍是“保持上一状态”；没有学到事件提前量。

### 4.4 地图块

scene-macro Brier：

| 方法 | map 0（6 场景） | map 1（6 场景） |
|---|---:|---:|
| constant | 0.0039270 | 0.0045406 |
| map_off mean | 0.0042582 | 0.0045239 |
| map_on mean | 0.0041449 | 0.0046087 |

两地图各 6 个独立场景，bootstrap CI 较宽；没有稳定的地图特定优势。map_on 在 map 1 也不优于 constant。

### 4.5 事件时间可观测性

- PR48 protocol 的 primary outage start 为 **tick 7**，secondary 为 **tick 13**；validation 的 onset 下一 tick 也正好是 7/13。
- 模型输入是 `node_features`（16 维，无 clock）+ 4 帧 history；`public_tick` 存在于输入 JSON，但**没有被网络使用**。
- 因 onset 都在 current tick 6/12，4 帧年龄都是 4；`age_h4` 无法区分 tick 7 与 13，也无法学习绝对时间风险。
- 精确年龄/绝对 tick 诊断能显著降低 Brier，但需要 4 帧以外的公开历史或显式 clock 特征；当前模型都没有。
- 不能把「加一个 public_tick 特征」直接当作部署方案：应先验证事件时间在目标测试分布中公开、稳定，且没有把固定合成 outage 记忆化。

## 5. 判读与下一步

1. **NO-GO**：不声称已实现 onset/recovery 判别、设备预测校准、事件驱动 H20 或在线调度收益。
2. **不是“一步世界模型无效”**：PR61 的 one-step public entity state / map MSE 结果仍成立；本审计只否定事件风险判别和“低总体 Brier = 可预测故障”的推断。
3. 当前模型没有超过 TRAIN-only constant hazard；map_on 在 scene-macro 上甚至更差。不能把 PR61 的节点状态改善外推成故障/恢复能力。
4. 先处理**信息可观测性**：确认事件时间/年龄是否在真实目标分布中公开且稳定；若稳定，评估显式 clock/age 特征或显式随机外生 transition hazard，而不是继续扩大网络。
5. 增加独立 onset/recovery 事件覆盖，再做 risk-set 校准/决策价值评估；在此之前不打开 BLIND、不启动 H20、不扩参数。
6. 16 个 onset/16 个 recovery 不足以稳定估计 PR/AUC/lift；本 PR 不报告这些指标。

## 6. 产物与可复现性

证据目录：`examples/evidence/pr63-risk-set-observability-20261113/`

- `baseline_params.json`：只在 TRAIN 拟合的 constant/age/tick 风险参数。
- `risk_set_metrics.json`：完整风险集、相位、地图、regime、per-scene、reliability、bootstrap 结果。
- `observability.json`：事件基数、tick/age 分布、PR48 固定 tick、模型输入可观测性。
- `phase_metrics.csv`：逐方法/逐 seed 的相位表。
- `tests.log`：6 个 CPU 单元测试通过。
- `SHA256SUMS`：以上文件校验和。

原始数据 SHA（写入 `baseline_params.json`）：

- PR59 TRAIN `6c75e4bf…fb90b0`
- PR60 congestion `c69c7cc6…8f6094`
- PR60 multi_fault `ff905160…f1f766`
- PR60 staggered `7ee5d838…25d43b`
- PR59 validation `c7a0a008…509712`

运行：`--device cpu`，单次 26 s；**无 CUDA 训练/拟合、无参数更新、无 BLIND**。六张 PR61 `best.pt` 的 SHA 仍由原 manifest 校验；权重不提交公开 Git。

## 7. 复现命令

```bash
cd /data/wm-a100-pr63-20261113/wm
CUDA_VISIBLE_DEVICES= /data/wm-a100-20261109/venv/bin/python -m \
  experiments.world_model_integration.audit_pr63.pr63_risk_set \
  --train-source /data/wm-a100-pr60-gpu-20261112/source/pr59-train/data.json \
  --train-source /data/wm-a100-pr60-gpu-20261112/source/pr60/pr60-true-new-training-43237354137-congestion/data.json \
  --train-source /data/wm-a100-pr60-gpu-20261112/source/pr60/pr60-true-new-training-43237354137-multi_fault/data.json \
  --train-source /data/wm-a100-pr60-gpu-20261112/source/pr60/pr60-true-new-training-43237354137-staggered_arrivals/data.json \
  --validation /data/wm-a100-pr60-gpu-20261112/source/pr59-validation/data.json \
  --checkpoint-root /data/wm-a100-pr62-audit-20261112/ckpt-verified \
  --checkpoint-manifest examples/evidence/pr60-gpu-training-20261112/checkpoint_manifest.csv \
  --pr48-protocol experiments/world_model_integration/stress_pr48/pr48_protocol.json \
  --output <new-directory> --device cpu
```
