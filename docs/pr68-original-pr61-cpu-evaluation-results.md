# PR68：原始 PR61 A100 权重 CPU 准入与 H5/H20 合法递推结果

日期：2026-11-13  
PR68 head：`a9e958c16e5376f8f0ac39524af89f533184534e`  
范围：**CPU only、无训练、无参数更新、无 BLIND/TEST**。使用原始 PR61 本地归档和原始 PR59/PR64 数据；仅作真实权重加载、旧验证复现和合法 H5/H20 递推描述，不作 factual H20 精度或在线调度收益声明。

## 1. 真实权重准入

| 项目 | 结果 |
|---|---|
| 归档 | `pr60-gpu-checkpoints-20261112.tar.gz` |
| 大小 / SHA256 | 6,661,520 B / `919a0b62…30a74e14` |
| tar 成员 | 12 个 best/last，全部与 PR61 manifest 的 SHA/大小匹配 |
| 安全加载 | `weights_only=True`，CPU；无随机权重 fallback |
| 模型 | 6/6 `best.pt`，52,953 参数，hidden 48 |
| 训练来源 | `git_head_source=e32c402…`；VALIDATION SHA `c7a0a008…509712` |
| PR61 GPU 训练重跑 | 否 |

六组 best 的 epoch（1-based）分别为 `map_off` 20/18/20、`map_on` 20/18/20（seeds 1810/1811/1812）。

## 2. 旧 PR59 validation 复现

使用原 PR59 12 场景 / 1,517 条真实公开 t→t+1 转移，完全相同的 teacher-forced one-step 评测；与 PR61 已保存指标比较：

| arm | seed | node MSE | device Brier | new-task MAE | max abs Δ vs PR61 |
|---|---:|---:|---:|---:|---:|
| map_off | 1810 | 0.005236378 | 0.004396612 | 0.232947129 | ≤6.62e-6 |
| map_off | 1811 | 0.006151418 | 0.004325422 | 0.188077526 | ≤6.62e-6 |
| map_off | 1812 | 0.006214235 | 0.004451115 | 0.158434267 | ≤6.62e-6 |
| map_on | 1810 | 0.004196060 | 0.004457187 | 0.163583382 | ≤6.62e-6 |
| map_on | 1811 | 0.005582503 | 0.004354666 | 0.155636628 | ≤6.62e-6 |
| map_on | 1812 | 0.005622389 | 0.004318512 | 0.212550747 | ≤6.62e-6 |

最大绝对差为 **6.614e-6**，低于 1e-4 复现门槛；六组权重与旧验证指标一致。该验证集是原模型选择用的旧 VALIDATION，不是新的独立测试。

## 3. 合法 H5/H20 递推

使用 PR64 的原始第一场景（TRAIN `PR64_TRAIN_0_0`、DEV `PR64_VALIDATION_0_0`），在同一 CPU 推理实验中运行：

- 动作：`nearest_rule_on_each_predicted_world`；
- 原始事实策略：`PR64_true_CPSAT`（与 rollout 策略不同）；
- 外生 seed：`common_exogenous_rng_seed`；
- arms：Rule / map_off / map_on × 原 PR61 seeds 1810/1811/1812；
- horizons：5 / 20；共 **36** 个合法递推。

### 3.1 H5

两源、三 seed、三臂的 H5 终态完全一致：完成阶段 `0`、等待阶段 `11`、busy ports `2`、active services `1`、设备变化 `0`。神经臂每步均执行 forward（`neural_forward_calls=5`），active service 查询 3 次；18 个 H5 递推全部合法。

### 3.2 H20

| source | arm | completed（seed 1810/1811/1812） | waiting | busy ports | active services | device changes | active queries |
|---|---|---|---:|---:|---:|---:|---:|
| train | rule | 5 / 5 / 3 | 11 / 11 / 10 | 2 / 2 / 2 | 1 / 1 / 1 | 0 / 0 / 2 | 0 |
| train | map_off | 5 / 3 / 3 | 11 / 10 / 10 | 2 / 4 / 0 | 1 / 2 / 0 | 0 / 0 / 2 | 12 / 12 / 12 |
| train | map_on | 5 / 5 / 3 | 11 / 11 / 10 | 2 / 2 / 0 | 1 / 1 / 0 | 0 / 0 / 2 | 12 / 12 / 14 |
| dev | rule | 5 / 5 / 3 | 11 / 11 / 10 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 2 | 0 |
| dev | map_off | 5 / 5 / 3 | 10 / 10 / 10 | 2 / 2 / 0 | 1 / 1 / 0 | 0 / 0 / 2 | 12 / 12 / 12 |
| dev | map_on | 5 / 5 / 3 | 11 / 11 / 10 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 2 | 11 / 11 / 12 |

- 36/36 个递推终态 `all_states_legal=True`；资源、任务生命周期、设备边界均通过检查。
- 神经臂 `neural_forward_calls` 在 H5/H20 分别严格等于 5/20；没有跳步。
- 观察到 H20 终态差异，例如 train seed 1811 的 map_off 完成 3，而 Rule/map_on 为 5；dev seed 1810/1811 的 map_off 出现 busy 2/active 1，而 Rule/map_on 为 0。  
  **这不是 factual H20 精度或调度优劣结论**：原事实轨迹由 CP-SAT 产生，rollout 使用 nearest；没有事实/反事实动作配对，也没有新的独立测试。

## 4. 环境与复现

- Python 3.11.15，torch 2.5.1（CPU inference；CUDA disabled），PyG 2.6.1，numpy 1.26.4，OR-Tools 9.14.6206，pandas 2.3.3。
- mbrl source commit `3f93cccfc8d635f74e335a2f07aab6e9a48fc021`；仅用于满足 PR64/PETS 模块导入，PR68 评测未训练 PETS。
- **环境偏差**：hydra-core 1.3.2 / omegaconf 2.3.0，而非 source lock 的 1.0.3；原因是 1.0.3 在 Python 3.11.15 的 dataclass 初始化失败。PR68 评测不调用 hydra runtime，也不训练 PETS；numpy/ortools/pandas 与 PR64 source freeze 一致。
- 15/15 CPU 单元测试通过（PR66+67+68）。
- 原始数据 SHA：PR59 validation `c7a0a008…509712`；PR64 train `4c115777…aee183`；PR64 dev `31a36e09…d0b86`；PR64 quality `f2e69dc6…af877`。
- 无新 GPU 训练、无 BLIND/TEST 读取或生成；未修改 PR61/PR62 原始数据或 checkpoint。

## 5. 产物

`examples/evidence/pr68-original-pr61-cpu-evaluation-20261113/`

- `verified_checkpoint.json`：archive、12 个成员、6 个 best 的准入/来源/epoch/指标。
- `evaluation.json`：6 个 teacher-forced 复现 + 36 个合法 H5/H20 递推。
- `environment.json` / `pip-freeze.txt`：CPU 环境与依赖。
- `tests.log`：15/15 单元测试。
- `source_head.txt`、`SHA256SUMS`。

**裁决：权重准入与旧验证复现已通过；真实预训练权重的合法 H5/H20 递推已执行。仍不能据此宣布新训练必要、长期预测收益或在线调度优势；BLIND 保持封存。**
