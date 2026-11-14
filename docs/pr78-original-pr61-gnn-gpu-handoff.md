# PR78 GPU 训练任务：关系图世界模型原训练主干 + 条件服务事件头

> **权重保密最高约束：原 PR61 模型和 PR78 新训练事件头均不公开。** GPU 在受控机器训练，所有 `*.pt`、优化器/RNG、原归档仅保存于私有持久存储；**绝不提交到公开仓库、PR、Issue、Release、GitHub Actions Artifact/缓存**。只能向公开任务同步非敏感指标和摘要，参见 [PR78 私有权重规程](pr78-private-checkpoint-policy.md)。已删除原可通过公开 Actions 传输权重的训练 workflow。

**最新执行决策（2026-11-14）**：用户选择 **GPU 为正式训练执行环境**，以缩短试验周转；CPU 保留代码回归、数据准入和可复现备选方案。GPU 不代表模型在数学上必须 GPU，也不改变已冻结的研究条件。

本文件为 **[研究 Project 入口 #79](https://github.com/lieuyimo/wm/issues/79) / [GPU 训练 Issue #81](https://github.com/lieuyimo/wm/issues/81)** 的执行规程，GPU Agent 应直接按 Issue 更新进度。**本文件是任务说明，不表示已完成 GPU 训练。**

## 前置门禁与职责

- **任务 #80：权重来源与 SHA 校验**。原 PR61 归档 `pr60-gpu-checkpoints-20261112.tar.gz`，期望大小 **6,661,520 bytes**，SHA256 `919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14`。验证原 [PR68](https://github.com/lieuyimo/wm/pull/68) 的六组 model best.pt manifest；用户表示原始权重已在 PR78 关联位置上传，GPU 执行方优先核对附件或受控服务器位置，不得凭文件名替代哈希。
- **任务 #81：原 GNN 真权重 GPU 训练**。冻结原 52,953 个参数 GRU/RGCN/SAGEConv 主干；仅对现有 48d public service query + 6d original task logits 的标准 BCE 事件预测头（新增 2,689 参数）训练。原模型不会被覆盖，旧权重不复制进新 head-only checkpoints；**新 head-only checkpoints 也属于私有资产，不上传公开 GitHub**。
- **任务 #82：条件事件 H1 比较**。全部 6 arms/seeds 在同一独立 DEV 12 场景、1,409 风险/594 下一步完成事件上评估；不以开发验证当 BLIND 独立泛化证据。H5/H20 和决策收益属于后续 #83/#84，未经评估不提前宣称。

## 不可改动数据与训练配置

- 原 PR64 TRAIN：24 场景、2,593 active service rows、1,106 真实下一步事件；SHA256 `4c1157777117f82f7deb44f98e1d8255e95611dfcb41c6dfff9e14dd23aee183`。
- 原 PR64 DEV：12 场景、1,409 risks、594 events；SHA256 `31a36e0940db7afe015fe35689cb629102dea45e8d61123ed89250abc01d0b86`。TRAIN/DEV 来自 [原始 GitHub Actions #43264954353](https://github.com/lieuyimo/wm/actions/runs/43264954353)，quality 与源 SHA256SUMS 一并校验。
- 使用原 PR73 已有 `jitter_ticks=2` 实际仿真生成机制，保持原 CP-SAT 行动、合法实验室业务规则和任务阶段，**不改随机物理机制**。
- `map_off` / `map_on` × `1810/1811/1812`，共六组；**20 epochs、batch16、Adam lr=1e−3**，按 DEV 12 独立场景宏平均 Brier 选择最佳 epoch；原 PR61 主干必须逐张量冻结不变。
- 已有 `pr78_frozen_cache.py` 会一次性提取原冻结 GNN 6d task 表征 + 48d 公开服务特征。GPU 优先运行，仍沿用完整端到端评分与缓存评分一致性校验。禁止随机权重回退、打开 PR59 BLIND/TEST 或混入暂停的 PR77 传送带实验。

## GPU Agent 执行步骤

在拥有受控 PR61 原 A100 归档的 CUDA 环境，checkout **PR78 当前最新 HEAD**，安装并保存 `requirements-paper-reuse.lock` 环境与所需 PyG，检查 `torch.cuda.is_available()` 和 `nvidia-smi`。先在 #80 留下经过校验的原 SHA 摘要和通过状态，**不写内部文件路径、权重附件**，再执行：

```bash
python -m experiments.world_model_integration.gnn_pr78.pr78_pretrained_train \
  --original-pr61-archive "$PR61_ARCHIVE" \
  --original-pr61-manifest examples/evidence/pr60-gpu-training-20261112/checkpoint_manifest.csv \
  --train "$PR64_TRAIN" \
  --dev "$PR64_DEV" \
  --quality "$PR64_QUALITY" \
  --device cuda \
  --output-dir "$NEW_UNTOUCHED_PR78_GPU_RESULT_DIR"
```

**可选** `--original-pr59-validation "$VERIFIED_ORIGINAL_PR59_VAL"` 仅供原 PR68 再校验历史同源数据；**不得用 PR64 DEV 代替 PR59 VAL**。启动前确认输出目录尚不存在，任何缺 CUDA/原权重 SHA mismatch/数据风险集错误立即停止。

## 产物和审查

每个 map/seed 必须在**受控私有存储**保存新的 **head-only** `best.pt`、优化器和 CPU/CUDA RNG 状态、best epoch、原 GNN 权重的内容 SHA 与已验证原归档哈希。不得作为公开 Artifact 或公开文件提交。私有环境中完整保存 `report.json` 每场景/相位 Brier、`runtime_environment.json`（GPU/驱动/Python/torch/CUDA）、`pip-freeze.txt`、源文件 SHA256、Git HEAD、日志、输出 SHA256SUMS；旧 PR61 权重原归档不覆盖；**原权重、新事件头和可恢复检查点一律禁止公开上传**。

同一批原 DEV active service 风险下的先验基线：

| 方法 | Brier |
|---|---:|
| 原 Rule | 0.288857 |
| 已训练 PR74 Deep Sets | 0.188937 |
| TRAIN-only public conditional hazard | **0.176655** |
| PR78 六组真 GNN + 新事件头 | **等待 GPU 实际结果，禁止预填** |

以 12 独立场景为统计单位报告配对效应与 CI、pickup/delivery 分层、三种业务 regime 分层。只将**脱敏统计指标与校验摘要**提交 [GPU 训练 Issue #81](https://github.com/lieuyimo/wm/issues/81) 和 [PR78 Draft](https://github.com/lieuyimo/wm/pull/78)；**不包含任何模型文件、优化器状态、公开 Artifact 或内部路径**；研究方依 [#82](https://github.com/lieuyimo/wm/issues/82) 判定是否进入多粒子 H5/H20 或是否需要针对性解冻。

**GPU 执行方负责实际提交结果**，本文件和 Issue 是任务委托与验收合同，不代表已远程启动 A100 训练。
