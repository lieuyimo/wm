# PR78：原 PR61 预训练关系图世界模型的 CPU 冻结主干训练

## 研究状态

**训练尚未实际执行**。我们已经通过 PR78 的完整原 PR64 J2 TRAIN/DEV 标签数据、原 GNN map_on/off 结构及梯度单元测试，且冻结主干的**公共节点预测 6d + 当前公开服务查询 48d**可以只在 CPU 前向提取一次、在后续 20 个 epoch 中仅使用普通小型 MLP 训练。代码已将 `--device` 默认值改为 `cpu`。

**硬阻塞不是 GPU，而是原 PR61 训练权重当前不在 GitHub Actions/本会话工作目录。**

原 [PR60 GPU 记录](pr60-gpu-training-results.md) 与 PR62、PR68 审计确认，受控 A100 服务器上的归档原路径为

```text
/data/wm-a100-pr60-gpu-20261112/pr60-gpu-checkpoints-20261112.tar.gz
```

原归档应为 **6,661,520 bytes**，SHA256：

```text
919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14
```

包含 map_on/map_off × seeds 1810、1811、1812 共六个训练结果的 best/last checkpoints。仓库中已有 manifest 位于 `examples/evidence/pr60-gpu-training-20261112/checkpoint_manifest.csv`。**任何无法从原归档/manifest 校验通过的模型都不得被视为原 PR61 真权重，不能用随机模型冒充。**

### 优化后的 CPU 路线

`experiments/world_model_integration/gnn_pr78/pr78_frozen_cache.py`：

1. 用 `pr78_pretrained_train.py` 的原 PR68 验证器严格加载六个原始 map_on/off 权重；冻结 52,953 个 GRU/RGCN/SAGEConv 参数。
2. 通过原 PR61 当前公开图、历史与地图，**每个 arm/seed 只进行一次** public next-task-node 6d 提取，与 PR73 相同、已测的服务会话 48d 当前公开查询拼接。严格保证样本身份、取样/交付阶段、t→t+1 标签和独立场景切分。
3. 每个 arm/seed 的 2,593 条 TRAIN 活动服务风险以行级 batch16，20 epoch，Adam(1e−3)，只更新现有标准条件 BCE 服务事件预测 MLP；12 DEV 场景 1,409 个活动服务风险只用于 best epoch 选择。
4. 用真实**原始端到端图网络前向**在选定最佳 head 上再评分一次，要求与缓存评分的 Brier 数值误差不超过 1e−5；每个原 PR61 主干张量训练前后必须严格相等。
5. 保存**新训练的 head 权重**、原始冻结主干**规范张量 SHA256**、优化器、CPU/CUDA RNG、原文件 SHA、每场景/相位指标、环境 `pip-freeze.txt` 和 `runtime_environment.json`。**不重复打包原 PR61 旧图权重**，以免破坏其受控归档。读取组合模型时，必须由 `PublicRelationalServiceRule.from_pretrained_output` 同时校验**原归档 SHA 与新 head SHA**，再验证规范张量哈希一致。

此方法并未修改原 PR61 网络结构、原 PR59/60 权重或数据；不使用任何新世界模型算法。冻结的 GNN 在 eval 模式输出不依赖当前 head 参数，因此缓存等价是确定性的，单元测试用真实业务图的两种服务阶段验证缓存与直接推理等价。

## 运行方式：CPU，无须 GPU

运行前在**具备原归档访问权限的机器**上确认：

```bash
sha256sum /data/wm-a100-pr60-gpu-20261112/pr60-gpu-checkpoints-20261112.tar.gz
```

然后在仓库 PR78 分支上准备原始 PR64 的 TRAIN、DEV、quality 文件（数据原来源 [Actions #43264954353](https://github.com/lieuyimo/wm/actions/runs/43264954353)，分别校验原 SHA256SUMS），执行：

```bash
python -m experiments.world_model_integration.gnn_pr78.pr78_pretrained_train \
  --original-pr61-archive /data/wm-a100-pr60-gpu-20261112/pr60-gpu-checkpoints-20261112.tar.gz \
  --original-pr61-manifest examples/evidence/pr60-gpu-training-20261112/checkpoint_manifest.csv \
  --train /path/to/verified-pr64-train/data.json \
  --dev /path/to/verified-pr64-developer-validation/data.json \
  --quality /path/to/verified-pr64-quality/quality.json \
  --device cpu \
  --output-dir /path/to/NEW_UNTOUCHED_PR78_CPU_RESULTS
```

原 PR59 validation 输入为可选，仅当提供原始 SHA 验证版本时再传 `--original-pr59-validation`，不能把 PR64 DEV 伪称为原 PR59 VAL。

**完整输出**：`report.json`、6 个头部 best.pt、`runtime_environment.json`、`pip-freeze.txt`；输出路径不得存在，先核对 git HEAD，最后自行 SHA256SUMS 归档。如果只需要公开或同步实验指标，请仅传输 result/report/日志与 **新事件头**，不上传旧 PR61 archive/主干权重到公开 Git 仓库。

## 必须对照的实际基线

同一个独立 DEV 1,409 风险/594 真实事件：

| 条件服务事件预测 | DEV Brier |
|---|---:|
| Original Rule | 0.28886 |
| PR74 已训练 Deep Sets | 0.18894 |
| PR73 TRAIN-only 公开条件 hazard | **0.17665** |
| PR78 真 PR61 预训练主干 + 新 head | **未训练，不能填数** |

多种子结果按原 12 个场景做宏均值、分取样/交付事件和场景 bootstrap；不要只展示最佳 seed 或将开发集选择结果当独立 TEST 泛化证据。根据结果决定是否进入有依据的原 GNN 参数解冻；H5/H20 粒子分布、调度收益仍是后续独立门禁。

## 当前阻塞裁决

- **CPU 训练可行性：PASS**，包括真实合法 54d 风险查询构造、真实标签、缓存与原 GNN 直接前向等价、CPU 标准 Adam 更新、不修改主干。
- **真权重训练结果：BLOCKED（缺归档访问）**，并非 GPU 性能不够。
- **0 GPU / 0 BLIND / 0 PR61 权重覆盖 / 0 新训练结果虚构**。
