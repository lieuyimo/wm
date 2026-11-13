# PR68：PR61 A100 真检查点恢复、SHA 准入及 CPU 递推评测执行记录

## 一、当前真实状态（不可用记录替代权重）

PR67 的合法 H5/H20 闭环是 **PR61 原网络架构 + 随机初始化权重** 的 CPU 合法性验证，不能据此评价训练模型能力。本 PR 延续 PR67，只完成 **真实检查点来源检查、安全加载器、可执行 CPU 评测和无权重明确阻断**。

已核对：
- [PR61](https://github.com/lieuyimo/wm/pull/61) 记载真实 A100 训练并保存 `pr60-gpu-checkpoints-20261112.tar.gz`，预期体积 **6,661,520 B**；完整 SHA256 **`919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14`**。
- 原记录的**受控本地**归档位置：
  `/data/wm-a100-pr60-gpu-20261112/pr60-gpu-checkpoints-20261112.tar.gz`。
- [PR62](https://github.com/lieuyimo/wm/pull/62) 曾在原 A100 环境按 SHA 验证并加载 **6/6 best.pt**，执行过 PR59 冻结 12 场景单步设备事件条件审查。证明该归档至少在 PR62 执行时确实可用；**不意味着当前 GitHub Actions 或本会话可以读到远程本地磁盘**。
- PR61 仓库已提交 `checkpoint_manifest.csv` 的 6×(best/last) 各 SHA、字节数，以及原 GPU 评测记录。Git tree 有其他 PETS/V3/V4 或 PR31 的训练权重，但**均不属于 PR61 的 52,953 参数 PublicMapTemporalRelationalWorld**，不得替换。
- GitHub 释放版本中 PR31 的 `full-training-and-environment.tar.gz` 属于其他世界模型，不能冒充 PR61。
- 已核实真实原 PR59 验证数据仍可从 Actions **#43234653164** 的 `pr59-real-public-world-43234653164-validation` 下载：SHA256 `c7a0a0081e0c901faea23b37816eb54b560efb9c8b66118387c3fbb0a9509712`，12 独立场景 1,517 真实 t→t+1 公开转移；此数据是**旧模型选择所用验证集**，不得当最终未知测试集。
- 原 PR64 的 TRAIN/DEV/quality 仍在 Actions **#43264954353**；DEV 也已用于方法开发，不是最终未知测试。

## 二、已经提交的可执行实现

- `experiments/world_model_integration/recovery_pr68/pr68_checkpoint.py`：强制输入原始 tar.gz + 固定 Git manifest，验证**完整 tar SHA/体积、tar 内文件无路径穿越/符号链接、12 个 best/last SHA 与体积、完整 arm/seed roster、PR61 training head、VALIDATION 来源 SHA、模型 schema/hidden/结构/参数数、Adam/CUDA&CPU RNG**。所有条件通过后才用 PyTorch `torch.load(weights_only=True)` CPU 安全加载六组真正 PR61 best 权重，严格 `state_dict`。
- `recovery_pr68/pr68_cpu_eval.py`：在 **真实权重输入不可缺少** 的前提下，复用原 PR59 `evaluate` 对旧 12 场景的 6 组模型验证单步 node MSE/device Brier/new task MAE 与已报告的 checkpoint metrics；随后加载 PR64 TRAIN/DEV 真实首帧的业务合法世界，在同一个 CPU 推理实验中分别以 Rule/map_off/map_on、3 个原 PR61 seed、H5/H20 比较法律约束下的任务阶段、资源、工作站与神经服务事件数。每一神经预测步都调用原训练好的 PyG forward，利用 PR67 服务状态映射和 PR66 设备外生边界，不使用随机权重 fallback。
- `test_pr68.py`：原始 manifest 配置，缺失归档、错误 archive SHA、错误验证源、损坏或重复模型条目等负例；阻断时无模型推理、无模型参数更新。
- `.github/workflows/pr68-original-pr61-checkpoint-admission-cpu.yml`：GitHub Actions CPU-only 下载 SHA 锁定的原 PR59、PR64 数据并运行 PR66/67/68 测试；**GitHub CI 不拥有受控本地 GPU 主机归档**，因此 CI 即便 PASS 也只能记录来源/入口 PASS、已训练模型评测 **BLOCKED**，不可假报预训练指标。

特别说明：PR68 闭环统计是**同一候选策略接口下的受约束预测轨迹描述**，不直接是 factual H20 ground-truth 误差：原 PR64 事实轨迹由 CP-SAT 控制，PR68 模型递推内选用当前预测状态下 nearest 行为；反事实动作及外生事件未与事实轨迹严格配对。因此当前 CPU 评测不能判断“真实长期预测更准”或“调度更优”，只作为加载实际权重后验证神经状态转移是否有合理动态以及何处出现偏差的第一步。

## 三、在原 A100 主机或可读该目录的 CPU 环境执行

无需占用 GPU，命令以 `CUDA_VISIBLE_DEVICES=""` 开始：

```bash
set -euo pipefail
cd /path/to/wm
git fetch origin
# 保持 PR68 分支/当前最新 HEAD，切勿覆盖别人的提交
git checkout research/pr68-original-pr61-checkpoint-cpu-evaluation-gate-20261113
export CUDA_VISIBLE_DEVICES=""

ARCHIVE=/data/wm-a100-pr60-gpu-20261112/pr60-gpu-checkpoints-20261112.tar.gz
test -f "$ARCHIVE" || { echo "BLOCKED: original PR61 local archive missing"; exit 2; }
test "$(stat -c %s "$ARCHIVE")" = 6661520
echo "919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14  $ARCHIVE" | sha256sum -c -

mkdir -p /data/wm-pr68-cpu-input/pr59-validation
mkdir -p /data/wm-pr68-cpu-input/pr64-train
mkdir -p /data/wm-pr68-cpu-input/pr64-dev
mkdir -p /data/wm-pr68-cpu-input/pr64-quality

gh run download 43234653164 -R lieuyimo/wm \
  -n pr59-real-public-world-43234653164-validation \
  -D /data/wm-pr68-cpu-input/pr59-validation
gh run download 43264954353 -R lieuyimo/wm \
  -n pr64-real-random-outage-43264954353-train \
  -D /data/wm-pr68-cpu-input/pr64-train
gh run download 43264954353 -R lieuyimo/wm \
  -n pr64-real-random-outage-43264954353-validation \
  -D /data/wm-pr68-cpu-input/pr64-dev
gh run download 43264954353 -R lieuyimo/wm \
  -n pr64-shifted-outage-true-risk-set-audit-43264954353 \
  -D /data/wm-pr68-cpu-input/pr64-quality

for f in /data/wm-pr68-cpu-input/*; do (cd "$f" && sha256sum -c SHA256SUMS); done

python -m experiments.world_model_integration.recovery_pr68.pr68_checkpoint \
  --archive "$ARCHIVE" \
  --original-validation /data/wm-pr68-cpu-input/pr59-validation/data.json \
  --out /data/wm-pr68-cpu-check/verified_checkpoint.json

python -m experiments.world_model_integration.recovery_pr68.pr68_cpu_eval \
  --archive "$ARCHIVE" \
  --original-validation /data/wm-pr68-cpu-input/pr59-validation/data.json \
  --pr64-train /data/wm-pr68-cpu-input/pr64-train/data.json \
  --pr64-dev /data/wm-pr68-cpu-input/pr64-dev/data.json \
  --pr64-quality /data/wm-pr68-cpu-input/pr64-quality/quality.json \
  --output /data/wm-pr68-cpu-check/evaluation.json

git rev-parse HEAD > /data/wm-pr68-cpu-check/source_head.txt
python -m pip freeze > /data/wm-pr68-cpu-check/pip-freeze.txt
(cd /data/wm-pr68-cpu-check && sha256sum *.json *.txt > SHA256SUMS)
```

环境建议优先使用原 A100 主机已有的 **Python 3.11.15、PyTorch 2.5.1+cu124、PyG 2.6.1** 解释器而仅禁用 CUDA；脚本本身始终 `map_location="cpu"`、调用 `.cpu()`。也可重新建立与 PR68 CPU CI 等价的版本冻结环境。若 `weights_only=True` 拒绝历史 checkpoint 的序列化对象，记录准确异常并暂停，不自动回退 `weights_only=False`。

## 四、条件性下一步与 GPU Agent 审查

此阶段唯一外部阻断是**受控本地检查点尚未提供给当前执行环境**。原 PR62 已执行一次权重真实推理，故应优先从上述原 A100 归档恢复，而不是重新训练。

恢复后先给出以下对照：原 6 best 的准入 SHA、真实 12 场景旧验证误差与 PR61 记录差异、Rule/map_off/map_on 的合法 H5/H20 演化、各 active service 查询次数及完成概率/事件计数、设备抽样变化、MAPF/资源/任务约束、逐场景分层描述。若要真正检验泛化，必须另预注册同策略且可执行的事实/反事实独立数据：原 PR59 BLIND 保持封存直至 protocol review。

**本 PR 的 CPU CI PASS 不等于 PR61 权重已经恢复。** 拿到上述 `evaluation.json` 之前，不启动新 GPU 训练或宣传决策收益。

## 五、交给可以访问原 A100 磁盘的 Agent 的 Prompt

> 使用 PR68 最新分支研究代码，先确认原 PR61 归档 `/data/wm-a100-pr60-gpu-20261112/pr60-gpu-checkpoints-20261112.tar.gz` 存在且完整 SHA256 为 `919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14`。严禁重训替代和使用 PR31/PETS/V4 检查点。CPU 禁用 CUDA，按 docs/pr68-original-pr61-recovery-cpu-review.md 的命令下载、SHA 复核 PR59 VAL 与 PR64 TRAIN/DEV/quality，执行 `pr68_checkpoint` 和 `pr68_cpu_eval`。先确保 6/6 原始 best 权重安全加载且旧 PR59 validation teacher-forced 指标复现，然后完成真实权重 Rule/map_off/map_on × 3 seeds × H5/H20 的合法业务状态递推，完整记录输入/环境/RNG/模型与评测结果 SHA，提交同一 PR 的新证据与评论。无 GPU 训练、无 BLIND、无原 PR 合并。若权重不可获取或 SHA 不符，直接标记 BLOCKED 并报告精确缺失项，不用随机网络代替、不声称 H20 已训练模型效果。
