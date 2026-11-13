# PR78 GPU 移交：仅使用原 PR61 真权重训练条件服务事件头

## 执行边界

**项目**：<https://github.com/lieuyimo/wm>  
**目标分支**：`research/pr78-original-gnn-public-service-event-alignment-20261114`  
**上游起点**：PR76 SHA `e79205e3441eee43dba3142189042d47a47d9a76`。**不要合并 PR77 传送带扩展**；该功能已暂停。  
**现有源码**：`experiments/world_model_integration/gnn_pr78/{pr78_model,pr78_pretrained_train,pr78_data_gate}.py`。  
**研究目标**：不修改现有 52,953 参数 PR61 GRU/RGCN/SAGEConv 主干，只在 SHA admission 后，用原有 48d 当前公开服务查询监督一个 2,689 参数的二分类下一步服务事件头。等价 map_on/map_off × 原 3 seeds，使用真实 PR73 J2 的 TRAIN/DEV（24/12 场景）。

原始数据不允许替换：PR64 TRAIN 文件 SHA256 `4c1157777117f82f7deb44f98e1d8255e95611dfcb41c6dfff9e14dd23aee183`，DEV `31a36e0940db7afe015fe35689cb629102dea45e8d61123ed89250abc01d0b86`，quality 原 artifact 从 [Actions #43264954353](https://github.com/lieuyimo/wm/actions/runs/43264954353) 下载。全部样本来自原 `BusinessSimulator`，已有 `jitter_ticks=2`，无需改数据合成规则。事前 CPU 门禁需要核对 **TRAIN 2,593 风险/1,106 完成事件；DEV 1,409/594**，且三类业务 regime 完整，标签严格 t→t+1。

原始 PR61 A100 6 个 best.pt archive：必须使用 PR68 的实际原始归档以及 manifest，**expected archive SHA256 `919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14`**。任何缺少归档、SHA mismatch、缺 CUDA 的情况都应直接停止，不允许以随机权重代替或用 PR74 Deep Sets 权重冒充 PR61 图网络。

## 执行命令

在具备 CUDA 的环境内安装并保留原项目冻结的 `requirements-paper-reuse.lock` 依赖与 `torch-geometric`；先运行本分支的 PR78 CPU 数据标签与梯度测试，然后执行：

```bash
python -m experiments.world_model_integration.gnn_pr78.pr78_pretrained_train \
  --original-pr61-archive /path/to/original-pr61-six-best-archive.tar.gz \
  --original-pr61-manifest /path/to/checkpoint_manifest.csv \
  --original-pr59-validation /path/to/original-pr59-validation.json \
  --train /path/to/original-pr64-train/data.json \
  --dev /path/to/original-pr64-validation/data.json \
  --quality /path/to/original-pr64-quality/quality.json \
  --device cuda \
  --output-dir /path/to/NEW_UNTOUCHED_PR78_RESULTS
```

`--original-pr59-validation` 可选，但若存在原始 PR59 验证文件，应传入，由 PR68 再校验其原 SHA；不能用 PR64 DEV 伪装成 PR59 VAL。运行前务必 `sha256sum` 校验文件并记录 HEAD。正式训练入口不接受无 SHA 权重。

**冻结的实验条件**：map_off/map_on × seeds `1810/1811/1812`，20 epochs，batch16，Adam lr1e−3；图主干参数冻结并验证前后逐张量完全一致。仅训练公开 48d + 原 GNN task next-state 6d 的标准服务事件 BCE head。按 DEV 12 场景 macro Brier 选最佳 epoch，检查 pickup/delivery 各自 Brier、事件数、置信区间。不能看 TEST/BLIND 再改种子、epoch、标签或数据机制。

**必须保存**：6 个新 best.pt（包含原主干 state、已训练新 head、优化器状态、Torch/CUDA 随机状态、best epoch、history、源 SHA、`online_verified=False`）；`report.json` 全量每场景/分相位结果；`runtime_environment.json`，`pip-freeze.txt`，Git HEAD、GPU型号、原 manifest+archive 哈希；所有输出 SHA256SUMS。不要覆盖原 PR61 A100 ckpt 或合并任何旧 PR。

## 结果判据与后续评估

在**完全相同**原 PR64 DEV J2 风险集上比较：
- Rule Brier **0.28886**，由原 PR73 确认。
- PR74 Deep Sets Brier **0.18894**，在相同条件事件任务上真实 CPU 训练。
- PR73 TRAIN-only 公开服务风险频率 Brier **0.17665**，最重要的简易基线。
- PR78 六种原训练图主干+新 head 的同一 scene-level 配对指标。报告 map_on 和 map_off 各三个种子，不能只展示最好一个。

**决策规则**：先检验公开信息修复和目标对齐之后，原关系图表示是否超越传统条件概率和 Deep Sets，或在有地图关系/拥堵条件的业务场景具有分层收益；若没有，先诊断输入、校准和是否需要有依据的有条件解冻 GNN，而非盲目加参数。首次 head 训练不等于证实 H5/H20 准确、粒子分布校准或 CP-SAT 调度收益。

正式模型与合法状态推演的接线已有 `PublicRelationalServiceRule.from_pretrained_output`：只有新 head SHA 和原 PR61 archive 双重校验才能使用原 PR67 状态合法化。新权重真正到达后才能运行**相同公开策略/外生条件的粒子预测分布**评测，不能以单粒子事实动作不合法的幸存 H20 样本评判准确性。

**禁止** GPU 盲目训练 PR61 整网、自动修改场景/种子、修改原校验 hash、打开 PR59 BLIND TEST、改变机器人调度策略、重新研究传送带、宣称本轮已有线上决策改善。

执行者应把完整事实结果、现有训练脚本问题和必要修复提交到 PR78 的评论，并保持 PR78 Draft，等研究端复核后再决定下一步。
