# PR70：实体生命周期修正与两种多步协议

日期：2026-11-13  
PR70 head：`13d1017`  
范围：**CPU inference only；无训练/微调/参数更新；无 BLIND/TEST**。使用 PR68 已验证 PR61 权重、PR59 validation、PR64 TRAIN/DEV。

## 1. 目标与实现

回应 PR69 review 的 A/B/C/D 门禁：

- **A. 公开实体到达**：只在事实当前公开观察中首次出现的 `stage1` 任务被注入条件递推；不复制事实 `stage2`，successor 仍由原 lawful workflow transition 生成。
- **B. 服务完成时序**：在事实 current observation + 事实 CP-SAT action 的 active-session risk set 上，比较 PR61 task head 的 next-status 概率、Rule 和真实 next status；不调阈值。
- **C. 两种协议**：
  - conditional diagnostic：事实动作/设备条件、stage1 arrival adapter；
  - fully legal predicted-state rollout：在各自预测状态上用 nearest legal batch 重新选动作，共同外生随机种子。
- **D. 统计单位**：12 个独立场景；seed 不作为独立场景；scene-level paired bootstrap。

## 2. A：stage1 到达修正

| 协议 | Rule H20 | map_off H20 | map_on H20 | H5 三臂 |
|---|---:|---:|---:|---:|
| 无到达修正（PR69） | 30/36 | 9/36 | 9/36 | 36/36 |
| + 公开 stage1 arrival | **36/36** | 14/36 | 13/36 | 36/36 |

- 共注入 **387** 个 stage1 arrival 记录（跨 run），只来自事实 current public observation；没有注入事实 stage2。
- Rule-only H20 回归从 30/36 提升到 **36/36**，确认新任务 arrival 缺口是原失败的主要原因之一。
- 神经臂剩余 H20 失败：map_off 20 个 stage2 + 2 illegal assignment；map_on 21 个 stage2 + 2 illegal assignment。  
  这些是**前序服务完成时序与 successor 生成**的真实分歧，不应再归因于 arrival adapter。
- H5 仍无事实 sample 完成（首次完成 tick 9/11），所以 H5 指标仍不能区分任务头；见 PR69 `h5_equality_cause.json`。

## 3. B：服务完成事件概率

在同一事实 active-session risk set 上：

| 项目 | 值 |
|---|---:|
| eligible active-session rows | 751 |
| actual service events | 508 |
| Rule Brier / log-loss | **0 / ~1e-6**（该 risk set 上确定性 rule 与事实事件一致） |
| map_off model Brier | 0.0244 / 0.0325 / 0.0279（seeds 1810/1811/1812） |
| map_on model Brier | 0.0215 / 0.0326 / 0.0309 |
| map_off model − Rule Brier | **+0.0283**, 95% CI [+0.0082, +0.0563] |
| map_on model − Rule Brier | **+0.0284**, 95% CI [+0.0096, +0.0539] |

- 神经 task head 的概率在 active-session 事件上明显差于该 Rule 基线；map_on 没有相对于 map_off 的稳定优势。
- 这是**事件概率/校准审计**，不是把确定性命中率直接等同于完整 H20 预测。
- **完成时序也明显偏早**：12 个场景 × 6 个模型的首个 model-sampled completion tick 全部是 **3**，而事实首个 completed observation tick 是 **9 或 11**。因此模型既在概率上更差，也在完成时序上过早；这与 H20 conditional 里 predecessor `in_transit` 而事实已 completed、successor stage2 缺失的现象一致。
- 这些“sampled complete”是模型内部事件，不代表规则状态真的完成了；它们不能当作 H20 事实精度。

## 4. C：fully legal predicted-state rollout

相同事实初始状态；各 arm 在自己的预测状态上使用 nearest legal action；共同外生 seed；216/216 个 rollout 完整且 `all_states_legal=True`。

| horizon | arm | completed mean | waiting mean | busy mean | active mean | full |
|---|---|---:|---:|---:|---:|---:|
| H5 | rule/map_off/map_on | 0 | 6.83 | 1.17–1.22 | 0.58–0.61 | 36/36 |
| H20 | rule | 3.583 | 6.444 | 1.306 | 0.500 | 36/36 |
| H20 | map_off | 3.500 | 6.278 | 1.444 | 0.556 | 36/36 |
| H20 | map_on | 3.500 | 6.167 | 1.500 | 0.583 | 36/36 |

Scene-level paired H20 completed difference (arm − rule)：

- map_off: **−0.0833**, 95% CI [−0.3333, +0.1667]
- map_on: **−0.0833**, 95% CI [−0.3333, +0.1667]

CI 包含 0，且该 rollout 不是 factual H20 accuracy；不能据此宣称模型更好或更差。

## 5. 判读

- **A GO**：stage1 public arrival 修正后 Rule H20 36/36；剩余神经失败是 stage2/service timing。
- **B NO-GO for learned service-event value**：模型事件概率显著差于该 Rule 基线；没有 map_on 优势。
- **C PASS legal rollout**：所有 arm 都能完成 5/20 合法 rollout；没有完成量优势；没有 factual 精度声明。
- **D**：12 个场景为独立单位；216 个 run 只描述训练 seed/场景重复，不当作 216 个独立样本。
- **总 NO-GO**：不重训、不扩参数、不晋升 checkpoint、不打开 BLIND、不宣称 H20 预测/在线调度收益。
- **下一问题**：如何在不复制事实 successor 的前提下，由公开 lifecycle/service 事件生成 stage2；以及为什么 task-head 在 active service 上比确定性 rule 差。

## 6. 产物

`examples/evidence/pr70-lifecycle-event-timing-20261113/`

- `conditional_injected.json` / `rule_only_injected.json`：stage1 arrival 协议与 Rule 回归。
- `service_timing.json`：active-session 事件概率审计。
- `legal_rollout.json`：fully legal predicted-state H5/H20。
- `environment.json`、`command.txt`、`source_head.txt`、`SHA256SUMS`。
- `tests.log`：2/2 arrival adapter unit tests。
- 无权重、无原始数据、无 BLIND。

**PR70 保持 Draft；不合并。**
