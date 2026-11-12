# PR62：PR61 权重的设备事件条件校准实测

日期：2026-11-13  
PR62 head：`6882dd360068bbca16e24e89f61bd682e8238caa`  
范围：真实 PR61 `best.pt` × PR59 冻结 12 场景 validation；**无新训练、无参数更新、无 BLIND/TEST 读取或生成、无 H20/在线推理**。

## 1. 结论

**NO-GO：设备故障发生与恢复的条件校准未通过。**

- 总体 Brier 约 `0.0042`，但 96.8% 的站点 tick 是 `healthy_hold`；该数值被健康状态基数主导，不能说明故障/恢复预测能力。
- `fault_onset` 的模型 Brier 为 `0.9904 / 0.9906`（map_off / map_on 均值），预测概率仍约 `0.995`；模型基本没有提前识别故障。
- `recovery` 的模型 Brier 为 `0.8350 / 0.8683`，预测概率仅约 `0.07–0.09`；虽然数值上略优于“保持上一状态”的 `1.0`，但仍远非可用校准。
- 仅有 **16 个 onset、16 个 recovery** 站点转移标签，分布在 12 个独立验证场景；不能把 1,517 个相关 tick 或 45,510 条模型预测当作独立泛化样本。
- map_on 的 one-step 节点 MSE 改善仍成立，但**不能**外推为设备事件、H5/H20 递归或在线调度收益。

## 2. 可信权重与数据校验

| 项目 | 值 |
|---|---|
| PR61 权重归档 | `pr60-gpu-checkpoints-20261112.tar.gz` |
| 归档 SHA256 | `919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14` |
| PR59 validation `data.json` SHA256 | `c7a0a0081e0c901faea23b37816eb54b560efb9c8b66118387c3fbb0a9509712` |
| PR62 reader head | `6882dd3` |
| 权重 SHA 校验 | 6/6 `best.pt` 与 PR61 manifest 一致 |
| 模型来源校验 | `arm/seed/git_head_source/validation SHA` 全部通过 |
| 新训练/参数更新 | 0 / 0 |
| BLIND_TEST 读取/生成 | 0 / 0 |
| 推理设备 | 单张空闲 A100-SXM4-80GB；结束后已释放 |

原始 archive 未修改；6 个 `best.pt` 解压到独立受控目录后逐一校验，再加载为 `map_enabled` / `map_off` 原始模型。

## 3. 事件基数

每个模型在 12 个独立验证场景上产生 **7,585 个站点 tick 标签**：

| 组别 | tick 数 | 说明 |
|---|---:|---|
| healthy_hold | 7,344 | 当前 up、下一时刻 up；占 96.8% |
| fault_onset | 16 | 当前 up、下一时刻 down |
| continuing_down | 209 | 当前 down、下一时刻 down |
| recovery | 16 | 当前 down、下一时刻 up |

- onset/recovery 各只有 16 个转移标签；每个验证场景 1–2 个。
- 两幅验证地图各 8 个 onset、8 个 recovery；按 regime 为 congestion 4、multi_fault 8、staggered_arrivals 4。
- 这些计数是高度相关的站点 tick，不是 16×1,517 个独立样本。事件基数不足以估计稀有事件校准或置信区间。

## 4. 条件事件结果

按 3 个训练 seed 聚合的模型 Brier 与“保持上一状态”持久性 baseline：

| 组别 | map_off 模型 | map_off 持久性 | map_off 差值 | map_on 模型 | map_on 持久性 | map_on 差值 |
|---|---:|---:|---:|---:|---:|---:|
| overall | 0.00423437 | 0.00421885 | +0.00001552 | 0.00420089 | 0.00421885 | −0.00001797 |
| healthy_hold | 0.00004854 | 0 | +0.00004854 | 0.00005184 | 0 | +0.00005184 |
| fault_onset | **0.99044468** | 1.00000000 | −0.00955532 | **0.99062777** | 1.00000000 | −0.00937223 |
| continuing_down | 0.01221817 | 0 | +0.01221817 | 0.00832705 | 0 | +0.00832705 |
| recovery | **0.83503461** | 1.00000000 | −0.16496539 | **0.86828905** | 1.00000000 | −0.13171095 |

- onset 模型预测均值：map_off `0.99516`、map_on `0.99529`；真实均为 0。它没有学到“即将故障”。
- recovery 模型预测均值：map_off `0.09051`、map_on `0.07090`；真实均为 1。模型仍接近“保持 down”。
- 逐 seed 总体 Brier 差值（模型 − 持久性）：
  - map_off：`+2.85e-5`, `−3.91e-5`, `+5.72e-5`
  - map_on：`+5.37e-5`, `−5.00e-5`, `−5.75e-5`
- 即使总体均值略有正负，事件子群仍是接近 1 的 Brier；不能据此选择模型或宣称校准。

## 5. 地图与 regime 分层

六模型汇总（每地图 8 个 onset、8 个 recovery）：

| 验证地图 | onset Brier | recovery Brier |
|---|---:|---:|
| map 0 | 0.99146 | 0.89111 |
| map 1 | 0.98961 | 0.81221 |

- 地图差异只是描述性；每地图仅 8 个事件，无法据此做稳定地图校准结论。
- regime 同样只有 4/8/4 个事件：congestion、multi_fault、staggered_arrivals 的 recovery Brier 分别约 `0.822 / 0.846 / 0.893`，样本不足。
- 分组使用**真实下一时刻标签**，仅用于评估；不是模型可用特征，也不是在线阈值。

## 6. 判读

1. 低总体 Brier 是健康状态主导，不是稀有事件预测能力。
2. 两个网络都近似持久性预测；在 onset 上预测概率接近 1，在 recovery 上预测概率接近 0。
3. 16 个 onset/16 个 recovery 事件不足以支持校准、置信区间或地图/regime 泛化声明。
4. 因此不满足进入 H5/H20 合法递归、决策感知或在线调度的门禁；也不应据此外推 PR61 的节点 MSE 改善。
5. 不扩大到更多参数或 GNN；优先解决事件数据覆盖与目标设计问题。

## 7. 产物

- 原始 JSON：`examples/evidence/pr62-device-event-conditional-20261113/pr62-device-event-conditional.json`
  - SHA256 `93f00186c30fb132746ddc9f383981ed4c08cc12f7e630cafbbb9be7136330d7`
- 运行日志：同目录 `pr62-device-event-conditional.log`
  - SHA256 `457594e5bb3a65996b4cc37801d7b994be0b7239f827197879ab2f025f7caef4`
- 逐模型/逐相位表：同目录 `phase_metrics.csv`
- 校验清单：同目录 `SHA256SUMS`
- 原始 6.66 MB checkpoint archive 与解压权重仍保存在受控本地实验目录，**未提交公开 Git**。

复核命令（在 GPU 主机，使用已校验 archive）：

```bash
python -m experiments.world_model_integration.audit_pr62.pr62_device_events \
  --validation /data/wm-a100-pr60-gpu-20261112/source/pr59-validation/data.json \
  --trusted-checkpoints /data/wm-a100-pr62-audit-20261112/ckpt-verified \
  --manifest examples/evidence/pr60-gpu-training-20261112/checkpoint_manifest.csv \
  --output /data/wm-a100-pr62-audit-20261112/pr62-device-event-conditional.json \
  --device cuda --batch-size 32
```

## 8. 下一步

- 先补充真实、独立的故障发生/恢复事件覆盖，或明确构造可观测的事件目标和评估协议；不要用同一 validation 反复调阈值后称为泛化。
- 在事件基数足够前，不启动 H5/H20 自回归、不启动在线调度、不扩大网络。
- 若事件条件模型仍只学持久性，应优先检查事件标签可预测性、前兆可观测性和损失/目标设计，而不是换更大模型。
