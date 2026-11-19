# 标准 FJSP 求解策略与 CPU 验证

HGNN+PPO 生成完整排程后可为 CP-SAT 提供初解。求解前选择策略直接读取总工序数与可选加工区间数。在总工序数不少于 150 且可选加工区间数不少于 267 时启用 HGNN，其余交给 CP-SAT。两个阈值来自 20 个开发实例的配对测量，独立测试期间保持固定。

## 已冻结的 Hurink 结果

18 个独立实例来自 edata、rdata 和 vdata，每个实例跑三个种子。10 秒预算覆盖输入读取、建模、探测、CPU 推理和排程校验。每例达标目标为独立纯 CP-SAT 参考运行在 60 秒搜索预算内得到的完工时间。

| 策略 | 1 秒达标 | 10 秒达标 | HGNN 调用 |
| --- | ---: | ---: | ---: |
| 纯 CP-SAT | 22/54 | 36/54 | 0/54 |
| 始终启用 HGNN | 13/54 | 37/54 | 54/54 |
| 短探测后选择 | 23/54 | 35/54 | 27/54 |
| 求解前选择 | 24/54 | 42/54 | 12/54 |

配置、输入指纹、目标和逐例结果见 [Hurink 实验归档](evidence/hurink-static-20261120/report.md)。实验使用同一批 18 个实例做三轮求解重复，支持这批正工时标准 FJSP 的比较。

## 重跑冻结续验

CP-SAT 环境使用 Python 3、OR-Tools 9.11.4210，推理环境使用 PyTorch 2.5.1、Gym 0.26.2。数据和模型实现分别固定为 [TU/e 基准 d088a265](https://github.com/ai-for-decision-making-tue/Job_Shop_Scheduling_Benchmark_Environments_and_Instances/tree/d088a26582f5cb79c6b4f2ad94f68ac7c50878cb) 和 [Song 等的 HGNN+PPO d7637b70](https://github.com/songwenas12/fjsp-drl/tree/d7637b70029be6cb8fe78e450dcb07767876daea)。墙钟时间受 CPU 与并行配置影响。当前代码通过受控路径使用私有模型文件。

将归档的 `prior-protocol.json` 复制到运行目录，在副本中填入由模型持有人验证的检查点 SHA-256。先从指定版本的 TU/e 检出生成 `source.tree`，再调用实验入口。

```bash
git -C "$TUE" ls-tree -r --full-tree d088a26582f5cb79c6b4f2ad94f68ac7c50878cb \
  data/fjsp/hurink > source.tree

"$CP_PYTHON" experiments/ai4opt_reproduction/fjsp_static_start.py \
  --benchmarks "$TUE" --upstream "$FJSP_DRL" --manifest "$EVIDENCE/benchmark-manifest.json" \
  --prior-protocol ./prior-protocol.json --prior-policy "$EVIDENCE/prior-policy.json" \
  --development "$EVIDENCE/development.jsonl" --source-tree ./source.tree \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --inference-python "$TORCH_PYTHON" --output ./fjsp-results --solver-workers 0
```

`TUE` 是固定版本的基准检出，`FJSP_DRL` 是固定版本的模型实现检出，`EVIDENCE` 指向本目录下的 `evidence/hurink-static-20261120`。模型持有人提供 `CHECKPOINT` 和 `CHECKPOINT_SHA256`。省去重复完成的样本可在同一配置下用 `--resume` 续跑。
