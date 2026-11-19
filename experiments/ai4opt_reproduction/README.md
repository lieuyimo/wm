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

## 求解入口

Linux 上由 OR-Tools 环境运行。给定一个标准 `.fjs` 实例和正数时限，命令输出 JSON。下面的公开示例在仅给定时限时返回找到的最好可行排程。

```bash
"$CP_PYTHON" experiments/ai4opt_reproduction/fjsp_solver.py \
  "$TUE/data/fjsp/brandimarte/Mk01.fjs" --benchmarks "$TUE" \
  --seconds 3 --output schedule.json
```

默认策略为 `auto`。可用 `--strategy cp-sat` 或 `--strategy hgnn` 做固定策略对照。`--target` 可指定 makespan 目标，达到目标后提前结束。启用 HGNN 时配置 `--upstream "$FJSP_DRL" --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" --inference-python "$TORCH_PYTHON"`。模型缺失、校验失败或推理超时会记录原因，并将剩余时间交给 CP-SAT。为回退保留总时限的 20%。到期时返回已经校验的最好可行解，未找到可行解时 `makespan` 为 `null`，排程为空。

JSON 含 `decision_reason`、`fallback_reason`、`makespan`、`solver_status`、`within_budget` 和完整工序排程。作业、工序和机器下标从 0 开始。`end_to_end_seconds` 从读取实例前开始，包含特征计算、按需模型加载、推理、搜索和排程校验，结束于构造返回对象。进程启动与输出文件写入发生在求解请求之外。`--output` 创建新文件，省略时写到标准输出。

求解入口的验收脚本为 `fjsp_solver_acceptance.py`。它使用固定版本的 `data/fjsp/dauzere_paulli` 全部 18 例，先检查 CLI 导出、短时限、加载失败、推理超时与真实 HGNN 推理，然后使用同一求解入口比较三种策略。跨基准比较使用常驻推理进程和统一的 10 秒请求预算。
