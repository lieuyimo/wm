# 系统演示、验收与交接

本功能分支以 PR #15 的 `d7d29923c77a026ed3f4ec5fb3e2934bc4385200` 为基线。默认演示复用 PR #15 的 `empty8 / 71003 / workflow_serial`、混合运动、5 步、3 粒子和 4 候选；它是**既有确认场景的可视化复跑**，不是新的独立测试。参数、任务、容量和故障均为仿真假设。

## 从干净 Linux 环境运行

以下示例需要 Python 3.10、Git、CMake、C++17 编译器、Node.js 和访问公开源码/依赖的网络；不需要 GPU。构建脚本运行上游示例和测试，独立检出固定提交，不改写上游源码。PIBT 上游部分依赖遗漏头文件时构建脚本记录原始失败，并仅通过编译选项包含标准头文件；详见原报告。

```bash
git clone https://github.com/lieuyimo/wm.git
cd wm
git fetch origin feature/integrated-system-demo-20261110
git switch --track origin/feature/integrated-system-demo-20261110

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install 'uv==0.13.5'
uv pip install --python "$(command -v python)" --index-strategy unsafe-best-match -r requirements-paper-reuse.lock

bash tools/pibt_motion/prepare_upstream.sh /tmp/wm-demo-pibt
bash tools/lacam3_recovery/prepare_upstream.sh /tmp/wm-demo-lacam3
python -m tools.pibt_motion.prepare_sources --output /tmp/wm-demo-hospital-sources
export WM_PIBT_LIBRARY=/tmp/wm-demo-pibt/bridge/libwm_pibt_bridge.so
export WM_LACAM3_WORKER=/tmp/wm-demo-lacam3/bridge/wm_lacam3_worker
export WM_HOSPITAL_SOURCES=/tmp/wm-demo-hospital-sources

python -m business_transport.demo compare --output runs/system-comparison
python -m business_transport.demo run --scenario clinic --seed 61000 --policy nearest --motion hybrid --output runs/system-clinic
python -m business_transport.demo run --scenario three_stage --policy workflow_rule --motion legacy --output runs/system-three-stage
python -m business_transport.demo run --scenario empty8 --regime workflow_outage --seed 71003 --policy nearest --output runs/system-outage
```

上述 `/tmp` 和 `runs` 目录必须尚不存在或为空，避免覆盖历史结果。打开 `runs/system-comparison/comparison.html`，点击四种策略进入相应离线回放；clinic 和其他场景打开各自 `report.html`。浏览器无需连接服务器，使用时间轴查看当前已释放任务与资源、候选评分及实际预测；候选和粒子可切换。clinic 在原固定设置下的 LaCAM3 事件可在时间轴 52、92 附近查看。仓库内预生成的演示见[证据目录](../examples/evidence/system-demo-20261110/README.md)。

可以用 `python -m business_transport.demo run --help` 查看选项。`run` 支持 `nearest`、`workflow_rule`、`workflow_pets`、`workflow_structured`，以及 `legacy`、`pibt`、`hybrid` 运动。`--model auto` 自动选择策略要求的原有模型；如显式指定不同模型会报错，不会静默替换。`compare` 固定四种策略，事实预测评价目前只支持 `hybrid`，以保持预测与执行运动接口一致。模型检查点、OR-Tools、原生库或地图源文件缺失时会明确报错。

## 结果与口径

每次 `run` 产生 `run.json`（完整配置与事实、预测）、`events.jsonl`、`decisions.json`、`summary.json` 和 `report.html`。`compare` 为每种方法保存同一组文件，另产生 `comparison.json`、`comparison.html`。记录包含实际策略和模型、窗口及候选预算、Git 提交、Python/PyTorch/OR-Tools 版本、固定上游来源和使用的二进制及检查点哈希。`run.json` 中的静态场景是**离线运行配置**，包含事后审计所需的完整任务表；传给策略的观测和保存的在线预测不包含未来事件表或隐藏倒计时。

页面展示的预测直接来自执行时的 `WorkflowDynamics`，而不是以后来的事实轨迹代替；只将被选候选与对应时刻事后观测并列，且后续实际动作可能不同。未执行候选不展示假想的“事实结果”。“连续未移动”包含正常服务和排队，不等于死锁；设备状态变化和服务暂停标注为从可见状态推断，PIBT/LaCAM3 来自真实执行器记录。

四策略共享同一场景、扰动和运动；其中后三组共享 CP-SAT 候选机制、评分、窗口与预算，最近优先不调用模型。结果表使用完整业务任务，分阶段等待另计；未完成任务纳入完成率分母和受限流转。规则/PETS/结构化模型 1、3、5 步准确率通过 PR #15 评估函数，在**同一条最近优先事实轨迹和同一组选中窗口**上单独计算。窗口重叠且有排除条件，不是独立样本的精度证明。演示开启候选完整轨迹记录，墙钟耗时包含记录开销，与不记录的历史基准不能直接相减。

## 自动验收

```bash
python -m unittest discover -s experiments/system_demo/tests -v
bash tools/system_demo/run_acceptance.sh "$WM_PIBT_LIBRARY" "$WM_LACAM3_WORKER" "$WM_HOSPITAL_SOURCES" /tmp/wm-demo-acceptance-a
bash tools/system_demo/run_acceptance.sh "$WM_PIBT_LIBRARY" "$WM_LACAM3_WORKER" "$WM_HOSPITAL_SOURCES" /tmp/wm-demo-acceptance-b
python tools/system_demo/compare_runs.py /tmp/wm-demo-acceptance-a /tmp/wm-demo-acceptance-b
python tools/system_demo/check_browser.py /tmp/wm-demo-acceptance-a/comparison/workflow_rule --focus decision
python tools/system_demo/check_browser.py /tmp/wm-demo-acceptance-a/clinic --focus recovery
```

重复比较排除墙钟、CPU、内存测量和运行环境来源字段，不排除任务、状态、评分、事件或预测。Node 测试执行实际回放脚本；浏览器检查需要已安装 Google Chrome/Chromium。CPU GitHub Actions 还构建官方原生后端并运行真实浏览器 DOM 检查。PR #14/#15 的历史运动回归由继承的对应工作流执行；各工作流的实际结果以 PR 检查记录为准。

## PR 依赖与合并建议

仓库实际编号中，#9 → #10 → #11 → #12 → #13 → #14 → #15 是一条依赖链；本系统演示以 #15 为目标，是链末的独立分支。仓库已有另一个开放的 [#16](https://github.com/lieuyimo/wm/pull/16)，同样以 #15 为目标，内容是有限候选 oracle 上界与决策瓶颈；它**不是**本系统演示。#17 以已有 #16 分支为目标，#18 是另一目标分支上的草稿，不应未经审查并入本次演示。

建议先逐一审查 #9 至 #15 的代码、检查和证据，再审查本演示及其他并行 PR；如用户将来决定合并，应按依赖从底向上处理，并在每一步核对目标分支和冲突。已开放的 #4 是独立 MAPF-World 迁移研究，#8 是独立 A100 训练证据；二者均以 main 为目标，不是本集成代码分支的 Git 父提交，是否合并应分别审查。该段仅为建议，本次不执行合并或删除任何分支。
