# 系统演示：可播放页面及运行证据

打开 [四策略对照](comparison/comparison.html)，可由表中链接进入对应离线回放。其他页面：[三阶段](three-stage/report.html)、[工作站故障](outage/report.html)、[clinic 活锁恢复](clinic/report.html)。网页内嵌运行数据，无需 Web 服务；选中时间、候选和粒子即可查看业务状态与实际在线预测。`comparison/comparison.json` 保存同一事实轨迹的准确率及完整对照指标。

每个场景目录含 `summary.json`、`report.html`，以及完整 `run.json.gz`、`decisions.json.gz`、`events.jsonl.gz`。例如 `gzip -dc comparison/workflow_pets/decisions.json.gz` 可查看实际 PETS 候选推演。离线配置包含完整场景，在线观测和预测中没有未来任务表、未来故障计划或隐藏服务倒计时。

`logs/wm-system-final-first.log.gz` 与 `wm-system-final-repeat.log.gz` 是两次完整执行；`wm-system-final-compare.log.gz` 为排除测量及环境后的逐项复核；`wm-regression-*.log.gz` 为 190 项本地回归；`wm-demo-pilot-chromium.log.gz` 记录本地浏览器环境无法启动的事实。实际 GitHub Actions 状态应查看对应 PR 检查。`SHA256SUMS` 覆盖本目录全部其他文件。

本例任务、工作站、容量、时间和扰动为仿真假设；已使用过的场景不能标为新的独立测试。指标、已知模型局限、复现命令见[验收记录](../../../docs/system-demo-results.md)与[运行指南](../../../docs/system-demo.md)。
