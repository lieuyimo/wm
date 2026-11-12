# PR62：PR61 GPU 单步世界模型的验证复核与故障条件校准门禁

## 1. 实际GPU实验结论与范围

[PR61 A100真实训练](https://github.com/lieuyimo/wm/pull/61)基于PR60封存的数据：TRAIN 60场景/9,046逐tick真值；VAL 12场景/1,517逐tick真值，2幅与训练障碍族不同的7×5验证地图；BLIND TEST 12场景尚未生成。3训练seed×2同参数arm，真实A100 GPU总六次运行，均20epoch、52,953参数，无模型训练失败。原始文件`docs/pr60-gpu-training-results.md`和`examples/evidence/pr60-gpu-training-20261112/`。

| 原GPU验证指标（每seed最佳epoch） | map_off | map_on | 改善幅度 |
|---|---:|---:|---:|
| 公开t+1实体节点masked MSE（12场景宏平均） | 0.0058672971 | **0.0051337263** | **−12.50%** |
| 站点设备可用性 Brier（12场景宏平均） | 0.0043910109 | 0.0043767649 | −0.32% |
| 匿名新任务阶段 MAE | 0.19315230 | 0.17726034 | −8.23% |

这一层次的比较是合法且有价值的：相同模型参数量、各训练seed独立初始化、真正不同地图族的VAL，map-on总体节点误差在3/3seed更好。但这**只是一阶真实历史输入的预测结果**，没有模型自回归、没有独立BLIND、没有真实滚动调度效果。

原始每场景CSV独立计算得：
- 12个真实不同validation场景上**11/12**个场景（先在场景内平均3个训练seed后）`map_on`节点MSE更低，1个相反。36组“场景×seed”中32/36更低。**不能把36当36个独立测试场景，更不能把1,517相关tick当1,517个独立泛化实例。**
- 设备Brier平均极微弱优势，场景平均仅**7/12**更好；3个seed只有1812更好。
- **验证地图0**六场景×3seed设备Brier差`map_on−off = −0.000113263`；**验证地图1**同样为`+0.000084771`，存在**相反方向**的校准表现。
- 整体设备设备Brier由正常/持续状态主导。需要单独按 **健康维持、故障发生、持续故障、故障恢复** 拆分，不能因为整体低至0.0044就断言能够准确提前预测恢复或故障。
- 20 epoch内多臂最佳checkpoint出现在末尾或接近末尾，数据尚不能判断拟合上限。**目前没有增大隐藏维数的合理证据，更没有用BLIND TEST调参的理由。**
- PR61 `best.pt/last.pt`及Adam/RNG都**只存在GPU主机的本地`pr60-gpu-checkpoints-20261112.tar.gz`**。Git已归档compact metrics和checkpoint_SHA清单，尚未归档模型权重。没有这个可信原始存档，第三方不能复算逐tick设备预报，也不能用原训练权重建真实H20展开。

## 2. 本PR新增的实际只读审计程序

1. `experiments/world_model_integration/audit_pr62/pr62_readonly.py`从PR61原`per_scenario.csv`、`summary.json`重新构造12真实场景×3seed×2arm的成对指标；报告按真实独立场景平均的符号、设备Brier按地图的反向差异、按实际业务regime的差值。**纯CPU、只读Git中可用的数据，不调用/加载任何模型或新轨迹，不读取BLIND。**
2. `experiments/world_model_integration/audit_pr62/pr62_device_events.py`是后续**可信权重在场**时由GPU Agent运行的验证子集分组校准程序；只从PR59官方验证`data.json`和PR61 `best.pt`恢复原模型。根据**当前PUBLIC**设备available状态和**真实下一步目标**，划分`healthy_hold / fault_onset / continuing_down / recovery`。每组报告样本数、概率均值、经验真率、模型Brier及**简单持久性baseline**的Brier差。此分组依据真实下一时刻标签，是**评估统计分组，不能作为模型特征或者未来事件预知器**。
3. 原`checkpoint_manifest.csv`中6个best.pt SHA逐一校验；验证其checkpoint的`arm/seed/source_sha/git_head_source`，以及VAL source标签、地图家族和独立场景数量；不修改权重、不推理BLIND、不做GPU训练。
4. `test_pr62.py`和`.github/workflows/pr62-pr61-evidence-readonly.yml`会在GitHub CPU上执行原compact evidence `SHA256SUMS`、成对指标独立复核、设备相位算术测试与校准脚本CLI语法；**这不等于已经加载到原GPU检查点或完成条件校准**。

## 3. 后续GPU Agent具体任务（当前无需新训练）

请在GPU主机执行下列命令。原权重存档由原GPU训练Agent生成，用户拥有其本地源，不能把未确认的路径冒充仓库artifact。

```bash
cd /data/wm-a100-pr60-gpu-20261112/wm  # 如仓库不在此路径，使用实际git clone工作目录
git fetch origin research/pr62-device-state-conditional-audit-20261112
git checkout research/pr62-device-state-conditional-audit-20261112

SRC=/data/wm-a100-pr60-gpu-20261112
ARCHIVE="$SRC/pr60-gpu-checkpoints-20261112.tar.gz"
printf '%s  %s\n' \
  919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14 \
  "$ARCHIVE" | sha256sum -c -

mkdir -p "$SRC/pr62-ckpt-verified"
tar -tzf "$ARCHIVE" | head -40
tar -xzf "$ARCHIVE" -C "$SRC/pr62-ckpt-verified"

# 验证PR59原始不可变validation数据。其真实路径见PR60 GPU交接文档
VALIDATION=/tmp/wm-gpu-source/pr59-validation
(cd "$VALIDATION" && sha256sum -c SHA256SUMS)

python -m experiments.world_model_integration.audit_pr62.pr62_device_events \
  --validation "$VALIDATION/data.json" \
  --trusted-checkpoints "$SRC/pr62-ckpt-verified" \
  --manifest examples/evidence/pr60-gpu-training-20261112/checkpoint_manifest.csv \
  --output "$SRC/pr62-device-event-conditional.json" \
  --device cuda --batch-size 32
```

上面的`cd .../wm`是示例路径而非确认存在的本地目录，必须按实际训练机git目录调整。验证完成后`pr62-device-event-conditional.json`应包含全部6个source-matched模型的逐场景/两地图、各设备事件相位真实样本数、模型与持久性Brier对照。归档新artifact到用户授权的私有GitHub Actions artifact、受控可读存储或用户明确指定安全位置，并保留checkpoint原archive SHA，**不把大型模型文件直接提交公开Git仓库**。

### 3.1 必须复核的统计和实验结论

- 先列**事件基数**（故障onset和真实recovery各有多少个独立站点事件），如只有15个恢复事件或更少，不得用平滑总体Brier/分样本置信区间夸大可靠性；
- 用3个训练seed、12独立场景和2地图族分别给出t+1 Brier、故障/恢复子群预测置信度，和**persistence**对照；若设备自然转移稀疏，识别指标偏差、前兆可观测性与真实潜在不可预测性；
- 报告行动条件下任务发布、工作站服务/资源拥堵的误差；别只关注站点健康概率；
- 对验证数据执行描述性分组分析不会打开BLIND，但**不能用同一验证集反复调校参数，然后把调过的结果称为独立泛化性能**；
- 若原权重在该阶段不可交付，明确`BLOCKED_CHECKPOINT_ARCHIVE_NOT_SHARED`，不能从3seed汇总CSV反推逐tick概率。

## 4. 后续世界模型路线（未执行）

只有PR62事件门禁证明系统动态可预测/可校准，才推进：

- **G2：H5/H20合法递归**。每一步世界模型接收`当前公开状态+合法调度动作+公开地图`，预测下一步真实状态概率；在原BusinessSimulator的生命周期/通行/资源合法规则约束下实现合理的逐状态更新，外生任务发布与设备维修抽样只来自公开历史条件概率。未来真实私有事件只作为离线标签对照。测H5/H20分组误差、物理合法性及校准，对照`map_off`/persistence/V4。
- **G3：决策感知学习**。证明H20候选动作未来状态和可行排序真实相关后，再考虑损失`L_pred + λ L_decision`。冻结`λ=0`控制臂、公平参数/数据/训练预算，报告预测校准与线上makespan双指标。不得仅对原调度评分做技巧性修改并改名叫“世界模型”。
- **G4：独立盲测**。选择模型/超参与最优checkpoint后才生成`horizontal_shelf_blind_topology`测试图真值，只进行一次首要评估；未经用户单独授权不触碰。

**当前裁决：PR61证明了公开地图对一步节点状态学习有效，但尚未证明设备故障与恢复的条件预测能力，也没有可从Git恢复的权重，因此下一阶段先做可复核的事件校准，而不是盲目扩大网络或宣称H20成功。**
