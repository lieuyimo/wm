# PR64：故障发生时间随机化后的外生事件风险建模审计

**研究性质：真实 CPU 仿真数据 + TRAIN-only 离散时间 hazard 基线 + 独立开发验证。没有新神经模型训练，没有 GPU，也没有打开原 PR59 的独立 BLIND TEST。**

## 1. 研究问题

原[PR63 风险集审计](https://github.com/lieuyimo/wm/pull/63)发现：在故障发生固定于 tick7/13 的原仿真环境中，使用模型尚未接收的绝对时间、精确状态年龄，可以让简单训练集风险基线在相同故障机制的验证环境上改善Brier；但原PR61神经关系世界模型只有4帧历史，没有显式时间。**不能把记住固定合成时刻当作世界模型学会随机设备动力学。**

PR64通过小规模前瞻性的**真实世界源随机化**调查：当不改变原BusinessSimulator工作流、机器人、站点、路径、PR48故障持续时间、调度器及数据结构时，仅在**新独立场景**中随机化隐藏故障**开始时刻**，TRAIN拟合的离散时间风险模型能否在不同事件时间支持的新DEV-VALIDATION场景上稳定好于常数风险。

方法参照：Suresh、Severn、Ghosh，[Survival prediction models: an introduction to discrete-time modeling](https://doi.org/10.1186/s12874-022-01679-6)，BMC Medical Research Methodology 2022。**离散风险概率不是PR64提出的新算法。** 后续多步预测可参考Chua等，[PETS](https://proceedings.neurips.cc/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)，NeurIPS 2018 的不确定性传播思想，不得把它当作已复现的算法。

## 2. 预先冻结的实验设计

`experiments/world_model_integration/outage_pr64/protocol.json`在第一批真实仿真执行前提交，所有数据生成和质量门槛均不按结果调节。

- **24个全新TRAIN事实场景**：复用原PR59六张训练地图及六种业务profile、三类stress，使用新种子命名空间，故障开始时刻通过SHA256哈希在`[6,10,14,18,22,26,30]`中随机选择。
- **12个全新DEV-VALIDATION事实场景**：复用原PR59两张验证开发地图，使用新的独立种子，故障开始时刻在`[8,12,16,20,24,28,32]`中选择；两组的可取时刻**刻意不重叠**。
- 每个primary/secondary station故障均保持原PR48通过私有seed抽取的**故障持续时间**，只改变其开始时刻；多故障不同站点时间重合则采用已注册的下一个候选tick确保真实公开事件能分别统计。
- 每个事实世界仍从原`build_quality_business`和`build_topology_business`构建，直接用原`BusinessSimulator`、原`TrueCPsatBranch`和`PublicTransitionCollector`采集全时序`t→t+1`；使用原资源容量/机器人运动/工序生命周期审计，原PyG公开`public_map`不变。新**隐藏事件时间只存在业务仿真私有状态，不能进入模型输入**。
- 不对这些新CPU事实轨迹运行原PR61神经网络，更不进行GPU参数更新；这是**生成机制可识别性试验**。
- 原 PR59 的12个`horizontal_shelf_blind_topology`测试场景既不使用，也不生成。**新12DEV-VALIDATION不是新的独立最终盲测**。
- 新TRAIN与新DEV-VALIDATION同时改变了地图家族、scenario seeds及故障开始时间支持，因此无法把两组Brier差解释成**纯粹时间变化的严格因果效应**。将来若需研究纯时间机制，应进行相同地图、相同seed和任务、仅故障开始时刻改变的配对试验。

定量门禁：全部真实连续公共时序与合法性检查通过，24 TRAIN /12 DEV源均完整保留（截尾标记不删除），TRAIN实际故障发生与恢复≥28、DEV实际故障发生与恢复≥14，TRAIN起始tick至少5种、DEV至少4种，支持无交叠，公开图无未来隐藏信息，0GPU，0神经训练。

## 3. 实际实验结果

原GitHub Actions：[Run #43264954353](https://github.com/lieuyimo/wm/actions/runs/43264954353)，三项任务全部success，全部源文件SHA审计通过，科学门禁为 **`PASS_TRUE_SHIFTED_OUTAGE_DATA_TO_REVIEW_MODEL_DESIGN`**，无失败项。`source_quality_verdict`仅说明数据质量符合协议，**不是事件预测能力PASS**。

| 真实新数据 | TRAIN | DEV-VALIDATION |
|---|---:|---:|
| 独立新场景 | 24 | 12 |
| 完整t→t+1公开状态转移 | 3,421 | 1,530 |
| UP→DOWN 故障发生 | 32 | 16 |
| DOWN→UP 真实恢复 | 32 | 16 |
| 原业务最终完成 | 22 | 12 |
| 原业务260tick右截尾 | 2 | 0 |
| 已实际观察故障开始tick | 6,10,14,18,22,26,30 | 8,16,20,24,28,32 |
| 静态地图数 | 6 | 2 |
| GPU训练、神经推理 | 0 | 0 |

原始CPU数据归档：[TRAIN Artifact #13966011445](https://github.com/lieuyimo/wm/actions/runs/43264954353/artifacts/13966011445)，[DEV Artifact #13964159663](https://github.com/lieuyimo/wm/actions/runs/43264954353/artifacts/13964159663)；[质量与全量风险集Artifact #13966496424](https://github.com/lieuyimo/wm/actions/runs/43264954353/artifacts/13966496424)。另有[不重新运行仿真的只读原始证据复核Run #43265617968](https://github.com/lieuyimo/wm/actions/runs/43265617968)。

### 3.1 所有设备状态风险时刻的真正验证Brier

模型风险率**仅用新TRAIN真实故障和恢复标签拟合**；DEV没有任何标签参与基线拟合。

| TRAIN-fitted风险方法 | DEV全部站点tick加权Brier | 12场景等权Brier | 相对常数（场景级） |
|---|---:|---:|---:|
| `constant`：设备当前UP/DOWN的条件常数事件概率 | **0.00404266** | 0.00417594 | 0 |
| `age_h4`：模型目前能看见的最近4个公开状态年龄 | 0.00403125 | 0.00415691 | −0.00001903 |
| `clock4`：每4tick一个固定公开绝对时间窗口 | 0.00417272 | 0.00429865 | +0.00012271 |
| `exact_age4`：全时序精确连续状态年龄，4tick分桶（诊断） | **0.00396149** | **0.00408637** | −0.00008957 |

`clock4`**没有沿用PR63为原固定故障时刻特殊设计的tick6/12独立分桶**。它使用事先固定的等宽4tick时段，以减少用既有失败时刻人为设置模型特征的风险。

### 3.2 独立场景配对不确定性

`pr64_gate.py`采用**12个完整开发验证场景**为bootstrap单位，2,000次固定随机种子重采样；不把7,650个相关站点tick或32次事件视为独立场景。

| 方法相对 `constant` 的场景级Brier差（负值更好） | 差值 | 95% 场景bootstrap CI | 12场景内数值较优 |
|---|---:|---|---:|
| `age_h4` | −0.00001903 | [−0.00007288, +0.00003140] | 7 |
| `clock4` | +0.00012271 | [−0.00005372, +0.00037015] | 5 |
| `exact_age4` | −0.00008957 | [−0.00019512, +0.00000745] | 7 |

**所有三个区间都跨0，没有任何年龄/绝对时间风险模型在该小规模随机化故障时间开发集上被确认为稳定优于训练集常数概率。**

这与PR63在固定故障tick7/13时的`abs_tick`诊断优势形成方法学上的警示：简单增加`public_tick`很容易学到合成数据生成器的固定事件规律。当前还不能证明“绝对时间一定坏”，因为场景数少、时钟分桶不同、地图族不同，而且没有原PR61检查点在本次新验证源上的推理。

## 4. Review 裁决和后续研究路线（待review）

**可直接确认的事实：**

1. **世界模型已学到的公开关系状态动态仍成立**：PR61对下一步实体状态的原始正向验证不因本轮外生故障风险实验而作废。
2. **固定时间的故障起点不可作为未来真实世界假设**；PR64控制机制已经验证真实业务仿真可以随机化隐藏故障时间，并保留完整合法公开状态转移数据。
3. **不能声称时钟或精确年龄特征在新时间分布下稳定有优势**；这一次即使`exact_age4`数值较优，场景级CI仍跨0。
4. **不应提高GNN参数量或从头训练新模型**，也不应凭这一轮就进入H5/H20递归调度；当前数据只支持讨论“事件是外生状态转移的概率过程”的架构选择。

**Reviewer 必须先处理的问题：**

- 在业务上，故障发生时刻、设备运行年龄、持续停机时间、维修进展和班次时刻究竟哪些是**真实系统公开可观测、稳定且可部署**的信息？合成固定起始/公开时钟不能替代真实可观测证据。
- 如果设备故障外生随机，是否采用**条件离散时间hazard**，明确`P(UP→DOWN | public history)`与`P(DOWN→UP | public history)`，并把抽样事件送入现有合法规则转移与图模型？如何保证概率校准和H20轨迹多样性？
- 不要让神经网络直接从不知道的隐私时间表或被验证集调优的特殊时刻桶预测下一步真值。
- 如果需要进一步确认时钟泛化，再冻结完全同seed、同地图、同真实业务配置只改变事件开始时刻的**严格配对对照**，不同真实随机排程与事件密度，至少增加独立的真实故障/恢复事件数量；保留此前全部原始失败场景与截尾。
- 在模型确有信息增益且概率合理前，仍**不要启动新GPU**。后续H5/H20可参考成熟PETS的不确定性传播思想，但先以独立合法规则、可观察系统状态和外生分布为起点，逐次验证物理一致性。

**本PR保持Draft等待Review，不自动建构新的GPU训练、不开BLIND、不覆盖原PR61–63任何结果。**

## 5. 源代码与复现

- 协议：`experiments/world_model_integration/outage_pr64/protocol.json`
- 真实业务CPU生成：`experiments/world_model_integration/outage_pr64/pr64_source.py`
- 前向验证风险集与场景bootstrap：`experiments/world_model_integration/outage_pr64/pr64_gate.py`
- 原始数据只读校验及CI呈现：`experiments/world_model_integration/outage_pr64/pr64_readonly_review.py`
- 先验合同测试：`experiments/world_model_integration/outage_pr64/test_pr64.py`
- 真实源CI：`.github/workflows/pr64-shifted-outage-source.yml`
- 真值只读Review CI：`.github/workflows/pr64-readonly-review.yml`

实际Run、原始Artifact及按SHA锁定的来源均保留。没有复制/改写原PR48固定事件起始数据，也没有将PR64新开发验证集冒充原独立BLIND。
