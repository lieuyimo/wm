# PR78 权重保密与实验可复现规程

**要求（2026-11-14）**：所有原 PR61 A100 训练权重及 PR78 新训练事件头，均为**非公开模型资产**。项目无意公开发布任何模型权重。原权重是否曾存于某台服务器或某次 PR 留言，不构成将其上传到公开 GitHub 的授权。

## 绝对禁止的公开渠道

不得将完整权重、部分权重、新训练 head 权重、含 `state_dict` / `torch.save` 的 checkpoint、优化器状态、checkpoint 归档上传到：

- 此公开 Git 仓库或历史分支（包括 Git LFS、releases、PR 变更和附件）；
- GitHub Actions Artifacts、Actions 缓存、公开训练日志（不得打印张量、密钥或私有内部绝对路径）；
- GitHub Issues / PR 评论、公开文档、公开实验附件；
- 对公众可访问的模型仓库或对象存储链接。

此前可通过 GitHub Actions 接收/上传权重的 `.github/workflows/pr78-verified-original-gnn-cpu-head-training.yml` 已**删除**。公开 CI 保留只需要代码和已公开的仿真数据的测试，不加载任何原始或新训练模型权重。

## 唯一允许的训练和保存方式

1. GPU Agent 在获得授权的受控训练机器上，以本地绝对路径读取原始 PR61 A100 权重，先执行 PR68 原 SHA256 / manifest 校验，禁止用随机权重替代。
2. 在受控训练机器使用 PR78 当前代码与 `--device cuda` 运行，`--output-dir` **必须位于 Git 工作区之外的权限受限目录**，不得将训练结果同步到公开 PR 或 GitHub Artifact。
3. 原始 `best.pt` 和新事件头 `*.pt`、优化器/RNG 状态、可恢复检查点均保存在**受控私有持久存储**，按内部权限/备份策略管理；私有存储位置由授权执行者决定，不在公开仓库披露。
4. PR78 训练入口 `require_private_training_output` 在 `GITHUB_ACTIONS=true` 时，或输出目录位于 Git 工作区时，直接硬失败。新增 `.gitignore` 仅是第二层误操作防护，不能视为安全隔离或覆盖已有跟踪文件。

## 允许公开的可复现研究证据

仅在经检查不含权重、私有样本/账号、凭据或内部敏感路径后，可公开：

- 源码 commit SHA、非敏感训练配置与种子、原模型/数据的**哈希摘要**、参数规模和目标定义；
- 按独立场景汇总的 Brier / ECE / NLL、相位分层、场景数、均值与置信区间；
- 训练轮数、最优 epoch、聚合损失/曲线（禁止日志中直接 dump `state_dict`、tensor 或私有路径）；
- 清理敏感信息后的执行环境版本，以及复现方法的文字说明；
- 是否通过原权重校验、是否生成了私有 checkpoint 的**布尔状态**。

实验复现权重由**获得授权的研究人员**在私有存储中访问。公开研究证据支持审计方法与结果，但不等于任何人可重放未公开的权重。若某项审查要求独立重现原权重推理，应在受控环境中进行，不上传模型。

## GitHub Issues / Project 的验收口径

- [#80](https://github.com/lieuyimo/wm/issues/80)：报告 SHA 校验通过与原权重是否在受控机器可访问；**不上传 checkpoint，不发布内部路径**。
- [#81](https://github.com/lieuyimo/wm/issues/81)：受控 GPU 训练六支，**私有**存储所有原始/新增 checkpoint；公开只提交非敏感汇总指标、六支身份与状态、SHA 摘要、代码版本/清理过的环境摘要。
- [#82](https://github.com/lieuyimo/wm/issues/82)：由获授权评估方在原受控机器执行六支模型评估，公开各模型/场景/相位汇总结果，不上传模型和未审查的逐样本推理资产。
- [#83](https://github.com/lieuyimo/wm/issues/83)、[#84](https://github.com/lieuyimo/wm/issues/84)：后续可公开纯合成、去敏的统计结果；任何训练权重保持私有。

**现状**：未在公开仓库发布 PR61 或 PR78 新训练权重；正式原权重六支 GPU 训练尚未确认完成。相关 Issue、PR 评论及历史训练文档中要求“上传新事件头 checkpoint / Artifact”的文字，应以本保密规程替代。
