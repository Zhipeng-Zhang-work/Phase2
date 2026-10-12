# 新电脑 Codex 交接入口

更新：2026-09-19。本目录按主题整理长期对话、代码状态和待办，不是逐字聊天记录。

请先完整阅读 [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)，再阅读对应源码和日志。不要把“代码存在”“语法检查通过”“历史冒烟通过”混为一谈。

## 不可忽略的约定

- 目标模型 `Qwen/Qwen3-32B`；SpecCache 小模型 `Qwen/Qwen3-0.6B`。
- 超算项目路径 `/home/chzhan.chzhipen/LMCache`，不同于本地路径。不要根据本地数据位置改集群 workload。
- 新实验尽量隔离，不破坏主实验和能正常工作的子实验脚本。
- 超算离线。C2KV 数据已全部上传后按用户要求删除本地大文件，仅留说明；不要自动重新下载。
- 最近修复了 C2KV 清洗 sbatch 的 Slurm spool 路径问题，尚待集群重新运行确认。
- 本轮只写交接文档，不修文档中发现的其他问题，不提交远端作业。

## 迁移清单

1. 搬整个仓库，包括未跟踪的新代码、配置、CSV、必要日志及本目录；不要只搬 Git 跟踪文件。
2. 保留 `C2KV/to_do.md`、`C2KV/DATASET_FORMAT.md`、`C2KV/datasets.md`。
3. 单独备份仓库外的作者代码 `/Users/zhangzhipeng/Desktop/C2KV` 和论文 PDF（如桌面的 `SpecCache.pdf`）。
4. 超算数据、模型和环境保持原样；新电脑重建本地 Python 环境，不照搬 macOS 虚拟环境或旧 SSD 配置。
5. SSH 凭据单独安全迁移，不写入本交接。

新 Codex 应按用户下一条任务继续，不要擅自把全部待办一次性实现。
