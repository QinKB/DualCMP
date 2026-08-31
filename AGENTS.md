# Repository Rules

默认用中文，先给结论；只报告实际观察、验证、风险和未完成项。

## Route

- 实质代码、ML、数据、性能、训练或评估工作先读 `code-implementation`。
- 涉及既有设计、checkpoint、实验、恢复或 Serena 时先读
  `memory-maintenance`，再核对当前代码与 Git 状态。
- 本地论文与关联代码优先 PaperMeld/PaperMason；仓库语义优先 Serena，
  字面搜索用 `rg`，当前 API/上游行为查官方来源。
- 阶段切换或上下文压缩后重新检查本文件和适用 Skill。

## Work

- 开始前明确最终用户功能、真实入口、数据、输出、失败行为和验收标准。
- 优先复用已验证实现；版本、接口、假设和许可证必须匹配。metric、loss、
  matching cost 不得自行发明或静默改变语义。
- 派发实现必须写明复用的仓库、文件和符号；未证明现有实现不适配并获 Lead
  批准前，不得自造算法、协议、数据语义或生命周期。
- 代码、配置、命令和文档优先项目相对路径；外部数据/工具通过配置或环境提供。
- 保护无关用户改动；不做破坏性 Git、外部写入或宽泛清理。

## Agents

- Lead 负责目标、架构、分解、集成、最终验收、Git、memory 和结论。
- 角色：`scout`=Luna/medium，`runner`=Luna/low，
  `worker`=Terra/medium，`reviewer`=Terra/high，`researcher`=Terra/medium。
- 常规派发使用 `fork_turns: "none"`；禁止 `fork_turns: "all"`，避免继承
  lead 模型而破坏角色路由。
- 任务必须有明确输入、边界、验收、停止条件，并带来实际并行、隔离、专业性或
  独立判断收益；不要为了占满槽位派发。
- 相同角色和相近上下文优先 `followup_task` 复用原 agent。agent 中断时先用原
  canonical task 名继续，并检查共享工作树是否已有部分修改；只有工具明确判定原
  agent 不可恢复且当前没有合适角色时才新建。
- 一个紧密代码路径只允许一个 writer；不要重叠写入。

## Acceptance

- 派发实现任务前先固定最终功能验收：真实入口、本地数据、资源配置、权威
  evaluator/protocol、样本数、预算和数值阈值；不得看结果后改门槛。
- Worker 必须自行完成最接近最终功能的短验收。语法、import、shape、mock、单元测试
  或孤立模块正确，只是中间证据，不代表功能完成。
- 数据/I/O 必须由真实下游入口消费；学习组件必须走通真实
  `forward -> loss -> backward -> optimizer -> checkpoint -> inference -> metric`。
  需要学习能力时，以预先声明的小子集过拟合或短训练 metric 达标为准；loss 下降
  本身不算通过。
- 未接入最终路径的模块必须标记为 incomplete。中间 Git checkpoint 只是恢复点，
  不得表述为功能验收完成。
- `runner` 只用于长时、资源密集或高噪声任务，如双 GPU 训练、过拟合、完整评估、
  benchmark 和长日志；不要派 runner 重跑 worker 的基本测试。
- 仅在最终端到端验收通过后创建 fresh `reviewer`；中间模块不创建 reviewer，除非
  用户明确要求提前审查。

## Environment

- 当前验证环境：`/home/qkb/miniconda3/envs/coop/bin/python`
  （Python 3.10.20）。CPU 检查设置 `CUDA_VISIBLE_DEVICES=''`。
- 仓库 `environment.yml` 声明 Python 3.7.11；迁移前新代码保持 Python 3.7 语法，
  并明确 grammar 检查不等于 3.7 ABI/runtime 验证。
- 所有 CUDA 测试、训练和评估必须使用本地两张 GPU 的两进程分布式路径，不接受
  单 GPU 结果。运行前检查 CPU、RAM、磁盘和双 GPU。
- `batch_size` 默认表示每 GPU 样本数。先保证正确，再基于实测提高 batch、workers、
  pin/persistent workers 和 prefetch，记录 global batch、峰值显存与吞吐。

## Finish

- 按“静态 -> 单元 -> 真实样本 -> 集成 -> 训练/评估”逐级验证，只报告实际运行项。
- 删除本任务确认无用的 scratch、缓存和中间产物；保留交付物、证据、用户文件和
  不确定文件。
- 功能达到验收后才做隔离本地 Git checkpoint；只暂存任务文件，不 push。
- 仅持久化已验证、非显然、未来有影响且重查代价高的 memory；保持简洁。
