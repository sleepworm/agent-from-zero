# Agent From Zero

Inventory & Quote Agent —— 从 Hello LLM 一路手写到 Production-Grade Agent，每个 commit 只引入一个概念，每个概念都先设计破坏性实验、再修复。完整执行计划见 [`learning-plans/Reliable_Agent_Learning_Plan.md`](learning-plans/Reliable_Agent_Learning_Plan.md)。

配套的长文课程草稿在独立仓库 [`docs`](https://github.com/sleepworm/docs) 的 `agent-from-zero-docs/` 子目录下维护——本仓库 `docs/*.md` 是每个 commit 的技术笔记，那边是整理成篇的发布稿，两者不是一份内容。

## 怎么跑

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # 填入 DEEPSEEK_API_KEY
pytest                 # 跑测试（全部用 mock，不需要真实 API Key）
```

## 章节目录

跟随 commit 顺序阅读，每个 commit 对应 `docs/` 下一篇同名文档（命名格式 `<phase>-<commit>-slug.md`），可以用 `git checkout <tag>` 回到对应阶段重新跑一遍。Tag 格式 `v0.<phase>.<commit>-slug`，完整 Phase/commit 拆解见执行计划第 4 节。

| Phase | 主题 | 状态 |
|---|---|---|
| 0 | Hello LLM（[0.1](docs/00-01-basic-chat.md) / [0.2](docs/00-02-error-handling.md) / [0.3](docs/00-03-cost-latency-tracking.md)） | ✅ |
| 1 | 第一个 Tool：查库存 | |
| 2 | 多 Tool + 自己实现 Agent Loop | |
| 3 | Tool Validation | |
| 4 | 主动破坏：Tool Failure Injection | |
| 5 | 从失败中恢复：Retry / Timeout / Idempotency / Circuit Breaker | |
| 6 | Context & Memory | |
| 7 | Context Pruning | |
| 8 | Tracing / Trajectory | |
| 9 | Planning：定价 + 折扣 + 报价 | |
| 10 | Evaluation：Golden Dataset + LLM-as-Judge | |
| 11 | Regression Testing / Quality Gate | |
| 12 | Permission & Guardrails（+ HITL 审批） | |
| 13 | State & Checkpoint | |
| 14 | Observability：Metrics + Dashboard | |
| 15 | Subagent / Multi-Agent | |
| 16 | 企业系统集成：真实 DB / CRM / API + MCP | |
| 17 | 换框架重做：MiniCordis / DeepSeek Harness / LangChain / LangGraph / LangSmith | |
| 18 | Production Architecture & Reliability Platform | |

## 项目原则

- 一个 commit 只引入一个核心概念
- 每个 commit 都必须能 `pytest` 跑通，不留半成品状态
- 每个引入新能力的 commit，能设计出破坏性实验就必须先写实验、再解决
- 每个阶段代码、测试、文档一起长出来

## 目录结构

```text
agent-from-zero/
├── src/
│   ├── tools/            # get_inventory / calculate_price / create_quote 等
│   ├── agent/             # loop / memory / guardrails / state
│   └── observability/     # tracing / metrics
├── tests/                 # 每个 commit 对应的测试
├── docs/                  # 每个 commit 对应的教材章节
├── eval/                  # Golden Dataset + Regression 报告（Phase 10 起）
├── frameworks/             # MiniCordis / DeepSeek Harness / LangChain / LangGraph 重写版本（Phase 17 起）
├── learning-plans/        # 执行计划文档（Phase/Commit 详细拆解）
└── failure_taxonomy.md    # 活文档，Phase 4.3 起持续更新，Phase 18.5 做最终核对
```

完整细节见 [`learning-plans/`](learning-plans/) 下的执行计划文档。待深入研究、暂时先记下来不细究的问题见 [`TODO.md`](TODO.md)。

## 这个仓库的来历

这是 Agent Design 教学项目的第二次尝试——沿用之前已经验证过的 Phase/Commit 大纲和章节文档模板（Problem / Minimal Example / Break It·Observe / Implement Fix / Industrial Solution / Evaluate / Exercises / What We Learned），但代码和每篇文档都重新手写，不是从旧仓库直接搬运历史。旧的执行计划文档本身作为路线图被带了过来，具体的实现和教材文字是全新的。
