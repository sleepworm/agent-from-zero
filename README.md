# Agent From Zero

大语言模型擅长理解上下文、生成文本，却不会自动知道公司的实时库存，也不能自己执行程序。Agent 是围绕模型搭建的一套应用程序：模型判断下一步，程序调用外部工具并返回结果，两者在一个受控制的循环里协作，直到完成任务或安全停止。

这个仓库从一次最小的模型调用开始，亲手搭建一个 Inventory & Quote Agent（库存与报价助手）——一个 commit 只引入一个概念，每个概念都先设计破坏性实验、再动手修复。

> 一个只会生成文本的模型，怎样逐步成长为能做事、会失败、可恢复，而且能够证明自己表现如何的可靠 Agent？

## 怎么跑

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # 填入 DEEPSEEK_API_KEY
pytest                 # 跑测试（全部用 mock，不需要真实 API Key）
```

## 章节目录

| 篇目 | Tag | 这一篇解决的问题 |
|---|---|---|
| [0.1 先让程序和模型说上第一句话](docs/00-01-basic-chat.md) | `v0.0.1-basic-chat` | 认识 Chat Completion 和请求边界；看清"会生成文本"不等于"知道业务事实" |
| [0.2 模型调用失败时，程序应该看见什么](docs/00-02-error-handling.md) | `v0.0.2-error-handling` | 真实触发认证、网络和请求错误，把 Provider 异常翻译成应用可以采取行动的错误类型 |
| [0.3 一次回答到底花了多少时间和钱](docs/00-03-cost-latency-tracking.md) | `v0.0.3-cost-latency-tracking` | 为每次模型调用记录 token、延迟和估算成本 |
| [1.1 先给模型一份查库存的说明书](docs/01-01-inventory-tool-schema.md) | `v0.1.1-inventory-tool-schema` | 写出真实库存函数和 Tool Schema，分清可执行实现和模型可读契约 |

`git checkout <tag>` 能回到任意一步重新跑一遍；`git diff <tag1> <tag2>` 能直接看某一步真正改了什么。这张表只列已经写完的章节，随着新 commit 落地继续往下加。

## 项目原则

- 一个 commit 只引入一个核心概念
- 每个 commit 都必须能 `pytest` 跑通，不留半成品状态
- 每个引入新能力的 commit，能设计出破坏性实验就必须先写实验、再解决
- 每个阶段代码、测试、文档一起长出来

## 目录结构

```text
agent-from-zero/
├── src/
│   ├── tools/            # get_inventory 等
│   └── agent/             # llm.py
├── tests/                 # 每个 commit 对应的测试
├── docs/                  # 每个 commit 对应的教材章节
└── failure_taxonomy.md    # 活文档，从系统性触发失败的那个 commit 起持续更新
```

待深入研究、暂时先记下来不细究的问题见 [`TODO.md`](TODO.md)。
