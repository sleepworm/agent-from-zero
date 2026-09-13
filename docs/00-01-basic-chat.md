# 00.1 — Basic Chat Completion

> Commit: `feat: basic chat completion`
> Tag: `v0.0.1-basic-chat`

## 1. Problem

在写任何"Agent"之前，先要能可靠地跑通最短的一条路：程序发一句话给模型，拿到一句回复。这一步不涉及工具、不涉及历史、不涉及任何业务逻辑——只是把"调用一次 LLM"这件事本身搞清楚：一次请求里到底带了什么字段、`temperature` 在控制什么、拿到的响应对象长什么样。这是后面十几个 Phase 的地基，跳过这一步直接上手写 Agent，遇到问题会分不清是"模型调用本身出了问题"还是"业务逻辑出了问题"。

主线场景（贯穿整本教材）：一家销售工业设备的小公司，客户会问诸如"A100 还有多少库存？"这样的问题。这一步先不接真实库存数据，只看模型在完全没有外部信息时会怎么回答。

## 2. Minimal Example

`src/agent/llm.py`：

```python
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_SYSTEM = "You are a helpful assistant for a small industrial equipment company."
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL)
    return _client


def chat(user_message: str, system: str = DEFAULT_SYSTEM, temperature: float = 0.0) -> str:
    response = get_client().chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content
```

几个动手前就定下来的设计决定：

- **DeepSeek 用的是 OpenAI 兼容协议**——直接用 `openai` 这个 SDK，只是把 `base_url` 换成 DeepSeek 的地址，不需要单独装一个 DeepSeek SDK。`system` 不是一个独立的参数，而是 `messages` 列表里的第一条记录，和 `user` 平级。
- **`get_client()` 是一个惰性单例**——只有真正调用 `chat()` 时才会去读环境变量、建立客户端。这样做的直接好处是：`import` 这个模块不需要环境变量已经存在，测试可以直接 `monkeypatch` 掉 `get_client`，完全绕开真实网络请求，不需要配一个假的 API Key 才能跑测试。这是一个会被后面几乎每一个 commit 的测试复用的模式。
- **`temperature=0.0` 是默认值**——Agent 场景要的是"同样的输入尽量得到同样的行为"，方便调试和验证，不是聊天场景那种鼓励"每次都有点新意"的用法。

### 2.0 先搞清楚几个基本概念：Chat Completion API、`messages`、`role`、`temperature`

这一节是给完全没接触过大模型 API 的读者补的背景知识——后面每一章都会直接用这些词，不再重复解释。

**Chat Completion 是什么**：它是一个 HTTP 接口的名字（`POST /chat/completions`），"OpenAI 兼容"说的就是这个接口的请求格式、响应格式几乎所有主流模型厂商都照抄了一份，DeepSeek 也是其中之一。它的核心行为其实很简单：**你把"到目前为止的整段对话"一次性发过去，服务器根据这段对话生成"接下来该说的下一句话"，然后把结果返回给你**。

关键的一点、也是后面 Phase 6（Context & Memory）才会正式处理的一点：**这个接口本身完全不记事**。服务器不会替你保存"上一次问了什么、这次是第几轮对话"——每次请求都是一次独立的、无状态的调用。如果想要"模型记得上一句话"，唯一的办法是自己把完整的历史拼进这次请求的 `messages` 里，一起发过去。这一步（Commit 0.1）每次调用只发一条 `user` 消息，天然就是"每次都从零开始"，这是当前这版代码最简单、但也最受限的地方。

**`messages` 是一个列表，列表里每一条都有 `role` 和 `content` 两个字段**。`role` 只有三种标准取值：

- `system`——设定这次对话的背景规则、身份、行为约束，通常放在列表第一条。相当于在对话开始之前，先给模型定一个"人设"和"行事准则"，用户看不到这句话，但它会影响模型接下来的每一句回答。这一步的 `DEFAULT_SYSTEM` 就是"你是一家小型工业设备公司的助手"这句话。
- `user`——真正的用户说的话。
- `assistant`——模型自己之前说过的话。这一步的代码还用不到这个角色（因为没有历史），但后面只要一涉及"多轮对话"或者"模型自己先说了一句话、再调用工具"这类场景，`assistant` 这个角色的消息就会开始出现在 `messages` 列表里。

**`model` 参数指定这次请求打给哪一个具体的模型**——不同厂商、同一厂商不同版本的模型，能力、价格、速度都不一样，这个字符串就是在"点名"要用哪一个。这一步用的是 `"deepseek-chat"`。

**`temperature` 是一个控制"输出有多随机"的旋钮**，取值范围通常是 0 到 2。模型每生成一个字/词，本质上是在一个"下一个字可能是什么"的概率分布里做一次选择——`temperature` 越接近 0，就越倾向于每次都选"概率最高的那个选项"，结果越稳定、越可预测；`temperature` 越高，就越会掺入一些"不是最优但也说得通"的选项，结果更有变化、但也更不可控。这一步选 `0.0`，是因为 Agent 要处理的是"库存够不够、要不要报错"这类需要**可预期**的判断，不是需要创意的写作场景。

顺带提一句这里还没用到、但后面（Commit 0.3）会正式登场的词——**token**。模型既不是按"字"也不是按"单词"计费和计算的，而是按 token：一个粗略的直觉是，token 是模型把文本切分成的最小处理单位（可能是半个词、一个词、也可能是一个标点），一段中文通常比看起来的字数占用更多 token。这一步先不用关心具体怎么切、怎么算，只需要知道：请求越长（`system` 越长、历史消息越多），花的 token、花的钱、花的时间就越多——这条直觉在 Phase 7（Context Pruning）会变得非常重要。

### 2.1 `client.chat.completions.create()` 和 `response.choices[0]` 分别在做什么

这两行代码后面会在每一章反复出现，这里一次性讲清楚，之后不再重复。

**`client.chat.completions.create(...)` 这条方法链，对应的就是前面说的那个 REST 接口**：`client.chat.completions` 是"对话补全"这一类资源的入口，`.create()` 是这类资源上"新建一次"的标准方法（对应 HTTP 里的 `POST`）。`openai` 这个 Python SDK 本质上是把"发一个 HTTP 请求、解析 JSON 响应"这件事包装成了看起来像调用本地函数的写法——`client.embeddings`、`client.models`、`client.files` 是这个 `client` 对象上其他类别的资源，用的是同一套设计。DeepSeek 选择做"OpenAI 兼容"，就是照抄了这一整套请求/响应的 JSON 结构和调用方式，只是把 `base_url` 换成了自己的服务器地址。

**返回值 `response` 里最关键的字段是 `choices`，它是一个数组，不是一个对象**——原因是这个接口本身支持"一次请求，让模型生成好几个不同版本的候选回答"，用请求参数 `n` 控制要几个（比如传 `n=3`，`choices` 数组里就会有 3 个元素）。**这一步（以及目前为止的每一步）都没有传 `n`，用的是它的默认值 `1`**，所以 `choices` 数组里永远只有一个元素。`response.choices[0]` 里的 `[0]` 不是什么防御性写法或者随手写的下标，它反映的就是这个接口真实的返回形状——即使只要一个结果，这个结果也还是被包在一个数组里给你。

**`choices[0].message` 才是真正装着回答内容的地方**——`message` 上有 `role`（这里永远是 `"assistant"`，因为这是模型说的话）和 `content`（真正的文本）。`choices[0]` 这一层本身还带着一个这一步还没用到的字段 `finish_reason`，它会说明"这次生成为什么停下来了"（正常说完了、还是撞到了长度上限、还是像后面 Commit 1.2 那样因为决定要调用一个工具而提前停止）——Phase 8（Tracing）会正式用到它，这里先记住有这么个字段。

上面这几条都是 Chat Completions API 本身有官方文档记录的标准行为，不是这个项目自己的设计，所以这一步没有专门发一次真实请求去验证"传 `n=3` 真的会返回 3 个 `choices`"——那样只是多花一次不产生新信息的真实调用。这本书"必须用真实调用验证"的原则，用在验证的是"我们自己代码的逻辑、我们对协议细节的理解对不对"，不是拿真金白银去反复确认厂商文档已经写清楚的标准行为。

### 2.2（进阶）`client` 上还有哪些方法链？为什么这一步偏偏选了 Chat Completion？

`chat.completions` 只是 `client` 这棵资源树上的一个分支，不是唯一选项。`openai` 这个 SDK（以及照抄它的各家"OpenAI 兼容"SDK）把整个 API 表面组织成 `client.<资源类别>.<动作>()` 这样的结构，常见的其他分支还有：

- **`client.completions.create()`**——比 Chat Completion 更早的接口，叫 Text Completion（现在已经是 legacy/deprecated 状态）。它没有 `role`/`messages` 这套结构，只接受一段裸文本 prompt，模型负责往后续写，没有"这句是谁说的"这个概念。
- **`client.embeddings.create()`**——把一段文本转换成一个向量（一串数字），不生成新文本，是给"语义检索/相似度比较"用的，Phase 16（企业系统集成/RAG）如果要做语义检索会用到它，但它解决的是完全不同的问题，不能拿来做对话。
- **`client.images.generate()`**、**`client.audio.transcriptions.create()`/`translations.create()`/`speech.create()`**——文生图、语音转文字/翻译/文字转语音，处理的是不同模态，和"文本对话"不是同一件事。
- **`client.moderations.create()`**——内容安全审核，判断一段文本是否违规，不生成回答。
- **`client.models`、`client.files`、`client.fine_tuning.jobs`、`client.batches`**——查询可用模型、文件管理、模型微调任务、批量异步请求，偏运维/训练场景，不是这个项目当下会用到的。
- **`client.responses.create()`**——OpenAI 相对新的一个接口（Responses API），想统一/替代 Chat Completions，加了一些内置工具和更方便的多轮状态管理，但还在演进期，不是所有 Provider（包括 DeepSeek）都跟进实现了。

DeepSeek 作为"OpenAI 兼容"厂商，只挑着实现了这棵树里的一部分，主力就是 `chat.completions`——这也是几乎所有想做"OpenAI 兼容"的 Provider 第一个、也是必须实现的部分。

**这本教材选 Chat Completion，核心原因就一条：`messages` 这套 role-based 结构，天然能装下"多轮对话 + 工具调用"这整套东西，其他接口都装不下。** 具体拆开看：

- **`role` 字段（`system`/`user`/`assistant`/后面 Phase 1 起会出现的 `tool`）是专门为了让"一次对话里发生的所有事情"能有序排列在一个列表里而设计的**——用户说了什么、模型自己说了什么（包括它决定调用工具这件事本身）、工具执行完返回了什么，全部是这个列表里的一条记录，靠 `role` 区分身份。老的 Text Completion 接口只有一段裸文本，没法干净地表达"这句是谁说的、这句是不是工具结果"。
- **Tool Calling（Phase 1 就会用到）这整套协议，是构建在 Chat Completions 之上的**——`tools` 这个请求参数、`tool_calls` 这个响应字段，只在 `chat.completions.create()` 这条方法链上存在。Embeddings、Images 这些接口天然不需要"调用外部函数"这个概念，没有对应的协议设计。
- **它是目前跨厂商兼容性最好的接口**——OpenAI、DeepSeek、以及很多其他厂商都实现了同一套 `chat.completions` 请求/响应形状，这也是为什么这本教材 Phase 17 要做框架/厂商切换对比时，底层协议这一层几乎不用改。Responses API 虽然更新、能力更全，但还没有这种跨厂商的普及程度，选它会牺牲"随便换个 OpenAI 兼容的 Provider 都能跑"这个前提。

一句话总结：选 Chat Completion，不是因为它是唯一选项，而是因为它是目前唯一原生支持"多轮对话结构 + Tool Calling"这两件 Agent 必需品、同时跨厂商兼容性最好的接口。

## 3. Break It / Observe

不是破坏性实验，是先看清楚"什么都不做"时模型的真实行为——这决定了 Phase 1 要解决的问题到底长什么样。

```python
from src.agent.llm import chat
print(chat("A100 还有多少库存？"))
```

真实调用（`deepseek-chat`，2026-09-13）：

```text
我目前无法直接查询实时库存数据，因为我没有连接到你们公司的库存系统或数据库。

要确认 A100 还有多少库存，建议你通过以下方式之一查询：
1. ERP/库存系统：如用友、金蝶、SAP、Odoo 等，查物料编码 A100 的现存量。
2. 仓库台账/Excel：查看最新《库存明细表》或《出入库记录》。
3. 联系仓库管理员/采购：确认是否有在途、锁定、待出库数量。
4. 电商/门店后台：如果有线上店铺，查看商品库存。

如果你愿意，我可以帮你：
- 根据你贴出的库存表/出入库记录，计算 A100 当前库存；
- 帮你做一个库存查询公式/模板；
- 帮你判断"可用库存 = 现有库存 - 锁定库存 + 在途库存"。
```

连续跑了 3 次同样的问题，又追加了一次更"逼"它给数字的问法（"客户在电话里问 A100 还有多少库存，你需要直接答复一个数字给他"），4 次全部一致：模型没有编造任何库存数字，每次都明确说明自己"没有连接到真实库存系统"，甚至在被追问的那次主动说出了理由——"直接报一个数字可能会误导客户"。

这个结果和这本教材最初设计这一步时预期会观察到的"模型编造一个看似合理的库存数字（Hallucination）"不一样——这次没有复现。这本身就是一个值得如实记录的发现：**同一个"没有工具"的场景，具体会不会编造数据，是模型当时的行为特征决定的，不是这段 Prompt 或代码本身能保证的**。也就是说，"模型这次没有撒谎"是运气好，不是这一步的代码提供了任何保障——`chat()` 本身完全没有做任何防止编造的事，纯粹是把消息转发给模型、把回复原样返回。

无论这次观察到的是"编造数字"还是"诚实拒绝"，暴露的都是同一个更根本的问题：**这个助手拿不到任何真实数据，回答不了任何真正有用的业务问题**。诚实拒绝不算错，但对一个要处理真实库存查询的业务场景来说同样没用——Phase 1 要解决的不是"让模型别撒谎"，是"让模型真的有地方能查到数据"。

## 4. Implement the Fix

见第 2 节。

## 5. Introduce Industrial Solution

不适用，第一次框架对比在 Phase 17。

## 6. Evaluate

```bash
pytest -v
```

实际输出：

```text
tests/test_llm.py::test_chat_returns_the_model_text PASSED               [ 25%]
tests/test_llm.py::test_chat_sends_system_and_user_messages PASSED       [ 50%]
tests/test_llm.py::test_chat_accepts_custom_system_and_temperature PASSED [ 75%]
tests/test_llm.py::test_get_client_reads_api_key_from_env PASSED         [100%]

4 passed in 0.64s
```

四个测试全部用 mock 覆盖，不需要真实 API Key：`chat()` 能正确返回模型文本、请求里 `messages`/`temperature`/`model` 字段都对、自定义 `system`/`temperature` 能生效、`get_client()` 能从环境变量正确读取 API Key。第 3 节那次真实调用没有写进自动化测试——模型具体会不会编造数据是服务端的真实行为，不是我们代码的逻辑，不能用测试断言去锁死一个我们控制不了的结果。

## 7. Exercises

1. 把 `temperature` 从 `0.0` 改成 `1.0`，重复第 3 节的实验，观察连续 5 次回答的措辞差异有多大——这为后面理解"Agent 场景为什么倾向低 temperature"打个直观印象。
2. 换一个和库存无关、模型大概率没有把握的问题（比如问一个不存在的产品型号，或者问一个需要实时数据的问题，如"现在几点"），观察模型是选择诚实说明"不知道"，还是给出一个自信但可能不对的回答。
3. 追查一下：如果把 `DEFAULT_SYSTEM` 里"you are a helpful assistant for a small industrial equipment company"这句话删掉，只留一句通用的"You are a helpful assistant"，第 3 节的实验结果会不会不一样？这本身也是"同一份代码，行为随不可控因素变化"这个主题的一次具体验证。

## 8. What We Learned

`chat()` 本身是最简单的一层转发——没有工具、没有历史、没有任何防护——它唯一的职责就是把"业务代码怎么调用 LLM 这一步的协议细节"封装清楚：`messages` 的结构、`get_client()` 的单例模式（为了让后面所有测试都能不依赖真实网络）、`temperature` 的取舍。

第 3 节最重要的收获不是"模型编不编造数字"这个具体结果，而是**这件事完全不受这一步的代码控制**——同一份 `chat()`，今天观察到的是诚实拒绝，教材原本预期的是编造数字，两者都有可能，且都不是好答案。这也是这本教材从这一步就要建立的心态：不能靠"这次跑起来正常"就当作可靠，需要真正的工具、真正的数据源，Phase 1 就是往这个方向迈出的第一步。
