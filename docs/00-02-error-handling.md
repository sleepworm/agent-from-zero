# 00.2 — Handle Three Basic API Failure Modes

> Commit: `feat: handle three basic API failure modes`
> Tag: `v0.0.2-error-handling`

## 1. Problem

Commit 0.1 的 `chat()` 完全没有异常处理——`get_client().chat.completions.create(...)` 一旦出错，异常会带着 `openai` SDK 内部的原始类型直接甩给调用方。在写任何业务逻辑之前，先要知道"调用失败"具体长什么样、能分成几类、彼此该怎么区分对待——这不是为了这一个 commit 本身：Phase 5（Retry）需要知道哪些失败值得重试、哪些不值得；Phase 4（Fail Closed）需要一套"我们自己定义的失败分类"，而不是散落在业务代码各处的 `openai` 库内部细节。

## 2. Minimal Example

`src/agent/llm.py` 新增三个异常类型，并给 `chat()` 包上一层分类：

```python
class ChatError(Exception):
    """chat() 所有已分类失败的公共基类。调用方既可以精确 catch 某一个子类，
    也可以只认这一个基类，一网打尽。"""


class AuthenticationFailure(ChatError):
    """API Key 缺失或错误——重试没有意义，必须先修好凭证。"""


class NetworkFailure(ChatError):
    """连不上或超时——本质都是"这次没打通"。"""


class BadRequestFailure(ChatError):
    """请求本身不合法（模型名不存在、参数超范围、prompt 太长等）——
    重试没有意义，必须先改请求内容。"""


def chat(user_message: str, system: str = DEFAULT_SYSTEM, temperature: float = 0.0) -> str:
    try:
        response = get_client().chat.completions.create(
            model=DEFAULT_MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
    except AuthenticationError as e:
        raise AuthenticationFailure(str(e)) from e
    except APIConnectionError as e:
        raise NetworkFailure(str(e)) from e
    except BadRequestError as e:
        raise BadRequestFailure(str(e)) from e

    return response.choices[0].message.content
```

三个新类型都继承同一个 `ChatError` 基类——这是为后面 Phase 5 的 Retry 逻辑打的地基：只有 `NetworkFailure` 值得重试（临时的连接问题，多试几次可能就通了），`AuthenticationFailure`/`BadRequestFailure` 不管重试多少次结果都一样（凭证错了就是错了，请求本身不合法也不会因为多试几次就变合法），不该在这两种情况上浪费时间。

**为什么要重新包一层，而不是让调用方直接 `except openai.AuthenticationError`**：这一层转换的意义不在于"现在看起来更好看"，而在于**让后面的 Agent Loop、Retry 逻辑，只需要认识这三个我们自己定义的类型，不需要认识 `openai` SDK 内部的异常体系**。以后如果要换一个 Provider（比如把 DeepSeek 换成别的 OpenAI 兼容服务，或者干脆换一个完全不同的 SDK），只需要改这一层的 `except` 映射，调用方的代码一行都不用动。

## 3. Break It / Observe

三种失败模式都是**先对着真实的 DeepSeek API 触发一次，观察真实返回的异常类型和消息，再照着写分类代码**——不是先写代码再假设失败会长什么样。

### 实验 1：错误的 API Key

```python
client = OpenAI(api_key="sk-invalid-wrong-key-00000000000000", base_url="https://api.deepseek.com")
client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": "hi"}])
```

真实观察到的异常：

```text
TYPE: openai.AuthenticationError
MESSAGE: Error code: 401 - {'error': {'message': 'Authentication Fails, Your api key: ****0000 is invalid', 'type': 'authentication_error', 'param': None, 'code': 'invalid_request_error'}}
```

### 实验 2：网络不可达 / 超时

两种子场景都单独测了，因为一开始不确定它们是不是同一个异常类型：

**指向一个不存在的域名**（模拟 DNS 解析失败）：

```text
TYPE: openai.APIConnectionError
MESSAGE: Connection error.
```

**真实超时**（`timeout=0.01` 秒，指向真实的 DeepSeek 地址）：

```text
TYPE: openai.APITimeoutError
MESSAGE: Request timed out.
```

查了一下这两个异常类型的继承关系：

```python
>>> openai.APITimeoutError.__mro__
(APITimeoutError, APIConnectionError, APIError, OpenAIError, Exception, BaseException, object)
```

**`APITimeoutError` 本身就是 `APIConnectionError` 的子类**——这是一个提前发现、省了一次返工的细节：原本以为超时需要单独一个分类和一个 `except` 分支，实际上 `except APIConnectionError` 一行代码就同时覆盖了"连不上"和"连上了但超时"两种情况，因为对调用方来说它们本质是同一类问题——这次没打通，值得用同一种策略（比如 Phase 5 的重试）去应对。

### 实验 3：请求本身不合法

```python
client.chat.completions.create(model="deepseek-chat-nonexistent-model", messages=[{"role": "user", "content": "hi"}])
```

真实观察到的异常：

```text
TYPE: openai.BadRequestError
MESSAGE: Error code: 400 - {'error': {'message': 'The supported API model names are deepseek-flash, deepseek-v4-pro, but you passed deepseek-chat-nonexistent-model.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}
```

这里有一个值得记录的插曲：错误信息说"支持的模型名是 `deepseek-flash`、`deepseek-v4-pro`"，里面并没有列出 Commit 0.1 到现在一直在用的 `deepseek-chat`——看起来像是在暗示这个模型名已经不再受支持了。立刻单独验证了一次：

```python
from src.agent.llm import chat
chat("A100 还有多少库存？")
```

结果 `deepseek-chat` 依然正常返回了回答，说明这条错误信息只是"你传的名字我不认识，这是几个我认识的建议"，不是"全部有效模型名单"——`deepseek-chat` 仍然可用。**这条经验和 Commit 0.1 观察到的"模型这次没有诚实拒绝"是同一个主题的另一次体现：拿到一条信息（不管是模型的回答还是 API 的错误信息）之后，先验证它字面上的意思是否准确，不要直接当真去改动别的地方。**

原计划里"超长 prompt"是触发 `BadRequestError` 的另一种常见方式，这里选择用"不存在的模型名"代替——构造一个真正超过 context window 的 prompt 需要生成几万到十几万 token 的文本，既慢又有不必要的输入成本，而且不同模型的 context window 大小不一样，容易测不准；不存在的模型名更快、更便宜、也更确定会触发同一类异常。

### 一个意外的插曲：`openai` SDK 底层换了 HTTP 客户端库

`pyproject.toml` 里 `openai` 的版本约束写的是 `>=1.50.0`（一个比较宽松的下限），这次实际装到的是 `3.13.0`。写测试时想复用 `httpx.Request`/`httpx.Response` 去手动构造和 Break It 里观察到的完全一致的异常实例，结果 `import httpx` 直接报 `ModuleNotFoundError`——查了一下这个版本的 `openai` 依赖的包，已经变成了一个叫 `httpx2` 的包（`openai.AuthenticationError.__init__` 的类型标注里写的是 `httpx2.Response`，不再是 `httpx.Response`）。这不是这个项目自己的选择，是上游依赖库自己的演进；这条约束写得越宽松，越有可能在不知不觉间撞上这类底层库变动，好在这次只影响测试代码怎么构造异常实例，`chat()` 本身的分类逻辑完全不受影响。

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
tests/test_llm.py::test_chat_returns_the_model_text PASSED               [ 11%]
tests/test_llm.py::test_chat_sends_system_and_user_messages PASSED       [ 22%]
tests/test_llm.py::test_chat_accepts_custom_system_and_temperature PASSED [ 33%]
tests/test_llm.py::test_get_client_reads_api_key_from_env PASSED         [ 44%]
tests/test_llm.py::TestFailureClassification::test_authentication_error_is_reclassified PASSED [ 55%]
tests/test_llm.py::TestFailureClassification::test_connection_error_is_reclassified_as_network_failure PASSED [ 66%]
tests/test_llm.py::TestFailureClassification::test_timeout_error_is_also_reclassified_as_network_failure PASSED [ 77%]
tests/test_llm.py::TestFailureClassification::test_bad_request_error_is_reclassified PASSED [ 88%]
tests/test_llm.py::TestFailureClassification::test_all_three_failures_share_a_common_base_class PASSED [100%]

9 passed in 0.52s
```

新增的五个 `TestFailureClassification` 测试**不重新触发真实的网络故障**（那样会让测试变慢、变得不确定），而是用 `openai.AuthenticationError`/`APIConnectionError`/`APITimeoutError`/`BadRequestError` 的真实构造函数，手动造出和第 3 节实验里观察到的完全一致的异常实例，再验证 `chat()` 有没有把它们正确重新分类成我们自己的类型、原始错误信息有没有被保留、三种类型是否共享同一个基类。这是"先用真实环境观察一次现象，再用 mock 稳定地验证这个现象背后的逻辑"这个方法论第二次出现（第一次是 Commit 0.1 的诚实拒绝观察）。

## 7. Exercises

1. 触发一次真实的"超长 prompt"导致的 `BadRequestError`，对比它的错误信息和"不存在的模型名"导致的错误信息，看 `type`/`code` 字段是否相同。
2. `APIConnectionError` 目前被统一归为 `NetworkFailure`。如果要进一步区分"DNS 解析失败"（大概率是配置错误，重试没用）和"真实超时"（大概率是临时的，重试可能有用），你会怎么改 `chat()` 的异常处理逻辑？
3. 故意在 `.env` 里填一个空字符串当 API Key（而不是完全不设置这个环境变量），跑一次 `chat()`，观察这和"完全没有 `DEEPSEEK_API_KEY`"（会在 `get_client()` 里直接触发 `KeyError`）是不是同一种失败，现在的代码有没有区分这两种情况。

## 8. What We Learned

三种失败模式不是凭空想象出来的分类，是对着真实 API 逐一触发、记录下真实异常类型和消息之后才定下来的；`APITimeoutError` 是 `APIConnectionError` 子类这个细节，提前避免了一次不必要的重复代码；"错误信息里提到的内容不一定是权威事实，用之前先验证"这个习惯，在 Commit 0.1（模型的回答）和这一步（API 的报错信息）里各出现了一次，是同一个更大原则的两次体现。

这一步还意外撞见了一个和"失败分类"本身无关、但同样值得记住的教训：**版本约束写得越宽松（`>=1.50.0`），代码就越可能在不知不觉间用上一个和写代码时假设的环境不完全一样的依赖版本**——这次只是测试代码需要多认识一个包名（`httpx2`），影响很小，但它提醒了一件事：真实环境里跑出来的结果，永远比"当初设计时脑子里想的样子"更权威，这也是这本教材第三次在 Commit 0.1/0.2 里强调"先跑起来看真实情况，再动手写代码或下结论"的原因。
