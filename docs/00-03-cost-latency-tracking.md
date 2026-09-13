# 00.3 — Track Token Usage, Cost, and Latency Per Call

> Commit: `feat: track token usage, cost, and latency per call`
> Tag: `v0.0.3-cost-latency-tracking`

## 1. Problem

到目前为止 `chat()` 只返回一句纯文本——每次调用花了多少钱、花了多久、用了多少 token，完全没有留下任何痕迹。这不是这一个 commit 才需要的信息：Phase 14（Observability）要做的是把"很多次调用"的数据聚合成趋势和 Dashboard，前提是"单次调用"这一层的数据先被真实、完整地记录下来。从第一次真正开始跑的调用起就养成这个习惯，比事后回头给几十个调用点里补埋点要容易得多。

## 2. Minimal Example

`src/agent/llm.py` 里，`chat()` 的返回值从裸字符串换成一个新定义的 `ChatResult`：

```python
PRICE_PER_MILLION_TOKENS_USD = {
    "input_cache_hit": 0.003,
    "input_cache_miss": 0.15,
    "output": 0.6,
}


@dataclass
class ChatResult:
    text: str
    tokens_in: int
    tokens_out: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    latency_ms: float
    estimated_cost_usd: float


def _estimate_cost_usd(cache_hit_tokens: int, cache_miss_tokens: int, tokens_out: int) -> float:
    return (
        cache_hit_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD["input_cache_hit"]
        + cache_miss_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD["input_cache_miss"]
        + tokens_out / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD["output"]
    )


def chat(user_message: str, system: str = DEFAULT_SYSTEM, temperature: float = 0.0) -> ChatResult:
    started_at = time.perf_counter()
    try:
        response = get_client().chat.completions.create(...)
    except AuthenticationError as e:
        raise AuthenticationFailure(str(e)) from e
    # ...（0.2 的失败分类保持不变）
    latency_ms = (time.perf_counter() - started_at) * 1000

    usage = response.usage
    cache_hit_tokens = usage.prompt_cache_hit_tokens or 0
    cache_miss_tokens = usage.prompt_cache_miss_tokens or 0
    tokens_out = usage.completion_tokens

    return ChatResult(
        text=response.choices[0].message.content,
        tokens_in=usage.prompt_tokens,
        tokens_out=tokens_out,
        cache_hit_tokens=cache_hit_tokens,
        cache_miss_tokens=cache_miss_tokens,
        latency_ms=latency_ms,
        estimated_cost_usd=_estimate_cost_usd(cache_hit_tokens, cache_miss_tokens, tokens_out),
    )
```

几个动手前定下来的设计决定：

- **延迟用 `time.perf_counter()` 量，不用 `time.time()`**——前者是单调时钟，专门为"测量一段代码经过了多久"设计，不会受系统时间被手动调整、闰秒之类的问题影响；后者是墙钟时间，语义上是"现在几点"，两者概念不同，混用是一个常见的隐性 bug 来源。
- **成本没有直接问 API 要，是自己按 `response.usage` 里的 token 明细乘官方单价算出来的**——Chat Completions API 本身不会在响应里直接告诉你"这次花了多少钱"，只会给"用了多少 token"；`estimated_cost_usd` 前缀里的"estimated"三个字是故意的，它是我们自己按官方定价表算出的估算值，不是账单上会出现的权威数字（真实账单可能因为汇率折算、账号折扣、计费周期取整等原因和这个数字有细微出入）。
- **价格分成三档：cache 命中的输入、cache 未命中的输入、输出**——这是 DeepSeek 定价页真实的三档结构（2026-09-13 抓取），命中缓存的输入 token 单价比未命中的便宜了 50 倍（\$0.003 vs \$0.15 / 1M tokens），这也是为什么 `response.usage` 要把这两类分开报告，而不是合并成一个笼统的 `prompt_tokens` 定价。
- **`chat()` 本身不打印任何东西，只负责返回数据**——这是和某些同类实现不一样的一个选择：把"测量一次调用花了什么"和"要不要把这次结果打印出来"分成两件事，`chat()` 只做前者，打印是调用方（这一步是 `if __name__ == "__main__":`）的责任。这样 `chat()` 可以被安静地在测试、在后面章节的 Agent Loop 内部反复调用，不会因为一个和"生成回答"无关的副作用（打印到控制台）而弄脏日志或拖慢批量调用。

### 2.0 `deepseek-chat` 这个模型名，定价该按哪一档算

`PRICE_PER_MILLION_TOKENS_USD` 里的具体数字，来自 DeepSeek 官方定价页，但那张表是按 `deepseek-flash`/`deepseek-v4-pro` 这两个具体型号报价的，不是按我们代码里写的 `deepseek-chat`——`deepseek-chat` 更像一个"稳定别名"，厂商可以在背后悄悄换它实际指向的具体模型。所以先发了一次真实请求，读 `response.model` 字段确认它现在到底解析到哪一个：

```python
r = chat("A100 还有多少库存？")
print(r.model)  # 实际读的是 response.model，ChatResult 目前没有暴露这个字段
```

结果：`deepseek-chat` 目前解析到 `deepseek-flash`，价格就按这一档抄的（cache 命中 \$0.003、未命中 \$0.15、输出 \$0.6，每百万 token）。这条对应关系不是永久保证的——厂商随时可能悄悄把 `deepseek-chat` 指向别的具体模型，价格跟着变——这也是为什么这份价格表被单独抽成一个模块级常量、加了抓取日期注释，而不是散落写死在计算逻辑里：以后价格变了，改一个地方就行，不用满代码找。

## 3. Break It / Observe

不是破坏性实验，是 DoD 要求的验证性实验：连续问 5 个长度、复杂度都不一样的问题，看 token 数、延迟、成本是不是真的随输入变化。

```python
questions = [
    "A100 还有多少库存？",                                          # 12 字，简单问题
    "A100 和 B200 这两款工业传感器，分别适合什么应用场景，各自的价格区间大概是多少？",  # 45 字，两个子问题
    "……（一个要求设计完整报价流程的长问题，135 字）",
    "你好",                                                        # 2 字，最短
    "……（一个要求详细解释 Chat Completion API、不少于 300 字的问题，97 字）",
]
for q in questions:
    r = chat(q)
    print(f"tokens_in={r.tokens_in} tokens_out={r.tokens_out} "
          f"cache_hit={r.cache_hit_tokens} cache_miss={r.cache_miss_tokens} "
          f"latency_ms={r.latency_ms:.0f} cost_usd={r.estimated_cost_usd:.6f}")
```

真实调用结果（`deepseek-chat`，2026-09-13）：

```text
--- Q1（12 字，"A100 还有多少库存？"）---
tokens_in=24 tokens_out=130  cache_hit=0 cache_miss=24 latency_ms=2312 cost_usd=0.000082

--- Q2（45 字，两个子问题）---
tokens_in=41 tokens_out=74   cache_hit=0 cache_miss=41 latency_ms=949  cost_usd=0.000051

--- Q3（135 字，要求设计报价流程）---
tokens_in=91 tokens_out=1610 cache_hit=0 cache_miss=91 latency_ms=8820 cost_usd=0.000980

--- Q4（2 字，"你好"）---
tokens_in=18 tokens_out=34   cache_hit=0 cache_miss=18 latency_ms=1122 cost_usd=0.000023

--- Q5（97 字，要求详细解释、不少于 300 字）---
tokens_in=62 tokens_out=968  cache_hit=0 cache_miss=62 latency_ms=5842 cost_usd=0.000590
```

**观察到的趋势，和最初设想的不完全一样**：`tokens_in` 确实随问题字数增长（18 → 24 → 41 → 62 → 91），但延迟和成本跟 `tokens_in` 的相关性很弱——真正主导延迟和成本的是 `tokens_out`。Q3 的输入只有 91 token（在 5 个问题里排第二），但因为问题本身要求"设计一个完整报价流程",模型写了 1610 个输出 token，延迟飙到 8820ms，是全场最慢、最贵的一次；Q4 输入最短（18 token），但因为"你好"这种问题模型也倾向于多说几句，输出反而不是最少的。

这条发现有实际意义：**这个场景下，控制成本和延迟的关键不是"用户问题写得多长"，是"模型这次打算写多长的回答"**——后面如果要做延迟/成本优化，`max_tokens` 这类限制输出长度的参数，或者在 `system` 里明确要求"回答要简洁"，可能比"想办法缩短用户输入"更直接有效。这一步先不动手做这类优化（那是后面 Phase 的事），只是把这条从真实数据里看出来的方向如实记下来。

**`cache_hit_tokens` 在 5 次请求里全部是 0**——即使 5 次请求共享同一个 `DEFAULT_SYSTEM`（理论上这段固定前缀有机会被缓存命中），也没有观察到任何一次命中。这条现象没有在这一步深入解释（缓存机制本身怎么工作、多短的前缀不会被缓存、命中窗口多长，都还不清楚），记进了 `TODO.md`，留到 Phase 6/7（涉及 Context 和 Prompt Cache 的两章）专门处理。

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
tests/test_llm.py::test_chat_returns_the_model_text PASSED
tests/test_llm.py::test_chat_sends_system_and_user_messages PASSED
tests/test_llm.py::test_chat_accepts_custom_system_and_temperature PASSED
tests/test_llm.py::test_get_client_reads_api_key_from_env PASSED
tests/test_llm.py::TestChatResult::test_token_counts_come_from_response_usage PASSED
tests/test_llm.py::TestChatResult::test_cache_hit_and_miss_tokens_are_reported_separately PASSED
tests/test_llm.py::TestChatResult::test_latency_is_measured_and_non_negative PASSED
tests/test_llm.py::TestChatResult::test_estimated_cost_matches_the_pricing_table PASSED
tests/test_llm.py::TestFailureClassification::test_authentication_error_is_reclassified PASSED
tests/test_llm.py::TestFailureClassification::test_connection_error_is_reclassified_as_network_failure PASSED
tests/test_llm.py::TestFailureClassification::test_timeout_error_is_also_reclassified_as_network_failure PASSED
tests/test_llm.py::TestFailureClassification::test_bad_request_error_is_reclassified PASSED
tests/test_llm.py::TestFailureClassification::test_all_three_failures_share_a_common_base_class PASSED

13 passed in 0.50s
```

新增的 `TestChatResult` 四个测试，全部用 mock 覆盖：`tokens_in`/`tokens_out` 忠实读自 `response.usage`、cache 命中/未命中的 token 数分开正确报告、延迟是真实测量出来的非负数、成本估算严格按 `PRICE_PER_MILLION_TOKENS_USD` 的公式算（构造了一个每一项都恰好是 100 万 token 的场景，让期望值直接等于三档单价之和，避免测试自己重新实现一遍计算逻辑再去对比）。第 3 节那组真实趋势数据没有写进自动化测试——`tokens_out`/`latency_ms` 这些数字本身是模型这次决定怎么回答、服务端这次跑得多快决定的，不是我们代码的逻辑，不能用断言去锁死一个我们控制不了的结果。

## 7. Exercises

1. 把 Q3（要求设计报价流程的长问题）在 `system` 里加一句"回答控制在 100 字以内"，重新跑一次，观察 `tokens_out`/`latency_ms`/`cost_usd` 分别变化了多少——这是验证第 3 节"输出长度主导成本"这条发现的直接办法。
2. 连续用完全相同的问题（比如反复问 5 次同一句"A100 还有多少库存？"）观察 `cache_hit_tokens` 会不会在某一次开始变成非 0——如果依然全是 0，说明这个场景下的缓存命中门槛比"重复同一个 system"更苛刻，值得记进 `TODO.md` 继续深挖。
3. `estimated_cost_usd` 目前完全没有考虑 DeepSeek 定价页提到的"高峰时段价格翻倍"（UTC 01:00-04:00、06:00-10:00，工作日）。想一想：要在现在的代码里加上这个判断，需要新增什么信息（提示：调用发生的具体时间），这个改动应该加在 `chat()` 内部还是 `_estimate_cost_usd()` 里？

## 8. What We Learned

`ChatResult` 这个结构本身很简单，但它体现了这本教材反复出现的一个态度：**衡量一件事的机制，要在第一次真正用到它之前就搭好**，不是等到"看起来快撑不住了"才回头补——Phase 14 的 Observability 不是凭空开始收集数据，是把 Commit 0.3 这里已经产出的单次数据聚合起来。

第 3 节最有价值的发现不是"token 数会变"这种预期之中的结果，而是**成本和延迟的主导因素是输出长度，不是输入长度**——这和很多人凭直觉认为的"问题问得越长、越复杂，就越贵"不完全一致，纠正了一个直觉误区。`cache_hit_tokens` 全程为 0 这条没能立刻解释的现象，被诚实地记进了 `TODO.md` 而不是含糊带过——这本教材的态度是"暂时解释不了的真实现象，先记下来，不假装看懂了"。
