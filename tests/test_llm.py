from unittest.mock import MagicMock

import httpx2
import openai
import pytest

from src.agent import llm


def _fake_client(
    response_text: str,
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    cache_hit_tokens: int = 0,
    cache_miss_tokens: int = 10,
) -> MagicMock:
    fake_message = MagicMock()
    fake_message.content = response_text
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_usage = MagicMock()
    fake_usage.prompt_tokens = prompt_tokens
    fake_usage.completion_tokens = completion_tokens
    fake_usage.prompt_cache_hit_tokens = cache_hit_tokens
    fake_usage.prompt_cache_miss_tokens = cache_miss_tokens
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage = fake_usage
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response
    return fake_client


def _client_raising(error: Exception) -> MagicMock:
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = error
    return fake_client


def _fake_request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.deepseek.com/chat/completions")


def test_chat_returns_the_model_text(monkeypatch):
    fake_client = _fake_client("Hello, Alex!")
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    result = llm.chat("Say hello to Alex")

    assert result.text == "Hello, Alex!"


def test_chat_sends_system_and_user_messages(monkeypatch):
    fake_client = _fake_client("ok")
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    llm.chat("A100 还有多少库存？")

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"] == [
        {"role": "system", "content": llm.DEFAULT_SYSTEM},
        {"role": "user", "content": "A100 还有多少库存？"},
    ]
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["model"] == llm.DEFAULT_MODEL


def test_chat_accepts_custom_system_and_temperature(monkeypatch):
    fake_client = _fake_client("ok")
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    llm.chat("hi", system="custom system", temperature=0.7)

    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"][0] == {"role": "system", "content": "custom system"}
    assert call_kwargs["temperature"] == 0.7


def test_get_client_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123")
    llm._client = None  # 重置单例，避免其他测试留下的 mock 状态影响这条测试

    client = llm.get_client()

    assert client.api_key == "test-key-123"
    assert client.base_url is not None
    llm._client = None  # 用完清理，避免影响后续测试


class TestChatResult:
    """验证 chat() 返回的 ChatResult 里，token 计数、成本估算是不是真的
    忠实反映了 response.usage 里的数字，不是这一版自己编出来的。延迟本身
    是真实时钟量出来的，测试里只断言"非负"，不断言具体数值。"""

    def test_token_counts_come_from_response_usage(self, monkeypatch):
        fake_client = _fake_client("ok", prompt_tokens=123, completion_tokens=45)
        monkeypatch.setattr(llm, "get_client", lambda: fake_client)

        result = llm.chat("hi")

        assert result.tokens_in == 123
        assert result.tokens_out == 45

    def test_cache_hit_and_miss_tokens_are_reported_separately(self, monkeypatch):
        fake_client = _fake_client("ok", cache_hit_tokens=80, cache_miss_tokens=20)
        monkeypatch.setattr(llm, "get_client", lambda: fake_client)

        result = llm.chat("hi")

        assert result.cache_hit_tokens == 80
        assert result.cache_miss_tokens == 20

    def test_latency_is_measured_and_non_negative(self, monkeypatch):
        fake_client = _fake_client("ok")
        monkeypatch.setattr(llm, "get_client", lambda: fake_client)

        result = llm.chat("hi")

        assert result.latency_ms >= 0

    def test_estimated_cost_matches_the_pricing_table(self, monkeypatch):
        fake_client = _fake_client(
            "ok", cache_hit_tokens=1_000_000, cache_miss_tokens=1_000_000, completion_tokens=1_000_000
        )
        monkeypatch.setattr(llm, "get_client", lambda: fake_client)

        result = llm.chat("hi")

        expected = (
            llm.PRICE_PER_MILLION_TOKENS_USD["input_cache_hit"]
            + llm.PRICE_PER_MILLION_TOKENS_USD["input_cache_miss"]
            + llm.PRICE_PER_MILLION_TOKENS_USD["output"]
        )
        assert result.estimated_cost_usd == pytest.approx(expected)


class TestFailureClassification:
    """
    三种失败模式都在 docs/00-02-error-handling.md 的 Break It 一节里真实触发过、
    观察过原始异常类型和信息，这里只验证"分类逻辑本身对不对"：给它一个真实构造
    出来的 openai 异常，看 chat() 有没有把它转换成我们自己的类型，同时保留住
    原始的错误信息。不在测试里重新触发真实的网络故障——那样会让测试变慢、
    变得不确定，真实环境已经在 Break It 一节验证过现象是真的。
    """

    def test_authentication_error_is_reclassified(self, monkeypatch):
        response = httpx2.Response(401, request=_fake_request())
        original = openai.AuthenticationError(
            "Authentication Fails, Your api key: ****0000 is invalid",
            response=response,
            body=None,
        )
        monkeypatch.setattr(llm, "get_client", lambda: _client_raising(original))

        with pytest.raises(llm.AuthenticationFailure) as exc_info:
            llm.chat("hi")

        assert "Authentication Fails" in str(exc_info.value)
        assert not isinstance(exc_info.value, (llm.NetworkFailure, llm.BadRequestFailure))

    def test_connection_error_is_reclassified_as_network_failure(self, monkeypatch):
        original = openai.APIConnectionError(message="Connection error.", request=_fake_request())
        monkeypatch.setattr(llm, "get_client", lambda: _client_raising(original))

        with pytest.raises(llm.NetworkFailure) as exc_info:
            llm.chat("hi")

        assert "Connection error" in str(exc_info.value)

    def test_timeout_error_is_also_reclassified_as_network_failure(self, monkeypatch):
        # openai.APITimeoutError 是 openai.APIConnectionError 的子类——见
        # docs/00-02-error-handling.md，断网和超时对调用方来说是同一类问题
        # （"这次没打通"），所以一个 except 分支能同时覆盖两种情况。
        original = openai.APITimeoutError(request=_fake_request())
        monkeypatch.setattr(llm, "get_client", lambda: _client_raising(original))

        with pytest.raises(llm.NetworkFailure):
            llm.chat("hi")

    def test_bad_request_error_is_reclassified(self, monkeypatch):
        response = httpx2.Response(400, request=_fake_request())
        original = openai.BadRequestError(
            "The supported API model names are deepseek-flash, deepseek-v4-pro, "
            "but you passed deepseek-chat-nonexistent-model.",
            response=response,
            body=None,
        )
        monkeypatch.setattr(llm, "get_client", lambda: _client_raising(original))

        with pytest.raises(llm.BadRequestFailure) as exc_info:
            llm.chat("hi")

        assert "supported API model names" in str(exc_info.value)

    def test_all_three_failures_share_a_common_base_class(self):
        assert issubclass(llm.AuthenticationFailure, llm.ChatError)
        assert issubclass(llm.NetworkFailure, llm.ChatError)
        assert issubclass(llm.BadRequestFailure, llm.ChatError)
