from unittest.mock import MagicMock

from src.agent import llm


def _fake_client(response_text: str) -> MagicMock:
    fake_message = MagicMock()
    fake_message.content = response_text
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response
    return fake_client


def test_chat_returns_the_model_text(monkeypatch):
    fake_client = _fake_client("Hello, Alex!")
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    result = llm.chat("Say hello to Alex")

    assert result == "Hello, Alex!"


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
