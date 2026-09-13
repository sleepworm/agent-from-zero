import os
import sys

from openai import OpenAI

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_SYSTEM = "You are a helpful assistant for a small industrial equipment company."
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """惰性单例：只有真正调用 chat() 时才会去读 DEEPSEEK_API_KEY、建立客户端——
    这样 import 这个模块本身不需要环境变量存在，测试可以直接 monkeypatch 掉
    这个函数，绕过真实的网络客户端。"""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL)
    return _client


def chat(user_message: str, system: str = DEFAULT_SYSTEM, temperature: float = 0.0) -> str:
    """
    最基础的一次 Chat Completion，不涉及工具、不涉及历史。DeepSeek 走的是
    OpenAI 兼容协议：system 不是单独的参数，而是 messages 列表里的第一条记录。

    temperature=0.0 是这里的默认值——Agent 场景要的是"同样的输入尽量得到同样
    的行为"，不是聊天场景那种"每次都想要点新意"。
    """
    response = get_client().chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("You: ")
    print(chat(question))
