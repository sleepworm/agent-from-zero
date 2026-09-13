import os
import sys

from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
)

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_SYSTEM = "You are a helpful assistant for a small industrial equipment company."
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_client: OpenAI | None = None


class ChatError(Exception):
    """chat() 所有已分类失败的公共基类。调用方既可以精确 catch 某一个子类，
    也可以只认这一个基类，一网打尽——这一层存在的意义是让后面的 Agent Loop
    /Retry 逻辑不需要认识 openai SDK 自己的异常体系，只需要认识这三种。"""


class AuthenticationFailure(ChatError):
    """API Key 缺失或错误——重试没有意义，必须先修好凭证。"""


class NetworkFailure(ChatError):
    """连不上或超时——本质都是"这次没打通"，见 chat() 里对
    APIConnectionError 的说明，超时被归进了同一类。"""


class BadRequestFailure(ChatError):
    """请求本身不合法（模型名不存在、参数超范围、prompt 太长等）——
    重试没有意义，必须先改请求内容。"""


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

    三种失败模式被重新分类成我们自己的异常类型，而不是让调用方直接接住
    openai SDK 原始的异常：AuthenticationFailure（凭证问题）、NetworkFailure
    （连接失败或超时——openai.APITimeoutError 本身就是 APIConnectionError
    的子类，一个 except 分支同时覆盖两种情况）、BadRequestFailure（请求本身
    不合法）。
    """
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


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("You: ")
    print(chat(question))
