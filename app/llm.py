"""共用對話 LLM client(摘要 ④ / 助理 ⑤⑥ 用)。

指向 config.CHAT_*(vLLM Qwen3.6 或 Ollama qwen3.6,OpenAI 相容)。
chat_template_kwargs 是 vLLM 專屬關 thinking;Ollama 會忽略(不影響)。
"""

from __future__ import annotations

from openai import AsyncOpenAI

from app import config

client = AsyncOpenAI(base_url=config.CHAT_BASE_URL, api_key=config.CHAT_API_KEY)

_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}


async def stream(messages: list[dict], temperature: float = 0.3):
    """串流 content deltas。"""
    s = await client.chat.completions.create(
        model=config.CHAT_MODEL, messages=messages, stream=True,
        temperature=temperature, extra_body=_EXTRA)
    async for ch in s:
        if ch.choices and ch.choices[0].delta.content:
            yield ch.choices[0].delta.content


async def once(messages: list[dict], temperature: float = 0.3) -> str:
    """一次取完整回覆(map-reduce 的 map 階段用)。"""
    r = await client.chat.completions.create(
        model=config.CHAT_MODEL, messages=messages,
        temperature=temperature, extra_body=_EXTRA)
    return r.choices[0].message.content or ""
