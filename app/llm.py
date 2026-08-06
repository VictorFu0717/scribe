"""共用對話 LLM 存取層 — 支援 vLLM 與 Ollama 兩種後端,thinking 可控。

為何要分後端(實測 2026-08):
  「關 thinking」兩邊機制完全不同,且不可互通——
  - vLLM  : /v1(OpenAI 相容)+ extra_body.chat_template_kwargs.enable_thinking=False。
  - Ollama: 只有原生 /api/chat 的 think=False 有效;/v1 端點忽略 chat_template_kwargs、
            也忽略 think,`/no_think` 軟開關也關不乾淨 → qwen3 每次照樣想 2000+ 字。
後端由 config.CHAT_BACKEND 決定(auto 依 URL:11434→ollama)。thinking 預設關(config.CHAT_THINK)。

對外統一介面(呼叫端不必知道後端):
  chat_stream(messages, tools=None, think=None) → 逐一 yield 事件 dict:
      {"type":"content","text": "..."}             內容片段
      {"type":"tool_calls","calls":[{id,name,arguments(str)}...]}   (該回合最後,若有工具呼叫)
  stream(messages) / once(messages)                純內容(摘要/翻譯/QA 用)
  assistant_tool_msg(content, calls) / tool_result_msg(call, result)
      → 依後端組出「工具往返」要塞回 messages 的訊息(格式兩邊不同)
"""

from __future__ import annotations

import json

import httpx
from openai import AsyncOpenAI

from app import config


def _detect_backend() -> str:
    b = (config.CHAT_BACKEND or "auto").lower()
    if b in ("ollama", "vllm"):
        return b
    u = (config.CHAT_BASE_URL or "").lower()
    return "ollama" if (":11434" in u or "ollama" in u) else "vllm"


BACKEND = _detect_backend()

# vLLM 走 OpenAI client;保留 client 名稱以相容既有引用
_openai = AsyncOpenAI(base_url=config.CHAT_BASE_URL, api_key=config.CHAT_API_KEY)
client = _openai

# Ollama 原生 base:把結尾的 /v1 去掉(config 給的是 …:11434/v1)
_ollama_base = (config.CHAT_BASE_URL or "").rstrip("/")
if _ollama_base.endswith("/v1"):
    _ollama_base = _ollama_base[:-3]


def _loads(s):
    try:
        return json.loads(s or "{}")
    except Exception:
        return {}


def _keep_alive():
    """Ollama keep_alive:數字(-1 永久 / 0 立即卸載 / 秒)給 int,時長字串(30m/1h)原樣。"""
    v = config.OLLAMA_KEEP_ALIVE
    try:
        return int(v)
    except (ValueError, TypeError):
        return v


# ---------------------------------------------------------------- 統一串流
async def chat_stream(messages, tools=None, think=None, temperature: float = 0.3):
    """統一串流介面,見模組 docstring。think=None → 用 config.CHAT_THINK(預設關)。"""
    if think is None:
        think = config.CHAT_THINK
    gen = _ollama_stream if BACKEND == "ollama" else _vllm_stream
    async for ev in gen(messages, tools, bool(think), temperature):
        yield ev


async def _vllm_stream(messages, tools, think, temperature):
    kw = dict(
        model=config.CHAT_MODEL, messages=messages, stream=True, temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": think}},
    )
    if tools:
        kw["tools"] = tools
    acc: dict = {}   # index -> {id,name,arguments}
    stream = await _openai.chat.completions.create(**kw)
    async for ch in stream:
        if not ch.choices:
            continue
        d = ch.choices[0].delta
        if getattr(d, "content", None):
            yield {"type": "content", "text": d.content}
        for tc in (getattr(d, "tool_calls", None) or []):
            e = acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                e["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    e["name"] += tc.function.name
                if tc.function.arguments:
                    e["arguments"] += tc.function.arguments
    if acc:
        yield {"type": "tool_calls", "calls": list(acc.values())}


async def _ollama_stream(messages, tools, think, temperature):
    payload = {
        "model": config.CHAT_MODEL, "messages": messages, "stream": True,
        "think": think, "options": {"temperature": temperature},
        "keep_alive": _keep_alive(),
    }
    if tools:
        payload["tools"] = tools
    calls: list = []
    async with httpx.AsyncClient(timeout=None) as hc:
        async with hc.stream("POST", f"{_ollama_base}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                msg = d.get("message") or {}
                if msg.get("content"):
                    yield {"type": "content", "text": msg["content"]}
                for tc in (msg.get("tool_calls") or []):    # ollama 的 tool_calls 一次給整包
                    fn = tc.get("function") or {}
                    args = fn.get("arguments", {})
                    calls.append({
                        "id": tc.get("id") or f"call_{len(calls)}",
                        "name": fn.get("name", ""),
                        "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
                    })
                if d.get("done"):
                    break
    if calls:
        yield {"type": "tool_calls", "calls": calls}


# ------------------------------------------------- 工具往返:組回填訊息(後端相異)
def assistant_tool_msg(content, calls):
    """assistant 決定呼叫工具的那則訊息。calls: [{id,name,arguments(str)}]。"""
    if BACKEND == "ollama":
        return {"role": "assistant", "content": content or "",
                "tool_calls": [{"function": {"name": c["name"], "arguments": _loads(c["arguments"])}}
                               for c in calls]}
    return {"role": "assistant", "content": content or None,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]}}
                           for c in calls]}


def tool_result_msg(call, result: str):
    """單一工具執行結果訊息。ollama 用 tool_name 對應,openai/vllm 用 tool_call_id。"""
    if BACKEND == "ollama":
        return {"role": "tool", "tool_name": call["name"], "content": result}
    return {"role": "tool", "tool_call_id": call["id"], "content": result}


# ---------------------------------------------------- 簡易介面(摘要/翻譯/QA:純內容)
async def stream(messages, temperature: float = 0.3):
    """串流 content deltas。"""
    async for ev in chat_stream(messages, tools=None, temperature=temperature):
        if ev["type"] == "content":
            yield ev["text"]


async def once(messages, temperature: float = 0.3) -> str:
    """一次取完整回覆(map-reduce 的 map 階段用)。"""
    out = ""
    async for ev in chat_stream(messages, tools=None, temperature=temperature):
        if ev["type"] == "content":
            out += ev["text"]
    return out
