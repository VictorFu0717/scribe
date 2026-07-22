"""單場會議 QA (方案 ③) — 把該場「定稿逐字稿」當 context,由 chat LLM 回答使用者問題。

chat LLM = 使用者自架的 vLLM (Qwen3.6-27B, OpenAI 相容),預設 http://localhost:8004/v1。
逐字稿來源:stateless —— 由 client 每次把逐字稿帶進來(之後接會議儲存再改成用 meeting_id 取)。
回答:SSE 串流、繁體中文(逐字稿本身已是繁體,加上 prompt 要求 → 模型自然輸出繁體)。

端點:
    POST /meeting/chat   (text/event-stream)
    body: {"transcript": "...", "question": "...", "history": [{"role","content"}...]}
    回傳: data: {"delta": "..."}\n\n  ...  最後 data: [DONE]\n\n
"""

from __future__ import annotations

import json
import os

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "http://localhost:8004/v1")
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "EMPTY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen3.6-27B")
# Qwen3 系列有 thinking 模式;會議 QA 求快求穩,關掉 thinking
CHAT_ENABLE_THINKING = os.getenv("CHAT_ENABLE_THINKING", "0") in ("1", "true", "True")

router = APIRouter()
_chat = AsyncOpenAI(base_url=CHAT_BASE_URL, api_key=CHAT_API_KEY)

SYSTEM_PROMPT = """你是專業的會議記錄助理。下面是一場會議的逐字稿。
請「只根據逐字稿內容」回答使用者的問題,並遵守:
- 一律使用繁體中文(台灣用語)回答。
- 只依據逐字稿,不要臆測或編造;若逐字稿中找不到相關內容,明確說「逐字稿中沒有提到相關資訊」。
- 回答精簡、切中重點,需要時用條列。

【會議逐字稿】
{transcript}"""


class ChatTurn(BaseModel):
    role: str          # "user" | "assistant"
    content: str


class MeetingChatReq(BaseModel):
    transcript: str
    question: str
    history: list[ChatTurn] = []


def _build_messages(req: MeetingChatReq) -> list[dict]:
    msgs: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(transcript=req.transcript)}
    ]
    for t in req.history:
        msgs.append({"role": t.role, "content": t.content})
    msgs.append({"role": "user", "content": req.question})
    return msgs


@router.post("/meeting/chat")
async def meeting_chat(req: MeetingChatReq):
    """依該場逐字稿回答問題,SSE 串流。"""
    messages = _build_messages(req)

    async def gen():
        try:
            stream = await _chat.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                stream=True,
                temperature=0.3,
                extra_body={"chat_template_kwargs": {"enable_thinking": CHAT_ENABLE_THINKING}},
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                piece = getattr(chunk.choices[0].delta, "content", None)
                if piece:
                    yield f"data: {json.dumps({'delta': piece}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
