"""單場會議 QA (方案 ③,舊端點) — 把逐字稿當 context,由 chat LLM 回答。

保留給相容;新的 agentic 助理端點見 app/routers/assistant.py(⑤⑥,待實作)。
逐字稿來源:stateless(client 帶 transcript)。SSE 串流、繁體。
"""

from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import llm

router = APIRouter()

SYSTEM_PROMPT = """你是專業的會議記錄助理。下面是一場會議的逐字稿。
請「只根據逐字稿內容」回答使用者的問題,並遵守:
- 一律使用繁體中文(台灣用語)回答。
- 只依據逐字稿,不要臆測或編造;若逐字稿中找不到相關內容,明確說「逐字稿中沒有提到相關資訊」。
- 回答精簡、切中重點,需要時用條列。

【會議逐字稿】
{transcript}"""


class ChatTurn(BaseModel):
    role: str
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
    messages = _build_messages(req)

    async def gen():
        try:
            async for piece in llm.stream(messages, temperature=0.3):   # thinking 由後端統一控(預設關)
                yield f"data: {json.dumps({'delta': piece}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
