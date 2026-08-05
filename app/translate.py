"""留檔翻譯 — POST /meetings/{id}/translate (SSE)。

把該場逐字稿用 chat LLM 翻成目標語言,串流回傳並存檔(之後 GET .../translation 可取)。
保留每行「說話者N：」標籤與換行(一行對一行)。長逐字稿分段翻(依行切,順序拼回)。

即時雙語字幕走 app 端裝置內翻譯;此端點是「會後留檔的高品質翻譯」。
"""

from __future__ import annotations

import json
import os

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import config, db, llm
from app.auth import get_current_user

router = APIRouter(tags=["translate"])

MAP_CHARS = int(os.getenv("TRANSLATE_MAP_CHARS", "3000"))   # 超過就分段翻

# 常見語言代碼 → 中文名(丟給 prompt);沒對到就原樣傳(LLM 也看得懂 en/English/日本語)
_LANG = {
    "en": "英文", "en-us": "英文", "ja": "日文", "jp": "日文", "ko": "韓文",
    "zh": "中文", "zh-hant": "繁體中文", "zh-hans": "簡體中文",
    "vi": "越南文", "th": "泰文", "id": "印尼文", "ms": "馬來文",
    "es": "西班牙文", "fr": "法文", "de": "德文", "ru": "俄文",
}


def _lang_name(t: str) -> str:
    return _LANG.get((t or "").strip().lower(), (t or "").strip() or "英文")


def _sys(target_name: str) -> str:
    return (f"你是專業翻譯。把以下會議逐字稿翻譯成{target_name}。\n"
            f"規則:保留每行開頭的「說話者N：」標籤與逐行換行結構(一行對一行);"
            f"忠實翻譯、不加註解、不輸出原文、不要開場白。")


def _split(text: str, n: int) -> list[str]:
    chunks, cur, size = [], [], 0
    for ln in text.splitlines():
        if size + len(ln) > n and cur:
            chunks.append("\n".join(cur)); cur, size = [], 0
        cur.append(ln); size += len(ln) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


class TranslateReq(BaseModel):
    target: str = "en"


def _sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@router.post("/meetings/{mid}/translate")
async def translate(mid: str,
                    req: TranslateReq | None = Body(default=None),
                    user: str = Depends(get_current_user)):
    if await db.get_meeting(user, mid) is None:
        raise HTTPException(404, "meeting not found")
    transcript = await db.get_transcript_text(mid)
    if not transcript.strip():
        raise HTTPException(400, "transcript is empty")

    target = (req.target if req else None) or "en"
    target_name = _lang_name(target)

    async def gen():
        full = ""
        try:
            chunks = _split(transcript, MAP_CHARS) if len(transcript) > MAP_CHARS else [transcript]
            for i, chunk in enumerate(chunks):
                async for piece in llm.stream([
                    {"role": "system", "content": _sys(target_name)},
                    {"role": "user", "content": chunk},
                ]):
                    full += piece
                    yield _sse({"delta": piece})
                if i < len(chunks) - 1:
                    full += "\n"
                    yield _sse({"delta": "\n"})
            await db.save_translation(mid, target, full.strip())
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield _sse({"error": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/meetings/{mid}/translation")
async def get_translation(mid: str, target: str = "en",
                          user: str = Depends(get_current_user)):
    text = await db.get_translation(user, mid, target)
    if text is None:
        raise HTTPException(404, "translation not found")
    return {"target": target, "text": text}
