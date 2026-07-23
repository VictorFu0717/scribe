"""④ 會議摘要 — POST /meetings/{id}/summarize (SSE)。

流程:取該場逐字稿 → chat LLM 產生固定格式 Markdown(串流 delta) → 串完解析成結構化 JSON、
存入 DB、再送出 JSON,最後 [DONE]。長逐字稿用 map-reduce(先分段濃縮,再合併)。

SSE:
    data: {"delta":"..."}   ...(邊產邊顯示)
    data: {"overview":..,"key_points":[..],"decisions":[..],
           "action_items":[{"task","owner?","due?"}],"follow_ups":[..]}
    data: [DONE]
"""

from __future__ import annotations

import json
import os
import re

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import config, db, llm
from app.auth import get_current_user

router = APIRouter(tags=["summary"])

MAP_MAX_CHARS = int(os.getenv("SUMMARY_MAP_CHARS", "6000"))   # 逐字稿超過就 map-reduce

_FORMAT = """## 會議摘要
（2-4 句話總結整場會議）

## 討論重點
- （重點,一行一項）

## 決議事項
- （做成的決議;若無寫「無」）

## 待辦事項
- 任務描述 ｜ 負責人：姓名 ｜ 期限：時間
- （沒有負責人或期限就省略該欄;若無待辦寫「無」）

## 後續追蹤
- （需追蹤事項;若無寫「無」）"""

_SYS = ("你是專業的會議記錄助理。根據會議逐字稿,產生繁體中文(台灣用語)的結構化會議紀錄。\n"
        "嚴格使用以下 Markdown 格式,標題文字與順序不可更動,只輸出內容不要開場白或結語:\n\n"
        + _FORMAT)

_MAP_SYS = ("把以下會議逐字稿片段濃縮成繁體中文重點筆記(條列),"
            "保留決議、待辦(含負責人/期限)、關鍵數字與結論。只輸出重點,不要開場白。")


def _split(text: str, n: int) -> list[str]:
    """依行切塊,盡量湊到 n 字元。"""
    chunks, cur, size = [], [], 0
    for ln in text.splitlines():
        if size + len(ln) > n and cur:
            chunks.append("\n".join(cur)); cur, size = [], 0
        cur.append(ln); size += len(ln) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _strip_think(md: str) -> str:
    return re.sub(r"<think>.*?</think>", "", md, flags=re.DOTALL).strip()


_EMPTY = {"", "無", "（無）", "(無)", "無。", "none", "N/A", "未指定"}


def _bullets(lines: list[str]) -> list[str]:
    out = []
    for l in lines:
        m = re.match(r"^\s*[-*•]\s*(.+)$", l)
        if m:
            v = m.group(1).strip()
            if v and v not in _EMPTY:
                out.append(v)
    return out


def _text(lines: list[str]) -> str:
    return "\n".join(l.strip() for l in lines if l.strip()).strip()


def _parse_action(a: str) -> dict:
    parts = [p.strip() for p in re.split(r"[｜|]", a)]
    item = {"task": parts[0]}
    for p in parts[1:]:
        m = re.match(r"(?:負責人|owner)\s*[:：]\s*(.+)", p)
        if m and m.group(1).strip() not in _EMPTY:
            item["owner"] = m.group(1).strip()
        m = re.match(r"(?:期限|due|時間)\s*[:：]\s*(.+)", p)
        if m and m.group(1).strip() not in _EMPTY:
            item["due"] = m.group(1).strip()
    return item


def parse_markdown(md: str) -> dict:
    """把固定格式 Markdown 解析成結構化 JSON;解析不到就給空欄(不會壞)。"""
    md = _strip_think(md)
    sections: dict[str, list[str]] = {}
    cur = None
    for line in md.splitlines():
        h = re.match(r"^#{1,6}\s*(.+?)\s*$", line)
        if h:
            cur = h.group(1)
            sections[cur] = []
        elif cur is not None:
            sections[cur].append(line)

    def find(*keys):
        for title, lines in sections.items():
            if any(k in title for k in keys):
                return lines
        return []

    return {
        "overview": _text(find("摘要", "概要", "overview")),
        "key_points": _bullets(find("重點", "討論")),
        "decisions": _bullets(find("決議", "決定")),
        "action_items": [_parse_action(a) for a in _bullets(find("待辦", "行動", "action"))],
        "follow_ups": _bullets(find("後續", "追蹤", "follow")),
    }


class SummarizeReq(BaseModel):
    language: str | None = "zh-Hant"


def _sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@router.post("/meetings/{mid}/summarize")
async def summarize(mid: str,
                    req: SummarizeReq | None = Body(default=None),
                    user: str = Depends(get_current_user)):
    if await db.get_meeting(user, mid) is None:
        raise HTTPException(404, "meeting not found")
    transcript = await db.get_transcript_text(mid)
    if not transcript.strip():
        raise HTTPException(400, "transcript is empty")

    async def gen():
        md = ""
        try:
            if len(transcript) <= MAP_MAX_CHARS:
                async for piece in llm.stream([
                    {"role": "system", "content": _SYS},
                    {"role": "user", "content": transcript},
                ]):
                    md += piece
                    yield _sse({"delta": piece})
            else:
                # map:各段濃縮成筆記
                notes = []
                for chunk in _split(transcript, MAP_MAX_CHARS):
                    notes.append(await llm.once([
                        {"role": "system", "content": _MAP_SYS},
                        {"role": "user", "content": chunk},
                    ]))
                combined = "\n\n".join(notes)
                # reduce:合併成最終紀錄(串流)
                async for piece in llm.stream([
                    {"role": "system", "content": _SYS},
                    {"role": "user", "content": "以下是各段會議重點筆記,請合併成一份完整會議紀錄:\n\n" + combined},
                ]):
                    md += piece
                    yield _sse({"delta": piece})

            data = parse_markdown(md)
            await db.save_summary(mid, data)
            yield _sse(data)
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield _sse({"error": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")
