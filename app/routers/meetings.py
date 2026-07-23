"""③ 會議 CRUD 端點。

多租戶:開發期用 `X-User-Id` header 或 config.DEFAULT_USER;auth(⑦)完成後改由 token 解出。
路徑與回傳形狀依 App HANDOFF 契約。
"""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Response
from pydantic import BaseModel

from app import config, db

router = APIRouter(prefix="/meetings", tags=["meetings"])


def _uid(x_user_id: str | None) -> str:
    return x_user_id or config.DEFAULT_USER


class CreateMeetingReq(BaseModel):
    title: str | None = None


@router.get("")
async def list_meetings(x_user_id: str | None = Header(default=None)):
    return {"items": await db.list_meetings(_uid(x_user_id))}


@router.post("")
async def create_meeting(req: CreateMeetingReq, x_user_id: str | None = Header(default=None)):
    return await db.create_meeting(_uid(x_user_id), req.title)


@router.get("/{mid}")
async def get_meeting(mid: str, x_user_id: str | None = Header(default=None)):
    m = await db.get_meeting(_uid(x_user_id), mid)
    if m is None:
        raise HTTPException(404, "meeting not found")
    return m


@router.delete("/{mid}")
async def delete_meeting(mid: str, x_user_id: str | None = Header(default=None)):
    await db.delete_meeting(_uid(x_user_id), mid)
    return Response(status_code=204)


@router.get("/{mid}/transcript")
async def get_transcript(mid: str, x_user_id: str | None = Header(default=None)):
    segs = await db.get_transcript(_uid(x_user_id), mid)
    if segs is None:
        raise HTTPException(404, "meeting not found")
    return {"segments": segs}


@router.get("/{mid}/summary")
async def get_summary(mid: str, x_user_id: str | None = Header(default=None)):
    data = await db.get_summary(_uid(x_user_id), mid)
    if data is None:
        raise HTTPException(404, "summary not found")   # ④ 尚未產生 → 404(依契約)
    return data
