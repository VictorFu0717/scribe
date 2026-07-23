"""③ 會議 CRUD 端點。

身分由 ⑦ auth 的 get_current_user 依賴解出(Bearer token;開發期沒帶則退回 X-User-Id/DEFAULT_USER)。
路徑與回傳形狀依 App HANDOFF 契約。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from app import db
from app.auth import get_current_user

router = APIRouter(prefix="/meetings", tags=["meetings"])


class CreateMeetingReq(BaseModel):
    title: str | None = None


@router.get("")
async def list_meetings(user: str = Depends(get_current_user)):
    return {"items": await db.list_meetings(user)}


@router.post("")
async def create_meeting(req: CreateMeetingReq, user: str = Depends(get_current_user)):
    return await db.create_meeting(user, req.title)


@router.get("/{mid}")
async def get_meeting(mid: str, user: str = Depends(get_current_user)):
    m = await db.get_meeting(user, mid)
    if m is None:
        raise HTTPException(404, "meeting not found")
    return m


@router.delete("/{mid}")
async def delete_meeting(mid: str, user: str = Depends(get_current_user)):
    await db.delete_meeting(user, mid)
    return Response(status_code=204)


@router.get("/{mid}/transcript")
async def get_transcript(mid: str, user: str = Depends(get_current_user)):
    segs = await db.get_transcript(user, mid)
    if segs is None:
        raise HTTPException(404, "meeting not found")
    return {"segments": segs}


@router.get("/{mid}/summary")
async def get_summary(mid: str, user: str = Depends(get_current_user)):
    data = await db.get_summary(user, mid)
    if data is None:
        raise HTTPException(404, "summary not found")   # ④ 尚未產生 → 404(依契約)
    return data
