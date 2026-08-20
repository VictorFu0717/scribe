"""③ 會議 CRUD 端點。

身分由 ⑦ auth 的 get_current_user 依賴解出(Bearer token;開發期沒帶則退回 X-User-Id/DEFAULT_USER)。
路徑與回傳形狀依 App HANDOFF 契約。
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from pydantic import BaseModel

from app import db, rag
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


@router.post("/reindex")
async def reindex_all(background_tasks: BackgroundTasks, user: str = Depends(get_current_user)):
    """把這個使用者的所有會議重建向量索引(背景執行)。

    什麼時候需要:embedding 服務曾經掛掉(那幾場只建了關鍵字索引、純向量模式搜不到)、
    ⑥ RAG 之前就存在的舊會議、或換了 embedding 模型。index_meeting 本身冪等,重跑安全。
    """
    ms = await db.list_meetings(user)
    ids = [m["id"] for m in ms]

    async def _run():
        ok = fail = 0
        for mid in ids:
            try:
                await rag.index_meeting(user, mid)
                ok += 1
            except Exception as e:
                fail += 1
                print(f"[reindex] {mid} 失敗: {e}")
        print(f"[reindex] {user}: {ok} 成功 / {fail} 失敗")

    background_tasks.add_task(_run)
    return {"meetings": len(ids), "status": "reindexing"}


@router.post("/{mid}/reindex")
async def reindex_one(mid: str, user: str = Depends(get_current_user)):
    """重建單一會議的向量索引(同步,通常一兩秒)。"""
    if await db.get_meeting(user, mid) is None:
        raise HTTPException(404, "meeting not found")
    try:
        await rag.index_meeting(user, mid)
    except Exception as e:
        raise HTTPException(503, f"索引失敗(embedding 服務可用嗎?): {e}")
    return {"id": mid, "status": "indexed"}
