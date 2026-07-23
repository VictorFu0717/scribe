"""SQLite 儲存層(aiosqlite)。

三張表,皆掛 user_id + meeting_id(多租戶,RAG 檢索靠 user_id 隔離):
  meetings             會議 metadata
  transcript_segments  逐字稿片段
  summaries            結構化摘要(④ 用)
之後接 RAG(⑥)時同一個 SQLite 檔用 sqlite-vec 加向量表即可,不需搬遷。
"""

from __future__ import annotations

import json
import struct
import uuid
from datetime import datetime, timezone

import aiosqlite
import sqlite_vec

from app import config


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _pack(vec) -> bytes:
    return struct.pack("%df" % len(vec), *vec)


async def _connect_vec() -> aiosqlite.Connection:
    """開一個已載入 sqlite-vec 擴充的連線(呼叫端負責 close)。"""
    conn = await aiosqlite.connect(config.DB_PATH)
    conn.row_factory = aiosqlite.Row
    await conn.enable_load_extension(True)
    await conn.load_extension(sqlite_vec.loadable_path())
    await conn.enable_load_extension(False)
    return conn


async def init_db():
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meetings(
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT,
                created_at TEXT, duration_sec INTEGER DEFAULT 0,
                status TEXT DEFAULT 'recording', has_summary INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS transcript_segments(
                id INTEGER PRIMARY KEY AUTOINCREMENT, meeting_id TEXT NOT NULL,
                seq INTEGER, text TEXT, speaker TEXT, start_ms INTEGER, end_ms INTEGER
            );
            CREATE TABLE IF NOT EXISTS summaries(
                meeting_id TEXT PRIMARY KEY, data TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS users(
                id TEXT PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT, created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_meetings_user ON meetings(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_seg_meeting ON transcript_segments(meeting_id, seq);
            CREATE TABLE IF NOT EXISTS chunks(
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, meeting_id TEXT,
                seq INTEGER, text TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_meeting ON chunks(meeting_id);
            """
        )
        await db.commit()
        # ⑥ RAG:sqlite-vec 向量表(user_id 分區,rowid = chunks.id)
        await db.enable_load_extension(True)
        await db.load_extension(sqlite_vec.loadable_path())
        await db.enable_load_extension(False)
        await db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"user_id text partition key, embedding float[{config.EMBED_DIM}])")
        await db.commit()
    print(f"[db] ready: {config.DB_PATH}")


def _meeting_row(r) -> dict:
    return {
        "id": r["id"], "title": r["title"], "created_at": r["created_at"],
        "duration_sec": r["duration_sec"], "status": r["status"],
        "has_summary": bool(r["has_summary"]), "audio_url": None,
    }


# ---- users (⑦ auth;儲存層只存,雜湊在 app/auth.py) ----
async def create_user(user_id: str, username: str, password_hash: str) -> dict:
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(
            "INSERT INTO users(id,username,password_hash,created_at) VALUES(?,?,?,?)",
            (user_id, username, password_hash, _now()))
        await db.commit()
    return {"id": user_id, "username": username}


async def get_user_by_username(username: str) -> dict | None:
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM users WHERE username=?", (username,))
        r = await cur.fetchone()
        return dict(r) if r else None


async def get_user_by_id(user_id: str) -> dict | None:
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT id,username,created_at FROM users WHERE id=?", (user_id,))
        r = await cur.fetchone()
        return dict(r) if r else None


def new_user_id() -> str:
    return _new_id()


# ---- meetings ----
async def create_meeting(user_id: str, title: str | None) -> dict:
    mid = _new_id()
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(
            "INSERT INTO meetings(id,user_id,title,created_at) VALUES(?,?,?,?)",
            (mid, user_id, title or "未命名會議", _now()))
        await db.commit()
    return await get_meeting(user_id, mid)


async def list_meetings(user_id: str) -> list[dict]:
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM meetings WHERE user_id=? ORDER BY created_at DESC", (user_id,))
        return [_meeting_row(r) for r in await cur.fetchall()]


async def get_meeting(user_id: str, mid: str) -> dict | None:
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM meetings WHERE user_id=? AND id=?", (user_id, mid))
        r = await cur.fetchone()
        return _meeting_row(r) if r else None


async def delete_meeting(user_id: str, mid: str) -> bool:
    async with aiosqlite.connect(config.DB_PATH) as db:
        cur = await db.execute("DELETE FROM meetings WHERE user_id=? AND id=?", (user_id, mid))
        await db.execute("DELETE FROM transcript_segments WHERE meeting_id=?", (mid,))
        await db.execute("DELETE FROM summaries WHERE meeting_id=?", (mid,))
        await db.commit()
        deleted = cur.rowcount > 0
    await delete_chunks(mid)   # 連帶刪向量索引
    return deleted


# ---- 向量索引 (⑥ RAG, sqlite-vec) ----
async def store_chunks(user_id: str, meeting_id: str, chunks: list[dict]):
    """先刪該會議舊塊,再存入新塊(chunks:{seq,text,embedding})。vec rowid = chunks.id。"""
    await delete_chunks(meeting_id)
    conn = await _connect_vec()
    try:
        for ch in chunks:
            cur = await conn.execute(
                "INSERT INTO chunks(user_id,meeting_id,seq,text) VALUES(?,?,?,?)",
                (user_id, meeting_id, ch["seq"], ch["text"]))
            cid = cur.lastrowid
            await conn.execute(
                "INSERT INTO vec_chunks(rowid,user_id,embedding) VALUES(?,?,?)",
                (cid, user_id, _pack(ch["embedding"])))
        await conn.commit()
    finally:
        await conn.close()


async def delete_chunks(meeting_id: str):
    conn = await _connect_vec()
    try:
        cur = await conn.execute("SELECT id FROM chunks WHERE meeting_id=?", (meeting_id,))
        ids = [r[0] for r in await cur.fetchall()]
        for cid in ids:
            await conn.execute("DELETE FROM vec_chunks WHERE rowid=?", (cid,))
        await conn.execute("DELETE FROM chunks WHERE meeting_id=?", (meeting_id,))
        await conn.commit()
    finally:
        await conn.close()


async def vector_search(user_id: str, query_emb, k: int = 8) -> list[dict]:
    """依 user_id 分區做 KNN,回傳 [{meeting_id,title,created_at,snippet,distance}]。"""
    conn = await _connect_vec()
    try:
        cur = await conn.execute(
            "SELECT rowid, distance FROM vec_chunks "
            "WHERE user_id=? AND embedding MATCH ? ORDER BY distance LIMIT ?",
            (user_id, _pack(query_emb), k))
        hits = [(r["rowid"], r["distance"]) for r in await cur.fetchall()]
        if not hits:
            return []
        ids = [h[0] for h in hits]
        ph = ",".join("?" * len(ids))
        cur = await conn.execute(
            f"SELECT c.id, c.meeting_id, c.text, m.title, m.created_at "
            f"FROM chunks c JOIN meetings m ON m.id=c.meeting_id WHERE c.id IN ({ph})", ids)
        meta = {r["id"]: dict(r) for r in await cur.fetchall()}
    finally:
        await conn.close()
    out = []
    for cid, dist in hits:
        r = meta.get(cid)
        if r:
            out.append({"meeting_id": r["meeting_id"], "title": r["title"],
                        "created_at": r["created_at"], "snippet": r["text"],
                        "distance": round(float(dist), 4)})
    return out


async def set_status(mid: str, status: str, duration_sec: int | None = None):
    async with aiosqlite.connect(config.DB_PATH) as db:
        if duration_sec is None:
            await db.execute("UPDATE meetings SET status=? WHERE id=?", (status, mid))
        else:
            await db.execute("UPDATE meetings SET status=?, duration_sec=? WHERE id=?",
                             (status, duration_sec, mid))
        await db.commit()


# ---- transcript ----
async def save_transcript(mid: str, segments: list[dict]):
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute("DELETE FROM transcript_segments WHERE meeting_id=?", (mid,))
        await db.executemany(
            "INSERT INTO transcript_segments(meeting_id,seq,text,speaker,start_ms,end_ms) "
            "VALUES(?,?,?,?,?,?)",
            [(mid, i, s.get("text", ""), s.get("speaker"),
              s.get("start_ms"), s.get("end_ms")) for i, s in enumerate(segments)])
        await db.commit()


async def get_transcript(user_id: str, mid: str) -> list[dict] | None:
    if await get_meeting(user_id, mid) is None:
        return None
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM transcript_segments WHERE meeting_id=? ORDER BY seq", (mid,))
        return [{"id": f"s{r['seq']}", "text": r["text"], "speaker": r["speaker"],
                 "is_final": True, "start_ms": r["start_ms"], "end_ms": r["end_ms"]}
                for r in await cur.fetchall()]


async def search_transcripts(user_id: str, query: str, limit: int = 8) -> list[dict]:
    """跨會議關鍵字搜尋(⑤ 前哨;⑥ 會升級成語意檢索)。回傳含 meeting_id/title/snippet。"""
    q = (query or "").strip()
    if not q:
        return []
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT s.meeting_id, m.title, m.created_at, s.text "
            "FROM transcript_segments s JOIN meetings m ON m.id = s.meeting_id "
            "WHERE m.user_id = ? AND s.text LIKE ? ORDER BY m.created_at DESC LIMIT ?",
            (user_id, f"%{q}%", limit))
        return [{"meeting_id": r["meeting_id"], "title": r["title"],
                 "created_at": r["created_at"], "snippet": r["text"]}
                for r in await cur.fetchall()]


async def get_transcript_text(mid: str) -> str:
    """整場逐字稿純文字(帶說話者前綴);給 ④摘要 / ⑤QA 用。"""
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM transcript_segments WHERE meeting_id=? ORDER BY seq", (mid,))
        rows = await cur.fetchall()
    lines = [f"{r['speaker']}：{r['text']}" if r["speaker"] else r["text"] for r in rows]
    return "\n".join(lines)


# ---- summary (④ 用;先備好介面) ----
async def save_summary(mid: str, data: dict):
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(
            "INSERT INTO summaries(meeting_id,data,created_at) VALUES(?,?,?) "
            "ON CONFLICT(meeting_id) DO UPDATE SET data=excluded.data, created_at=excluded.created_at",
            (mid, json.dumps(data, ensure_ascii=False), _now()))
        await db.execute("UPDATE meetings SET has_summary=1 WHERE id=?", (mid,))
        await db.commit()


async def get_summary(user_id: str, mid: str) -> dict | None:
    if await get_meeting(user_id, mid) is None:
        return None
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT data FROM summaries WHERE meeting_id=?", (mid,))
        r = await cur.fetchone()
        return json.loads(r["data"]) if r else None
