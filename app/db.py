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
            CREATE TABLE IF NOT EXISTS translations(
                meeting_id TEXT, target TEXT, text TEXT, created_at TEXT,
                PRIMARY KEY (meeting_id, target)
            );
            CREATE TABLE IF NOT EXISTS users(
                id TEXT PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT, created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_meetings_user ON meetings(user_id, created_at);
            CREATE UNIQUE INDEX IF NOT EXISTS uniq_seg ON transcript_segments(meeting_id, seq);
            CREATE TABLE IF NOT EXISTS chunks(
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, meeting_id TEXT,
                seq INTEGER, text TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_meeting ON chunks(meeting_id);
            -- 會議標籤(使用者自訂,如「專案會議」「每週會議」):一場可多個,用來縮小 RAG 檢索範圍
            CREATE TABLE IF NOT EXISTS meeting_tags(
                meeting_id TEXT NOT NULL, tag TEXT NOT NULL,
                PRIMARY KEY (meeting_id, tag)
            );
            CREATE INDEX IF NOT EXISTS idx_mtags_tag ON meeting_tags(tag);
            """
        )
        await db.commit()
        # ⑥ RAG:sqlite-vec 向量表(user_id 分區,rowid = chunks.id)
        await db.enable_load_extension(True)
        await db.load_extension(sqlite_vec.loadable_path())
        await db.enable_load_extension(False)
        # ⚠️ 距離用 vec0 的**預設 L2**,不是 cosine。目前正確是因為 bge-m3 回傳單位長度向量,
        # 此時 ‖a-b‖² = 2(1-cos) → 排序與 cosine 完全等價(已實測)。
        # 換成**不做正規化**的 EMBED_MODEL 時排序會悄悄變錯且無錯誤訊息,
        # 屆時要改成 `embedding float[N] distance_metric=cosine` 並重建索引(POST /meetings/reindex)。
        await db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"user_id text partition key, embedding float[{config.EMBED_DIM}])")
        await db.commit()
        # hybrid 檢索的關鍵字側:FTS5 + **trigram** 分詞(rowid = chunks.id)。
        # 預設的 unicode61 對中文完全無效 —— 中文沒有空白,整串會變成單一 token,
        # 實測查「健保署」「預算成長」都是 0 筆。trigram 以 3 字滑窗建索引才搜得到。
        await db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks "
                         "USING fts5(text, tokenize='trigram')")
        # 舊資料庫回填(⑥ 之前建的 chunks 沒有 FTS 列)
        cur = await db.execute("SELECT (SELECT count(*) FROM chunks), "
                               "(SELECT count(*) FROM fts_chunks)")
        n_chunks, n_fts = await cur.fetchone()
        if n_chunks and not n_fts:
            await db.execute("INSERT INTO fts_chunks(rowid, text) SELECT id, text FROM chunks")
            print(f"[db] FTS 回填 {n_chunks} 塊")
        await db.commit()
    print(f"[db] ready: {config.DB_PATH}")


def _meeting_row(r, tags=None) -> dict:
    return {
        "id": r["id"], "title": r["title"], "created_at": r["created_at"],
        "duration_sec": r["duration_sec"], "status": r["status"],
        "has_summary": bool(r["has_summary"]), "audio_url": None,
        "tags": tags if tags is not None else [],
    }


def norm_tags(tags) -> list[str]:
    """去空白、去空字串、去重(不分大小寫)、限長。保留使用者原本的大小寫寫法。"""
    out, seen = [], set()
    for t in (tags or []):
        if t is None or not isinstance(t, (str, int, float)):
            continue                      # str(None) 會變成字串 "None",得先擋掉
        t = str(t).strip()[:40]
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[:20]


async def _tags_of(db, mids: list[str]) -> dict:
    """一次查多場會議的標籤(避免 N+1)。"""
    if not mids:
        return {}
    ph = ",".join("?" * len(mids))
    cur = await db.execute(
        f"SELECT meeting_id, tag FROM meeting_tags WHERE meeting_id IN ({ph}) ORDER BY tag",
        mids)
    out: dict = {m: [] for m in mids}
    for r in await cur.fetchall():
        out[r[0]].append(r[1])
    return out


async def set_tags(user_id: str, mid: str, tags) -> list[str] | None:
    """整組覆寫某會議的標籤(空陣列 = 清空)。會議不屬於此使用者則回 None。"""
    if await get_meeting(user_id, mid) is None:
        return None
    clean = norm_tags(tags)
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute("DELETE FROM meeting_tags WHERE meeting_id=?", (mid,))
        for t in clean:
            await db.execute("INSERT OR IGNORE INTO meeting_tags(meeting_id,tag) VALUES(?,?)",
                             (mid, t))
        await db.commit()
    return clean


async def list_tags(user_id: str) -> list[dict]:
    """這個使用者用過的所有標籤 + 各自的會議數(給 App 下拉、給助理注入 prompt)。"""
    async with aiosqlite.connect(config.DB_PATH) as db:
        cur = await db.execute(
            "SELECT t.tag, COUNT(*) n FROM meeting_tags t JOIN meetings m ON m.id=t.meeting_id "
            "WHERE m.user_id=? GROUP BY t.tag ORDER BY n DESC, t.tag", (user_id,))
        return [{"tag": r[0], "count": r[1]} for r in await cur.fetchall()]


async def meetings_with_tags(user_id: str, tags) -> list[str]:
    """帶有「任一個」指定標籤的會議 id(標籤比對不分大小寫)。"""
    clean = norm_tags(tags)
    if not clean:
        return []
    ph = ",".join("?" * len(clean))
    async with aiosqlite.connect(config.DB_PATH) as db:
        cur = await db.execute(
            f"SELECT DISTINCT t.meeting_id FROM meeting_tags t JOIN meetings m ON m.id=t.meeting_id "
            f"WHERE m.user_id=? AND LOWER(t.tag) IN ({ph})",
            [user_id] + [c.lower() for c in clean])
        return [r[0] for r in await cur.fetchall()]


async def update_meeting(user_id: str, mid: str, title=None, tags=None) -> dict | None:
    """更新會議(目前支援 title / tags);None = 該欄位不動。"""
    if await get_meeting(user_id, mid) is None:
        return None
    if title is not None:
        async with aiosqlite.connect(config.DB_PATH) as db:
            await db.execute("UPDATE meetings SET title=? WHERE id=?",
                             (str(title).strip()[:200] or "未命名會議", mid))
            await db.commit()
    if tags is not None:
        await set_tags(user_id, mid, tags)
    return await get_meeting(user_id, mid)


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
async def create_meeting(user_id: str, title: str | None, tags=None) -> dict:
    mid = _new_id()
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(
            "INSERT INTO meetings(id,user_id,title,created_at) VALUES(?,?,?,?)",
            (mid, user_id, title or "未命名會議", _now()))
        for t in norm_tags(tags):
            await db.execute("INSERT OR IGNORE INTO meeting_tags(meeting_id,tag) VALUES(?,?)",
                             (mid, t))
        await db.commit()
    return await get_meeting(user_id, mid)


async def list_meetings(user_id: str, tags=None) -> list[dict]:
    """tags 有值時,只回帶有「任一個」該標籤的會議。"""
    only = set(await meetings_with_tags(user_id, tags)) if norm_tags(tags) else None
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM meetings WHERE user_id=? ORDER BY created_at DESC", (user_id,))
        rows = [r for r in await cur.fetchall() if only is None or r["id"] in only]
        tmap = await _tags_of(db, [r["id"] for r in rows])
        return [_meeting_row(r, tmap.get(r["id"], [])) for r in rows]


async def get_meeting(user_id: str, mid: str) -> dict | None:
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM meetings WHERE user_id=? AND id=?", (user_id, mid))
        r = await cur.fetchone()
        if not r:
            return None
        return _meeting_row(r, (await _tags_of(db, [mid])).get(mid, []))


async def delete_meeting(user_id: str, mid: str) -> bool:
    # 先確認會議屬於此使用者;不是就完全不動(避免用別人的 meeting_id 刪到別人的逐字稿/向量)
    if await get_meeting(user_id, mid) is None:
        return False
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute("DELETE FROM meetings WHERE id=?", (mid,))
        await db.execute("DELETE FROM transcript_segments WHERE meeting_id=?", (mid,))
        await db.execute("DELETE FROM summaries WHERE meeting_id=?", (mid,))
        await db.execute("DELETE FROM translations WHERE meeting_id=?", (mid,))
        await db.execute("DELETE FROM meeting_tags WHERE meeting_id=?", (mid,))
        await db.commit()
    await delete_chunks(mid)   # 連帶刪向量索引(chunks + vec_chunks)
    return True


# ---- 向量索引 (⑥ RAG, sqlite-vec) ----
async def store_chunks(user_id: str, meeting_id: str, chunks: list[dict]):
    """先刪該會議舊塊,再存入新塊(chunks:{seq,text,embedding})。vec rowid = chunks.id。

    embedding 為 None 的塊仍會寫入 chunks + fts_chunks,只是不進向量表 ——
    這樣「embedding 失敗的內容」至少關鍵字檢索還找得到(hybrid 的好處之一)。
    """
    await delete_chunks(meeting_id)
    conn = await _connect_vec()
    try:
        for ch in chunks:
            cur = await conn.execute(
                "INSERT INTO chunks(user_id,meeting_id,seq,text) VALUES(?,?,?,?)",
                (user_id, meeting_id, ch["seq"], ch["text"]))
            cid = cur.lastrowid
            if ch.get("embedding") is not None:
                await conn.execute(
                    "INSERT INTO vec_chunks(rowid,user_id,embedding) VALUES(?,?,?)",
                    (cid, user_id, _pack(ch["embedding"])))
            await conn.execute("INSERT INTO fts_chunks(rowid,text) VALUES(?,?)",
                               (cid, ch["text"]))
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
            await conn.execute("DELETE FROM fts_chunks WHERE rowid=?", (cid,))
        await conn.execute("DELETE FROM chunks WHERE meeting_id=?", (meeting_id,))
        await conn.commit()
    finally:
        await conn.close()


async def vector_search(user_id: str, query_emb, k: int = 8,
                        meeting_ids: list[str] | None = None) -> list[dict]:
    """依 user_id 分區做 KNN,回傳 [{meeting_id,title,created_at,snippet,distance}]。

    meeting_ids 有值時,**在向量檢索當下就限制候選**(sqlite-vec 原生支援 rowid IN)。
    這比「先取 top-k 再過濾」正確得多:500 場會議裡只有 3 場帶某標籤時,
    後過濾很可能整個篩空;先限制候選則保證回得滿 k 筆。
    """
    conn = await _connect_vec()
    try:
        if meeting_ids is not None:
            if not meeting_ids:
                return []
            ph = ",".join("?" * len(meeting_ids))
            cur = await conn.execute(
                f"SELECT id FROM chunks WHERE user_id=? AND meeting_id IN ({ph})",
                [user_id] + list(meeting_ids))
            ids = [r[0] for r in await cur.fetchall()]
            if not ids:
                return []
            iph = ",".join("?" * len(ids))
            cur = await conn.execute(
                f"SELECT rowid, distance FROM vec_chunks "
                f"WHERE user_id=? AND embedding MATCH ? AND rowid IN ({iph}) "
                f"ORDER BY distance LIMIT ?",
                [user_id, _pack(query_emb)] + ids + [k])
        else:
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


async def upsert_segment(meeting_id: str, seq: int, text: str, speaker: str | None,
                         start_ms: int | None, end_ms: int | None):
    """逐句寫入(即時串流用):按 (meeting_id, seq) upsert,斷線也不丟已定稿的句子。"""
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(
            "INSERT INTO transcript_segments(meeting_id,seq,text,speaker,start_ms,end_ms) "
            "VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(meeting_id,seq) DO UPDATE SET "
            "text=excluded.text, speaker=excluded.speaker, "
            "start_ms=excluded.start_ms, end_ms=excluded.end_ms",
            (meeting_id, seq, text, speaker, start_ms, end_ms))
        await db.commit()


async def count_segments(meeting_id: str) -> int:
    """該會議 DB 既有段數(reconnect-continue 的續錄起點)。"""
    async with aiosqlite.connect(config.DB_PATH) as db:
        cur = await db.execute(
            "SELECT COUNT(*) FROM transcript_segments WHERE meeting_id=?", (meeting_id,))
        return (await cur.fetchone())[0]


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


def fts_terms(query: str, max_terms: int = 24) -> list[str]:
    """把查詢拆成可餵給 trigram FTS5 的詞。

    中文沒有空白可切,又不想引進斷詞器(字典外的專有名詞如「長照2.0」反而切不好),
    所以對中文字串取 **3 字滑窗**:「健保署預算」→ 健保署 / 保署預 / 署預算。
    雜訊窗(保署預)幾乎不會命中,bm25 又會壓低常見窗的權重,實際上等同「字元三元組檢索」。
    英數詞(>=3)直接當一個詞。
    """
    import re
    out: list[str] = []
    for w in re.findall(r"[A-Za-z0-9_]{3,}", query or ""):
        out.append(w.lower())
    for run in re.findall(r"[^\x00-\x7f]{3,}", query or ""):     # 連續非 ASCII(中文等)
        for i in range(len(run) - 2):
            out.append(run[i:i + 3])
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq[:max_terms]


async def keyword_search(user_id: str, query: str, k: int = 8,
                         meeting_ids: list[str] | None = None) -> list[dict]:
    """關鍵字檢索(FTS5 trigram + bm25),多租戶以 chunks.user_id 過濾。

    trigram 需要 >=3 字元;查詢過短(如「長照」)拆不出詞 → 退回 LIKE 子字串比對,
    否則那類查詢會靜默回空,反而比原本的 LIKE 還差。
    """
    q = (query or "").strip()
    if not q:
        return []
    terms = fts_terms(q)
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if meeting_ids is not None and not meeting_ids:
            return []
        mfil = ""
        margs: list = []
        if meeting_ids:
            mfil = f" AND c.meeting_id IN ({','.join('?' * len(meeting_ids))})"
            margs = list(meeting_ids)
        if terms:
            expr = " OR ".join(f'"{t}"' for t in terms)   # 雙引號:避免 FTS5 語法字元
            cur = await db.execute(
                "SELECT c.meeting_id, c.text AS snippet, m.title, m.created_at, "
                "       bm25(fts_chunks) AS score "
                "FROM fts_chunks f JOIN chunks c ON c.id = f.rowid "
                "JOIN meetings m ON m.id = c.meeting_id "
                "WHERE fts_chunks MATCH ? AND c.user_id = ?" + mfil + " "
                "ORDER BY score LIMIT ?", [expr, user_id] + margs + [k])
        else:
            cur = await db.execute(
                "SELECT c.meeting_id, c.text AS snippet, m.title, m.created_at, 0 AS score "
                "FROM chunks c JOIN meetings m ON m.id = c.meeting_id "
                "WHERE c.user_id = ? AND c.text LIKE ?" + mfil + " "
                "ORDER BY m.created_at DESC LIMIT ?", [user_id, f"%{q}%"] + margs + [k])
        return [{"meeting_id": r["meeting_id"], "title": r["title"],
                 "created_at": r["created_at"], "snippet": r["snippet"]}
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


# ---- translations (留檔翻譯;每個 (meeting_id, target) 一份) ----
async def save_translation(mid: str, target: str, text: str):
    async with aiosqlite.connect(config.DB_PATH) as db:
        await db.execute(
            "INSERT INTO translations(meeting_id,target,text,created_at) VALUES(?,?,?,?) "
            "ON CONFLICT(meeting_id,target) DO UPDATE SET text=excluded.text, created_at=excluded.created_at",
            (mid, target, text, _now()))
        await db.commit()


async def get_translation(user_id: str, mid: str, target: str) -> str | None:
    if await get_meeting(user_id, mid) is None:
        return None
    async with aiosqlite.connect(config.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT text FROM translations WHERE meeting_id=? AND target=?", (mid, target))
        r = await cur.fetchone()
        return r["text"] if r else None
