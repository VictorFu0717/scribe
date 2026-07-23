"""⑥ RAG:逐字稿切塊 → embedding → sqlite-vec 向量庫;語意檢索(多租戶 user_id 隔離 + 日期過濾)。

- index_meeting:定稿/上傳後呼叫,把該會議逐字稿切塊、embedding、存入向量庫(先刪舊的)。
- semantic_search:query embedding → vec KNN(依 user_id 分區)→ 可選日期範圍過濾 → top-k。
被 assistant 的 search_meetings 工具使用(把原本的關鍵字搜尋升級成語意)。
"""

from __future__ import annotations

from app import config, db, embed as _embed


def _chunk_segments(segs: list[dict], max_chars: int) -> list[str]:
    """把逐字稿片段合併成 ~max_chars 的塊(帶說話者前綴)。"""
    chunks, cur = [], ""
    for s in segs:
        line = f"{s['speaker']}：{s['text']}" if s.get("speaker") else s.get("text", "")
        if not line.strip():
            continue
        if cur and len(cur) + len(line) > max_chars:
            chunks.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


async def index_meeting(user_id: str, meeting_id: str):
    """建立/更新某會議的向量索引(冪等,會先刪舊塊)。"""
    segs = await db.get_transcript(user_id, meeting_id)
    if not segs:
        return
    chunks = _chunk_segments(segs, config.RAG_CHUNK_CHARS)
    if not chunks:
        return
    embs = await _embed.embed(chunks)
    await db.store_chunks(user_id, meeting_id,
                          [{"seq": i, "text": chunks[i], "embedding": embs[i]}
                           for i in range(len(chunks))])


async def semantic_search(user_id: str, query: str, k: int = 6,
                          date_from: str | None = None, date_to: str | None = None) -> list[dict]:
    """語意檢索;date_from/date_to 為 YYYY-MM-DD(含)日期範圍過濾(依會議 created_at)。"""
    if not (query or "").strip():
        return []
    qemb = (await _embed.embed([query]))[0]
    over = k * 4 if (date_from or date_to) else k
    hits = await db.vector_search(user_id, qemb, over)
    if date_from or date_to:
        lo = date_from or ""
        hi = (date_to + "T23:59:59Z") if date_to else "9999"
        hits = [h for h in hits if lo <= (h.get("created_at") or "") <= hi]
    return hits[:k]
