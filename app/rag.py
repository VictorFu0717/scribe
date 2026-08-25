"""⑥ RAG:逐字稿切塊 → embedding → sqlite-vec 向量庫;語意檢索(多租戶 user_id 隔離 + 日期過濾)。

- index_meeting:定稿/上傳後呼叫,把該會議逐字稿切塊、embedding、存入向量庫(先刪舊的)。
- semantic_search:query embedding → vec KNN(依 user_id 分區)→ 可選日期範圍過濾 → top-k。
被 assistant 的 search_meetings 工具使用(把原本的關鍵字搜尋升級成語意)。
"""

from __future__ import annotations

import asyncio

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
    try:
        embs = await _embed.embed(chunks)
        if not all(_embed.is_valid(e) for e in embs):
            raise ValueError("embedding 含 NaN/inf")
    except Exception as e:
        # 整批失敗(常見於某一塊觸發 bge-m3 的 NaN)→ 逐筆重試,只丟掉真正壞的那幾塊
        print(f"[rag] {meeting_id} 整批 embedding 失敗({e}),改逐筆重試")
        embs = await _embed.embed_each(chunks)
    # embedding 壞掉的塊仍然寫入(embedding=None):不進向量表,但關鍵字檢索照樣搜得到
    rows = [{"seq": i, "text": chunks[i],
             "embedding": embs[i] if _embed.is_valid(embs[i]) else None}
            for i in range(len(chunks))]
    bad = sum(1 for r in rows if r["embedding"] is None)
    if bad:
        print(f"[rag] {meeting_id} 有 {bad}/{len(rows)} 塊無法 embedding,僅建關鍵字索引")
    await db.store_chunks(user_id, meeting_id, rows)


def _in_range(h, date_from, date_to) -> bool:
    if not (date_from or date_to):
        return True
    lo = date_from or ""
    hi = (date_to + "T23:59:59Z") if date_to else "9999"
    return lo <= (h.get("created_at") or "") <= hi


async def _tag_scope(user_id: str, tags):
    """標籤 → 候選會議 id;沒指定標籤回 None(不限制)。"""
    if not db.norm_tags(tags):
        return None
    return await db.meetings_with_tags(user_id, tags)


async def semantic_search(user_id: str, query: str, k: int = 6,
                          date_from: str | None = None, date_to: str | None = None,
                          tags=None) -> list[dict]:
    """純語意檢索。date_from/date_to 為 YYYY-MM-DD(含)日期範圍(依會議 created_at);
    tags 有值時只在帶有「任一個」該標籤的會議裡找(在向量檢索當下就限制候選,不是後過濾)。"""
    if not (query or "").strip():
        return []
    scope = await _tag_scope(user_id, tags)
    if scope is not None and not scope:
        return []
    qemb = (await _embed.embed([query]))[0]
    over = k * 4 if (date_from or date_to) else k
    hits = await db.vector_search(user_id, qemb, over, meeting_ids=scope)
    return [h for h in hits if _in_range(h, date_from, date_to)][:k]


async def hybrid_search(user_id: str, query: str, k: int = 6,
                        date_from: str | None = None, date_to: str | None = None,
                        tags=None) -> list[dict]:
    """向量 + 關鍵字 混合檢索,以 RRF(Reciprocal Rank Fusion)合併。

    兩者互補:向量擅長「意思相近但用詞不同」(問『營收表現』命中『這季賺了多少』),
    關鍵字擅長「專有名詞精確比對」(健保署、長照2.0、人名)—— 這類詞的 embedding
    常被周圍語意稀釋,正是純向量最容易漏掉的。

    RRF 只用「名次」而非分數,所以不必去校正 cosine 距離與 bm25 兩種不同尺度:
        score(doc) = Σ 1/(RRF_K + rank_i)
    任一側失敗(例如 embedding 服務掛了)仍以另一側的結果作答,不整個壞掉。
    """
    q = (query or "").strip()
    if not q:
        return []
    scope = await _tag_scope(user_id, tags)
    if scope is not None and not scope:
        return []
    over = k * 4 if (date_from or date_to) else k * 2

    async def _vec():
        try:
            qemb = (await _embed.embed([q]))[0]
            return await db.vector_search(user_id, qemb, over, meeting_ids=scope)
        except Exception:
            return []

    async def _kw():
        try:
            return await db.keyword_search(user_id, q, over, meeting_ids=scope)
        except Exception:
            return []

    vec_hits, kw_hits = await asyncio.gather(_vec(), _kw())

    fused: dict = {}
    for hits in (vec_hits, kw_hits):
        for rank, h in enumerate(hits):
            if not _in_range(h, date_from, date_to):
                continue
            key = (h.get("meeting_id"), h.get("snippet"))
            row = fused.setdefault(key, {**h, "_score": 0.0})
            row["_score"] += 1.0 / (config.RAG_RRF_K + rank + 1)
    out = sorted(fused.values(), key=lambda r: -r["_score"])[:k]
    for r in out:
        r.pop("_score", None)
        r.pop("distance", None)
    return out
