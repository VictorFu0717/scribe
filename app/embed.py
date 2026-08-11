"""Embedding client(⑥ RAG)。預設 Ollama bge-m3(1024 維),OpenAI 相容 /v1/embeddings。"""

from __future__ import annotations

import math

from openai import AsyncOpenAI

from app import config

_client = AsyncOpenAI(base_url=config.EMBED_BASE_URL, api_key=config.EMBED_API_KEY)


def is_valid(vec) -> bool:
    """向量可用嗎?擋掉 NaN/inf —— 存進 sqlite-vec 會污染整個索引的距離計算。"""
    return bool(vec) and all(math.isfinite(x) for x in vec)


async def embed(texts: list[str] | str) -> list[list[float]]:
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []
    r = await _client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [d.embedding for d in r.data]


async def embed_each(texts: list[str]) -> list[list[float] | None]:
    """逐筆 embedding,壞掉的回 None。

    為什麼需要:實測 bge-m3(Ollama)會對**特定內容**回 NaN 而讓整個請求 500
    (可 100% 重現,與順序/分隔符無關),而且**同批其他文字會被一起拖下水**。
    原本 index_meeting 整批送,一塊踩雷 → 整場會議索引失敗 → 該會議在 RAG 中
    直接消失(且只送一個 error 事件,使用者不會發現)。改成先試整批(快),
    失敗才逐筆重試,把損失侷限在真正有問題的那一塊。
    """
    out: list[list[float] | None] = []
    for t in texts:
        try:
            r = await _client.embeddings.create(model=config.EMBED_MODEL, input=[t])
            v = r.data[0].embedding
            out.append(v if is_valid(v) else None)
        except Exception:
            out.append(None)
    return out
