"""Embedding client(⑥ RAG)。預設 Ollama bge-m3(1024 維),OpenAI 相容 /v1/embeddings。"""

from __future__ import annotations

from openai import AsyncOpenAI

from app import config

_client = AsyncOpenAI(base_url=config.EMBED_BASE_URL, api_key=config.EMBED_API_KEY)


async def embed(texts: list[str] | str) -> list[list[float]]:
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []
    r = await _client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [d.embedding for d in r.data]
