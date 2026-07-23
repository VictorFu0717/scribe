#!/usr/bin/env python3
"""scribe server — 入口。

語音會議助理的 server 端。組裝 FastAPI app、啟動時載入模型 + 初始化 DB、掛載路由。
細節見各模組與 README.md:
    app/config.py            設定
    app/models.py            本地 ASR/語者 模型
    app/db.py                SQLite 儲存(meetings/transcripts/summaries)
    app/ws.py                /ws/asr 即時轉錄 + 說話者辨識 + 定稿寫入
    app/routers/meetings.py  會議 CRUD
    app/chat_qa.py           /meeting/chat 單場問答(舊端點,待 agentic 助理取代)

執行:
    python main.py           # 預設 :8005(需先起 Qwen3-ASR@:9000、對話 LLM@:8004)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config, db, models
from app.assistant import router as assistant_router
from app.chat_qa import router as qa_router
from app.routers.meetings import router as meetings_router
from app.summarize import router as summary_router
from app.upload import router as upload_router
from app.ws import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    await models.startup()
    yield
    await models.shutdown()


app = FastAPI(title="scribe server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)
app.include_router(meetings_router)
app.include_router(summary_router)
app.include_router(upload_router)
app.include_router(assistant_router)
app.include_router(qa_router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "loaded": models.is_loaded(),
        "preview_model": config.STREAM_MODEL,
        "vad_model": config.VAD_MODEL,
        "finalize_model": config.QWEN_MODEL,
        "vllm": config.VLLM_BASE_URL,
        "chat_llm": config.CHAT_BASE_URL,
    }


if __name__ == "__main__":
    import uvicorn

    # 單一 worker:模型只載一份;併發靠 async 連線 + vLLM continuous batching(見 README「併發模型」)
    uvicorn.run(app, host="0.0.0.0", port=config.PORT, workers=1)
