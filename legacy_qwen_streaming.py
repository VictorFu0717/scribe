#!/usr/bin/env python3
"""
即時串流 ASR 伺服器 — FastAPI WebSocket + Qwen3-ASR-1.7B (vLLM backend)

用途:
    手機 / 瀏覽器把麥克風音訊一段段透過 WebSocket 送進來,伺服器用
    Qwen3-ASR 的串流 API 邊收邊辨識,把「已定稿 / 未定」的逐字稿即時
    推回去。這是「即時預覽」用的服務;正式逐字稿建議會後另用離線整段
    重轉(精度較高、可掛 forced aligner 出時間軸)。

安裝:
    pip install "qwen-asr[vllm]" fastapi "uvicorn[standard]" numpy

執行(重點:單一 process、不要開 --reload;vLLM 會 spawn 子進程,多
worker / reload 會互相衝突):
    python asr_ws_server.py
    # 或
    # uvicorn asr_ws_server:app --host 0.0.0.0 --port 8001 --workers 1

可用環境變數覆寫設定:
    ASR_MODEL_PATH, ASR_GPU_MEM_UTIL, ASR_MAX_NEW_TOKENS,
    ASR_UNFIXED_CHUNK_NUM, ASR_UNFIXED_TOKEN_NUM, ASR_CHUNK_SIZE_SEC, PORT

WebSocket 協定 (/ws/asr):
    Client -> Server:
        - binary frame : 原始音訊 = PCM 16-bit little-endian, 單聲道, 16000 Hz
        - text  frame  : {"type": "end"}    結束這段語音,伺服器做最終定稿
        - text  frame  : {"type": "reset"}  丟棄目前狀態、重新開始一段
    Server -> Client (JSON):
        - {"type":"partial","committed":...,"tentative":...,"text":...,"language":...}
        - {"type":"final","text":...,"language":...}
        - {"type":"error","detail":...}
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# --- 設定 ------------------------------------------------------------------
MODEL_PATH = os.getenv("ASR_MODEL_PATH", "Qwen/Qwen3-ASR-1.7B")
# 4090 測試機獨佔時可用 0.8~0.9;搬到共卡的 mohw-ai-1 記得壓低(例如 0.10)
GPU_MEM_UTIL = float(os.getenv("ASR_GPU_MEM_UTIL", "0.1"))
MAX_NEW_TOKENS = int(os.getenv("ASR_MAX_NEW_TOKENS", "32"))   # 串流每個窗口設小值
SAMPLE_RATE = 16000                                          # 協定固定 16k;client 需自行 resample

# 串流穩定度參數:數字越大 -> 尾巴修正空間越大、較準但晃越久;
#                 越小 -> 定稿快、跳動少,但可能較早定錯。可即時調整找手感。
UNFIXED_CHUNK_NUM = int(os.getenv("ASR_UNFIXED_CHUNK_NUM", "2"))
UNFIXED_TOKEN_NUM = int(os.getenv("ASR_UNFIXED_TOKEN_NUM", "5"))
CHUNK_SIZE_SEC = float(os.getenv("ASR_CHUNK_SIZE_SEC", "2.0"))

# 一個模型實例,所有連線共用;GPU 呼叫用鎖序列化以確保安全
_model = None
_model_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """啟動時載入模型一次,關閉時釋放。"""
    global _model
    # 延後匯入,避免在 uvicorn reload / import 階段提早初始化 vLLM
    from qwen_asr import Qwen3ASRModel

    print(f"[startup] loading {MODEL_PATH} (gpu_memory_utilization={GPU_MEM_UTIL}) ...")
    _model = Qwen3ASRModel.LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print("[startup] model ready.")
    yield
    _model = None
    print("[shutdown] model released.")


app = FastAPI(title="Qwen3-ASR streaming server", lifespan=lifespan)

# 只有瀏覽器測試頁跨網域連線才需要;native app 走 ws 不受 CORS 限制。
# 正式環境請把 allow_origins 收斂成你的來源。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH, "loaded": _model is not None}


def _longest_common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


async def _run_blocking(fn, *args) -> None:
    """在 threadpool 內跑阻塞的 GPU 呼叫,並用鎖確保同時只有一個串流在動模型。"""
    loop = asyncio.get_running_loop()
    async with _model_lock:
        await loop.run_in_executor(None, fn, *args)


def _new_state():
    return _model.init_streaming_state(
        unfixed_chunk_num=UNFIXED_CHUNK_NUM,
        unfixed_token_num=UNFIXED_TOKEN_NUM,
        chunk_size_sec=CHUNK_SIZE_SEC,
    )


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    await ws.accept()

    state = _new_state()
    prev_text = ""       # 上一次的完整假設,用來算穩定前綴
    committed_len = 0    # 已定稿長度(只增不減),伺服器端以 LocalAgreement 推導

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            # --- 控制訊息(text frame)---
            if msg.get("text") is not None:
                try:
                    ctrl = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue

                if ctrl.get("type") == "end":
                    await _run_blocking(_model.finish_streaming_transcribe, state)
                    await ws.send_text(json.dumps({
                        "type": "final",
                        "text": state.text or "",
                        "language": state.language,
                    }, ensure_ascii=False))
                    # 重置以接續下一段語音
                    state = _new_state()
                    prev_text, committed_len = "", 0
                elif ctrl.get("type") == "reset":
                    state = _new_state()
                    prev_text, committed_len = "", 0
                continue

            # --- 音訊資料(binary frame)= PCM16 LE mono 16k ---
            data = msg.get("bytes")
            if not data:
                continue
            seg = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if seg.size == 0:
                continue

            await _run_blocking(_model.streaming_transcribe, seg, state)

            cur = state.text or ""
            # LocalAgreement:已定稿 = 這次與上次假設的最長共同前綴(只增不減、且不超過目前長度)
            common = _longest_common_prefix_len(prev_text, cur)
            committed_len = min(max(committed_len, common), len(cur))
            prev_text = cur

            await ws.send_text(json.dumps({
                "type": "partial",
                "committed": cur[:committed_len],   # 前端用正常樣式呈現
                "tentative": cur[committed_len:],   # 前端用灰色 / 斜體(這段還會變)
                "text": cur,
                "language": state.language,
            }, ensure_ascii=False))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps(
                {"type": "error", "detail": str(e)}, ensure_ascii=False))
        except Exception:
            pass
    finally:
        # 盡量收尾(釋放該串流狀態相關資源;失敗忽略)
        try:
            await _run_blocking(_model.finish_streaming_transcribe, state)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    # 單一 worker、不要 reload:vLLM 會 spawn 子進程,多 worker / reload 會衝突
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8005")), workers=1)