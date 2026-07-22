#!/usr/bin/env python3
"""
即時串流 ASR 伺服器 — 混合雙模型 (方案 2)

架構:
    [瀏覽器/手機 麥克風]
        │  WebSocket, PCM16 LE mono 16k, 一段段送
        ▼
    [本服務 FastAPI]
        ├─ 即時預覽 (低延遲):  FunASR paraformer-zh-streaming
        │     邊收邊吐字 → partial.tentative(灰字),即時顯示
        ├─ 斷句 (endpoint):    FunASR fsmn-vad (streaming)
        │     偵測到句子結束 → 把整句音訊丟去定稿
        └─ 定稿 (高準/併發):   Qwen3-ASR @ vllm serve (OpenAI 相容端點)
              每句結束打一次 async HTTP,回來的高準文字「就地覆蓋」該句預覽

    為什麼這樣切:
    - 多個連線可同時服務;每句「定稿」是 async 打到 vllm serve,
      vLLM 自己做 continuous batching → 定稿是「真併發」。
    - 預覽 / VAD 是很小的本地模型,呼叫極快;以一把鎖序列化確保
      同一個 torch 模型實例不會被多執行緒同時呼叫(安全 > 極致併發,
      而且真正吃資源的定稿已經外包給 vLLM 了)。

前置步驟:
    1) 起 Qwen3-ASR 定稿服務 (另一個 process):
         vllm serve Qwen/Qwen3-ASR-1.7B --port 8000 --gpu-memory-utilization 0.6
    2) 裝預覽 / VAD 模型套件:
         pip install funasr
    3) 起本服務:
         python main.py            # 預設 :8005

環境變數:
    VLLM_BASE_URL  (預設 http://localhost:8000/v1)
    VLLM_API_KEY   (預設 EMPTY)
    QWEN_MODEL     (預設 Qwen/Qwen3-ASR-1.7B)
    STREAM_MODEL   (預設 paraformer-zh-streaming)
    VAD_MODEL      (預設 fsmn-vad)
    FUNASR_HUB     (預設 hf;本機下載較快。中國內網可設 ms 用 modelscope)
    DEVICE         (預設 cuda)
    ASR_LANG       (選填;不設 = Qwen3 自動偵測語言)
    ASR_TRADITIONAL(預設 1;簡->繁台灣用語轉換。設 0 關閉)
    MAX_SEG_SEC    (預設 30;連續講不停時的安全切段秒數)
    PORT           (預設 8005)

WebSocket 協定 (/ws/asr) — 與舊版相容,差別在 committed 會被高準結果「升級」:
    Client -> Server:
        - binary : PCM16 LE mono 16k
        - text   : {"type":"end"}    結束本段,等所有句子定稿後回 final
        - text   : {"type":"reset"}  丟棄狀態重來
    Server -> Client (JSON):
        - {"type":"partial","committed":..,"tentative":..,"text":..,"language":..}
              committed = 已斷句的句子(先放預覽字佔位,Qwen 回來後就地升級成高準字)
              tentative = 目前這句的即時預覽(灰字,還會變)
        - {"type":"final","text":..}
        - {"type":"error","detail":..}
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# --- 設定 ------------------------------------------------------------------
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:9000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3-ASR-1.7B")

STREAM_MODEL = os.getenv("STREAM_MODEL", "paraformer-zh-streaming")
VAD_MODEL = os.getenv("VAD_MODEL", "fsmn-vad")
FUNASR_HUB = os.getenv("FUNASR_HUB", "hf")   # 本機實測 hf 下載遠快於 ms;離線可預先快取
DEVICE = os.getenv("DEVICE", "cuda")

ASR_LANG = os.getenv("ASR_LANG") or None
MAX_SEG_SEC = float(os.getenv("MAX_SEG_SEC", "30"))
# 簡->繁(台灣用語)轉換:預覽(paraformer 只輸出簡體)與定稿都會過一次
ASR_TW = os.getenv("ASR_TRADITIONAL", "1") not in ("0", "false", "False", "")

SAMPLE_RATE = 16000                 # 協定固定 16k;client 需自行 resample

# paraformer-zh-streaming 串流參數(FunASR 官方建議值)
# chunk_size=[0,10,5] → 600ms 一格;數字越大越準但延遲越高
PF_CHUNK = [0, 10, 5]
ENC_LOOKBACK = 4
DEC_LOOKBACK = 1
CHUNK_STRIDE = PF_CHUNK[1] * 960    # 10 * 960 = 9600 samples = 600ms @16k
CHUNK_MS = int(CHUNK_STRIDE / SAMPLE_RATE * 1000)  # 600

# 本地模型實例(所有連線共用),用鎖序列化 GPU 呼叫確保 thread 安全
_pf_model = None      # 即時預覽
_vad_model = None     # 斷句
_oai = None           # 定稿用的 async OpenAI client(打 vllm serve)
_cc = None            # OpenCC s2twp 轉換器(簡->繁台灣)
_funasr_lock = asyncio.Lock()


def _to_tw(text: str) -> str:
    """簡體 -> 繁體(台灣用語);未啟用或轉換失敗則原樣回傳。"""
    if not text or _cc is None:
        return text
    try:
        return _cc.convert(text)
    except Exception:
        return text


def _clean_qwen(raw: str) -> str:
    """Qwen3-ASR 經 vLLM 回傳會夾帶模板標記,例如
    'language Chinese<asr_text>實際文字'。取 <asr_text> 之後、並清掉殘留角括號標記。"""
    if not raw:
        return ""
    if "<asr_text>" in raw:
        raw = raw.split("<asr_text>", 1)[1]
    raw = re.sub(r"<[^>]*>", "", raw)   # 去掉任何殘留的 <...> 標記
    return raw.strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """啟動時載入本地預覽 / VAD 模型,並建立到 vllm serve 的 client。"""
    global _pf_model, _vad_model, _oai, _cc
    from funasr import AutoModel
    from openai import AsyncOpenAI

    print(f"[startup] loading FunASR preview={STREAM_MODEL} vad={VAD_MODEL} "
          f"(hub={FUNASR_HUB}, device={DEVICE}) ...")
    _pf_model = AutoModel(model=STREAM_MODEL, hub=FUNASR_HUB, device=DEVICE,
                          disable_update=True)
    _vad_model = AutoModel(model=VAD_MODEL, hub=FUNASR_HUB, device=DEVICE,
                           disable_update=True)
    _oai = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    if ASR_TW:
        try:
            import opencc
            _cc = opencc.OpenCC("s2twp")
            print("[startup] OpenCC s2twp 已啟用 (簡體 -> 繁體台灣用語)")
        except Exception as e:
            print(f"[startup] OpenCC 不可用,略過繁簡轉換: {e}")
    print(f"[startup] ready. finalize -> {VLLM_BASE_URL} ({QWEN_MODEL})")
    yield
    _pf_model = _vad_model = _oai = _cc = None
    print("[shutdown] released.")


app = FastAPI(title="Qwen3-ASR hybrid streaming server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ③ 單場會議 QA:POST /meeting/chat(把該場逐字稿當 context 問答)
from chat_qa import router as qa_router  # noqa: E402
app.include_router(qa_router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "preview_model": STREAM_MODEL,
        "vad_model": VAD_MODEL,
        "finalize_model": QWEN_MODEL,
        "vllm": VLLM_BASE_URL,
        "loaded": _pf_model is not None,
    }


async def _funasr_generate(model, chunk: np.ndarray, cache: dict, is_final: bool, **kw):
    """在 threadpool 跑阻塞的 FunASR 呼叫,並用鎖確保同一模型不被同時呼叫。"""
    loop = asyncio.get_running_loop()
    async with _funasr_lock:
        return await loop.run_in_executor(
            None,
            lambda: model.generate(input=chunk, cache=cache, is_final=is_final, **kw),
        )


async def _finalize_qwen(seg: np.ndarray) -> str:
    """把一句音訊送到 vllm serve 的 Qwen3-ASR 做高準定稿(async,可併發)。"""
    bio = io.BytesIO()
    sf.write(bio, seg, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    kwargs = {"model": QWEN_MODEL, "file": ("seg.wav", bio.getvalue(), "audio/wav")}
    if ASR_LANG:
        kwargs["language"] = ASR_LANG
    resp = await _oai.audio.transcriptions.create(**kwargs)
    return _clean_qwen(getattr(resp, "text", "") or "")


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    await ws.accept()

    # --- 每連線狀態 ---
    pf_cache: dict = {}                     # 預覽 streaming cache(每句重置)
    vad_cache: dict = {}                    # VAD streaming cache(整條連線連續)
    audio_buf = np.zeros(0, dtype=np.float32)   # 尚未湊滿一個 chunk 的殘餘
    seg_samples: list[np.ndarray] = []      # 目前這句累積的音訊(給 Qwen 定稿)
    seg_dur = 0.0                           # 目前這句長度(秒)
    segments: list[str] = []                # 已斷句句子:先放預覽字,Qwen 回來就升級
    cur_preview = ""                        # 目前這句的即時預覽

    send_lock = asyncio.Lock()              # 避免收發兩邊同時 send
    fin_queue: asyncio.Queue = asyncio.Queue()   # 待定稿佇列 (idx, audio)

    async def send(obj: dict):
        async with send_lock:
            try:
                await ws.send_text(json.dumps(obj, ensure_ascii=False))
            except Exception:
                pass

    async def push_partial():
        committed = _to_tw("".join(segments))
        tentative = _to_tw(cur_preview)
        await send({
            "type": "partial",
            "committed": committed,
            "tentative": tentative,
            "text": committed + tentative,
            "language": ASR_LANG,
        })

    async def finalizer():
        """背景 worker:逐句送 Qwen3 定稿,回來就地升級 segments 並推更新。"""
        while True:
            item = await fin_queue.get()
            try:
                if item is None:
                    break
                idx, seg = item
                try:
                    text = await _finalize_qwen(seg)
                    if text:
                        segments[idx] = text
                        await push_partial()
                except Exception as e:
                    await send({"type": "error", "detail": f"finalize: {e}"})
            finally:
                fin_queue.task_done()

    fin_task = asyncio.create_task(finalizer())

    def close_segment():
        """收掉目前這句:預覽字佔位入列,音訊排進定稿佇列,重置預覽狀態。"""
        nonlocal cur_preview, pf_cache, seg_samples, seg_dur
        seg = np.concatenate(seg_samples) if seg_samples else np.zeros(0, np.float32)
        idx = len(segments)
        segments.append(cur_preview)        # 先用預覽字佔位(committed 立即有字)
        cur_preview = ""
        pf_cache = {}                       # 新的一句 → 預覽 cache 重置
        seg_samples = []
        seg_dur = 0.0
        if seg.size > 0:
            fin_queue.put_nowait((idx, seg))   # 交給 finalizer 併發定稿

    async def process_chunk(chunk: np.ndarray, is_final: bool):
        nonlocal cur_preview, seg_dur
        if chunk.size:
            seg_samples.append(chunk)
            seg_dur += chunk.size / SAMPLE_RATE

        # 1) VAD 斷句偵測
        seg_ended = False
        try:
            vad_res = await _funasr_generate(
                _vad_model, chunk, vad_cache, is_final, chunk_size=CHUNK_MS)
            if vad_res and vad_res[0].get("value"):
                for s in vad_res[0]["value"]:
                    # [beg_ms, end_ms];end != -1 代表偵測到一個句尾
                    if isinstance(s, (list, tuple)) and len(s) == 2 and s[1] != -1:
                        seg_ended = True
        except Exception:
            pass  # VAD 掛掉不致命,最壞情況整段到 end 才定稿

        # 2) 即時預覽解碼
        try:
            pf_res = await _funasr_generate(
                _pf_model, chunk, pf_cache, is_final,
                chunk_size=PF_CHUNK,
                encoder_chunk_look_back=ENC_LOOKBACK,
                decoder_chunk_look_back=DEC_LOOKBACK,
            )
            if pf_res and pf_res[0].get("text"):
                cur_preview += pf_res[0]["text"]
        except Exception as e:
            await send({"type": "error", "detail": f"preview: {e}"})

        # 3) 講太久沒停 → 安全切段
        if seg_dur >= MAX_SEG_SEC:
            seg_ended = True

        # 4) 正常串流途中偵測到句尾就收段(is_final 的收段在外層統一處理)
        if seg_ended and not is_final:
            close_segment()

        await push_partial()

    def reset_all():
        nonlocal pf_cache, vad_cache, audio_buf, seg_samples, seg_dur, cur_preview
        pf_cache = {}
        vad_cache = {}
        audio_buf = np.zeros(0, dtype=np.float32)
        seg_samples = []
        seg_dur = 0.0
        cur_preview = ""
        segments.clear()

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            # --- 控制訊息 (text frame) ---
            if msg.get("text") is not None:
                try:
                    ctrl = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue

                t = ctrl.get("type")
                if t == "end":
                    # flush 殘餘音訊 + 讓預覽吐出尾巴
                    if audio_buf.size > 0:
                        await process_chunk(audio_buf, is_final=True)
                        audio_buf = np.zeros(0, dtype=np.float32)
                    # 收掉最後一句 → 排定稿
                    close_segment()
                    # 等所有句子都定稿完成再回 final
                    await fin_queue.join()
                    await send({"type": "final", "text": _to_tw("".join(segments))})
                    reset_all()
                elif t == "reset":
                    reset_all()
                continue

            # --- 音訊資料 (binary frame) = PCM16 LE mono 16k ---
            data = msg.get("bytes")
            if not data:
                continue
            seg = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if seg.size == 0:
                continue

            audio_buf = np.concatenate([audio_buf, seg])
            # 湊滿一格(600ms)就處理一次
            while audio_buf.size >= CHUNK_STRIDE:
                chunk = audio_buf[:CHUNK_STRIDE]
                audio_buf = audio_buf[CHUNK_STRIDE:]
                await process_chunk(chunk, is_final=False)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await send({"type": "error", "detail": str(e)})
    finally:
        # 收掉背景 finalizer
        try:
            fin_queue.put_nowait(None)
            await asyncio.wait_for(fin_task, timeout=5)
        except Exception:
            fin_task.cancel()


if __name__ == "__main__":
    import uvicorn

    # 單一 worker:本地模型只載一份,連線靠 async 併發;定稿併發交給 vllm serve
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8005")), workers=1)