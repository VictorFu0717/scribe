"""整段錄音檔上傳轉錄(離線批次)。

App 上傳完整錄音 → server VAD 切段 → 每段送 Qwen3-ASR 定稿(併發)→(可選)說話者辨識
→ 存入該會議。長檔案處理較久,故用背景工作:上傳後立即回 status=transcribing,
App 輪詢 GET /meetings/{id} 直到 status=ready,再 GET .../transcript。

端點:
    POST /meetings/{id}/audio   multipart: file=<音檔>, diarization=<bool>
    → {"id":..,"status":"transcribing"}
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile

import librosa
import numpy as np
from fastapi import (APIRouter, BackgroundTasks, Depends, File, Form,
                     HTTPException, UploadFile)

from app import config, db, models, rag
from app.auth import get_current_user
from app.diarize import SpeakerClusterer

router = APIRouter(tags=["upload"])

MAX_SEG_MS = int(float(os.getenv("UPLOAD_MAX_SEG_SEC", "30")) * 1000)   # 過長 VAD 段再切
CONCURRENCY = int(os.getenv("UPLOAD_CONCURRENCY", "8"))                 # 同時打 Qwen3-ASR 上限
SR = config.SAMPLE_RATE


def _load_audio(raw: bytes, suffix: str = "") -> np.ndarray:
    """任意音檔 bytes → 16k mono float32(librosa 會自動 resample/降混)。

    兩段式:先在記憶體解(soundfile 直接吃 wav/flac/ogg/mp3,免磁碟 IO);
    失敗才落地成暫存檔再解。**手機錄的 m4a/aac 只有後者能解** —— soundfile 不支援
    m4a,而 librosa 用來救場的 audioread/ffmpeg 後備是 spawn `ffmpeg -i <路徑>`,
    只吃檔案路徑、吃不了 BytesIO,所以光裝 ffmpeg 而不落地是解不開的。
    """
    try:
        audio, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=suffix or ".bin") as f:
            f.write(raw)
            f.flush()
            audio, _ = librosa.load(f.name, sr=SR, mono=True)   # 需系統有 ffmpeg
    return audio.astype(np.float32)


def _cap_segments(segs: list, max_ms: int) -> list:
    """把 VAD 段整理:過濾非法、把 > max_ms 的段再切成數段。"""
    out = []
    for s in segs:
        if not (isinstance(s, (list, tuple)) and len(s) == 2):
            continue
        b, e = int(s[0]), int(s[1])
        if b < 0 or e < 0 or e <= b:
            continue
        while e - b > max_ms:
            out.append([b, b + max_ms]); b += max_ms
        out.append([b, e])
    return out


async def _process(mid: str, user: str, audio: np.ndarray, diarize: bool):
    """背景:VAD → 併發定稿 → (可選)語者分群 → 寫入儲存 → 建向量索引。"""
    try:
        raw_segs = await models.vad_offline(audio)
        segs = _cap_segments(raw_segs, MAX_SEG_MS)
        if not segs:                       # VAD 沒切到 → 整段當一段
            segs = [[0, int(audio.size / SR * 1000)]]

        clips = [audio[int(b * SR / 1000):int(e * SR / 1000)] for b, e in segs]

        # 定稿:併發(vLLM 會 continuous batching),用 semaphore 控上限
        sem = asyncio.Semaphore(CONCURRENCY)

        async def fin(clip):
            if not models.is_speech_segment(clip):   # 非語音段不送定稿(會被幻覺成假句子)
                return ""
            async with sem:
                try:
                    return await models.finalize_qwen(clip)
                except Exception:
                    return ""                        # 逾時/失敗:略過該段,不讓整批上傳失敗

        texts = await asyncio.gather(*[fin(c) for c in clips])

        # 說話者:依時間順序分群(assign 會更新中心,需序列化)
        speakers = [None] * len(segs)
        if diarize:
            cl = SpeakerClusterer(config.SPK_THRESHOLD, config.SPK_PREFIX)
            for i, clip in enumerate(clips):
                if clip.size and texts[i].strip():
                    speakers[i] = cl.assign(await models.spk_embed(clip))

        result = [{"text": models.to_tw(texts[i]), "speaker": speakers[i],
                   "start_ms": int(segs[i][0]), "end_ms": int(segs[i][1])}
                  for i in range(len(segs)) if texts[i].strip()]

        await db.save_transcript(mid, result)
        await db.set_status(mid, "ready", int(audio.size / SR))
        try:
            await rag.index_meeting(user, mid)   # ⑥ 建向量索引
        except Exception as e:
            print(f"[upload] {mid} 索引失敗: {e}")
        print(f"[upload] {mid} done: {len(result)} 段")
    except Exception as e:
        await db.set_status(mid, "error")
        print(f"[upload] {mid} 失敗: {e}")


@router.post("/meetings/{mid}/audio")
async def upload_audio(mid: str, background_tasks: BackgroundTasks,
                       file: UploadFile = File(...),
                       diarization: bool = Form(False),
                       user: str = Depends(get_current_user)):
    if await db.get_meeting(user, mid) is None:
        raise HTTPException(404, "meeting not found")

    raw = await file.read()
    src = f"{file.filename!r} type={file.content_type} {len(raw)} bytes"
    if not raw:
        print(f"[upload] {mid} 400 empty file: {src}")
        raise HTTPException(400, f"empty file ({src})")
    suffix = os.path.splitext(file.filename or "")[1][:10]
    loop = asyncio.get_running_loop()
    try:
        audio = await loop.run_in_executor(None, _load_audio, raw, suffix)
    except Exception as e:
        print(f"[upload] {mid} 400 decode fail: {src} -> {e}")
        raise HTTPException(400, f"cannot decode audio ({src});m4a/aac 需系統有 ffmpeg: {e}")
    if audio.size == 0:
        print(f"[upload] {mid} 400 decoded empty: {src}")
        raise HTTPException(400, f"decoded audio is empty ({src})")

    await db.set_status(mid, "transcribing")
    background_tasks.add_task(_process, mid, user, audio, diarization)
    return {"id": mid, "status": "transcribing", "duration_sec": int(audio.size / SR)}
