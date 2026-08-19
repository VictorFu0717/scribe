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

from app import config, db, models, pyannote_diar, rag
from app.auth import get_current_user
from app.diarize import assign_all, build_spans, labels_from_timeline

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


def _quietest(audio: np.ndarray, b_ms: int, e_ms: int) -> int:
    """在 [b,e] 的中段找最安靜的 100ms,當作切點(比盲切不容易切在字中間)。"""
    lo, hi = b_ms + (e_ms - b_ms) // 4, e_ms - (e_ms - b_ms) // 4
    win = int(SR * 0.1)
    best, best_r = (lo + hi) // 2, None
    for t in range(lo, max(lo + 1, hi), 100):
        seg = audio[int(t * SR / 1000):int(t * SR / 1000) + win]
        if seg.size == 0:
            continue
        r = float(np.sqrt((seg.astype(np.float64) ** 2).mean()))
        if best_r is None or r < best_r:
            best, best_r = t, r
    return best


def _split_long(audio: np.ndarray, spans: list, max_ms: int) -> list:
    """A2:把超過 max_ms 的發言切開(同一人講很久 → ASR 太長、定稿延遲高)。"""
    out = []
    for b, e, lab in spans:
        stack = [(b, e)]
        while stack:
            s0, e0 = stack.pop(0)
            if e0 - s0 <= max_ms:
                out.append((s0, e0, lab))
            else:
                cut = _quietest(audio, s0, e0)
                stack = [(s0, cut), (cut, e0)] + stack
    return sorted(out, key=lambda x: x[0])


async def _process(mid: str, user: str, audio: np.ndarray, diarize: bool):
    """背景:VAD → 併發定稿 → (可選)語者分群 → 寫入儲存 → 建向量索引。"""
    try:
        # 語者時間軸要先算:A2 用它來「切段」,A1 只用它「貼標籤」
        timeline = await pyannote_diar.diarize(audio) if (diarize and pyannote_diar.enabled()) else None

        preset_spk = None
        if timeline and config.DIARIZE_SEGMENT == "speaker":
            # A2:依「語者轉換」切段 —— 每段天生單一語者,不必再事後貼標籤
            spans = _split_long(audio, build_spans(timeline, config.A2_MERGE_GAP,
                                                   config.A2_MIN_DUR, config.SPK_PREFIX),
                                MAX_SEG_MS)
            if spans:
                segs = [[b, e] for b, e, _ in spans]
                preset_spk = [l for _, _, l in spans]
                print(f"[upload] {mid} 切段: 語者轉換(A2), {len(segs)} 段 / "
                      f"{len(set(preset_spk))} 位")
        if preset_spk is None:             # 現況 / A2 不可用時的退路:依停頓切
            segs = _cap_segments(await models.vad_offline(audio), MAX_SEG_MS)
            if not segs:                   # VAD 沒切到 → 整段當一段
                segs = [[0, int(audio.size / SR * 1000)]]

        def _clip(i):
            """取段落音訊。A2 的切點正好落在語音邊界上,補一點 lead-in 給 ASR;
            但**夾限在鄰段之外**,免得把隔壁那個人的聲音也收進來。"""
            b, e = segs[i]
            pad = int(config.SEG_PAD_MS) if preset_spk is not None else 0
            if pad:
                lo = segs[i - 1][1] if i > 0 else 0
                hi = segs[i + 1][0] if i + 1 < len(segs) else int(audio.size / SR * 1000)
                b, e = max(b - pad, lo, 0), min(e + pad, hi)
            return audio[int(b * SR / 1000):int(e * SR / 1000)]

        clips = [_clip(i) for i in range(len(segs))]

        # 定稿:併發(vLLM 會 continuous batching),用 semaphore 控上限
        sem = asyncio.Semaphore(CONCURRENCY)

        fails: list = []

        async def fin(clip):
            if not models.is_speech_segment(clip):   # 非語音段不送定稿(會被幻覺成假句子)
                return ""
            async with sem:
                try:
                    return await models.finalize_qwen(clip)
                except Exception as e:
                    fails.append(str(e))             # 逾時/失敗:略過該段,不讓整批上傳失敗
                    return ""

        texts = await asyncio.gather(*[fin(c) for c in clips])
        if fails:   # 別靜默:定稿服務掛掉時,原本只會看到「0 段」而查不出原因
            print(f"[upload] {mid} 定稿失敗 {len(fails)}/{len(clips)} 段,例:{fails[0][:120]}")

        # 說話者:A2 已在切段時決定;A1 用時間軸貼標籤;都不可用才退回 campplus
        # (每個 VAD 段一個聲紋 —— 段內混多人時必然失準)。
        speakers = [None] * len(segs)
        if diarize:
            idxs = [i for i, c in enumerate(clips) if c.size and texts[i].strip()]
            labels = None
            if preset_spk is not None:
                labels = [preset_spk[i] for i in idxs]      # A2:切段時就已經知道是誰
            elif timeline:
                # A1:分段仍由 VAD 決定,只把時間軸上「重疊最久」的語者貼回每一段
                labels = labels_from_timeline(timeline, [segs[i] for i in idxs],
                                              config.SPK_PREFIX)
                print(f"[upload] {mid} 語者分離: pyannote/A1, {len({l for l in labels if l})} 位")
            if labels is None:
                embs = [await models.spk_embed(clips[i]) for i in idxs]   # spk_embed 內部有鎖
                durs = [(segs[i][1] - segs[i][0]) / 1000.0 for i in idxs]
                labels, _, _ = assign_all(embs, durs, config.SPK_THRESHOLD,
                                          config.SPK_PREFIX, config.SPK_MIN_NEW_SEC)
                print(f"[upload] {mid} 語者分離: campplus, {len({l for l in labels if l})} 位")
            for i, label in zip(idxs, labels):
                speakers[i] = label

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
