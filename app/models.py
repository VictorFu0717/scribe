"""本地 ASR / 語者 模型狀態與存取。

所有連線共用一份模型;GPU 呼叫以一把鎖序列化確保 thread 安全。
真正吃資源的定稿外包給 vLLM(async HTTP),所以序列化本地小模型不是瓶頸。
"""

from __future__ import annotations

import asyncio
import io
import re

import numpy as np
import soundfile as sf
from openai import AsyncOpenAI

from app import config

_pf = None            # 即時預覽 (paraformer-streaming)
_vad = None           # 斷句 (fsmn-vad)
_oai: AsyncOpenAI | None = None   # 定稿 client (打 vllm serve)
_cc = None            # OpenCC s2twp
_spk = None           # 語者向量模型 (lazy-load)
_lock = asyncio.Lock()
_spk_load_lock = asyncio.Lock()


async def startup():
    """載入預覽 / VAD 模型 + OpenCC + 定稿 client。"""
    global _pf, _vad, _oai, _cc
    from funasr import AutoModel

    print(f"[startup] loading FunASR preview={config.STREAM_MODEL} vad={config.VAD_MODEL} "
          f"(hub={config.FUNASR_HUB}, device={config.DEVICE}) ...")
    _pf = AutoModel(model=config.STREAM_MODEL, hub=config.FUNASR_HUB,
                    device=config.DEVICE, disable_update=True)
    _vad = AutoModel(model=config.VAD_MODEL, hub=config.FUNASR_HUB,
                     device=config.DEVICE, disable_update=True)
    _oai = AsyncOpenAI(base_url=config.VLLM_BASE_URL, api_key=config.VLLM_API_KEY)
    if config.ASR_TW:
        try:
            import opencc
            _cc = opencc.OpenCC("s2twp")
            print("[startup] OpenCC s2twp 已啟用 (簡體 -> 繁體台灣用語)")
        except Exception as e:
            print(f"[startup] OpenCC 不可用,略過繁簡轉換: {e}")
    print(f"[startup] ready. finalize -> {config.VLLM_BASE_URL} ({config.QWEN_MODEL})")


async def shutdown():
    global _pf, _vad, _oai, _cc, _spk
    _pf = _vad = _oai = _cc = _spk = None
    print("[shutdown] released.")


def is_loaded() -> bool:
    return _pf is not None


def to_tw(text: str) -> str:
    """簡體 -> 繁體(台灣用語);未啟用或失敗則原樣回傳。"""
    if not text or _cc is None:
        return text
    try:
        return _cc.convert(text)
    except Exception:
        return text


def clean_qwen(raw: str) -> str:
    """Qwen3-ASR 經 vLLM 回傳夾帶的模板標記(如 language Chinese<asr_text>...)剝乾淨。"""
    if not raw:
        return ""
    if "<asr_text>" in raw:
        raw = raw.split("<asr_text>", 1)[1]
    raw = re.sub(r"<[^>]*>", "", raw)
    return raw.strip()


async def _generate(model, chunk, cache, is_final, **kw):
    loop = asyncio.get_running_loop()
    async with _lock:
        return await loop.run_in_executor(
            None, lambda: model.generate(input=chunk, cache=cache, is_final=is_final, **kw))


async def vad(chunk, cache, is_final):
    return await _generate(_vad, chunk, cache, is_final, chunk_size=config.CHUNK_MS)


async def preview(chunk, cache, is_final):
    return await _generate(
        _pf, chunk, cache, is_final, chunk_size=config.PF_CHUNK,
        encoder_chunk_look_back=config.ENC_LOOKBACK,
        decoder_chunk_look_back=config.DEC_LOOKBACK)


async def finalize_qwen(seg: np.ndarray) -> str:
    """一段音訊 → Qwen3-ASR 高準定稿(async,可併發)。"""
    bio = io.BytesIO()
    sf.write(bio, seg, config.SAMPLE_RATE, format="WAV", subtype="PCM_16")
    kwargs = {"model": config.QWEN_MODEL, "file": ("seg.wav", bio.getvalue(), "audio/wav")}
    if config.ASR_LANG:
        kwargs["language"] = config.ASR_LANG
    resp = await _oai.audio.transcriptions.create(**kwargs)
    return clean_qwen(getattr(resp, "text", "") or "")


async def get_spk_model():
    """lazy-load 語者向量模型(第一次啟用說話者辨識才載入)。"""
    global _spk
    if _spk is None:
        async with _spk_load_lock:
            if _spk is None:
                from funasr import AutoModel
                loop = asyncio.get_running_loop()
                print(f"[diarize] loading speaker model {config.SPK_MODEL} (hub={config.SPK_HUB}) ...")
                _spk = await loop.run_in_executor(None, lambda: AutoModel(
                    model=config.SPK_MODEL, hub=config.SPK_HUB,
                    device=config.DEVICE, disable_update=True))
                print("[diarize] speaker model ready.")
    return _spk


async def spk_embed(audio: np.ndarray) -> np.ndarray:
    """一段音訊的語者向量(192 維);與其他 FunASR 呼叫共用同一把鎖。"""
    model = await get_spk_model()
    loop = asyncio.get_running_loop()
    async with _lock:
        res = await loop.run_in_executor(
            None, lambda: model.generate(input=audio.astype(np.float32)))
    return res[0]["spk_embedding"].detach().cpu().numpy().ravel()
