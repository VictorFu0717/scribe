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
                     device=config.DEVICE, disable_update=True,
                     max_end_silence_time=config.VAD_MAX_END_SILENCE_MS,
                     max_single_segment_time=int(config.VAD_MAX_SEGMENT_SEC * 1000))
    # timeout 必設:非語音/靜音段會讓 Qwen3-ASR 失控生成,SDK 預設 600s 會把定稿佇列堵死。
    # max_retries=0:卡住的請求重試只是再等一次,對 localhost 沒有意義。
    _oai = AsyncOpenAI(base_url=config.VLLM_BASE_URL, api_key=config.VLLM_API_KEY,
                       timeout=config.FINALIZE_TIMEOUT, max_retries=0)
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


_WORDCH = re.compile(r"[0-9A-Za-z\u00c0-\u024f]")   # 拉丁字母/數字(含重音字母)


def join_text(prev: str, piece: str) -> str:
    """把新片段接到已累積文字後面,英文之間補空白。

    paraformer 串流「單一片段內」是有空白的('your country'),但**片段之間從來沒有**,
    所以直接 prev + piece 會把英文黏成一團:
        'and so'+'my'+'low'+'ask'+'not' -> 'and somylowasknot'
    中文不需要空白('我們'+'開會' -> '我們開會' 是對的),因此只在**兩側都是英數**時
    才補一個空白;中英交界(如 '我們用'+'GPU')維持不加,那是中文的習慣寫法。
    """
    if not piece:
        return prev
    if not prev:
        return piece
    if _WORDCH.match(prev[-1]) and _WORDCH.match(piece[0]):
        return prev + " " + piece
    return prev + piece


def join_all(parts) -> str:
    """依 join_text 規則串接一串片段。"""
    out = ""
    for p in parts:
        out = join_text(out, p)
    return out


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


async def vad_offline(audio: np.ndarray) -> list:
    """離線 VAD:整段音訊 → [[beg_ms, end_ms], ...](檔案上傳轉錄用)。"""
    loop = asyncio.get_running_loop()
    async with _lock:
        res = await loop.run_in_executor(
            None, lambda: _vad.generate(input=audio.astype(np.float32)))
    return res[0].get("value", []) if res else []


async def preview(chunk, cache, is_final):
    return await _generate(
        _pf, chunk, cache, is_final, chunk_size=config.PF_CHUNK,
        encoder_chunk_look_back=config.ENC_LOOKBACK,
        decoder_chunk_look_back=config.DEC_LOOKBACK)


def frame_peak_rms(a: np.ndarray, frame_ms: float = 30.0) -> float:
    """整段音訊中「最大的 30ms 框 RMS」。

    比整段平均 RMS 穩健:一段 15s 裡只有 1s 在講話時,平均 RMS 會被靜音稀釋到很低,
    用平均值當門檻會誤刪真語音;取最大框則只問「有沒有任何一刻夠大聲」。
    """
    if a is None or a.size == 0:
        return 0.0
    n = max(1, int(config.SAMPLE_RATE * frame_ms / 1000))
    x = a.astype(np.float64)
    if x.size < n:
        return float(np.sqrt(np.mean(x ** 2)))
    k = x.size // n
    fr = x[:k * n].reshape(k, n)
    return float(np.sqrt((fr ** 2).mean(axis=1)).max())


def is_speech_segment(seg: np.ndarray) -> bool:
    """這段音訊值不值得送定稿?擋掉碎段與近乎無聲的段(理由見 config 的守門區)。"""
    if seg is None or seg.size == 0:
        return False
    if seg.size / config.SAMPLE_RATE * 1000 < config.MIN_SEG_MS:
        return False
    return frame_peak_rms(seg) >= config.MIN_SEG_RMS


def normalize_segment(seg: np.ndarray) -> np.ndarray:
    """把小聲的段落放大到目標 RMS(只放大不壓低,增益有上限)。SEG_NORM_RMS=0 關閉。"""
    if config.SEG_NORM_RMS <= 0 or seg is None or seg.size == 0:
        return seg
    rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
    if rms <= 1e-6:
        return seg
    gain = min(config.SEG_NORM_RMS / rms, config.SEG_NORM_MAX_GAIN)
    if gain <= 1.0:
        return seg                       # 已經夠大聲:不動(壓低救不了削波)
    out = seg.astype(np.float32) * gain
    peak = float(np.abs(out).max())
    if peak > 0.99:                      # 防削波
        out *= 0.99 / peak
    return out


async def finalize_qwen(seg: np.ndarray) -> str:
    """一段音訊 → Qwen3-ASR 高準定稿(async,可併發)。送出前做音量正規化。"""
    seg = normalize_segment(seg)
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
