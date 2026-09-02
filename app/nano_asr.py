"""Fun-ASR-Nano 即時預覽後端(STREAM_BACKEND=nano 時啟用)。

與 paraformer 的根本差別:paraformer 是**原生串流**(有 cache,每次只算新 chunk);
Nano 沒有串流 cache,做法是**每次重解「語音起點到現在」的整個視窗**——
這正是官方 serve_realtime_ws.py 的做法。代價是視窗越長越貴,好處是最後一次
更新已經看過全部音訊,所以串流品質 == 離線品質(實測 MER 13.2% vs 13.3%)。

為什麼要換:paraformer-zh 是純中文模型,英文會爛掉。ASCEND 語料 120 句真人
中英夾雜實測,英文詞召回 34.1% → 72.9%,而純中文只從 6.8% 退到 7.1%。

設計要點(與 pyannote_diar 同一套原則):
- **lazy-load**:STREAM_BACKEND=paraformer 時完全不載入(零顯存)。
- **獨立 semaphore、不共用 models._lock**:那把鎖是給 FunASR 用的,
  借過來會讓 VAD/定稿跟著卡住。
- **失敗一律回 None**:呼叫端自動退回 paraformer,不讓預覽拖垮整條連線。
"""

from __future__ import annotations

import asyncio
import re
import time

import numpy as np
import regex

from app import config

_engine = None
_load_lock = asyncio.Lock()
_gpu_sem = asyncio.Semaphore(1)      # 同時最多一個解碼
_unavailable = False                 # 載入失敗過就不再重試


def enabled() -> bool:
    return config.STREAM_BACKEND == "nano" and not _unavailable


def _clean(text: str) -> str:
    """剝掉 vLLM 輸出夾帶的標記與雜訊(照搬官方 serve_realtime_ws._clean_asr_text)。"""
    text = re.sub(r"<[^>]*>", "", text or "")
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[Ｏ\[\]&＆|｜]", "", text)
    text = re.sub(r"/sil|endofbreak|FFFF", "", text)
    return re.sub(r"\s+", " ", text).strip()


def fix_repetition(text: str, max_ngram: int = 12, max_occ: int = 3,
                   min_len: int = 12) -> tuple[str, bool]:
    """偵測「重複繞圈」的幻覺並截斷,只留一次。

    Nano 是自迴歸 LLM,餵到靜音或雜訊時會反覆吐同一段字。官方即時服務內建這道
    防護就是為此。我們的 audio gate(models.is_speech_segment)已經擋掉數位靜音,
    但預覽是逐 chunk 解碼、還沒經過那道門,所以這裡再擋一次。
    """
    # 官方用 len(text) < max_ngram*2(=24)才跳過,對中文太寬 —— 24 個中文字是一整句話,
    # 20 字的純重複(「好的好的好的…」)會被放行。這裡降到 12。
    # 取捨:誤判的代價只是預覽被截短一次(定稿走 Qwen3 不受影響),
    # 漏判的代價是使用者直接看到一整串鬼打牆,後者明顯比較糟。
    if not text or len(text) < min_len:
        return text, False
    cleaned = regex.sub(r"\p{P}+", "", text)

    def _truncate(rep: str) -> str:
        pos = text.find(rep)
        if pos >= 0:
            end = text.find(rep, pos + len(rep))
            if end >= 0:
                return text[:end + len(rep)]
        return text[:len(text) // 2]

    # ① 整個「詞」被重複(英文常見:hello hello hello)
    m = regex.search(rf"(?<!\S)(?!\d+$)(\w+)(?:\s+\1){{{max_occ - 1},}}(?!\S)",
                     cleaned, regex.IGNORECASE)
    if m:
        return _truncate(m.group(1)), True
    # ② 固定長度的字串被重複(中文常見:好的好的好的)
    for n in range(1, max_ngram):
        m = regex.search(rf"(?=.*\D)(?<!\d)(\S{{{n}}})\1{{{max_occ - 1},}}(?!\d)", cleaned)
        if m:
            return _truncate(m.group(1)), True
    return text, False


def _build():
    from funasr.auto.auto_model_vllm import AutoModelVLLM
    dev = config.DEVICE if config.DEVICE.startswith("cuda") else "cpu"
    return AutoModelVLLM(
        model=config.NANO_MODEL, hub=config.FUNASR_HUB,
        device=dev if dev != "cuda" else "cuda:0",
        dtype="bf16", tensor_parallel_size=1,
        gpu_memory_utilization=config.NANO_GPU_FRAC,
        max_model_len=config.NANO_MAX_LEN,
    )


async def _get():
    global _engine, _unavailable
    if _engine is None and not _unavailable:
        async with _load_lock:
            if _engine is None and not _unavailable:
                loop = asyncio.get_running_loop()
                try:
                    print(f"[nano] loading {config.NANO_MODEL} (in-process vLLM, "
                          f"gpu_frac={config.NANO_GPU_FRAC}) ...")
                    _engine = await loop.run_in_executor(None, _build)
                    print("[nano] ready.")
                except Exception as e:
                    _unavailable = True
                    print(f"[nano] 不可用,即時預覽退回 paraformer: {e}")
    return _engine


async def warmup():
    """啟動時預載(STREAM_BACKEND=nano 才會真的做事)。

    不預載的話,第一條 WebSocket 連線要等 vLLM 起來(數十秒),使用者會以為當掉。
    """
    if not enabled():
        return
    eng = await _get()
    if eng is not None:
        await preview(np.zeros(config.SAMPLE_RATE, np.float32))   # 觸發 CUDA graph / 編譯


async def preview(audio: np.ndarray) -> str | None:
    """重解整個視窗 → 預覽文字。不可用或失敗回 None(呼叫端退回 paraformer)。"""
    if not enabled() or audio is None or audio.size == 0:
        return None
    eng = await _get()
    if eng is None:
        return None
    import torch

    def _run():
        r = eng.generate(
            inputs=[torch.from_numpy(np.ascontiguousarray(audio)).float()],
            language=config.NANO_LANG,
            hotwords=config.NANO_HOTWORDS or None,
            max_new_tokens=config.NANO_MAX_TOKENS,
        )
        return _clean(r[0]["text"] if r else "")

    loop = asyncio.get_running_loop()
    async with _gpu_sem:
        try:
            txt = await loop.run_in_executor(None, _run)
        except Exception as e:
            print(f"[nano] 解碼失敗,本次退回 paraformer: {e}")
            return None
    txt, looped = fix_repetition(txt)
    if looped:
        print("[nano] 偵測到重複繞圈,已截斷")
    return txt


def shutdown():
    global _engine
    _engine = None
