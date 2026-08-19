"""pyannote 語者分離(目前用於整檔上傳路徑)。

與現行 campplus 做法的差別:現行是「每個 VAD 段抽一個聲紋再分群」,一段若混了多人
(實測 84% 的段混了平均 3 人)聲紋就是混合的,分群必然失準。pyannote 改在**幀層級**
(每 16ms)判定誰在說話,只從「單人、無重疊」的幀抽聲紋,再做全域分群。

實測(AliMeeting 5 場,重疊率 2%~64%,同樣輸出在 VAD 段上的公平比較):
DER campplus 44.1% → pyannote 37.2%;搭配 DIARIZE_SEGMENT=speaker(依語者轉換切段)
可再降到 20.8%。**最穩定的優勢是語者人數每場都對**(campplus 會塌成 1~2 位或碎成 8~9 位)。
速度相當(1 小時錄音約 19 秒)。

設計要點:
- **lazy-load**:不開語者辨識就完全不載入(零 VRAM),與 campplus 同樣策略。
- **獨立 semaphore、不共用 models._lock**:一個 1 小時的上傳會佔用 GPU 約 20 秒,
  若跟 FunASR 共用那把鎖,期間所有即時串流連線的 VAD/預覽都會卡死。
- **失敗一律回 None**:呼叫端自動退回 campplus,不讓語者分離拖垮整個上傳。
"""

from __future__ import annotations

import asyncio

import numpy as np

from app import config

_pipeline = None
_load_lock = asyncio.Lock()
_gpu_sem = asyncio.Semaphore(1)      # 同時最多一個 pyannote 工作
_unavailable = False                 # 載入失敗過就不再重試(避免每次上傳都卡一次)


def enabled() -> bool:
    return config.DIARIZE_BACKEND in ("auto", "pyannote") and not _unavailable


def _build():
    """組裝與官方 speaker-diarization-3.1 等價的 pipeline。

    不用 Pipeline.from_pretrained("pyannote/speaker-diarization-3.1"):pyannote.audio 4.x
    會連帶去抓 `speaker-diarization-community-1` 的 PLDA(**另一個 gated repo**)。
    這裡直接指定 3.1 的兩個元件與官方超參數;AgglomerativeClustering 不使用 PLDA
    (已核對原始碼),故把 get_plda 短路掉。
    """
    import torch
    import pyannote.audio.pipelines.speaker_diarization as sd

    _orig = sd.get_plda
    sd.get_plda = lambda plda, **kw: None if plda is None else _orig(plda, **kw)

    from pyannote.audio.pipelines import SpeakerDiarization

    pipe = SpeakerDiarization(
        legacy=True,                       # 直接輸出 Annotation
        segmentation=config.PYANNOTE_SEG,
        embedding=config.PYANNOTE_EMB,
        clustering="AgglomerativeClustering", plda=None,
        segmentation_batch_size=32, embedding_batch_size=32,
        embedding_exclude_overlap=True,    # 只從「沒有重疊」的幀抽聲紋 —— 品質關鍵
        token=config.HF_TOKEN,
    )
    pipe.instantiate({
        "clustering": {"method": "centroid",
                       "min_cluster_size": config.PYANNOTE_MIN_CLUSTER,
                       "threshold": config.PYANNOTE_THRESHOLD},
        "segmentation": {"min_duration_off": 0.0},
    })
    dev = "cuda" if (config.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    return pipe.to(torch.device(dev))


async def _get():
    global _pipeline, _unavailable
    if _pipeline is None and not _unavailable:
        async with _load_lock:
            if _pipeline is None and not _unavailable:
                loop = asyncio.get_running_loop()
                try:
                    print("[pyannote] loading speaker diarization pipeline ...")
                    _pipeline = await loop.run_in_executor(None, _build)
                    print("[pyannote] ready.")
                except Exception as e:
                    _unavailable = True
                    print(f"[pyannote] 不可用,語者分離退回 campplus: {e}")
    return _pipeline


async def diarize(audio: np.ndarray, num_speakers: int | None = None):
    """整段音訊 → [(起秒, 訖秒, 標籤)];不可用或失敗時回 None(呼叫端自行退回)。"""
    if not enabled() or audio is None or audio.size == 0:
        return None
    pipe = await _get()
    if pipe is None:
        return None
    import torch

    def _run():
        # 餵記憶體波形,避開 torchcodec(本機 CUDA 版本不合,無法解碼檔案)
        f = {"waveform": torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0),
             "sample_rate": config.SAMPLE_RATE}
        kw = {"num_speakers": num_speakers} if num_speakers else {}
        ann = pipe(f, **kw)
        return [(float(t.start), float(t.end), str(s))
                for t, _, s in ann.itertracks(yield_label=True)]

    loop = asyncio.get_running_loop()
    async with _gpu_sem:
        try:
            return await loop.run_in_executor(None, _run)
        except Exception as e:
            print(f"[pyannote] 分離失敗,退回 campplus: {e}")
            return None


def shutdown():
    global _pipeline
    _pipeline = None
