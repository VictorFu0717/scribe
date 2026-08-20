"""串流語者分離:邊錄邊算,依「語者轉換」切段(即時路徑的層次①)。

與 upload 的 A2 差別:不能等錄完。作法是
  ① 每收滿一個 10 秒視窗就跑 pyannote 前端(分段+聲紋)—— 實測與離線**逐位元相同**,
     零品質損失,代價只有約 10 秒延遲(視窗要滑過去才知道那一刻是誰在講)。
  ② 累積 segmentations + embeddings(約 36MB/小時),**音訊用完即丟**(只留滾動緩衝),
     所以不需要保存錄音 —— 這兩樣都還原不成可聽的語音。
  ③ 每隔幾秒用「累積到目前為止的全部」重跑**全域**分群(實測只要 0.01 秒),
     所以不必維護線上增量狀態,而且能回頭修正先前判錯的標籤。
  ④ 只吐出「已定案」的段(結束時間早於 now - latency)給 ASR。

已知限制:開頭一分鐘分群還不穩時,可能把兩個人塞進同一段(該切沒切),那段文字就混了 —
事後無法補救(除非重跑 ASR)。反之「該合沒合」可以靠合併相鄰同人的文字補回。
"""

from __future__ import annotations

import asyncio

import numpy as np

from app import config
from app.diarize import build_spans

SR = 16000


class StreamingDiarizer:
    """單一連線的串流語者分離狀態機。"""

    def __init__(self, pipe, latency_sec: float, recluster_sec: float):
        self._pipe = pipe
        self.latency = latency_sec
        self.recluster_every = recluster_sec

        spec = pipe._segmentation.model.specifications
        self.win = float(spec.duration)                     # 10s
        self.step = float(pipe.segmentation_step) * self.win  # 1s

        self._buf = np.zeros(0, np.float32)   # 滾動音訊緩衝
        self._buf_start_ms = 0                # 緩衝第一個 sample 的絕對時間
        self._total_ms = 0.0                  # 已收到的音訊總長
        self._segs: list = []                 # 每個視窗的分段輸出
        self._embs: list = []                 # 每個視窗的聲紋
        self._n_win = 0                       # 已算完的視窗數
        self._emitted_ms = 0.0                # 已吐出到哪裡
        self._emitted: list = []              # [{seq,start_ms,end_ms,speaker}]
        self._last_recluster = 0.0
        self._timeline: list = []
        self._spans: list = []

    # ---------------------------------------------------------------- 內部
    def _window_audio(self, i: int):
        """第 i 個視窗 [i*step, i*step+win) 的音訊;緩衝不夠則回 None。"""
        b = int(i * self.step * 1000)
        e = b + int(self.win * 1000)
        if e > self._total_ms:
            return None
        s0 = int((b - self._buf_start_ms) * SR / 1000)
        s1 = s0 + int(self.win * SR)
        if s0 < 0 or s1 > self._buf.size:
            return None
        return self._buf[s0:s1]

    def _front_end(self, wav: np.ndarray):
        """①:單一視窗的分段 + 聲紋(與離線逐位元相同)。"""
        import torch
        from pyannote.core import SlidingWindow, SlidingWindowFeature

        p = self._pipe
        cf = {"waveform": torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0),
              "sample_rate": SR, "uri": "w"}
        one = p._segmentation(cf)
        d = one.data if one.data.ndim == 2 else one.data[0]
        sw1 = SlidingWindowFeature(d[np.newaxis],
                                   SlidingWindow(duration=self.win, step=self.step, start=0.0))
        emb = p.get_embeddings(cf, sw1, exclude_overlap=p.embedding_exclude_overlap,
                              hook=p.setup_hook(cf, hook=None))
        return d, emb[0]

    def _recluster(self):
        """③:用累積到目前為止的全部,重跑全域分群 → 語者時間軸。"""
        import numpy as _np
        from pyannote.core import SlidingWindow, SlidingWindowFeature

        p = self._pipe
        if len(self._segs) < 2:
            return []
        sw = SlidingWindow(duration=self.win, step=self.step, start=0.0)
        seg = SlidingWindowFeature(_np.stack(self._segs), sw)
        emb = _np.stack(self._embs)
        cnt = p.speaker_count(seg, p._segmentation.model.receptive_field, warm_up=(0.0, 0.0))
        hard, _, _ = p.clustering(embeddings=emb, segmentations=seg,
                                  num_clusters=None, min_clusters=1, max_clusters=20)
        cnt.data = _np.minimum(cnt.data, 20).astype(_np.int8)
        hard[_np.sum(seg.data, axis=1) == 0] = -2
        ann = p.to_annotation(p.reconstruct(seg, hard, cnt), min_duration_on=0.0,
                              min_duration_off=p.segmentation.min_duration_off)
        return [(float(t.start), float(t.end), str(s))
                for t, _, s in ann.itertracks(yield_label=True)]

    def _slice(self, b_ms: float, e_ms: float):
        s0 = int((b_ms - self._buf_start_ms) * SR / 1000)
        s1 = int((e_ms - self._buf_start_ms) * SR / 1000)
        return self._buf[max(0, s0):max(0, s1)]

    def _trim(self):
        """丟掉不會再用到的音訊:已吐出、且不在任何未算視窗需求範圍內的部分。"""
        need_ms = min(self._emitted_ms, self._n_win * self.step * 1000)
        keep_from = max(0.0, need_ms - 2000)          # 留 2 秒餘裕給 padding
        drop = int((keep_from - self._buf_start_ms) * SR / 1000)
        if drop > SR:                                  # 至少累積 1 秒才值得搬移
            self._buf = self._buf[drop:]
            self._buf_start_ms += drop * 1000 / SR

    # ---------------------------------------------------------------- 對外
    async def push(self, chunk: np.ndarray):
        """餵音訊。回傳 (可送 ASR 的新段, 既有段的標籤更新)。"""
        if chunk is not None and chunk.size:
            self._buf = np.concatenate([self._buf, chunk.astype(np.float32)])
            self._total_ms += chunk.size * 1000 / SR

        loop = asyncio.get_running_loop()
        while True:                                    # ① 補算所有已收滿的視窗
            wav = self._window_audio(self._n_win)
            if wav is None:
                break
            d, e = await loop.run_in_executor(None, self._front_end, wav)
            self._segs.append(d)
            self._embs.append(e)
            self._n_win += 1

        relabels: dict = {}
        settle_ms = self._total_ms - self.latency * 1000
        if (self._total_ms - self._last_recluster >= self.recluster_every * 1000
                and self._n_win >= 2):
            self._last_recluster = self._total_ms
            self._timeline = await loop.run_in_executor(None, self._recluster)
            # 切段與標籤必須來自**同一份** spans,否則兩邊各自編號會對不起來
            self._spans = build_spans(self._timeline, config.A2_MERGE_GAP,
                                      config.A2_MIN_DUR, config.SPK_PREFIX)
            for row in self._emitted:                  # ③ 回頭修正既有標籤
                lab = self._label_of(row["start_ms"], row["end_ms"])
                if lab and lab != row["speaker"]:
                    row["speaker"] = lab
                    relabels[row["seq"]] = lab

        out = []
        if self._spans:                                # ④ 只吐已定案的段
            for b, e, lab in self._spans:
                # 每次重分群都會重新推導 spans,邊界會變。若一個段的起點落在「已吐出」
                # 範圍內,**要截斷後吐出剩下的部分**,不能整段跳過 —— 否則重算把兩段併長時,
                # 中間那截語音會永遠消失(實測會漏掉一半的語音、DER 50%)。
                b = max(b, self._emitted_ms)
                if e > settle_ms or (e - b) < config.A2_MIN_DUR * 1000:
                    continue
                audio = self._slice(b, e)
                if audio.size:
                    row = {"seq": len(self._emitted), "start_ms": int(b), "end_ms": int(e),
                           "speaker": lab, "audio": audio}
                    self._emitted.append({k: row[k] for k in
                                          ("seq", "start_ms", "end_ms", "speaker")})
                    out.append(row)
                    self._emitted_ms = e
        self._trim()
        return out, relabels

    async def flush(self):
        """收尾:剩下的音訊補算視窗,再把尾巴全部吐出。"""
        if self._buf.size and self._total_ms > self._emitted_ms:
            pad = int(self.win * SR)                   # 補零讓最後一個視窗算得出來
            self._buf = np.concatenate([self._buf, np.zeros(pad, np.float32)])
            self._total_ms += pad * 1000 / SR
            self.latency = 0.0
            self._last_recluster = -1e9
            out, relabels = await self.push(np.zeros(0, np.float32))
            return out, relabels
        return [], {}

    def label_for(self, b_ms, e_ms):
        return self._label_of(b_ms, e_ms)

    def _label_of(self, b_ms, e_ms):
        """目前 spans 中與 [b,e] 重疊最久的語者(標籤修正用;標籤已是 說話者N)。"""
        acc: dict = {}
        for b, e, lab in self._spans:
            ov = min(e_ms, e) - max(b_ms, b)
            if ov > 0:
                acc[lab] = acc.get(lab, 0.0) + ov
        return max(acc, key=acc.get) if acc else None
