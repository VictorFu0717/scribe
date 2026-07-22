"""說話者線上分群(純 numpy,不碰 GPU/torch,方便測試)。

用法:每段定稿時算出該段的說話者向量(192 維, CAM++/ERes2NetV2),丟進 assign():
    - 與已知語者中心的最大 cosine 相似度 >= threshold → 判為同一人,並更新該中心
    - 否則 → 新語者
回傳標籤如「說話者1」「說話者2」。中心以 running mean 累積、每次重新正規化。

門檻校準(CAM++):同語者 ~0.67+、語音 vs 雜訊 ~0;不同真人通常 <0.4 → 0.5 是穩健預設。
"""

from __future__ import annotations

import numpy as np


class SpeakerClusterer:
    def __init__(self, threshold: float = 0.5, prefix: str = "說話者"):
        self.threshold = threshold
        self.prefix = prefix
        self._centroids: list[np.ndarray] = []   # 每個語者的正規化中心向量
        self._counts: list[int] = []

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).ravel()
        return v / (np.linalg.norm(v) + 1e-9)

    def assign(self, embedding: np.ndarray) -> str:
        """回傳語者標籤;會就地更新分群狀態。"""
        e = self._norm(embedding)
        if self._centroids:
            sims = [float(np.dot(e, c)) for c in self._centroids]
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                n = self._counts[best]
                merged = (self._centroids[best] * n + e) / (n + 1)
                self._centroids[best] = self._norm(merged)
                self._counts[best] += 1
                return f"{self.prefix}{best + 1}"
        self._centroids.append(e)
        self._counts.append(1)
        return f"{self.prefix}{len(self._centroids)}"

    @property
    def num_speakers(self) -> int:
        return len(self._centroids)
