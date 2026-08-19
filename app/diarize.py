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
    def __init__(self, threshold: float = 0.5, prefix: str = "說話者",
                 min_new_sec: float = 0.0):
        self.threshold = threshold
        self.prefix = prefix
        self.min_new_sec = min_new_sec            # 短於此的段不得新增語者(0=不限制)
        self._centroids: list[np.ndarray] = []   # 每個語者的正規化中心向量
        self._counts: list[int] = []

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).ravel()
        return v / (np.linalg.norm(v) + 1e-9)

    def assign(self, embedding: np.ndarray, duration_sec: float | None = None) -> str:
        """回傳語者標籤;會就地更新分群狀態。

        duration_sec:該段的真實長度(不含 lead-in padding)。**短段只能歸入既有語者,
        不得新增語者、也不更新中心** —— 實測 CAM++ 聲紋在短段極不穩定:
            段長 0.3s→同一人 cos 僅 0.18(99% 會被判成不同人)、1.0s→0.37(80%)、
            2.0s→0.56(27%)、3.0s→0.67(0%);而「不同人」始終穩定在 0.07~0.16。
        亦即短段的問題不是「兩人分不開」,而是「同一人自己跟自己都對不上」,
        分布完全重疊、沒有任何門檻救得了 → 只能不讓它產生新語者,否則每個短句
        都會生出一個幽靈語者(使用者實測回報「語者變很多」的成因)。
        """
        e = self._norm(embedding)
        reliable = duration_sec is None or duration_sec >= self.min_new_sec
        if self._centroids:
            sims = [float(np.dot(e, c)) for c in self._centroids]
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                if reliable:                      # 只用可信的段更新中心,免得被短段污染
                    n = self._counts[best]
                    merged = (self._centroids[best] * n + e) / (n + 1)
                    self._centroids[best] = self._norm(merged)
                    self._counts[best] += 1
                return f"{self.prefix}{best + 1}"
            if not reliable:                      # 短段:硬歸入最像的,不新增語者
                return f"{self.prefix}{best + 1}"
        self._centroids.append(e)
        self._counts.append(1)
        return f"{self.prefix}{len(self._centroids)}"

    def load_state(self, centroids, counts) -> None:
        """用外部(重分群)算出的結果覆寫狀態,讓後續線上判定與新標籤一致。"""
        self._centroids = [self._norm(c) for c in centroids]
        self._counts = [max(1, int(n)) for n in counts]

    @property
    def num_speakers(self) -> int:
        return len(self._centroids)


def cluster_offline(embeddings, threshold: float = 0.5, prefix: str = "說話者",
                    n_speakers: int | None = None) -> list[str]:
    """離線全域分群:一次看完所有聲紋再決定分組(整檔上傳用)。

    與線上 assign() 的差別 —— assign 是貪婪、一次定案、不可回頭:第 3 段判錯了,
    後面所有證據都無法回溯修正它,錯誤的中心還會繼續吸收後續段落。離線時整份音訊
    都在手上,沒有理由忍受這個限制,所以改用凝聚式階層分群:全域決定分組、與輸入順序無關。

    **用 centroid linkage(比「群中心」)而非 average linkage(比「群間平均兩兩距離」)**:
    這是實測選出來的。average linkage 在同一個門檻下明顯更嚴格 —— 群一變大,平均兩兩
    相似度就被拉低而過度切分;貪婪版比的是 running-mean 中心,中心會把雜訊平均掉。
    語意不對齊的後果:模擬測試(200 次/組)在同語者 cos≈0.52 的高難度下,
    average+cosine 只有 86.8%,反而**輸給**貪婪的 94.2%;centroid 則有 99.5%。
    (同語者 cos≈0.68 的真實 CAM++ 水準下三者皆 100%,差異出現在難分的邊界情況。)

    對 L2 正規化向量,歐氏距離² = 2(1-cos),與 cosine 單調等價,
    故門檻換算 d = sqrt(2*(1-threshold)),語意仍是「相似度 >= threshold 才合併」。

    n_speakers 若指定則直接聚成該群數(忽略 threshold),供未來 speaker_count 使用。
    這條路**改用 average linkage**:centroid linkage 的合併高度非單調(inversion),
    配 maxclust 會失準(實測指定 2 群卻塌成 1 群);average 單調且在已知 K 時
    與 ward/complete 幾乎等價(99.5~99.7%)。

    回傳與輸入等長的標籤;編號依「首次出現時間」排序,與線上版慣例一致
    (說話者1 = 最早開口的人)。
    """
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    embs = [SpeakerClusterer._norm(e) for e in (embeddings or [])]
    if not embs:
        return []
    if len(embs) == 1:
        return [f"{prefix}1"]

    d = pdist(np.vstack(embs), metric="euclidean")
    if n_speakers and n_speakers > 0:
        raw = fcluster(linkage(d, method="average"),
                       t=min(n_speakers, len(embs)), criterion="maxclust")
    else:
        raw = fcluster(linkage(d, method="centroid"),
                       t=float(np.sqrt(2.0 * (1.0 - threshold))), criterion="distance")
    return _relabel(raw, prefix)


def _relabel(raw, prefix: str) -> list[str]:
    """把任意群編號改成依「首次出現順序」的 說話者N(scipy 的編號與時間無關)。"""
    order: dict = {}
    for c in raw:
        if c not in order:
            order[c] = len(order) + 1
    return [f"{prefix}{order[c]}" for c in raw]


def flatten_turns(turns, rule: str = "continue"):
    """把可重疊的語者時間軸攤平成「每個瞬間只有一位」。

    重疊時誰勝出:
      continue 上一刻就已在說話的人(把插話/附和視為背景) —— 預設
      longest  所屬 turn 較長者
    實測兩者差異極小(平均 DER 20.8% vs 20.9%),不是關鍵決策。

    ⚠️ 端點必須先四捨五入到同一精度再比較,否則會因浮點差異把語音判成「沒人在說」
    而整段漏掉(這個 bug 在評估腳本上讓 DER 從 5% 暴增到 74%)。
    """
    turns = [(round(float(s), 3), round(float(e), 3), l) for s, e, l in (turns or [])
             if float(e) > float(s)]
    if not turns:
        return []
    pts = sorted({x for s, e, _ in turns for x in (s, e)})
    out, prev = [], None
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if b - a < 1e-6:
            continue
        act = [t for t in turns if t[0] <= a + 1e-6 and t[1] >= b - 1e-6]
        if not act:
            prev = None
            continue
        if len(act) == 1:
            win = act[0][2]
        elif rule == "continue" and prev is not None and any(l == prev for _, _, l in act):
            win = prev
        else:
            win = max(act, key=lambda t: t[1] - t[0])[2]
        if out and out[-1][2] == win and abs(out[-1][1] - a) < 1e-6:
            out[-1] = (out[-1][0], b, win)
        else:
            out.append((a, b, win))
        prev = win
    return out


def build_spans(turns, merge_gap: float = 0.5, min_dur: float = 0.2,
                prefix: str = "說話者", rule: str = "continue"):
    """語者時間軸 → 可直接送 ASR 的分段 [(起ms, 訖ms, 標籤)]。

    這是 A2 的核心:切段依據從「停頓」換成「換人」,每段天生只有一位語者,
    標籤必然正確、文字也不會混雜多人(現行 VAD 切段實測 84% 的段混了平均 3 人)。

    merge_gap 合併相鄰同人的間隔;min_dur 丟棄過短的碎段。
    實測(5 場真實會議)這兩個參數對 DER 幾乎沒影響(0.6 個百分點內),
    但對段數影響很大(75 → 52 段)→ **應該用「段數」而不是 DER 來選**:
    段少 = ASR 呼叫少、處理快、App 顯示整齊。
    """
    flat = flatten_turns(turns, rule)
    merged: list = []
    for a, b, l in flat:
        if merged and merged[-1][2] == l and a - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], b, l)
        else:
            merged.append((a, b, l))
    kept = [(a, b, l) for a, b, l in merged if b - a >= min_dur]
    order: dict = {}
    for _, _, l in kept:
        if l not in order:
            order[l] = len(order) + 1
    return [(int(a * 1000), int(b * 1000), f"{prefix}{order[l]}") for a, b, l in kept]


def labels_from_timeline(timeline, spans, prefix: str = "說話者", min_overlap: float = 0.0):
    """把語者時間軸貼回既有的 ASR 分段:每段取「重疊最久」的語者(A1 做法)。

    timeline: [(起秒, 訖秒, 任意標籤)],可重疊(pyannote 會標出同時說話)。
    spans:    [(起ms, 訖ms)] —— ASR 的分段(仍由 VAD 決定,A1 不改切段)。

    一段若跨越多位語者(實測 84% 的段會),這裡只能取主要發言者 —— 這是 A1 的已知取捨;
    要根治得改用語者轉換點來切段(A2)。回傳與 spans 等長的標籤,編號依首次出現排序。
    """
    out_raw = []
    for b_ms, e_ms in spans:
        b, e = b_ms / 1000.0, e_ms / 1000.0
        acc: dict = {}
        for s, t, lab in timeline or []:
            ov = min(e, t) - max(b, s)
            if ov > 0:
                acc[lab] = acc.get(lab, 0.0) + ov
        if acc:
            best = max(acc, key=acc.get)
            out_raw.append(best if acc[best] >= min_overlap else None)
        else:
            out_raw.append(None)
    order: dict = {}
    for lab in out_raw:
        if lab is not None and lab not in order:
            order[lab] = len(order) + 1
    return [None if lab is None else f"{prefix}{order[lab]}" for lab in out_raw]


def assign_all(embeddings, durations=None, threshold: float = 0.5, prefix: str = "說話者",
               min_reliable_sec: float = 0.0, n_speakers: int | None = None):
    """全域分群 + 短段救援。回傳 (labels, centroids, counts)。

    分兩步,理由是短段聲紋不可信(見 assign 的說明):
      1. **只用「夠長的段」決定有幾個人、各是誰**(min_reliable_sec 以上)。
         短段若參與分群,會各自變成獨立小群 → 語者暴增。
      2. 再把**每一段(含短段)**歸到最近的中心 —— 不設門檻、強制歸入,
         因為短段不該有「開新語者」的權利,只能挑一個最像的。

    可信段不足 2 段時無法分群:退回「全部視為同一人」(寧可少分,也不要生出幽靈語者)。
    回傳的 centroids/counts 可餵回 SpeakerClusterer.load_state(),讓線上判定接續一致。
    """
    import numpy as np

    embs = [SpeakerClusterer._norm(e) for e in (embeddings or [])]
    if not embs:
        return [], [], []
    durs = list(durations) if durations is not None else [None] * len(embs)

    rel = [i for i, d in enumerate(durs) if d is None or d >= min_reliable_sec]
    if len(rel) >= 2:
        base = cluster_offline([embs[i] for i in rel], threshold, prefix, n_speakers)
        groups: dict = {}
        for i, lab in zip(rel, base):
            groups.setdefault(lab, []).append(embs[i])
        cents = [SpeakerClusterer._norm(np.mean(v, axis=0)) for v in groups.values()]
    elif len(rel) == 1:
        cents = [embs[rel[0]]]
    else:
        cents = [SpeakerClusterer._norm(np.mean(embs, axis=0))]   # 全是短段 → 當成一個人

    raw = [int(np.argmax([float(np.dot(e, c)) for c in cents])) for e in embs]
    labels = _relabel(raw, prefix)

    # 依新標籤重算中心與計數(只採計可信段;沒有可信段就全採)
    idx_of = {}
    for lab in labels:
        idx_of.setdefault(lab, len(idx_of))
    members: list[list] = [[] for _ in idx_of]
    for i, lab in enumerate(labels):
        if durs[i] is None or durs[i] >= min_reliable_sec or not rel:
            members[idx_of[lab]].append(embs[i])
    out_c, out_n = [], []
    for m in members:
        out_c.append(SpeakerClusterer._norm(np.mean(m, axis=0)) if m else cents[0])
        out_n.append(max(1, len(m)))
    return labels, out_c, out_n
