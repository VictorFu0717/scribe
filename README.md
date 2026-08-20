# scribe — 語音會議助理（Server 端）

即時把會議錄音轉成**逐字稿**（邊講邊出字）、句子結束自動**定稿**（高準），
並可針對該場會議**問答**。目標是一個個人助理 app 的後端；本 repo 是 server 端。

> 📱 手機 App（iOS/Android 客戶端）：[**scribe-app**](https://github.com/VictorFu0717/scribe-app)
>
> 目錄名目前仍是 `websocket_ASR`，專案代稱為 **scribe**。

---

## 架構

```
[瀏覽器/App 麥克風]
   │ WebSocket, PCM16 LE mono 16k
   ▼
┌─────────────────────── scribe server (FastAPI, :8005) ───────────────────────┐
│  ① 即時 ASR                                     ③ 單場會議 QA                  │
│  逐字預覽 + VAD 斷句 + 定稿                       POST /meeting/chat (SSE)      │
│     │ 預覽/斷句(本地)      │ 定稿(HTTP,併發)         │ grounding 在逐字稿         │
│     ▼                     ▼                        ▼                           │
│  FunASR                Qwen3-ASR @ vLLM         Qwen3.6-27B @ vLLM             │
│  paraformer-streaming  (docker, :9000)          (docker, :8004)               │
│  + fsmn-vad            OpenAI 相容               OpenAI 相容                    │
└──────────────────────────────────────────────────────────────────────────────┘
  簡→繁:OpenCC s2twp(逐字稿與定稿一律繁體台灣用語)
```

**為什麼這樣切**：Qwen3-ASR 的「串流」API 不支援 batch、無法併發；因此
- **即時預覽**用輕量的 FunASR paraformer-streaming（本地、低延遲）；
- **定稿**丟給 `vllm serve` 的 Qwen3-ASR，vLLM 做 continuous batching → **真併發**；
### 併發模型（為什麼 `workers=1` 仍能同時服務多人）

`workers=1` 指的是「**單一 process、模型只載一份**」，**不是**一次只能處理一個請求：

- 單一 **async event loop** 可同時 juggle 多條 WebSocket 連線（每條各自維護狀態）。
- 逐字預覽 / VAD 是本地小模型呼叫，以一把鎖序列化，但每次 ~毫秒級，不是瓶頸。
- **定稿**是 async 打到 vLLM → vLLM 做 **continuous batching**，跨所有連線**真併發**。

開更多 worker 反而**有害**：每個 worker 會各載一份 FunASR 模型、VRAM 翻倍；GPU 才是瓶頸，
不是 web 層。要再擴充併發是加大 vLLM（或多卡 / 多 vLLM 實例），不是加 uvicorn worker。

---

## 元件與埠

| 服務 | 說明 | 埠 (host) | 啟動方式 |
|------|------|:---:|----------|
| **scribe** | 本 server（ASR + SQLite 儲存 + 會議 CRUD + QA）| 8005 | `python main.py` |
| **Qwen3-ASR** | 定稿 vLLM 服務（音訊→文字，Ollama 做不了）| 9000 | `docker/`（本 repo）|
| **對話 LLM** | 摘要/QA/助理；Qwen3.6-27B(vLLM) **或** Ollama `qwen3.6` | 8004 / 11434 | vLLM 或 `ollama serve` |

### 專案結構

```
main.py                    精簡入口(組裝 app、lifespan、掛路由)
app/
├── config.py              所有設定(env)
├── models.py              本地 ASR/語者 模型 + OpenCC + 定稿呼叫
├── db.py                  SQLite 儲存(aiosqlite;meetings/transcripts/summaries)
├── ws.py                  /ws/asr 即時轉錄 + 說話者 + 定稿寫入
├── routers/meetings.py    會議 CRUD
├── chat_qa.py             /meeting/chat 單場問答(舊端點)
└── diarize.py             說話者線上分群
```

---

## 部署與連線（Tailscale + JWT）

- **連線一律走 Tailscale**：app／使用者連 server 的 tailnet IP `http://100.68.0.81:8005`
  （WS 用 `ws://100.68.0.81:8005/ws/asr`）。公司內、外出都一樣，不受網段影響。
- **為什麼不走公司 WiFi 直連**：WiFi 客戶端（如 `192.168.68.x`）與 server 有線網段（`192.168.0.0/23`）**不同網段**，
  直連需請 MIS 開通跨網段路由；Tailscale 走 WireGuard 隧道、與網段無關、零網管成本，故直接全走 Tailscale。
- **身分辨識走 JWT**（`AUTH_MODE=jwt`）：登入拿 token，與連線層無關 → 換網路也是同一身分（多租戶 `user_id` 一致）。
- **對外只開 `8005`**（`ufw allow in on tailscale0`；LAN 直連才需 `ufw allow from 192.168.0.0/23 to any port 8005`）。
  `9000`(Qwen3-ASR)、`11434`(Ollama) 為內部服務，防火牆擋著即可（Ollama 若要給內網同仁，只對 LAN 網段開）。
- **人數**：Tailscale 免費上限 6 users；~15 人需付費方案，或自架 **Headscale**（開源、無使用者上限）。

---

## 前置需求

- NVIDIA GPU + driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（Windows 用 Docker Desktop + WSL2）
- Docker / Docker Compose v2
- Python 3.12 + 本 repo 的 `.venv`（已裝 funasr / vllm / qwen-asr / opencc 等）
- **`ffmpeg`（整檔上傳必備）**：`sudo apt install ffmpeg`。手機錄音預設是 **m4a/aac**，
  libsndfile 不支援，只能靠 librosa 的 audioread→ffmpeg 後備解碼。缺了會讓
  `POST /meetings/{id}/audio` 回 **400 cannot decode audio**。
  ⚠️ 光裝 ffmpeg 還不夠：該後備是 spawn `ffmpeg -i <路徑>`，**只吃檔案路徑、吃不了記憶體物件**，
  所以 `upload.py` 的 `_load_audio` 會在記憶體解碼失敗時把上傳內容落地成暫存檔再解。

---

## 啟動流程（三步）

```bash
# 1) 定稿服務:Qwen3-ASR (vLLM)
cd docker && docker compose up -d          # host :9000;首次會 build + 載入模型
docker compose logs -f                      # 等到 "Application startup complete"

# 2) 對話 LLM(擇一)
#    a) vLLM Qwen3.6-27B(另一個 repo):
cd ~/PycharmProjects/RAG_LangChain/vllm && docker compose up -d   # host :8004
#    b) 或用 Ollama(較省事):
ollama serve && ollama pull qwen3.6        # host :11434

# 3) scribe 主服務
cd ~/PycharmProjects/websocket_ASR
.venv/bin/python main.py                                   # 用 vLLM 對話(預設 :8004)
# 若對話用 Ollama(thinking 自動關;keep-alive 常駐避免冷重載):
CHAT_BASE_URL=http://localhost:11434/v1 CHAT_MODEL=qwen3.6:latest CHAT_API_KEY=ollama \
  OLLAMA_KEEP_ALIVE=-1 .venv/bin/python main.py            # :8005

# 測試:瀏覽器開 test.html(錄音→逐字→定稿→問這場會議)
```

---

## 端點

### ① 即時轉錄 — `WS /ws/asr`
```
Client → Server:
  binary                              PCM16 LE mono 16k 音訊
  {"type":"config","diarization":true,"meeting_id":"<id>"}  開/關語者辨識 + 關聯會議(定稿會寫入此 meeting)
  {"type":"end"}                       結束本段,定稿(+寫入儲存)後回 final
  {"type":"reset"}                     丟棄狀態重來
Server → Client (JSON):
  {"type":"partial","committed":..,"tentative":..,"text":..,"diarization":bool,
   "segments":[{"speaker":"說話者1","text":..}, ...]}   committed=已定稿句,tentative=即時灰字
  {"type":"final","text":..,"segments":[...],"meeting_id":..}
  {"type":"config","diarization":bool,"meeting_id":..}   config 回覆
  {"type":"error","detail":..}
```
> **說話者辨識**：可開關、用到才載入（不開＝零 VRAM）。開啟後每句定稿會標上「說話者N」，
> `segments` 提供結構化結果，`committed`/`final.text` 會加「說話者N：」前綴並逐句換行。
> 準度取決於**斷句細緻度**：一段若含多位說話者，只會判給一人。用 `VAD_MAX_END_SILENCE_MS`（越小切越細）
> 調整；但「兩人零停頓連續交談」VAD 切不開，需真正的語者分離（未來可加 pyannote 離線精修）。
>
> **⚠️ 段落太短時聲紋根本不可用**（實測 CAM++，這是「語者變很多」的根因）：
>
> | 段長 | 同一人 cos（平均/最低）| 不同人 cos | 同一人被判成不同人 |
> |---|---|---|---|
> | 0.3s | 0.18 / −0.14 | 0.07 | **99%** |
> | 1.0s | 0.37 / −0.08 | 0.10 | **80%** |
> | 2.0s | 0.56 / 0.28 | 0.16 | 27% |
> | 3.0s | 0.67 / 0.51 | — | 0% |
>
> 「同語者 cos≈0.67」**只在 3 秒以上成立**。短段的問題不是「兩人分不開」（不同人始終穩定在 0.07~0.16），
> 而是**同一人自己跟自己都對不上**，同人/異人分布完全重疊 → **調門檻救不了**，只能不讓短段產生新語者。
>
> **分群策略**（`app/diarize.py`）：
> - **即時串流** → `SpeakerClusterer.assign(emb, 秒數)` 線上貪婪 + 兩道防護：
>   (A) 短於 `SPK_MIN_NEW_SEC` 的段**只能歸入既有語者、不得新增、也不更新中心**；
>   (B) 每 `SPK_RECLUSTER_EVERY` 段用 `assign_all()` 回頭全域重分群、改寫先前標籤並同步 DB
>   （WS 每次 partial 本就重送整個 `segments`，App 會自動更新；收尾時再強制跑一次）。
> - **整檔上傳** → 同樣走 `assign_all()`：用夠長的段決定分群（centroid linkage），再把每段（含短段）歸到最近的中心。
>
> **整檔上傳可改用 pyannote 語者分離**（`DIARIZE_BACKEND`/`DIARIZE_SEGMENT`）。
> 5 場真實會議（AliMeeting，重疊率 2%~64%，含 RTTM 標準答案）實測 DER：
>
> | 切段依據 | 標籤來源 | 平均 DER | 語者人數 |
> |---|---|---|---|
> | VAD（停頓）| campplus（現狀）| 44.1% | 常塌成 1~2 位或碎成 8~9 位 |
> | VAD（停頓）| pyannote（A1）| 37.2% | **每場都對** |
> | **語者轉換（A2）**| pyannote | **20.8%** | **每場都對** |
>
> ⚠️ **不要拿「campplus 44.1% vs pyannote 原始時間軸 16.1%」宣稱改善 24pp** —— 那混淆了
> 「標籤方法」與「輸出顆粒度」。真正的改善要靠 A2 換掉切段依據。
>
> **即時串流也可以依語者切段**（`WS_SEGMENT=speaker`）。同一段 3 分鐘真實會議實測：
>
> | 模式 | 段數 | 辨識語者 | DER |
> |---|---|---|---|
> | `vad`（現狀）| 39 | 5 位（真人 3）| 14.4% |
> | `speaker` | 27 | **3 位** | **4.1%** |
>
> 代價是定稿慢約 10~15 秒（要等 pyannote 的 10 秒視窗滑過去才知道那一刻是誰在講），
> 但 paraformer 預覽灰字仍即時，畫面不會空著。只吃 4% 即時預算、約 36MB/小時/連線，
> **且不需要保存錄音**（只留聲紋與語音活動圖譜，還原不成可聽的內容）。
> 已知限制：開頭一分鐘分群未穩時可能「該切沒切」（兩人塞進同一段），事後無法補救。
>
> A2 的取捨：段落從「一句話」變成「一段發言」（較長、較少：83 段 → 43 段）；
> 重疊語音只保留主導者；重疊嚴重的場合改善有限（64% 重疊那場僅 64.2%→57.0%）。
> **App 端不需改**（欄位格式不變），但上傳與即時錄音的段落顆粒度會不一致。
>
> 模擬實測（100 次/組，段長分布貼近真實對話）：
>
> | 情境 | 原版貪婪 | 加 (A) | **加 (A)+(B)** |
> |---|---|---|---|
> | 2人/30段 | 74.1%／**11.4 位語者** | 85.4%／2.5 | **100%／2.0** |
> | 3人/40段 | 82.5%／**15.9 位** | 86.2%／3.3 | **98.7%／2.9** |
> | 4人/60段 | 87.3%／**22.7 位** | 88.2%／4.4 | **99.4%／4.0** |
>
> linkage 是實測選出來的：模擬 200 次/組，在同語者 cos≈0.52 的高難度下 centroid 達 **99.5%**、
> 貪婪 94.2%、而 average+cosine 只有 86.8%（**反而輸給貪婪** —— 群一大平均兩兩相似度就被拉低而過度切分，
> 與「比中心」的貪婪語意不匹配）。真實 CAM++ 水準（cos≈0.68）下三者皆 100%，差異只在難分的邊界情況。

### ⑤ agentic 助理 — `POST /assistant/chat`（SSE 串流）
```
body: {"messages":[{"role","content"}...], "meeting_id":str|null, "language":"zh-Hant"}
回傳(text/event-stream): data: {"delta":"..."}  ...  data: [DONE]
```
> 手寫 **agent loop**:LLM 自行決定是否呼叫工具(多輪),最後串流答案。工具:
> `get_meeting_transcript` / `get_meeting_summary` / `list_meetings` / `search_meetings`(**語意檢索**,可帶日期範圍)。
> 帶 `meeting_id` → 以該場為「目前會議」;不帶 → 可跨會議。系統會注入「今天日期」,agent 能自行把
> 「上週/上個月5號」換算成 `date_from/date_to` 傳給 `search_meetings`。工具註冊表好擴充(加工具 = 加 schema + handler)。

### ③ 單場會議問答（舊）— `POST /meeting/chat`（SSE 串流）
```
body: {"transcript":"逐字稿全文","question":"問題","history":[{"role","content"}...]}
回傳(text/event-stream): data: {"delta":"..."}  ...  data: [DONE]
```
> stateless(client 帶逐字稿)。已由 ⑤ `/assistant/chat` 取代(server 用 meeting_id 自取 + 工具);此端點保留相容。

### 會議 CRUD + 儲存（①②③）
存於 SQLite（`scribe.db`），皆掛 `user_id`（多租戶）。開發期以 `X-User-Id` header 指定使用者（預設 `dev`）。

| 端點 | 說明 |
|------|------|
| `POST /meetings` | 建會議（App 開始錄音時），body `{"title"}` → 回 meeting（`status:"recording"`）|
| `GET /meetings` | 列出使用者的會議 → `{"items":[...]}` |
| `GET /meetings/{id}` | 單一會議 metadata |
| `DELETE /meetings/{id}` | 刪除（連帶逐字稿/摘要）→ 204 |
| `GET /meetings/{id}/transcript` | `{"segments":[{id,text,speaker,is_final,start_ms,end_ms}]}` |
| `GET /meetings/{id}/summary` | 有摘要回 JSON；沒有回 **404** |

> **② 定稿寫入（逐句即時、斷線不丟）**：WS `config` 帶 `meeting_id` 後，**每句定稿就立刻寫入 DB**（不等 `end`）。
> 所以 iOS 背景中斷／WS 斷線也不會丟已定稿的句子；`end` 或斷線收尾時把會議設 `status="ready"` + 更新 `duration_sec` + 建 RAG 索引（不會卡在「轉錄中」）。
> **重連續錄**：用同一 `meeting_id` 重新 `config`，會接在既有段落之後續號（不覆蓋）。

### 整段錄音上傳轉錄 — `POST /meetings/{id}/audio`（背景批次）
```
multipart/form-data: file=<音檔 wav/flac/ogg;mp3/m4a 需 ffmpeg>, diarization=<true|false>
→ {"id":..,"status":"transcribing","duration_sec":N}
```
> 上傳後 server **背景**處理:VAD 切段 → 每段送 Qwen3-ASR 定稿(併發,vLLM batching)→ 可選說話者辨識
> → 寫入該會議。App 上傳後輪詢 `GET /meetings/{id}` 直到 `status="ready"`,再 `GET .../transcript`。
> 過長的 VAD 段會自動切 ≤30s(`UPLOAD_MAX_SEG_SEC`);併發上限 `UPLOAD_CONCURRENCY`(預設 8)。

### ④ 會議摘要 — `POST /meetings/{id}/summarize`（SSE 串流）
```
body: {"language":"zh-Hant"}   (可省略)
回傳(text/event-stream):
  data: {"delta":"..."}            邊產邊顯示的 Markdown 文字
  ...
  data: {"overview":"..","key_points":[..],"decisions":[..],
         "action_items":[{"task","owner?","due?"}],"follow_ups":[..]}   結構化卡片
  data: [DONE]
```
> 先串流 Markdown 供顯示,串完解析成結構化 JSON、存入 DB(之後 `GET .../summary` 可取)。
> 長逐字稿自動 **map-reduce**(分段濃縮再合併)。摘要會存檔並把會議 `has_summary` 設為 true。

### ⑦ 身分辨識（多租戶 user_id 來源）

由 `AUTH_MODE` 決定；**切換只改這一個設定**，RAG / 會議 / 摘要等其他程式完全不用動（都只吃 `get_current_user` 給的 `user_id`）。

> **為什麼預設 `jwt`**：使用者會**同時**走公司 WiFi（同網段直連 `192.168.x.x`）和 Tailscale（外出 `100.x`）。
> LAN 不提供「使用者是誰」，只有**應用層登入**才能在兩條路徑上得到一致身分。帳密登入一次（app 存 token）→
> 走哪條網路都認得同一人。此時 Tailscale 只是**純遠端連線通道（VPN）**，不再是身分來源。

**`AUTH_MODE=jwt`（預設；WiFi + Tailscale 混用、或對公網）— 帳密登入 → JWT**
```
POST /auth/register  {"username","password"}       → {access_token,token_type,expires_in,user_id,username}
POST /auth/token     form: username=&password=     → 同上(OAuth2 標準)
GET  /auth/me        Authorization: Bearer <jwt>   → 目前使用者
```
- 端點以 `Authorization: Bearer <jwt>` 認身分；WS 可用 `?token=`／`Authorization` header／`config` 訊息帶 `token`。
- `AUTH_REQUIRED=1` 強制 token（否則 401）；`=0`（開發）沒帶退回 `X-User-Id`／`DEFAULT_USER`，且 `/auth/token` 未知帳號自動註冊。
- 帳號管理：目前**開放註冊**（網路已被 WiFi/Tailscale 閘控）；審核制／邀請制待補（register→待審→admin 核准）。

**`AUTH_MODE=tailscale`（選用；純內部、且只走 tailnet 時）— 身分取自 `tailscale whois`**
- 免 app 登入，tailnet 邀請名單即白名單；同一人多裝置＝同一 email＝同一租戶。
- ⚠️ **不適用「同時有公司 WiFi 直連」**：LAN 連線 whois 認不出人（會全退回 `DEFAULT_USER`）→ 這種混用要用 `jwt`。

### 留檔翻譯 — `POST /meetings/{id}/translate`（SSE 串流）
```
body: {"target":"en"}    (語言代碼或名稱;en/ja/ko/zh-Hant/vi/…)
回傳(SSE): data: {"delta":"..."}  ...  data: [DONE]
GET /meetings/{id}/translation?target=en  → {"target","text"}   (未翻過回 404)
```
> 把逐字稿用 chat LLM 翻成目標語言、保留每行「說話者N：」逐行結構、串流回傳並**存檔**（可重取）。長逐字稿分段翻。
> **即時雙語字幕請走 app 端「裝置內翻譯」**（google_mlkit_translation / Apple Translation，零延遲、免 server 負載）；
> 此端點是「會後留檔的高品質翻譯」。

### `GET /health`
回傳各模型載入狀態。

---

## 設定（環境變數）

| 變數 | 預設 | 說明 |
|------|------|------|
| `VLLM_BASE_URL` | `http://localhost:9000/v1` | Qwen3-ASR 定稿服務 |
| `QWEN_MODEL` | `Qwen/Qwen3-ASR-1.7B` | 定稿模型名 |
| `CHAT_BASE_URL` | `http://localhost:8004/v1` | 對話 LLM 服務;用 Ollama 設 `http://localhost:11434/v1` |
| `CHAT_MODEL` | `Qwen3.6-27B` | 對話模型名;Ollama 設 `qwen3.6:latest` |
| `CHAT_API_KEY` | `EMPTY` | 對話 LLM 金鑰（Ollama 隨意填如 `ollama`）|
| `CHAT_BACKEND` | `auto` | 對話後端:`auto`(URL 含 `:11434`→ollama,否則 vllm) / `ollama` / `vllm`。決定「關 thinking」與 API 呼叫方式（見下方說明）|
| `CHAT_THINK` | `0` | 是否開 thinking。`0`=關（預設,快;會議任務不需深度推理）、`1`=開 |
| `OLLAMA_KEEP_ALIVE` | `30m` | (ollama 後端) 每次呼叫帶入,模型保留多久;設 `-1`=永久保留（根治 36B 冷重載 ~90s） |
| `SCRIBE_DB` | `scribe.db` | SQLite 資料庫路徑（含 sqlite-vec 向量表）|
| `DEFAULT_USER` | `dev` | 開發期預設 user_id（auth ⑦ 前的多租戶佔位）|
| `EMBED_BASE_URL` | `http://localhost:11434/v1` | Embedding 服務（預設 Ollama）|
| `EMBED_MODEL` | `bge-m3` | Embedding 模型（1024 維）|
| `EMBED_DIM` | `1024` | 向量維度（換模型要一起改）|
| `RAG_CHUNK_CHARS` | `400` | 逐字稿切塊字元數 |
| `RAG_HYBRID` | `0` | `0`=**純向量（預設）**；`1`=向量＋關鍵字 RRF 混合。實測混合未帶來品質提升，見下方 |
| `RAG_RRF_K` | `60` | RRF 常數：`score = Σ 1/(K + 名次)`。越大越像投票、越小越偏袒各自第一名 |
| `TRANSLATE_MAP_CHARS` | `3000` | 留檔翻譯:超過此長度就分段翻 |
| `AUTH_MODE` | `jwt` | 身分來源:`jwt`(帳密登入,預設;WiFi+Tailscale 混用一致身分) 或 `tailscale`(whois,純 tailnet 才適用) |
| `AUTH_SECRET` | `dev-insecure...` | JWT 簽章密鑰（`jwt` 模式;**正式務必覆寫**,>=32 bytes）|
| `AUTH_TTL` | `43200` | token 有效秒數（12h）|
| `AUTH_REQUIRED` | `0` | (`jwt` 模式) `1`=強制 Bearer;`0`=沒帶退回 `X-User-Id`/`DEFAULT_USER` |
| `UPLOAD_MAX_SEG_SEC` | `30` | 上傳轉錄:過長 VAD 段的再切秒數 |
| `UPLOAD_CONCURRENCY` | `8` | 上傳轉錄:同時打 Qwen3-ASR 的段數上限 |
| `STREAM_MODEL` / `VAD_MODEL` | `paraformer-zh-streaming` / `fsmn-vad` | 預覽 / 斷句模型 |
| `FUNASR_HUB` | `hf` | FunASR 下載來源（`hf`/`ms`）|
| `ASR_TRADITIONAL` | `1` | 簡→繁台灣用語轉換 |
| `MAX_SEG_SEC` | `20` | ws 端安全切段秒數（VAD 沒斷時的後盾）|
| `FINALIZE_TIMEOUT` | `30` | 單段定稿逾時（秒）。**必要**：非語音段會讓 Qwen3-ASR 失控生成，SDK 預設 600s 會把定稿佇列堵死。逾時退回預覽文字，不整句丟掉 |
| `MIN_SEG_MS` | `150` | 短於此的碎段不送定稿 |
| `MIN_SEG_RMS` | `0.0005` | 30ms 框最大 RMS 低於此視為「死訊號」→ 不送定稿。刻意設得極低（≈PCM16 的 16 個量化階），只擋數位靜音；拉高會誤刪真語音（見下方說明）|
| `SEG_PAD_MS` | `200` | 段前補前一段尾巴當 lead-in（斷句處是靜音，不會產生重複字）；`0`=關 |
| `SEG_NORM_RMS` | `0.05` | 音量正規化目標 RMS（只放大不壓低）；`0`=關 |
| `SEG_NORM_MAX_GAIN` | `8` | 正規化增益上限 |
| `VAD_MAX_END_SILENCE_MS` | `500` | VAD 斷句停頓門檻(ms)。太小(350)句子被切碎、太大(800)一段混多人。語者不夠細→調小(400);句子被切碎→調大(600~700) |
| `VAD_MAX_SEGMENT_SEC` | `15` | VAD 單段上限秒數（fsmn 原生 60s）|
| `DIARIZE` | `0` | 說話者辨識是否預設開（通常由 app 用 config 訊息控制）|
| `SPK_MODEL` | `funasr/campplus` | 語者向量模型;ERes2NetV2 用 `iic/speech_eres2netv2_sv_zh-cn_16k-common` |
| `SPK_HUB` | 同 `FUNASR_HUB` | 語者模型下載來源（ERes2NetV2 在 ModelScope 需設 `ms`）|
| `SPK_THRESHOLD` | `0.5` | 同語者 cosine 門檻（越高越嚴、越容易判成新語者）|
| `SPK_MIN_NEW_SEC` | `2.0` | **短於此的段不得新增語者**（只能歸入最像的既有語者，也不更新中心）。防「語者暴增」的關鍵，見下方說明 |
| `SPK_RECLUSTER_EVERY` | `10` | 即時串流每累積幾段就回頭全域重分群、修正先前標籤；`0`=關 |
| `SPK_PREFIX` | `說話者` | 語者標籤前綴 |
| `DIARIZE_BACKEND` | `auto` | `auto`（有 pyannote 就用，否則退回）/ `campplus` / `pyannote`。**只影響整檔上傳**，即時串流仍走 campplus |
| `DIARIZE_SEGMENT` | `vad` | 上傳路徑的切段依據：`vad`=依停頓（現狀）/ `speaker`=依語者轉換（A2，見下方）|
| `A2_MERGE_GAP` | `0.5` | (A2) 合併相鄰同人的間隔秒數 |
| `A2_MIN_DUR` | `0.2` | (A2) 丟棄短於此的碎段 |
| `HF_TOKEN` | — | 下載 pyannote gated 模型用；正式機不能連外時請預載 HF cache 並設 `HF_HUB_OFFLINE=1` |
| `WS_SEGMENT` | `vad` | **即時串流**的切段依據：`vad`=聽到停頓就切（現狀，定稿約 1 秒）/ `speaker`=依語者轉換切（定稿慢約 10~15 秒，但預覽灰字仍即時）|
| `WS_DIAR_LATENCY` | `10` | (speaker) 等多久才算「定案」可送 ASR |
| `WS_DIAR_RECLUSTER` | `10` | (speaker) 每隔多久重跑全域分群、回頭修正標籤 |
| `PORT` | `8005` | scribe 埠 |

### 定稿前的音訊守門（為什麼需要）

實測（2026-08，直接打 `:9000`）發現 Qwen3-ASR 對**非語音輸入**有兩種病態行為：

| 輸入 | 行為 |
|------|------|
| 純數位靜音 3s | **失控生成、25s 不回應** ← 最嚴重 |
| 極低雜訊 | 憑空生出「嗯。」「no.」（還會判成西班牙文）|
| 中等雜訊 / 嗡嗡聲 | 正常回空字串 |

定稿佇列是**單一序列**處理，所以一段靜音卡住 → 該場會議後續全部定稿被堵住；斷線收尾只等 8 秒，逾時就丟掉還沒寫入的句子。因此：

- **`FINALIZE_TIMEOUT`**：讓卡住有上界（原本 SDK 預設 600s）。逾時保留 paraformer 預覽文字寫入 DB，不讓整句消失。
- **`MIN_SEG_RMS` / `MIN_SEG_MS`**：非語音段根本不送定稿。
- **`SEG_NORM_RMS`**：意外發現的好處——把微弱雜訊放大到正常音量後，模型更能認出「這只是雜訊」，12 次雜訊測試的幻覺數 **3 → 0**。

⚠️ **`MIN_SEG_RMS` 不要隨意調高**：實測 Qwen3-ASR 對小聲音訊極穩健（語音縮到 **0.002 倍**、框峰值 0.0004 仍逐字正確），門檻拉高只會誤刪真語音，換不到抗雜訊的好處。能量本來就分不開「環境底噪」與「很小聲的人聲」，雜訊幻覺交給正規化與逾時處理。

### 對話後端與 thinking（重要）

摘要/翻譯/QA/助理全走 `app/llm.py` 的統一層,依 `CHAT_BACKEND` 自動選對後端與「關 thinking」的方式——**兩邊機制不可互通**（實測 2026-08，Ollama qwen3.6）:

| 後端 | 關 thinking 的方式 | API |
|------|------|-----|
| **vLLM** | `chat_template_kwargs.enable_thinking=False` | `/v1`(OpenAI 相容) |
| **Ollama** | 原生 `think:False` | **`/api/chat`**（`/v1` 端點會**忽略** `enable_thinking` 與 `think`，關不掉！）|

- Qwen3 的 thinking 實質是**開/關二元,沒有 low/med/high 檔位**;想「短思考」不可行,實務就是關掉（`CHAT_THINK=0`）。
- 切後端只改 `CHAT_BASE_URL`（或強制 `CHAT_BACKEND`），**程式不用動**;對外 SSE 契約不變。
- Ollama 若回應慢多半是 **36B 模型被踢出重載**（非 thinking）→ 設 `OLLAMA_KEEP_ALIVE=-1` 常駐。

---

## 測試工具

- `test.html` — 瀏覽器：錄音 → 逐字/定稿 → 同頁問這場會議
- `test_qa.py` — 命令列驗證 `/meeting/chat`（多輪對話 + 一題故意問逐字稿沒有的）

---

## 疑難排解（Docker / vLLM 常見雷）

- **`No module named ...qwen3_asr`**：image 的 vLLM/transformers 不認得 `qwen3_asr` → 定稿服務要裝 `qwen-asr` 並用 `qwen-asr-serve` 入口（見 `docker/Dockerfile`）。
- **不要 `FROM vllm/vllm-openai` 疊 qwen-asr**：會撞 blinker(distutils)、torchvision::nms(ABI) 等衝突。改 `FROM python:3.12-slim` + `pip install "qwen-asr[vllm]"`。
- **`InductorError` / `torchvision::nms does not exist`**：slim image 缺編譯器 → 加 `--enforce-eager`（跳過 torch.compile）+ 裝 `build-essential`。
- **定稿文字夾雜 `language Chinese<asr_text>`**：Qwen3-ASR 經 vLLM 的格式外洩 → `main.py` 的 `_clean_qwen()` 已處理。
- **埠對應**：容器內一律 8000；對外映射（如 `9000:8000`）改的是 host 埠，`main.py` 要用 `VLLM_BASE_URL` 指到 host 埠。

---

## Roadmap（依 App HANDOFF 契約）

- [x] 即時 ASR（逐字 + 定稿 + 繁體 + 併發）+ 說話者辨識（可開關、lazy-load）
- [x] **① SQLite 儲存**（meetings / transcripts / summaries，掛 user_id 多租戶）
- [x] **② 定稿寫入**（WS 帶 `meeting_id` → 定稿存入、status=ready、duration）
- [x] **③ 會議 CRUD**（`/meetings` 系列端點）
- [x] **④ 摘要**（`POST /meetings/{id}/summarize`，SSE + 結構化 JSON，長逐字稿 map-reduce）
- [x] **⑤ agentic 助理**（`POST /assistant/chat`，手寫 loop + 工具:get_transcript/get_summary/list/search）
- [x] **⑥ RAG**（sqlite-vec + bge-m3 語意檢索;定稿/上傳後自動索引;多租戶 user_id 分區 + 日期過濾）
- [x] **hybrid 檢索**（向量 + FTS5 關鍵字，RRF 合併）—— 見下方說明

### hybrid 檢索（向量 + 關鍵字）— **預設關閉，`RAG_HYBRID=1` 開啟**

實測未帶來檢索品質提升（數據見下），故預設走純向量；程式與 FTS 索引都保留，隨時可切換。

`rag.hybrid_search()` 同時跑兩側再以 RRF 合併：`score = Σ 1/(RRF_K + 名次)`。只用名次、不用分數，
所以不必去校正 cosine 距離與 bm25 兩種尺度；任一側失敗（例：embedding 服務掛了）仍以另一側作答。

⚠️ **中文 FTS5 必須用 `trigram` 分詞器**：預設的 `unicode61` 對中文完全無效（中文沒有空白，
整串變成單一 token，實測查「健保署」「預算成長」都是 **0 筆**）。查詢端也要對應處理 ——
中文取 **3 字滑窗**（「健保署預算」→ 健保署／保署預／署預算）；trigram 需 ≥3 字，
所以 2 字查詢（如「長照」）拆不出詞 → **退回 `LIKE` 子字串比對**，否則會靜默回空。

**實測誠實結論**：在我建的測試語料上（24 場、含高度相似的干擾），hybrid 與純向量**打平**
（Top-1 各 5/6；罕見識別碼 NC-2731／人名／法規條號／EIP-Portal 也是各 4/4）——
bge-m3 對中文專有名詞本來就夠強，**沒有量測到檢索品質的提升**。

hybrid 目前**已證實**的價值是**韌性**：
- embedding 服務掛掉 → 關鍵字側照樣作答
- **某些內容根本 embedding 不了**：實測 bge-m3(Ollama) 會對特定文字組合回 `NaN` 而讓整個請求 500，
  可 100% 重現、且**同批其他文字會被一起拖下水**。這類塊仍會寫入 `chunks`+`fts_chunks`（只是不進向量表），
  **靠關鍵字側仍搜得到**（已驗證：向量側完全找不到，hybrid 救回）。
- [x] 整段錄音上傳轉錄（`POST /meetings/{id}/audio`，背景批次）
- [x] **⑦ 身分辨識**（`AUTH_MODE`:jwt=帳密登入(預設;WiFi+Tailscale 混用一致身分) / tailscale=whois(純 tailnet 選用)）
- [ ] 帳號審核制 + admin 管理（產品化時;register→待審→核准）
- [ ] diarization 指定人數（`speaker_count`;上傳路徑 K 群聚類,選配）
