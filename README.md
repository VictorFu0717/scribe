# scribe — 語音會議助理（Server 端）

即時把會議錄音轉成**逐字稿**（邊講邊出字）、句子結束自動**定稿**（高準），
並可針對該場會議**問答**。目標是一個個人助理 app 的後端；本 repo 是 server 端。

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

## 前置需求

- NVIDIA GPU + driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（Windows 用 Docker Desktop + WSL2）
- Docker / Docker Compose v2
- Python 3.12 + 本 repo 的 `.venv`（已裝 funasr / vllm / qwen-asr / opencc 等）

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
# 若對話用 Ollama:
CHAT_BASE_URL=http://localhost:11434/v1 CHAT_MODEL=qwen3.6:latest CHAT_API_KEY=ollama \
  .venv/bin/python main.py                                 # :8005

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

### ⑤ agentic 助理 — `POST /assistant/chat`（SSE 串流）
```
body: {"messages":[{"role","content"}...], "meeting_id":str|null, "language":"zh-Hant"}
回傳(text/event-stream): data: {"delta":"..."}  ...  data: [DONE]
```
> 手寫 **agent loop**:LLM 自行決定是否呼叫工具(多輪),最後串流答案。工具:
> `get_meeting_transcript` / `get_meeting_summary` / `list_meetings` / `search_meetings`(關鍵字,⑥ 升級語意)。
> 帶 `meeting_id` → 以該場為「目前會議」;不帶 → 可跨會議檢索。工具註冊表好擴充(加工具 = 加 schema + handler)。

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

> **② 定稿寫入**：WS `config` 帶 `meeting_id` 後，`end` 定稿完成會把逐字稿 segments 寫入該會議，
> 並更新 `duration_sec` 與 `status="ready"`。

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
| `SCRIBE_DB` | `scribe.db` | SQLite 資料庫路徑 |
| `DEFAULT_USER` | `dev` | 開發期預設 user_id（auth ⑦ 前的多租戶佔位）|
| `UPLOAD_MAX_SEG_SEC` | `30` | 上傳轉錄:過長 VAD 段的再切秒數 |
| `UPLOAD_CONCURRENCY` | `8` | 上傳轉錄:同時打 Qwen3-ASR 的段數上限 |
| `STREAM_MODEL` / `VAD_MODEL` | `paraformer-zh-streaming` / `fsmn-vad` | 預覽 / 斷句模型 |
| `FUNASR_HUB` | `hf` | FunASR 下載來源（`hf`/`ms`）|
| `ASR_TRADITIONAL` | `1` | 簡→繁台灣用語轉換 |
| `MAX_SEG_SEC` | `30` | 連續講不停的安全切段秒數 |
| `DIARIZE` | `0` | 說話者辨識是否預設開（通常由 app 用 config 訊息控制）|
| `SPK_MODEL` | `funasr/campplus` | 語者向量模型;ERes2NetV2 用 `iic/speech_eres2netv2_sv_zh-cn_16k-common` |
| `SPK_HUB` | 同 `FUNASR_HUB` | 語者模型下載來源（ERes2NetV2 在 ModelScope 需設 `ms`）|
| `SPK_THRESHOLD` | `0.5` | 同語者 cosine 門檻（越高越嚴、越容易判成新語者）|
| `SPK_PREFIX` | `說話者` | 語者標籤前綴 |
| `PORT` | `8005` | scribe 埠 |

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
- [ ] **⑥ RAG**（把 search_meetings 從關鍵字升級成 sqlite-vec + embedding 語意檢索）
- [x] 整段錄音上傳轉錄（`POST /meetings/{id}/audio`，背景批次）
- [ ] **⑦ 登入**（`POST /auth/token`）+ diarization 指定人數
