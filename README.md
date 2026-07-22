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
| **scribe** | 本 server（ASR 邏輯 + QA 端點）| 8005 | `python main.py` |
| **Qwen3-ASR** | 定稿 vLLM 服務 | 9000 | `docker/`（本 repo）|
| **Qwen3.6-27B** | 對話 LLM（QA/統整）| 8004 | `~/PycharmProjects/RAG_LangChain/vllm/` |

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

# 2) 對話 LLM:Qwen3.6-27B (vLLM) — 在另一個 repo
cd ~/PycharmProjects/RAG_LangChain/vllm && docker compose up -d   # host :8004

# 3) scribe 主服務
cd ~/PycharmProjects/websocket_ASR
.venv/bin/python main.py                    # :8005

# 測試:瀏覽器開 test.html(錄音→逐字→定稿→問這場會議)
```

---

## 端點

### ① 即時轉錄 — `WS /ws/asr`
```
Client → Server:
  binary                              PCM16 LE mono 16k 音訊
  {"type":"config","diarization":true} 開/關說話者辨識(開啟時 server 才 lazy-load 語者模型)
  {"type":"end"}                       結束本段,等所有句子定稿後回 final
  {"type":"reset"}                     丟棄狀態重來
Server → Client (JSON):
  {"type":"partial","committed":..,"tentative":..,"text":..,"diarization":bool,
   "segments":[{"speaker":"說話者1","text":..}, ...]}   committed=已定稿句,tentative=即時灰字
  {"type":"final","text":..,"segments":[...]}
  {"type":"config","diarization":bool}                  config 回覆
  {"type":"error","detail":..}
```
> **說話者辨識**：可開關、用到才載入（不開＝零 VRAM）。開啟後每句定稿會標上「說話者N」，
> `segments` 提供結構化結果，`committed`/`final.text` 會加「說話者N：」前綴並逐句換行。

### ③ 單場會議問答 — `POST /meeting/chat`（SSE 串流）
```
body: {"transcript":"逐字稿全文","question":"問題","history":[{"role","content"}...]}
回傳(text/event-stream): data: {"delta":"..."}  ...  data: [DONE]
```
> 逐字稿目前為 **stateless**（由 client 帶入）；context 上限 16384，長會議需靠 RAG（見 Roadmap ④）。

### `GET /health`
回傳各模型載入狀態。

---

## 設定（環境變數）

| 變數 | 預設 | 說明 |
|------|------|------|
| `VLLM_BASE_URL` | `http://localhost:9000/v1` | Qwen3-ASR 定稿服務 |
| `QWEN_MODEL` | `Qwen/Qwen3-ASR-1.7B` | 定稿模型名 |
| `CHAT_BASE_URL` | `http://localhost:8004/v1` | 對話 LLM 服務 |
| `CHAT_MODEL` | `Qwen3.6-27B` | 對話模型名 |
| `CHAT_ENABLE_THINKING` | `0` | 開啟 Qwen3 thinking（QA 預設關,較快）|
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

## Roadmap

- [x] ① 即時 ASR（逐字 + 定稿 + 繁體 + 併發）
- [x] 說話者辨識（可開關、lazy-load;CAM++/ERes2NetV2 + 線上分群）
- [x] ③ 單場會議 QA（grounding 在逐字稿）
- [ ] ② 會議儲存 + 自動統整（摘要 / 待辦 / 決議）
- [ ] ④ RAG 個人助手：逐字稿依 `user_id` 存向量DB，跨會議問答（「上週待辦」「上個月5號重點」）
      — 需 embedding 模型 + 向量DB（pgvector/Qdrant）+ 時間語意解析
