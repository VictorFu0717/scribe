"""集中管理環境設定與常數。"""
import os

# --- 定稿 LLM (Qwen3-ASR @ vLLM) ---
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:9000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3-ASR-1.7B")

# --- 本地 ASR 模型 ---
STREAM_MODEL = os.getenv("STREAM_MODEL", "paraformer-zh-streaming")
VAD_MODEL = os.getenv("VAD_MODEL", "fsmn-vad")
FUNASR_HUB = os.getenv("FUNASR_HUB", "hf")   # 本機 hf 下載遠快於 ms
DEVICE = os.getenv("DEVICE", "cuda")

ASR_LANG = os.getenv("ASR_LANG") or None

# --- 即時預覽後端 ────────────────────────────────────────────────────────────
#   paraformer  FunASR paraformer-zh-streaming(預設)。原生串流、有 cache,
#               每次只算新 chunk → 成本與視窗長度無關,約 3% GPU/連線。
#               但它是**純中文模型**,英文會爛掉(見下方實測)。
#   nano        Fun-ASR-Nano-2512(800M,in-process vLLM)。沒有串流 cache,
#               每次重解「語音起點到現在」的整個視窗 → 越長越貴,20s 視窗約 20% GPU/連線。
#
# ASCEND 語料實測(120 句真人中英夾雜 + 60 句純中文,MER = 中文按字/英文按詞):
#              中英夾雜 MER   英文詞召回   純中文 MER
#   paraformer     22.3%        34.1%        6.8%
#   nano           13.2%        72.9%        7.1%
# 也就是:英文大幅改善、中文幾乎不動(SenseVoice 則是拿中文換英文,故未採用)。
# nano 的串流品質與離線 batch 完全相同(13.2% vs 13.3%)—— 因為每次都重解整個視窗,
# 最後一次更新已看過全部音訊,沒有「串流看不到後文」的劣勢。
STREAM_BACKEND = os.getenv("STREAM_BACKEND", "paraformer")   # paraformer | nano
NANO_MODEL = os.getenv("NANO_MODEL", "FunAudioLLM/Fun-ASR-Nano-2512")
# vLLM 是 in-process(不是另一個 HTTP 服務),要自己讓出顯存給同機的 Qwen3-ASR 定稿服務。
# ⚠️ 這是**佔總顯存的比例**,不是絕對值 —— 換到小卡要重算(96GB 上的 0.10 = 9.6GB,
#    24GB 卡上只有 2.4GB,會載不起來)。
# 實測(96GB 卡,max_model_len=2048):
#   0.06 以下  KV cache 算出來是負的 → 載不起來(固定開銷約 6.7GB:權重+音訊編碼器+CUDA graph)
#   0.08       KV 0.85GB / 7,904 tok  → 可併發 3.8
#   0.10       KV 2.74GB / 25,680 tok → 可併發 12.5   ← 預設
#   0.25       KV 17GB  / 159,056 tok → 可併發 77(延遲與 0.10 完全相同,多的全浪費)
# 一個請求約 450 token(20s 音訊 embedding + ≤200 輸出),所以 0.10 已是需求的數十倍。
NANO_GPU_FRAC = float(os.getenv("NANO_GPU_FRAC", "0.10"))
NANO_MAX_LEN = int(os.getenv("NANO_MAX_LEN", "2048"))
NANO_LANG = os.getenv("NANO_LANG", "中文")      # 夾雜英文照樣轉得出來,這只是提示
NANO_MAX_TOKENS = int(os.getenv("NANO_MAX_TOKENS", "200"))
# 熱詞(context biasing):公司術語、人名、專案代號。逗號分隔。
NANO_HOTWORDS = [w.strip() for w in os.getenv("NANO_HOTWORDS", "").split(",") if w.strip()]
# 兩次預覽更新的最小間隔(ms)。0 = 每個 chunk 都更新(最即時)。
# 視窗長時每次要 ~150-300ms,調大可省 GPU,代價是預覽更新變慢。
NANO_MIN_MS = int(os.getenv("NANO_MIN_MS", "0"))
MAX_SEG_SEC = float(os.getenv("MAX_SEG_SEC", "20"))   # ws 端安全切段(VAD 沒斷時的後盾)

# VAD 斷句停頓門檻(ms):停頓超過這麼久才斷句。權衡:
#   太大(fsmn 預設 800)→ 一段混多人 → 語者判錯;太小(350)→ 句內小停頓也切、句子被切碎。
#   500 是甜蜜點(抓句尾停頓、不抓句內猶豫)。語者還分不夠細→調小(400);句子還被切碎→調大(600~700)。
VAD_MAX_END_SILENCE_MS = int(os.getenv("VAD_MAX_END_SILENCE_MS", "600"))
VAD_MAX_SEGMENT_SEC = float(os.getenv("VAD_MAX_SEGMENT_SEC", "15"))   # 單段上限(fsmn 預設 60s→15s)
ASR_TW = os.getenv("ASR_TRADITIONAL", "1") not in ("0", "false", "False", "")

# --- 定稿前的音訊守門(擋幻覺 + 擋卡死)---
# 實測(2026-08,直接打 :9000):送「純靜音 3s」→ 模型失控生成、25s 不回應;
#   送「低雜訊」→ 憑空生出「嗯。」「no.」(還判成西班牙文)。非語音段若照送,
#   假句子會寫進 DB 還配一個說話者;長靜音更會把「單一序列」的定稿佇列整個堵住。
FINALIZE_TIMEOUT = float(os.getenv("FINALIZE_TIMEOUT", "30"))   # 單段定稿逾時(秒);逾時退回預覽文字,不整句丟掉
MIN_SEG_MS = float(os.getenv("MIN_SEG_MS", "150"))              # 短於此的碎段不送定稿(單音節通常 >150ms)
# 30ms 框的最大 RMS 低於此 → 視為非語音。**刻意設得很低**:實測 Qwen3-ASR 對小聲音訊極穩健
# (把語音縮到 0.002 倍、框峰值 0.0004,仍逐字正確),所以門檻拉高只會誤刪真語音,換不到好處。
# 這道門的職責僅限「擋掉數位靜音/近乎零訊號」(MAX_SEG_SEC 硬切出的空白段、手機靜音或中斷送來的零),
# 那才是實測會讓模型失控生成、把定稿佇列卡死的輸入。環境雜訊的幻覺不靠它,靠下面的正規化 + 逾時。
# 0.0005 ≈ PCM16 的 16 個量化階(量化底是 1/32768≈3e-5),等於在問「這段訊號是不是死的」。
# 實測真實語音縮到 0.005 倍(框峰值 0.0009)仍可正確辨識,設在這個位置才不會誤刪它。
MIN_SEG_RMS = float(os.getenv("MIN_SEG_RMS", "0.0005"))
# 段前補上前一段的尾巴當 lead-in(VAD 要靜音才斷句 → 補進來的是靜音,不會產生重複字);0=關。
SEG_PAD_MS = float(os.getenv("SEG_PAD_MS", "200"))
# 音量正規化:小聲的段落放大到目標 RMS 再送定稿。只放大不壓低(壓低救不了已削波的音訊),增益有上限。
# 實測價值不在「提升小聲語音準確度」(模型本來就夠穩),而在**抑制幻覺**:微弱雜訊放大到正常音量後,
# 模型更能認出那只是雜訊 —— 12 次雜訊測試的幻覺數 3 → 0。0=關。
SEG_NORM_RMS = float(os.getenv("SEG_NORM_RMS", "0.05"))
SEG_NORM_MAX_GAIN = float(os.getenv("SEG_NORM_MAX_GAIN", "8"))

# --- 說話者辨識(可開關、lazy-load)---
DIARIZE_DEFAULT = os.getenv("DIARIZE", "0") in ("1", "true", "True")
SPK_MODEL = os.getenv("SPK_MODEL", "funasr/campplus")   # ERes2NetV2: iic/speech_eres2netv2_sv_zh-cn_16k-common (SPK_HUB=ms)
SPK_HUB = os.getenv("SPK_HUB", FUNASR_HUB)
SPK_THRESHOLD = float(os.getenv("SPK_THRESHOLD", "0.65"))
SPK_PREFIX = os.getenv("SPK_PREFIX", "說話者")
# 短於此的段不得「新增語者」(只能歸入最像的既有語者)。實測 CAM++ 聲紋對段長極敏感:
#   同一人 cos 在 3.0s 是 0.67(0% 誤判)、2.0s 0.56(27%)、1.0s 0.37(80%)、0.3s 0.18(99%),
#   而不同人始終 0.07~0.16 → 短段的同人/異人分布完全重疊,調門檻救不了,
#   只能不讓它生出新語者(否則每個短句一個幽靈語者,即「語者變很多」)。
SPK_MIN_NEW_SEC = float(os.getenv("SPK_MIN_NEW_SEC", "2.0"))
# 即時串流每累積這麼多段就回頭全域重分群一次(修正先前判錯的標籤);0=關。
SPK_RECLUSTER_EVERY = int(os.getenv("SPK_RECLUSTER_EVERY", "10"))

# --- 語者分離後端(目前只影響「整檔上傳」路徑,即時串流仍走 campplus)---
# auto(預設):裝了 pyannote 且模型抓得到就用它,否則自動退回 campplus。
# 依據:5 場真實會議(AliMeeting,重疊率 2%~64%)實測 DER,**同樣輸出在 VAD 段上的公平比較**:
#   campplus 44.1% → pyannote 37.2%(改善 6.9pp)。
#   ⚠️ 別拿「campplus 61.6% vs pyannote 原始時間軸 16.1%」來宣稱改善 45pp —— 那混淆了
#   「標籤方法」與「輸出顆粒度」兩個變數;細顆粒度要靠 DIARIZE_SEGMENT=speaker(A2)才拿得到。
#   pyannote 真正穩定的優勢是**語者人數每場都對**(campplus 會塌成 1~2 位或碎成 8~9 位)。
#   速度兩者相當(1 小時錄音各約 16s/19s)。
DIARIZE_BACKEND = os.getenv("DIARIZE_BACKEND", "auto")   # auto | campplus | pyannote
PYANNOTE_SEG = os.getenv("PYANNOTE_SEG", "pyannote/segmentation-3.0")
PYANNOTE_EMB = os.getenv("PYANNOTE_EMB", "pyannote/wespeaker-voxceleb-resnet34-LM")
# 官方 speaker-diarization-3.1 的超參數(Optuna 調出來的,別隨手改)
PYANNOTE_THRESHOLD = float(os.getenv("PYANNOTE_THRESHOLD", "0.7045654963945799"))
PYANNOTE_MIN_CLUSTER = int(os.getenv("PYANNOTE_MIN_CLUSTER", "12"))
# gated 模型下載用;正式機若不能連外,請先把 HF cache 預載進去並設 HF_HUB_OFFLINE=1
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 上傳路徑的「切段依據」:
#   vad(預設,維持現狀)    依停頓切 → 標籤只能取主要發言者(一段常混多人)
#   speaker(A2)          依「語者轉換」切 → 每段天生單一語者
# 實測 5 場真實會議平均 DER:VAD 切段 + pyannote 標籤 37.2% → 依語者切段 20.8%。
# 但重疊嚴重的場合改善有限(64% 重疊那場 64.2%→56.3%),因為攤平重疊必然有損。
# pyannote 不可用時自動退回 vad。
DIARIZE_SEGMENT = os.getenv("DIARIZE_SEGMENT", "vad")     # vad | speaker
# 合併相鄰同人的間隔 / 丟棄過短碎段。實測對 DER 幾乎無影響(0.6pp 內),但段數 75→52,
# 所以是照「段數」選的:段少 = ASR 呼叫少、處理快、App 顯示整齊。
A2_MERGE_GAP = float(os.getenv("A2_MERGE_GAP", "0.5"))
A2_MIN_DUR = float(os.getenv("A2_MIN_DUR", "0.2"))

# 即時串流的切段依據(獨立於上傳,預設維持現狀):
#   vad(預設)  聽到停頓就切,定稿約 1 秒後出現 —— 現行行為,完全不動
#   speaker    依語者轉換切(層次①)。定稿慢約 10~15 秒(要等 pyannote 的 10 秒視窗滑過去
#              才知道那一刻是誰在講),但**預覽灰字仍即時**,畫面不會空著。
# 只吃 4% 即時預算,記憶體約 36MB/小時/連線(不留音訊)。
WS_SEGMENT = os.getenv("WS_SEGMENT", "vad")               # vad | speaker
# latency:等多久才算「定案」可送 ASR。**別設太小** —— 剛錄到的區域只被少數視窗覆蓋,
# 時間軸還是暫定的,會看到較多語者變化 → 段落被切碎、文字支離破碎。
# 3 場真實會議實測(上傳 A2 基準 DER 12.8% / 中位段長 6.3s):
#   latency=10 recluster=10 → DER 15.2%, 中位 3.7s   ← 明顯比上傳差,體感就是這個
#   latency=15 recluster=5  → DER 13.2%, 中位 5.3s   ← 預設值,幾乎追平上傳
#   latency=20 recluster=10 → DER 13.2%, 中位 5.4s   (再拉長沒有更好)
WS_DIAR_LATENCY = float(os.getenv("WS_DIAR_LATENCY", "15"))
WS_DIAR_RECLUSTER = float(os.getenv("WS_DIAR_RECLUSTER", "5"))
# 即時串流的「語者標籤」來源(與切段依據分開):
#   auto(預設) 有 pyannote 就用它的時間軸貼標籤,否則 campplus
#   campplus   每段抽一次聲紋再分群(舊行為)
# 搭配 WS_SEGMENT=vad 時 → **字照樣約 1 秒出來**(即時翻譯不受影響),
# 只有語者標籤晚約 15 秒才貼上、之後持續修正。5 場實測 DER:campplus 44.1% → pyannote 37.2%。
# (想要更準的 21.3% 就得用 WS_SEGMENT=speaker,但字會慢 20 秒 —— 先知道誰在講才切得了段。)
WS_DIARIZE = os.getenv("WS_DIARIZE", "auto")              # auto | campplus | pyannote

# --- 音訊 / 串流參數 ---
SAMPLE_RATE = 16000                 # 協定固定 16k;client 需自行 resample
PF_CHUNK = [0, 10, 5]               # paraformer 串流 chunk(600ms)
ENC_LOOKBACK = 4
DEC_LOOKBACK = 1
CHUNK_STRIDE = PF_CHUNK[1] * 960    # 9600 samples = 600ms @16k
CHUNK_MS = int(CHUNK_STRIDE / SAMPLE_RATE * 1000)

# --- 對話 LLM (Qwen3.6-27B @ vLLM;摘要/助理用)---
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "http://localhost:8004/v1")
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "EMPTY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen3.6-27B")
# 後端:auto(:11434→ollama,否則 vllm) | ollama | vllm。決定「關 thinking」與 API 呼叫方式:
#   vllm → /v1 + chat_template_kwargs.enable_thinking;ollama → /api/chat + think。
CHAT_BACKEND = os.getenv("CHAT_BACKEND", "auto")
CHAT_THINK = os.getenv("CHAT_THINK", "0") in ("1", "true", "True")   # 預設關 thinking(快;會議任務不需深度推理)
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")            # ollama 原生呼叫帶,避免模型被踢出重載

# --- Embedding (⑥ RAG;預設 Ollama bge-m3) ---
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11434/v1")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "ollama")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
RAG_CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "400"))   # 逐字稿切塊字元數
# hybrid 檢索的 RRF 常數:score = Σ 1/(K + 名次)。K 越大 → 各來源的名次差異被壓平、越像投票;
# 越小 → 越偏袒各自的第一名。60 是文獻慣用值。
RAG_RRF_K = int(os.getenv("RAG_RRF_K", "60"))
# 檢索策略。**預設純向量**:實測 hybrid 在測試語料上與純向量打平(Top-1 5/6 vs 5/6;
# 罕見識別碼 4/4 vs 4/4)—— bge-m3 對中文專有名詞已經夠強,量測不到品質提升,
# 故不預設啟用。設 1 可開 hybrid(向量+FTS5 關鍵字 RRF),好處是韌性:embedding 服務掛掉、
# 或某些內容 embedding 不了(bge-m3 對特定文字會回 NaN)時,關鍵字側仍搜得到。
RAG_HYBRID = os.getenv("RAG_HYBRID", "0") not in ("0", "false", "False", "")

# --- 儲存 ---
DB_PATH = os.getenv("SCRIBE_DB", "scribe.db")

# --- 認證 (⑦, JWT bearer) ---
AUTH_SECRET = os.getenv("AUTH_SECRET", "dev-insecure-secret-change-me-in-production-please")   # 正式務必用環境變數覆寫(>=32 bytes)
AUTH_ALGO = "HS256"
AUTH_TTL = int(os.getenv("AUTH_TTL", "43200"))     # token 有效秒數(預設 12h)
# false(開發):端點不強制 token,沒帶就退回 X-User-Id / DEFAULT_USER;/auth/token 未知帳號自動註冊
# true (正式):所有端點強制 Bearer,沒帶 401;不自動註冊
AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "0") in ("1", "true", "True")
# 身分來源(RAG 多租戶靠這個 user_id):
#   "jwt"(預設):帳密登入 → JWT。使用者同時走公司 WiFi(LAN 直連)與 Tailscale(外出),
#               LAN 沒有網路層身分,只能靠 app 登入才能在兩條路徑上得到「一致」的身分。
#   "tailscale":純內部、且只走 tailnet 時可用,身分直接取自 tailscale whois(免 app 登入)。
#               不適用「同時有 WiFi 直連」的情況。
AUTH_MODE = os.getenv("AUTH_MODE", "jwt")

# --- 開發期多租戶佔位(AUTH_REQUIRED=false 時的退回身分)---
DEFAULT_USER = os.getenv("DEFAULT_USER", "dev")

PORT = int(os.getenv("PORT", "8005"))
