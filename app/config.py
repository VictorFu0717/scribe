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
MAX_SEG_SEC = float(os.getenv("MAX_SEG_SEC", "20"))   # ws 端安全切段(VAD 沒斷時的後盾)

# VAD 斷句停頓門檻(ms):停頓超過這麼久才斷句。權衡:
#   太大(fsmn 預設 800)→ 一段混多人 → 語者判錯;太小(350)→ 句內小停頓也切、句子被切碎。
#   500 是甜蜜點(抓句尾停頓、不抓句內猶豫)。語者還分不夠細→調小(400);句子還被切碎→調大(600~700)。
VAD_MAX_END_SILENCE_MS = int(os.getenv("VAD_MAX_END_SILENCE_MS", "550"))
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
SPK_THRESHOLD = float(os.getenv("SPK_THRESHOLD", "0.5"))
SPK_PREFIX = os.getenv("SPK_PREFIX", "說話者")

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
