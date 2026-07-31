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
VAD_MAX_END_SILENCE_MS = int(os.getenv("VAD_MAX_END_SILENCE_MS", "500"))
VAD_MAX_SEGMENT_SEC = float(os.getenv("VAD_MAX_SEGMENT_SEC", "15"))   # 單段上限(fsmn 預設 60s→15s)
ASR_TW = os.getenv("ASR_TRADITIONAL", "1") not in ("0", "false", "False", "")

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
