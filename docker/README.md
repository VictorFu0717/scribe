# Qwen3-ASR 定稿服務 (Docker)

`main.py` 的即時預覽 (FunASR) 跑在 host;**高準定稿**這半外包給這個容器裡的
vLLM (Qwen3-ASR, OpenAI 相容端點)。用 Docker 的好處:跨平台一致、不用在每台機器
處理 vLLM 的 CUDA / 相依性問題。

## 啟動

```bash
cd docker
cp .env.example .env        # 選用,按需修改
docker compose up -d        # 第一次會 build image
docker compose logs -f      # 等到看到 "Application startup complete"
curl http://localhost:8000/health
```

起來後在 host 跑 `python main.py`,它預設就連 `http://localhost:8000/v1`。

## 常用

```bash
docker compose down                 # 停
docker compose up -d --build        # 改 Dockerfile 後重建
docker compose logs -f vllm-qwen3-asr
```

## 設定重點 (改 docker-compose.yml)

| 項目 | 說明 |
|------|------|
| `--gpu-memory-utilization` | vLLM 佔多少 VRAM;與 host 上 FunASR 共卡時別設滿 |
| `count: all` | 用哪些 GPU;單卡可改 `device_ids: ["0"]` |
| image tag `v0.14.0` | 需 ≥ 含 qwen3_asr 支援的版本,想升級改 Dockerfile 的 FROM |
| volumes HF 快取 | 掛 host 快取避免容器重抓 ~3.5GB 模型 |

## 前置

- **Linux**: NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **Windows**: Docker Desktop (WSL2 後端),設定裡開 GPU;driver 需支援 WSL2

## 驗證定稿端點

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@some_16k.wav
```
