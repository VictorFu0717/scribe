# 重現官方安裝方式(HF 頁面: "clean Python 3.12 environment" + pip install qwen-asr[vllm])。
#
# 為什麼不 FROM vllm/vllm-openai:
#   那個 image 已 pin 好一整套相依(含 distutils 的 blinker、與其 torch 不匹配的 torchvision)。
#   在上面疊 qwen-asr 會把兩套相依世界混在一起 → blinker 無法移除 / torchvision::nms ABI 不符 等錯。
#   改用乾淨底層,讓 pip 一次把 qwen-asr[vllm] 及其相依(vllm/torch/torchvision/transformers)
#   解成互相匹配的版本(等同你本機已驗證可跑的組合)。
FROM python:3.12-slim

# vllm / torch / 音訊處理 執行期需要的系統庫
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 版本 pin 0.0.6:與本機驗證過的組合一致(vllm 0.14.0 / torch 2.9.1 / transformers 4.57.6)
# (最耗時的一層:會下載 torch/vllm 數 GB;放在 build-essential 之前以便被快取重用)
RUN pip install --no-cache-dir "qwen-asr[vllm]==0.0.6"

# build-essential 放最後:torch Inductor / Triton 執行期可能需要 C/C++ 編譯器,
# slim image 沒有會炸 InductorError。放在 pip 之後 → 不動到上面已快取的重量級 pip 層。
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# qwen-asr-serve:先把 qwen3_asr 註冊進 transformers + vLLM,再轉呼 vllm serve
ENTRYPOINT ["qwen-asr-serve"]
