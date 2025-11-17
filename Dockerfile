# ------------------------------------
# 階段 1: 基礎環境
# ------------------------------------
# 使用官方 Python 基礎映像，選擇輕量級的 slim 版本
# 選擇 3.10-slim-buster/bullseye 或更高版本，確保與您的環境兼容
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# ------------------------------------
# 階段 2: 處理依賴關係
# ------------------------------------
# 複製依賴文件 (只複製需要的，以利用 Docker 快取)
# ⚠️ 注意：由於我們使用 PyTorch，直接複製整個 requirements.txt (見下方步驟)
# 如果沒有 requirements.txt，請手動列出所有依賴
COPY requirements.txt .

# 安裝依賴。--no-cache-dir 減少映像大小
# ⚠️ 注意：如果需要 CUDA/GPU 支援，這裡需要使用 NVIDIA 的基礎映像
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------
# 階段 3: 複製應用程式和模型
# ------------------------------------
# 複製您的所有應用程式檔案和模型權重
# 假設所有檔案都在當前目錄
COPY main.py .
COPY model.py .
COPY model_weights.pth . 
# 如果 model_weights.pth 很大，Docker 映像也會很大，這是正常的。

# ------------------------------------
# 階段 4: 運行服務
# ------------------------------------
# 暴露服務端口 (FastAPI/Uvicorn 默認使用 8000)
EXPOSE 8000

# 定義容器啟動時執行的命令
# 這裡使用 Uvicorn 的多 worker 模式來支持多核並發（類似於我們在 Windows 上的解決方案）
# --host 0.0.0.0: 允許外部訪問
# --workers 4: 運行 4 個進程（請根據您最終部署環境的核心數調整）
# main:app: 啟動 main.py 檔案中的 app 實例
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]