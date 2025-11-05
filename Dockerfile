# 使用一個輕量級的 Python 官方映像檔作為基礎
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# ⭐️ 新增的部分：安裝系統依賴 ⭐️
# 更新套件庫並安裝 WeasyPrint 和 FFmpeg 所需的系統依賴
# --no-install-recommends 可以減少不必要的安裝，讓映像檔小一點
RUN apt-get update && apt-get install -y \
    libpango-1.0-0 \
    libharfbuzz0b \
    libpangoft2-1.0-0 \
    libgobject-2.0-0 \
    ffmpeg \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製你專案的所有檔案到映像檔中
COPY . .

# Heroku 會自動注入 PORT 環境變數
# 這行指令會啟動你的 Gunicorn 伺服器
CMD gunicorn --timeout 120 --workers 1 --bind 0.0.0.0:$PORT run:app