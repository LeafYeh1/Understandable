# 使用一個輕量級的 Python 官方映像檔作為基礎
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt 並安裝套件
# --no-cache-dir 可以稍微減少映像檔大小
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製你專案的所有檔案到映像檔中
COPY . .

# Heroku 會自動注入 PORT 環境變數
# 這行指令會啟動你的 Gunicorn 伺服器，跟你的 Procfile 做的事情一樣
CMD gunicorn --bind 0.0.0.0:$PORT app:app