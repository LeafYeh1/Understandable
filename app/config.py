import os

class Config:
    """基礎設定類別"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'a_very_strong_default_secret_key')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # 你的資料庫 URL 處理邏輯
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if DATABASE_URL:
        db_url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        if "sslmode=" not in db_url:
            sep = "&" if "?" in db_url else "?"
            db_url = f"{db_url}{sep}sslmode=require"
        SQLALCHEMY_DATABASE_URI = db_url
    else:
        SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:123456@localhost/emotion_app'

    # 你的模型路徑
    MODEL_URL = os.getenv("EMOTION_MODEL_URL")
    MODEL_SHA256 = os.getenv("EMOTION_MODEL_SHA256","").lower()
    MODEL_PATH = os.getenv("MODEL_PATH","/tmp/emotion_model.h5")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")