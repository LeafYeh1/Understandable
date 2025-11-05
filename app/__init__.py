import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from .config import Config

# 載入 .env 檔案
load_dotenv() 

# 1. 在工廠外部建立擴充套件實例
db = SQLAlchemy()

def create_app():
    """
    應用程式工廠函數
    """
    app = Flask(__name__)
    
    # 2. 載入設定
    app.config.from_object(Config)

    # 3. 將 app 綁定到擴充套件
    db.init_app(app)

    # 4. 設定日誌
    app.logger.setLevel(logging.DEBUG)

    # 5. 註冊 Blueprints 
    from .auth import routes as auth_routes
    from .core import routes as core_routes
    from .chat import routes as chat_routes
    from .analysis import routes as analysis_routes
    
    app.register_blueprint(auth_routes.auth_bp, url_prefix='/auth')
    app.register_blueprint(core_routes.core_bp, url_prefix='/core')
    app.register_blueprint(chat_routes.chat_bp, url_prefix='/chat')
    app.register_blueprint(analysis_routes.analysis_bp, url_prefix='/analysis')

    # 6. 預載入模型
    from .services import ml_models
    with app.app_context():
        app.logger.info("Starting model preload...")
        ml_models.preload_models(app.logger, app.config)
        app.logger.info("Model preload complete.")

    return app