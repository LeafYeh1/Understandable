from app import create_app, db
import os

# 建立 app 實例
app = create_app()

if __name__ == "__main__":
    with app.app_context():
        # 確保資料庫被建立
        db.create_all() 
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)