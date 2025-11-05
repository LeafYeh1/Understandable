from . import db  
from datetime import datetime

# 定義情緒類別
class_names = ['ang', 'dis', 'fear', 'happy', 'neu', 'sad']

# 使用者(諮商師)資料庫模板
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True) # UID
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    account = db.Column(db.String(100), nullable=False)
    clinic = db.Column(db.String(300))
    role = db.Column(db.String(20), nullable=False)  # 角色類型（counselor/user）
    counselor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # 綁定諮商師

# 患者資料庫模板
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    note = db.Column(db.String(500))
    counselor_id = db.Column(db.Integer, db.ForeignKey('user.id'))  # UID
    
# 患者的語音情續報告模板
class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)  # 儲存檔名或路徑
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # 時間戳
    description = db.Column(db.String(300))  
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id')) # UID
    patient = db.relationship('Patient', backref=db.backref('reports', lazy=True))

# 紀錄聊天內容模板
class ChatRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    date = db.Column(db.String(20))  # YYYY-MM-DD
    sender = db.Column(db.String(10))  # "user" or "ai"
    text = db.Column(db.Text)
    time = db.Column(db.String(10))   # HH:MM