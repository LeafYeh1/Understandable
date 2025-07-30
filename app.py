# flask 基礎設定
from flask import Flask, request, jsonify
from flask import render_template, send_file, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy

# 第三方函式庫
from tensorflow.keras.models import load_model
import numpy as np
import librosa
from pydub import AudioSegment
from io import BytesIO
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from weasyprint import HTML

# 初始化 Flask 應用
app = Flask(__name__)

# 設定密鑰和資料庫
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:123456@localhost/emotion_app' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 定義情緒類別
class_names = ['ang', 'dis', 'fear', 'happy', 'neu', 'sad']

# 使用者(諮商師)資料庫模板
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True) # UID
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    account = db.Column(db.String(100), nullable=False)
    clinic = db.Column(db.String(300))

# 患者資料庫模板
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    note = db.Column(db.String(500))
    counselor_id = db.Column(db.Integer, db.ForeignKey('user.id'))  # UID
    
# 患者的語音情續報告模板(尚未完成)
class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)  # 儲存檔名或路徑
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # 時間戳
    description = db.Column(db.String(300))  
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id')) # UID
    patient = db.relationship('Patient', backref=db.backref('reports', lazy=True))

# 情緒分析核心
class EmotionAnalyzer:
    def __init__(self, model_path="emotion_model.h5"):
        self.model = load_model(model_path)
        self.sr = 16000  # 取樣率
        self.segment_duration = 2  # 每段切割長度（秒）

    def extract_features(self, y):
        """
        從音訊 array 中提取 MFCC、Delta、Chroma、Contrast 特徵並整理成固定形狀
        """
        n_fft = 512
        hop_length = 128

        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=15, n_fft=n_fft, hop_length=hop_length)
        delta_mfcc = librosa.feature.delta(mfcc, order=1)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, n_fft=n_fft, hop_length=hop_length)
        contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr, n_fft=n_fft, hop_length=hop_length)

        min_frames = min(mfcc.shape[1], delta_mfcc.shape[1], delta2_mfcc.shape[1], chroma.shape[1], contrast.shape[1])

        # 裁切時間軸長度一致
        features = np.vstack([
            mfcc[:, :min_frames],
            delta_mfcc[:, :min_frames],
            delta2_mfcc[:, :min_frames],
            chroma[:, :min_frames],
            contrast[:, :min_frames],
        ])

        # 統一長度
        max_len = 500
        if features.shape[1] < max_len:
            pad_width = max_len - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :max_len]

        return features.T  # shape = (500, 64)

    def classify_segment(self, y, threshold_energy=0.01, min_duration_speech=0.5):
        """
        判斷該段音訊是靜音(0)、不確定(1)、語音(2)
        """
        if len(y) < self.sr * self.segment_duration:
            y = np.pad(y, (0, self.sr * self.segment_duration - len(y)))

        energy = np.sum(y ** 2) / len(y)
        if energy < threshold_energy:
            return 0

        voiced = sum([(e-s)/self.sr for s, e in librosa.effects.split(y, top_db=30)])
        if voiced >= min_duration_speech:
            return 2
        return 1

    def analyze(self, file_stream):
        """
        主邏輯處理流程：切割 -> 標註 -> 黏合 -> 預測 -> 統計
        """
        audio = AudioSegment.from_file(file_stream, format="wav")
        segment_ms = self.segment_duration * 1000
        segments = [audio[i:i+segment_ms] for i in range(0, len(audio), segment_ms)]

        # 對每段分類（靜音、不確定、語音）
        labels = [self.classify_segment(np.array(seg.get_array_of_samples()).astype(np.float32) / (2**15)) for seg in segments]

        # 修正不確定的段落標籤
        for i in range(len(labels)):
            if labels[i] == 1:
                if (i > 0 and labels[i-1] == 2) or (i < len(labels)-1 and labels[i+1] == 2):
                    labels[i] = 2
                elif (i == 0 or labels[i-1] == 0) and (i == len(labels)-1 or labels[i+1] == 0):
                    labels[i] = 0

        emotion_counts = {name: 0 for name in class_names}
        line_results = []

        # 找出連續語音段（4 秒）黏合
        for i in range(len(labels) - 1):
            if labels[i] == labels[i+1] == 2:
                combined = segments[i] + segments[i+1]
                y = np.array(combined.get_array_of_samples()).astype(np.float32) / (2**15)
                features = self.extract_features(y)
                input_data = np.expand_dims(features, axis=0)
                prediction = self.model.predict(input_data)[0]
                predicted_label = class_names[int(np.argmax(prediction))]
                emotion_counts[predicted_label] += 1
                line_results.append(predicted_label)

        return emotion_counts, line_results
    
# 初始化情緒分析器
analyzer = EmotionAnalyzer()

# 登入要求
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        account = request.form["account"]
        password = request.form["password"]
        user = User.query.filter_by(account=account).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            return redirect(url_for("home"))
        else:
            flash("帳號或密碼錯誤")
    return render_template("login.html")

# 登出要求
@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))

# 註冊要求，含密碼強度檢查
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        account = request.form["account"]
        clinic = request.form["clinic"]

        # 檢查是否重複帳號或 email
        if User.query.filter_by(username=username).first():
            flash("使用者名稱已被註冊")
            return render_template("register.html")
        if User.query.filter_by(email=email).first():
            flash("電子郵件已被註冊")
            return render_template("register.html")

        # 密碼不一致
        if password != confirm_password:
            flash("兩次輸入的密碼不一致")
            return render_template("register.html")

        # 密碼強度檢查（至少8字元，含大小寫與數字）
        import re
        if not re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$", password):
            flash("密碼需至少8位，包含大寫、小寫與數字")
            return render_template("register.html")

        # 建立帳號
        hashed_pw = generate_password_hash(password)
        new_user = User(
            username=username,
            email=email,
            password=hashed_pw,
            account=account,
            clinic=clinic
        )
        db.session.add(new_user)
        db.session.commit()

        session["user_id"] = new_user.id
        flash("註冊成功，歡迎使用！")
        return redirect(url_for("home"))

    return render_template("register.html")

# 前往首頁
@app.route("/home")
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user = User.query.get(session["user_id"])
    return render_template("home.html", user=user)

# 患者列表頁面
@app.route("/patients")
def patients():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    user = User.query.get(session["user_id"])
    patient_list = Patient.query.filter_by(counselor_id=user.id).all()
    return render_template("patients.html", user=user, patients=patient_list)

# 完整音檔上傳頁面
@app.route("/predict_upload")
def predict_upload():
    # 檢查使用者是否登入
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    user = User.query.get(session["user_id"])
    return render_template("index.html", user=user)

# 前往錄音頁面
@app.route("/record")
def record():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user = User.query.get(session["user_id"])
    return render_template("index-audio.html", user=user)

# 新增患者頁面
@app.route("/patients/add", methods=["GET", "POST"])
def add_patient():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    if request.method == "POST":
        name = request.form["name"]
        age = int(request.form["age"])
        gender = request.form["gender"]
        note = request.form["note"]

        new_patient = Patient(
            name=name,
            age=age,
            gender=gender,
            note=note,
            counselor_id=session["user_id"]
        )
        db.session.add(new_patient)
        db.session.commit()
        return redirect(url_for("patients"))

    return render_template("add_patient.html")

# 產出文件報告
@app.route("/generate_report", methods=["POST"])
def generate_report():
    data = request.json
    emotion_summary = data.get("pie_chart", {})
    emotion_sequence = data.get("line_chart", [])
    patient_name = data.get("patient_name", "Unknown")

    # 簡易 HTML 模板
    html_content = f"""
    <html>
    <head><meta charset='utf-8'><style>
        body {{ font-family: Arial; padding: 20px; }}
        h2 {{ color: #3366cc; }}
        .section {{ margin-bottom: 30px; }}
    </style></head>
    <body>
        <h2>情緒分析報告</h2>
        <div class="section">
            <strong>患者姓名：</strong> {patient_name}<br>
            <strong>分析時間：</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
        <div class="section">
            <h3>情緒圓餅圖</h3>
            <img src="{data.get('pie_image')}" width="300">
         </div>
        <div class="section">
            <h3>時間序列折線圖</h3>
            <img src="{data.get('line_image')}" width="300">
        </div>
    </body></html>
    """

    # 轉成 PDF
    pdf_buffer = BytesIO()
    HTML(string=html_content).write_pdf(pdf_buffer)
    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=f"emotion_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf",
        mimetype="application/pdf"
    )
# 音檔預測
@app.route("/predict", methods=["POST"])
def predict():
    """
    接收音檔並回傳情緒分析結果（餅圖 + 折線圖資料）
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = file.filename.lower()

    # 如果是 webm 檔案就轉成 wav
    if filename.endswith(".webm"):
        try:
            print("Converting webm to wav...")
            audio = AudioSegment.from_file(file, format="webm")
            wav_io = BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)
            file_stream = wav_io
        except Exception as e:
            return jsonify({"error": f"Failed to convert webm: {str(e)}"}), 500
    else:
        # 直接從記憶體處理檔案，不儲存到硬碟
        file_stream = BytesIO(file.read())

    pie_chart, line_chart = analyzer.analyze(file_stream)

    return jsonify({
        "pie_chart": pie_chart,
        "line_chart": line_chart
    }), 200

if __name__ == "__main__":
    with app.app_context():
        db.create_all() # 確保資料庫和表格已建立
    app.run(debug=True)
