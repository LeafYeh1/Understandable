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
import json
import whisper

# 文字建議 gemini
from Phi_3mini import generate_response

# 初始化 Flask 應用
app = Flask(__name__)

# 初始化 Whisper 模型（可選 tiny, base, small, medium, large）
whisper_model = whisper.load_model("small")

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
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    account = db.Column(db.String(100), nullable=False)
    clinic = db.Column(db.String(300))
    role = db.Column(db.String(20), nullable=False)  # 角色類型（counselor/user）

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
        delta_mfcc = librosa.feature.delta(mfcc)
        log_mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels=64))
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)

        min_frames = min(mfcc.shape[1], delta_mfcc.shape[1], log_mel.shape[1], rms.shape[1], zcr.shape[1])
        
        # 裁切時間軸長度一致
        features = np.vstack([
            mfcc[:, :min_frames],
            delta_mfcc[:, :min_frames],
            log_mel[:, :min_frames],
            rms[:, :min_frames],
            zcr[:, :min_frames]
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
@app.route("/")
def choose_role():
    return render_template("choose_role.html")

@app.route("/login_counselor", methods=["GET","POST"])
def login_counselor():
    if request.method == "POST":
        account = request.form["account"]
        password = request.form["password"]
        user = User.query.filter_by(account=account, role="counselor").first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["role"] = "counselor" # 設定角色為諮商師
            return redirect(url_for("home"))
        else:
            flash("帳號或密碼錯誤，或角色不符")
    return render_template("login_counselor.html")

@app.route("/login_user", methods=["GET","POST"])
def login_user():
    if request.method == "POST":
        account = request.form["account"]
        password = request.form["password"]
        user = User.query.filter_by(account=account, role="user").first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["role"] = "user" # 設定角色為使用者
            return redirect(url_for("home"))
        else:
            flash("帳號或密碼錯誤，或角色不符")
    return render_template("login_user.html")
    
# 登出要求
@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.clear()
    return redirect(url_for("choose_role"))

# 註冊要求，含密碼強度檢查
@app.route("/register_user", methods=["GET", "POST"])
def register_user():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        account = request.form["account"]

        # 檢查是否重複帳號或 email
        if User.query.filter_by(account=account).first():
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
            email=email,
            password=hashed_pw,
            account=account,
            role="user"  # 設定角色為使用者
        )
        db.session.add(new_user)
        db.session.commit()

        session["user_id"] = new_user.id
        flash("註冊成功，歡迎使用！")
        return redirect(url_for("home"))

    return render_template("register.html", role="user")

@app.route("/register_counselor", methods=["GET", "POST"])
def register_counselor():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        account = request.form["account"]
        clinic = request.form["clinic"]

        # 檢查是否重複帳號或 email
        if User.query.filter_by(account=account).first():
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
            email=email,
            password=hashed_pw,
            account=account,
            clinic=clinic,
            role="counselor"  # 設定角色為諮商師
        )
        db.session.add(new_user)
        db.session.commit()

        session["user_id"] = new_user.id
        flash("註冊成功，歡迎使用！")
        return redirect(url_for("home"))
    return render_template("register.html", role="counselor")
    
# 前往首頁
@app.route("/home")
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user = User.query.get(session["user_id"])
    role = session.get("role", None)
    return render_template("home.html", user=user, role=role)

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
    patients = Patient.query.filter_by(counselor_id=user.id).all()
    role = session.get("role", None)
    return render_template("index.html", user=user, patients=patients, role=role)

# 前往錄音頁面
@app.route("/record")
def record():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user = User.query.get(session["user_id"])
    patients = Patient.query.filter_by(counselor_id=user.id).all()
    role = session.get("role", None)
    return render_template("index-audio.html", user=user, patients=patients, role=role)

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
    patient_name = data.get("patient_name", "Unknown")
    suggestion = data.get("suggestion", "無建議內容。")
    role = session.get("role", None)

    if role == "counselor":
        subject_label = "患者姓名"
    else:
        subject_label = "使用者帳號"
    print(f"Generating report for {subject_label}: {patient_name}")
    user = db.session.get(User, session["user_id"])
    subject_value = patient_name if role == "counselor" else user.account
    
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
            <strong>{subject_label}：</strong> {subject_value}<br>
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
        <div class="section">
            <h3>情緒建議</h3>
            <p>{suggestion}</p>
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

    # ===== Whisper（純記憶體）=====
    transcript_text = ""
    try:
        file_stream.seek(0)  # very important：把指標拉回開頭
        # 直接用 librosa 從記憶體載入為 16kHz、單聲道的 float32 numpy
        y, sr = librosa.load(file_stream, sr=16000, mono=True)

        # 第一次先假設中文，失敗就自動偵測重試
        asr_result = whisper_model.transcribe(
            y, language="zh", fp16=False, condition_on_previous_text=False
        )
        transcript_text = (asr_result.get("text") or "").strip()
        if not transcript_text:
            asr_result = whisper_model.transcribe(
                y, fp16=False, condition_on_previous_text=False
            )
        transcript_text = (asr_result.get("text") or "").strip()

        print("=== Whisper 語音轉文字結果 ===")
        print(repr(transcript_text))
        print("===========================")
        # 存在 session，讓 suggestion() 可直接取用        
        session["last_transcript"] = transcript_text
        
    except Exception as e:
        print(f"[Whisper] 轉文字失敗：{e}")
        session["last_transcript"] = ""
        
    # ============================
    
    pie_chart, line_chart = analyzer.analyze(file_stream)

    return jsonify({
        "pie_chart": pie_chart,
        "line_chart": line_chart,
        "transcript": transcript_text
    }), 200

@app.route("/suggestion", methods=["POST"])
def suggestion():
    data = request.get_json()
    emotion_stats = data.get("emotion_stats", {})  # pie_chart 統計資料
    line_series = data.get("line_series", [])      # 折線圖情緒序列
    transcript = data.get("transcript") or session.get("last_transcript", "")
    
    # 組合 prompt 給 Gemini
    prompt = [
        {"role": "system", "content": "你是一位心理諮商師，請根據使用者情緒給出建議與舒緩方式，請使用繁體中文、自然口吻，先是一句話建議，再列點具體方法。"},
        {"role": "user", "content": (
            "情緒分布：{'sad': 4, 'ang': 3, 'happy': 0}\n"
            "時間序列：['sad', 'sad', 'ang', 'ang', 'sad']\n"
            "請給出建議與方法。"
        )},
        {"role": "assistant", "content": (
            "你可能正經歷負面情緒，試著溫柔地照顧自己的內在。\n"
            "舒緩方式：\n"
            "1. 嘗試冥想或靜坐 5 分鐘\n"
            "2. 書寫當下感受與想法\n"
            "3. 聆聽放鬆音樂\n"
            "4. 與親近的人談談心情"
        )},
        {"role": "user", "content": (
            f"情緒分布：{json.dumps(emotion_stats, ensure_ascii=False)}\n"
            f"時間序列：{line_series}\n"
            f"語音轉文字（摘要）：{transcript}\n"
            "請給出建議與方法。"
        )}
    ]

    suggestion_text = generate_response(prompt)
    
    print("=== 模型產出內容如下 ===")
    print(suggestion_text)
    print("=======================")
    return jsonify({"suggestion": suggestion_text})

if __name__ == "__main__":
    with app.app_context():
        db.create_all() # 確保資料庫和表格已建立
    app.run(debug=True)