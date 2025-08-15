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
import whisper # 語音轉文字

# 文字建議 gemini
from Qwen import local_llm_generate
import json as pyjson 

# 高效音訊讀取
import soundfile as sf
import webrtcvad
import io

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

def decode_audio_to_16k_mono(file_bytes: io.BytesIO, filename: str | None = None):
    """
    優先用 soundfile 一次讀 → 轉單聲道 → 16k。
    若不支援（常見 webm），用 pydub+ffmpeg 轉 WAV 再讀
    回傳: (y: float32, sr=16000)
    """
    file_bytes.seek(0)
    try:
        # 首選：soundfile 直接讀成 float32
        y, sr = sf.read(file_bytes, dtype='float32', always_2d=False)
    except Exception:
        # soundfile 讀失敗（可能 webm），用 pydub 轉
        file_bytes.seek(0)
        fmt = "webm" if (filename and filename.lower().endswith(".webm")) else None
        audio = AudioSegment.from_file(file_bytes, format=fmt) if fmt else AudioSegment.from_file(file_bytes)
        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        y, sr = sf.read(wav_io, dtype='float32', always_2d=False)

    # 轉單聲道
    if y.ndim > 1:
        y = np.mean(y, axis=1)
        
    # 重取樣到 16k
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="kaiser_fast")
        sr = 16000
    return y.astype(np.float32), sr

class SafeSileroVAD:
    """
    以 WebRTC VAD 取代 Silero（避免 Windows 上 torch/DLL 問題）
    先偵測語音的時間區段（timestamps），只對語音區段切窗與特徵
    介面維持 speech_timestamps(y) -> [(start, end)]（樣本索引）
    """
    def __init__(self, sr=16000, aggressiveness=2, frame_ms=20):
        self.sr = sr
        self.vad = webrtcvad.Vad(aggressiveness)  # aggressiveness: 0~3，越大越嚴格
        self.frame_ms = frame_ms                   # WebRTC VAD 只能處理 10/20/30ms 的短幀（16-bit PCM）

    def _float_to_int16(self, y: np.ndarray) -> bytes:
        y_clip = np.clip(y, -1.0, 1.0)
        pcm16 = (y_clip * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    def speech_timestamps(self, y: np.ndarray, min_speech_s=0.3, max_silence_s=0.5):
        """
        先把 float32 轉成 int16 PCM bytes，依序切成固定長度的 frame。
        用 VAD 判斷每個 frame 是否為語音，將連續語音 frame 併成一段。
        容許短暫靜音（max_silence_s）仍屬同一段，避免說話中間停頓被切斷。
        若抓不到語音，最後退回 energy-based（librosa.effects.split），避免空結果。
        """
        pcm_bytes = self._float_to_int16(y)

        bytes_per_sample = 2
        samples_per_frame = int(self.sr * (self.frame_ms / 1000.0))
        frame_bytes = samples_per_frame * bytes_per_sample

        frames = []
        for i in range(0, len(pcm_bytes) - frame_bytes + 1, frame_bytes):
            frame = pcm_bytes[i:i + frame_bytes]
            is_speech = self.vad.is_speech(frame, self.sr)
            frames.append(is_speech)

        ts_list = []
        in_speech = False
        start_idx = 0
        silence_run = 0
        max_silence_frames = int(max_silence_s / (self.frame_ms / 1000.0))
        min_speech_frames = int(min_speech_s / (self.frame_ms / 1000.0))

        for idx, flag in enumerate(frames):
            if flag: # 目前是語音
                if not in_speech:
                    in_speech = True
                    start_idx = idx
                silence_run = 0
            else:    # 目前是靜音
                if in_speech:
                    silence_run += 1
                    # 靜音累積超過上限 → 前一段語音結束
                    if silence_run > max_silence_frames:
                        end_idx = idx - silence_run
                        if end_idx - start_idx >= min_speech_frames:
                            s = start_idx * samples_per_frame
                            e = end_idx * samples_per_frame
                            ts_list.append((s, e))
                        in_speech = False
                        silence_run = 0
                        
        # 收尾：若最後仍在語音狀態 → 補上一段
        if in_speech:
            end_idx = len(frames)
            if end_idx - start_idx >= min_speech_frames:
                s = start_idx * samples_per_frame
                e = end_idx * samples_per_frame
                ts_list.append((s, e))

        # 若抓不到語音，退回 energy-based，避免空結果
        if not ts_list:
            intervals = librosa.effects.split(y, top_db=30)
            ts_list = [(int(s), int(e)) for s, e in intervals]

        return ts_list


def window_slices_from_ts(y, sr, ts_list, win_s=3.0, hop_s=1.5, max_windows_per_ts=999):
    """
    片段長度固定、重疊 50%（3s / 1.5s）→ 對模型最友善，也能產生足夠樣本數。    
    """
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    slices = []
    for (s, e) in ts_list:
        start = s
        count = 0
        # 覆疊切窗
        while start + win <= e and count < max_windows_per_ts:
            slices.append((start, start + win))
            start += hop
            count += 1
        # 視情況補尾窗：避免在語音尾端剛好少一點點而沒有被取到
        if e - win > s and (e - win) - (start - hop) > (0.3 * win):
            slices.append((e - win, e))
    return slices

def extract_features_batch(y_list, sr=16000, max_len=500):
    """把多段 y 做成一個 batch 特徵：(B, 500, feat_dim)"""
    feats = []
    n_fft = 512
    hop_length = 128
    for y in y_list:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15, n_fft=n_fft, hop_length=hop_length)
        delta_mfcc = librosa.feature.delta(mfcc)
        log_mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=64))
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)

        min_frames = min(mfcc.shape[1], delta_mfcc.shape[1], log_mel.shape[1], rms.shape[1], zcr.shape[1])
        f = np.vstack([mfcc[:, :min_frames], delta_mfcc[:, :min_frames], log_mel[:, :min_frames], rms[:, :min_frames], zcr[:, :min_frames]])
        if f.shape[1] < max_len:
            f = np.pad(f, ((0, 0), (0, max_len - f.shape[1])), mode='constant')
        else:
            f = f[:, :max_len]
        feats.append(f.T)  # (500, feat_dim)
    X = np.stack(feats, axis=0)
    return X


class EmotionAnalyzer:
    def __init__(self, model_path="emotion_model.h5", sr=16000, win_s=3.0, hop_s=1.5):
        # 關掉 compile 警告：推論不需要 metrics
        self.model = load_model(model_path, compile=False)
        self.sr = sr
        self.win_s = win_s
        self.hop_s = hop_s
        self.vad = SafeSileroVAD(sr=sr, aggressiveness=2, frame_ms=20)

    def analyze(self, y_all: np.ndarray):
        """
        1) WebRTC VAD 找語音區段
        2) 各區段滑動窗（預設 win=3s, hop=1.5s）
        3) 批次特徵 → 一次 model.predict
        4) 回傳：({emotion: count}, [label序列])
        """
        ts_list = self.vad.speech_timestamps(y_all)
        if not ts_list:
            return {c: 0 for c in class_names}, []

        slices = window_slices_from_ts(y_all, self.sr, ts_list, self.win_s, self.hop_s)
        win_samps = int(self.win_s * self.sr)
        chunks = [y_all[s:e] for (s, e) in slices if e - s == win_samps]
        if not chunks:
            return {c: 0 for c in class_names}, []

        X = extract_features_batch(chunks, sr=self.sr, max_len=500)
        preds = self.model.predict(X, verbose=0)  # 一次推論
        labels_idx = np.argmax(preds, axis=1)
        labels = [class_names[i] for i in labels_idx]

        counts = {c: 0 for c in class_names}
        for lb in labels:
            counts[lb] += 1
        return counts, labels
    
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
    user = db.session.get(User, session["user_id"])
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
    一次解碼 → Whisper 與情緒分析共用 y → VAD + 滑動窗 + 批次推論
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    up = request.files["file"]
    filename = up.filename or ""
    file_bytes = io.BytesIO(up.read())

    # 1) 只解碼一次（16k/mono/float32）
    y, sr = decode_audio_to_16k_mono(file_bytes, filename=filename)

    # 2) Whisper 轉文字（openai/whisper）
    transcript_text = ""
    try:
        asr_result = whisper_model.transcribe(y, language="zh", fp16=False, condition_on_previous_text=False)
        transcript_text = (asr_result.get("text") or "").strip()
        if not transcript_text:
            asr_result = whisper_model.transcribe(y, fp16=False, condition_on_previous_text=False)
            transcript_text = (asr_result.get("text") or "").strip()
        session["last_transcript"] = transcript_text
        print(f"[Whisper] 轉文字成功：{transcript_text}")  # 只顯示前50字
    except Exception as e:
        print(f"[Whisper] 轉文字失敗：{e}")
        session["last_transcript"] = ""

    # 3) 情緒分析：VAD → 滑動窗 → 批次 → 一次 predict
    pie_chart, line_chart = analyzer.analyze(y)

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
    
    # 避免過長：可視需要截斷
    def shorten(txt, max_chars=800):
        return (txt[:max_chars] + "…") if txt and len(txt) > max_chars else (txt or "")

    transcript_short = shorten(transcript, 800)

    # 用單一大 prompt（/api/generate 比較合適）
    prompt = (
        "你是一位心理諮商師，請用繁體中文回覆，並嚴格遵守以下格式與規範：\n"
        "任務：根據使用者的情緒統計、時間序列，以及語音轉文字摘要，"
        "先提供 1 句同理且務實的建議，再列出 3–4 點具體、可執行的調節方法。\n\n"
        f"【情緒分布】{pyjson.dumps(emotion_stats, ensure_ascii=False)}\n"
        f"【時間序列】{line_series}\n"
        f"【語音轉文字（摘要）】{transcript_short}\n\n"
        "請遵守以下規範：\n"
        "1) 以情緒分布中佔比最高的情緒作為主要關注點，若情緒差異不大，則以語音內容判斷主要情緒\n"
        "2) 必須同時參考情緒分布、時間序列趨勢與語音內容，不能只依賴其中之一\n"
        "3) 口吻自然、友善、非醫療診斷\n"
        "4) 方法務實清楚，務必為可立即採取的行動，例如「深呼吸 3 次」、「散步 5 分鐘」、「寫下 3 件感恩的事」。避免使用空泛詞語如「放輕鬆」。\n"
        "5) 若時間序列顯示情緒惡化，先安撫再給穩定情緒的方法；若顯示好轉，先肯定再提供維持的方法。\n"
        "6) 輸出格式必須固定如下（不得省略標題與編號）：\n"
        "【同理建議】\n(一句溫暖務實的話)\n"
        "【情緒調節方法】\n1. ...\n2. ...\n3. ...\n4. ...\n"
    )
    suggestion_text = local_llm_generate(prompt, num_ctx=2048, temperature=0.6) or "（暫無建議，請稍後再試）"
    
    print("=== 模型產出內容如下 ===")
    print(suggestion_text)
    print("=======================")
    return jsonify({"suggestion": suggestion_text})

if __name__ == "__main__":
    with app.app_context():
        db.create_all() # 確保資料庫和表格已建立
    app.run(debug=True)