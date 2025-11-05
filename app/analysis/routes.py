from flask import Blueprint, request, jsonify, session, send_file, current_app
from .. import db
from ..models import User
from ..services import audio_processor, ml_models, ai_services
import io
import traceback
from datetime import datetime
from io import BytesIO
from weasyprint import HTML
from pathlib import Path

analysis_bp = Blueprint('analysis', __name__, template_folder='../templates')

# 取得 emotion_analyzer 實例 (可以優化，例如放在 app context)
def _get_analyzer():
    logger = current_app.logger
    config = current_app.config
    return ml_models.EmotionAnalyzer(logger, config)

@analysis_bp.route("/predict", methods=["POST"])
def predict():
    stage = "start"
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file_bytes = io.BytesIO(request.files["file"].read())
        filename = request.files["file"].filename or ""
        
        # === 呼叫 Service Layer ===
        stage = "decode_audio"
        y, sr = audio_processor.decode_audio_to_16k_mono(file_bytes, filename=filename)
        current_app.logger.info("[predict] decoded: len=%d sr=%d", len(y), sr)

        transcript_text = ""
        try:
            stage = "whisper"
            wm = ml_models.get_whisper(current_app.logger, current_app.config)
            asr_result = wm.transcribe(y, language="zh", fp16=False, condition_on_previous_text=False)
            transcript_text = (getattr(asr_result, "text", None) or asr_result.get("text") or "").strip()
            session["last_transcript"] = transcript_text
        except Exception as e:
            current_app.logger.warning("[Whisper] 轉文字失敗：%r", e)
            session["last_transcript"] = ""

        stage = "analyze"
        analyzer = _get_analyzer()
        pie_chart, line_chart = analyzer.analyze(y) # 呼叫 Service
        current_app.logger.info("[predict] analyze done: %s", pie_chart)
        # ==========================

        return jsonify({
            "pie_chart": pie_chart,
            "line_chart": line_chart,
            "transcript": transcript_text
        }), 200

    except Exception as e:
        tb = traceback.format_exc()
        current_app.logger.error("[predict] failed at stage=%s: %r\n%s", stage, e, tb)
        # 直接回 JSON，前端不要只顯示「發生錯誤」，把 error/stage 印出來
        return jsonify({"error": str(e), "stage": stage, "trace": tb}), 500

@analysis_bp.route("/suggestion", methods=["POST"])
def suggestion():
    data = request.get_json()
    transcript = data.get("transcript") or session.get("last_transcript", "")
    
    # === 呼叫 Service Layer ===
    suggestion_text = ai_services.get_emotion_suggestion(
        data.get("emotion_stats", {}),
        data.get("line_series", []),
        transcript,
        current_app.logger
    )
    # ==========================
    
    print("=== 模型產出內容如下 ===")
    print(suggestion_text)
    print("=======================")
    return jsonify({"suggestion": suggestion_text})

# 產出文件報告
@analysis_bp.route("/generate_report", methods=["POST"])
def generate_report():
    data = request.json
    patient_name = data.get("patient_name", "Unknown")
    suggestion_html = data.get("suggestion", "無建議內容。")
    
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
    <head><meta charset='utf-8'>
    <style>
        @font-face {{
          font-family: "NotoSansTC";
          src: url("static/fonts/NotoSansTC-Regular.ttf") format("truetype");
        }}
        html, body {{
          font-family: "NotoSansTC", "Arial", sans-serif;
          font-size: 14px;
        }}
        body {{ padding: 20px; }}
        h2 {{ color: #3366cc; margin-top: 0; }}
        .section {{ margin-bottom: 24px; }}
        p {{ line-height: 1.6; }}
        img {{ display:block; margin: 0 auto; }}
      </style></head>
    <body>
        <h2>情緒分析報告</h2>
        <div class="section">
            <strong>{subject_label}：</strong> {subject_value}<br>
            <strong>分析時間：</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
        <div class="section">
            <h3>情緒圓餅圖</h3>
            <img src="{data.get('pie_image')}" style="width:300px; height:auto; display:block; margin:auto;">
         </div>
        <div class="section">
            <h3>時間序列折線圖</h3>
            <img src="{data.get('line_image')}" width="300">
        </div>
        <div class="section">
            <h3>情緒建議</h3>
            <p>{suggestion_html}</p>
        </div>
    </body></html>
    """

    # 轉成 PDF
    pdf_buffer = BytesIO()
    base_url = str(Path(current_app.root_path))
    HTML(string=html_content, base_url=base_url).write_pdf(pdf_buffer)
    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=f"emotion_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf",
        mimetype="application/pdf"
    )