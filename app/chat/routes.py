from flask import Blueprint, request, jsonify, session, render_template, redirect, url_for, current_app
from .. import db
from ..models import User, ChatRecord
from ..services import ai_services  
from datetime import datetime
import re

# 建立 Blueprint
chat_bp = Blueprint('chat', __name__, template_folder='../templates')

@chat_bp.route("/", methods=["GET"]) 
def chat_page():
    if "user_id" not in session:
        return redirect(url_for("auth.login_user")) 
    user = db.session.get(User, session["user_id"])
    role = session.get("role", None)
    return render_template("chat.html", user=user, role=role)

@chat_bp.route("/ai", methods=["POST"]) 
def chat_ai():
    data = request.get_json()
    user_msg = data.get("message", "")
    history = data.get("history", [])
    date = data.get("date")
    if not user_msg:
        return jsonify({"reply": "請輸入訊息"}), 400

    # === 呼叫 Service Layer ===
    try:
        reply = ai_services.get_chat_reply(
            user_msg, 
            history, 
            current_app.logger
        )
    except Exception as e:
        current_app.logger.error(f"[Chat AI] Gemini call failed: {e}")
        return jsonify({"reply": "AI 助理發生錯誤，請稍後再試。"}), 500
    # ==========================
        
    # 存進資料庫
    if not date or not re.match(r"^\d{4}-\d{2}-\d{2}$", str(date)):
        date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M")
    user_id = session.get("user_id")
    
    db.session.add(ChatRecord(user_id=user_id, date=date, sender="user", text=user_msg, time=time))
    db.session.add(ChatRecord(user_id=user_id, date=date, sender="ai", text=reply, time=time))
    db.session.commit()

    return jsonify({"reply": reply})

@chat_bp.route("/history", methods=["GET"])
def chat_history():
    date = request.args.get("date")
    user_id = session.get("user_id")
    records = ChatRecord.query.filter_by(user_id=user_id, date=date).all()
    out = [{"text": r.text, "sender": r.sender, "time": r.time} for r in records]
    return jsonify({"history": out})

@chat_bp.route("/dates")
def chat_dates():
    user_id = session.get("user_id")
    dates = db.session.query(ChatRecord.date).filter_by(user_id=user_id).distinct().all()
    return jsonify({"dates": [d[0] for d in dates]})

@chat_bp.route("/delete_date", methods=["POST"])
def delete_chat_date():
    data = request.get_json()
    date = data.get("date")
    user_id = session.get("user_id")
    if not date or not user_id:
        return jsonify({"success": False, "error": "缺少日期或未登入"}), 400
    ChatRecord.query.filter_by(user_id=user_id, date=date).delete()
    db.session.commit()
    return jsonify({"success": True})

@chat_bp.route("/patient_view/<int:user_id>")
def patient_chat(user_id):
    if "user_id" not in session or session.get("role") != "counselor":
        return redirect(url_for("auth.login_counselor"))

    # 確認這個 user_id 的患者真的屬於自己
    patient = User.query.filter_by(id=user_id, role="user", counselor_id=session["user_id"]).first_or_404()

    # 抓取他的聊天紀錄
    records = ChatRecord.query.filter_by(user_id=user_id).order_by(ChatRecord.date, ChatRecord.time).all()

    return render_template("patient_chat.html", patient=patient, records=records)
    
@chat_bp.route("/summarize", methods=["POST"]) 
def summarize_chat():
    # 權限檢查：確保是登入的諮商師
    if "user_id" not in session or session.get("role") != "counselor":
        return jsonify({"error": "權限不足"}), 403

    data = request.get_json()
    patient_id = data.get("patient_id")
    if not patient_id:
        return jsonify({"error": "未提供患者 ID"}), 400

    # 安全性檢查
    patient = User.query.filter_by(id=patient_id, role="user", counselor_id=session["user_id"]).first()
    if not patient:
        return jsonify({"error": "找不到指定的患者或權限不符"}), 404

    # 獲取該患者的所有聊天紀錄
    records = ChatRecord.query.filter_by(user_id=patient_id).order_by(ChatRecord.date, ChatRecord.time).all()
    if not records:
        return jsonify({"summary": "這位使用者還沒有任何聊天紀錄。"})

    # 格式化
    chat_log = ""
    for r in records:
        sender = '使用者' if r.sender == 'user' else 'AI'
        chat_log += f"{sender}: {r.text}\n"

    # === 呼叫 Service Layer ===
    try:
        summary_text = ai_services.get_chat_summary(
            chat_log, 
            current_app.logger
        )
    except Exception as e:
        current_app.logger.error(f"[Chat Summary] Gemini call failed: {e}")
        return jsonify({"summary": "AI 總結功能發生錯誤，請稍後再試。"}), 500
    # ==========================
        
    return jsonify({"summary": summary_text})