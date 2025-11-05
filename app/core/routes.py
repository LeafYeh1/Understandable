from flask import Blueprint, request, render_template, redirect, url_for, session, flash, jsonify
from .. import db
from ..models import User, Patient, ChatRecord

# 建立 Blueprint
core_bp = Blueprint('core', __name__, template_folder='../templates')

# 檢查是否已經登入，是的話就導向 home
@core_bp.route("/")
def choose_role():    
    if "user_id" in session:
        return redirect(url_for("core.home"))
    return render_template("choose_role.html")

@core_bp.route("/home")
def home():
    if "user_id" not in session:
        # 如果沒登入，導回角色選擇
        return redirect(url_for("core.choose_role"))
    user = db.session.get(User, session["user_id"])
    role = session.get("role", None)
    return render_template("home.html", user=user, role=role)

# 患者列表頁面 (諮商師)
@core_bp.route("/patients")
def patients():
    if "user_id" not in session or session.get("role") != "counselor":
        flash("請先以諮商師身分登入")
        return redirect(url_for("auth.login_counselor"))
    
    user = User.query.get(session["user_id"])
    patient_list = Patient.query.filter_by(counselor_id=user.id).all()
    return render_template("patients.html", user=user, patients=patient_list)

# 新增患者頁面 (諮商師)
@core_bp.route("/patients/add", methods=["GET", "POST"])
def add_patient():
    if "user_id" not in session or session.get("role") != "counselor":
        return redirect(url_for("auth.login_counselor"))
        
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
        return redirect(url_for("core.patients"))

    return render_template("add_patient.html")

# 完整音檔上傳頁面
@core_bp.route("/predict_upload")
def predict_upload():
    if "user_id" not in session:
        return redirect(url_for("core.choose_role"))
    
    user = User.query.get(session["user_id"])
    patients = []
    if session.get("role") == "counselor":
        patients = Patient.query.filter_by(counselor_id=user.id).all()
        
    role = session.get("role", None)
    return render_template("index.html", user=user, patients=patients, role=role)

# 前往錄音頁面
@core_bp.route("/record")
def record():
    if "user_id" not in session:
        return redirect(url_for("core.choose_role"))
        
    user = User.query.get(session["user_id"])
    patients = []
    if session.get("role") == "counselor":
        patients = Patient.query.filter_by(counselor_id=user.id).all()
        
    role = session.get("role", None)
    return render_template("index-audio.html", user=user, patients=patients, role=role)

# 綁定諮商師 (使用者)
@core_bp.route("/mycounselor", methods=["GET", "POST"])
def my_counselor():
    if "user_id" not in session:
        return redirect(url_for("auth.login_user"))
    if session.get("role") != "user":
        return redirect(url_for("core.home"))

    user = User.query.get(session["user_id"])

    if request.method == "POST":
        counselor_name = request.form.get("counselor_name")
        counselor = User.query.filter_by(account=counselor_name, role="counselor").first()
        if counselor:
            user.counselor_id = counselor.id
            db.session.commit()
            flash(f"已綁定諮商師：{counselor.account}")
            return redirect(url_for("core.my_counselor"))
        else:
            flash("找不到此諮商師名稱")

    counselors = User.query.filter_by(role="counselor").all()
    bound_counselor = None
    if user.counselor_id:
        bound_counselor = User.query.get(user.counselor_id)

    return render_template(
        "mycounselor.html",
        user=user,
        counselors=counselors,
        bound_counselor=bound_counselor
    )

# 諮商師查看 "使用者" 紀錄 
@core_bp.route("/user_records")
def user_records():
    if "user_id" not in session or session.get("role") != "counselor":
        return redirect(url_for("auth.login_counselor"))

    # 找到所有綁定自己的使用者 (User model)
    patients = User.query.filter_by(role="user", counselor_id=session["user_id"]).all()

    return render_template("patients_record.html", patients=patients)

@core_bp.route("/get_user_chat_dates/<int:user_id>")
def get_user_chat_dates(user_id):
    if "user_id" not in session or session.get("role") != "counselor":
        return jsonify([])

    # 安全檢查：確認這個 user_id 是綁定在自己底下的
    patient = User.query.filter_by(id=user_id, role="user", counselor_id=session["user_id"]).first()
    if not patient:
        return jsonify([])

    # 從聊天紀錄取得日期
    dates = db.session.query(ChatRecord.date)\
            .filter_by(user_id=user_id)\
            .distinct()\
            .order_by(ChatRecord.date.desc())\
            .all()
    return jsonify([d[0] for d in dates])

@core_bp.route("/get_user_chat_logs/<int:user_id>/<date>")
def get_user_chat_logs(user_id, date):
    if "user_id" not in session or session.get("role") != "counselor":
        return jsonify([])

    # 安全檢查
    patient = User.query.filter_by(id=user_id, role="user", counselor_id=session["user_id"]).first()
    if not patient:
        return jsonify([])

    records = ChatRecord.query.filter_by(user_id=user_id, date=date)\
                .order_by(ChatRecord.time.asc()).all()

    logs = [{"sender": r.sender, "text": r.text, "time": r.time} for r in records]
    return jsonify(logs)