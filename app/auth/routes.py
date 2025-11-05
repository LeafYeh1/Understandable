from flask import Blueprint, request, render_template, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from .. import db  
from ..models import User 

# 建立 Blueprint 物件
auth_bp = Blueprint('auth', __name__, template_folder='../templates')

@auth_bp.route("/login_counselor", methods=["GET","POST"])
def login_counselor():
    if request.method == "POST":
        account = request.form["account"]
        password = request.form["password"]
        user = User.query.filter_by(account=account, role="counselor").first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["role"] = "counselor" # 設定角色為諮商師
            return redirect(url_for("core.home"))
        else:
            flash("帳號或密碼錯誤，或角色不符")
    return render_template("login_counselor.html")

@auth_bp.route("/login_user", methods=["GET","POST"])
def login_user():
    if request.method == "POST":
        account = request.form["account"]
        password = request.form["password"]
        user = User.query.filter_by(account=account, role="user").first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["role"] = "user" # 設定角色為使用者
            return redirect(url_for("core.home"))
        else:
            flash("帳號或密碼錯誤，或角色不符")
    return render_template("login_user.html")
    
# 註冊要求，含密碼強度檢查
@auth_bp.route("/register_user", methods=["GET", "POST"])
def register_user():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        account = request.form["account"]

        # 檢查是否重複帳號或 email
        if User.query.filter_by(account=account).first():
            flash("使用者名稱已被註冊")
            return render_template("register.html", role="user")
        
        if User.query.filter_by(email=email).first():
            flash("電子郵件已被註冊")
            return render_template("register.html", role="user")

        # 密碼不一致
        if password != confirm_password:
            flash("兩次輸入的密碼不一致")
            return render_template("register.html", role="user")

        # 密碼強度檢查（至少8字元，含大小寫與數字）
        import re
        if not re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$", password):
            flash("密碼需至少8位，包含大寫、小寫與數字")
            return render_template("register.html", role="user")

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

        flash("註冊成功，請登入。", "success")
        return redirect(url_for("auth.login_user"))

    return render_template("register.html", role="user")

@auth_bp.route("/register_counselor", methods=["GET", "POST"])
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
            return render_template("register.html", role="counselor")
        if User.query.filter_by(email=email).first():
            flash("電子郵件已被註冊")
            return render_template("register.html", role="counselor")

        # 密碼不一致
        if password != confirm_password:
            flash("兩次輸入的密碼不一致")
            return render_template("register.html", role="counselor")

        # 密碼強度檢查（至少8字元，含大小寫與數字）
        import re
        if not re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$", password):
            flash("密碼需至少8位，包含大寫、小寫與數字")
            return render_template("register.html", role="counselor")

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

        flash("註冊成功，請登入。", "success")
        return redirect(url_for("auth.login_counselor"))
    return render_template("register.html", role="counselor")

# 登出要求
@auth_bp.route("/logout")
def logout():
    session.pop("user_id", None)
    session.clear()
    return redirect(url_for("core.choose_role"))