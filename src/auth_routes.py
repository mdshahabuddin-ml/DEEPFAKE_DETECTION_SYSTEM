from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from auth_models import db, User

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        role = request.form.get("role", "user")

        if role not in {"admin", "user"}:
            role = "user"

        exists = User.query.filter((User.username == username) | (User.email == email)).first()
        if exists:
            flash("User already exists")
            return redirect(url_for("auth.signup"))

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            role=role,
        )
        db.session.add(user)
        db.session.commit()
        flash("Account created")
        return redirect(url_for("auth.signin"))

    return render_template("signup.html")

@auth_bp.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid credentials")
            return redirect(url_for("auth.signin"))

        login_user(user)
        return redirect(url_for("admin_dashboard") if user.role == "admin" else url_for("user_dashboard"))

    return render_template("signin.html")

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.signin"))