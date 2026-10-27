import os
import secrets

from flask import Flask
from flask_login import LoginManager, login_required, current_user
from auth_models import db, User
from auth_routes import auth_bp

def create_app():
    app = Flask(__name__)
    secret_key = os.environ.get("FLASK_SECRET_KEY")
    if not secret_key:
        if os.environ.get("FLASK_ENV") == "production":
            raise RuntimeError(
                "FLASK_SECRET_KEY must be set in production; refusing to start "
                "with an ephemeral key (sessions will not survive restarts)."
            )
        secret_key = secrets.token_hex(32)
    app.config["SECRET_KEY"] = secret_key
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///auth.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = "auth.signin"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    app.register_blueprint(auth_bp)

    @app.route("/admin/dashboard")
    @login_required
    def admin_dashboard():
        if current_user.role != "admin":
            return "Forbidden", 403
        return "Admin dashboard"

    @app.route("/user/dashboard")
    @login_required
    def user_dashboard():
        return "User dashboard"

    with app.app_context():
        db.create_all()

    return app

if __name__ == "__main__":
    create_app().run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")