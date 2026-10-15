"""LeafIQ Flask application factory."""
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

db            = SQLAlchemy()
login_manager = LoginManager()
csrf_protect  = CSRFProtect()


def create_app(config_name: str = "development") -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")

    from webapp.config import config as cfg_map
    app.config.from_object(cfg_map[config_name])

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Init extensions
    db.init_app(app)
    login_manager.init_app(app)
    csrf_protect.init_app(app)   # enables csrf_token() in all templates

    login_manager.login_view            = "auth.login"
    login_manager.login_message         = "Please sign in to access the dashboard."
    login_manager.login_message_category = "info"

    # Register blueprints
    from webapp.app.auth.routes      import auth_bp
    from webapp.app.dashboard.routes import dashboard_bp
    from webapp.app.api.routes       import api_bp
    from webapp.app.main.routes      import main_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp,      url_prefix="/auth")
    app.register_blueprint(dashboard_bp, url_prefix="/dashboard")
    app.register_blueprint(api_bp,       url_prefix="/api")

    # Exempt API endpoints from CSRF (they use session auth, no forms)
    csrf_protect.exempt(api_bp)

    # Register new Phase 8 blueprints
    from webapp.app.dashboard import routes as _  # ensure evaluate/monitor registered

    with app.app_context():
        db.create_all()

    return app
