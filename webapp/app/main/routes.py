"""Main blueprint — public pages: home."""
from flask import Blueprint, render_template, redirect, url_for
from flask_login import current_user

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.detect"))
    return render_template("main/home.html")


@main_bp.route("/health")
def health():
    return {"status": "ok", "service": "LeafIQ"}, 200
