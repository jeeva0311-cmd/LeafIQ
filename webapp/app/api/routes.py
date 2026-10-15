"""API blueprint — JSON endpoints for AJAX use."""
from flask import Blueprint, jsonify, current_app
from flask_login import login_required, current_user
from webapp.app.models import Prediction

api_bp = Blueprint("api", __name__)


@api_bp.route("/predictions")
@login_required
def get_predictions():
    preds = (current_user.predictions
             .order_by(Prediction.created_at.desc())
             .limit(50).all())
    return jsonify([p.to_dict() for p in preds])


@api_bp.route("/predictions/<int:pred_id>")
@login_required
def get_prediction(pred_id: int):
    from webapp.app import db
    pred = db.session.get(Prediction, pred_id)
    if not pred or pred.user_id != current_user.id:
        return jsonify({"error": "Not found"}), 404
    return jsonify(pred.to_dict())


@api_bp.route("/benchmark")
def get_benchmark():
    """Return verified benchmark results (no auth required — public data)."""
    return jsonify(current_app.config["BENCHMARK"])


@api_bp.route("/stats")
@login_required
def get_stats():
    total = current_user.predictions.count()
    per_class = {}
    for p in current_user.predictions.all():
        per_class[p.predicted_class] = per_class.get(p.predicted_class, 0) + 1
    return jsonify({"total": total, "per_class": per_class})
