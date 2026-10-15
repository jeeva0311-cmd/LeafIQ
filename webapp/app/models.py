"""Database models for LeafIQ webapp."""
import json
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from webapp.app import db, login_manager

# ── Health-score thresholds (documented, never presented as diagnosis) ────
HEALTH_SCORE_THRESHOLDS = {
    # (label, min_confidence, description)
    "Likely Healthy":   0.80,   # Healthy class predicted at >= 80% confidence
    "Possible Disease": 0.60,   # Disease class predicted at >= 60% confidence
    "Uncertain":        0.0,    # Anything else
}

HEALTHY_CLASS = "Tomato_Healthy"


def compute_health_score(predicted_class: str, confidence: float) -> dict:
    """
    Return a plant-health indicator based on predicted class and confidence.

    Rules (documented thresholds):
      - 'Likely Healthy'   : class == Tomato_Healthy AND confidence >= 0.80
      - 'Possible Disease' : class != Tomato_Healthy AND confidence >= 0.60
      - 'Uncertain'        : everything else (low confidence or ambiguous)

    IMPORTANT: This is NOT a medical/agronomic diagnosis. It is a UI
    indicator derived from model output. The confidence thresholds are
    based on the LeafIQ test-set calibration (Phase 4): 89.5% of correct
    CNN predictions occur with >= 95% confidence; the >=80% / >=60% cutoffs
    are conservative UI labels only.
    """
    label: str
    color: str
    icon: str
    note: str

    if predicted_class == HEALTHY_CLASS and confidence >= 0.80:
        label = "Likely Healthy"
        color = "#27ae60"
        icon  = "✓"
        note  = ("The model predicted Healthy with high confidence. "
                 "Continue routine monitoring.")
    elif predicted_class != HEALTHY_CLASS and confidence >= 0.60:
        label = "Possible Disease"
        color = "#e67e22"
        icon  = "⚠"
        note  = ("A disease class was predicted with moderate-to-high confidence. "
                 "Inspect the leaf and consider treatment.")
    else:
        label = "Uncertain"
        color = "#7f8c8d"
        icon  = "?"
        note  = ("The model's confidence is low. The result may not be reliable. "
                 "Submit a clearer image or consult an expert.")

    return {
        "label":      label,
        "color":      color,
        "icon":       icon,
        "note":       note,
        "confidence": round(confidence * 100, 1),
        "threshold_note": (
            "Thresholds: Healthy ≥80% → Likely Healthy; "
            "Disease ≥60% → Possible Disease; else Uncertain. "
            "These are UI labels, not agronomic diagnoses."
        ),
    }


# ── User model ────────────────────────────────────────────────────────────

class User(UserMixin, db.Model):
    __tablename__ = "users"

    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username      = db.Column(db.String(64),  unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime,
                               default=lambda: datetime.now(timezone.utc))
    is_active     = db.Column(db.Boolean, default=True)

    predictions = db.relationship("Prediction", backref="user",
                                   lazy="dynamic",
                                   cascade="all, delete-orphan")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def __repr__(self) -> str:
        return f"<User {self.email}>"


@login_manager.user_loader
def load_user(user_id: int):
    return db.session.get(User, int(user_id))


# ── Prediction record ─────────────────────────────────────────────────────

class Prediction(db.Model):
    __tablename__ = "predictions"

    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    filename        = db.Column(db.String(256), nullable=False)
    original_name   = db.Column(db.String(256), nullable=False)
    model_used      = db.Column(db.String(32),  nullable=False)
    predicted_class = db.Column(db.String(64),  nullable=False, index=True)
    confidence      = db.Column(db.Float,        nullable=False)
    probabilities   = db.Column(db.Text,         nullable=False)   # JSON
    inference_ms    = db.Column(db.Float,         nullable=True)
    gradcam_file    = db.Column(db.String(256),   nullable=True)
    created_at      = db.Column(db.DateTime,
                                 default=lambda: datetime.now(timezone.utc),
                                 index=True)

    # ── instance helpers ──────────────────────────────────────────────────

    def probs_dict(self) -> dict:
        return json.loads(self.probabilities)

    def health_score(self) -> dict:
        return compute_health_score(self.predicted_class, self.confidence)

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "filename":        self.filename,
            "original_name":   self.original_name,
            "model_used":      self.model_used,
            "predicted_class": self.predicted_class,
            "confidence":      round(self.confidence * 100, 2),
            "probabilities":   self.probs_dict(),
            "inference_ms":    self.inference_ms,
            "gradcam_file":    self.gradcam_file,
            "created_at":      self.created_at.strftime("%Y-%m-%d %H:%M"),
            "health_score":    self.health_score(),
        }

    # ── class-level filtered query ────────────────────────────────────────

    @classmethod
    def filtered_query(cls, user_id: int,
                       cls_filter: str  = "",
                       date_from: str   = "",
                       date_to: str     = "",
                       conf_min: float  = 0.0,
                       conf_max: float  = 1.0,
                       model_filter: str = "",
                       search: str      = ""):
        """
        Return a SQLAlchemy query with all supplied filters applied.
        All parameters are optional and safe against injection (parameterized).

        Args:
            user_id      : required — scopes to this user only
            cls_filter   : exact predicted_class name, or "" for all
            date_from    : "YYYY-MM-DD" lower bound on created_at
            date_to      : "YYYY-MM-DD" upper bound on created_at
            conf_min     : float 0–1, minimum confidence (inclusive)
            conf_max     : float 0–1, maximum confidence (inclusive)
            model_filter : "custom_cnn" | "cnn_xgboost" | "" for all
            search       : substring match on original_name (case-insensitive)
        """
        from datetime import datetime as dt

        q = cls.query.filter_by(user_id=user_id)

        if cls_filter and cls_filter.strip():
            q = q.filter(cls.predicted_class == cls_filter.strip())

        if date_from and date_from.strip():
            try:
                df = dt.strptime(date_from.strip(), "%Y-%m-%d")
                q  = q.filter(cls.created_at >= df)
            except ValueError:
                pass  # silently ignore malformed date

        if date_to and date_to.strip():
            try:
                # End of day
                dt_obj = dt.strptime(date_to.strip(), "%Y-%m-%d")
                dt_end = dt_obj.replace(hour=23, minute=59, second=59)
                q      = q.filter(cls.created_at <= dt_end)
            except ValueError:
                pass

        # Clamp confidence bounds to [0,1]
        conf_min = max(0.0, min(1.0, float(conf_min)))
        conf_max = max(0.0, min(1.0, float(conf_max)))
        if conf_min > 0.0:
            q = q.filter(cls.confidence >= conf_min)
        if conf_max < 1.0:
            q = q.filter(cls.confidence <= conf_max)

        if model_filter and model_filter.strip() in ("custom_cnn", "cnn_xgboost"):
            q = q.filter(cls.model_used == model_filter.strip())

        if search and search.strip():
            q = q.filter(cls.original_name.ilike(f"%{search.strip()}%"))

        return q.order_by(cls.created_at.desc())

    # ── similar-case finder ───────────────────────────────────────────────

    @classmethod
    def similar_cases(cls, user_id: int, predicted_class: str,
                      exclude_id: int, confidence: float,
                      limit: int = 5) -> list:
        """
        Find up to `limit` previous predictions for this user that share
        the same predicted_class, ordered by confidence proximity to
        the current prediction.

        Strategy: same-class predictions, sorted by |conf_current - conf_row|.
        This surfaces cases the model was similarly certain about, giving
        a meaningful visual reference without requiring stored embeddings.

        These are labelled in the UI as "Reference Cases", NOT as
        confirmations of the current diagnosis.
        """
        # Fetch all same-class predictions excluding the current one
        candidates = (
            cls.query
               .filter_by(user_id=user_id, predicted_class=predicted_class)
               .filter(cls.id != exclude_id)
               .all()
        )
        if not candidates:
            return []

        # Sort by confidence proximity
        candidates.sort(key=lambda p: abs(p.confidence - confidence))
        return candidates[:limit]
