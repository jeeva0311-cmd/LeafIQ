"""LeafIQ webapp configuration.

Environment variables read at startup (never hard-coded in source):
  LEAFIQ_SECRET_KEY     — Flask secret key (REQUIRED in production)
  LEAFIQ_ENV            — "development" | "production" (default: development)
  LEAFIQ_DB_URI         — SQLAlchemy DB URI (default: SQLite in webapp/)
  LEAFIQ_UPLOAD_FOLDER  — absolute path for uploads
  LEAFIQ_MAX_UPLOAD_MB  — max upload size in MB (default: 16)
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # LeafIQ/
WEBAPP_ROOT  = Path(__file__).resolve().parent          # LeafIQ/webapp/

_DEFAULT_SECRET = "leafiq-dev-secret-2026-change-in-prod"


class Config:
    # ── Flask core ─────────────────────────────────────────────────────────
    SECRET_KEY = os.environ.get("LEAFIQ_SECRET_KEY", _DEFAULT_SECRET)
    DEBUG      = False
    TESTING    = False

    # ── Database ────────────────────────────────────────────────────────────
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "LEAFIQ_DB_URI",
        f"sqlite:///{WEBAPP_ROOT / 'leafiq.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ── Uploads ─────────────────────────────────────────────────────────────
    UPLOAD_FOLDER = os.environ.get(
        "LEAFIQ_UPLOAD_FOLDER",
        str(WEBAPP_ROOT / "app" / "static" / "uploads")
    )
    MAX_CONTENT_LENGTH = int(os.environ.get("LEAFIQ_MAX_UPLOAD_MB", "16")) * 1024 * 1024
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
    MAX_IMAGE_PIXELS   = 178_956_970    # PIL decompression-bomb limit

    # ── ML model paths ──────────────────────────────────────────────────────
    CNN_CHECKPOINT    = os.environ.get(
        "LEAFIQ_CNN_CHECKPOINT",
        str(PROJECT_ROOT / "models" / "custom_cnn_best.keras")
    )
    XGB_PKL           = os.environ.get(
        "LEAFIQ_XGB_PKL",
        str(PROJECT_ROOT / "models" / "xgboost_best.pkl")
    )
    CNN_INPUT_HW      = 64
    XGB_EMBEDDING_DIM = 256

    # ── Confidence thresholds ──────────────────────────────────────────────
    # Applied at prediction time and in the UI.
    # Values are fractions (0–1); UI shows percentages.
    CONF_THRESHOLD_HIGH   = 0.80   # above this → high-confidence result
    CONF_THRESHOLD_MEDIUM = 0.60   # above this → medium confidence
    CONF_THRESHOLD_LOW    = 0.40   # below this → show uncertainty warning
    CONF_DEFAULT_MIN_UI   = 0      # default min-confidence slider on detect page
    UNCERTAINTY_WARNING   = (
        "This prediction has low confidence. "
        "The image may be unclear, out-of-focus, or not a tomato leaf. "
        "Consider re-uploading a clearer image."
    )

    # ── Evaluation cache ────────────────────────────────────────────────────
    EVAL_CACHE_PATH = str(PROJECT_ROOT / "outputs" / "eval_cache.json")
    TEST_CACHE_DIR  = str(PROJECT_ROOT / "outputs" / "cache")

    # ── Verified benchmark results (Phase 4 / Phase 5) ─────────────────────
    BENCHMARK = {
        "custom_cnn": {
            "display": "Custom CNN",
            "accuracy": 98.17,
            "f1": 98.07,
            "precision": 97.89,
            "recall": 98.28,
            "errors": 15,
            "total": 821,
            "params": 323172,
            "size_kb": 3902,
            "inference_ms": 74.4,
            "status": "recommended",
        },
        "mobilenetv2": {
            "display": "MobileNetV2-A",
            "accuracy": 61.88,
            "f1": 55.39,
            "precision": 63.81,
            "recall": 55.67,
            "errors": 313,
            "total": 821,
            "params": 2422852,
            "size_kb": 11343,
            "inference_ms": 324.5,
            "status": "underfitted",
        },
        "cnn_xgboost": {
            "display": "CNN + XGBoost",
            "accuracy": 98.29,
            "f1": 98.21,
            "precision": 98.15,
            "recall": 98.28,
            "errors": 14,
            "total": 821,
            "params": "N/A",
            "size_kb": 835,
            "inference_ms": 76.0,
            "status": "experimental",
        },
    }

    CLASS_NAMES = [
        "Tomato_Healthy",
        "Tomato_Early_Blight",
        "Tomato_Late_Blight",
        "Tomato_Leaf_Mold",
    ]

    CLASS_DISPLAY = {
        "Tomato_Healthy":      "Healthy",
        "Tomato_Early_Blight": "Early Blight",
        "Tomato_Late_Blight":  "Late Blight",
        "Tomato_Leaf_Mold":    "Leaf Mold",
    }

    CLASS_INFO = {
        "Tomato_Healthy": {
            "color": "#27ae60",
            "icon": "✓",
            "description": "No disease detected. The leaf appears healthy.",
            "action": "Continue regular monitoring and care.",
        },
        "Tomato_Early_Blight": {
            "color": "#e67e22",
            "icon": "⚠",
            "description": "Alternaria solani infection detected. Brown concentric ring lesions.",
            "action": "Apply fungicide (copper-based). Remove affected leaves. Improve airflow.",
        },
        "Tomato_Late_Blight": {
            "color": "#c0392b",
            "icon": "✕",
            "description": "Phytophthora infestans detected. Water-soaked dark lesions.",
            "action": "Urgent: Apply systemic fungicide immediately. Isolate affected plants.",
        },
        "Tomato_Leaf_Mold": {
            "color": "#8e44ad",
            "icon": "◈",
            "description": "Passalora fulva infection. Yellow patches with grey-brown mould.",
            "action": "Reduce humidity. Improve ventilation. Apply appropriate fungicide.",
        },
    }


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False

    @property
    def SECRET_KEY(self):  # type: ignore[override]
        key = os.environ.get("LEAFIQ_SECRET_KEY", "")
        if not key or key == _DEFAULT_SECRET:
            raise RuntimeError(
                "LEAFIQ_SECRET_KEY environment variable must be set in production. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        return key


config = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
    "default":     DevelopmentConfig,
}
