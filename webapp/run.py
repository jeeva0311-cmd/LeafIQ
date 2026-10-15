"""
LeafIQ web application entry point.

Development:
    cd "C:\\Users\\JEEVA PRAKASINI G\\Documents\\Resume project\\LeafIQ"
    .\\venv311\\Scripts\\Activate.ps1
    python webapp\\run.py

Production (example with Waitress):
    set LEAFIQ_ENV=production
    set LEAFIQ_SECRET_KEY=<your-secret-key>
    python webapp\\run.py
"""
import sys, os
from pathlib import Path

# ── Add project root to sys.path ──────────────────────────────────────────
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── Load .env if python-dotenv is available ───────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent.parent / ".env"
    if _env.exists():
        load_dotenv(str(_env))
        print(f"  Loaded .env from {_env}")
except ImportError:
    pass   # python-dotenv not installed; rely on actual env vars

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")

# ── Dependency version check ──────────────────────────────────────────────
def _check_versions():
    import importlib.metadata as md
    required = {
        "flask":          "3.0",
        "flask-sqlalchemy":"3.0",
        "flask-login":    "0.6",
        "flask-wtf":      "1.0",
        "tensorflow":     "2.13",
        "xgboost":        "2.0",
        "fpdf2":          "2.7",
    }
    warnings = []
    for pkg, min_ver in required.items():
        try:
            ver = md.version(pkg)
            # Simple major.minor check
            def _v(s): return tuple(int(x) for x in s.split(".")[:2])
            if _v(ver) < _v(min_ver):
                warnings.append(f"  {pkg} {ver} < recommended {min_ver}")
        except Exception:
            warnings.append(f"  {pkg} NOT INSTALLED")
    return warnings

_ver_warnings = _check_versions()

# ── Create app ────────────────────────────────────────────────────────────
config_name = os.environ.get("LEAFIQ_ENV", "development")

from webapp.app import create_app
app = create_app(config_name)

if __name__ == "__main__":
    print("=" * 60)
    print(f"  LeafIQ — Plant Disease Detection Web App")
    print(f"  Environment : {config_name}")
    print(f"  URL         : http://127.0.0.1:5000")
    if _ver_warnings:
        print("  Dependency warnings:")
        for w in _ver_warnings: print(w)
    print("=" * 60)

    if config_name == "production":
        # Try Waitress for production
        try:
            from waitress import serve
            print("  Using Waitress WSGI server")
            serve(app, host="0.0.0.0", port=5000)
        except ImportError:
            print("  Waitress not installed; falling back to Flask dev server")
            print("  Install for production: pip install waitress")
            app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
