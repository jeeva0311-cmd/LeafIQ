import importlib.metadata as md, sys
pkgs = ["flask","flask_sqlalchemy","flask_login","werkzeug",
        "flask_wtf","wtforms","pillow","tensorflow","xgboost"]
out = []
for p in pkgs:
    pip_name = {"flask_sqlalchemy":"Flask-SQLAlchemy","flask_login":"Flask-Login",
                "flask_wtf":"Flask-WTF","wtforms":"WTForms"}.get(p, p)
    try: v = md.version(pip_name); out.append(f"OK   {p:<22} {v}")
    except Exception: out.append(f"MISS {p:<22} not installed")
from pathlib import Path
(Path(__file__).parent.parent/"outputs"/"_web_pkgs.txt").write_text(
    "\n".join(out)+"\n", encoding="utf-8")
print("\n".join(out))
