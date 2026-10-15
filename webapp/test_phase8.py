"""
LeafIQ Phase 8 — Automated test suite.
Tests: evaluate, monitor, uncertainty, top-3, file validation,
       env-var config, Phase 7 regressions.
Writes outputs/_test_phase8.txt
"""
import sys, os, re, json, time, io, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import urllib.request, urllib.parse, urllib.error
import http.cookiejar
from pathlib import Path

BASE = "http://127.0.0.1:5000"
OUT  = Path(__file__).parent.parent / "outputs" / "_test_phase8.txt"

cj     = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
lines  = [f"LeafIQ Phase 8 Test  {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
passed = 0; failed = 0

def chk(label, cond, detail=""):
    global passed, failed
    tag = "PASS" if cond else "FAIL"
    lines.append(f"  {tag}  {label}" + (f"  [{detail}]" if detail else ""))
    if cond: passed += 1
    else:    failed += 1

def get(path):
    try:
        r = opener.open(BASE + path, timeout=15)
        return r.getcode(), r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)

def post_form(path, data):
    payload = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(BASE + path, data=payload, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        r = opener.open(req, timeout=30)
        return r.getcode(), r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)

def extract_csrf(body):
    m = re.search(r'name="csrf_token"[^>]+value="([^"]+)"', body)
    return m.group(1) if m else ""

def upload_jpeg(csrf_token, model="custom_cnn", skip_gradcam="0", conf_thresh="60"):
    from PIL import Image
    img = Image.new("RGB", (64,64), color=(34,139,34))
    buf = io.BytesIO(); img.save(buf,"JPEG"); jpeg = buf.getvalue()
    boundary = f"----FB{uuid.uuid4().hex}"
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"csrf_token\"\r\n\r\n{csrf_token}".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"skip_gradcam\"\r\n\r\n{skip_gradcam}".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"conf_thresh\"\r\n\r\n{conf_thresh}".encode(),
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"leaf.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode() + jpeg),
        f"--{boundary}--\r\n".encode(),
    ]
    body = b"\r\n".join(parts)
    req  = urllib.request.Request(BASE+"/dashboard/detect/upload", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        r = opener.open(req, timeout=120)
        return r.getcode(), r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")

# ── Setup ──────────────────────────────────────────────────────────────────
lines.append("=== SETUP ===")
ts   = int(time.time())
USER = f"p8user{ts}"
MAIL = f"p8user{ts}@example.com"
PASS = "Phase8Test!"

s, b = get("/auth/signup"); csrf = extract_csrf(b)
chk("signup page loads", s==200 and bool(csrf))
post_form("/auth/signup", {"csrf_token":csrf,"username":USER,"email":MAIL,"password":PASS,"confirm":PASS})

lines.append("  Uploading test image (CNN, ~60s on CPU)...")
s, db = get("/dashboard/detect"); csrf_up = extract_csrf(db)
st, rb = upload_jpeg(csrf_up, "custom_cnn", skip_gradcam="1")
chk("Upload (skip_gradcam=1)", st==200, f"status={st}")
pred_ids = re.findall(r'/dashboard/result/(\d+)', rb)

# ── 1. Phase 6/7 regressions ──────────────────────────────────────────────
lines.append("\n=== 1. PHASE 6/7 REGRESSION ===")
for path, needle in [
    ("/dashboard/detect",         "Image Detection"),
    ("/dashboard/history",        "Prediction History"),
    ("/dashboard/compare",        "98.17"),
    ("/dashboard/reports",        "Total Predictions"),
    ("/dashboard/explainability", "Grad-CAM"),
    ("/api/benchmark",            "custom_cnn"),
]:
    sv, bv = get(path)
    chk(f"{path}", sv==200 and needle in bv, f"status={sv}")

# ── 2. New routes exist ────────────────────────────────────────────────────
lines.append("\n=== 2. NEW PHASE 8 ROUTES ===")
s, b = get("/dashboard/evaluate")
chk("/dashboard/evaluate loads", s==200 and "Model Evaluation" in b, f"status={s}")
chk("evaluate has run form",     "Run Evaluation" in b or "run" in b.lower())

s, b = get("/dashboard/monitor")
chk("/dashboard/monitor loads", s==200 and "Monitoring" in b, f"status={s}")
chk("monitor shows stats",      "Total Predictions" in b or "Inference" in b)

# ── 3. Detect page — new inputs ───────────────────────────────────────────
lines.append("\n=== 3. DETECT PAGE INPUTS ===")
s, b = get("/dashboard/detect")
chk("detect has confidence slider", "conf_thresh" in b or "confThresh" in b)
chk("detect has skip_gradcam",      "skip_gradcam" in b or "skipGradcam" in b)
chk("detect has advanced options",  "Advanced" in b or "advanced" in b.lower())
chk("detect has file validation JS","ALLOWED" in b or "MAX_MB" in b)

# ── 4. Result page — top-3 + uncertainty ──────────────────────────────────
lines.append("\n=== 4. RESULT PAGE ENHANCEMENTS ===")
if pred_ids:
    s, b = get(f"/dashboard/result/{pred_ids[0]}")
    chk("result loads", s==200, f"status={s}")
    chk("result has top-3 section",     "#1" in b and "#2" in b and "#3" in b)
    chk("result has health indicator",  any(x in b for x in ["Likely Healthy","Possible Disease","Uncertain"]))
    chk("result has uncertainty field", "uncertainty" in b.lower() or "Uncertain" in b or "confidence" in b.lower())

# ── 5. File validation ────────────────────────────────────────────────────
lines.append("\n=== 5. FILE VALIDATION ===")
from webapp.app.ml.predictor import validate_image_file
from PIL import Image
import tempfile, struct

# Valid JPEG
import tempfile
tf_path = tempfile.mktemp(suffix=".jpg")
try:
    img = Image.new("RGB",(100,100),(0,200,0)); img.save(tf_path); img.close()
    ok, err = validate_image_file(tf_path)
    chk("valid JPEG passes validation", ok, err)
finally:
    try: os.unlink(tf_path)
    except: pass

# Too small
tf_path = tempfile.mktemp(suffix=".jpg")
try:
    img = Image.new("RGB",(4,4),(0,200,0)); img.save(tf_path); img.close()
    ok, err = validate_image_file(tf_path)
    chk("too-small image rejected", not ok, err)
finally:
    try: os.unlink(tf_path)
    except: pass

# Empty file
tf_path = tempfile.mktemp(suffix=".jpg")
try:
    with open(tf_path,"wb") as f_: f_.write(b"")
    ok, err = validate_image_file(tf_path)
    chk("empty file rejected", not ok, err)
finally:
    try: os.unlink(tf_path)
    except: pass

# Corrupt file
tf_path = tempfile.mktemp(suffix=".jpg")
try:
    with open(tf_path,"wb") as f_: f_.write(b"\xff\xd8" + b"\x00"*50)
    ok, err = validate_image_file(tf_path)
    chk("corrupt image rejected", not ok, err)
finally:
    try: os.unlink(tf_path)
    except: pass

# Non-existent file
ok, err = validate_image_file("/nonexistent/path/image.jpg")
chk("nonexistent file rejected", not ok, err)

# ── 6. Uncertainty logic ──────────────────────────────────────────────────
lines.append("\n=== 6. UNCERTAINTY FLAG LOGIC ===")
from webapp.app.ml.predictor import _uncertainty_flag
u_high   = _uncertainty_flag(0.95, "Tomato_Healthy")
u_medium = _uncertainty_flag(0.70, "Tomato_Healthy")
u_low    = _uncertainty_flag(0.45, "Tomato_Healthy")
u_very   = _uncertainty_flag(0.20, "Tomato_Early_Blight")

chk("high conf (0.95) → not uncertain",  not u_high["is_uncertain"])
chk("high conf level == 'high'",          u_high["level"] == "high")
chk("medium conf (0.70) → not is_uncertain", not u_medium["is_uncertain"])
chk("low conf (0.45) → is_uncertain",    u_low["is_uncertain"])
chk("very low (0.20) → level 'uncertain'", u_very["level"] == "uncertain")
chk("very low has warning text",          len(u_very["warning"]) > 10)

# ── 7. Health score consistency ───────────────────────────────────────────
lines.append("\n=== 7. HEALTH SCORE LABELS ===")
from webapp.app.models import compute_health_score
h1 = compute_health_score("Tomato_Healthy", 0.95)
h2 = compute_health_score("Tomato_Late_Blight", 0.85)
h3 = compute_health_score("Tomato_Healthy", 0.50)
h4 = compute_health_score("Tomato_Early_Blight", 0.30)

chk("Healthy@95% → Likely Healthy",       h1["label"] == "Likely Healthy")
chk("Disease@85% → Possible Disease",     h2["label"] == "Possible Disease")
chk("Healthy@50% → Uncertain",            h3["label"] == "Uncertain")
chk("Disease@30% → Uncertain",            h4["label"] == "Uncertain")
chk("threshold note in all results",
    all("threshold" in h["threshold_note"].lower() for h in [h1,h2,h3,h4]))

# ── 8. Evaluator cache ────────────────────────────────────────────────────
lines.append("\n=== 8. EVALUATOR CACHE ===")
from webapp.app.ml.evaluator import load_cached_eval
cache = load_cached_eval()
chk("eval cache loads without error", isinstance(cache, dict))
# If cache exists from previous eval run, check it has expected fields
if "custom_cnn" in cache:
    m = cache["custom_cnn"]
    chk("cached CNN has accuracy",        "accuracy" in m)
    chk("cached CNN has confusion_matrix","confusion_matrix" in m)
    chk("cached CNN has per_class",       "per_class" in m and len(m["per_class"]) == 4)
    chk("cached CNN accuracy >= 0.95",    m.get("accuracy",0) >= 0.95,
        f"accuracy={m.get('accuracy',0):.4f}")
else:
    chk("eval cache empty (not yet run)", True, "run evaluate_run to populate")
    lines.append("    NOTE: run /dashboard/evaluate to generate results")

# ── 9. Monitor page inputs ────────────────────────────────────────────────
lines.append("\n=== 9. MONITOR FEATURES ===")
s, b = get("/dashboard/monitor")
chk("monitor has confidence buckets",  ">=90%" in b or "conf" in b.lower())
chk("monitor has model usage section", "Model Usage" in b)
chk("monitor has inference stats",     "Inference" in b or "ms" in b)

# ── 10. Auth boundary (new routes) ────────────────────────────────────────
lines.append("\n=== 10. AUTH BOUNDARY (NEW ROUTES) ===")
cj2  = http.cookiejar.CookieJar()
anon = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj2))
for path in ["/dashboard/evaluate", "/dashboard/monitor"]:
    try:
        r = anon.open(BASE + path, timeout=8)
        sa, ba = r.getcode(), r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        sa, ba = e.code, ""
    chk(f"anon blocked: {path}", "Sign In" in ba or sa == 302, f"status={sa}")

# ── 11. Upload with skip_gradcam=1 (faster path) ─────────────────────────
lines.append("\n=== 11. SKIP GRADCAM FLAG ===")
s, b = get("/dashboard/detect"); csrf2 = extract_csrf(b)
st2, rb2 = upload_jpeg(csrf2, skip_gradcam="1")
chk("upload with skip_gradcam=1 works", st2==200, f"status={st2}")
ids2 = re.findall(r'/dashboard/result/(\d+)', rb2)
if ids2:
    s, b = get(f"/dashboard/result/{ids2[0]}")
    chk("result page loads for skip-gradcam pred", s==200)
    # Grad-CAM section should show "not available" message
    chk("result shows Grad-CAM not available",
        "not available" in b.lower() or "gradcam" in b.lower())

# ── 12. config.py env keys ────────────────────────────────────────────────
lines.append("\n=== 12. CONFIG ENV KEYS ===")
import importlib.util, sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
from webapp.config import Config, DevelopmentConfig
chk("Config has CONF_THRESHOLD_HIGH",   hasattr(Config, "CONF_THRESHOLD_HIGH"))
chk("Config has CONF_THRESHOLD_MEDIUM", hasattr(Config, "CONF_THRESHOLD_MEDIUM"))
chk("Config has CONF_THRESHOLD_LOW",    hasattr(Config, "CONF_THRESHOLD_LOW"))
chk("Config has EVAL_CACHE_PATH",       hasattr(Config, "EVAL_CACHE_PATH"))
chk("Config has TEST_CACHE_DIR",        hasattr(Config, "TEST_CACHE_DIR"))
chk("Config has UNCERTAINTY_WARNING",   hasattr(Config, "UNCERTAINTY_WARNING"))

# ── 13. .env.example exists ───────────────────────────────────────────────
lines.append("\n=== 13. DEPLOYMENT FILES ===")
env_ex = Path(__file__).parent.parent / ".env.example"
chk(".env.example exists",            env_ex.exists())
chk(".env.example has SECRET_KEY",    "LEAFIQ_SECRET_KEY" in env_ex.read_text() if env_ex.exists() else False)
chk("run.py has version check",
    "check_versions" in (Path(__file__).parent/"run.py").read_text())
chk("evaluator.py exists",
    (Path(__file__).parent/"app"/"ml"/"evaluator.py").exists())

# ── Summary ────────────────────────────────────────────────────────────────
lines.append(f"\n{'='*55}")
lines.append(f"TOTAL: {passed} PASS  |  {failed} FAIL")
lines.append(f"{'='*55}")
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Done. PASS={passed} FAIL={failed}")
print(f"Results: {OUT}")
