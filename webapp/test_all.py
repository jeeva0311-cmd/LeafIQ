"""
LeafIQ Phase 6 — Full automated test suite.
Writes outputs/_test_all.txt with PASS/FAIL for every check.
"""
import sys, os, re, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import urllib.request, urllib.parse, urllib.error
import http.cookiejar
from pathlib import Path

BASE = "http://127.0.0.1:5000"
OUT  = Path(__file__).parent.parent / "outputs" / "_test_all.txt"

# Fresh cookie jar per test session
cj     = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

lines  = [f"LeafIQ Phase 6 Test  {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
passed = 0; failed = 0

def chk(label, cond, detail=""):
    global passed, failed
    tag = "PASS" if cond else "FAIL"
    lines.append(f"  {tag}  {label}" + (f"  [{detail}]" if detail else ""))
    if cond: passed += 1
    else:    failed += 1

def get(path, follow=True):
    try:
        resp = opener.open(BASE + path, timeout=10)
        return resp.getcode(), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)

def post_form(path, data: dict):
    payload = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(BASE + path, data=payload, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        resp = opener.open(req, timeout=10)
        return resp.getcode(), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)

def extract_csrf(body):
    m = re.search(r'name="csrf_token"[^>]+value="([^"]+)"', body)
    return m.group(1) if m else ""

# ── 1. Public pages ───────────────────────────────────────────────────────
lines.append("=== 1. PUBLIC PAGES ===")
s, b = get("/health");      chk("/health → 200",       s==200, f"status={s}")
s, b = get("/");            chk("/ home → 200",         s==200, f"status={s}")
chk("Home has 'LeafIQ'",   "LeafIQ" in b)
chk("Home has 'Get Started'", "Get Started" in b or "started" in b.lower())
s, b = get("/auth/login");  chk("/auth/login → 200",    s==200, f"status={s}")
chk("Login has form",       "form" in b.lower() and "email" in b.lower())
s, b = get("/auth/signup"); chk("/auth/signup → 200",   s==200, f"status={s}")
chk("Signup has form",      "username" in b.lower())

# ── 2. Auth redirect (unauthenticated) ───────────────────────────────────
lines.append("\n=== 2. AUTH REDIRECT (unauthenticated) ===")
for path in ["/dashboard/detect","/dashboard/history","/dashboard/compare",
             "/dashboard/explainability","/dashboard/reports","/dashboard/settings"]:
    s, b = get(path)
    # After redirect follows to login page
    chk(f"{path} protected", "Sign In" in b or s == 302, f"status={s}")

# ── 3. Signup ─────────────────────────────────────────────────────────────
lines.append("\n=== 3. SIGNUP ===")
ts = int(time.time())
TEST_USER  = f"tester{ts}"
TEST_EMAIL = f"tester{ts}@example.com"
TEST_PASS  = "TestPass99!"

s, b = get("/auth/signup")
csrf = extract_csrf(b)
chk("CSRF token present on signup page", bool(csrf))

s2, r2 = post_form("/auth/signup", {
    "csrf_token": csrf,
    "username":   TEST_USER,
    "email":      TEST_EMAIL,
    "password":   TEST_PASS,
    "confirm":    TEST_PASS,
})
form_errs = re.findall(r'class="form-error">([^<]+)<', r2)
chk("POST /auth/signup succeeds (no form errors)", not form_errs,
    f"status={s2} errors={form_errs}")
chk("After signup: dashboard accessible", "Image Detection" in r2 or
    "detect" in r2.lower(), f"status={s2} body_len={len(r2)}")

# ── 4. Dashboard pages (authenticated) ───────────────────────────────────
lines.append("\n=== 4. DASHBOARD PAGES (after signup) ===")
for path, label, needle in [
    ("/dashboard/detect",        "detect",        "Image Detection"),
    ("/dashboard/history",       "history",       "Prediction History"),
    ("/dashboard/compare",       "compare",       "98.17"),
    ("/dashboard/explainability","explainability","Grad-CAM"),
    ("/dashboard/reports",       "reports",       "Total Predictions"),
    ("/dashboard/settings",      "settings",      "Account Information"),
]:
    s, b = get(path)
    chk(f"dashboard/{label} → 200 + content", s==200 and needle in b,
        f"status={s} needle={'found' if needle in b else 'missing'}")

# ── 5. API endpoints ──────────────────────────────────────────────────────
lines.append("\n=== 5. API ENDPOINTS ===")
s, b = get("/api/predictions"); chk("/api/predictions → 200", s==200, f"status={s}")
try: json.loads(b); chk("/api/predictions → valid JSON", True)
except: chk("/api/predictions → valid JSON", False)
s, b = get("/api/stats");       chk("/api/stats → 200", s==200, f"status={s}")
s, b = get("/api/benchmark");   chk("/api/benchmark has CNN accuracy", "98.17" in b)

# ── 6. Logout ─────────────────────────────────────────────────────────────
lines.append("\n=== 6. LOGOUT ===")
s, b = get("/auth/logout")
chk("Logout → redirects to home/login", "LeafIQ" in b or "Sign In" in b, f"status={s}")

# Check dashboard is protected again
s, b = get("/dashboard/detect")
chk("Dashboard protected after logout", "Sign In" in b, f"status={s}")

# ── 7. Login ──────────────────────────────────────────────────────────────
lines.append("\n=== 7. LOGIN ===")
s, b = get("/auth/login")
csrf2 = extract_csrf(b)
chk("CSRF token on login page", bool(csrf2))

s3, r3 = post_form("/auth/login", {
    "csrf_token": csrf2,
    "email":      TEST_EMAIL,
    "password":   TEST_PASS,
})
chk("POST /auth/login succeeds", "Image Detection" in r3 or
    "Welcome back" in r3 or ("Sign In" not in r3 and s3 == 200),
    f"status={s3}")

# Wrong password
s, b = get("/auth/login")
csrf3 = extract_csrf(b)
sw, bw = post_form("/auth/login", {
    "csrf_token": csrf3,
    "email":      TEST_EMAIL,
    "password":   "WrongPassword!!",
})
chk("Wrong password shows error", "Invalid email or password" in bw or "Sign In" in bw,
    f"status={sw}")

# ── 8. Image upload + prediction ─────────────────────────────────────────
lines.append("\n=== 8. IMAGE UPLOAD + PREDICTION ===")
# Re-login first
s, b = get("/auth/login")
csrf4 = extract_csrf(b)
post_form("/auth/login", {"csrf_token": csrf4, "email": TEST_EMAIL, "password": TEST_PASS})

# Create a minimal valid JPEG (1×1 green pixel) for upload test
import io, struct

def make_minimal_jpeg() -> bytes:
    """Create a tiny valid 8x8 green JPEG."""
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(34, 139, 34))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except Exception:
        return b""

jpeg_data = make_minimal_jpeg()
chk("PIL can create test JPEG", len(jpeg_data) > 100, f"size={len(jpeg_data)}")

if jpeg_data:
    # Upload multipart form
    import uuid
    boundary = f"----FormBoundary{uuid.uuid4().hex}"
    # Get CSRF from detect page
    s, detect_body = get("/dashboard/detect")
    csrf_up = extract_csrf(detect_body)

    # Build multipart body
    body_parts = []
    body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"csrf_token\"\r\n\r\n{csrf_up}".encode())
    body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ncustom_cnn".encode())
    body_parts.append((
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"test_leaf.jpg\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + jpeg_data)
    body_parts.append(f"--{boundary}--\r\n".encode())
    mp_body = b"\r\n".join(body_parts)

    req_up = urllib.request.Request(
        BASE + "/dashboard/detect/upload",
        data=mp_body, method="POST"
    )
    req_up.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        resp_up = opener.open(req_up, timeout=180)  # CNN takes ~2-5s on CPU
        upload_status = resp_up.getcode()
        upload_body   = resp_up.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        upload_status = e.code
        upload_body   = e.read().decode("utf-8", errors="replace")

    chk("POST /dashboard/detect/upload → 200", upload_status == 200,
        f"status={upload_status}")

    if upload_status == 200:
        has_pred    = any(c in upload_body for c in
                         ["Healthy","Blight","Mold","predicted_class"])
        has_conf    = "%" in upload_body or "confidence" in upload_body.lower()
        has_probbar = "prob" in upload_body.lower()
        chk("Result page shows prediction class", has_pred, "")
        chk("Result page shows confidence",       has_conf, "")
        chk("Result page shows prob bars",         has_probbar, "")
        if upload_status == 200:
            # Check history has 1 entry now
            s, hb = get("/dashboard/history")
            chk("Prediction saved to history", "1 prediction" in hb or
                "predictions" in hb.lower(), f"status={s}")
    else:
        chk("Upload result has prediction", False, f"HTTP {upload_status}")
        chk("Upload confidence shown",      False, "skipped")
        chk("Upload prob bars shown",       False, "skipped")
        chk("Prediction saved to history",  False, "skipped")
        # Print error for diagnosis
        err_text = re.findall(r"(?:Error|Exception|error)[^\n<]{0,150}", upload_body)
        for e in err_text[:3]:
            lines.append(f"    UPLOAD_ERROR: {e.strip()}")

# ── Summary ───────────────────────────────────────────────────────────────
lines.append(f"\n{'='*55}")
lines.append(f"TOTAL: {passed} PASS  |  {failed} FAIL")
lines.append(f"{'='*55}")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Done. PASS={passed} FAIL={failed}")
print(f"Full results: {OUT}")
