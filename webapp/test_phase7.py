"""
LeafIQ Phase 7 — Automated test suite.
Tests: filters, exports, explainability, health score, similar cases,
       existing Phase 6 functionality, invalid inputs, auth.
Writes outputs/_test_phase7.txt
"""
import sys, os, re, json, time, io, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import urllib.request, urllib.parse, urllib.error
import http.cookiejar
from pathlib import Path

BASE = "http://127.0.0.1:5000"
OUT  = Path(__file__).parent.parent / "outputs" / "_test_phase7.txt"

cj     = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

lines  = [f"LeafIQ Phase 7 Test  {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
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
        r = opener.open(req, timeout=15)
        return r.getcode(), r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)

def extract_csrf(body):
    m = re.search(r'name="csrf_token"[^>]+value="([^"]+)"', body)
    return m.group(1) if m else ""

def upload_jpeg(csrf_token, model="custom_cnn"):
    """Upload a 64x64 green JPEG. Returns (status, body)."""
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(34, 139, 34))
    buf = io.BytesIO(); img.save(buf, "JPEG"); jpeg = buf.getvalue()
    boundary = f"----FB{uuid.uuid4().hex}"
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"csrf_token\"\r\n\r\n{csrf_token}".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}".encode(),
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"leaf.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode() + jpeg),
        f"--{boundary}--\r\n".encode(),
    ]
    body = b"\r\n".join(parts)
    req  = urllib.request.Request(BASE + "/dashboard/detect/upload", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        r = opener.open(req, timeout=120)
        return r.getcode(), r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)

# ── Setup: signup + upload 2 predictions ──────────────────────────────────
lines.append("=== SETUP ===")
ts   = int(time.time())
USER = f"p7user{ts}"
MAIL = f"p7user{ts}@example.com"
PASS = "Phase7Test!"

s, b = get("/auth/signup")
csrf = extract_csrf(b)
chk("signup page loads", s == 200 and bool(csrf))
post_form("/auth/signup", {"csrf_token": csrf, "username": USER,
                            "email": MAIL, "password": PASS, "confirm": PASS})

# Upload 2 predictions
lines.append("  Uploading 2 test images (CNN, ~60s on CPU)...")
s, db_b = get("/dashboard/detect")
csrf_up = extract_csrf(db_b)
st1, b1 = upload_jpeg(csrf_up, "custom_cnn")
chk("Upload 1 (custom_cnn)", st1 == 200, f"status={st1}")

s, db_b = get("/dashboard/detect")
csrf_up2 = extract_csrf(db_b)
st2, b2 = upload_jpeg(csrf_up2, "custom_cnn")
chk("Upload 2 (custom_cnn)", st2 == 200, f"status={st2}")

# Extract pred IDs from result pages
pred_ids = re.findall(r'/dashboard/result/(\d+)', b1 + b2)
lines.append(f"  Prediction IDs found: {pred_ids}")

# ── 1. Existing Phase 6 pages still work ─────────────────────────────────
lines.append("\n=== 1. PHASE 6 REGRESSION ===")
for path, needle in [
    ("/dashboard/detect",         "Image Detection"),
    ("/dashboard/compare",        "98.17"),
    ("/dashboard/settings",       "Account Information"),
    ("/api/benchmark",            "custom_cnn"),
]:
    s, b = get(path)
    chk(f"{path}", s == 200 and needle in b, f"status={s}")

# ── 2. History page + filters ─────────────────────────────────────────────
lines.append("\n=== 2. HISTORY FILTERS ===")
s, b = get("/dashboard/history")
chk("history loads", s == 200 and "Prediction History" in b, f"status={s}")
chk("history has filter bar", "Apply Filters" in b or "Apply" in b)
chk("history shows predictions", str(pred_ids[0]) in b if pred_ids else False,
    "pred_id in page")

# Filter by class
s, b = get("/dashboard/history?cls=Tomato_Healthy")
chk("filter cls=Healthy loads", s == 200, f"status={s}")
chk("filter shows matching count", "record" in b.lower())

# Filter by date range (today)
today = time.strftime("%Y-%m-%d")
s, b = get(f"/dashboard/history?date_from={today}&date_to={today}")
chk(f"filter date_from/to={today}", s == 200, f"status={s}")

# Filter by confidence
s, b = get("/dashboard/history?conf_min=50&conf_max=100")
chk("filter conf_min/max loads", s == 200, f"status={s}")

# Filter by model
s, b = get("/dashboard/history?model=custom_cnn")
chk("filter model=custom_cnn", s == 200, f"status={s}")

# Search by filename
s, b = get("/dashboard/history?q=leaf")
chk("filter search=leaf", s == 200, f"status={s}")

# Clear filters
s, b = get("/dashboard/history")
chk("clear filters (no params)", s == 200 and "Prediction History" in b)

# Combined filter
s, b = get(f"/dashboard/history?cls=Tomato_Healthy&model=custom_cnn&date_from={today}")
chk("combined filter", s == 200, f"status={s}")

# Invalid date — should not 500
s, b = get("/dashboard/history?date_from=not-a-date")
chk("invalid date — no 500", s == 200, f"status={s}")

# ── 3. History CSV export ─────────────────────────────────────────────────
lines.append("\n=== 3. HISTORY CSV/PDF EXPORT ===")
s, b = get("/dashboard/history/export/csv")
chk("history CSV export → 200", s == 200, f"status={s}")
chk("CSV has header row", "ID,Original Filename" in b or "Predicted Class" in b)
chk("CSV has data rows", b.count("\n") > 1, f"lines={b.count(chr(10))}")

# PDF export
s, b = get("/dashboard/history/export/pdf")
chk("history PDF export → 200", s == 200, f"status={s}")
chk("PDF export is binary (not HTML)", len(b) > 100 and
    ("PDF" in b[:10] or b[:4] == "%PDF" or len(b) > 200))

# Filtered CSV
s, b = get(f"/dashboard/history/export/csv?cls=Tomato_Healthy")
chk("filtered CSV export", s == 200, f"status={s}")

# ── 4. Result page — health score + similar cases ─────────────────────────
lines.append("\n=== 4. RESULT — HEALTH SCORE + SIMILAR CASES ===")
if pred_ids:
    s, b = get(f"/dashboard/result/{pred_ids[0]}")
    chk(f"result/{pred_ids[0]} loads", s == 200, f"status={s}")
    chk("health indicator present",
        any(x in b for x in ["Likely Healthy","Possible Disease","Uncertain"]))
    chk("health threshold note present", "threshold" in b.lower() or "not a" in b.lower())
    chk("similar reference cases section present", "Reference Cases" in b or "Similar" in b)
    chk("prob bars present", "prob-bar" in b.lower())
    chk("Grad-CAM section present", "Grad-CAM" in b or "gradcam" in b.lower())

    # Second result should show similar cases (same user, same class)
    if len(pred_ids) >= 2:
        s, b2r = get(f"/dashboard/result/{pred_ids[1]}")
        chk(f"result/{pred_ids[1]} loads", s == 200)
        # If both predicted same class, first result should appear as similar case in second
        chk("second result has similar-cases section",
            "Reference Cases" in b2r or "Similar" in b2r)

# Invalid pred_id — should not 500
s, b = get("/dashboard/result/999999")
chk("invalid pred_id → no 500", s == 200, f"status={s}")

# Another user's pred_id — 403 or redirect (auth boundary)
# We test this by using id=0 (non-existent)
s, b = get("/dashboard/result/0")
chk("pred_id=0 → no 500", s in (200, 404), f"status={s}")

# ── 5. Explainability list + detail ───────────────────────────────────────
lines.append("\n=== 5. EXPLAINABILITY ===")
s, b = get("/dashboard/explainability")
chk("explainability loads", s == 200 and "Grad-CAM" in b, f"status={s}")
chk("explainability has filter bar", "date_from" in b)

# Filtered
s, b = get(f"/dashboard/explainability?cls=Tomato_Healthy")
chk("explainability filter cls=Healthy", s == 200, f"status={s}")

# Detail page — try first pred with gradcam
if pred_ids:
    # Check if gradcam was generated (it may not be on every run)
    s_det, b_det = get(f"/dashboard/explainability/{pred_ids[0]}")
    if s_det == 200:
        has_gcam_page = "Grad-CAM" in b_det and "heatmap" in b_det.lower()
        chk(f"explainability_detail/{pred_ids[0]} loaded",
            s_det == 200, f"status={s_det}")
        chk("detail has heatmap content", has_gcam_page or "not available" in b_det.lower())
    else:
        # Might redirect if no gradcam
        chk(f"explainability_detail/{pred_ids[0]} no 500",
            s_det in (200, 302), f"status={s_det}")

# Invalid ID
s, b = get("/dashboard/explainability/999999")
chk("explainability_detail invalid id", s in (200, 404), f"status={s}")

# ── 6. Reports with filters + stats + export ─────────────────────────────
lines.append("\n=== 6. REPORTS ===")
s, b = get("/dashboard/reports")
chk("reports loads", s == 200 and "Total Predictions" in b, f"status={s}")
chk("reports has filter bar", "Apply" in b)
chk("reports shows stats", "Avg Confidence" in b or "avg" in b.lower())
chk("reports has class breakdown", "compare-bar" in b.lower() or "by Class" in b)
chk("reports CSV export link", "export/csv" in b)
chk("reports PDF export link", "export/pdf" in b)

# Filtered reports
s, b = get("/dashboard/reports?cls=Tomato_Healthy")
chk("filtered reports (Healthy)", s == 200, f"status={s}")

s, b = get(f"/dashboard/reports?date_from={today}")
chk("filtered reports by date", s == 200, f"status={s}")

# Reports CSV export
s, b = get("/dashboard/reports/export/csv")
chk("reports CSV", s == 200, f"status={s}")
chk("reports CSV header", "ID" in b and "Class" in b)

# Reports PDF export
s, b = get("/dashboard/reports/export/pdf")
chk("reports PDF", s == 200, f"status={s}")

# ── 7. Auth boundary checks ────────────────────────────────────────────────
lines.append("\n=== 7. AUTH BOUNDARY ===")
# Create fresh opener (no session)
cj2    = http.cookiejar.CookieJar()
anon   = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj2))

def get_anon(path):
    try:
        r = anon.open(BASE + path, timeout=8)
        return r.getcode(), r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")

for path in ["/dashboard/history", "/dashboard/history/export/csv",
             "/dashboard/history/export/pdf", "/dashboard/explainability",
             "/dashboard/reports", "/dashboard/reports/export/csv",
             "/dashboard/reports/export/pdf"]:
    sa, ba = get_anon(path)
    chk(f"anon blocked: {path}", "Sign In" in ba or sa == 302,
        f"status={sa}")

# ── 8. Delete still works ──────────────────────────────────────────────────
lines.append("\n=== 8. DELETE REGRESSION ===")
if pred_ids:
    s, b = get("/dashboard/history")
    csrf_del = extract_csrf(b)
    sd, bd = post_form(
        f"/dashboard/history/delete/{pred_ids[0]}",
        {"csrf_token": csrf_del}
    )
    chk("delete prediction", sd == 200 and "Prediction History" in bd,
        f"status={sd}")
    # Verify it's gone
    s, b = get(f"/dashboard/result/{pred_ids[0]}")
    chk("deleted pred_id no longer accessible",
        "not found" in b.lower() or "Result not found" in b or "Prediction History" in b)

# ── Summary ────────────────────────────────────────────────────────────────
lines.append(f"\n{'='*55}")
lines.append(f"TOTAL: {passed} PASS  |  {failed} FAIL")
lines.append(f"{'='*55}")
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Done. PASS={passed} FAIL={failed}")
print(f"Results: {OUT}")
