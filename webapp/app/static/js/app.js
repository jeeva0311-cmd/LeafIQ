/* LeafIQ — Frontend JavaScript */
'use strict';

// ── Flash message auto-dismiss ─────────────────────────────
document.querySelectorAll('.alert[data-dismiss]').forEach(el => {
  setTimeout(() => el.remove(), parseInt(el.dataset.dismiss, 10) || 5000);
});

// ── Image upload preview ───────────────────────────────────
const uploadInput = document.getElementById('imageUpload');
const previewWrap = document.getElementById('previewWrap');
const previewImg  = document.getElementById('previewImg');
const uploadText  = document.getElementById('uploadText');
const submitBtn   = document.getElementById('submitBtn');

if (uploadInput) {
  uploadInput.addEventListener('change', function () {
    const file = this.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      previewWrap.classList.remove('hidden');
      if (uploadText) uploadText.classList.add('hidden');
      if (submitBtn) submitBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  });
}

// Drag-and-drop
const uploadZone = document.querySelector('.upload-zone');
if (uploadZone) {
  ['dragenter','dragover'].forEach(evt =>
    uploadZone.addEventListener(evt, e => { e.preventDefault(); uploadZone.classList.add('drag-over'); })
  );
  ['dragleave','drop'].forEach(evt =>
    uploadZone.addEventListener(evt, e => { e.preventDefault(); uploadZone.classList.remove('drag-over'); })
  );
  uploadZone.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0];
    if (file && uploadInput) {
      const dt = new DataTransfer();
      dt.items.add(file);
      uploadInput.files = dt.files;
      uploadInput.dispatchEvent(new Event('change'));
    }
  });
}

// ── Loading overlay on form submit ────────────────────────
const uploadForm = document.getElementById('uploadForm');
const loadingOverlay = document.getElementById('loadingOverlay');
if (uploadForm && loadingOverlay) {
  uploadForm.addEventListener('submit', () => {
    loadingOverlay.classList.remove('hidden');
  });
}

// ── Probability bars animation ─────────────────────────────
document.querySelectorAll('.prob-bar-fill[data-pct]').forEach(bar => {
  const pct = parseFloat(bar.dataset.pct) || 0;
  requestAnimationFrame(() => {
    setTimeout(() => { bar.style.width = pct + '%'; }, 100);
  });
});

// ── Compare chart bars animation ──────────────────────────
document.querySelectorAll('.compare-bar-fill[data-pct]').forEach(bar => {
  const pct = parseFloat(bar.dataset.pct) || 0;
  requestAnimationFrame(() => {
    setTimeout(() => { bar.style.width = pct + '%'; }, 200);
  });
});

// ── Password visibility toggle ────────────────────────────
document.querySelectorAll('[data-toggle-pw]').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = document.getElementById(btn.dataset.togglePw);
    if (!target) return;
    if (target.type === 'password') {
      target.type = 'text';
      btn.textContent = '🙈';
    } else {
      target.type = 'password';
      btn.textContent = '👁';
    }
  });
});

// ── Real-time password strength ───────────────────────────
const pwField    = document.getElementById('password');
const pwStrength = document.getElementById('pwStrength');
if (pwField && pwStrength) {
  pwField.addEventListener('input', () => {
    const val = pwField.value;
    let score = 0;
    if (val.length >= 8)  score++;
    if (/[A-Z]/.test(val)) score++;
    if (/[0-9]/.test(val)) score++;
    if (/[^A-Za-z0-9]/.test(val)) score++;
    const labels  = ['', 'Weak', 'Fair', 'Good', 'Strong'];
    const colors  = ['', '#e74c3c', '#e67e22', '#f1c40f', '#27ae60'];
    pwStrength.textContent = val.length ? labels[score] : '';
    pwStrength.style.color = colors[score];
  });
}

// ── Smooth scroll for anchor links ───────────────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// ── Auto-hide flash messages ──────────────────────────────
setTimeout(() => {
  document.querySelectorAll('.flash-auto-hide').forEach(el => {
    el.style.transition = 'opacity 0.5s';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 500);
  });
}, 4000);
