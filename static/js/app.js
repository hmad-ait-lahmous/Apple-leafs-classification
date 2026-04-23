'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let webcamStream   = null;
let autoInterval   = null;
let isAnalyzing    = false;
let lastAutoResult = null;

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(tab, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(`panel-${tab}`).classList.add('active');

  if (tab !== 'webcam') {
    stopCamera();
  }
}

// ── Upload ────────────────────────────────────────────────────────────────────
function onDragOver(e) {
  e.preventDefault();
  document.getElementById('dropzone').classList.add('drag-over');
}
function onDragLeave(e) {
  document.getElementById('dropzone').classList.remove('drag-over');
}
function onDrop(e) {
  e.preventDefault();
  document.getElementById('dropzone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    document.getElementById('fileInput').files = e.dataTransfer.files;
    loadFilePreview(file);
  }
}

function onFileSelected(e) {
  const file = e.target.files[0];
  if (file) loadFilePreview(file);
}

function loadFilePreview(file) {
  const reader = new FileReader();
  reader.onload = function (e) {
    document.getElementById('preview-img').src = e.target.result;
    document.getElementById('dropzone').classList.add('hidden');
    document.getElementById('preview-area').classList.remove('hidden');
    hideResult();
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  document.getElementById('preview-area').classList.add('hidden');
  document.getElementById('dropzone').classList.remove('hidden');
  document.getElementById('fileInput').value = '';
  hideResult();
}

async function analyzeUpload() {
  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files[0]) return;

  showSkeleton();

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    showResult(data);
  } catch (err) {
    showError(err.message);
  }
}

// ── Webcam ────────────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 960 } }
    });
    const video = document.getElementById('webcam');
    video.srcObject = webcamStream;
    document.getElementById('webcam-off').classList.add('hidden');
    document.getElementById('btn-stop').style.display    = 'inline-flex';
    document.getElementById('btn-capture').style.display = 'inline-flex';
    document.getElementById('auto-label').style.display  = 'inline-flex';
  } catch (err) {
    alert('Could not access camera: ' + err.message);
  }
}

function stopCamera() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  clearInterval(autoInterval);
  autoInterval = null;

  const toggle = document.getElementById('auto-toggle');
  if (toggle) toggle.checked = false;

  const pulse = document.getElementById('auto-pulse');
  if (pulse) pulse.remove();

  document.getElementById('webcam').srcObject = null;
  document.getElementById('webcam-off').classList.remove('hidden');
  document.getElementById('btn-stop').style.display    = 'none';
  document.getElementById('btn-capture').style.display = 'none';
  document.getElementById('auto-label').style.display  = 'none';
}

function captureFrame() {
  const video  = document.getElementById('webcam');
  const canvas = document.getElementById('webcam-canvas');
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.85);
}

async function captureAndAnalyze() {
  if (!webcamStream || isAnalyzing) return;
  const dataUrl = captureFrame();
  await analyzeBase64(dataUrl);
}

async function analyzeBase64(dataUrl) {
  if (isAnalyzing) return;
  isAnalyzing = true;
  showSkeleton();

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    showResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    isAnalyzing = false;
  }
}

function toggleAuto(enabled) {
  const container = document.querySelector('.webcam-container');
  let pulse = document.getElementById('auto-pulse');

  if (enabled) {
    if (!pulse) {
      pulse = document.createElement('div');
      pulse.id = 'auto-pulse';
      pulse.className = 'auto-pulse';
      container.appendChild(pulse);
    }
    autoInterval = setInterval(async () => {
      if (webcamStream && !isAnalyzing) {
        const dataUrl = captureFrame();
        await analyzeBase64(dataUrl);
      }
    }, 3000);
  } else {
    clearInterval(autoInterval);
    autoInterval = null;
    if (pulse) pulse.remove();
  }
}

// ── Result rendering ──────────────────────────────────────────────────────────
function showSkeleton() {
  const card = document.getElementById('result-card');
  card.classList.remove('hidden', 'healthy', 'moderate', 'severe');
  document.getElementById('skeleton').classList.remove('hidden');
  document.getElementById('result-content').classList.add('hidden');
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideResult() {
  document.getElementById('result-card').classList.add('hidden');
  document.getElementById('skeleton').classList.add('hidden');
  document.getElementById('result-content').classList.add('hidden');
}

function showResult(data) {
  document.getElementById('skeleton').classList.add('hidden');
  document.getElementById('result-content').classList.remove('hidden');

  const card = document.getElementById('result-card');
  card.classList.remove('healthy', 'moderate', 'severe');
  card.classList.add(data.severity);

  // Badge
  const badge = document.getElementById('result-badge');
  badge.textContent = data.emoji;
  badge.className   = `result-badge badge-${data.severity}`;

  // Label & confidence
  document.getElementById('result-label').textContent = data.label;
  document.getElementById('result-confidence').textContent =
    `Confidence: ${data.confidence}%  ·  Analyzed in ${data.elapsed_ms} ms`;

  // Summary
  document.getElementById('result-summary').textContent = data.summary;

  // Score bars
  const barsEl = document.getElementById('score-bars');
  barsEl.innerHTML = '';

  const sorted = Object.entries(data.all_scores).sort((a, b) => b[1] - a[1]);
  const winnerClass = data.class_name;
  const barColors   = [winnerClass, 'orange', 'yellow'];

  sorted.forEach(([cls, pct], i) => {
    const isWinner = cls === winnerClass;
    const colorClass = isWinner ? 'winner' : (i === 1 ? 'orange' : 'yellow');
    const friendlyName = getFriendlyName(cls);

    barsEl.innerHTML += `
      <div class="score-item">
        <div class="score-meta">
          <span>${friendlyName}</span>
          <span>${pct}%</span>
        </div>
        <div class="score-track">
          <div class="score-fill ${colorClass}" style="width: 0%" data-target="${pct}"></div>
        </div>
      </div>`;
  });

  // Animate bars after paint
  requestAnimationFrame(() => {
    document.querySelectorAll('.score-fill').forEach(bar => {
      const target = bar.getAttribute('data-target');
      setTimeout(() => { bar.style.width = target + '%'; }, 50);
    });
  });

  // Tips
  const tipsEl = document.getElementById('tips-list');
  tipsEl.innerHTML = '';
  data.tips.forEach(tip => {
    tipsEl.innerHTML += `
      <div class="tip-card">
        <div class="tip-icon">${tip.icon}</div>
        <div class="tip-content">
          <div class="tip-title">${tip.title}</div>
          <div class="tip-body">${tip.body}</div>
        </div>
      </div>`;
  });

  // Footer
  const now = new Date().toLocaleTimeString();
  document.getElementById('result-footer').textContent =
    `Analysis at ${now} — For informational purposes only.`;

  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
  document.getElementById('skeleton').classList.add('hidden');
  const card = document.getElementById('result-card');
  card.classList.remove('hidden', 'healthy', 'moderate', 'severe');
  card.classList.add('moderate');

  const content = document.getElementById('result-content');
  content.classList.remove('hidden');
  content.innerHTML = `
    <div class="result-header">
      <div class="result-badge badge-moderate">⚠️</div>
      <div>
        <h2 class="result-label">Analysis Failed</h2>
        <p class="result-confidence">Please try again with a clearer photo</p>
      </div>
    </div>
    <p class="result-summary">${message}</p>`;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function getFriendlyName(cls) {
  const map = {
    'Apple black_spot': '🟠 Black Spot',
    'Apple Brown_spot': '🟡 Brown Spot',
    'Apple Normal':     '🟢 Healthy',
  };
  return map[cls] || cls;
}
