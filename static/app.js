// InsectVision -- plain JS, no build step, no framework.

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // must match app.py's config.MAX_UPLOAD_BYTES
const SEVERITY_LABELS = {
  none: "No concern",
  low: "Low risk",
  moderate: "Moderate risk",
  high: "High risk",
};

let health = null;
let selectedFile = null;

const el = (id) => document.getElementById(id);

async function init() {
  try {
    const res = await fetch("/api/v1/health");
    health = await res.json();
  } catch (e) {
    health = { mode: "not_configured", class_names: [], taxon_status: {}, species_info: {} };
  }

  // Shown regardless of mode -- the class list comes from species.json,
  // not from whether a trained model is actually loaded, so it's useful
  // reference info even before training finishes.
  if (health.class_names && health.class_names.length > 0) {
    renderSupportedSpecies();
    el("supportedCard").hidden = false;
  }

  if (health.mode === "not_configured") {
    el("captureCard").hidden = true;
    el("notConfiguredCard").hidden = false;
    return;
  }

  const badge = el("modeBadge");
  badge.hidden = false;
  badge.textContent = health.mode === "cascade" ? "Auto-counting" : "Manual count";

  // The count field only matters in classifier-only mode -- cascade mode
  // counts detections itself, so asking the user to also type a number
  // would be redundant and confusing about which number actually drives
  // the severity decision.
  el("countField").hidden = health.mode === "cascade";

  wireEvents();
}

function wireEvents() {
  el("takePhotoBtn").addEventListener("click", () => el("cameraInput").click());
  el("chooseFileBtn").addEventListener("click", () => el("galleryInput").click());
  el("cameraInput").addEventListener("change", (e) => onFilePicked(e.target.files[0]));
  el("galleryInput").addEventListener("change", (e) => onFilePicked(e.target.files[0]));
  el("replaceBtn").addEventListener("click", resetCapture);
  el("retakeBtn").addEventListener("click", showCaptureState);
  el("scanAnotherBtn").addEventListener("click", showCaptureState);
  el("analyseBtn").addEventListener("click", submitAnalysis);

  const dz = el("dropzone");
  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("drag-over");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag-over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag-over");
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) onFilePicked(file);
  });
}

// ---------------------------------------------------------------- //
// Capture
// ---------------------------------------------------------------- //
function onFilePicked(file) {
  hideFileError();
  if (!file) return;

  // A client-side check is a courtesy, not a security boundary -- app.py
  // enforces both the content-type allowlist and the 10 MB limit itself
  // regardless of what happens here.
  if (file.type && !file.type.startsWith("image/")) {
    showFileError("That doesn't look like an image file. Please choose a photo.");
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    showFileError(`Image is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max is 10 MB.`);
    return;
  }

  selectedFile = file;
  el("previewImg").src = URL.createObjectURL(file);
  el("emptyState").hidden = true;
  el("previewState").hidden = false;
  el("analyseBtn").disabled = false;
}

function resetCapture() {
  selectedFile = null;
  el("cameraInput").value = "";
  el("galleryInput").value = "";
  el("emptyState").hidden = false;
  el("previewState").hidden = true;
  el("analyseBtn").disabled = true;
  hideFileError();
}

function showFileError(msg) {
  const e = el("fileError");
  e.textContent = msg;
  e.hidden = false;
}
function hideFileError() {
  el("fileError").hidden = true;
}

function showCaptureState() {
  el("resultCard").hidden = true;
  el("captureCard").hidden = false;
  resetCapture();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---------------------------------------------------------------- //
// Submit
// ---------------------------------------------------------------- //
async function submitAnalysis() {
  if (!selectedFile) return;

  el("analyseBtn").disabled = true;
  el("analyseBtnText").textContent = "Analysing…";
  el("analyseSpinner").hidden = false;
  el("submitError").hidden = true;

  const form = new FormData();
  form.append("image", selectedFile);
  form.append("growth_stage", el("growthStage").value);
  form.append("observed_count", el("observedCount").value || "1");

  try {
    const res = await fetch("/api/v1/analyse", { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }
    renderResult(await res.json());
  } catch (err) {
    const e = el("submitError");
    e.textContent = err.message || "Something went wrong. Check your connection and try again.";
    e.hidden = false;
  } finally {
    el("analyseBtn").disabled = false;
    el("analyseBtnText").textContent = "Analyse photo";
    el("analyseSpinner").hidden = true;
  }
}

// ---------------------------------------------------------------- //
// Species lookups (from /api/v1/health)
// ---------------------------------------------------------------- //
function speciesLabel(taxon) {
  const info = health.species_info && health.species_info[taxon];
  return (info && info.common_name) || taxon.replace(/_/g, " ");
}
function speciesStatus(taxon) {
  return (health.taxon_status && health.taxon_status[taxon]) || "pest";
}

// ---------------------------------------------------------------- //
// Supported-species reference list (shown at the top of the page)
// ---------------------------------------------------------------- //
function renderSupportedSpecies() {
  const groups = { pest: [], beneficial: [], neutral: [] };
  health.class_names.forEach((taxon) => {
    const status = speciesStatus(taxon);
    (groups[status] || groups.neutral).push(taxon);
  });

  const groupMeta = {
    pest: { heading: "⚠️ Pests", order: 0 },
    beneficial: { heading: "🐝 Beneficial", order: 1 },
    neutral: { heading: "➖ Neutral", order: 2 },
  };

  const container = el("supportedGroups");
  container.innerHTML = "";

  Object.entries(groups)
    .sort((a, b) => groupMeta[a[0]].order - groupMeta[b[0]].order)
    .forEach(([status, taxa]) => {
      if (taxa.length === 0) return;

      const group = document.createElement("div");
      group.className = "species-group";

      const heading = document.createElement("h3");
      heading.className = "species-group-heading";
      heading.textContent = `${groupMeta[status].heading} (${taxa.length})`;
      group.appendChild(heading);

      const list = document.createElement("div");
      list.className = "species-chip-list";
      taxa
        .slice()
        .sort((a, b) => speciesLabel(a).localeCompare(speciesLabel(b)))
        .forEach((taxon) => {
          const chip = document.createElement("span");
          chip.className = "species-chip " + status;
          chip.textContent = speciesLabel(taxon);
          list.appendChild(chip);
        });
      group.appendChild(list);

      container.appendChild(group);
    });
}

// ---------------------------------------------------------------- //
// Render result
// ---------------------------------------------------------------- //
function renderResult(data) {
  el("captureCard").hidden = true;
  el("resultCard").hidden = false;

  if (!data.quality.passed) {
    el("rejectedView").hidden = false;
    el("successView").hidden = true;
    el("rejectReason").textContent = data.quality.reason || "Image quality too low.";
    el("rejectScore").textContent = data.quality.score.toFixed(2);
    window.scrollTo({ top: 0, behavior: "smooth" });
    return;
  }

  el("rejectedView").hidden = true;
  el("successView").hidden = false;

  const decision = data.decision;
  const needsReview = decision.action === "flag_for_review";

  const band = el("severityBand");
  band.className = "severity-band severity-" + (needsReview ? "review" : decision.severity);
  el("severityLabel").textContent = needsReview
    ? "Needs expert review"
    : SEVERITY_LABELS[decision.severity] || decision.severity;

  el("advisoryText").textContent = decision.advisory;
  el("reviewNotice").hidden = !(decision.flagged && decision.flagged.length > 0);

  const pathList = el("decisionPath");
  pathList.innerHTML = "";
  (decision.path || []).forEach((step) => {
    const li = document.createElement("li");
    li.textContent = step;
    pathList.appendChild(li);
  });

  el("latencyMs").textContent = `${Math.round(data.latency_ms)} ms`;

  if (data.mode === "cascade") {
    el("cascadeResults").hidden = false;
    el("classifierResults").hidden = true;
    renderCascade(data);
  } else {
    el("cascadeResults").hidden = true;
    el("classifierResults").hidden = false;
    renderClassifierOnly(data);
  }

  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderCascade(data) {
  const tally = el("tally");
  const list = el("detectionList");
  tally.innerHTML = "";
  list.innerHTML = "";

  if (data.detections.length === 0) {
    const p = document.createElement("p");
    p.textContent = "No insects detected in this photo.";
    p.style.color = "var(--grey-700)";
    list.appendChild(p);
    return;
  }

  const counts = {};
  data.detections.forEach((d) => {
    if (d.flagged) return; // an unconfirmed guess must never be counted as a confirmed identification
    counts[d.taxon] = (counts[d.taxon] || 0) + 1;
  });

  if (Object.keys(counts).length === 0) {
    const chip = document.createElement("span");
    chip.className = "tally-chip";
    chip.textContent = "No confirmed identifications";
    tally.appendChild(chip);
  } else {
    Object.entries(counts).forEach(([taxon, n]) => {
      const chip = document.createElement("span");
      chip.className = "tally-chip " + speciesStatus(taxon);
      chip.textContent = `${n}× ${speciesLabel(taxon)}`;
      tally.appendChild(chip);
    });
  }

  data.detections.forEach((d) => list.appendChild(detectionRow(d.taxon, d.confidence, d.flagged, false)));
}

function renderClassifierOnly(data) {
  const list = el("topPredictionsList");
  list.innerHTML = "";
  data.top_predictions.forEach((p, i) => {
    // top_predictions[0] is always the one actually fed into the decision
    // tree (see app.py) -- marking it avoids the user wondering which of
    // the three numbers the advisory above is actually about.
    list.appendChild(detectionRow(p.taxon, p.confidence, p.confidence < 0.75, i === 0));
  });
}

function detectionRow(taxon, confidence, flagged, isBest) {
  const row = document.createElement("div");
  row.className = "detection-row";

  const top = document.createElement("div");
  top.className = "detection-row-top";

  const nameSpan = document.createElement("span");
  nameSpan.innerHTML = `<span class="detection-name">${escapeHtml(speciesLabel(taxon))}</span>`;

  const statusTag = document.createElement("span");
  statusTag.className = "status-tag " + speciesStatus(taxon);
  statusTag.textContent = speciesStatus(taxon);
  nameSpan.appendChild(statusTag);

  if (isBest) {
    const bestTag = document.createElement("span");
    bestTag.className = "status-tag beneficial";
    bestTag.style.marginLeft = "4px";
    bestTag.textContent = "used";
    nameSpan.appendChild(bestTag);
  }
  if (flagged) {
    const ft = document.createElement("span");
    ft.className = "flagged-tag";
    ft.style.marginLeft = "4px";
    ft.textContent = "Needs review";
    nameSpan.appendChild(ft);
  }
  top.appendChild(nameSpan);
  row.appendChild(top);

  const pct = Math.round(confidence * 100);
  const barWrap = document.createElement("div");
  barWrap.className = "confidence-bar-wrap";
  const fill = document.createElement("div");
  fill.className = "confidence-bar-fill" + (confidence < 0.75 ? " below-threshold" : "");
  fill.style.width = pct + "%";
  const tick = document.createElement("div");
  tick.className = "confidence-threshold-tick";
  tick.title = "75% confirmation threshold";
  barWrap.appendChild(fill);
  barWrap.appendChild(tick);
  row.appendChild(barWrap);

  const label = document.createElement("div");
  label.className = "confidence-label";
  label.innerHTML = `<span>Confidence</span><span>${pct}%</span>`;
  row.appendChild(label);

  return row;
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

init();
