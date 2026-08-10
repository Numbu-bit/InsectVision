// InsectVision -- plain JS, no build step, no framework.

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // must match app.py's config.MAX_UPLOAD_BYTES

let health = null;
let selectedFile = null;

// The exact bytes last sent to /api/v1/analyse (post-crop), kept only so a
// successful cascade result can draw detection boxes over precisely what
// the server saw -- box coordinates from the API are in that image's pixel
// space, not the original uncropped photo's.
let lastAnalysedBlob = null;

// Crop/zoom tool state. `crop` is in CSS pixels relative to the displayed
// (possibly scaled-down) preview image, NOT the photo's actual pixel
// dimensions -- getCropRectNatural() converts between the two when it's
// time to actually cut the image.
let crop = null;
let cropContainerEl, cropBoxEl, cropImgEl;
let activeDrag = null; // { type: "move"|"nw"|"ne"|"sw"|"se", startX, startY, startCrop }
const MIN_CROP_DISPLAY_PX = 60;

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
  el("resetCropBtn").addEventListener("click", resetCrop);
  el("zoomCenterBtn").addEventListener("click", zoomToCenter);

  initCropper();

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

  // Read as a data URL rather than URL.createObjectURL(): sidesteps blob
  // URL lifecycle/revocation edge cases entirely, and -- more importantly
  // -- lets the preview stay hidden until we KNOW there's something real
  // to show. Nothing here reveals previewState; that only happens inside
  // resetCrop(), triggered by the <img>'s own "load" event once the
  // browser has actually decoded the image successfully. That way a
  // format the browser can't render (or a slow decode) never gets a
  // chance to show a broken-image icon -- the empty state just stays put
  // until there's a confirmed result, success or failure.
  const reader = new FileReader();
  reader.onload = () => {
    el("previewImg").src = reader.result;
  };
  reader.onerror = () => {
    showFileError("Couldn't read that file. Please try a different photo.");
  };
  reader.readAsDataURL(file);
}

// ---------------------------------------------------------------- //
// Crop/zoom tool
//
// The native camera app (opened via <input capture="environment">) is
// outside the page's control -- there's no way to overlay a live framing
// guide on it or auto-zoom it from JS. This is the practical equivalent:
// let the user tighten the frame around the insect AFTER capture, so a
// small/distant subject can still be zoomed in on before the image is
// sent for detection.
// ---------------------------------------------------------------- //
function initCropper() {
  cropContainerEl = el("cropContainer");
  cropImgEl = el("previewImg");
  cropBoxEl = el("cropBox");

  // The crop box starts hidden and ONLY becomes visible once
  // renderCropBox() has run with real, measured dimensions (see
  // resetCrop/zoomToCenter). Its darkening effect is a box-shadow with a
  // huge spread, which still covers the entire page even when the box
  // itself is sized 0x0 -- so it must never be shown before a real
  // position exists, not just left at whatever it defaulted to.
  cropBoxEl.hidden = true;

  cropImgEl.addEventListener("load", resetCrop);
  cropImgEl.addEventListener("error", onPreviewImageError);

  cropBoxEl.addEventListener("pointerdown", (e) => {
    if (e.target === cropBoxEl) startDrag(e, "move");
  });
  cropBoxEl.querySelectorAll(".crop-handle").forEach((handle) => {
    handle.addEventListener("pointerdown", (e) => startDrag(e, handle.dataset.handle));
  });
  document.addEventListener("pointermove", onPointerMove);
  document.addEventListener("pointerup", () => { activeDrag = null; });
}

/** Default state: the full photo, unmodified. A user who never touches
 * the crop tool gets exactly the same image sent as before this feature
 * existed -- cropping is an available aid, not a silent default that
 * could clip a subject that wasn't centred. */
function resetCrop() {
  // Reveal the preview HERE, not in onFilePicked -- this only runs once
  // the <img> has genuinely finished decoding (it's wired to the "load"
  // event), so the container is never shown with nothing valid to display.
  // Unhiding must happen before measuring clientWidth/clientHeight below:
  // a hidden (display:none) element always measures 0x0 regardless of the
  // image's real size.
  el("emptyState").hidden = true;
  el("previewState").hidden = false;
  el("analyseBtn").disabled = false;

  const w = cropImgEl.clientWidth;
  const h = cropImgEl.clientHeight;
  if (!w || !h) return; // not laid out yet
  crop = { x: 0, y: 0, w, h };
  renderCropBox();
}

/** One-tap suggestion for "the insect is small in the frame": crop to a
 * centred 70% box. Only runs when the user asks for it (see wireEvents),
 * since assuming the subject is centred by default would be wrong often
 * enough to do more harm than good. */
function zoomToCenter() {
  const w = cropImgEl.clientWidth;
  const h = cropImgEl.clientHeight;
  if (!w || !h) return;
  const boxW = w * 0.7;
  const boxH = h * 0.7;
  crop = { x: (w - boxW) / 2, y: (h - boxH) / 2, w: boxW, h: boxH };
  renderCropBox();
}

function renderCropBox() {
  cropBoxEl.hidden = false;
  cropBoxEl.style.left = crop.x + "px";
  cropBoxEl.style.top = crop.y + "px";
  cropBoxEl.style.width = crop.w + "px";
  cropBoxEl.style.height = crop.h + "px";
}

/** Fires when the browser can't decode the selected file as an image --
 * most commonly a HEIC/HEIF photo (the default format on many iPhones),
 * which plenty of browsers accept as a file but can't render in an <img>
 * tag. Without this handler the page was left showing a broken-image icon
 * with the crop tool's full-page darkening stuck on, since resetCrop()
 * (which is what reveals the crop box) never runs when "load" never fires. */
function onPreviewImageError() {
  // resetCapture() itself calls hideFileError() -- it must run BEFORE
  // showFileError(), not after, or it immediately wipes the very message
  // this function exists to show.
  resetCapture();
  showFileError(
    "That photo couldn't be opened -- your browser may not support its format " +
    "(this happens with HEIC photos from some phones). Try a different photo, " +
    "or save/export it as JPEG or PNG first."
  );
}

function clampCrop() {
  const maxW = cropImgEl.clientWidth;
  const maxH = cropImgEl.clientHeight;
  crop.w = Math.max(MIN_CROP_DISPLAY_PX, Math.min(crop.w, maxW));
  crop.h = Math.max(MIN_CROP_DISPLAY_PX, Math.min(crop.h, maxH));
  crop.x = Math.max(0, Math.min(crop.x, maxW - crop.w));
  crop.y = Math.max(0, Math.min(crop.y, maxH - crop.h));
}

function startDrag(e, type) {
  e.preventDefault();
  activeDrag = { type, startX: e.clientX, startY: e.clientY, startCrop: { ...crop } };
  e.target.setPointerCapture && e.target.setPointerCapture(e.pointerId);
}

function onPointerMove(e) {
  if (!activeDrag || !crop) return;
  const dx = e.clientX - activeDrag.startX;
  const dy = e.clientY - activeDrag.startY;
  const s = activeDrag.startCrop;

  if (activeDrag.type === "move") {
    crop.x = s.x + dx;
    crop.y = s.y + dy;
    crop.w = s.w;
    crop.h = s.h;
  } else {
    let { x, y, w, h } = s;
    if (activeDrag.type.includes("e")) w = s.w + dx;
    if (activeDrag.type.includes("s")) h = s.h + dy;
    if (activeDrag.type.includes("w")) { x = s.x + dx; w = s.w - dx; }
    if (activeDrag.type.includes("n")) { y = s.y + dy; h = s.h - dy; }
    crop = { x, y, w, h };
  }
  clampCrop();
  renderCropBox();
}

/** Maps the on-screen crop box (display pixels) to the photo's actual
 * pixel coordinates -- the preview is often shown scaled down. */
function getCropRectNatural() {
  const scaleX = cropImgEl.naturalWidth / cropImgEl.clientWidth;
  const scaleY = cropImgEl.naturalHeight / cropImgEl.clientHeight;
  return {
    x: crop.x * scaleX,
    y: crop.y * scaleY,
    w: crop.w * scaleX,
    h: crop.h * scaleY,
  };
}

/** Cuts the selected region out of the original photo via canvas and
 * returns it as a JPEG Blob. Falls back to null on any failure so the
 * caller can just send the original, uncropped file instead. */
async function cropToBlob() {
  if (!crop) return null;
  try {
    const rect = getCropRectNatural();
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(rect.w));
    canvas.height = Math.max(1, Math.round(rect.h));
    const ctx = canvas.getContext("2d");
    ctx.drawImage(cropImgEl, rect.x, rect.y, rect.w, rect.h, 0, 0, canvas.width, canvas.height);
    return await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
  } catch (e) {
    return null;
  }
}

function resetCapture() {
  selectedFile = null;
  crop = null;
  if (cropBoxEl) cropBoxEl.hidden = true; // avoid its full-page darkening lingering into the empty state
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

  // Send the cropped region if the crop tool produced one (i.e. the user
  // zoomed in, or even just the default 70% box); fall back to the
  // original file if cropping failed for any reason -- better to analyse
  // the whole photo than to fail the scan outright.
  const croppedBlob = await cropToBlob();
  const imageToSend = croppedBlob
    ? new File([croppedBlob], "cropped.jpg", { type: "image/jpeg" })
    : selectedFile;

  // Kept around so a successful result can draw detection boxes over the
  // exact bytes the server actually analysed (see drawDetections).
  lastAnalysedBlob = imageToSend;

  try {
    renderResult(await postForAnalysis(imageToSend));
  } catch (err) {
    const e = el("submitError");
    e.textContent = friendlySubmitError(err);
    e.hidden = false;
  } finally {
    el("analyseBtn").disabled = false;
    el("analyseBtnText").textContent = "Analyse photo";
    el("analyseSpinner").hidden = true;
  }
}

/** Thrown only for an HTTP response the server actually sent back (4xx/5xx)
 * -- as opposed to fetch() itself throwing a plain TypeError, which happens
 * when no response came back at all (dropped mobile connection, DNS blip,
 * or a free-tier server that was still finishing waking up from idle). */
class HttpError extends Error {}

/** POSTs the image, retrying once after a short pause if the FIRST attempt
 * never got an HTTP response at all. A real HTTP error response is not
 * retried -- the server already answered, so trying again wouldn't change
 * anything. This single retry is enough to ride out the two most common
 * real-world causes of a bare "Failed to fetch": a brief mobile-network
 * drop, and a Render free-tier instance that had gone to sleep and was
 * still coming back up mid-request. */
async function postForAnalysis(imageFile, attempt = 1) {
  const form = new FormData();
  form.append("image", imageFile);
  try {
    const res = await fetch("/api/v1/analyse", { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new HttpError(body.detail || `Request failed (${res.status})`);
    }
    return await res.json();
  } catch (err) {
    if (attempt < 2 && !(err instanceof HttpError)) {
      await new Promise((resolve) => setTimeout(resolve, 3000));
      return postForAnalysis(imageFile, attempt + 1);
    }
    throw err;
  }
}

function friendlySubmitError(err) {
  if (err instanceof HttpError) return err.message;
  return "Couldn't reach the server after a couple of tries. This can happen with a weak " +
    "connection, or right after the server has been idle and is still waking back up -- " +
    "wait a few seconds and tap Analyse photo again.";
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

  drawDetections(data.detections);

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

  data.detections.forEach((d) => {
    const runnerUp = d.runner_up_taxon
      ? { taxon: d.runner_up_taxon, confidence: d.runner_up_confidence }
      : null;
    list.appendChild(detectionRow(d.taxon, d.confidence, d.flagged, false, runnerUp));
  });
}

// Matches the status-tag / tally-chip colour language elsewhere in the UI
// (see style.css --red-700/--green-700/--grey-700), plus a distinct blue
// for anything flagged below the 75% confirmation threshold (matching
// .flagged-tag) so an unconfirmed guess never LOOKS as certain as a
// confirmed one when drawn on the photo itself.
const BOX_COLORS = { pest: "#b71c1c", beneficial: "#2e7d32", neutral: "#4a4a4a", flagged: "#0d47a1" };

/** Draws the exact image the server analysed, with a box and label over
 * each detection -- since the dataset this model trained on (Roboflow) is
 * itself bounding-box annotated, showing those boxes back is the natural
 * visual confirmation of what got detected, not just a text list. Box
 * coordinates from the API are in the pixel space of the image that was
 * actually sent (lastAnalysedBlob), which is why that exact blob -- not
 * the original unmodified photo -- is what gets drawn underneath them. */
function drawDetections(detections) {
  const wrap = el("resultImageWrap");
  if (!lastAnalysedBlob || detections.length === 0) {
    wrap.hidden = true;
    return;
  }

  const img = new Image();
  const url = URL.createObjectURL(lastAnalysedBlob);
  img.onload = () => {
    URL.revokeObjectURL(url);
    const canvas = el("resultCanvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);

    const lineWidth = Math.max(2, Math.round(img.naturalWidth / 250));
    const fontSize = Math.max(14, Math.round(img.naturalWidth / 45));
    ctx.font = `700 ${fontSize}px system-ui, sans-serif`;
    ctx.textBaseline = "top";

    detections.forEach((d) => {
      const [x1, y1, x2, y2] = d.box;
      const color = d.flagged ? BOX_COLORS.flagged : (BOX_COLORS[speciesStatus(d.taxon)] || BOX_COLORS.pest);

      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const label = `${speciesLabel(d.taxon)} ${Math.round(d.confidence * 100)}%`;
      const pad = Math.round(fontSize * 0.3);
      const textW = ctx.measureText(label).width;
      const labelH = fontSize + pad * 2;
      // Label sits just above the box normally, but flips inside the top
      // edge instead when the box is too close to the photo's edge for
      // an above-box label to fit on the canvas at all.
      const labelY = y1 - labelH >= 0 ? y1 - labelH : y1;
      ctx.fillStyle = color;
      ctx.fillRect(x1, labelY, textW + pad * 2, labelH);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x1 + pad, labelY + pad);
    });

    wrap.hidden = false;
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    wrap.hidden = true; // fall back to the text-only detection list below
  };
  img.src = url;
}

function renderClassifierOnly(data) {
  const list = el("topPredictionsList");
  list.innerHTML = "";
  data.top_predictions.forEach((p, i) => {
    // top_predictions[0] is the model's best guess -- marked so it's clear
    // which of the three numbers is the primary answer. All three are
    // already listed as separate rows here, so no extra runner-up note is
    // needed the way cascade mode's single-row-per-box needs one.
    list.appendChild(detectionRow(p.taxon, p.confidence, p.confidence < 0.75, i === 0, null));
  });
}

function detectionRow(taxon, confidence, flagged, isBest, runnerUp) {
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
    bestTag.textContent = "best match";
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

  // Surface the second-best guess when it's not negligible -- e.g. for a
  // weevil/beetle-type mix-up, this shows the user both candidates the
  // model was actually choosing between, instead of one label that reads
  // as more certain than it is.
  if (runnerUp && runnerUp.confidence >= 0.10) {
    const alt = document.createElement("div");
    alt.className = "runner-up-note";
    alt.textContent = `Could also be ${speciesLabel(runnerUp.taxon)} (${Math.round(runnerUp.confidence * 100)}%)`;
    row.appendChild(alt);
  }

  return row;
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

init();
