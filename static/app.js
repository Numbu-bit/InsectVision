// InsectVision -- plain JS, no build step, no framework.

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // must match app.py's config.MAX_UPLOAD_BYTES

let health = null;
let selectedFile = null;

// Fallbacks only for when /api/v1/health is unreachable -- the live values
// come from the server so the UI can never disagree with the decision the
// backend actually made (decision_tree.CONFIDENCE_THRESHOLD, species.json
// reject_class).
const DEFAULT_CONFIDENCE_THRESHOLD = 0.75;
function confidenceThreshold() {
  return (health && typeof health.confidence_threshold === "number")
    ? health.confidence_threshold : DEFAULT_CONFIDENCE_THRESHOLD;
}
/** The classifier's "not an insect I recognise" class, or null for a legacy
 * closed-set model. */
function rejectClass() {
  return (health && health.reject_class) || null;
}
/** False for the reject class (and anything species.json marks
 * is_specimen:false) -- such a prediction is a statement that there is NO
 * recognisable insect, not an identification of one. */
function isSpecimen(taxon) {
  if (taxon === rejectClass()) return false;
  const info = health && health.species_info && health.species_info[taxon];
  return !(info && info.is_specimen === false);
}

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
    health = { mode: "not_configured", class_names: [], taxon_status: {}, species_info: {},
               reject_class: null, confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD };
  }

  // Shown regardless of mode -- the class list comes from species.json,
  // not from whether a trained model is actually loaded, so it's useful
  // reference info even before training finishes.
  if (health.class_names && health.class_names.length > 0) {
    renderSupportedSpecies();
    el("supportedCard").hidden = false;
    el("supportedCount").textContent = `${health.class_names.filter(isSpecimen).length} species`;
  }
  // Model badge: class count and the recorded validation macro-F1, when
  // training has written one -- so a user can see which model answered.
  if (typeof health.classifier_macro_f1 === "number") {
    const b = el("modelBadge");
    b.textContent = `${health.class_names.filter(isSpecimen).length} species · F1 ${health.classifier_macro_f1.toFixed(2)}`;
    b.hidden = false;
  }
  setStep(1);
  // Both threshold mentions in the static copy read the live value too.
  document.querySelectorAll("[data-threshold-pct]").forEach((n) => {
    n.textContent = Math.round(confidenceThreshold() * 100) + "%";
  });
  el("rejectNote").hidden = !rejectClass();

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
  initWebcam();
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

/** 1 = choose photo, 2 = ready to analyse / analysing, 3 = verdict shown. */
function setStep(n) {
  document.querySelectorAll(".step").forEach((li) => {
    const k = Number(li.dataset.step);
    li.classList.toggle("is-active", k === n);
    li.classList.toggle("is-done", k < n);
  });
}

// ---------------------------------------------------------------- //
// Webcam (desktop and laptops -- phones get the native camera via the
// capture="environment" input, which is a better experience there)
//
// getUserMedia needs a secure context: https:// in production (Render is),
// or localhost during development. On plain http:// over a LAN the browser
// hides the API entirely, and the button stays hidden with it.
// ---------------------------------------------------------------- //
let webcamStream = null;
let webcamFacing = "environment"; // rear camera first on devices that have one

function webcamSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function initWebcam() {
  if (!webcamSupported()) return;
  el("webcamBtn").hidden = false;
  el("webcamBtn").addEventListener("click", openWebcam);
  el("webcamCancelBtn").addEventListener("click", closeWebcam);
  el("webcamCaptureBtn").addEventListener("click", captureWebcamFrame);
  el("webcamSwitchBtn").addEventListener("click", () => {
    webcamFacing = webcamFacing === "environment" ? "user" : "environment";
    startWebcamStream();
  });
  el("webcamModal").addEventListener("click", (e) => {
    if (e.target === el("webcamModal")) closeWebcam(); // click on the backdrop
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !el("webcamModal").hidden) closeWebcam();
  });
  // Never leave a camera running in the background when the tab is hidden.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && webcamStream) closeWebcam();
  });
}

async function openWebcam() {
  hideFileError();
  el("webcamError").hidden = true;
  el("webcamModal").hidden = false;
  el("webcamCaptureBtn").disabled = true;
  await startWebcamStream();
}

async function startWebcamStream() {
  stopWebcamTracks();
  const video = el("webcamVideo");
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      // "ideal", not "exact": ask for a high-resolution frame (the model
      // wants detail) but accept whatever the camera can do rather than
      // failing outright on a 720p laptop webcam.
      video: { facingMode: webcamFacing, width: { ideal: 1920 }, height: { ideal: 1080 } },
    });
    video.srcObject = webcamStream;
    try { await video.play(); } catch (e) { /* autoplay attribute handles it */ }
    el("webcamCaptureBtn").disabled = false;
    el("webcamError").hidden = true;
    // Offer the flip button only when there is more than one camera.
    try {
      const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
      el("webcamSwitchBtn").hidden = cams.length < 2;
    } catch (e) {
      el("webcamSwitchBtn").hidden = true;
    }
  } catch (err) {
    showWebcamError(friendlyWebcamError(err));
  }
}

function friendlyWebcamError(err) {
  const name = err && err.name;
  if (!window.isSecureContext) {
    return "The webcam only works over https:// or on localhost -- your browser blocks camera access on plain http.";
  }
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Camera permission was denied. Allow camera access for this site in your browser's address bar, then try again.";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError" || name === "DevicesNotFoundError") {
    return "No camera was found on this device. Use \"Choose File\" instead.";
  }
  if (name === "NotReadableError" || name === "AbortError") {
    return "The camera is busy or unavailable -- another app may be using it. Close it and try again.";
  }
  return "Couldn't start the webcam. Try again, or use \"Choose File\".";
}

function showWebcamError(msg) {
  const e = el("webcamError");
  e.textContent = msg;
  e.hidden = false;
  el("webcamCaptureBtn").disabled = true;
}

function stopWebcamTracks() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
  const video = el("webcamVideo");
  if (video) video.srcObject = null;
}

function closeWebcam() {
  stopWebcamTracks();
  el("webcamModal").hidden = true;
}

/** Grabs the current video frame at the camera's native resolution and
 * hands it to the same onFilePicked() path a chosen file takes -- so the
 * crop tool, size check and upload behave identically for webcam shots. */
function captureWebcamFrame() {
  const video = el("webcamVideo");
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) {
    showWebcamError("The camera hasn't delivered a frame yet -- give it a second and try again.");
    return;
  }
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d").drawImage(video, 0, 0, w, h);
  canvas.toBlob((blob) => {
    if (!blob) {
      showWebcamError("Couldn't capture the frame. Try again.");
      return;
    }
    closeWebcam();
    onFilePicked(new File([blob], `webcam-${Date.now()}.jpg`, { type: "image/jpeg" }));
  }, "image/jpeg", 0.92);
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
  setStep(2);

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
  setStep(1);
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
// Defaults to "neutral", not "pest": a taxon missing from taxon_status is a
// config gap, and painting it red would invent a risk judgement nobody made.
function speciesStatus(taxon) {
  return (health.taxon_status && health.taxon_status[taxon]) || "neutral";
}

// ---------------------------------------------------------------- //
// Supported-species reference list (shown at the top of the page)
// ---------------------------------------------------------------- //
function renderSupportedSpecies() {
  const groups = { pest: [], beneficial: [], neutral: [] };
  health.class_names.forEach((taxon) => {
    if (!isSpecimen(taxon)) return; // "other" is a model output, not a species the app identifies
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
  setStep(3);

  el("latencyMs").textContent = `${Math.round(data.latency_ms)} ms`;
  renderVerdictBanner(data);

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

/** The one-line answer above the details. Three states only, matching
 * decision_tree.Verdict: confirmed (green), uncertain (amber), none (grey). */
function renderVerdictBanner(data) {
  const banner = el("verdictBanner");
  banner.className = "verdict-banner";
  banner.innerHTML = "";
  let kind, title, sub;

  if (data.mode === "cascade") {
    const dets = data.detections || [];
    const confirmed = dets.filter((d) => d.verdict === "confirmed" || (!d.verdict && !d.flagged));
    const uncertain = dets.filter((d) => d.verdict === "uncertain" || (!d.verdict && d.flagged && d.is_specimen !== false));
    const rejected = dets.filter((d) => d.is_specimen === false);
    if (dets.length === 0) {
      kind = "none"; title = "No insects detected"; sub = "Nothing in the photo looked like an insect to the detector. Try cropping closer.";
    } else if (confirmed.length > 0) {
      const names = [...new Set(confirmed.map((d) => speciesLabel(d.taxon)))];
      kind = "confirmed";
      title = confirmed.length === 1 ? `Identified: ${names[0]}` : `${confirmed.length} insects identified`;
      sub = [uncertain.length ? `${uncertain.length} more need${uncertain.length === 1 ? "s" : ""} review` : null,
             rejected.length ? `${rejected.length} region${rejected.length === 1 ? "" : "s"} not an insect` : null]
            .filter(Boolean).join(" · ");
    } else if (uncertain.length > 0) {
      kind = "uncertain";
      title = `Possibly ${speciesLabel(uncertain[0].taxon)} -- needs review`;
      sub = `Below the ${Math.round(confidenceThreshold() * 100)}% confidence needed to confirm.`;
    } else {
      kind = "none"; title = "No known insect recognised"; sub = "The regions found don't match any species this app knows.";
    }
  } else {
    const preds = data.top_predictions || [];
    const best = preds[0];
    if (!best || data.verdict === "no_specimen" || best.is_specimen === false) {
      kind = "none"; title = "No known insect recognised"; sub = "This photo doesn't look like any species this app is trained on.";
    } else if (data.verdict === "confirmed" || best.confidence >= confidenceThreshold()) {
      kind = "confirmed"; title = `Identified: ${speciesLabel(best.taxon)}`; sub = `${Math.round(best.confidence * 100)}% confidence · ${speciesStatus(best.taxon)}`;
    } else {
      kind = "uncertain"; title = `Possibly ${speciesLabel(best.taxon)} -- needs review`; sub = `${Math.round(best.confidence * 100)}% is below the ${Math.round(confidenceThreshold() * 100)}% needed to confirm.`;
    }
  }

  const icon = document.createElement("span");
  icon.className = "verdict-icon";
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = kind === "confirmed" ? "✅" : kind === "uncertain" ? "🤔" : "🔍";
  const text = document.createElement("span");
  text.className = "verdict-text";
  const t = document.createElement("span"); t.textContent = title;
  text.appendChild(t);
  if (sub) { const sEl = document.createElement("span"); sEl.className = "verdict-sub"; sEl.textContent = sub; text.appendChild(sEl); }
  banner.appendChild(icon);
  banner.appendChild(text);
  banner.classList.add(kind);
  banner.hidden = false;
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
    // Only a CONFIRMED specimen is ever counted. `flagged` is true for both
    // an uncertain insect and a no_specimen region, and is_specimen is
    // checked explicitly too so a server that only sent the older field
    // set still can't get a "Not an insect" chip into the tally.
    if (d.flagged || d.is_specimen === false) return;
    counts[d.taxon] = (counts[d.taxon] || 0) + 1;
  });

  const noSpecimenCount = data.detections.filter((d) => d.is_specimen === false).length;
  const allNoSpecimen = noSpecimenCount === data.detections.length;

  if (allNoSpecimen) {
    // Every region the detector proposed was rejected by the classifier:
    // the honest headline is "nothing recognised", not a list of guesses.
    list.appendChild(noSpecimenBlock(bestClosestSpecies(data.detections)));
    return;
  }

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
    if (d.is_specimen === false) {
      list.appendChild(noSpecimenRow(d));
      return;
    }
    const runnerUp = d.runner_up_taxon
      ? { taxon: d.runner_up_taxon, confidence: d.runner_up_confidence }
      : null;
    list.appendChild(detectionRow(d.taxon, d.confidence, d.flagged, false, runnerUp));
  });
}

/** Across several no_specimen regions, the single most plausible insect
 * candidate the model considered (or null if none clears 10%). */
function bestClosestSpecies(detections) {
  let best = null;
  detections.forEach((d) => {
    if (d.is_specimen !== false || !d.runner_up_taxon || !isSpecimen(d.runner_up_taxon)) return;
    if (d.runner_up_confidence >= 0.10 && (!best || d.runner_up_confidence > best.confidence)) {
      best = { taxon: d.runner_up_taxon, confidence: d.runner_up_confidence };
    }
  });
  return best;
}

/** Neutral "nothing recognised" state. Deliberately has no species name in
 * the heading, no pest/beneficial colour and no confidence bar: the
 * classifier's confidence in "other" is confidence that there's NO known
 * insect here, and drawing it as a green bar would read as the opposite. */
function noSpecimenBlock(closest) {
  const box = document.createElement("div");
  box.className = "no-specimen";
  const icon = document.createElement("div");
  icon.className = "no-specimen-icon";
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = "🔍";
  const h = document.createElement("h3");
  h.textContent = "No known insect recognised";
  const p = document.createElement("p");
  p.textContent = "This photo doesn't look like any of the species this app is trained on. " +
    "If there is an insect in it, try getting closer or cropping tighter around it.";
  box.appendChild(icon);
  box.appendChild(h);
  box.appendChild(p);
  if (closest) {
    const alt = document.createElement("p");
    alt.className = "runner-up-note";
    alt.textContent = `Closest species the model considered: ${speciesLabel(closest.taxon)} ` +
      `(${Math.round(closest.confidence * 100)}%) -- far too low to be an identification.`;
    box.appendChild(alt);
  }
  return box;
}

/** One detected region that the classifier rejected, listed alongside real
 * detections when a photo has both. */
function noSpecimenRow(d) {
  const row = document.createElement("div");
  row.className = "detection-row no-specimen-row";
  const top = document.createElement("div");
  top.className = "detection-row-top";
  const name = document.createElement("span");
  name.innerHTML = `<span class="detection-name">Not an insect</span>`;
  const tag = document.createElement("span");
  tag.className = "status-tag none";
  tag.textContent = "not recognised";
  name.appendChild(tag);
  top.appendChild(name);
  row.appendChild(top);
  const note = document.createElement("div");
  note.className = "runner-up-note";
  const closest = (d.runner_up_taxon && isSpecimen(d.runner_up_taxon) && d.runner_up_confidence >= 0.10)
    ? ` Closest species considered: ${speciesLabel(d.runner_up_taxon)} (${Math.round(d.runner_up_confidence * 100)}%).`
    : "";
  note.textContent = "This region doesn't match any species the app knows." + closest;
  row.appendChild(note);
  return row;
}

// Matches the status-tag / tally-chip colour language elsewhere in the UI
// (see style.css --red-700/--green-700/--grey-700), plus a distinct blue
// for anything flagged below the 75% confirmation threshold (matching
// .flagged-tag) so an unconfirmed guess never LOOKS as certain as a
// confirmed one when drawn on the photo itself.
const BOX_COLORS = { pest: "#b71c1c", beneficial: "#2e7d32", neutral: "#4a4a4a", flagged: "#0d47a1", none: "#757575" };

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
      const rejected = d.is_specimen === false;
      const color = rejected ? BOX_COLORS.none
        : d.flagged ? BOX_COLORS.flagged
        : (BOX_COLORS[speciesStatus(d.taxon)] || BOX_COLORS.neutral);

      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.setLineDash(rejected ? [lineWidth * 3, lineWidth * 2] : []);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);

      // A rejected region shows no percentage: the number would be the
      // model's confidence that it is NOT an insect, which reads backwards
      // next to the real detections' confidence-in-a-species numbers.
      const label = rejected ? "Not an insect" : `${speciesLabel(d.taxon)} ${Math.round(d.confidence * 100)}%`;
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
  const heading = el("topMatchesHeading");
  list.innerHTML = "";

  const preds = data.top_predictions;
  if (data.verdict === "no_specimen" || (preds.length && preds[0].is_specimen === false)) {
    // The whole image was rejected: no species card at all, just the
    // neutral state plus (maybe) the closest insect for context.
    heading.hidden = true;
    const closest = preds.find((p) => p.is_specimen !== false && p.confidence >= 0.10) || null;
    list.appendChild(noSpecimenBlock(closest));
    return;
  }

  heading.hidden = false;
  const thr = confidenceThreshold();
  let bestShown = false;
  preds.forEach((p) => {
    if (p.is_specimen === false) {
      // "other" appearing as a runner-up is useful context ("the model
      // partly doubts this is an insect") but it's not a species row.
      if (p.confidence >= 0.10) {
        const note = document.createElement("div");
        note.className = "runner-up-note";
        note.textContent = `The model also considered that this may not be an insect at all (${Math.round(p.confidence * 100)}%).`;
        list.appendChild(note);
      }
      return;
    }
    // The first specimen row is the model's best guess -- marked so it's
    // clear which of the numbers is the primary answer.
    list.appendChild(detectionRow(p.taxon, p.confidence, p.confidence < thr, !bestShown, null));
    bestShown = true;
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

  const thr = confidenceThreshold();
  const pct = Math.round(confidence * 100);
  const barWrap = document.createElement("div");
  barWrap.className = "confidence-bar-wrap";
  const fill = document.createElement("div");
  fill.className = "confidence-bar-fill" + (confidence < thr ? " below-threshold" : "");
  fill.style.width = pct + "%";
  const tick = document.createElement("div");
  tick.className = "confidence-threshold-tick";
  tick.style.left = Math.round(thr * 100) + "%";
  tick.title = `${Math.round(thr * 100)}% confirmation threshold`;
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
    alt.textContent = isSpecimen(runnerUp.taxon)
      ? `Could also be ${speciesLabel(runnerUp.taxon)} (${Math.round(runnerUp.confidence * 100)}%)`
      : `The model also considered that this may not be an insect at all (${Math.round(runnerUp.confidence * 100)}%).`;
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
