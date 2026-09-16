// InsectVision front end. Plain JS, no build step, no framework.
//
// The page is a "stage + sidebar" dashboard. Everything visual about the
// photo happens on the stage (left): the empty drop zone, the chosen photo
// with its crop tool, the live camera, and finally the analysed image with
// boxes drawn on it. The sidebar (right) holds the verdict, the detection
// list, the species reference and the model facts. On a phone the two
// columns stack, stage first.
//
// Stage states: "empty" -> "preview" -> "result", plus "camera".

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // must match app.py's config.MAX_UPLOAD_BYTES

let health = null;
let selectedFile = null;

// Fallbacks for when /api/v1/health is unreachable. The live values come
// from the server so the UI can never disagree with the decision the
// backend made (decision_tree.CONFIDENCE_THRESHOLD, species.json reject_class).
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
 * is_specimen:false). Such a prediction says there is NO recognisable
 * insect here; it is not an identification of one. */
function isSpecimen(taxon) {
  if (taxon === rejectClass()) return false;
  const info = health && health.species_info && health.species_info[taxon];
  return !(info && info.is_specimen === false);
}

// The exact bytes last sent to /api/v1/analyse (post-crop), kept so the
// result can draw detection boxes over precisely what the server saw. Box
// coordinates from the API are in that image's pixel space.
let lastAnalysedBlob = null;

// Crop tool state. `crop` is in CSS pixels relative to the displayed
// (possibly scaled-down) preview image, not the photo's real pixels;
// getCropRectNatural() converts when it is time to cut.
let crop = null;
let cropContainerEl, cropBoxEl, cropImgEl;
let activeDrag = null; // { type: "move"|"nw"|"ne"|"sw"|"se", startX, startY, startCrop }
const MIN_CROP_DISPLAY_PX = 60;

const el = (id) => document.getElementById(id);

function isDesktop() {
  return !!(window.matchMedia && window.matchMedia("(min-width: 1024px)").matches);
}

// ---------------------------------------------------------------- //
// Boot
// ---------------------------------------------------------------- //
async function init() {
  try {
    const res = await fetch("/api/v1/health");
    health = await res.json();
  } catch (e) {
    health = { mode: "not_configured", class_names: [], taxon_status: {}, species_info: {},
               reject_class: null, confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD };
  }

  const specimenCount = (health.class_names || []).filter(isSpecimen).length;
  if (specimenCount > 0) {
    renderSupportedSpecies();
    el("supportedCard").hidden = false;
    el("supportedCount").textContent = `${specimenCount} species`;
    el("factClasses").textContent = rejectClass()
      ? `${specimenCount} species plus a "not an insect" class`
      : `${specimenCount} species`;
  }
  document.querySelectorAll("[data-threshold-pct]").forEach((n) => {
    n.textContent = Math.round(confidenceThreshold() * 100) + "%";
  });
  el("rejectNote").hidden = !rejectClass();

  if (typeof health.classifier_macro_f1 === "number") {
    const b = el("modelBadge");
    b.textContent = `${specimenCount} species · F1 ${health.classifier_macro_f1.toFixed(2)}`;
    b.hidden = false;
    el("factF1").textContent = `${health.classifier_macro_f1.toFixed(3)} (validation macro-F1)`;
  }
  if (typeof health.detector_map50 === "number") {
    el("factMap").textContent = health.detector_map50.toFixed(3);
  }

  if (health.mode === "not_configured") {
    document.querySelector(".stage-col").hidden = true;
    document.querySelector(".side-col").hidden = true;
    el("notConfiguredCard").hidden = false;
    return;
  }

  const badge = el("modeBadge");
  badge.hidden = false;
  badge.textContent = health.mode === "cascade" ? "Auto-counting" : "Single insect";

  wireEvents();
  setStage("empty");
}

function wireEvents() {
  el("takePhotoBtn").addEventListener("click", () => el("cameraInput").click());
  el("chooseFileBtn").addEventListener("click", () => el("galleryInput").click());
  el("cameraInput").addEventListener("change", (e) => onFilePicked(e.target.files[0]));
  el("galleryInput").addEventListener("change", (e) => onFilePicked(e.target.files[0]));
  el("replaceBtn").addEventListener("click", () => el("galleryInput").click());
  el("scanAnotherBtn").addEventListener("click", startOver);
  el("reanalyseBtn").addEventListener("click", () => setStage("preview"));
  el("analyseBtn").addEventListener("click", submitAnalysis);
  el("resetCropBtn").addEventListener("click", resetCrop);
  el("zoomCenterBtn").addEventListener("click", zoomToCenter);
  el("tabUpload").addEventListener("click", () => setSource("upload"));
  el("tabCamera").addEventListener("click", () => setSource("camera"));

  initCropper();
  initWebcam();

  // The whole stage accepts a dropped file, not just the empty panel.
  const dz = el("stage");
  dz.addEventListener("dragover", (e) => { e.preventDefault(); el("stageEmpty").classList.add("drag-over"); });
  dz.addEventListener("dragleave", () => el("stageEmpty").classList.remove("drag-over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    el("stageEmpty").classList.remove("drag-over");
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) onFilePicked(file);
  });
}

// ---------------------------------------------------------------- //
// Stage state machine
// ---------------------------------------------------------------- //
let stageState = "empty";

function setStage(state) {
  stageState = state;
  el("stage").dataset.state = state;
  el("stageEmpty").hidden = state !== "empty";
  el("stagePreview").hidden = state !== "preview";
  el("stageCamera").hidden = state !== "camera";
  el("stageResult").hidden = state !== "result";

  el("toolsPreview").hidden = state !== "preview";
  el("toolsCamera").hidden = state !== "camera";
  el("toolsResult").hidden = state !== "result";
  el("analyseBtn").hidden = state !== "preview";
  el("cropHint").hidden = state !== "preview";

  if (state !== "camera") stopWebcamTracks();

  const tabCamera = state === "camera";
  el("tabCamera").classList.toggle("is-active", tabCamera);
  el("tabUpload").classList.toggle("is-active", !tabCamera);
  el("tabCamera").setAttribute("aria-selected", String(tabCamera));
  el("tabUpload").setAttribute("aria-selected", String(!tabCamera));

  if (state === "empty") setStatus("Ready", "");
  if (state === "preview") setStatus("Photo loaded. Adjust the crop, then analyse.", "ok");
  if (state === "camera" && !webcamStream) setStatus("Camera is off", "");
}

/** Tabs: "upload" shows the photo (or the empty drop zone if none yet),
 * "camera" shows the live viewfinder. */
function setSource(source) {
  if (source === "camera") {
    setStage("camera");
    return;
  }
  if (stageState === "camera") setStage(selectedFile ? "preview" : "empty");
}

function setStatus(text, kind) {
  const s = el("stageStatus");
  s.className = "stage-status" + (kind ? " " + kind : "");
  el("stageStatusText").textContent = text;
}

function startOver() {
  resetCapture();
  resetStats();
  el("verdictBanner").hidden = true;
  el("rejectedView").hidden = true;
  el("verdictEmpty").hidden = false;
  el("cascadeResults").hidden = true;
  el("classifierResults").hidden = true;
  el("detectionsEmpty").hidden = false;
  setStage("empty");
  if (!isDesktop()) window.scrollTo({ top: 0, behavior: "smooth" });
}

function resetStats() {
  ["statDetections", "statConfirmed", "statReview", "statNone"].forEach((id) => { el(id).textContent = "0"; });
  el("statQuality").textContent = "n/a";
  el("latencyWrap").hidden = true;
}

// ---------------------------------------------------------------- //
// Live camera (inline on the stage)
//
// Phones also get the native camera app through the capture="environment"
// file input ("Take photo"), which is usually the better experience there.
// getUserMedia needs a secure context: https:// in production (Render is),
// or localhost during development. On plain http:// over a LAN the browser
// hides the API entirely, and the Live camera tab stays hidden with it.
// ---------------------------------------------------------------- //
let webcamStream = null;
let webcamFacing = "environment"; // rear camera first on devices that have one

function webcamSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function initWebcam() {
  if (!webcamSupported()) return;
  el("tabCamera").hidden = false;
  el("webcamStartBtn").addEventListener("click", startWebcamStream);
  el("webcamStopBtn").addEventListener("click", () => { stopWebcamTracks(); setStatus("Camera is off", ""); });
  el("webcamCaptureBtn").addEventListener("click", captureWebcamFrame);
  el("webcamSwitchBtn").addEventListener("click", () => {
    webcamFacing = webcamFacing === "environment" ? "user" : "environment";
    startWebcamStream();
  });
  // Never leave a camera running when the tab is hidden.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && webcamStream) { stopWebcamTracks(); setStatus("Camera is off", ""); }
  });
}

async function startWebcamStream() {
  stopWebcamTracks();
  el("webcamError").hidden = true;
  const video = el("webcamVideo");
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      // "ideal", not "exact": ask for a detailed frame but accept whatever
      // the camera can do rather than failing on a 720p laptop webcam.
      video: { facingMode: webcamFacing, width: { ideal: 1920 }, height: { ideal: 1080 } },
    });
    video.srcObject = webcamStream;
    try { await video.play(); } catch (e) { /* the autoplay attribute covers it */ }
    video.dataset.live = "1";
    el("webcamOff").hidden = true;
    el("webcamGuide").hidden = false;
    el("webcamCaptureBtn").disabled = false;
    el("webcamStopBtn").hidden = false;
    el("webcamStartBtn").hidden = true;
    setStatus("Camera live", "live");
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
    return "The camera only works over https:// or on localhost. Your browser blocks camera access on plain http.";
  }
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Camera permission was denied. Allow camera access for this site in your browser's address bar, then try again.";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError" || name === "DevicesNotFoundError") {
    return "No camera was found on this device. Use \"Choose file\" instead.";
  }
  if (name === "NotReadableError" || name === "AbortError") {
    return "The camera is busy or unavailable. Another app may be using it. Close that app and try again.";
  }
  return "Couldn't start the camera. Try again, or use \"Choose file\".";
}

function showWebcamError(msg) {
  const e = el("webcamError");
  e.textContent = msg;
  e.hidden = false;
  el("webcamCaptureBtn").disabled = true;
  setStatus("Camera unavailable", "");
}

function stopWebcamTracks() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
  const video = el("webcamVideo");
  if (video) { video.srcObject = null; delete video.dataset.live; }
  el("webcamOff").hidden = false;
  el("webcamGuide").hidden = true;
  el("webcamCaptureBtn").disabled = true;
  el("webcamStopBtn").hidden = true;
  el("webcamStartBtn").hidden = false;
}

/** Grabs the current frame at the camera's native resolution and hands it
 * to the same onFilePicked() path a chosen file takes, so the crop tool,
 * size check and upload behave identically for camera shots. */
function captureWebcamFrame() {
  const video = el("webcamVideo");
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) {
    showWebcamError("The camera hasn't delivered a frame yet. Give it a second and try again.");
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
    onFilePicked(new File([blob], `camera-${Date.now()}.jpg`, { type: "image/jpeg" }));
  }, "image/jpeg", 0.92);
}

// ---------------------------------------------------------------- //
// Choosing a photo
// ---------------------------------------------------------------- //
function onFilePicked(file) {
  hideFileError();
  if (!file) return;

  // A client-side check is a courtesy, not a security boundary. app.py
  // enforces both the content-type allowlist and the 10 MB limit itself.
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
  // URL lifecycle edge cases and, more importantly, lets the preview stay
  // hidden until we KNOW there is something real to show. The preview is
  // revealed inside resetCrop(), on the <img>'s own "load" event, so a
  // format the browser can't render never gets to show a broken image.
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
// The native camera app is outside the page's control, so there is no way
// to overlay a framing guide on it. This is the practical equivalent: let
// the user tighten the frame around the insect AFTER capture, so a small
// or distant subject can still be zoomed in on before it is analysed.
// ---------------------------------------------------------------- //
function initCropper() {
  cropContainerEl = el("cropContainer");
  cropImgEl = el("previewImg");
  cropBoxEl = el("cropBox");

  // The crop box only becomes visible once renderCropBox() has run with
  // real, measured dimensions. Its darkening effect is a box-shadow with a
  // huge spread, so it must never be shown before a real position exists.
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
 * the crop tool gets exactly the same image sent as if the tool did not
 * exist. Cropping is an aid, not a silent default. */
function resetCrop() {
  // Reveal the preview HERE, not in onFilePicked: this only runs once the
  // <img> has genuinely decoded. Unhiding must happen before measuring
  // clientWidth/clientHeight; a display:none element measures 0x0.
  setStage("preview");
  el("analyseBtn").disabled = false;

  const w = cropImgEl.clientWidth;
  const h = cropImgEl.clientHeight;
  if (!w || !h) return; // not laid out yet
  crop = { x: 0, y: 0, w, h };
  renderCropBox();
}

/** One-tap help for "the insect is small in the frame": a centred 70% box.
 * Only on request; assuming the subject is centred would be wrong often
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

/** The browser could not decode the selected file as an image, most often
 * a HEIC/HEIF photo from an iPhone. Without this the page would show a
 * broken-image icon with the crop overlay stuck on. */
function onPreviewImageError() {
  resetCapture(); // calls hideFileError(), so it must run BEFORE showFileError()
  setStage("empty");
  showFileError(
    "That photo couldn't be opened. Your browser may not support its format " +
    "(this happens with HEIC photos from some phones). Try a different photo, " +
    "or save it as JPEG or PNG first."
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

/** Maps the on-screen crop box (display pixels) to the photo's real pixel
 * coordinates; the preview is usually shown scaled down. */
function getCropRectNatural() {
  const scaleX = cropImgEl.naturalWidth / cropImgEl.clientWidth;
  const scaleY = cropImgEl.naturalHeight / cropImgEl.clientHeight;
  return { x: crop.x * scaleX, y: crop.y * scaleY, w: crop.w * scaleX, h: crop.h * scaleY };
}

/** Cuts the selected region out of the original photo and returns it as a
 * JPEG Blob, or null on any failure so the caller sends the original. */
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
  if (cropBoxEl) cropBoxEl.hidden = true;
  el("cameraInput").value = "";
  el("galleryInput").value = "";
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

// ---------------------------------------------------------------- //
// Submit
// ---------------------------------------------------------------- //
async function submitAnalysis() {
  if (!selectedFile) return;

  el("analyseBtn").disabled = true;
  el("analyseBtnText").textContent = "Analysing…";
  el("analyseSpinner").hidden = false;
  el("submitError").hidden = true;
  setStatus("Analysing photo", "busy");

  // Send the cropped region if the crop tool produced one; fall back to the
  // original file if cropping failed for any reason. Analysing the whole
  // photo beats failing the scan.
  const croppedBlob = await cropToBlob();
  const imageToSend = croppedBlob
    ? new File([croppedBlob], "cropped.jpg", { type: "image/jpeg" })
    : selectedFile;

  lastAnalysedBlob = imageToSend;

  try {
    renderResult(await postForAnalysis(imageToSend));
  } catch (err) {
    const e = el("submitError");
    e.textContent = friendlySubmitError(err);
    e.hidden = false;
    setStatus("Analysis failed", "");
  } finally {
    el("analyseBtn").disabled = false;
    el("analyseBtnText").textContent = "Analyse photo";
    el("analyseSpinner").hidden = true;
  }
}

/** Thrown only for an HTTP response the server actually sent back (4xx/5xx),
 * as opposed to fetch() itself throwing a TypeError when no response came
 * back at all (dropped connection, or a free-tier server still waking up). */
class HttpError extends Error {}

/** POSTs the image, retrying once after a short pause if the FIRST attempt
 * never got an HTTP response. A real HTTP error is not retried; the server
 * already answered. One retry rides out a brief mobile-network drop or a
 * Render free-tier instance coming back from sleep mid-request. */
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
  return "Couldn't reach the server after a couple of tries. This can happen on a weak " +
    "connection, or right after the server has been idle and is still waking up. " +
    "Wait a few seconds and press Analyse photo again.";
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
// Supported-species reference (sidebar)
// ---------------------------------------------------------------- //
function renderSupportedSpecies() {
  const groups = { pest: [], beneficial: [], neutral: [] };
  health.class_names.forEach((taxon) => {
    if (!isSpecimen(taxon)) return; // "other" is a model output, not a species
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
      taxa.slice().sort((a, b) => speciesLabel(a).localeCompare(speciesLabel(b))).forEach((taxon) => {
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
  el("verdictEmpty").hidden = true;
  el("latencyMs").textContent = `${Math.round(data.latency_ms)} ms`;
  el("latencyWrap").hidden = false;
  el("statQuality").textContent = data.quality.score.toFixed(2);

  if (!data.quality.passed) {
    // The photo stays on the stage so the user can re-crop or replace it.
    el("verdictBanner").hidden = true;
    el("rejectedView").hidden = false;
    el("rejectReason").textContent = data.quality.reason || "Image quality too low.";
    el("rejectScore").textContent = data.quality.score.toFixed(2);
    el("cascadeResults").hidden = true;
    el("classifierResults").hidden = true;
    el("detectionsEmpty").hidden = false;
    ["statDetections", "statConfirmed", "statReview", "statNone"].forEach((id) => { el(id).textContent = "0"; });
    setStatus("Photo not usable", "");
    scrollToResultsOnSmallScreens();
    return;
  }

  el("rejectedView").hidden = true;
  el("detectionsEmpty").hidden = true;
  renderVerdictBanner(data);
  renderStats(data);

  if (data.mode === "cascade") {
    el("cascadeResults").hidden = false;
    el("classifierResults").hidden = true;
    renderCascade(data);
  } else {
    el("cascadeResults").hidden = true;
    el("classifierResults").hidden = false;
    renderClassifierOnly(data);
    showAnalysedImage();
  }

  setStage("result");
  setStatus(`Done in ${Math.round(data.latency_ms)} ms`, "ok");
  scrollToResultsOnSmallScreens();
}

/** On phones the sidebar sits below the stage, so bring the verdict into view. */
function scrollToResultsOnSmallScreens() {
  if (isDesktop()) return;
  const card = el("verdictCard");
  if (card.scrollIntoView) card.scrollIntoView({ behavior: "smooth", block: "start" });
}

/** The instrument strip under the stage. */
function renderStats(data) {
  let dets = 0, confirmed = 0, review = 0, none = 0;
  if (data.mode === "cascade") {
    (data.detections || []).forEach((d) => {
      dets += 1;
      if (d.is_specimen === false) none += 1;
      else if (d.verdict === "confirmed" || (!d.verdict && !d.flagged)) confirmed += 1;
      else review += 1;
    });
  } else {
    const best = (data.top_predictions || [])[0];
    if (best) {
      dets = 1;
      if (data.verdict === "no_specimen" || best.is_specimen === false) none = 1;
      else if (data.verdict === "confirmed" || best.confidence >= confidenceThreshold()) confirmed = 1;
      else review = 1;
    }
  }
  el("statDetections").textContent = String(dets);
  el("statConfirmed").textContent = String(confirmed);
  el("statReview").textContent = String(review);
  el("statNone").textContent = String(none);
}

/** The one-line answer. Three states only, matching decision_tree.Verdict:
 * confirmed (green), uncertain (amber), none (grey). */
function renderVerdictBanner(data) {
  const banner = el("verdictBanner");
  banner.className = "verdict-banner";
  banner.innerHTML = "";
  let kind, title, sub;
  const thrPct = Math.round(confidenceThreshold() * 100);

  if (data.mode === "cascade") {
    const dets = data.detections || [];
    const confirmed = dets.filter((d) => d.verdict === "confirmed" || (!d.verdict && !d.flagged));
    const uncertain = dets.filter((d) => d.verdict === "uncertain" || (!d.verdict && d.flagged && d.is_specimen !== false));
    const rejected = dets.filter((d) => d.is_specimen === false);
    if (dets.length === 0) {
      kind = "none"; title = "No insects detected";
      sub = "Nothing in the photo looked like an insect to the detector. Try cropping closer.";
    } else if (confirmed.length > 0) {
      const names = [...new Set(confirmed.map((d) => speciesLabel(d.taxon)))];
      kind = "confirmed";
      title = confirmed.length === 1 ? `Identified: ${names[0]}` : `${confirmed.length} insects identified`;
      sub = [uncertain.length ? `${uncertain.length} more need${uncertain.length === 1 ? "s" : ""} review` : null,
             rejected.length ? `${rejected.length} region${rejected.length === 1 ? "" : "s"} not an insect` : null]
            .filter(Boolean).join(" · ");
    } else if (uncertain.length > 0) {
      kind = "uncertain";
      title = `Possibly ${speciesLabel(uncertain[0].taxon)} (needs review)`;
      sub = `Below the ${thrPct}% confidence needed to confirm.`;
    } else {
      kind = "none"; title = "No known insect recognised";
      sub = "The regions found don't match any species this app knows.";
    }
  } else {
    const preds = data.top_predictions || [];
    const best = preds[0];
    if (!best || data.verdict === "no_specimen" || best.is_specimen === false) {
      kind = "none"; title = "No known insect recognised";
      sub = "This photo doesn't look like any species this app is trained on.";
    } else if (data.verdict === "confirmed" || best.confidence >= confidenceThreshold()) {
      kind = "confirmed"; title = `Identified: ${speciesLabel(best.taxon)}`;
      sub = `${Math.round(best.confidence * 100)}% confidence · ${speciesStatus(best.taxon)}`;
    } else {
      kind = "uncertain"; title = `Possibly ${speciesLabel(best.taxon)} (needs review)`;
      sub = `${Math.round(best.confidence * 100)}% is below the ${thrPct}% needed to confirm.`;
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
    p.className = "muted";
    p.textContent = "No insects detected in this photo.";
    list.appendChild(p);
    return;
  }

  const counts = {};
  data.detections.forEach((d) => {
    // Only a CONFIRMED specimen is ever counted. `flagged` covers both an
    // uncertain insect and a rejected region; is_specimen is checked too so
    // a "Not an insect" chip can never reach the tally.
    if (d.flagged || d.is_specimen === false) return;
    counts[d.taxon] = (counts[d.taxon] || 0) + 1;
  });

  const noSpecimenCount = data.detections.filter((d) => d.is_specimen === false).length;
  if (noSpecimenCount === data.detections.length) {
    // Every region the detector proposed was rejected: the honest headline
    // is "nothing recognised", not a list of guesses.
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
    const runnerUp = d.runner_up_taxon ? { taxon: d.runner_up_taxon, confidence: d.runner_up_confidence } : null;
    list.appendChild(detectionRow(d.taxon, d.confidence, d.flagged, false, runnerUp));
  });
}

/** Across several rejected regions, the most plausible insect the model
 * considered (or null if none clears 10%). */
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

/** Neutral "nothing recognised" state: no species name in the heading, no
 * pest/beneficial colour and no confidence bar. The model's confidence in
 * "other" is confidence that there is NO known insect here; a green bar
 * would read as the opposite. */
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
      `(${Math.round(closest.confidence * 100)}%). Too low to count as an identification.`;
    box.appendChild(alt);
  }
  return box;
}

/** One detected region the classifier rejected, listed alongside real
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

// Same colour language as the status tags and the header legend, plus a
// distinct blue for anything below the confirmation threshold, so an
// unconfirmed guess never LOOKS as certain as a confirmed one on the photo.
const BOX_COLORS = { pest: "#b71c1c", beneficial: "#2e7d32", neutral: "#4a4a4a", flagged: "#0d47a1", none: "#757575" };

/** Show the analysed image on the stage without boxes (classifier-only
 * mode, or a cascade result with no detections). */
function showAnalysedImage() {
  const img = el("resultImg");
  el("resultImageWrap").hidden = true;
  if (!lastAnalysedBlob) { img.hidden = true; return; }
  const url = URL.createObjectURL(lastAnalysedBlob);
  img.onload = () => URL.revokeObjectURL(url);
  img.src = url;
  img.hidden = false;
}

/** Draws the exact image the server analysed with a box and label over
 * each detection. Box coordinates from the API are in the pixel space of
 * the image that was actually sent (lastAnalysedBlob). */
function drawDetections(detections) {
  const wrap = el("resultImageWrap");
  if (!lastAnalysedBlob || detections.length === 0) {
    showAnalysedImage();
    return;
  }
  el("resultImg").hidden = true;

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

      // A rejected region shows no percentage: that number would be the
      // model's confidence that it is NOT an insect, which reads backwards
      // next to the real detections' confidence-in-a-species numbers.
      const label = rejected ? "Not an insect" : `${speciesLabel(d.taxon)} ${Math.round(d.confidence * 100)}%`;
      const pad = Math.round(fontSize * 0.3);
      const textW = ctx.measureText(label).width;
      const labelH = fontSize + pad * 2;
      const labelY = y1 - labelH >= 0 ? y1 - labelH : y1; // flip inside when too close to the top edge
      ctx.fillStyle = color;
      ctx.fillRect(x1, labelY, textW + pad * 2, labelH);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x1 + pad, labelY + pad);
    });

    wrap.hidden = false;
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    showAnalysedImage();
  };
  img.src = url;
}

function renderClassifierOnly(data) {
  const list = el("topPredictionsList");
  const heading = el("topMatchesHeading");
  list.innerHTML = "";

  const preds = data.top_predictions;
  if (data.verdict === "no_specimen" || (preds.length && preds[0].is_specimen === false)) {
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
      // "other" as a runner-up is useful context (the model partly doubts
      // this is an insect) but it is not a species row.
      if (p.confidence >= 0.10) {
        const note = document.createElement("div");
        note.className = "runner-up-note";
        note.textContent = `The model also considered that this may not be an insect at all (${Math.round(p.confidence * 100)}%).`;
        list.appendChild(note);
      }
      return;
    }
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

  // Show the second-best guess when it is not negligible, e.g. a weevil vs
  // beetle mix-up, so the user sees both candidates instead of one label
  // that reads as more certain than it is.
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
