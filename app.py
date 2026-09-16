"""
FastAPI backend AND web UI host for InsectVision, deliberately one service:
a single Render instance serves both the JSON API and the static UI (Step 5)
from the same origin, so there's no CORS configuration to get wrong and no
second deployment to keep in sync.

A general-purpose insect identifier: given a photo, say what species it
looks like, how confident that guess is, and whether that species is
generally a pest, beneficial, or neutral. Nothing about crop type, growth
stage, or advisory follows from that -- this app deliberately doesn't
guess at things it was never told.

Every classification resolves to one of three verdicts (src/decision_tree.py):
    confirmed    a known species at/above the 0.75 threshold
    uncertain    a known species below it -- shown, never counted
    no_specimen  the classifier's reject class ("other") won: the photo
                 (or this detected region) is not an insect it recognises.
                 This is a definite answer, not a low-confidence one.

    uvicorn app:app --reload
"""
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import config
from src.classifier_onnx import ClassifierOnnx
from src.decision_tree import CONFIDENCE_THRESHOLD, Identification
from src.detector_onnx import YoloOnnxDetector
from src.imageops import assess_quality, load_image

log = logging.getLogger("insectvision")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="InsectVision", version="1.1.0")

config.STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

ACCEPTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
# REVIEW: image/heic and image/heif are accepted here but Pillow cannot decode
# them without the pillow-heif plugin (not in requirements-deploy.txt), so an
# iPhone HEIC upload currently gets a 400 "could not be decoded" rather than a
# 415. Either add pillow-heif (native wheel, ~10 MB) or drop the two types and
# let the UI's existing "export as JPEG" hint do the work.

# A detected region narrower than this (in original-image pixels) is skipped
# before classification. The old cutoff was 4px, which is merely "not empty";
# below ~8px there is no image content for the classifier to upscale from.
# Deliberately still small: a distant insect crop that's borderline is better
# sent through and allowed to land on the reject class than silently dropped.
MIN_CROP_PX = 8


# --------------------------------------------------------------------- #
# Lazy-loaded, process-wide singletons -- models load once per process,
# not once per request, and only once either mode actually needs them.
#
# The endpoint below is a plain `def`, so FastAPI runs it in a threadpool:
# two first requests arriving together would otherwise BOTH construct an
# InferenceSession (each a ~100 MB+ allocation on a 512 MB tier). The lock
# makes the first loader win and the second reuse it.
# --------------------------------------------------------------------- #
_species_cfg: Optional[dict] = None
_classifier: Optional[ClassifierOnnx] = None
_detector: Optional[YoloOnnxDetector] = None
# RLock, not Lock: get_classifier() calls get_species_cfg() while holding it.
_load_lock = threading.RLock()


def get_species_cfg() -> dict:
    global _species_cfg
    if _species_cfg is None:
        with _load_lock:
            if _species_cfg is None:
                if not config.SPECIES_CONFIG.exists():
                    raise HTTPException(500, "config/species.json is missing.")
                cfg = json.loads(config.SPECIES_CONFIG.read_text(encoding="utf-8"))
                _validate_species_cfg(cfg)
                _species_cfg = cfg
    return _species_cfg


def _validate_species_cfg(cfg: dict) -> None:
    """Fail at first use with a message that names the problem, rather than
    with a KeyError deep inside a request."""
    names = cfg.get("class_names")
    if not isinstance(names, list) or not names:
        raise HTTPException(500, "species.json: class_names must be a non-empty list.")
    if len(names) != len(set(names)):
        raise HTTPException(500, "species.json: class_names contains duplicates.")
    reject = cfg.get("reject_class")
    if reject is not None and reject not in names:
        raise HTTPException(500, f"species.json: reject_class '{reject}' is not in class_names.")
    for taxon in names:
        if taxon != reject and taxon not in cfg.get("taxon_status", {}):
            log.warning("species.json: taxon '%s' has no taxon_status entry; UI will show neutral", taxon)


def reject_class() -> Optional[str]:
    return get_species_cfg().get("reject_class")


def classifier_path() -> Path:
    return config.resolve(get_species_cfg().get("classifier_onnx", "models/classifier.onnx"))


def detector_path() -> Path:
    return config.resolve(get_species_cfg().get("detector_onnx", "models/detector.onnx"))


def current_mode() -> str:
    """Cascade if both model files are present, classifier-only if just the
    classifier is, else not_configured -- checked fresh each call (cheap
    stat() calls) rather than cached, so dropping detector.onnx in after
    the process has already started is picked up without a restart."""
    if not classifier_path().exists():
        return "not_configured"
    return "cascade" if detector_path().exists() else "classifier_only"


def get_classifier() -> ClassifierOnnx:
    global _classifier
    if _classifier is None:
        with _load_lock:
            if _classifier is None:
                cfg = get_species_cfg()
                path = classifier_path()
                if not path.exists():
                    raise HTTPException(503, f"Classifier model {path.name} not found in models/ -- "
                                             "train it with notebooks/train_cascade_colab.ipynb.")
                try:
                    _classifier = ClassifierOnnx(str(path), cfg["class_names"],
                                                 cfg.get("classifier_input_size", 224))
                except ValueError as e:
                    # Model/config mismatch (e.g. a 12-class ONNX against a
                    # 13-entry class list). Refusing to serve is the right
                    # call: serving would return wrong species names.
                    log.error("Classifier rejected at load: %s", e)
                    raise HTTPException(503, f"Classifier unavailable: {e}")
                log.info("Loaded classifier %s (%d classes)", path.name, len(cfg["class_names"]))
    return _classifier


def get_detector() -> YoloOnnxDetector:
    global _detector
    if _detector is None:
        with _load_lock:
            if _detector is None:
                cfg = get_species_cfg()
                path = detector_path()
                try:
                    _detector = YoloOnnxDetector(
                        str(path), input_size=cfg.get("detector_input_size", 640),
                        conf_threshold=cfg.get("detector_conf_threshold",
                                               YoloOnnxDetector.DEFAULT_CONF_THRESHOLD))
                except ValueError as e:
                    log.error("Detector rejected at load: %s", e)
                    raise HTTPException(503, f"Detector unavailable: {e}")
                log.info("Loaded detector %s (conf>=%.2f)", path.name, _detector.conf_threshold)
    return _detector


# --------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------- #
class QualityOut(BaseModel):
    score: float
    passed: bool
    reason: str = ""


class DetectionOut(BaseModel):
    box: list[float]
    objectness: float
    taxon: str
    confidence: float
    # One of "confirmed" | "uncertain" | "no_specimen" -- see module docstring.
    verdict: str
    # False when taxon is the reject class: this region is not a recognised
    # insect. The UI must render a neutral "not an insect" state for it --
    # no species card, no pest/beneficial colour, and never a tally entry.
    is_specimen: bool
    # HARD RULE: True for anything that is not a confirmed identification
    # (both uncertain AND no_specimen) -- the one bit every consumer needs:
    # "may I count this?" Kept for backwards compatibility with clients that
    # only know about the old flagged/not-flagged split.
    flagged: bool
    # Second-best guess for this same box. For a specimen it shows when the
    # model is choosing between two look-alikes (weevil vs beetle -- weevils
    # ARE beetles taxonomically, so split confidence there is expected). For
    # a no_specimen verdict it's the closest insect the model considered,
    # which the UI may mention but must never present as an identification.
    runner_up_taxon: Optional[str] = None
    runner_up_confidence: Optional[float] = None


class TopPredictionOut(BaseModel):
    taxon: str
    confidence: float
    is_specimen: bool = True


class AnalyseResponse(BaseModel):
    mode: str
    quality: QualityOut
    detections: list[DetectionOut] = []            # cascade mode
    top_predictions: list[TopPredictionOut] = []   # classifier-only mode
    # classifier-only mode: verdict for the whole image (its single
    # specimen). Cascade mode carries one verdict per detection instead.
    verdict: Optional[str] = None
    latency_ms: float


class SpeciesInfoOut(BaseModel):
    common_name: str = ""
    scientific_name: Optional[str] = None
    damage_symptoms: Optional[str] = None
    is_specimen: bool = True


class HealthResponse(BaseModel):
    status: str
    mode: str
    class_names: list[str]
    # Name of the "not an insect" class, or null for a legacy closed-set
    # model. The UI hides this class from the supported-species list.
    reject_class: Optional[str] = None
    # Published so the UI draws its threshold tick from the same number the
    # server decides with, instead of a second hardcoded 0.75.
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    taxon_status: dict[str, str] = {}
    species_info: dict[str, SpeciesInfoOut] = {}
    classifier_macro_f1: Optional[float] = None
    detector_map50: Optional[float] = None


# --------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------- #
@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    cfg = get_species_cfg()
    return HealthResponse(
        status="ok",
        mode=current_mode(),
        class_names=cfg.get("class_names", []),
        reject_class=cfg.get("reject_class"),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        taxon_status=cfg.get("taxon_status", {}),
        species_info=cfg.get("species_info", {}),
        classifier_macro_f1=cfg.get("classifier_macro_f1"),
        detector_map50=cfg.get("detector_map50"),
    )


def _identify(candidates, reject: Optional[str]) -> Identification:
    top = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    return Identification(
        taxon=top.taxon, confidence=top.confidence,
        runner_up_taxon=runner_up.taxon if runner_up else None,
        runner_up_confidence=runner_up.confidence if runner_up else None,
        # A legacy model with no reject class must never accidentally match
        # a real taxon named like the default, so pass a name no class has.
        reject_class=reject if reject is not None else "\x00none")


# Plain `def`, not `async def`: ONNX inference and PIL decoding are CPU-bound
# and synchronous. Inside an `async def` they'd run ON the event loop and
# freeze every other request (including /health) for the duration; as a sync
# endpoint FastAPI runs this in its threadpool instead.
@app.post("/api/v1/analyse", response_model=AnalyseResponse)
def analyse(image: UploadFile = File(...)):
    start = time.perf_counter()

    if image.content_type not in ACCEPTED_CONTENT_TYPES:
        raise HTTPException(415, "Unsupported image format.")

    # Read at most limit+1 bytes: enough to know the limit was exceeded
    # without buffering an arbitrarily large body first.
    raw = image.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB).")
    if not raw:
        raise HTTPException(400, "Empty upload.")

    try:
        img = load_image(raw)
    except Exception:
        raise HTTPException(400, "Image could not be decoded.")

    quality = assess_quality(img)
    mode = current_mode()

    # Quality gate first, before touching any model: a refusal to answer is
    # preferable to an answer derived from unusable evidence, and it's the
    # one check that costs nothing regardless of which mode is active.
    if not quality.passed:
        return AnalyseResponse(
            mode=mode,
            quality=QualityOut(score=quality.score, passed=False, reason=quality.reason),
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    if mode == "not_configured":
        raise HTTPException(503, f"Classifier model {classifier_path().name} is not present in "
                                 "models/ -- train it with notebooks/train_cascade_colab.ipynb "
                                 "and drop the exported file in.")

    try:
        detections_out, top_predictions_out, whole_image_verdict = _run_models(img, mode)
    except HTTPException:
        raise
    except Exception:
        # Anything unexpected inside inference (a corrupt-but-decodable
        # image, an ONNX Runtime error) must not surface as a raw traceback
        # with a 500 page -- log it server-side, tell the client cleanly.
        log.exception("Inference failed (mode=%s, image=%sx%s)", mode, *img.size)
        raise HTTPException(500, "Analysis failed unexpectedly. Please try another photo.")

    return AnalyseResponse(
        mode=mode,
        quality=QualityOut(score=quality.score, passed=True),
        detections=detections_out,
        top_predictions=top_predictions_out,
        verdict=whole_image_verdict,
        latency_ms=(time.perf_counter() - start) * 1000,
    )


def _run_models(img, mode: str):
    detections_out: list[DetectionOut] = []
    top_predictions_out: list[TopPredictionOut] = []
    whole_image_verdict: Optional[str] = None
    reject = reject_class()

    if mode == "cascade":
        detector = get_detector()
        classifier = get_classifier()
        # Design note: when the detector finds nothing, the cascade returns
        # zero detections and does NOT fall back to classifying the whole
        # image -- "no insect found" is the honest answer, and the user can
        # crop in with the UI's zoom tool if the insect was just small.
        for det in detector.detect(img):
            x1, y1, x2, y2 = det.box  # already clamped to the image by the detector
            if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
                continue
            crop = img.crop((int(x1), int(y1), int(round(x2)), int(round(y2))))
            ident = _identify(classifier.classify(crop, top_k=2), reject)
            detections_out.append(DetectionOut(
                box=[x1, y1, x2, y2], objectness=det.objectness,
                taxon=ident.taxon, confidence=ident.confidence,
                verdict=ident.verdict.value, is_specimen=ident.is_specimen,
                flagged=ident.flagged,
                runner_up_taxon=ident.runner_up_taxon,
                runner_up_confidence=ident.runner_up_confidence))
    else:  # classifier_only -- whole image treated as one specimen
        classifier = get_classifier()
        top3 = classifier.classify(img, top_k=3)
        top_predictions_out = [TopPredictionOut(taxon=t.taxon, confidence=t.confidence,
                                                is_specimen=(t.taxon != reject)) for t in top3]
        whole_image_verdict = _identify(top3, reject).verdict.value

    return detections_out, top_predictions_out, whole_image_verdict


@app.get("/")
def index():
    index_path = config.STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "InsectVision API is running. Web UI not built yet (Step 5)."}
