"""
FastAPI backend AND web UI host for InsectVision, deliberately one service:
a single Render instance serves both the JSON API and the static UI (Step 5)
from the same origin, so there's no CORS configuration to get wrong and no
second deployment to keep in sync.

    uvicorn app:app --reload
"""
import json
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import config
from src.classifier_onnx import ClassifierOnnx
from src.decision_tree import CONFIDENCE_THRESHOLD, Identification, TaxonStatus, evaluate
from src.detector_onnx import YoloOnnxDetector
from src.imageops import assess_quality, load_image

app = FastAPI(title="InsectVision", version="1.0.0")

config.STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


# --------------------------------------------------------------------- #
# Lazy-loaded, process-wide singletons -- models load once per process,
# not once per request, and only once either mode actually needs them.
# --------------------------------------------------------------------- #
_species_cfg: Optional[dict] = None
_classifier: Optional[ClassifierOnnx] = None
_detector: Optional[YoloOnnxDetector] = None


def get_species_cfg() -> dict:
    global _species_cfg
    if _species_cfg is None:
        if not config.SPECIES_CONFIG.exists():
            raise HTTPException(500, "config/species.json is missing.")
        _species_cfg = json.loads(config.SPECIES_CONFIG.read_text())
    return _species_cfg


def current_mode() -> str:
    """Cascade if both model files are present, classifier-only if just the
    classifier is, else not_configured -- checked fresh each call (cheap
    stat() calls) rather than cached, so dropping detector.onnx in after
    the process has already started is picked up without a restart."""
    cfg = get_species_cfg()
    classifier_path = config.resolve(cfg.get("classifier_onnx", "models/classifier.onnx"))
    detector_path = config.resolve(cfg.get("detector_onnx", "models/detector.onnx"))
    if not classifier_path.exists():
        return "not_configured"
    return "cascade" if detector_path.exists() else "classifier_only"


def get_classifier() -> ClassifierOnnx:
    global _classifier
    if _classifier is None:
        cfg = get_species_cfg()
        path = config.resolve(cfg.get("classifier_onnx", "models/classifier.onnx"))
        if not path.exists():
            raise HTTPException(503, "Classifier model not trained yet -- see "
                                "notebooks/train_cascade_colab.ipynb.")
        _classifier = ClassifierOnnx(str(path), cfg["class_names"],
                                     cfg.get("classifier_input_size", 224))
    return _classifier


def get_detector() -> YoloOnnxDetector:
    global _detector
    if _detector is None:
        cfg = get_species_cfg()
        path = config.resolve(cfg.get("detector_onnx", "models/detector.onnx"))
        _detector = YoloOnnxDetector(str(path), input_size=cfg.get("detector_input_size", 640))
    return _detector


def get_taxon_status() -> dict[str, TaxonStatus]:
    cfg = get_species_cfg()
    return {k: TaxonStatus(v) for k, v in cfg.get("taxon_status", {}).items()}


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
    flagged: bool


class TopPredictionOut(BaseModel):
    taxon: str
    confidence: float


class DecisionOut(BaseModel):
    severity: str
    action: str
    advisory: str
    path: list[str]
    flagged: list[TopPredictionOut] = []


class AnalyseResponse(BaseModel):
    mode: str
    quality: QualityOut
    detections: list[DetectionOut] = []
    top_predictions: list[TopPredictionOut] = []
    decision: Optional[DecisionOut] = None
    latency_ms: float


class SpeciesInfoOut(BaseModel):
    common_name: str = ""
    scientific_name: str = ""
    damage_symptoms: str = ""


class HealthResponse(BaseModel):
    status: str
    mode: str
    class_names: list[str]
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
        taxon_status=cfg.get("taxon_status", {}),
        species_info=cfg.get("species_info", {}),
        classifier_macro_f1=cfg.get("classifier_macro_f1"),
        detector_map50=cfg.get("detector_map50"),
    )


@app.post("/api/v1/analyse", response_model=AnalyseResponse)
async def analyse(image: UploadFile = File(...),
                  growth_stage: str = Form("vegetative"),
                  observed_count: int = Form(1)):
    start = time.perf_counter()

    if image.content_type not in {"image/jpeg", "image/png", "image/webp",
                                  "image/heic", "image/heif"}:
        raise HTTPException(415, "Unsupported image format.")

    raw = await image.read()
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB).")

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
        raise HTTPException(503, "No trained classifier yet -- see "
                            "notebooks/train_cascade_colab.ipynb, then drop "
                            "classifier.onnx into models/.")

    taxon_status = get_taxon_status()
    cfg = get_species_cfg()
    economic_thresholds = cfg.get("economic_thresholds", {})

    detections_out: list[DetectionOut] = []
    top_predictions_out: list[TopPredictionOut] = []
    identifications: list[Identification] = []

    if mode == "cascade":
        detector = get_detector()
        classifier = get_classifier()
        raw_detections = detector.detect(img)

        for det in raw_detections:
            x1, y1, x2, y2 = [max(0.0, v) for v in det.box]
            crop = img.crop((x1, y1, x2, y2))
            if crop.size[0] < 4 or crop.size[1] < 4:
                continue  # degenerate box from an unusable region -- skip rather than crash

            top = classifier.classify(crop, top_k=1)[0]
            flagged = top.confidence < CONFIDENCE_THRESHOLD
            detections_out.append(DetectionOut(
                box=[x1, y1, x2, y2], objectness=det.objectness,
                taxon=top.taxon, confidence=top.confidence, flagged=flagged))
            identifications.append(Identification(taxon=top.taxon, confidence=top.confidence, count=1))
    else:  # classifier_only -- whole image is one specimen, count is manual
        classifier = get_classifier()
        top3 = classifier.classify(img, top_k=3)
        top_predictions_out = [TopPredictionOut(taxon=t.taxon, confidence=t.confidence) for t in top3]

        best = top3[0]
        count = max(1, observed_count)
        identifications.append(Identification(taxon=best.taxon, confidence=best.confidence, count=count))

    decision = evaluate(identifications, taxon_status, growth_stage, economic_thresholds)

    return AnalyseResponse(
        mode=mode,
        quality=QualityOut(score=quality.score, passed=True),
        detections=detections_out,
        top_predictions=top_predictions_out,
        decision=DecisionOut(
            severity=decision.severity.value,
            action=decision.action,
            advisory=decision.advisory,
            path=decision.path,
            flagged=[TopPredictionOut(taxon=f.taxon, confidence=f.confidence) for f in decision.flagged],
        ),
        latency_ms=(time.perf_counter() - start) * 1000,
    )


@app.get("/")
def index():
    index_path = config.STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "InsectVision API is running. Web UI not built yet (Step 5)."}
