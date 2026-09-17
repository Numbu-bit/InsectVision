"""API-level tests through FastAPI's TestClient.

Tests that need live inference use the trained classifier named in
config/species.json and skip cleanly if it isn't present. The verdict wiring
is exercised independently with a stubbed classifier.
"""
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as appmod
from src.classifier_onnx import ClassResult
from tests.conftest import CLASSIFIER, DETECTOR, ROOT, requires_classifier, requires_detector


def _reset(monkeypatch, cfg: dict, tmp_path):
    p = tmp_path / "species.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setattr(appmod.config, "SPECIES_CONFIG", p)
    monkeypatch.setattr(appmod, "_species_cfg", None)
    monkeypatch.setattr(appmod, "_classifier", None)
    monkeypatch.setattr(appmod, "_detector", None)
    return TestClient(appmod.app)


def _jpeg(img: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    buf.seek(0)
    return buf


def _sharp_photo(w=640, h=480, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(60, 200, (h, w, 3), dtype=np.uint8)
    arr[::7, :, :] = 255
    return Image.fromarray(arr)


def _post(client, buf, ctype="image/jpeg"):
    return client.post("/api/v1/analyse", files={"image": ("x.jpg", buf, ctype)})


@pytest.fixture
def real_cfg(species_cfg):
    return dict(species_cfg)


# ----------------------------------------------------------------- health / config
def test_health_exposes_reject_class_and_threshold(monkeypatch, species_cfg, tmp_path):
    c = _reset(monkeypatch, species_cfg, tmp_path)
    h = c.get("/api/v1/health").json()
    assert h["reject_class"] == species_cfg["reject_class"]
    assert h["confidence_threshold"] == 0.75
    assert h["species_info"]["other"]["is_specimen"] is False
    assert h["species_info"]["other"]["scientific_name"] is None


def test_missing_classifier_reports_not_configured_and_503(monkeypatch, species_cfg, tmp_path):
    cfg = dict(species_cfg, classifier_onnx="models/does_not_exist.onnx")
    c = _reset(monkeypatch, cfg, tmp_path)
    assert c.get("/api/v1/health").json()["mode"] == "not_configured"
    r = _post(c, _jpeg(_sharp_photo()))
    assert r.status_code == 503 and "does_not_exist.onnx" in r.json()["detail"]


def test_invalid_species_config_is_a_clear_500(monkeypatch, species_cfg, tmp_path):
    cfg = dict(species_cfg, reject_class="not_a_class")
    c = _reset(monkeypatch, cfg, tmp_path)
    r = c.get("/api/v1/health")
    assert r.status_code == 500 and "reject_class" in r.json()["detail"]


@requires_classifier()
def test_class_count_mismatch_is_refused_not_served(monkeypatch, species_cfg, specimen_class_names, tmp_path):
    cfg = dict(species_cfg, class_names=specimen_class_names, reject_class=None)  # 12 names, 13-output file
    cfg["detector_onnx"] = "models/does_not_exist.onnx"                            # force classifier-only
    c = _reset(monkeypatch, cfg, tmp_path)
    r = _post(c, _jpeg(_sharp_photo()))
    assert r.status_code == 503 and "out of sync" in r.json()["detail"]


# ----------------------------------------------------------------- input validation
@requires_classifier()
class TestInputValidation:
    def test_unsupported_type(self, monkeypatch, real_cfg, tmp_path):
        c = _reset(monkeypatch, real_cfg, tmp_path)
        assert _post(c, _jpeg(_sharp_photo()), ctype="text/plain").status_code == 415

    def test_undecodable(self, monkeypatch, real_cfg, tmp_path):
        c = _reset(monkeypatch, real_cfg, tmp_path)
        assert _post(c, io.BytesIO(b"definitely not a jpeg")).status_code == 400

    def test_empty(self, monkeypatch, real_cfg, tmp_path):
        c = _reset(monkeypatch, real_cfg, tmp_path)
        assert _post(c, io.BytesIO(b"")).status_code == 400

    def test_too_large(self, monkeypatch, real_cfg, tmp_path):
        c = _reset(monkeypatch, real_cfg, tmp_path)
        assert _post(c, io.BytesIO(b"\xff" * (appmod.config.MAX_UPLOAD_BYTES + 1))).status_code == 413

    def test_quality_gate_short_circuits_before_models(self, monkeypatch, real_cfg, tmp_path):
        c = _reset(monkeypatch, real_cfg, tmp_path)
        d = _post(c, _jpeg(Image.new("RGB", (50, 50)))).json()
        assert d["quality"]["passed"] is False and d["detections"] == [] and d["top_predictions"] == []


# ----------------------------------------------------------------- verdict wiring (stubbed classifier)
class _StubClassifier:
    def __init__(self, results):
        self.results = results

    def classify(self, img, top_k=3):
        return self.results[:top_k]


def test_classifier_only_no_specimen_verdict(monkeypatch, species_cfg, tmp_path):
    cfg = dict(species_cfg, detector_onnx="models/nope.onnx", classifier_onnx="config/species.json")  # any existing file -> classifier_only
    c = _reset(monkeypatch, cfg, tmp_path)
    monkeypatch.setattr(appmod, "get_classifier", lambda: _StubClassifier(
        [ClassResult("other", 0.90), ClassResult("beetle", 0.06), ClassResult("moth", 0.02)]))
    d = _post(c, _jpeg(_sharp_photo())).json()
    assert d["mode"] == "classifier_only"
    assert d["verdict"] == "no_specimen"
    assert d["top_predictions"][0] == {"taxon": "other", "confidence": 0.90, "is_specimen": False}
    assert d["top_predictions"][1]["is_specimen"] is True


def test_classifier_only_uncertain_and_confirmed(monkeypatch, species_cfg, tmp_path):
    cfg = dict(species_cfg, detector_onnx="models/nope.onnx", classifier_onnx="config/species.json")
    c = _reset(monkeypatch, cfg, tmp_path)
    monkeypatch.setattr(appmod, "get_classifier", lambda: _StubClassifier(
        [ClassResult("beetle", 0.56), ClassResult("other", 0.30), ClassResult("weevil", 0.10)]))
    assert _post(c, _jpeg(_sharp_photo())).json()["verdict"] == "uncertain"
    monkeypatch.setattr(appmod, "get_classifier", lambda: _StubClassifier(
        [ClassResult("beetle", 0.92), ClassResult("weevil", 0.05), ClassResult("other", 0.01)]))
    assert _post(c, _jpeg(_sharp_photo())).json()["verdict"] == "confirmed"


def test_legacy_model_without_reject_class_never_yields_no_specimen(monkeypatch, species_cfg, specimen_class_names, tmp_path):
    cfg = dict(species_cfg, class_names=specimen_class_names, reject_class=None,
               detector_onnx="models/nope.onnx", classifier_onnx="config/species.json")
    c = _reset(monkeypatch, cfg, tmp_path)
    monkeypatch.setattr(appmod, "get_classifier", lambda: _StubClassifier([ClassResult("beetle", 0.4), ClassResult("ants", 0.3)]))
    d = _post(c, _jpeg(_sharp_photo())).json()
    assert d["verdict"] == "uncertain" and all(p["is_specimen"] for p in d["top_predictions"])


def test_inference_exception_is_a_clean_500(monkeypatch, species_cfg, tmp_path):
    cfg = dict(species_cfg, detector_onnx="models/nope.onnx", classifier_onnx="config/species.json")
    c = _reset(monkeypatch, cfg, tmp_path)

    class Boom:
        def classify(self, *a, **k):
            raise RuntimeError("onnxruntime exploded")
    monkeypatch.setattr(appmod, "get_classifier", lambda: Boom())
    r = _post(c, _jpeg(_sharp_photo()))
    assert r.status_code == 500 and "exploded" not in r.json()["detail"]


# ----------------------------------------------------------------- cascade with real models
@requires_classifier()
@requires_detector()
def test_cascade_detections_carry_verdict_fields(monkeypatch, real_cfg, tmp_path):
    c = _reset(monkeypatch, real_cfg, tmp_path)
    sample = next((p for p in [ROOT.parent.parent / "beetle.jpg", ROOT.parent.parent / "ant.jpg"] if p.exists()), None)
    if sample is None:
        pytest.skip("no sample insect photo next to the project")
    d = _post(c, open(sample, "rb")).json()
    assert d["mode"] == "cascade" and d["detections"], d
    for det in d["detections"]:
        assert det["verdict"] in {"confirmed", "uncertain", "no_specimen"}
        assert det["is_specimen"] is True  # a real insect photo must not be rejected
        assert det["flagged"] == (det["verdict"] != "confirmed")
        x1, y1, x2, y2 = det["box"]
        assert 0 <= x1 <= x2 and 0 <= y1 <= y2  # clamped, ordered
