import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The trained classifier named by config/species.json (classifier_13cls.onnx).
# Tests that need real inference use it; they skip cleanly if it hasn't been
# trained/dropped in yet.
_cfg = json.loads((ROOT / "config" / "species.json").read_text(encoding="utf-8"))
CLASSIFIER = ROOT / _cfg.get("classifier_onnx", "models/classifier.onnx")
DETECTOR = ROOT / "models" / "detector.onnx"


@pytest.fixture(scope="session")
def species_cfg() -> dict:
    return json.loads((ROOT / "config" / "species.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def specimen_class_names(species_cfg) -> list[str]:
    """The 12 insect classes, i.e. class_names minus the reject class."""
    return [c for c in species_cfg["class_names"] if c != species_cfg.get("reject_class")]


def requires_classifier():
    return pytest.mark.skipif(not CLASSIFIER.exists(), reason=f"{CLASSIFIER.name} not present -- train it first")


def requires_detector():
    return pytest.mark.skipif(not DETECTOR.exists(), reason="models/detector.onnx not present")
