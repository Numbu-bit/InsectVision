import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The shipped 12-class model, if present. Tests that need a real classifier
# use it with a matching 12-entry class list; the app's own species.json may
# by now describe a 13-class model that hasn't been dropped in yet.
LEGACY_CLASSIFIER = ROOT / "models" / "classifier.onnx"
DETECTOR = ROOT / "models" / "detector.onnx"


@pytest.fixture(scope="session")
def species_cfg() -> dict:
    return json.loads((ROOT / "config" / "species.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def legacy_class_names(species_cfg) -> list[str]:
    return [c for c in species_cfg["class_names"] if c != species_cfg.get("reject_class")]


def requires_legacy_classifier():
    return pytest.mark.skipif(not LEGACY_CLASSIFIER.exists(), reason="models/classifier.onnx not present")


def requires_detector():
    return pytest.mark.skipif(not DETECTOR.exists(), reason="models/detector.onnx not present")
