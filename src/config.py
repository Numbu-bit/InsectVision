"""
Filesystem locations, resolved from this file's own position rather than
the current working directory -- so `python app.py`, `uvicorn app:app` from
inside insectvision/, and pytest from anywhere all agree on where
config/species.json and models/ actually are.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Overridable so tests (Step 6) can point at a throwaway species.json with
# synthetic model paths instead of touching the real, not-yet-trained one.
SPECIES_CONFIG = Path(os.environ.get("INSECTVISION_SPECIES_CONFIG", ROOT / "config" / "species.json"))
MODELS_DIR = ROOT / "models"
STATIC_DIR = ROOT / "static"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def resolve(path_str: str) -> Path:
    """species.json stores model paths like 'models/classifier.onnx' --
    relative to ROOT, not to whatever the caller's cwd happens to be."""
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p
