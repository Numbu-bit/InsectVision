import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tests.conftest import ROOT


def _img(path: Path, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(path)


def _run(tmp_path: Path, with_negatives: bool):
    raw = tmp_path / "raw" / "wrapper" / "Agri Pests"
    for k, cls in enumerate(["Ants", "Beetle", "Moth"]):
        for i in range(6):
            _img(raw / cls / f"{cls.lower()} ({i}).jpg", seed=k * 100 + i)
        # exact duplicate under another name -> must be dropped
        src = raw / cls / f"{cls.lower()} (0).jpg"
        (raw / cls / "dupe.jpg").write_bytes(src.read_bytes())
    neg = tmp_path / "negatives"
    if with_negatives:
        for i in range(10):
            _img(neg / ("leaves" if i % 2 else "soil") / f"neg{i}.png", seed=900 + i)
    cfg_path = tmp_path / "species.json"
    out = tmp_path / "classify"
    cmd = [sys.executable, str(ROOT / "scripts" / "prepare_data.py"), "--source", str(tmp_path / "raw"),
           "--negatives", str(neg), "--out", str(out), "--species-config", str(cfg_path),
           "--val-fraction", "0.34", "--min-images", "1"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stdout + res.stderr
    return res.stdout, json.loads(cfg_path.read_text()), out


def test_negatives_become_other_class_and_config_is_reject_aware(tmp_path):
    stdout, cfg, out = _run(tmp_path, with_negatives=True)
    assert cfg["class_names"] == ["ants", "beetle", "moth", "other"]
    assert cfg["reject_class"] == "other"
    assert "other" not in cfg["taxon_status"]
    assert cfg["species_info"]["other"] == {"common_name": "Not an insect", "scientific_name": None,
                                            "damage_symptoms": None, "is_specimen": False}
    assert cfg["classifier_onnx"] == "models/classifier_4cls.onnx"
    assert cfg["detector_conf_threshold"] == 0.25
    assert (out / "train" / "other").is_dir() and (out / "val" / "other").is_dir()
    assert sum(1 for _ in (out / "train" / "other").iterdir()) + sum(1 for _ in (out / "val" / "other").iterdir()) == 10


def test_exact_duplicates_are_dropped(tmp_path):
    stdout, cfg, out = _run(tmp_path, with_negatives=True)
    assert "1 exact dupes dropped" in stdout
    for cls in ("ants", "beetle", "moth"):
        n = sum(1 for _ in (out / "train" / cls).iterdir()) + sum(1 for _ in (out / "val" / cls).iterdir())
        assert n == 6, f"{cls}: {n} files, duplicate not removed"


def test_missing_negatives_warns_and_produces_closed_set_config(tmp_path):
    stdout, cfg, out = _run(tmp_path, with_negatives=False)
    assert "WARNING: no negative images" in stdout
    assert cfg["class_names"] == ["ants", "beetle", "moth"]
    assert cfg["reject_class"] is None
    assert "other" not in cfg["species_info"]
    assert not (out / "train" / "other").exists()
