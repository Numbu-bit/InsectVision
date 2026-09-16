import tracemalloc

import numpy as np
import pytest
from PIL import Image, ImageFilter

from src import imageops


def _textured(w, h, seed=0):
    """A sharp, well-lit synthetic image: random blobs + edges, mid-grey mean."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(60, 200, (h, w, 3), dtype=np.uint8)
    arr[::7, :, :] = 255  # hard edges every 7 rows
    arr[:, ::11, :] = 0
    return Image.fromarray(arr)


def test_quality_score_is_resolution_independent():
    """The same sharp content must score the same whether it arrives as a
    500px web image or a 12MP phone photo. Before the fix, the 12MP version
    of a passing photo scored 0.52 and was rejected as blurry."""
    small = _textured(500, 375)
    big = small.resize((4000, 3000), Image.LANCZOS)
    q_small, q_big = imageops.assess_quality(small), imageops.assess_quality(big)
    assert q_small.passed and q_big.passed
    assert abs(q_small.score - q_big.score) < 0.15


def test_quality_gate_memory_is_bounded_on_large_photos():
    big = _textured(4000, 3000)
    tracemalloc.start()
    imageops.assess_quality(big)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 60 * 1024 * 1024, f"quality gate peaked at {peak/1e6:.0f} MB on a 12MP image"


def test_blurred_photo_is_rejected_and_sharp_passes():
    sharp = _textured(800, 600)
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=8))
    assert imageops.assess_quality(sharp).passed
    q = imageops.assess_quality(blurred)
    assert not q.passed and "blurred" in q.reason


def test_too_small_image_is_rejected_with_reason():
    q = imageops.assess_quality(Image.new("RGB", (50, 50)))
    assert not q.passed and q.score == 0.0 and "too small" in q.reason


def test_near_black_textured_image_is_rejected_as_dark():
    """Regression: luminance alone could never fail a sharp image under the
    blended score, so an effectively black (mean ~10/255) textured image
    passed at 0.78. The hard bound catches it, with the right reason."""
    arr = np.asarray(_textured(400, 400)).astype(np.float32) * 0.08
    q = imageops.assess_quality(Image.fromarray(arr.astype(np.uint8)))
    assert not q.passed and "dark" in q.reason


def test_washed_out_image_is_rejected():
    arr = 255 - (255 - np.asarray(_textured(400, 400)).astype(np.float32)) * 0.05
    q = imageops.assess_quality(Image.fromarray(arr.astype(np.uint8)))
    assert not q.passed and "glare" in q.reason


def test_dim_but_usable_photo_still_passes():
    # Mean luminance ~45: below ideal, above the hard floor; sharp -> should pass.
    arr = np.asarray(_textured(400, 400)).astype(np.float32) * 0.34
    assert imageops.assess_quality(Image.fromarray(arr.astype(np.uint8))).passed


def test_letterbox_roundtrip():
    img = Image.new("RGB", (800, 400))
    canvas, scale, pad = imageops.letterbox(img, 640)
    assert canvas.size == (640, 640)
    # a box in letterboxed coords maps back to original coords
    box = imageops.unletterbox_box([pad[0], pad[1], pad[0] + 640, pad[1] + 320], scale, pad)
    assert np.allclose(box, [0, 0, 800, 400])


def test_clamp_box():
    assert imageops.clamp_box([-5, 10, 700, 650], 640, 640) == [0.0, 10.0, 640.0, 640.0]
    assert imageops.clamp_box([300, 300, 100, 100], 640, 640) == [100.0, 100.0, 300.0, 300.0]


def test_to_chw_float32_layout_and_range():
    arr = imageops.to_chw_float32(Image.new("RGB", (4, 3), (255, 0, 128)))
    assert arr.shape == (3, 3, 4) and arr.dtype == np.float32
    assert arr[0].max() == 1.0 and arr[1].max() == 0.0 and abs(arr[2].max() - 128 / 255) < 1e-6
