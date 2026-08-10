"""
Image quality gate and geometric normalisation, using Pillow + numpy only.

No OpenCV here on purpose: this module runs both in data prep and at serve
time on Render's 512 MB free tier, where an extra native-code dependency
(OpenCV ships large compiled shared libraries) costs RAM and cold-start time
the instance doesn't have to spare. Everything below is implemented with
plain array arithmetic instead.
"""
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

# Calibrated by eye against a handful of sharp vs. blurred test photos using
# the same variance-of-Laplacian idea OpenCV's cv2.Laplacian().var() uses.
# Revisit once real field photos are available in Step 2/3 -- this is a
# starting point, not a measured threshold.
BLUR_VARIANCE_THRESHOLD = 100.0
LUMINANCE_MIN = 40.0
LUMINANCE_MAX = 215.0

# Below this on the shorter side, even a perfect blur/luminance score can't
# be trusted -- letterboxing something this small up to the model's working
# resolution is mostly fabricating detail, not preserving it. This is a hard
# floor, unlike RESOLUTION_TARGET below.
MIN_USABLE_DIMENSION = 96

# Shorter-side pixel count at which the *resolution component* of the score
# reaches 1.0. Deliberately NOT a hard cutoff on its own -- a small but
# genuinely sharp, well-lit photo should still be able to pass on the
# strength of blur/luminance, since the model resizes to its own working
# resolution regardless of what came in. An earlier version rejected
# anything under 320px outright before even measuring blur, which meant a
# perfectly clear close-up photo could get rejected with a misleading
# "move closer" message for a problem that had nothing to do with framing.
RESOLUTION_TARGET = 320

QUALITY_PASS_MARK = 0.60


@dataclass
class QualityReport:
    score: float
    passed: bool
    reason: str = ""


def load_image(data: bytes) -> Image.Image:
    """Decode uploaded bytes into a normalised RGB image.

    exif_transpose matters specifically for phone photos: many cameras
    write the "this way up" rotation into EXIF metadata rather than
    rotating the pixels, so skipping this step silently sends sideways
    images into the model.
    """
    img = Image.open(BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the discrete Laplacian, as a blur proxy.

    A sharp image has strong local intensity swings (high-frequency edge
    content), which the Laplacian responds to strongly; a blurred image
    doesn't, so its Laplacian response is low and flat -- hence low
    variance. Computed here via shifted-array subtraction (the standard
    [[0,1,0],[1,-4,1],[0,1,0]] kernel applied to interior pixels only) so no
    convolution library is needed.
    """
    center = gray[1:-1, 1:-1]
    up, down = gray[:-2, 1:-1], gray[2:, 1:-1]
    left, right = gray[1:-1, :-2], gray[1:-1, 2:]
    laplacian = up + down + left + right - 4.0 * center
    return float(laplacian.var())


def assess_quality(img: Image.Image) -> QualityReport:
    """Score image usability in [0, 1] and explain any rejection.

    Refusing to analyse an unusable photo is preferable to producing a
    severity judgement from evidence too poor to support it. Blur,
    luminance and resolution are blended into one weighted score rather
    than resolution acting as a separate hard veto, so a small-but-sharp
    photo isn't punished for a dimension that doesn't actually determine
    whether it's readable.
    """
    w, h = img.size
    shorter_side = min(w, h)
    if shorter_side < MIN_USABLE_DIMENSION:
        return QualityReport(
            0.0, False,
            f"Image is only {w}x{h}px, too small to analyse reliably. "
            "Retake closer to the subject or at a higher camera resolution.")

    gray = np.asarray(img.convert("L"), dtype=np.float64)
    blur_var = _laplacian_variance(gray)
    luminance = float(gray.mean())

    blur_score = min(blur_var / BLUR_VARIANCE_THRESHOLD, 1.0)
    resolution_score = min(shorter_side / RESOLUTION_TARGET, 1.0)
    if luminance < LUMINANCE_MIN:
        lum_score = luminance / LUMINANCE_MIN
        lum_reason = "Image is too dark; find better light."
    elif luminance > LUMINANCE_MAX:
        lum_score = max(0.0, (255.0 - luminance) / (255.0 - LUMINANCE_MAX))
        lum_reason = "Image is over-exposed; avoid direct glare."
    else:
        lum_score, lum_reason = 1.0, ""

    # Blur dominates (it destroys detail outright); luminance next;
    # resolution weighted lightest since the model resizes regardless of
    # input size and a modest-resolution sharp photo is still useful.
    score = 0.5 * blur_score + 0.3 * lum_score + 0.2 * resolution_score
    if score >= QUALITY_PASS_MARK:
        return QualityReport(score, True)

    # Explain whichever factor is actually weakest, rather than a fixed
    # priority order -- a low-resolution image that's ALSO blurry should
    # say so, not always blame the same thing.
    candidates = [(blur_score, "Image is blurred; hold the phone steady and retake.")]
    if lum_reason:
        candidates.append((lum_score, lum_reason))
    if resolution_score < 1.0:
        candidates.append((resolution_score,
                           f"Image is only {w}x{h}px, which limits reliable analysis. "
                           "Retake at a higher camera resolution if you can."))
    reason = min(candidates, key=lambda c: c[0])[1]
    return QualityReport(score, False, reason)


def letterbox(img: Image.Image, size: int = 640,
             pad_value: int = 114) -> tuple[Image.Image, float, tuple[int, int]]:
    """Resize preserving aspect ratio, padding to a square canvas.

    Aspect ratio must be preserved: a plain resize would distort the body
    proportions fine-grained species discrimination depends on. Returns the
    scale and padding needed to map boxes back to original coordinates.
    """
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (pad_value, pad_value, pad_value))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas, scale, (left, top)


def unletterbox_box(box, scale: float, pad: tuple[int, int]) -> list[float]:
    """Map a box from letterboxed coordinates back to the original image."""
    left, top = pad
    x1, y1, x2, y2 = box
    return [(x1 - left) / scale, (y1 - top) / scale,
            (x2 - left) / scale, (y2 - top) / scale]


def to_chw_float32(img: Image.Image) -> np.ndarray:
    """HWC uint8 RGB -> CHW float32 in [0, 1] -- the layout ONNX Runtime
    expects and the layout both models were exported to accept."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))
