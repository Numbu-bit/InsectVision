"""
YOLOv8 detection via ONNX Runtime only -- no ultralytics, no torch.

Ultralytics' own postprocessing helpers require torch, and torch alone is
far too large for Render's 512 MB free-tier serve process. This module
reimplements just the two pieces of postprocessing actually needed --
mapping letterboxed model output back to image coordinates, and
non-maximum suppression -- as plain numpy.
"""
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from PIL import Image

from . import imageops


@dataclass
class RawDetection:
    box: list[float]      # x1, y1, x2, y2 in ORIGINAL image coordinates
    objectness: float
    class_id: int


class YoloOnnxDetector:
    """Loads once per process (see app.py's lazy loader) and is reused
    across requests -- constructing an InferenceSession is comparatively
    expensive and there's no reason to pay that cost per request."""

    def __init__(self, onnx_path: str, input_size: int = 640,
                conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, img: Image.Image) -> list[RawDetection]:
        canvas, scale, pad = imageops.letterbox(img, self.input_size)
        tensor = imageops.to_chw_float32(canvas)[np.newaxis, ...]  # add batch dim

        raw = self.session.run(None, {self.input_name: tensor})[0]
        # Ultralytics YOLOv8 ONNX export shape is (1, 4 + num_classes,
        # num_boxes): box coords (cx, cy, w, h in the letterboxed frame) in
        # the first 4 rows, one per-class score row per class -- YOLOv8 has
        # no separate objectness channel like YOLOv5 did, so the per-box
        # "objectness" used for thresholding here is just its best class score.
        preds = raw[0].T  # -> (num_boxes, 4 + num_classes)

        boxes_letterboxed = _cxcywh_to_xyxy(preds[:, :4])
        class_scores = preds[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        objectness = class_scores.max(axis=1)

        keep = objectness >= self.conf_threshold
        boxes_letterboxed = boxes_letterboxed[keep]
        class_ids = class_ids[keep]
        objectness = objectness[keep]

        keep_idx = _nms(boxes_letterboxed, objectness, self.iou_threshold)

        detections = []
        for i in keep_idx:
            box = imageops.unletterbox_box(boxes_letterboxed[i], scale, pad)
            detections.append(RawDetection(
                box=box, objectness=float(objectness[i]), class_id=int(class_ids[i])))
        return detections


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression (standard reference algorithm),
    reimplemented here rather than imported from torchvision.ops so the
    serve process doesn't need torch just for this one function."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)

        order = rest[iou <= iou_threshold]
    return keep
