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

    # The detector is deliberately tuned for RECALL, not precision: its only
    # job is to propose regions, and the classifier's reject class ("other")
    # is what decides whether a proposal is actually an insect. Raising this
    # to suppress boxes on trees would also drop real insects (measured:
    # tree boxes scored 0.25-0.29, a real slug scored 0.33 -- no clean gap),
    # so rejection belongs downstream where there IS a clean signal.
    DEFAULT_CONF_THRESHOLD = 0.25
    DEFAULT_IOU_THRESHOLD = 0.45

    def __init__(self, onnx_path: str, input_size: int = 640,
                 conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                 iou_threshold: float = DEFAULT_IOU_THRESHOLD):
        if not 0.0 < conf_threshold < 1.0:
            raise ValueError(f"conf_threshold must be in (0, 1), got {conf_threshold}")
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._validate_model_shape(onnx_path)

    def _validate_model_shape(self, onnx_path: str) -> None:
        """A YOLOv8 export is (1, 4 + num_classes, num_anchors). Anything
        else -- a YOLOv5 export with its extra objectness row, a segmentation
        head, a classifier dropped in by mistake -- would be decoded into
        garbage boxes rather than an error, so check once at load time."""
        in_shape = self.session.get_inputs()[0].shape
        if len(in_shape) != 4:
            raise ValueError(f"{onnx_path}: expected a 4-D image input, got shape {in_shape}")
        if isinstance(in_shape[2], int) and in_shape[2] != self.input_size:
            raise ValueError(
                f"{onnx_path} has a fixed input size of {in_shape[2]}px but species.json's "
                f"detector_input_size is {self.input_size}.")
        out_shape = self.session.get_outputs()[0].shape
        if len(out_shape) != 3 or (isinstance(out_shape[1], int) and out_shape[1] < 5):
            raise ValueError(
                f"{onnx_path}: expected YOLOv8 output (1, 4+num_classes, anchors), got {out_shape}")
        self.num_classes = out_shape[1] - 4 if isinstance(out_shape[1], int) else None

    def detect(self, img: Image.Image) -> list[RawDetection]:
        if img.size[0] < 1 or img.size[1] < 1:
            return []
        canvas, scale, pad = imageops.letterbox(img.convert("RGB"), self.input_size)
        tensor = imageops.to_chw_float32(canvas)[np.newaxis, ...]  # add batch dim

        raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
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
        w, h = img.size
        for i in keep_idx:
            box = imageops.unletterbox_box(boxes_letterboxed[i], scale, pad)
            # Anchors near the letterbox border routinely predict boxes that
            # spill past the image edge; clamp so downstream crop() never
            # gets negative or out-of-range coordinates (PIL would pad those
            # with black, feeding the classifier a border it never trained on).
            box = imageops.clamp_box(box, w, h)
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
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
