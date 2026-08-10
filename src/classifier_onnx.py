"""
EfficientNet-B0 ONNX inference using ONNX Runtime only -- no torch,
matching detector_onnx.py's reasoning: torch alone is too large for
Render's free-tier serve process.
"""
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class ClassResult:
    taxon: str
    confidence: float


class ClassifierOnnx:
    def __init__(self, onnx_path: str, class_names: list[str], input_size: int = 224):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = class_names
        self.input_size = input_size

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        # Resize-then-centre-crop, matching the *validation* transform used
        # during training (Resize(1.14x) -> CenterCrop) in the Colab
        # notebook -- inference needs to see the same framing the model was
        # evaluated on, not the training-time random-crop augmentation.
        target = self.input_size
        resize_to = int(target * 1.14)
        w, h = img.size
        scale = resize_to / min(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        w, h = img.size
        left, top = (w - target) // 2, (h - target) // 2
        img = img.crop((left, top, left + target, top + target))

        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(arr, (2, 0, 1))
        return chw[None].astype(np.float32)

    def classify(self, img: Image.Image, top_k: int = 3) -> list[ClassResult]:
        tensor = self._preprocess(img)
        logits = self.session.run(None, {self.input_name: tensor})[0][0]

        shifted = logits - logits.max()  # numerically stable softmax
        exp = np.exp(shifted)
        probs = exp / exp.sum()

        order = probs.argsort()[::-1][:top_k]
        return [ClassResult(self.class_names[i], float(probs[i])) for i in order]
