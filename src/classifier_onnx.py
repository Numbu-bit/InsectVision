"""
EfficientNet-B0 ONNX inference using ONNX Runtime only -- no torch,
matching detector_onnx.py's reasoning: torch alone is too large for
Render's free-tier serve process.

Preprocessing here MUST stay byte-for-byte equivalent to the *validation*
transform in notebooks/train_cascade_colab.ipynb:
    Resize(int(224 * 1.14)) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)
The notebook's export cell runs this very class against the torch model on
real validation images and fails the export if predictions disagree, so a
drift between the two code paths is caught at training time, not in
production.
"""
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# torchvision's Resize(int(size * 1.14)) followed by CenterCrop(size) -- the
# standard ImageNet eval framing (256/224 = 1.143). Kept as one named
# constant so the ratio can't drift between the two code paths.
RESIZE_RATIO = 1.14


@dataclass
class ClassResult:
    taxon: str
    confidence: float


class ClassifierOnnx:
    def __init__(self, onnx_path: str, class_names: list[str], input_size: int = 224):
        if len(class_names) != len(set(class_names)):
            raise ValueError("class_names contains duplicates")
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.class_names = list(class_names)
        self.input_size = input_size
        self._validate_model_shape(onnx_path)

    def _validate_model_shape(self, onnx_path: str) -> None:
        """Refuse to serve a model whose output width doesn't match the class
        list. species.json says 13 classes and the ONNX file emits 12 (or vice
        versa) is exactly the failure mode of swapping a model without its
        config -- and without this check it doesn't crash, it silently
        returns the wrong species name for every prediction.

        Shapes from get_outputs() can carry symbolic dims (e.g. 'batch'), so
        only the last, concrete dim is checked statically; a real forward
        pass on a blank image is then used as the authoritative check, which
        also confirms the runtime can actually execute the graph."""
        out_shape = self.session.get_outputs()[0].shape
        if out_shape and isinstance(out_shape[-1], int) and out_shape[-1] != len(self.class_names):
            raise ValueError(
                f"{onnx_path} outputs {out_shape[-1]} classes but species.json lists "
                f"{len(self.class_names)}. The model file and config/species.json are out "
                f"of sync -- did you drop in a new classifier without its species.json "
                f"(or vice versa)?")

        in_shape = self.session.get_inputs()[0].shape
        if len(in_shape) == 4 and isinstance(in_shape[1], int) and in_shape[1] != 3:
            raise ValueError(f"{onnx_path} expects {in_shape[1]} input channels, not RGB (3).")
        if (len(in_shape) == 4 and isinstance(in_shape[2], int) and in_shape[2] != self.input_size):
            raise ValueError(
                f"{onnx_path} has a fixed input size of {in_shape[2]}px but species.json's "
                f"classifier_input_size is {self.input_size}.")

        probe = self.logits(Image.new("RGB", (self.input_size, self.input_size)))
        if probe.shape != (len(self.class_names),):
            raise ValueError(
                f"{onnx_path} produced {probe.shape[0]} logits but species.json lists "
                f"{len(self.class_names)} classes -- model file and config are out of sync.")

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        # Convert to RGB *before* resizing: PIL silently falls back to NEAREST
        # interpolation for palette ('P') images, which would change the
        # pixels the model sees for e.g. an indexed-colour PNG.
        img = img.convert("RGB")
        target = self.input_size
        resize_to = int(target * RESIZE_RATIO)
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError("Cannot classify an empty image.")

        # Replicates torchvision.transforms.Resize(int) + CenterCrop(int)
        # EXACTLY, including two easy-to-miss integer conventions:
        #   * the long side is int()-truncated, not rounded
        #     (torchvision: new_long = int(new_short * long / short));
        #   * the crop offset is int(round(margin / 2.0)), not margin // 2 --
        #     for the standard 255->224 crop the margin is 31, so this is
        #     16 vs 15: a one-pixel shift on every single image.
        # An earlier version used round() and // here; the notebook's
        # serve-path parity check flagged the resulting pixel mismatch.
        short, long = (w, h) if w <= h else (h, w)
        new_short, new_long = resize_to, int(resize_to * long / short)
        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
        if (new_w, new_h) != (w, h):  # torchvision skips the resize when already the right size
            img = img.resize((max(1, new_w), max(1, new_h)), Image.BILINEAR)
        w, h = img.size
        left = int(round((w - target) / 2.0))
        top = int(round((h - target) / 2.0))
        img = img.crop((left, top, left + target, top + target))

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(arr, (2, 0, 1))
        return np.ascontiguousarray(chw[None], dtype=np.float32)

    def logits(self, img: Image.Image) -> np.ndarray:
        tensor = self._preprocess(img)
        out = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def probabilities(self, img: Image.Image) -> np.ndarray:
        """Full softmax vector, index-aligned with class_names."""
        logits = self.logits(img)
        shifted = logits - logits.max()  # numerically stable softmax
        exp = np.exp(shifted)
        return exp / exp.sum()

    def classify(self, img: Image.Image, top_k: int = 3) -> list[ClassResult]:
        probs = self.probabilities(img)
        top_k = max(1, min(top_k, len(self.class_names)))
        order = probs.argsort()[::-1][:top_k]
        return [ClassResult(self.class_names[i], float(probs[i])) for i in order]
