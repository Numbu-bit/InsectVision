# Python runtime modules

These modules are imported by `app.py`. They are designed for CPU inference with ONNX Runtime, Pillow, and NumPy, so the deployed service does not need PyTorch, OpenCV, or Ultralytics.

| File | What it does |
|---|---|
| `__init__.py` | Marks `src` as a Python package. It intentionally contains no runtime logic. |
| `config.py` | Defines project-root paths, the species configuration path, model/static directories, the 10 MB upload limit, and path resolution for model paths stored in `species.json`. |
| `imageops.py` | Decodes and normalises images, corrects phone-photo EXIF rotation, scores blur/luminance/resolution quality, letterboxes images for YOLO, maps boxes back to the original image, and converts images to ONNX CHW float tensors. |
| `classifier_onnx.py` | Loads an EfficientNet-B0 ONNX model once and classifies an image. It applies the same resize/centre-crop and ImageNet normalisation used during validation, then returns top-k softmax predictions. |
| `detector_onnx.py` | Loads a YOLOv8 ONNX model and detects objects. It performs letterboxing, output conversion from centre-width-height boxes to corner boxes, confidence filtering, and NumPy non-maximum suppression. |
| `decision_tree.py` | Defines pest/beneficial/neutral status values, the fixed confidence threshold of `0.75`, the reject class name (`REJECT_CLASS = "other"`) and the three-way `Verdict` (`confirmed` / `uncertain` / `no_specimen`). `Identification.verdict` applies the reject-class check first, then the threshold. |

## How the modules work together

`app.py` calls `imageops.load_image()` and `imageops.assess_quality()` first. If the image passes, it calls either `YoloOnnxDetector.detect()` plus `ClassifierOnnx.classify()` in cascade mode, or `ClassifierOnnx.classify()` directly in classifier-only mode. `decision_tree.CONFIDENCE_THRESHOLD` controls the uncertainty flag returned to the UI.

## Important runtime assumptions

- The classifier class order must match `config/species.json`.
- Model files must be valid ONNX files and use the input sizes in `species.json` (`224` for the classifier and `640` for the detector by default).
- Detector boxes are returned in original-image coordinates after undoing letterboxing.
- The quality gate rejects very small, dark, over-exposed, or blurry photos before model inference. Blur and luminance are measured on a copy downscaled to 512 px on the shorter side (Lanczos): the Laplacian-variance blur score is strongly resolution-dependent (a sharp 12 MP phone photo scored 1.7 at native size against a threshold of 100 and was rejected as blurry), and full-resolution float arrays peaked at ~384 MB on a 512 MB serve tier.
- `ClassifierOnnx._preprocess` replicates torchvision's `Resize(int(224*1.14)) -> CenterCrop(224)` bit-for-bit, including the truncated long side and the `round(margin/2)` crop offset. The training notebook's export cell runs this class on real validation images and fails the export if it disagrees with the torch pipeline.
- `ClassifierOnnx` and `YoloOnnxDetector` validate the ONNX input/output shapes against `species.json` at load and raise `ValueError` on a mismatch; `app.py` turns that into a 503 rather than serving wrong labels.
