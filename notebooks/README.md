# Training notebook

## `train_cascade_colab.ipynb`

This is the end-to-end training and export workflow for Google Colab with a GPU. It trains the two independent stages used by the application:

1. **Classifier:** EfficientNet-B0 trained on the Kaggle folder-per-class dataset **plus a 13th `other` class of non-insect images** supplied as `negatives.zip` (or a Drive folder) in section 5. Class-weighted cross-entropy (inverse frequency) with label smoothing 0.1; model selection on macro-F1. Produces `classifier_13cls.onnx` (opset 13, dynamic batch, input `input`, output `output`).
2. **Detector:** YOLOv8n trained on the Roboflow YOLO-format bounding-box dataset. This stage is optional and produces `detector.onnx`. Skipping it leaves the app in classifier-only mode.

## Main stages

- Check that a Colab GPU is available.
- Optionally mount Google Drive for checkpoints and recovery after disconnection.
- Install training/export dependencies.
- Upload the reviewed `src/`, `scripts/`, and `config/` project files.
- Authenticate with Kaggle and download/prepare the classifier data by running the real `scripts/prepare_data.py`.
- Train and validate the classifier, including macro-F1, a per-class classification report and a confusion matrix.
- Gate the export: any class under 0.80 F1, `other` recall under 0.85, or insect-only macro-F1 under 0.90 blocks the export unless `EXPORT_DESPITE_WARNINGS = True`.
- Verify the export three ways: torch vs ONNX Runtime on real validation images (softmax drift < 1e-4), output width == `len(class_names)`, and serve-path parity (the app's own `src/classifier_onnx.py` must agree with the torch `val_tf` pipeline on real images).
- Download the Roboflow detector data when detector training is enabled.
- Train and validate the detector, including mAP@0.50.
- Export both models to single-file ONNX and verify their outputs.
- Download the resulting ONNX files and updated `species.json` for the application.

The notebook also contains recovery instructions for resuming detector training and re-exporting a detector from a saved Drive checkpoint.

## How to describe the results

The notebook's recorded metrics are validation metrics:

- 12-class classifier (superseded) validation macro-F1: `0.9441` (approximately `94.41%`), calculated from `data/classify/val/`. The 13-class model's metrics are written into `species.json` by the notebook (`classifier_macro_f1`, `classifier_specimen_macro_f1` for the like-for-like insect-only comparison, `classifier_reject_recall`).
- Detector validation mAP@0.50: `0.9636` (approximately `96.36%`), calculated by the YOLO validation call.

The classifier has no independent `data/classify/test/` split, so its result must not be described as test accuracy. The detector dataset does contain a `test/` folder, but the stored detector metric was produced by validation, not by an explicit test-split evaluation.

The project also has no automated software tests yet; `insectvision/tests/` is currently empty. Manual image uploads test the application workflow but do not replace a held-out model test set.

## Outputs used by the app

Copy these results into the local project after training:

```text
insectvision/models/classifier_13cls.onnx
insectvision/models/detector.onnx       # optional
insectvision/config/species.json
```

The serve-time app does not run the notebook. It only loads the exported ONNX files through the modules in `src/`.
