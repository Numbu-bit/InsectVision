# InsectVision application

This folder contains the runnable application, its model-runtime code, the browser interface, configuration, and training workflow.

## Main files

| File or folder | Purpose |
|---|---|
| `app.py` | FastAPI server, API endpoints, model selection, and static UI hosting. |
| `src/` | Python modules used by the server for configuration, image quality, classification, detection, and confidence rules. See `src/README.md`. |
| `scripts/` | Dataset preparation utilities. See `scripts/README.md`. |
| `config/species.json` | Model paths, supported class names, pest/beneficial/neutral labels, species descriptions, and stored evaluation metrics. See `config/README.md`. |
| `static/index.html` | Browser page structure. |
| `static/app.js` | Browser interactions, image selection/cropping, API calls, and result rendering. |
| `static/style.css` | Mobile-first visual styling and accessibility states. See `static/README.md`. |
| `notebooks/train_cascade_colab.ipynb` | Colab workflow for downloading data, training the classifier and optional detector, evaluating them, and exporting ONNX models. See `notebooks/README.md`. |
| `models/*.onnx` | Runtime model files loaded by the app. They are not source scripts. |
| `requirements.txt` | Full local/training/test dependency set. |
| `requirements-deploy.txt` | Smaller CPU-only dependency set used when serving the app. |
| `render.yaml` | Render deployment instructions. |

## Run locally

From this directory:

```powershell
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. The health endpoint is `http://127.0.0.1:8000/api/v1/health`.

The app uses classifier-only mode when the classifier file named in `config/species.json` (`classifier_onnx`, currently `models/classifier_13cls.onnx`) exists without a detector. It uses cascade mode when both ONNX files exist. If the classifier is missing, the health endpoint reports `not_configured` and image analysis is unavailable. At load time the app checks that the ONNX file's output width equals `len(class_names)` and refuses to serve a mismatched pair (for example the old 12-class `classifier.onnx` against the 13-entry class list).

## The `other` class and verdicts

The classifier has 13 outputs: 12 insect taxa plus `other`, a reject class trained on non-insect images (leaves, soil, hands, diagrams...). Without it, softmax is a forced choice between insects and a photo of a tree comes back as "beetle 56%". Every prediction resolves to one of three verdicts (`src/decision_tree.py`):

| Verdict | Meaning | Counted? |
|---|---|---|
| `confirmed` | a real taxon at or above the 0.75 threshold | yes |
| `uncertain` | a real taxon below the threshold; shown as needing review | no |
| `no_specimen` | the top class is `other`: no recognisable insect here. A definite answer, not a low-confidence one | no |

The reject-class check runs before the threshold, so `other` at 90% is "no insect", never "flagged other". `other` has no pest/beneficial status, never appears in the supported-species list or a tally, and the UI renders a neutral "No known insect recognised" state for it.

## Request flow

1. The browser loads the static page and requests `/api/v1/health`.
2. The user uploads an image; the browser checks its type and size and can crop it.
3. `/api/v1/analyse` decodes the image and applies the quality gate.
4. In cascade mode, YOLO detects regions and the classifier labels each region.
5. In classifier-only mode, the classifier labels the complete image and returns the top three predictions.
6. The response includes confidence values, uncertainty flags, quality information, and latency.
