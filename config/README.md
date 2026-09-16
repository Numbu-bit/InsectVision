# Configuration

## `species.json`

This JSON file is the shared contract between training, the backend, and the browser UI.

| Property | Purpose |
|---|---|
| `detector_onnx` | Relative or absolute path to the YOLO detector model. |
| `classifier_onnx` | Relative or absolute path to the image classifier model. |
| `detector_input_size` | Detector square input size, normally `640`. |
| `classifier_input_size` | Classifier crop size, normally `224`. |
| `class_names` | Alphabetically ordered class names returned by the classifier. This order must match the model output indexes. Includes the reject class `other` (index 8 in the standard 13-class list -- it sits in the middle, not at the end). |
| `reject_class` | Name of the "not an insect" class (`"other"`), or `null` for a legacy closed-set model. Must be one of `class_names`. Excluded from `taxon_status`. |
| `detector_conf_threshold` | Minimum detector score for a region to be proposed (default `0.25`). Kept low deliberately: the detector is tuned for recall and the classifier's reject class decides whether a proposal is really an insect. |
| `classifier_specimen_macro_f1`, `classifier_per_class_f1`, `classifier_reject_recall` | Extra evaluation metrics written by the notebook: insect-only macro-F1 (comparable to the old 12-class 0.944), per-class F1, and recall of the reject class. |
| `taxon_status` | Maps each class to `pest`, `beneficial`, or `neutral`. |
| `species_info` | Display metadata such as common name, scientific name, and damage symptoms. |
| `classifier_macro_f1` | Recorded classifier evaluation metric, if training has been evaluated. |
| `detector_map50` | Recorded detector mAP@0.50 metric, if detector training has been evaluated. |

`app.py` reads this file for the health response, model paths, supported species list, status labels, and stored metrics. `prepare_data.py` updates the class list, adds missing status/species entries, sets `reject_class` when a `--negatives` folder was supplied, writes `species_info.other` with `"is_specimen": false`, and names `classifier_onnx` by class count (`models/classifier_13cls.onnx`).

Edit the status and species information deliberately. The automatically guessed values from `prepare_data.py` must be reviewed by a human before deployment.
