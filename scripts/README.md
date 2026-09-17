# Dataset scripts

## `prepare_data.py`

```powershell
python scripts/prepare_data.py --source data/raw --negatives data/negatives
```

`--negatives` is a folder (any nesting) of images with **no insect** in them -- leaves, soil, bark, hands, tools, diagrams. It becomes the classifier's reject class `other` (capped at `--max-negatives`, default 2x `--max-per-class`). Without it the script warns loudly and produces the old closed-set 12-class layout, in which a photo of a tree is forced onto the nearest insect class. Byte-identical duplicate images are dropped before the train/val split (4 groups were found in the original Kaggle data, one straddling train and val). The script also writes `reject_class`, `species_info.other` (`is_specimen: false`) and a class-count-named `classifier_onnx` path into `species.json`.

Prepares the Kaggle agricultural-pests image dataset for classifier training. It handles the classifier dataset only; it does not prepare the Roboflow YOLO detector dataset.

### What it does

1. Finds the folder-per-class directory, including common wrapper directories inside downloaded archives.
2. Canonicalises class names, for example `Field Crickets` becomes `field_crickets`.
3. Prints every discovered class and its image count.
4. Warns about classes below `--min-images`.
5. Randomly shuffles each class with a reproducible seed.
6. Copies up to `--max-per-class` images into `data/classify/train/<class>` and `data/classify/val/<class>`.
7. Writes the discovered class order and guessed taxon statuses to `config/species.json`.
8. Prints classes guessed as beneficial or neutral so they can be reviewed manually.

The status guesses are only starting points. Review `config/species.json` before training because a wrong status can suppress an alert in the application.

### Usage from the `insectvision` directory

```powershell
python scripts/prepare_data.py --source data/raw
```

The source directory must already contain the downloaded and extracted Kaggle dataset. To download it with the Kaggle CLI:

```powershell
kaggle datasets download -d vencerlanz09/agricultural-pests-image-dataset -p data/raw --unzip
```

### Useful options

```text
--source          Input dataset root; default data/raw
--out             Prepared output; default data/classify
--species-config  JSON file to update; default config/species.json
--max-per-class   Maximum source images per class; default 400
--val-fraction    Validation fraction; default 0.15
--min-images      Warning threshold; default 20
--seed            Reproducible split seed; default 42
```

The script copies files rather than deleting the source. Re-running it can leave old class files in the output directory if the source classes change, so use a clean output directory when changing datasets.

The detector data is already stored in `FAIR-D_V2.5.v2i.yolov8/` and is downloaded/prepared by the training notebook rather than this script.
