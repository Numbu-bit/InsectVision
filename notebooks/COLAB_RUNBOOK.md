# Colab runbook -- training the 13-class classifier

End-to-end steps for `train_cascade_colab.ipynb`. Only the **Kaggle API token** is
needed: it downloads both the insect dataset and the non-insect ("other")
images. The detector is **not** retrained -- the existing `models/detector.onnx`
(mAP@0.50 = 0.964) stays as it is.

Expect **35-50 minutes** wall-clock on a free T4: ~10 min of downloads,
~15-25 min of training, a few minutes for evaluation and export.

---

## Part A -- open the notebook (10 seconds)

Paste this into your browser:

**<https://colab.research.google.com/github/Numbu-bit/InsectVision/blob/retrain-13class/notebooks/train_cascade_colab.ipynb>**

The notebook pulls the project code straight from GitHub (section 3,
`CODE_SOURCE = "github"`), so there is nothing to zip or upload -- the branch
`retrain-13class` already has everything it needs. (The zip-upload route still
exists as `CODE_SOURCE = "upload"` if you ever need it.)

---

## Part B -- in Colab

### B1. Open the notebook and get a GPU

1. Open the link from Part A. Colab shows the notebook read-only from GitHub;
   the first time you run a cell it asks to save a copy to your Drive
   (**File -> Save a copy in Drive**) -- accept, so your run is saved.
2. **Runtime -> Change runtime type -> Hardware accelerator: T4 GPU -> Save.**
3. **Runtime -> Restart session** (needed whenever you change the accelerator).

### B2. Run the setup cells, top to bottom

Run each cell with **Shift+Enter** and wait for it to finish before the next.

| Section | Cell | What to do / what to expect |
|---|---|---|
| 1. GPU check | run | Prints `GPU OK: Tesla T4`. If it raises, redo B1 step 2-3. |
| 1b. Google Drive | run **or** set `USE_DRIVE = False` first | Only useful for detector training (which you are skipping). Set `USE_DRIVE = False` to avoid the Drive permission prompt. |
| 2. Install dependencies | run | ~1-2 min. Warnings about pip dependency resolution are normal. |
| 3. Get project code | run | Clones the GitHub repo and prints `Checked out commit ...` then `Loaded 13 classes:` including `other`. If it says 12, the branch on GitHub is behind -- push first. |
| 4. Kaggle authentication | run, paste your token | Get it at kaggle.com -> Settings -> API -> **Create New Token** if you don't have it to hand. The input is hidden. |

### B3. Data -- insects from Kaggle, negatives from Kaggle

**Section 5, first cell (negatives).** Leave the defaults:

```python
NEGATIVES_SOURCE = "kaggle"
```

It downloads three public datasets with your token (outdoor scenes, everyday
objects/people/flowers/fruit, plant leaves), removes anything with an insect
word in its folder name, and samples ~350 from each -> ~1,000 negatives. The
PlantVillage download is the slow one (~2 GB); the whole cell takes 5-10 min.

Expected last line: `~1000 negative images ready under /content/data/negatives`.

- If one dataset fails to download, the cell prints Kaggle's message and
  **skips it**; as long as the total is >= 300 you can continue. If it is
  below 300, add another slug to `KAGGLE_NEGATIVE_SETS` or switch to
  `NEGATIVES_SOURCE = "upload"` and upload a `negatives.zip` of your own.

**Section 5, second cell (insects + prepare).** Run it. It downloads the
agricultural-pests dataset, runs `scripts/prepare_data.py --negatives ...`,
and rewrites `species.json`. Check the printout:

```
13 classes: ['ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig',
             'grasshopper', 'moth', 'other', 'slug', 'snail', 'wasp', 'weevil']
reject_class = 'other'
```

`other` must be at **index 8** (alphabetical, in the middle). The split
summary shows `other` with ~680 train / ~120 val images.

### B4. Train (section 6) -- 15-25 minutes

Run the training cell. First it prints the class table and the per-class loss
weights (all insects ~1.0-1.2, `other` ~0.4-0.5 because it has more images --
that is the inverse-frequency weighting doing its job). Then one line per
epoch:

```
epoch   1/30  train_loss=1.4321  val_macro_f1=0.8012
...
```

Early stopping usually ends it around epoch 18-25. Leave the tab open; Colab
disconnects idle tabs.

### B5. Evaluate and read the gates (section 6, next cell)

Run it. Read three things:

1. **`Validation macro-F1 (insects only, vs 0.944 baseline)`** -- the
   like-for-like number. 0.90-0.95 is what you want.
2. **`'other' recall=...`** -- the fraction of non-insect photos correctly
   rejected. Wants >= 0.85. The confusion matrix's `other` **row** shows what
   the leaks were mistaken for; the `other` **column** shows real insects
   wrongly rejected.
3. The final block: `OK: all export gates passed.` or a `WARNING: EXPORT
   GATES FAILED` list. Gates: any class F1 < 0.80, `other` recall < 0.85,
   insect-only macro-F1 < 0.90.

If a gate fails: `other` recall low -> set `OTHER_WEIGHT_BOOST = 1.5` in the
training cell and re-run B4-B5; an insect class collapsed -> look at its
confusion-matrix row, it is usually a look-alike pair (weevil/beetle) and a
few more epochs help (`patience = 12`). `EXPORT_DESPITE_WARNINGS = True` in
the export cell overrides the block -- only do that knowingly.

### B6. Skip the detector (sections 7-8)

Run the first cell of section 7 as-is (`TRAIN_DETECTOR = False`). It prints
`Detector training is OFF -- skipping sections 7-8.` **Do not run** the
Roboflow download / YOLO training / resume cells -- just scroll past them to
section 9. (If you run them anyway they are no-ops while `TRAIN_DETECTOR` is
False.)

### B7. Export and verify (section 9)

Run the helpers cell, then the **classifier export** cell. Expected output:

```
BatchNorm channels: 21008, running_var min=..., collapsed(<1e-6)=0
classifier: torch vs onnx on 8 real val images -- max |dprob| = 2e-07, argmax agree = True
ONNX output width 13 == len(class_names) OK
serve-path parity on 96 val images: argmax agreement=1.000, max |dprob|=0.0000
Serve-path parity OK -- src/classifier_onnx.py reproduces the training-time preprocessing.
```

Any assertion here means **do not deploy** -- the message says which of the
three checks failed. Then run the **detector export** cell; it prints that
detector training was skipped, which is expected.

### B8. Write species.json and download (sections 10-11)

Run both cells. The download cell hands you **two files**:

- `classifier_13cls.onnx`
- `species.json`

(no `detector.onnx`, because it wasn't retrained.)

---

## Part C -- back on your computer (5 minutes)

1. Copy the files in:
   - `classifier_13cls.onnx` -> `insectvision/models/`
   - `species.json` -> `insectvision/config/` (overwrite)
2. Optional but tidy: delete the old `insectvision/models/classifier.onnx`.
   The app no longer references it and would refuse to load it against the
   13-class list anyway.
3. Run the tests from inside `insectvision/`:

   ```powershell
   python -m pytest tests -q
   ```

4. Start the app and test it:

   ```powershell
   uvicorn app:app --reload
   ```

   Open <http://127.0.0.1:8000>. The header badge should read
   **Auto-counting** and **12 species · F1 0.9x**.
   - Upload the tree photo -> "No known insect recognised".
   - Upload `beetle.jpg`, `ant.jpg` etc. from the project folder -> green
     "Identified" banner at ~90%.
   - Click **Use Webcam** -> allow the camera -> Capture -> Analyse.
     (Webcam works on `localhost` and on the https Render URL; browsers block
     it on plain `http://` over a LAN.)
5. Commit the two files on `retrain-13class`, then merge to `main` and push
   -> Render redeploys with code and model together:

   ```powershell
   git add models/classifier_13cls.onnx config/species.json
   git commit -m "Add trained 13-class classifier"
   git checkout main
   git merge retrain-13class
   git push origin main
   ```

---

## If something goes wrong

| Symptom | Fix |
|---|---|
| `Loaded 12 classes` in section 3 | The `retrain-13class` branch on GitHub is behind your local code. Commit and push, then re-run the cell. |
| Kaggle `403`/`404` on a negatives dataset | That slug moved or needs you to accept its terms on the Kaggle page. Open the dataset on kaggle.com once, or swap the slug. The cell skips it and continues. |
| `Only N negative images (need >= 300)` | Add a dataset slug, or switch to `"upload"`. |
| `Class order mismatch` assertion | `data/classify/*/` folders and `species.json` disagree -- re-run both section-5 cells. |
| Disconnected during training | Re-run from the top; classifier training is short enough that there is no resume logic. `USE_DRIVE` only matters for the detector. |
| App says `not_configured` after copying files | `species.json` names `models/classifier_13cls.onnx` -- check the file is in `models/` with exactly that name. |
| App says `Classifier unavailable: ... out of sync` | The ONNX file and `species.json` are from different runs. Copy both from the same Colab session. |
