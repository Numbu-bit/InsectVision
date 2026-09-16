"""
Turn a downloaded Kaggle folder-per-class insect image dataset into the
train/val layout the classifier trainer (Step 3, Colab) expects, and write
config/species.json with a pest/beneficial guess for each discovered class.

This handles the CLASSIFIER data only. The detector needs bounding-box
annotations, which folder-per-class datasets don't have -- detector data
comes from a Roboflow YOLO-format export instead, downloaded and prepared
directly inside the Colab notebook (Step 3), not here.

Usage:
    kaggle datasets download -d vencerlanz09/agricultural-pests-image-dataset \\
        -p data/raw --unzip
    python scripts/prepare_data.py --source data/raw --negatives data/negatives

--negatives is a folder (flat or nested, any depth) of images that contain
NO insect: leaves, bark, soil, sky, hands, tools, walls, diagrams, blurry
nothing. They become the classifier's reject class, "other" (see
src/decision_tree.REJECT_CLASS). Without it the classifier is closed-set:
a softmax over only insect classes is *forced* to call a photo of a tree
some insect, which is how "beetle 56%" on a tree happens. The script warns
loudly, but still runs, if the folder is missing -- so an old-style
12-class model can still be reproduced deliberately.

This script does not call the Kaggle API itself -- getting the raw dataset
onto disk (via the `kaggle` CLI above, or a manual download-and-unzip from
the Kaggle website) is a separate, one-time step outside its scope. This
script only *processes* whatever is on disk at --source: default is
data/raw, matching the command above, so the two are meant to be run back to
back.
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.decision_tree import REJECT_CLASS, TaxonStatus  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Keyword -> status. Checked against the canonicalised class name as a
# substring match. Order matters: more specific phrases first, so e.g.
# "parasitic wasp" hits the beneficial rule before a bare "wasp" would.
# This is a starting guess, not a judgement to trust blindly -- every class
# it marks beneficial (or leaves ambiguous) is printed for you to check,
# because getting this wrong has real consequences: a genuine pest wrongly
# marked beneficial would silently suppress every alert for it.
STATUS_RULES: list[tuple[str, "TaxonStatus"]] = [
    ("parasitic wasp", TaxonStatus.BENEFICIAL),
    ("parasitoid", TaxonStatus.BENEFICIAL),
    ("ladybird", TaxonStatus.BENEFICIAL),
    ("ladybug", TaxonStatus.BENEFICIAL),
    ("lady beetle", TaxonStatus.BENEFICIAL),
    ("honeybee", TaxonStatus.BENEFICIAL),
    ("honey bee", TaxonStatus.BENEFICIAL),
    ("bee", TaxonStatus.BENEFICIAL),
    ("earthworm", TaxonStatus.BENEFICIAL),
    ("mantis", TaxonStatus.BENEFICIAL),
    ("dragonfly", TaxonStatus.BENEFICIAL),
    ("spider", TaxonStatus.BENEFICIAL),
    ("wasp", TaxonStatus.NEUTRAL),   # ambiguous: many wasps sting but aren't crop pests
    ("ant", TaxonStatus.NEUTRAL),    # ambiguous: farming/aphid-tending ants vs harmless
]


def canonicalise(name: str) -> str:
    """'Field Crickets' -> 'field_crickets'. Folder names in Kaggle exports
    are inconsistently capitalised (and sometimes pluralised); the rest of
    the system only ever sees this canonical key, never the raw folder name."""
    cleaned = "".join(c if c.isalnum() else "_" for c in name.strip().lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _word_matches(word: str, kw_word: str) -> bool:
    if word == kw_word:
        return True
    # Simple plural tolerance ("bees" ~ "bee", "beetles" ~ "beetle") without
    # unconditionally stripping a trailing "s" from every word -- that would
    # mangle words that are singular but happen to end in s, e.g. "mantis"
    # would become "manti" and stop matching its own keyword.
    if word.endswith("s") and word[:-1] == kw_word:
        return True
    if kw_word.endswith("s") and kw_word[:-1] == word:
        return True
    return False


def guess_status(taxon: str) -> TaxonStatus:
    # Whole-word matching, not substring: a naive `"bee" in "beetle"` check
    # is True (b-e-e-tle) and would wrongly mark a beetle class beneficial.
    # Match keyword phrases only against contiguous whole words instead.
    words = taxon.replace("_", " ").split()
    for keyword, status in STATUS_RULES:
        kw_words = keyword.split()
        n = len(kw_words)
        for i in range(len(words) - n + 1):
            if all(_word_matches(words[i + j], kw_words[j]) for j in range(n)):
                return status
    return TaxonStatus.PEST


def _has_images(d: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in d.iterdir())


def _looks_like_class_root(d: Path) -> bool:
    subdirs = [p for p in d.iterdir() if p.is_dir()]
    if len(subdirs) < 2:
        return False
    with_images = sum(1 for s in subdirs if _has_images(s))
    return with_images >= max(2, round(0.6 * len(subdirs)))


def find_class_root(root: Path) -> Path:
    """Kaggle exports often wrap the real class folders in one or two extra
    directories (the zip's own name, sometimes repeated). Descend through
    single-child wrapper directories until the folder-per-class layout is
    found, rather than assuming --source already points at it exactly."""
    current = root
    for _ in range(6):
        if _looks_like_class_root(current):
            return current
        subdirs = [p for p in current.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            current = subdirs[0]
            continue
        break
    raise RuntimeError(
        f"Could not find a folder-per-class layout under {root} "
        f"(looked {6} levels deep). Point --source directly at the "
        "directory whose immediate subdirectories are the class folders."
    )


def discover_classes(class_root: Path) -> dict[str, list[Path]]:
    classes: dict[str, list[Path]] = {}
    for d in sorted(p for p in class_root.iterdir() if p.is_dir()):
        images = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not images:
            continue
        classes[canonicalise(d.name)] = images
    return classes


def discover_negatives(negatives_root: Path) -> list[Path]:
    """Every image under --negatives, at any depth. Unlike the Kaggle
    classes, negatives are usually gathered from several sources into
    sub-folders (leaves/, soil/, hands/ ...) -- all of it is one class."""
    return sorted(p for p in negatives_root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _content_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def dedupe_exact(images: list[Path]) -> tuple[list[Path], int]:
    """Drop byte-identical duplicates. Kaggle scrapes commonly contain the
    same file under two names; if both copies land on opposite sides of the
    train/val split, the validation score is inflated for free. Exact-hash
    only -- near-duplicates (re-encoded/resized copies) would need
    perceptual hashing, which is out of scope here.
    # REVIEW: consider a pHash pass if val macro-F1 looks too good to be true."""
    seen: set[str] = set()
    kept: list[Path] = []
    for p in images:
        h = _content_hash(p)
        if h in seen:
            continue
        seen.add(h)
        kept.append(p)
    return kept, len(images) - len(kept)


def split_and_copy(classes: dict[str, list[Path]], out_dir: Path,
                   max_per_class: int, val_fraction: float, seed: int) -> dict[str, dict]:
    rng = random.Random(seed)
    summary: dict[str, dict] = {}

    for taxon, images in classes.items():
        pool, n_dupes = dedupe_exact(images)
        rng.shuffle(pool)
        pool = pool[:max_per_class]

        n_val = max(1, round(len(pool) * val_fraction)) if len(pool) > 1 else 0
        val_files, train_files = pool[:n_val], pool[n_val:]

        for split, files in (("train", train_files), ("val", val_files)):
            split_dir = out_dir / split / taxon
            split_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                shutil.copy2(src, split_dir / src.name)

        summary[taxon] = {
            "available": len(images),
            "duplicates_dropped": n_dupes,
            "used": len(pool),
            "train": len(train_files),
            "val": len(val_files),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/raw",
                    help="Downloaded+unzipped Kaggle dataset root (default: data/raw)")
    ap.add_argument("--out", default="data/classify",
                    help="Output train/val directory (default: data/classify)")
    ap.add_argument("--negatives", default="data/negatives",
                    help=f"Folder of NON-insect images that become the '{REJECT_CLASS}' "
                         "reject class (default: data/negatives). Searched recursively.")
    ap.add_argument("--max-negatives", type=int, default=None,
                    help=f"Cap on '{REJECT_CLASS}' images (default: 2x --max-per-class). "
                         "Negatives are far more visually diverse than any one insect "
                         "class, so they warrant more examples -- but not so many they "
                         "swamp training; the class-weighted loss in the notebook "
                         "rebalances whatever is left.")
    ap.add_argument("--species-config", default="config/species.json")
    ap.add_argument("--max-per-class", type=int, default=400)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--min-images", type=int, default=20,
                    help="Warn (not fail) if a class has fewer images than this")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    source = (ROOT / args.source) if not Path(args.source).is_absolute() else Path(args.source)
    if not source.exists():
        print(f"ERROR: --source {source} does not exist.")
        print("Download the dataset first, e.g.:")
        print("  kaggle datasets download -d vencerlanz09/agricultural-pests-image-dataset "
              f"-p {args.source} --unzip")
        raise SystemExit(1)

    class_root = find_class_root(source)
    print(f"Found folder-per-class layout at: {class_root}")

    classes = discover_classes(class_root)
    if not classes:
        raise SystemExit(f"No class folders with images found under {class_root}")
    if REJECT_CLASS in classes:
        raise SystemExit(
            f"The source dataset already has a class named '{REJECT_CLASS}', which is "
            "reserved for the reject class. Rename that folder or pick a different "
            "REJECT_CLASS in src/decision_tree.py.")

    negatives_root = (ROOT / args.negatives) if not Path(args.negatives).is_absolute() else Path(args.negatives)
    negatives = discover_negatives(negatives_root) if negatives_root.exists() else []
    if negatives:
        max_neg = args.max_negatives if args.max_negatives is not None else 2 * args.max_per_class
        classes[REJECT_CLASS] = negatives
        print(f"Found {len(negatives)} negative (non-insect) images under {negatives_root} "
              f"-> class '{REJECT_CLASS}' (cap {max_neg})")
    else:
        max_neg = args.max_per_class
        print("\n" + "!" * 70)
        print(f"WARNING: no negative images found at {negatives_root}.")
        print(f"The classifier will have NO '{REJECT_CLASS}' class and will be closed-set:")
        print("any photo of a non-insect will be forced onto the nearest insect class.")
        print("Pass --negatives <folder of non-insect images> to fix this.")
        print("!" * 70 + "\n")

    print(f"\nDiscovered {len(classes)} classes:")
    low_count_warnings = []
    for taxon, images in sorted(classes.items()):
        flag = ""
        if len(images) < args.min_images:
            flag = f"  ** only {len(images)} images, below --min-images {args.min_images} **"
            low_count_warnings.append(taxon)
        print(f"  {taxon:<25} {len(images):>5} images{flag}")

    out_dir = ROOT / args.out
    print(f"\nCopying into {out_dir} (train/val split, capped at "
          f"{args.max_per_class}/class, seed={args.seed})...")
    # The reject class gets its own, larger cap: split it separately so the
    # shared --max-per-class cap doesn't silently throw most negatives away.
    specimen_classes = {t: imgs for t, imgs in classes.items() if t != REJECT_CLASS}
    summary = split_and_copy(specimen_classes, out_dir, args.max_per_class, args.val_fraction, args.seed)
    if REJECT_CLASS in classes:
        summary.update(split_and_copy({REJECT_CLASS: classes[REJECT_CLASS]}, out_dir,
                                      max_neg, args.val_fraction, args.seed))

    print("\nSplit summary:")
    for taxon, s in sorted(summary.items()):
        dupes = f"  ({s['duplicates_dropped']} exact dupes dropped)" if s["duplicates_dropped"] else ""
        print(f"  {taxon:<25} used {s['used']:>4}/{s['available']:<4}  "
              f"train {s['train']:>4}  val {s['val']:>4}{dupes}")

    # The reject class is not a taxon: it has no pest/beneficial status and
    # must never appear in the supported-species list or a tally.
    taxon_status = {taxon: guess_status(taxon).value for taxon in specimen_classes}
    beneficial_or_neutral = {t: s for t, s in taxon_status.items() if s != TaxonStatus.PEST.value}

    species_config_path = ROOT / args.species_config
    cfg = json.loads(species_config_path.read_text()) if species_config_path.exists() else {}
    # class_names is THE index contract: position i here == model output i.
    # torchvision's ImageFolder sorts folder names the same way, and the
    # notebook asserts the two agree before training starts.
    cfg["class_names"] = sorted(classes.keys())
    cfg["taxon_status"] = taxon_status
    cfg["reject_class"] = REJECT_CLASS if REJECT_CLASS in classes else None
    cfg.setdefault("species_info", {})
    for taxon in specimen_classes:
        cfg["species_info"].setdefault(taxon, {
            "common_name": taxon.replace("_", " ").title(),
            "scientific_name": "",
            "damage_symptoms": "",
        })
    if REJECT_CLASS in classes:
        cfg["species_info"][REJECT_CLASS] = {
            "common_name": "Not an insect",
            "scientific_name": None,
            "damage_symptoms": None,
            "is_specimen": False,
        }
    else:
        cfg["species_info"].pop(REJECT_CLASS, None)
    cfg.setdefault("detector_onnx", "models/detector.onnx")
    # Named by class count so a 12-class and a 13-class export can never be
    # confused for one another on disk -- the serve-time loader also checks
    # the ONNX output width against len(class_names) before accepting it.
    cfg["classifier_onnx"] = f"models/classifier_{len(classes)}cls.onnx"
    cfg.setdefault("detector_input_size", 640)
    cfg.setdefault("classifier_input_size", 224)
    cfg.setdefault("detector_conf_threshold", 0.25)
    cfg.setdefault("classifier_macro_f1", None)
    cfg.setdefault("detector_map50", None)
    species_config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"\nWrote {species_config_path}")

    print("\n" + "=" * 70)
    if beneficial_or_neutral:
        print("REVIEW THESE -- guessed as beneficial or neutral (never triggers")
        print("an intervention advisory). Check config/species.json is correct")
        print("before training; a real pest marked beneficial here would")
        print("silently suppress every alert for it:")
        for taxon, status in sorted(beneficial_or_neutral.items()):
            print(f"  {taxon:<25} -> {status}")
    else:
        print("No classes were guessed as beneficial or neutral -- if you "
              "expected bees/ladybirds/earthworms etc. to appear, check the "
              "class names above matched the keyword list in this script.")
    print("=" * 70)

    if low_count_warnings:
        print(f"\nNote: {len(low_count_warnings)} class(es) have very few source "
              f"images ({', '.join(low_count_warnings)}) -- expect weak accuracy "
              "on those specific taxa regardless of training settings.")


if __name__ == "__main__":
    main()
