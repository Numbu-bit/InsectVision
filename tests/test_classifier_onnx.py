import numpy as np
import pytest
from PIL import Image

from src.classifier_onnx import IMAGENET_MEAN, IMAGENET_STD, ClassifierOnnx
from tests.conftest import CLASSIFIER, requires_classifier

torchvision = pytest.importorskip("torchvision", reason="preprocessing parity test needs torchvision")
from torchvision import transforms  # noqa: E402


def _preprocess_only(img, size=224):
    """Run _preprocess without a real session (geometry + normalisation only)."""
    clf = ClassifierOnnx.__new__(ClassifierOnnx)
    clf.input_size = size
    return clf._preprocess(img)


@pytest.mark.parametrize("size", [(400, 300), (301, 500), (555, 555), (640, 427), (1000, 333),
                                  (255, 400), (224, 224), (97, 131), (3000, 2000)])
def test_preprocess_matches_torchvision_val_transform_exactly(size):
    """Train/serve skew guard. The notebook trains and validates with
    Resize(int(224*1.14)) -> CenterCrop(224) -> ToTensor -> Normalize; the
    serve-time code must produce the identical tensor, to the bit. An earlier
    version differed by one pixel (round() vs int(), // vs round(/2))."""
    val_tf = transforms.Compose([
        transforms.Resize(int(224 * 1.14)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
    ])
    w, h = size
    img = Image.fromarray(np.random.default_rng(w * h).integers(0, 255, (h, w, 3), dtype=np.uint8))
    expected = val_tf(img)[None].numpy()
    actual = _preprocess_only(img)
    assert actual.shape == (1, 3, 224, 224) and actual.dtype == np.float32
    assert np.array_equal(expected, actual), f"max diff {np.abs(expected - actual).max()}"


def test_palette_image_is_converted_before_resize():
    rgb = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8))
    pal = rgb.convert("P")
    # must not crash; and must equal preprocessing the RGB conversion of the palette image
    assert np.array_equal(_preprocess_only(pal), _preprocess_only(pal.convert("RGB")))


@requires_classifier()
class TestWithTrainedModel:
    def test_refuses_class_count_mismatch(self, specimen_class_names):
        # 12 insect names against the 13-output file -- the exact mistake of
        # dropping in a new model without its species.json.
        with pytest.raises(ValueError, match="out of sync"):
            ClassifierOnnx(str(CLASSIFIER), specimen_class_names)

    def test_refuses_duplicate_class_names(self, species_cfg):
        names = species_cfg["class_names"]
        with pytest.raises(ValueError, match="duplicates"):
            ClassifierOnnx(str(CLASSIFIER), names[:-1] + [names[0]])

    def test_probabilities_sum_to_one_and_top_k_bounded(self, species_cfg):
        names = species_cfg["class_names"]
        clf = ClassifierOnnx(str(CLASSIFIER), names)
        img = Image.fromarray(np.random.default_rng(3).integers(0, 255, (300, 300, 3), dtype=np.uint8))
        probs = clf.probabilities(img)
        assert probs.shape == (len(names),) and abs(probs.sum() - 1) < 1e-5
        assert len(clf.classify(img, top_k=99)) == len(names)
        assert len(clf.classify(img, top_k=0)) == 1

    def test_non_insect_images_resolve_to_reject_class(self, species_cfg):
        """The reason the 13th class exists: synthetic non-insect inputs must
        land on 'other', not on the nearest insect (the old 'beetle 56%' bug)."""
        clf = ClassifierOnnx(str(CLASSIFIER), species_cfg["class_names"])
        rng = np.random.default_rng(0)
        grey = Image.new("RGB", (400, 400), (128, 128, 128))
        noise = Image.fromarray(rng.integers(0, 255, (400, 400, 3), dtype=np.uint8))
        for img in (grey, noise):
            top = clf.classify(img, top_k=1)[0]
            assert top.taxon == species_cfg["reject_class"], f"got {top.taxon} {top.confidence:.2f}"
