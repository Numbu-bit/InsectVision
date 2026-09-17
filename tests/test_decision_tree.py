import pytest

from src.decision_tree import CONFIDENCE_THRESHOLD, REJECT_CLASS, Identification, Verdict


def test_reject_class_wins_regardless_of_confidence():
    # "other" at 90% is a confident statement that there is no insect --
    # it must never be presented as a flagged/uncertain identification.
    ident = Identification(REJECT_CLASS, 0.90, "beetle", 0.05)
    assert ident.verdict is Verdict.NO_SPECIMEN
    assert not ident.is_specimen
    assert ident.flagged and not ident.confirmed


def test_reject_class_below_threshold_is_still_no_specimen():
    ident = Identification(REJECT_CLASS, 0.45, "beetle", 0.40)
    assert ident.verdict is Verdict.NO_SPECIMEN
    assert ident.closest_species == ("beetle", 0.40)


def test_specimen_verdicts_follow_threshold():
    assert Identification("beetle", CONFIDENCE_THRESHOLD).verdict is Verdict.CONFIRMED
    assert Identification("beetle", CONFIDENCE_THRESHOLD - 1e-9).verdict is Verdict.UNCERTAIN
    assert Identification("beetle", 0.56, REJECT_CLASS, 0.30).verdict is Verdict.UNCERTAIN


def test_closest_species_only_for_no_specimen():
    assert Identification("beetle", 0.9, "earwig", 0.05).closest_species is None
    assert Identification(REJECT_CLASS, 0.9).closest_species is None


def test_flagged_means_do_not_count():
    assert Identification("beetle", 0.92).flagged is False
    assert Identification("beetle", 0.50).flagged is True
    assert Identification(REJECT_CLASS, 0.99).flagged is True


def test_custom_reject_class_name():
    ident = Identification("background", 0.8, reject_class="background")
    assert ident.verdict is Verdict.NO_SPECIMEN
    # and the default name is then an ordinary taxon
    assert Identification(REJECT_CLASS, 0.8, reject_class="background").verdict is Verdict.CONFIRMED


def test_confidence_must_be_probability():
    with pytest.raises(ValueError):
        Identification("beetle", 1.5)


def test_threshold_is_the_documented_hard_rule():
    assert CONFIDENCE_THRESHOLD == 0.75
