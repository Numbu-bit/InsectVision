"""
Confidence-based confirmation, reject-class handling, and
pest/beneficial/neutral status.

This is a general-purpose insect identifier, not a crop-management tool --
it has no business inferring severity, advising an intervention, or
assuming anything about a crop growth stage it was never told about. Its
job stops at: here is what species this looks like, here is how sure the
model is, and here is whether that species is generally considered a
pest, beneficial, or neither. What to do with that is left entirely to
the user.

Every identification resolves to exactly one of three verdicts:

  NO_SPECIMEN  the classifier's top class is the reject class ("other").
               The model is saying "this is not one of the insects I
               know". This is NOT an uncertain identification of an
               insect -- an "other" prediction at 90% is a confident
               statement that there's no recognisable insect here, and
               must never be shown as "flagged other" or counted.
  UNCERTAIN    a real taxon, but below CONFIDENCE_THRESHOLD. Shown to the
               user as a guess needing review; never counted as confirmed.
  CONFIRMED    a real taxon at or above CONFIDENCE_THRESHOLD.

The reject-class check comes FIRST, before any threshold comparison --
the threshold only has meaning for classes that are actual species.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TaxonStatus(str, Enum):
    PEST = "pest"
    BENEFICIAL = "beneficial"
    NEUTRAL = "neutral"


class Verdict(str, Enum):
    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    NO_SPECIMEN = "no_specimen"


# HARD RULE: a species prediction below this confidence must never be
# presented as a confirmed identification -- it stays flagged for the user
# to treat as uncertain. Do not raise this threshold.
CONFIDENCE_THRESHOLD = 0.75

# Name of the classifier's "not an insect I recognise" class. It must be a
# real output of the trained model (a folder in data/classify/{train,val}/),
# so that out-of-distribution images -- trees, soil, hands, diagrams -- have
# somewhere to go other than the nearest insect class. Without it a 12-way
# softmax is *forced* to put 100% across the 12 insects, which is exactly
# how a photo of a tree used to come back as "beetle 56%".
#
# species.json carries the same name under "reject_class"; that copy is
# the serve-time contract (checked at model load), this constant is the
# default for tooling that runs before species.json exists.
REJECT_CLASS = "other"


@dataclass
class Identification:
    taxon: str
    confidence: float
    runner_up_taxon: Optional[str] = None
    runner_up_confidence: Optional[float] = None
    reject_class: str = REJECT_CLASS

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be a probability in [0, 1], got {self.confidence}")

    @property
    def is_specimen(self) -> bool:
        """False when the model's best answer is 'not an insect I know'."""
        return self.taxon != self.reject_class

    @property
    def verdict(self) -> Verdict:
        if not self.is_specimen:
            return Verdict.NO_SPECIMEN
        if self.confidence >= CONFIDENCE_THRESHOLD:
            return Verdict.CONFIRMED
        return Verdict.UNCERTAIN

    @property
    def confirmed(self) -> bool:
        return self.verdict is Verdict.CONFIRMED

    @property
    def flagged(self) -> bool:
        """Anything that is not a confirmed species identification. Kept as
        a single boolean because it's the one question every consumer asks
        ("may I count this?"), and the answer is 'no' for both an uncertain
        insect and a non-insect."""
        return not self.confirmed

    @property
    def closest_species(self) -> Optional[tuple[str, float]]:
        """For a NO_SPECIMEN verdict: the best *insect* candidate the model
        considered, so the UI can say "closest match: beetle (31%)" without
        ever presenting it as an identification. None for other verdicts,
        or if the runner-up is unavailable."""
        if self.is_specimen or self.runner_up_taxon is None:
            return None
        if self.runner_up_taxon == self.reject_class:
            return None
        return self.runner_up_taxon, float(self.runner_up_confidence or 0.0)
