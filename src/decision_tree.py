"""
Confidence-based confirmation and pest/beneficial/neutral status.

This is a general-purpose insect identifier, not a crop-management tool --
it has no business inferring severity, advising an intervention, or
assuming anything about a crop growth stage it was never told about. Its
job stops at: here is what species this looks like, here is how sure the
model is, and here is whether that species is generally considered a
pest, beneficial, or neither. What to do with that is left entirely to
the user.
"""
from dataclasses import dataclass
from enum import Enum


class TaxonStatus(str, Enum):
    PEST = "pest"
    BENEFICIAL = "beneficial"
    NEUTRAL = "neutral"


# HARD RULE: a species prediction below this confidence must never be
# presented as a confirmed identification -- it stays flagged for the user
# to treat as uncertain. Do not raise this threshold.
CONFIDENCE_THRESHOLD = 0.75


@dataclass
class Identification:
    taxon: str
    confidence: float

    @property
    def confirmed(self) -> bool:
        return self.confidence >= CONFIDENCE_THRESHOLD
