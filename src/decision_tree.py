"""
Rule-based severity decision tree.

Deliberately rule-based rather than learned end-to-end: an advisory that may
lead a farmer to apply a control measure must be auditable, and thresholds
must be revisable (in config/species.json) without retraining anything.
Every branch taken is recorded in Decision.path for that reason.

This module has zero dependency on how a taxon was identified -- it doesn't
know or care whether counts came from a detector auto-counting boxes or from
a farmer typing a number next to a single classified crop. That's what lets
the same decision tree serve both app modes described in the project spec.
"""
from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class TaxonStatus(str, Enum):
    PEST = "pest"
    BENEFICIAL = "beneficial"
    NEUTRAL = "neutral"


# HARD RULE: a species prediction below this confidence is referred to a
# human for review and must never be presented as a confirmed identification.
# Do not raise this threshold to make the system look more decisive --
# that would be presenting guesses as answers.
CONFIDENCE_THRESHOLD = 0.75

# Generic fallback: pest instance count per image above which a taxon is
# treated as economically significant, used when config/species.json has no
# taxon-specific override. Deliberately NOT a dict of hardcoded species names
# here -- the real taxon list depends on whatever dataset prepare_data.py
# ends up building (Step 2), so per-taxon thresholds live in config, and this
# module only needs a sane number to fall back on.
DEFAULT_ECONOMIC_THRESHOLD = 5

SUSCEPTIBLE_STAGES = {"seedling", "vegetative", "tasselling", "flowering"}


@dataclass
class Identification:
    """One taxon identification contributing to the count for an image.

    In cascade mode the caller creates one Identification per confirmed
    detection (count=1 each) and lets `evaluate` aggregate them. In
    classifier-only mode the caller creates a single Identification carrying
    the farmer-typed `count` directly. Either way `evaluate` just sums counts
    per taxon -- it doesn't need to know which mode produced them.
    """
    taxon: str
    confidence: float
    count: int = 1


@dataclass
class Decision:
    severity: Severity
    action: str
    advisory: str
    path: list[str] = field(default_factory=list)
    # Identifications below CONFIDENCE_THRESHOLD, carried through unmodified
    # so the UI can show "an insect was seen but we're not sure what it is"
    # without them ever influencing severity or being labelled as confirmed.
    flagged: list[Identification] = field(default_factory=list)


def evaluate(identifications: list[Identification],
             taxon_status: dict[str, TaxonStatus],
             growth_stage: str = "vegetative",
             economic_thresholds: dict[str, int] | None = None) -> Decision:
    """Walk the decision tree and return a severity, action and advisory.

    Assumes the caller has already gated image quality (see imageops.py) --
    this function is only about turning an already-accepted image's
    identifications into a severity judgement, not about whether the image
    was usable in the first place.
    """
    thresholds = economic_thresholds or {}
    path: list[str] = []

    # D1 - anything identified at all?
    if not identifications:
        path.append("no identification for this image")
        return Decision(Severity.NONE, "log_clean_scan",
                        "No insects were detected. Logged as a clean scan; "
                        "continue routine monitoring.", path)

    # D2 - species confidence. HARD RULE: below CONFIDENCE_THRESHOLD is
    # referred to a human, never presented as a confirmed identification.
    confirmed = [i for i in identifications if i.confidence >= CONFIDENCE_THRESHOLD]
    flagged = [i for i in identifications if i.confidence < CONFIDENCE_THRESHOLD]
    if not confirmed:
        path.append(f"all {len(flagged)} identification(s) below confidence "
                    f"{CONFIDENCE_THRESHOLD}")
        return Decision(Severity.NONE, "flag_for_review",
                        "An insect was seen but the system is not confident "
                        "enough to identify it. Please consult an extension "
                        "officer to confirm the species.", path, flagged=flagged)
    path.append(f"{len(confirmed)} confirmed, {len(flagged)} flagged for review")

    # D3 - aggregate confirmed counts per taxon (flagged ones never count
    # towards severity -- an uncertain guess must not be able to trigger a
    # high-severity alert).
    counts: dict[str, int] = {}
    for ident in confirmed:
        counts[ident.taxon] = counts.get(ident.taxon, 0) + ident.count

    # D4 - pest or beneficial/neutral?
    pests = {t: n for t, n in counts.items()
             if taxon_status.get(t, TaxonStatus.PEST) is TaxonStatus.PEST}
    if not pests:
        path.append("no pest taxa among confirmed identifications")
        names = ", ".join(counts)
        return Decision(Severity.NONE, "no_intervention",
                        f"Identified {names}, which is beneficial or neutral. "
                        "No intervention is advised; these insects help "
                        "control pests.", path, flagged=flagged)
    path.append(f"pest taxa present: {', '.join(pests)}")

    # D5 - economic threshold, per taxon (config-overridable).
    over_threshold = {t: n for t, n in pests.items()
                      if n >= thresholds.get(t, DEFAULT_ECONOMIC_THRESHOLD)}
    if over_threshold:
        path.append("count >= economic threshold")
        worst = max(over_threshold, key=lambda t: over_threshold[t])
        threshold = thresholds.get(worst, DEFAULT_ECONOMIC_THRESHOLD)
        return Decision(
            Severity.HIGH, "alert",
            f"High infestation of {worst.replace('_', ' ')} "
            f"({over_threshold[worst]} counted, threshold {threshold}). "
            "Consult your extension officer about an integrated pest "
            "management intervention without delay.", path, flagged=flagged)
    path.append("count < economic threshold")

    # D6 - crop growth stage susceptibility.
    if growth_stage.lower() in SUSCEPTIBLE_STAGES:
        path.append(f"growth stage '{growth_stage}' is susceptible")
        worst = max(pests, key=lambda t: pests[t])
        return Decision(
            Severity.MODERATE, "cultural_biological_control",
            f"Moderate risk: {worst.replace('_', ' ')} detected at a "
            "susceptible crop stage. Consider cultural or biological control "
            "such as hand picking, field sanitation or conserving natural "
            "enemies. Re-scan within 72 hours.", path, flagged=flagged)

    path.append(f"growth stage '{growth_stage}' is tolerant")
    return Decision(Severity.LOW, "monitor",
                    "Low risk: pest numbers are below the economic threshold "
                    "and the crop is at a tolerant stage. Logged; continue "
                    "monitoring.", path, flagged=flagged)
