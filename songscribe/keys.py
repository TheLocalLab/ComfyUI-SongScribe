"""Key / scale estimation via Krumhansl-Schmuckler profile correlation."""

from __future__ import annotations

import numpy as np

# Krumhansl-Kessler probe-tone profiles.
_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Two spellings per pitch class. Which one we use depends on the key signature:
# flat keys get flat names, sharp keys get sharp names.
_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Pitch classes whose conventional major/minor spelling uses flats.
_FLAT_MAJOR_ROOTS = {1, 3, 5, 8, 10}  # Db, Eb, F, Ab, Bb
_FLAT_MINOR_ROOTS = {1, 3, 5, 8, 10}  # C#/Db m, Eb m, F m, G#/Ab m, Bb m

_SPOKEN = {"#": " sharp", "b": " flat"}


def _spell(pitch_class: int, is_major: bool) -> str:
    flat_roots = _FLAT_MAJOR_ROOTS if is_major else _FLAT_MINOR_ROOTS
    table = _FLAT if pitch_class in flat_roots else _SHARP
    return table[pitch_class]


def spoken_key(name: str) -> str:
    """'Db' -> 'D flat', 'F#' -> 'F sharp'. MiniMax captions spell accidentals out."""
    if len(name) > 1 and name[1] in _SPOKEN:
        return name[0] + _SPOKEN[name[1]]
    return name


def estimate_key(chroma: np.ndarray) -> dict:
    """Estimate key from a chromagram.

    Args:
        chroma: (12, n_frames) chromagram.

    Returns dict with root/mode/name/spoken/confidence. Confidence is the
    normalised margin between the best and second-best hypothesis, which is a
    far more honest signal than the raw correlation - a track that correlates
    0.9 with both C major and A minor is genuinely ambiguous, not confident.
    """
    profile = chroma.mean(axis=1)
    total = profile.sum()
    if total <= 0:
        return {
            "root": None,
            "mode": None,
            "name": None,
            "spoken": None,
            "confidence": 0.0,
        }
    profile = profile / total

    scores: list[tuple[float, int, bool]] = []
    for pc in range(12):
        rotated = np.roll(profile, -pc)
        for template, is_major in ((_MAJOR, True), (_MINOR, False)):
            corr = float(np.corrcoef(rotated, template)[0, 1])
            if not np.isfinite(corr):
                corr = -1.0
            scores.append((corr, pc, is_major))

    scores.sort(key=lambda s: s[0], reverse=True)
    corr, pc, is_major = scores[0]

    # A key and its relative (Db major / Bb minor) share all seven pitch
    # classes, so they always score within a hair of each other. Measuring
    # confidence against the relative would report every correctly-identified
    # key as uncertain, so the margin is taken against the best *genuinely
    # different* hypothesis instead, and the relative is reported separately.
    rel_pc = (pc + 9) % 12 if is_major else (pc + 3) % 12
    rel_major = not is_major
    rival = next(
        (s for s in scores[1:] if not (s[1] == rel_pc and s[2] == rel_major)),
        None,
    )
    margin = corr - rival[0] if rival else corr

    # Two independent requirements: the winner must fit the profile well in
    # absolute terms, and it must beat the field. Either one failing should
    # pull confidence down, so they multiply.
    #
    # The separation scale is deliberately small. Neighbouring keys on the
    # circle of fifths share six of seven pitch classes, so even an unambiguous
    # key typically wins by only 0.02-0.10 of correlation. Scaling against a
    # larger spread would report every correct answer as a coin flip.
    fit = np.clip((corr - 0.2) / 0.5, 0.0, 1.0)
    separation = np.clip(margin / 0.06, 0.0, 1.0)
    confidence = float(fit * separation)

    rel_score = next(
        (s[0] for s in scores if s[1] == rel_pc and s[2] == rel_major), None
    )
    rel_name = _spell(rel_pc, rel_major)

    name = _spell(pc, is_major)
    mode = "major" if is_major else "minor"
    return {
        "root": name,
        "mode": mode,
        "name": f"{name} {mode}",
        "spoken": f"{spoken_key(name)} {mode}",
        "confidence": round(confidence, 3),
        "correlation": round(corr, 3),
        "margin": round(float(margin), 3),
        "relative": f"{rel_name} {'major' if rel_major else 'minor'}",
        # How close the relative is: near 0 means the two are indistinguishable
        # from the chroma alone, which is the honest answer for most tracks.
        "relative_margin": (
            None if rel_score is None else round(float(corr - rel_score), 3)
        ),
    }
