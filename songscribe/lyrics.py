"""Lyric section tags and length fitting.

MiniMax treats the bracketed section tags as the only executable structural
instruction - the lyric text itself only conveys mood. A malformed tag
therefore doesn't produce a slightly-off song, it silently drops a structural
instruction from a render that may take minutes.
"""

from __future__ import annotations

import re

# The canonical tags MiniMax responds to.
CANONICAL = ("Intro", "Verse", "Pre-Chorus", "Chorus", "Bridge", "Instrumental", "Outro")

# Common ways people write each one. Matching is case-insensitive and ignores
# trailing numbering ("Verse 2", "chorus_1").
ALIASES = {
    "intro": "Intro",
    "introduction": "Intro",
    "verse": "Verse",
    "vs": "Verse",
    "prechorus": "Pre-Chorus",
    "pre-chorus": "Pre-Chorus",
    "pre chorus": "Pre-Chorus",
    "build": "Pre-Chorus",
    "chorus": "Chorus",
    "hook": "Chorus",
    "refrain": "Chorus",
    "bridge": "Bridge",
    "middle8": "Bridge",
    "middle 8": "Bridge",
    "break": "Instrumental",
    "breakdown": "Instrumental",
    "instrumental": "Instrumental",
    "solo": "Instrumental",
    "interlude": "Instrumental",
    "outro": "Outro",
    "ending": "Outro",
    "coda": "Outro",
    "end": "Outro",
}

# [Verse], (Chorus), {bridge}, and bare "Verse 2:" on its own line.
_BRACKETED = re.compile(r"^[\s]*[\[\(\{]\s*([^\]\)\}]+?)\s*[\]\)\}][\s:]*$")
_BARE = re.compile(r"^\s*([A-Za-z][A-Za-z\- ]{1,14}?)\s*(\d+)?\s*:\s*$")

# Rough syllable counting: vowel groups, minus silent trailing 'e'.
_VOWEL_GROUPS = re.compile(r"[aeiouy]+")

# Sung syllables per second at a moderate tempo. Real delivery varies hugely,
# which is why the fit check reports a range rather than a single number.
SYLLABLES_PER_SECOND = 3.2


def _canonicalise(raw: str) -> str | None:
    """Map a tag's inner text onto a canonical tag, or None if unrecognised."""
    cleaned = re.sub(r"\s+", " ", raw.strip().lower()).strip()
    if not cleaned:
        return None

    # Match the full name before stripping trailing digits. Some canonical
    # names end in a number themselves - "middle 8" is a bridge - and removing
    # the digits first turns it into "middle", which matches nothing.
    if cleaned in ALIASES:
        return ALIASES[cleaned]

    # "verse 2" / "chorus_1" -> "verse" / "chorus"
    without_number = re.sub(r"[\s_]*\d+$", "", cleaned).strip()
    return ALIASES.get(without_number)


def count_syllables(text: str) -> int:
    total = 0
    for word in re.findall(r"[A-Za-z']+", text):
        word = word.lower()
        groups = len(_VOWEL_GROUPS.findall(word))
        if word.endswith("e") and groups > 1:
            groups -= 1
        total += max(1, groups)
    return total


def normalise(text: str) -> dict:
    """Rewrite section tags into canonical bracketed form.

    Returns the normalised lyrics plus a structured report of what changed and
    what could not be understood.
    """
    lines = (text or "").splitlines()
    out: list[str] = []
    sections: list[str] = []
    changes: list[str] = []
    unknown: list[str] = []
    lyric_lines = 0
    syllables = 0

    for number, line in enumerate(lines, start=1):
        bracket_match = _BRACKETED.match(line)
        bare_match = None if bracket_match else _BARE.match(line)
        raw = None

        if bracket_match:
            raw = bracket_match.group(1)
        elif bare_match:
            candidate = bare_match.group(1)
            # Only treat a bare "Word:" as a tag if it names a known section;
            # otherwise it is a lyric line that happens to end in a colon.
            if _canonicalise(candidate):
                raw = candidate

        if raw is not None:
            canonical = _canonicalise(raw)
            if canonical:
                replacement = f"[{canonical}]"
                if line.strip() != replacement:
                    changes.append(f"line {number}: {line.strip()!r} -> {replacement}")
                out.append(replacement)
                sections.append(canonical)
                continue
            unknown.append(f"line {number}: {line.strip()!r}")
            out.append(line)
            continue

        out.append(line)
        if line.strip():
            lyric_lines += 1
            syllables += count_syllables(line)

    return {
        "lyrics": "\n".join(out),
        "sections": sections,
        "section_count": len(sections),
        "changes": changes,
        "unknown_tags": unknown,
        "lyric_lines": lyric_lines,
        "syllables": syllables,
    }


def estimate_duration(syllables: int, section_count: int) -> dict:
    """Very rough sung-duration estimate, deliberately reported as a range.

    Delivery speed varies enormously between a ballad and a rap verse, so a
    single number here would imply a precision that does not exist.
    """
    if syllables <= 0:
        return {"low": 0.0, "high": 0.0, "mid": 0.0}

    mid = syllables / SYLLABLES_PER_SECOND
    # Instrumental space between sections that carries no syllables.
    mid += section_count * 4.0
    return {
        "low": round(mid * 0.65, 1),
        "mid": round(mid, 1),
        "high": round(mid * 1.6, 1),
    }


def check(text: str, target_duration: float | None = None) -> dict:
    """Normalise, then report anything that would degrade a render."""
    result = normalise(text)
    warnings: list[str] = []

    if not result["sections"]:
        warnings.append(
            "No section tags found. MiniMax treats tags as the only structural "
            "instruction - without them the model chooses its own structure."
        )
    else:
        if result["sections"][0] != "Intro":
            warnings.append(
                f"First section is [{result['sections'][0]}], not [Intro]."
            )
        if result["sections"][-1] != "Outro":
            warnings.append(
                f"Last section is [{result['sections'][-1]}], not [Outro]."
            )

    if result["unknown_tags"]:
        warnings.append(
            f"{len(result['unknown_tags'])} tag(s) not recognised and left "
            f"untouched: {'; '.join(result['unknown_tags'][:3])}"
        )

    estimate = estimate_duration(result["syllables"], result["section_count"])
    result["estimated_duration"] = estimate

    if target_duration and estimate["mid"] > 0:
        if target_duration < estimate["low"]:
            warnings.append(
                f"max_duration is {target_duration:.0f}s but these lyrics need "
                f"roughly {estimate['low']:.0f}-{estimate['high']:.0f}s. Expect "
                "words to be cut or rushed."
            )
        elif target_duration > estimate["high"] * 1.5:
            warnings.append(
                f"max_duration is {target_duration:.0f}s but these lyrics only "
                f"fill roughly {estimate['mid']:.0f}s. Expect long instrumental "
                "stretches."
            )

    result["warnings"] = warnings
    return result


def format_report(result: dict) -> str:
    lines = []
    counts: dict[str, int] = {}
    for section in result["sections"]:
        counts[section] = counts.get(section, 0) + 1

    if counts:
        summary = ", ".join(f"{name} x{n}" if n > 1 else name for name, n in counts.items())
        lines.append(f"Structure: {summary}")
    lines.append(
        f"{result['lyric_lines']} lyric line(s), ~{result['syllables']} syllables"
    )

    estimate = result.get("estimated_duration") or {}
    if estimate.get("mid"):
        lines.append(
            f"Estimated sung length: ~{estimate['mid']:.0f}s "
            f"(range {estimate['low']:.0f}-{estimate['high']:.0f}s)"
        )

    if result["changes"]:
        lines.append(f"\nNormalised {len(result['changes'])} tag(s):")
        lines.extend(f"  {c}" for c in result["changes"][:10])
        if len(result["changes"]) > 10:
            lines.append(f"  ... and {len(result['changes']) - 10} more")

    if result["warnings"]:
        lines.append("\nWarnings:")
        lines.extend(f"  ! {w}" for w in result["warnings"])
    else:
        lines.append("\nNo problems found.")

    return "\n".join(lines)
