"""Stage 3: assemble the three-section MiniMax caption.

Phase 1 builds the caption from measured facts and embedded tags only. The
`descriptors` argument is the seam the CLAP layer plugs into: when stage 2 is
present it supplies the adjective vocabulary (genre, mood, instruments, vocal
character) and this module places those words into the right slots.

The composer never invents a number. If something was not measured or scored,
the corresponding clause is omitted rather than filled with a plausible guess -
a caption that says less is far more useful than one that says something wrong.
"""

from __future__ import annotations

import random

SECTION_HEADERS = ("Global Metadata", "Vocal Details", "Arrangement")


def _join(parts, separator=", ", final=" and "):
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    # Vocabulary labels are phrases, and many already contain "and". Appending
    # another one yields "energetic and driving and confident and swaggering",
    # which reads as four items instead of two. Comma-join those instead.
    if any(" and " in part for part in parts):
        return separator.join(parts)
    return separator.join(parts[:-1]) + final + parts[-1]


def _dedupe(parts: list[str]) -> list[str]:
    """Drop entries that restate one already present.

    The CLAP vocabulary and the DSP measurements deliberately overlap, so a
    track can score "a deep sub-heavy club mix" and separately measure a
    "deep sub-heavy low end". Both are true; saying both is padding.
    """
    stopwords = {
        "a", "an", "the", "and", "with", "of", "in", "on", "everything",
        "very", "heavily", "slightly", "mix", "sound", "recording",
    }

    kept: list[str] = []
    kept_tokens: list[set[str]] = []

    for part in parts:
        tokens = {
            word.strip(",.").lower()
            for word in part.split()
            if word.strip(",.").lower() not in stopwords
        }
        if not tokens:
            continue
        # Two thirds shared content words means they are saying the same thing.
        if any(
            len(tokens & seen) / min(len(tokens), len(seen)) >= 0.66
            for seen in kept_tokens
        ):
            continue
        kept.append(part)
        kept_tokens.append(tokens)

    return kept


def _sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if not text.endswith((".", "!", "?")):
        text += "."
    return text[0].upper() + text[1:]


def _descriptor_list(descriptors: dict | None, axis: str, limit: int = 3) -> list[str]:
    if not descriptors:
        return []
    values = descriptors.get(axis) or []
    out = []
    for item in values[:limit]:
        # Stage 2 emits either bare strings or {"label", "score"} dicts.
        out.append(item["label"] if isinstance(item, dict) else str(item))
    return out


def compose_global(analysis: dict, tags: dict, descriptors: dict | None) -> str:
    tempo = analysis.get("tempo", {})
    key = analysis.get("key", {})
    timbre = analysis.get("timbre", {})
    loudness = analysis.get("loudness", {})
    balance = analysis.get("balance", {})
    structure = analysis.get("structure", {})

    clauses: list[str] = []

    genres = _descriptor_list(descriptors, "genre", 2)
    if not genres and tags.get("genre"):
        genres = [tags["genre"].lower()]
    if genres:
        clauses.append(_join(genres))

    # Measured facts: tempo and key.
    facts = []
    if tempo.get("bpm_int"):
        facts.append(f"{tempo['bpm_int']} BPM")
    if key.get("spoken") and (key.get("confidence") or 0) >= 0.15:
        facts.append(key["spoken"])
    if key.get("harmonic_complexity") and (key.get("confidence") or 0) >= 0.15:
        facts.append(key["harmonic_complexity"])
    if facts:
        clauses.append(_join(facts, ", ", ", "))

    moods = _descriptor_list(descriptors, "mood", 2)
    if moods:
        clauses.append(_join(moods))

    # Feel of the groove - measured from beat-interval variance.
    feel_bits = []
    if tempo.get("feel") and tempo["feel"] != "unknown":
        feel_bits.append(f"{tempo['feel']} timing")
    if tempo.get("swing") and tempo["swing"] > 0.06:
        feel_bits.append("noticeable swing")
    if feel_bits:
        clauses.append(_join(feel_bits))

    if structure.get("arc") and structure["arc"] != "unknown":
        clauses.append(f"energy arc: {structure['arc']}")

    scene = _descriptor_list(descriptors, "scene", 2)
    if scene:
        clauses.append(_join(scene))

    # Production profile, from spectral shape and dynamics.
    production = _descriptor_list(descriptors, "production", 3)
    measured_production = []
    if timbre.get("brightness"):
        measured_production.append(f"tonally {timbre['brightness']}")
    if loudness.get("descriptor"):
        measured_production.append(loudness["descriptor"])
    if balance.get("low_end"):
        measured_production.append(balance["low_end"])
    if timbre.get("noisiness") == "noisy / textured":
        measured_production.append("audible noise/texture in the mix")

    # CLAP's production labels come first so that when they collide with a
    # measured descriptor, the more specific scored phrase is the one kept.
    production_text = _join(_dedupe(production + measured_production))
    if production_text:
        clauses.append(f"production: {production_text}")

    return _sentence(". ".join(_sentence(c).rstrip(".") for c in clauses if c))


def compose_vocal(analysis: dict, tags: dict, descriptors: dict | None) -> str:
    timbre_words = _descriptor_list(descriptors, "vocal_timbre", 2)
    delivery_words = _descriptor_list(descriptors, "vocal_delivery", 2)
    presence = (descriptors or {}).get("vocal_presence")

    if presence == "instrumental":
        return "Fully instrumental - no vocals at any point."

    if not (timbre_words or delivery_words):
        # No stage-2 scoring available: say only what is defensible.
        return (
            "Vocal character not analysed - describe the voice you want here, "
            "or leave as-is for the model to choose."
        )

    # Lead with what kind of vocal performance it is, then the voice itself,
    # then how it is delivered - the order MiniMax's own captions use.
    opener = {
        "spoken": "Spoken-word / rapped delivery",
        "wordless": "Wordless vocal textures rather than sung lyrics",
    }.get(presence)

    parts = []
    if opener:
        parts.append(opener)
    if timbre_words:
        parts.append(_join(timbre_words))
    if delivery_words:
        parts.append(_join(delivery_words))

    return _sentence(", ".join(parts))


def compose_arrangement(
    analysis: dict, tags: dict, descriptors: dict | None, rng: random.Random
) -> str:
    balance = analysis.get("balance", {})
    structure = analysis.get("structure", {})
    sections = structure.get("sections") or []

    parts: list[str] = []

    instruments = _descriptor_list(descriptors, "instruments", 5)
    if instruments:
        parts.append(_sentence(f"Core instrumentation: {_join(instruments)}"))

    if balance.get("drum_presence") or balance.get("low_end"):
        bed = f"Mix sits {balance.get('drum_presence', 'balanced')}"
        if balance.get("low_end"):
            bed += f", with a {balance['low_end']}"
        parts.append(_sentence(bed))

    # Section-by-section energy shape. This is the part that makes the caption
    # describe a song that develops rather than a static loop.
    if len(sections) >= 2:
        described = _describe_sections(sections)
        if described:
            parts.append(described)

    if not parts:
        return "Arrangement not analysed."
    return " ".join(parts)


def _describe_sections(sections: list[dict]) -> str:
    """Turn measured per-section energy into an arrangement narrative."""
    labels = []
    total = len(sections)
    # Intro/Outro naming only makes sense when there is a middle for them to
    # bracket. With two sections, calling them intro and outro asserts a
    # structure the segmentation did not actually find.
    named_ends = total >= 3

    for index, section in enumerate(sections):
        if named_ends and index == 0:
            name = "Intro"
        elif named_ends and index == total - 1:
            name = "Outro"
        else:
            name = f"Section {index + 1}"

        energy = section.get("energy_label", "steady")
        phrasing = {
            "sparse": "stripped back and quiet",
            "steady": "holding the main groove",
            "full": "full arrangement, everything in",
        }.get(energy, "steady")

        labels.append(
            f"{name} ({section['start']:.0f}-{section['end']:.0f}s): {phrasing}"
        )

    return _sentence("Section map - " + "; ".join(labels))


def compose_caption(
    analysis: dict,
    tags: dict | None = None,
    descriptors: dict | None = None,
    seed: int = 0,
) -> dict:
    """Build the full caption plus its three parts separately.

    Returning the parts as well as the joined text is what lets the Caption
    Splitter node round-trip without re-parsing prose.
    """
    tags = tags or {}
    rng = random.Random(seed)

    global_text = compose_global(analysis, tags, descriptors)
    vocal_text = compose_vocal(analysis, tags, descriptors)
    arrangement_text = compose_arrangement(analysis, tags, descriptors, rng)

    caption = "\n\n".join(
        f"{header}: {body}"
        for header, body in zip(
            SECTION_HEADERS, (global_text, vocal_text, arrangement_text)
        )
        if body
    )

    return {
        "caption": caption,
        "global": global_text,
        "vocal": vocal_text,
        "arrangement": arrangement_text,
    }
