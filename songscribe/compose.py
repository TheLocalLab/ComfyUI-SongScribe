"""Stage 3: assemble the three-section MiniMax caption.

Sentence order follows MiniMax's own captions: genre, then measured facts, then
mood and its arc, then listening context, then production - and in Arrangement,
instrumentation and groove before the section-by-section walk.

Two rules govern everything here:

1. The composer never invents a number. Anything not measured or scored is
   omitted rather than filled with a plausible guess, because every sentence in
   a caption becomes an instruction to the music model.
2. Detail is a dial, not a maximum. Feeding a verbatim analysis of a song back
   into a generator produces a clone of it; `style` controls how much of the
   source survives.
"""

from __future__ import annotations

import random

SECTION_HEADERS = ("Global Metadata", "Vocal Details", "Arrangement")

# How literally the caption reproduces the analysed track.
STYLES = {
    # Everything measured, including exact section timings. Best for
    # reproducing a reference track as closely as the model allows.
    "verbatim": {
        "exact_numbers": True,
        "section_timings": True,
        "section_detail": "full",
        "max_instruments": 6,
        "max_production": 4,
    },
    # Keeps tempo and key, drops second-level timings. The default: enough to
    # steer the model, loose enough that it writes a new song.
    "balanced": {
        "exact_numbers": True,
        "section_timings": False,
        "section_detail": "narrative",
        "max_instruments": 5,
        "max_production": 3,
    },
    # Vibe only. No tempo, no key, no structure - genre, mood and texture.
    "loose": {
        "exact_numbers": False,
        "section_timings": False,
        "section_detail": "none",
        "max_instruments": 3,
        "max_production": 2,
    },
}

DEFAULT_STYLE = "balanced"

# Seeded phrasing alternatives. The section *labels* are never varied - the
# three headers and the "Production:" prefix are part of the format MiniMax
# expects - but the connective prose around them can be, so re-rolling the seed
# gives a genuinely different caption without re-analysing the audio.
PHRASINGS = {
    "mood_suffix": ["throughout", "from start to finish", "across the whole track"],
    "harmony_role": [
        "carrying the harmony",
        "holding the harmonic bed",
        "as the harmonic bed",
    ],
    "mix_opener": ["Mix sits", "The mix sits", "Overall balance is"],
    "energy_sparse": [
        "stripped back, fewer elements",
        "pared down, elements dropping away",
        "sparse, space opening up",
    ],
    "energy_steady": [
        "the main groove holding",
        "the core groove running",
        "steady, the groove locked in",
    ],
    "energy_full": [
        "full arrangement, everything present",
        "everything in, at full weight",
        "the fullest texture in the track",
    ],
}


# --------------------------------------------------------------------------
# text helpers


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
        if any(
            len(tokens & seen) / min(len(tokens), len(seen)) >= 0.66
            for seen in kept_tokens
        ):
            continue
        kept.append(part)
        kept_tokens.append(tokens)

    return kept


def _sentence(text: str) -> str:
    text = (text or "").strip().rstrip(",")
    if not text:
        return ""
    if not text.endswith((".", "!", "?")):
        text += "."
    return text[0].upper() + text[1:]


def _paragraph(sentences: list[str]) -> str:
    return " ".join(s for s in (_sentence(x) for x in sentences) if s)


def _labels(descriptors: dict | None, axis: str, limit: int = 3) -> list[str]:
    if not descriptors:
        return []
    out = []
    for item in (descriptors.get(axis) or [])[:limit]:
        out.append(item["label"] if isinstance(item, dict) else str(item))
    return out


def _detail(descriptors: dict | None, axis: str) -> dict:
    return (descriptors or {}).get(f"_{axis}_detail") or {}


# --------------------------------------------------------------------------
# Global Metadata


def _tempo_phrase(tempo: dict, style: dict, rng: random.Random) -> str:
    if not tempo:
        return ""
    if style["exact_numbers"] and tempo.get("bpm_int"):
        return f"{tempo['bpm_int']} BPM"
    # Loose mode still conveys pace, just not a number to lock onto.
    return tempo.get("descriptor", "")


def _key_phrase(key: dict, style: dict) -> str:
    if not key or not key.get("spoken"):
        return ""
    if (key.get("confidence") or 0) < 0.12:
        return ""
    if not style["exact_numbers"]:
        # Without a specific key, the mode alone still shapes the harmony.
        return f"{key.get('mode', '')} tonality".strip()

    phrase = key["spoken"]
    complexity = key.get("harmonic_complexity")
    if complexity == "chromatic / extended harmony":
        phrase += ", chromatic extended harmony"
    elif complexity == "moderately extended":
        phrase += ", with extended chord voicings"
    return phrase


def _arc_phrase(structure: dict, rng: random.Random) -> str:
    """Narrate how energy moves through the track.

    Built from measured per-section RMS relative to the median, so it says
    something real about the arrangement rather than restating the mood.
    """
    sections = structure.get("sections") or []
    if len(sections) < 3:
        return ""

    energies = [s.get("relative_energy", 0.0) for s in sections]
    peak = max(range(len(energies)), key=lambda i: energies[i])
    position = peak / max(len(energies) - 1, 1)

    # These sit between two other clauses in a comma-joined sentence, so none
    # of them may contain a comma of its own or the whole thing reads as a
    # flat list rather than a shape.
    if position < 0.33:
        shape = rng.choice(
            [
                "hitting its fullest early and easing off after",
                "front-loaded with the biggest moment near the top",
            ]
        )
    elif position > 0.66:
        shape = rng.choice(
            [
                "building steadily toward a peak late on",
                "saving its fullest moment for the closing stretch",
            ]
        )
    else:
        shape = rng.choice(
            [
                "swelling through the middle before settling again",
                "deepening in the middle and easing back out",
            ]
        )

    opening = "opening sparse" if energies[0] < -1.5 else "opening full"
    ending = (
        "dissolving softly at the end"
        if energies[-1] < -1.5
        else "holding its weight to the end"
    )
    return f"{opening}, {shape}, {ending}"


def compose_global(
    analysis: dict, tags: dict, descriptors: dict | None, style: dict, rng
) -> str:
    tempo = analysis.get("tempo", {})
    key = analysis.get("key", {})
    timbre = analysis.get("timbre", {})
    loudness = analysis.get("loudness", {})
    balance = analysis.get("balance", {})
    structure = analysis.get("structure", {})

    sentences: list[str] = []

    # 1. Genre.
    genres = _labels(descriptors, "genre", 2)
    if not genres and tags.get("genre"):
        genres = [tags["genre"].lower()]
    if genres:
        sentences.append(", ".join(genres))

    # 2. Measured facts.
    facts = [_tempo_phrase(tempo, style, rng), _key_phrase(key, style)]
    if style["exact_numbers"]:
        feel = tempo.get("feel")
        if feel and feel != "unknown":
            facts.append(f"{feel} timing")
        if tempo.get("swing") and tempo["swing"] > 0.06:
            facts.append("with a noticeable swing")
    fact_text = ", ".join(f for f in facts if f)
    if fact_text:
        sentences.append(fact_text)

    # 3. Mood, carried across the track's shape.
    moods = _labels(descriptors, "mood", 2)
    arc = _arc_phrase(structure, rng) if style["section_detail"] != "none" else ""
    suffix = rng.choice(PHRASINGS["mood_suffix"])
    if moods and arc:
        sentences.append(f"{_join(moods)} {suffix}, {arc}")
    elif moods:
        sentences.append(f"{_join(moods)} {suffix}")
    elif arc:
        sentences.append(arc)

    # 4. Listening context.
    scene = _labels(descriptors, "scene", 2)
    if scene:
        sentences.append(_join(scene))

    # 5. Production profile: scored texture words first, measured ones after,
    # so the more specific phrase wins any collision in _dedupe.
    production = _labels(descriptors, "production", style["max_production"])
    measured = []
    if timbre.get("brightness"):
        measured.append(f"tonally {timbre['brightness']}")
    if loudness.get("descriptor"):
        measured.append(loudness["descriptor"])
    if balance.get("low_end"):
        measured.append(balance["low_end"])
    if timbre.get("noisiness") == "noisy / textured":
        measured.append("audible noise and texture in the mix")

    if not style["exact_numbers"]:
        measured = measured[:1]

    production_text = _join(_dedupe(production + measured))
    if production_text:
        sentences.append(f"Production: {production_text}")

    return _paragraph(sentences)


# --------------------------------------------------------------------------
# Vocal Details


def compose_vocal(
    analysis: dict, tags: dict, descriptors: dict | None, style: dict, rng
) -> str:
    presence = (descriptors or {}).get("vocal_presence")

    if presence == "instrumental":
        return "Fully instrumental - no vocals at any point."

    timbre_words = _labels(descriptors, "vocal_timbre", 2)
    delivery_words = _labels(descriptors, "vocal_delivery", 2)

    if not (timbre_words or delivery_words):
        return (
            "Vocal character not analysed - describe the voice you want here, "
            "or leave as-is for the model to choose."
        )

    sentences = []

    opener = {
        "spoken": "Spoken-word and rapped delivery rather than sung melody",
        "wordless": "Wordless vocal textures rather than sung lyrics",
    }.get(presence)
    if opener:
        sentences.append(opener)

    voice = _join(_dedupe(timbre_words + delivery_words))
    if voice:
        sentences.append(voice)

    # Vocal treatment worth restating here: the reader of this section wants to
    # know how the voice sits, and reverb/delay is the biggest part of that.
    treatments = [
        label
        for label in _labels(descriptors, "production", 3)
        if any(word in label for word in ("reverb", "delay", "autotun", "vocod"))
    ]
    if treatments:
        sentences.append(f"Voice treated with {_join(treatments)}")

    return _paragraph(sentences)


# --------------------------------------------------------------------------
# Arrangement


def _instrument_roles(instruments: list[str], balance: dict, rng) -> str:
    """Give the instrument list a little structure instead of a flat dump."""
    if not instruments:
        return ""

    percussive_words = ("drum", "percussion", "hi-hat", "808", "clap", "tambourine")
    bass_words = ("bass",)

    drums = [i for i in instruments if any(w in i.lower() for w in percussive_words)]
    bass = [i for i in instruments if any(w in i.lower() for w in bass_words)]
    melodic = [i for i in instruments if i not in drums and i not in bass]

    parts = []
    if drums:
        parts.append(_join(drums))
    if bass:
        parts.append(_join(bass))
    if melodic:
        parts.append(f"{_join(melodic)} {rng.choice(PHRASINGS['harmony_role'])}")

    return _join(parts, separator="; ", final="; ")


def _section_walk(structure: dict, descriptors: dict | None, style: dict, rng) -> str:
    """Walk the measured section map.

    Sections are named by position and energy only. The segmentation finds
    *where* the music changes, not *what* a section is - calling something a
    chorus would be an assertion the analysis cannot support.
    """
    sections = structure.get("sections") or []
    mode = style["section_detail"]

    if mode == "none" or len(sections) < 2:
        return ""

    total = len(sections)
    named_ends = total >= 3

    # One phrasing per energy level for the whole caption, rather than a fresh
    # roll per section. Re-rolling per section makes identical sections read as
    # if they differ, and blocks the run-collapsing below.
    chosen = {
        level: rng.choice(PHRASINGS.get(f"energy_{level}", PHRASINGS["energy_steady"]))
        for level in {s.get("energy_label", "steady") for s in sections}
    }

    def name_for(index: int) -> str:
        if named_ends and index == 0:
            return "Intro"
        if named_ends and index == total - 1:
            return "Outro"
        return f"Section {index + 1}"

    # Collapse consecutive sections that say the same thing. A seven-section
    # track that holds one groove should read "Sections 2-6: ..." rather than
    # repeating an identical clause five times.
    runs: list[dict] = []
    for index, section in enumerate(sections):
        phrasing = chosen[section.get("energy_label", "steady")]
        if runs and runs[-1]["phrasing"] == phrasing:
            runs[-1]["last"] = index
            runs[-1]["end"] = section["end"]
        else:
            runs.append(
                {
                    "first": index,
                    "last": index,
                    "start": section["start"],
                    "end": section["end"],
                    "phrasing": phrasing,
                }
            )

    entries = []
    for run in runs:
        first_name = name_for(run["first"])
        if run["first"] == run["last"]:
            label = first_name
        else:
            last_name = name_for(run["last"])
            both_numbered = first_name.startswith("Section") and last_name.startswith(
                "Section"
            )
            if both_numbered:
                label = f"Sections {run['first'] + 1}-{run['last'] + 1}"
            else:
                label = f"{first_name} through {last_name}"

        if mode == "full" and style["section_timings"]:
            entries.append(
                f"{label} ({run['start']:.0f}-{run['end']:.0f}s): {run['phrasing']}"
            )
        else:
            entries.append(f"{label}: {run['phrasing']}")

    return "; ".join(entries)


def _entry_exit(descriptors: dict | None) -> str:
    """Note instrumentation that changes between the start and end of the track.

    Uses the per-window winners CLAP already produced, so it costs nothing and
    describes actual change rather than an assumed song structure.
    """
    detail = _detail(descriptors, "instruments")
    per_window = detail.get("per_window") or []
    if len(per_window) < 3:
        return ""

    opening, closing = per_window[0], per_window[-1]
    if opening == closing:
        return ""
    return f"Opens around {opening} and ends closer to {closing}"


def compose_arrangement(
    analysis: dict, tags: dict, descriptors: dict | None, style: dict, rng
) -> str:
    balance = analysis.get("balance", {})
    tempo = analysis.get("tempo", {})
    structure = analysis.get("structure", {})

    sentences: list[str] = []

    instruments = _labels(descriptors, "instruments", style["max_instruments"])
    roles = _instrument_roles(instruments, balance, rng)
    if roles:
        sentences.append(roles)

    groove = []
    if balance.get("drum_presence"):
        opener = rng.choice(PHRASINGS["mix_opener"])
        groove.append(f"{opener} {balance['drum_presence']}")
    if balance.get("low_end") and style["exact_numbers"]:
        groove.append(f"with a {balance['low_end']}")
    if groove:
        sentences.append(_join(groove, separator=", ", final=", "))

    transition = _entry_exit(descriptors)
    if transition and style["section_detail"] != "none":
        sentences.append(transition)

    walk = _section_walk(structure, descriptors, style, rng)
    if walk:
        sentences.append(f"Section map - {walk}")

    if not sentences:
        return "Arrangement not analysed."
    return _paragraph(sentences)


# --------------------------------------------------------------------------


def compose_caption(
    analysis: dict,
    tags: dict | None = None,
    descriptors: dict | None = None,
    seed: int = 0,
    style: str = DEFAULT_STYLE,
) -> dict:
    """Build the full caption plus its three parts separately.

    Returning the parts as well as the joined text is what lets the Caption
    Splitter node round-trip without re-parsing prose.
    """
    tags = tags or {}
    rng = random.Random(seed)
    style_config = STYLES.get(style, STYLES[DEFAULT_STYLE])

    global_text = compose_global(analysis, tags, descriptors, style_config, rng)
    vocal_text = compose_vocal(analysis, tags, descriptors, style_config, rng)
    arrangement_text = compose_arrangement(
        analysis, tags, descriptors, style_config, rng
    )

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
        "style": style,
    }
