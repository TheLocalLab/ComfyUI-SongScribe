"""Style presets.

Presets are structured, not prose. Each one declares the same fields the
analyzer produces - genre, mood, instruments, production and so on - which
means a preset can be fed through the exact same composer as a real analysis.
One grammar, one set of tests, and presets and analysed tracks come out
speaking the same language.

It also makes blending meaningful: merging two structured presets is a list
operation, whereas blending two paragraphs of prose is not well defined.
"""

from __future__ import annotations

import os

from . import compose
from .keys import spoken_key

PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

# Fields that hold descriptor lists and can therefore be blended.
LIST_FIELDS = (
    "genre",
    "mood",
    "scene",
    "production",
    "instruments",
    "vocal_timbre",
    "vocal_delivery",
)

# Optional single-line modifier axes, applied on top of any preset.
MODIFIERS = {
    "era": {
        "none": {},
        "1960s": {"production": ["a raw live-room recording", "a narrow mono-leaning image"]},
        "1970s": {"production": ["warm analog tape saturation"]},
        "1980s": {"production": ["a wide spacious stereo image", "long cavernous hall reverb"]},
        "1990s": {"production": ["a crisp studio multitrack production"]},
        "2000s": {"production": ["a loud heavily compressed master"]},
        "modern": {"production": ["a clean modern polished mix"]},
    },
    "texture": {
        "none": {},
        "lo-fi": {"production": ["a muddy lo-fi bedroom mix", "heavy vinyl crackle and surface noise"]},
        "tape": {"production": ["tape hiss and wow-flutter pitch wobble"]},
        "hi-fi": {"production": ["a bright airy hi-fi mix"]},
        "intimate": {"production": ["a dry close-mic'd intimate sound"]},
        "cavernous": {"production": ["heavy reverb drenching everything"]},
    },
    "mood_shift": {
        "none": {},
        "darker": {"mood": ["dark and brooding"]},
        "brighter": {"mood": ["bright and joyful"]},
        "sadder": {"mood": ["melancholy and wistful"]},
        "dreamier": {"mood": ["hazy and drifting"]},
        "harder": {"mood": ["energetic and driving"]},
    },
}


class PresetError(RuntimeError):
    pass


def list_presets(preset_dir: str = PRESET_DIR) -> list[str]:
    """Filename stems, e.g. 'world-reggae'."""
    if not os.path.isdir(preset_dir):
        return []
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(preset_dir)
        if name.endswith((".yaml", ".yml"))
    )


def preset_index(preset_dir: str = PRESET_DIR) -> dict[str, str]:
    """Map display name -> filename stem.

    The dropdown shows "World / Reggae" rather than "world-reggae"; with three
    dozen entries the readable form is the difference between a usable menu and
    a wall of slugs. A preset with no `name:` falls back to its stem, so a
    hand-dropped file still appears.
    """
    import yaml

    index: dict[str, str] = {}
    for stem in list_presets(preset_dir):
        display = stem
        for extension in (".yaml", ".yml"):
            path = os.path.join(preset_dir, stem + extension)
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        data = yaml.safe_load(fh) or {}
                    display = str(data.get("name") or stem)
                except Exception:
                    pass
                break
        # Collisions would make one preset unreachable, so disambiguate.
        if display in index:
            display = f"{display} ({stem})"
        index[display] = stem
    return index


def list_choices(preset_dir: str = PRESET_DIR) -> list[str]:
    """Display names, sorted so categories group together."""
    return sorted(preset_index(preset_dir))


def load_choice(display: str, preset_dir: str = PRESET_DIR) -> dict:
    """Load by display name, falling back to treating it as a stem."""
    stem = preset_index(preset_dir).get(display, display)
    return load_preset(stem, preset_dir)


def load_preset(name: str, preset_dir: str = PRESET_DIR) -> dict:
    import yaml

    for extension in (".yaml", ".yml"):
        path = os.path.join(preset_dir, name + extension)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            data.setdefault("name", name)
            return data
    raise PresetError(f"preset not found: {name}")


def blend(primary: dict, secondary: dict, weight: float) -> dict:
    """Merge two presets. `weight` is how much of the secondary to admit (0-1).

    Interleaves each list field, taking proportionally more from the secondary
    as weight rises. Scalars (bpm, key) cross over at the halfway point rather
    than being averaged - the mean of 78 and 174 BPM is a tempo neither preset
    asked for.
    """
    weight = max(0.0, min(1.0, float(weight)))
    result = dict(primary)
    result["name"] = f"{primary.get('name', '?')} x {secondary.get('name', '?')}"

    for field in LIST_FIELDS:
        first = list(primary.get(field) or [])
        second = list(secondary.get(field) or [])
        if not second:
            result[field] = first
            continue
        if not first:
            result[field] = second
            continue

        total = max(len(first), len(second))
        take_second = int(round(total * weight))
        take_first = total - take_second

        merged = first[:take_first] + second[:take_second]
        # Preserve order of first appearance while removing duplicates.
        seen = set()
        result[field] = [x for x in merged if not (x in seen or seen.add(x))]

    if weight >= 0.5:
        for scalar in ("bpm", "key", "vocal_presence"):
            if secondary.get(scalar) is not None:
                result[scalar] = secondary[scalar]

    return result


def apply_modifiers(preset: dict, **choices) -> dict:
    """Layer modifier axes onto a preset. Modifiers add, never replace."""
    result = {k: (list(v) if isinstance(v, list) else v) for k, v in preset.items()}

    for axis, choice in choices.items():
        table = MODIFIERS.get(axis)
        if not table or not choice or choice == "none":
            continue
        for field, additions in (table.get(choice) or {}).items():
            existing = list(result.get(field) or [])
            # Modifier phrases go first: they are the explicit user choice,
            # and the composer's dedupe keeps whichever it sees first.
            result[field] = additions + [x for x in existing if x not in additions]

    return result


def _as_descriptors(preset: dict) -> dict:
    """Shape a preset like the CLAP scorer's output."""
    descriptors: dict = {}
    for field in LIST_FIELDS:
        descriptors[field] = [{"label": label, "score": 1.0} for label in (preset.get(field) or [])]
    if preset.get("vocal_presence"):
        descriptors["vocal_presence"] = preset["vocal_presence"]
    return descriptors


def _as_analysis(preset: dict) -> dict:
    """Shape a preset like the DSP analyser's output.

    Only fields the preset actually declares are populated; the composer omits
    any clause whose inputs are missing, so an underspecified preset produces a
    shorter caption rather than an invented one.
    """
    analysis: dict = {"tempo": {}, "key": {}, "timbre": {}, "loudness": {}, "balance": {}, "structure": {}}

    bpm = preset.get("bpm")
    if bpm:
        analysis["tempo"] = {"bpm_int": int(bpm), "bpm": float(bpm)}
        if preset.get("feel"):
            analysis["tempo"]["feel"] = preset["feel"]
        if preset.get("swing"):
            analysis["tempo"]["swing"] = float(preset["swing"])

    key = preset.get("key")
    if key:
        parts = str(key).split()
        root = parts[0]
        mode = parts[1] if len(parts) > 1 else "major"
        analysis["key"] = {
            "name": f"{root} {mode}",
            "spoken": f"{spoken_key(root)} {mode}",
            "mode": mode,
            # Declared by a human, so it is not a guess needing a confidence.
            "confidence": 1.0,
        }
        if preset.get("harmony"):
            analysis["key"]["harmonic_complexity"] = preset["harmony"]

    if preset.get("low_end"):
        analysis["balance"]["low_end"] = preset["low_end"]
    if preset.get("drums"):
        analysis["balance"]["drum_presence"] = preset["drums"]

    return analysis


def to_caption(preset: dict, seed: int = 0, style: str = compose.DEFAULT_STYLE) -> dict:
    """Render a preset through the same composer used for analysed audio."""
    return compose.compose_caption(
        _as_analysis(preset),
        tags={},
        descriptors=_as_descriptors(preset),
        seed=seed,
        style=style,
    )
