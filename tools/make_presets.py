"""Generate the shipped preset library.

Presets are plain YAML and meant to be hand-edited; this script exists so the
*shipped* set stays internally consistent - same field names, same vocabulary
phrasing, same level of detail - rather than drifting as entries are added by
hand over time. Editing a preset afterwards is entirely expected.

Vocabulary note: production/instrument/mood/scene values are drawn from
songscribe/vocab/*.yaml wherever possible, so a preset and an analysed track
describe the same thing with the same words.

    python tools/make_presets.py
"""

from __future__ import annotations

import os
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "songscribe", "presets")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from presets_core import P  # noqa: E402
from presets_yue2 import P2  # noqa: E402

# The YuE2-taxonomy wave lives in its own module purely to keep this
# table readable; both halves share one schema and one emitter.
P = P + P2


def quote(value: str) -> str:
    return f'"{value}"' if any(c in value for c in ":#") else value


def emit(entry) -> str:
    (category, slug, name, bpm, key, harmony, feel, drums, low_end,
     genre, mood, scene, production, instruments, vocals) = entry
    presence, timbre, delivery = vocals

    lines = [
        f"name: {name}",
        f"category: {category}",
        f"bpm: {bpm}",
        f"key: {key}",
        f"harmony: {harmony}",
        f"feel: {feel}",
        f"drums: {drums}",
        f"low_end: {low_end}",
    ]
    for field, values in (
        ("genre", genre), ("mood", mood), ("scene", scene),
        ("production", production), ("instruments", instruments),
    ):
        lines.append(f"{field}:")
        lines.extend(f"  - {quote(v)}" for v in values)

    lines.append(f"vocal_presence: {presence}")
    if presence != "instrumental":
        for field, values in (("vocal_timbre", timbre), ("vocal_delivery", delivery)):
            if values:
                lines.append(f"{field}:")
                lines.extend(f"  - {quote(v)}" for v in values)

    return "\n".join(lines) + "\n"


def main() -> int:
    os.makedirs(OUT, exist_ok=True)

    # Remove the previous generation so renamed entries do not linger.
    for existing in os.listdir(OUT):
        if existing.endswith((".yaml", ".yml")):
            os.remove(os.path.join(OUT, existing))

    for entry in P:
        category, slug = entry[0], entry[1]
        path = os.path.join(OUT, f"{category}-{slug}.yaml")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(emit(entry))

    print(f"wrote {len(P)} presets to {OUT}")
    categories: dict[str, int] = {}
    for entry in P:
        categories[entry[0]] = categories.get(entry[0], 0) + 1
    for category, count in sorted(categories.items()):
        print(f"  {category:<12} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
