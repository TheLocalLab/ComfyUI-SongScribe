"""Calibrate the per-axis min_z gate against labelled tracks.

Prints, for each axis, the z-score of every label it would emit, so a threshold
can be chosen from what the data does rather than from intuition. For genre and
vocal presence - where ground truth is known - it sweeps min_z and reports how
many correct calls survive versus how many wrong ones are suppressed.

    python tools/calibrate.py <folder> [model]
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from songscribe import audio_io, descriptors  # noqa: E402

# Ground truth. Genre is judged by family: a label counts if it is in the set.
TRUTH = {
    "Ali - Dejavu": {
        "genre": {"contemporary R&B", "neo-soul", "classic soul", "mainstream pop",
                  "indie pop", "funk", "gospel"},
        "voiced": True,
    },
    "DISZ - Rightouesness": {"genre": {"reggae", "dub", "afrobeats"}, "voiced": True},
    "Dalgona - Music": {
        "genre": {"trap", "drill", "boom bap hip-hop", "k-pop", "reggaeton"},
        "voiced": True,
    },
    "Max2buy - Bloodstones": {
        "genre": {"trap", "drill", "boom bap hip-hop", "lo-fi hip-hop"},
        "voiced": True,
    },
    "NAH - One Last Rose": {
        "genre": {"indie rock", "alternative rock", "classic rock", "hard rock",
                  "power pop", "punk rock", "grunge", "heavy metal", "shoegaze"},
        "voiced": True,
    },
    "REXXO - Its Lit": {
        "genre": {"trap", "drill", "boom bap hip-hop", "lo-fi hip-hop"},
        "voiced": True,
    },
}


def main() -> int:
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    model = sys.argv[2] if len(sys.argv) > 2 else "general"

    files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(audio_io.SUPPORTED_EXTENSIONS)
    )
    axes = descriptors.load_vocabularies()

    print(f"model: {descriptors.resolve_model(model)}")
    print(f"tracks: {len(files)}\n")

    # axis -> list of (track, label, z, is_correct_or_None)
    records: dict[str, list] = {a: [] for a in axes}

    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        loaded = audio_io.load_from_path(path)
        samples = loaded.samples_at(descriptors.CLAP_SR)
        embeddings = descriptors._embed_audio(
            descriptors._windows(samples, descriptors.CLAP_SR), model
        )
        truth = TRUTH.get(name, {})

        print(f"--- {name} ---")
        for axis, spec in sorted(axes.items()):
            # Score with the gate wide open so every candidate is visible.
            loose = dict(spec)
            loose["min_z"] = -99.0
            loose["threshold"] = 0.0
            loose["top_k"] = 4
            result = descriptors.score_axis(embeddings, loose, axis, model)

            tops = result["top"] or []
            shown = ", ".join(f"{t['label']} (z={t['z']:.2f})" for t in tops[:3])
            print(f"  {axis:<16} {shown}")

            for rank, item in enumerate(tops):
                correct = None
                if axis == "genre" and truth.get("genre"):
                    correct = item["label"] in truth["genre"]
                if axis == "vocal_presence" and "voiced" in truth:
                    got_voiced = "instrumental" not in item["label"]
                    correct = got_voiced == truth["voiced"]
                records[axis].append((name, item["label"], item["z"], correct, rank))
        print()

    # ---- z distributions per axis
    print("=" * 78)
    print("Z-SCORE DISTRIBUTION (rank-1 label per track)")
    print("=" * 78)
    print(f"{'axis':<16}{'min':>8}{'median':>9}{'max':>8}   current min_z")
    for axis in sorted(records):
        firsts = [r[2] for r in records[axis] if r[4] == 0]
        if not firsts:
            continue
        current = axes[axis].get("min_z", descriptors.DEFAULT_MIN_Z)
        print(
            f"{axis:<16}{min(firsts):>8.2f}{float(np.median(firsts)):>9.2f}"
            f"{max(firsts):>8.2f}   {current}"
        )

    # ---- sweep for the axes with ground truth
    for axis in ("genre", "vocal_presence"):
        judged = [r for r in records[axis] if r[3] is not None and r[4] == 0]
        if not judged:
            continue
        print()
        print("=" * 78)
        print(f"MIN_Z SWEEP - {axis} (rank-1 calls, n={len(judged)})")
        print("=" * 78)
        print(f"{'min_z':>7}{'kept':>7}{'correct':>9}{'wrong':>7}{'silent':>8}  verdict")
        for threshold in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            kept = [r for r in judged if r[2] >= threshold]
            correct = sum(1 for r in kept if r[3])
            wrong = len(kept) - correct
            silent = len(judged) - len(kept)
            note = ""
            if wrong == 0 and correct > 0:
                note = "<- all survivors correct"
            print(
                f"{threshold:>7.1f}{len(kept):>7}{correct:>9}{wrong:>7}{silent:>8}  {note}"
            )

        print("\n  per-track rank-1 calls:")
        for name, label, z, correct, _ in sorted(judged, key=lambda r: -r[2]):
            mark = "OK " if correct else "BAD"
            print(f"    {mark} z={z:>5.2f}  {name[:26]:<28} {label[:44]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
