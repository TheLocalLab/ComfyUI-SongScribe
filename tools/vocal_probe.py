"""Find a vocal-presence prompt set that both CLAP checkpoints agree on.

A wrong "instrumental" is the worst error this pack can make: it gates the
entire Vocal Details section, so the caption instructs the music model to
produce no vocals at all. It is worth more effort than any other axis.

Tests several phrasings, including prompt ensembles, against tracks whose
vocal status is known.

    python tools/vocal_probe.py <folder> [model ...]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from songscribe import audio_io, descriptors  # noqa: E402

# Every track here has vocals; several are rapped, which is the case the
# current wording gets wrong.
VOICED = {
    "Ali - Dejavu": True,
    "DISZ - Rightouesness": True,
    "Dalgona - Music": True,
    "Max2buy - Bloodstones": True,
    "NAH - One Last Rose": True,
    "REXXO - Its Lit": True,
}

# Candidate label sets. Each maps label -> outcome.
CANDIDATES = {
    "current": {
        "prompts": ["{label}"],
        "labels": {
            "an instrumental track with no singing at all": "instrumental",
            "a song with lead singing vocals": "sung",
            "a track with spoken word or rapping": "spoken",
            "a track with wordless humming and vocal textures": "wordless",
        },
    },
    "explicit_voice": {
        "prompts": ["{label}"],
        "labels": {
            "instrumental music with no human voice anywhere": "instrumental",
            "music with a human voice singing words": "sung",
            "music with a human voice rapping words": "spoken",
            "music with wordless vocal humming and no lyrics": "wordless",
        },
    },
    "ensemble": {
        "prompts": ["{label}", "a recording of {label}", "this is {label}"],
        "labels": {
            "instrumental music with no human voice anywhere": "instrumental",
            "music with a human voice singing words": "sung",
            "music with a human voice rapping words": "spoken",
            "music with wordless vocal humming and no lyrics": "wordless",
        },
    },
    # Two-stage: decide voice/no-voice first, which is an easier question than
    # simultaneously deciding sung vs rapped vs hummed.
    "binary_first": {
        "prompts": ["{label}", "a recording of {label}"],
        "labels": {
            "instrumental music with no singer and no voice": "instrumental",
            "music with a singer, rapper or human voice": "voiced",
        },
    },
}


def run(name: str, spec_data: dict, embeddings, model: str) -> tuple[str, float]:
    labels = list(spec_data["labels"])
    spec = {
        "labels": labels,
        "prompt": "{label}",
        "prompts": spec_data["prompts"],
        "top_k": 1,
        "threshold": 0.0,
        "min_z": -99.0,
        "temperature": 0.1,
        "mode": "exclusive",
        "outcomes": {k: v for k, v in spec_data["labels"].items()},
    }
    result = descriptors.score_axis(embeddings, spec, f"probe_{name}", model)
    top = result["top"][0] if result["top"] else {"label": "?", "z": 0.0}
    return spec_data["labels"].get(top["label"], "?"), top["z"]


def main() -> int:
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    models = sys.argv[2:] or ["general", "music_and_speech"]

    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(audio_io.SUPPORTED_EXTENSIONS)
    ]

    signals = {}
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        if name in VOICED:
            loaded = audio_io.load_from_path(path)
            signals[name] = loaded.samples_at(descriptors.CLAP_SR)
    print(f"{len(signals)} labelled track(s)\n")

    totals: dict[tuple[str, str], int] = {}

    for model in models:
        print("=" * 74)
        print(model)
        print("=" * 74)
        embeddings = {
            name: descriptors._embed_audio(
                descriptors._windows(samples, descriptors.CLAP_SR), model
            )
            for name, samples in signals.items()
        }

        for candidate, data in CANDIDATES.items():
            hits = 0
            details = []
            for name, emb in embeddings.items():
                outcome, z = run(candidate, data, emb, model)
                voiced = outcome not in ("instrumental", "?")
                ok = voiced == VOICED[name]
                hits += ok
                details.append(f"{'ok' if ok else 'XX'} {name[:18]}={outcome}")
            totals[(model, candidate)] = hits
            print(f"  {candidate:<16} {hits}/{len(embeddings)}   " + "  ".join(details[:3]))
            print(f"  {'':<16}       " + "  ".join(details[3:]))
        print()

    print("=" * 74)
    print("SUMMARY (voiced/instrumental correctness)")
    print("=" * 74)
    print(f"{'candidate':<18}" + "".join(f"{m:>22}" for m in models) + f"{'total':>8}")
    for candidate in CANDIDATES:
        row = [totals.get((m, candidate), 0) for m in models]
        n = len(signals)
        print(
            f"{candidate:<18}"
            + "".join(f"{f'{v}/{n}':>22}" for v in row)
            + f"{sum(row):>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
