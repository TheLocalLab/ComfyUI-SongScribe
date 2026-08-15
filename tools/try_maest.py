"""Evaluate MAEST (supervised Discogs-style tagger) against CLAP zero-shot.

MAEST is trained on 400 Discogs styles rather than matching free text, so it
should be strictly better at genre than a zero-shot model that has never seen
"reggae" as a label.

SECURITY NOTE: MAEST ships custom modelling code and requires
trust_remote_code=True, which executes code from the model repository. That is
a deliberate trust decision about mtg-upf (Music Technology Group, UPF
Barcelona - the Essentia authors), not something to enable casually. This
script is an investigation; whether the node ships it is a separate call.

    python tools/try_maest.py <folder>
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from songscribe import audio_io  # noqa: E402

MODEL = "mtg-upf/discogs-maest-10s-pw-129e"
MAEST_SR = 16000

TRUTH = {
    "Ali - Dejavu": "R&B / pop",
    "DISZ - Rightouesness": "reggae",
    "Dalgona - Music": "hip-hop / k-pop",
    "Max2buy - Bloodstones": "hip-hop",
    "NAH - One Last Rose": "rock",
    "REXXO - Its Lit": "hip-hop / trap",
}


def main() -> int:
    folder = sys.argv[1] if len(sys.argv) > 1 else "."

    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        print(f"transformers/torch unavailable: {exc}")
        return 2

    print(f"loading {MODEL} (trust_remote_code=True)...")
    started = time.perf_counter()
    try:
        classifier = pipeline(
            "audio-classification",
            model=MODEL,
            trust_remote_code=True,
            device=-1,  # CPU, consistent with the rest of the pack
            top_k=5,
        )
    except Exception as exc:
        print(f"FAILED to load MAEST: {type(exc).__name__}: {exc}")
        print("\nThis is the outcome that matters - if it will not load on this")
        print("stack, the supervised route is not available without more work.")
        return 1
    print(f"loaded in {time.perf_counter() - started:.0f}s\n")

    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(audio_io.SUPPORTED_EXTENSIONS)
    ]

    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        loaded = audio_io.load_from_path(path)
        samples = loaded.samples_at(MAEST_SR)

        started = time.perf_counter()
        try:
            predictions = classifier({"raw": samples, "sampling_rate": MAEST_SR})
        except Exception as exc:
            print(f"{name}: inference failed - {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - started

        print(f"--- {name}   (truth: {TRUTH.get(name, '?')})   {elapsed:.1f}s")
        for prediction in predictions:
            print(f"    {prediction['score']:.3f}  {prediction['label']}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
