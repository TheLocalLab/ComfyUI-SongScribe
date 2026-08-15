"""Compare CLAP checkpoints on a folder of labelled songs.

Genre labels in a generation prompt are far more trustworthy than tempo labels
- a track prompted "reggae" does sound like reggae, whereas a requested BPM may
simply not have been honoured - so genre is scored against the label here while
tempo is deliberately not.

    python_embeded\\python.exe ComfyUI-SongScribe\\tests\\compare_models.py <folder>
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from songscribe import audio_io, descriptors  # noqa: E402

# Genre families the labels fall into, and the vocabulary entries that would
# count as a hit for each. Judged generously: any label in the family counts.
FAMILIES = {
    "rnb_pop": {
        "contemporary R&B", "neo-soul", "classic soul", "mainstream pop",
        "indie pop", "funk", "gospel",
    },
    "reggae": {"reggae", "dub", "afrobeats"},
    "hiphop": {
        "trap", "drill", "boom bap hip-hop", "lo-fi hip-hop", "chillhop",
        "reggaeton",
    },
    "rock": {
        "indie rock", "alternative rock", "classic rock", "hard rock",
        "power pop", "punk rock", "grunge", "emo", "heavy metal", "shoegaze",
    },
}

TRUTH = {
    "Ali - Dejavu": "rnb_pop",
    "DISZ - Rightouesness": "reggae",
    "Dalgona - Music": "hiphop",
    "Max2buy - Bloodstones": "hiphop",
    "NAH - One Last Rose": "rock",
}

VOCAL_TRUTH = {
    "Ali - Dejavu": "voiced",
    "DISZ - Rightouesness": "voiced",
    "Dalgona - Music": "voiced",
    "Max2buy - Bloodstones": "voiced",
    "NAH - One Last Rose": "voiced",
}


def main() -> int:
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(audio_io.SUPPORTED_EXTENSIONS)
    )
    if not files:
        print(f"no audio in {folder}")
        return 2

    # Decode once; every model scores the identical signal.
    print("decoding audio...")
    signals = {}
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        loaded = audio_io.load_from_path(path)
        signals[name] = loaded.samples_at(descriptors.CLAP_SR)
    print(f"{len(signals)} track(s) ready\n")

    summary: dict[str, dict] = {}

    for key in ("general", "music", "music_and_speech"):
        model_id = descriptors.MODELS[key]
        print("=" * 78)
        print(f"{key}  ({model_id})")
        print("=" * 78)

        genre_hits = 0
        genre_scored = 0
        vocal_hits = 0
        vocal_scored = 0

        for name, samples in signals.items():
            try:
                result = descriptors.describe(samples, model_id=model_id)
            except Exception as exc:
                print(f"  {name}: FAILED {exc}")
                continue

            genres = [g["label"] for g in result.get("genre", [])]
            presence = result.get("vocal_presence")

            family = TRUTH.get(name)
            verdict = "-"
            if family:
                genre_scored += 1
                accepted = FAMILIES[family]
                hit = any(g in accepted for g in genres)
                genre_hits += hit
                verdict = "HIT " if hit else "miss"

            if VOCAL_TRUTH.get(name):
                vocal_scored += 1
                voiced = presence != "instrumental"
                vocal_hits += voiced == (VOCAL_TRUTH[name] == "voiced")

            print(
                f"  {verdict} {name[:26]:<28} {', '.join(genres) or '(none)':<38}"
                f" [{presence}]"
            )

        summary[key] = {
            "genre": (genre_hits, genre_scored),
            "vocal": (vocal_hits, vocal_scored),
        }
        print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'checkpoint':<20}{'genre family':>16}{'vocal presence':>18}")
    for key, data in summary.items():
        g, gt = data["genre"]
        v, vt = data["vocal"]
        print(f"{key:<20}{f'{g}/{gt}':>16}{f'{v}/{vt}':>18}")

    best = max(summary.items(), key=lambda kv: kv[1]["genre"][0])
    print(f"\nBest on genre: {best[0]} ({best[1]['genre'][0]}/{best[1]['genre'][1]})")
    print(f"Current default: {descriptors.DEFAULT_MODEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
