"""Exercise the Whisper lyric fallback.

With a folder argument it transcribes real songs and, where a sibling .txt
holds the true lyrics, reports rough word overlap so the quality claim in the
README is grounded rather than asserted.

    python_embeded\\python.exe ComfyUI-SongScribe\\tests\\transcribe_test.py [folder]
"""

from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lyrics are frequently not ASCII - accents, CJK - and the Windows console
# defaults to cp1252, which raises rather than degrading.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from songscribe import audio_io, lyrics as lyrics_engine, tags, transcribe  # noqa: E402

FAILURES: list[str] = []


def check(ok, label):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)


def test_structure_offline():
    """The tagging logic is pure and testable without running Whisper."""
    print("=== STRUCTURE (synthetic segments) ===")

    segments = [
        {"start": 5.0, "end": 9.0, "text": "you give me deja vu"},
        {"start": 9.2, "end": 13.0, "text": "like I loved you once"},
        # long gap -> instrumental
        {"start": 30.0, "end": 34.0, "text": "yeah you stole my heart"},
        {"start": 34.2, "end": 38.0, "text": "didnt take you long"},
        # repeat of the first block -> chorus
        {"start": 50.0, "end": 54.0, "text": "you give me deja vu"},
        {"start": 54.2, "end": 58.0, "text": "like I loved you once"},
    ]
    out = transcribe.structure(segments, duration=70.0)
    print(out)
    print()

    check("[Intro]" in out, "leading silence produces [Intro]")
    check("[Instrumental]" in out, "long vocal gap produces [Instrumental]")
    check(out.count("[Chorus]") == 2, f"repeated block tagged [Chorus] twice (got {out.count('[Chorus]')})")
    check("[Verse]" in out, "non-repeating block tagged [Verse]")

    empty = transcribe.structure([], duration=10.0)
    check(empty == "", "no segments yields empty lyrics, not a stray tag")

    # A song transcribed as one continuous block must not be labelled [Outro];
    # the closing section only exists relative to something before it.
    single = transcribe.structure(
        [{"start": 1.0, "end": 90.0, "text": "one long continuous vocal take"}],
        duration=95.0,
    )
    check("[Outro]" not in single, f"single-block song not tagged [Outro] (got {single[:40]!r})")
    check("[Verse]" in single, "single-block song tagged [Verse]")

    # Output must survive the phase-4 validator.
    validated = lyrics_engine.check(out)
    check(validated["section_count"] >= 4, "generated tags parse back out")
    check(not validated["unknown_tags"], "generated tags are all canonical")


def test_sidecar_guard():
    print("\n=== SIDECAR GUARD ===")
    prose = (
        "Style - Contemporary R&B and Pop track at 95 BPM in the key of G Major, "
        "the arrangement features a syncopated electric bass line with a rounded "
        "warm tone and a drum kit with a crisp snare and tight kick, a clean "
        "electric guitar plays rhythmic staccato chords on the off-beats."
    )
    check(not tags.looks_like_lyrics(prose), "long prose rejected as lyrics")

    real = "[Intro]\nMmm...\n\n[Verse]\nMidnight and the canvas glows\nDragging little wires"
    check(tags.looks_like_lyrics(real), "tagged lyrics accepted")

    plain = "the moon is low in the july sky\ntonight I leave you behind\nbut I wont say goodbye"
    check(tags.looks_like_lyrics(plain), "untagged short-line lyrics accepted")

    check(not tags.looks_like_lyrics("one line only"), "single line rejected")


def word_overlap(produced: str, reference: str) -> float:
    def words(text):
        text = re.sub(r"\[[^\]]*\]", " ", text)
        return set(re.findall(r"[a-z']{3,}", text.lower()))

    a, b = words(produced), words(reference)
    if not a or not b:
        return 0.0
    return len(a & b) / len(b)


def test_real_audio(folder: str):
    print(f"\n=== REAL AUDIO ({folder}) ===")
    files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(audio_io.SUPPORTED_EXTENSIONS)
    )[:3]

    if not files:
        print("no audio found")
        return

    for path in files:
        name = os.path.basename(path)
        loaded = audio_io.load_from_path(path)
        started = time.perf_counter()
        result = transcribe.transcribe_to_lyrics(
            loaded.samples_at(transcribe.WHISPER_SR),
            duration=loaded.duration,
            model_size="base",
        )
        elapsed = time.perf_counter() - started

        ratio = elapsed / max(loaded.duration, 1)
        print(f"\n--- {name} ---")
        print(
            f"  {loaded.duration:.0f}s audio in {elapsed:.0f}s "
            f"({ratio:.2f}x realtime), language={result['language']} "
            f"p={result['language_probability']}"
        )

        truth_path = os.path.splitext(path)[0] + ".txt"
        if os.path.isfile(truth_path):
            with open(truth_path, "r", encoding="utf-8", errors="replace") as fh:
                truth = fh.read()
            if "Lyrics -" in truth:
                truth = truth.split("Lyrics -", 1)[1]
            overlap = word_overlap(result["lyrics"], truth)
            print(f"  word overlap with true lyrics: {overlap:.0%}")

        preview = result["lyrics"][:400]
        print("  " + preview.replace("\n", "\n  "))


def main() -> int:
    test_structure_offline()
    test_sidecar_guard()

    if not transcribe.is_available():
        print("\nfaster-whisper not installed; skipping real-audio pass")
    elif len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        test_real_audio(sys.argv[1])
    else:
        print("\n(no folder given; skipping real-audio pass)")

    print()
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}): {'; '.join(FAILURES)}")
        return 1
    print("Transcription OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
