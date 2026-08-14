"""Compare the caption at each style level, and check the dial actually works.

    python_embeded\\python.exe ComfyUI-SongScribe\\tests\\compose_test.py [audio]
"""

from __future__ import annotations

import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from songscribe import audio_io, compose, descriptors, features, tags  # noqa: E402


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        tempfile.gettempdir(), "songscribe_smoke.wav"
    )
    if not os.path.isfile(path):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import soundfile as sf

        from smoke_test import SR, synthesise  # type: ignore

        sf.write(path, synthesise(), SR)

    loaded = audio_io.load_from_path(path)
    analysis = features.analyse(loaded)
    file_tags = tags.read_tags(path)

    scored = None
    if descriptors.is_available():
        try:
            scored = descriptors.describe(loaded.samples_at(descriptors.CLAP_SR))
        except Exception as exc:
            print(f"(descriptors unavailable: {exc})")

    captions = {}
    for style in compose.STYLES:
        result = compose.compose_caption(
            analysis, file_tags, scored, seed=0, style=style
        )
        captions[style] = result
        print(f"\n{'=' * 70}\n{style.upper()}  ({len(result['caption'])} chars)\n{'=' * 70}")
        print(result["caption"])

    print(f"\n{'=' * 70}\nCHECKS\n{'=' * 70}")
    failures = []

    def check(ok, label):
        print(f"{'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            failures.append(label)

    verbatim = captions["verbatim"]["caption"]
    balanced = captions["balanced"]["caption"]
    loose = captions["loose"]["caption"]

    # The dial has to actually reduce specificity, monotonically.
    check(
        len(verbatim) > len(loose),
        f"verbatim ({len(verbatim)}) longer than loose ({len(loose)})",
    )
    check(
        len(verbatim) >= len(balanced) >= len(loose),
        "length decreases monotonically across styles",
    )

    bpm = analysis["tempo"]["bpm_int"]
    check(f"{bpm} BPM" in verbatim, f"verbatim states exact tempo ({bpm} BPM)")
    check(f"{bpm} BPM" in balanced, "balanced states exact tempo")
    check("BPM" not in loose, "loose omits BPM entirely")

    # Second-level timings are the strongest clone signal; only verbatim keeps them.
    timing = re.compile(r"\d+-\d+s")
    check(bool(timing.search(verbatim)), "verbatim includes section timings")
    check(not timing.search(balanced), "balanced drops section timings")
    check(not timing.search(loose), "loose drops section timings")

    for style, result in captions.items():
        has_all = all(h in result["caption"] for h in compose.SECTION_HEADERS)
        check(has_all, f"{style} has all three section headers")
        parts_ok = all(
            result[k] for k in ("global", "vocal", "arrangement")
        )
        check(parts_ok, f"{style} exposes all three parts separately")

    # Seeded phrasing must be reproducible, or cached runs would drift.
    again = compose.compose_caption(analysis, file_tags, scored, seed=0, style="balanced")
    check(again["caption"] == balanced, "same seed reproduces identical caption")

    # A seed that changes nothing would be a lie in the node's tooltip.
    varied = compose.compose_caption(analysis, file_tags, scored, seed=7, style="balanced")
    check(varied["caption"] != balanced, "different seed produces different phrasing")
    print(f"\n  seed 7 balanced:\n  {varied['caption'].splitlines()[0][:200]}\n")

    # No conjunction pile-ups from phrase labels that already contain "and".
    for style, result in captions.items():
        check(
            " and and " not in result["caption"]
            and not re.search(r"\band\b[^.]{0,40}\band\b[^.]{0,15}\band\b",
                              result["caption"]),
            f"{style} has no conjunction pile-up",
        )

    print()
    if failures:
        print(f"FAILURES: {', '.join(failures)}")
        return 1
    print("Composer OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
