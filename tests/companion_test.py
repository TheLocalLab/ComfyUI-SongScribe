"""Exercise the phase-4 companion nodes: presets, splitter/composer, lyrics.

    python_embeded\\python.exe ComfyUI-SongScribe\\tests\\companion_test.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from songscribe import compose, lyrics, presets  # noqa: E402
from songscribe.compose import split_caption  # noqa: E402

FAILURES: list[str] = []


def check(ok, label):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)


def test_presets():
    print("\n=== PRESETS ===")
    names = presets.list_presets()
    choices = presets.list_choices()
    print(f"found {len(names)} presets in {len(set(n.split('-')[0] for n in names))} categories")
    check(len(names) >= 30, f"at least 30 presets ship (got {len(names)})")
    check(len(choices) == len(names), "every preset has a unique display name")
    check(
        all("/" in c for c in choices),
        "display names are category-qualified for a readable dropdown",
    )
    check(
        presets.load_choice(choices[0]).get("name") == choices[0],
        "display name round-trips back to its preset",
    )

    for name in names:
        data = presets.load_preset(name)
        composed = presets.to_caption(data, seed=0)
        caption = composed["caption"]
        ok = all(h in caption for h in compose.SECTION_HEADERS)
        check(ok, f"{name} renders all three sections")
        if not ok:
            print(f"      got: {caption[:200]}")

    lofi = presets.load_choice("Hip-Hop / Lo-Fi Chillhop")
    rendered = presets.to_caption(lofi, seed=0)["caption"]
    print(f"\n--- lofi_hiphop ---\n{rendered}\n")
    check("78 BPM" in rendered, "preset tempo reaches the caption")
    check("D flat major" in rendered, "preset key is spelled out")

    # Blending must actually mix, and must be directional.
    techno = presets.load_choice("Electronic / Dark Techno")
    low = presets.to_caption(presets.blend(lofi, techno, 0.0), seed=0)["caption"]
    high = presets.to_caption(presets.blend(lofi, techno, 1.0), seed=0)["caption"]
    mid = presets.to_caption(presets.blend(lofi, techno, 0.5), seed=0)["caption"]
    check(low != high, "blend weight changes the output")
    check("78 BPM" in low, "blend 0.0 keeps the first preset's tempo")
    check("132 BPM" in high, "blend 1.0 takes the second preset's tempo")
    print(f"--- blend 0.5 (lofi x techno) ---\n{mid.splitlines()[0]}\n")

    # Modifiers add without replacing.
    modified = presets.apply_modifiers(lofi, era="1980s", texture="none", mood_shift="darker")
    # Compared case-insensitively: the composer capitalises whichever phrase
    # lands at the start of a sentence.
    text = presets.to_caption(modified, seed=0)["caption"].lower()
    check("dark and brooding" in text, "mood_shift modifier reaches the caption")
    check(
        any(g in text for g in ("lo-fi hip-hop", "chillhop")),
        "modifiers add rather than replace the preset's genre",
    )


def test_split_roundtrip():
    print("\n=== SPLITTER / COMPOSER ===")
    original = (
        "Global Metadata: Lo-fi hip-hop, chillhop. 78 BPM.\n\n"
        "Vocal Details: Soft androgynous vocal.\n\n"
        "Arrangement: Dusty boom-bap drums."
    )
    parts = split_caption(original)
    check(parts["global"].startswith("Lo-fi hip-hop"), "global section extracted")
    check(parts["vocal"].startswith("Soft androgynous"), "vocal section extracted")
    check(parts["arrangement"].startswith("Dusty boom-bap"), "arrangement extracted")

    rebuilt = "\n\n".join(
        f"{h}: {b}"
        for h, b in zip(
            compose.SECTION_HEADERS,
            (parts["global"], parts["vocal"], parts["arrangement"]),
        )
    )
    check(rebuilt == original, "split -> rebuild is lossless")

    # Degenerate inputs must not lose text.
    headerless = split_caption("just some free text with no headers")
    check(
        headerless["global"] == "just some free text with no headers",
        "caption with no headers is preserved, not discarded",
    )
    check(split_caption("") == {"global": "", "vocal": "", "arrangement": ""},
          "empty caption yields empty parts")

    bolded = split_caption("**Global Metadata:** Test.\n\n**Arrangement:** Drums.")
    check(bolded["global"] == "Test.", "markdown-bold headers are handled")
    check(bolded["arrangement"] == "Drums.", "bold arrangement extracted")

    preamble = split_caption("stray note\n\nGlobal Metadata: Real content.")
    check("stray note" in preamble["global"], "text before the first header survives")


def test_lyrics():
    print("\n=== LYRICS ===")
    messy = (
        "(intro)\n"
        "Mmm...\n\n"
        "Verse 1:\n"
        "Midnight and the canvas glows\n"
        "Dragging little wires where the current flows\n\n"
        "{HOOK}\n"
        "Let it render on\n\n"
        "[middle 8]\n"
        "Rain keeps drawing pictures on the glass\n\n"
        "ending:\n"
        "Goodnight"
    )
    result = lyrics.check(messy, target_duration=120.0)
    print(lyrics.format_report(result))

    check(result["sections"] == ["Intro", "Verse", "Chorus", "Bridge", "Outro"],
          f"all five tag styles normalised (got {result['sections']})")
    check("[Chorus]" in result["lyrics"], "{HOOK} became [Chorus]")
    check("[Bridge]" in result["lyrics"], "[middle 8] became [Bridge]")
    check("[Outro]" in result["lyrics"], "bare 'ending:' became [Outro]")
    check(result["syllables"] > 20, "syllables counted")

    # A lyric line ending in a colon must not be mistaken for a tag.
    tricky = lyrics.check("[Intro]\nShe told me this:\nand then she left\n[Outro]\nBye")
    check(tricky["sections"] == ["Intro", "Outro"],
          f"lyric line ending in ':' not treated as a tag (got {tricky['sections']})")

    untagged = lyrics.check("just some words\nwith no structure at all")
    check(len(untagged["warnings"]) >= 1, "untagged lyrics produce a warning")

    # Fit checking in both directions.
    long_lyrics = "[Intro]\n" + "\n".join(["a line of words here"] * 80) + "\n[Outro]\nend"
    tight = lyrics.check(long_lyrics, target_duration=30.0)
    check(
        any("cut or rushed" in w for w in tight["warnings"]),
        "warns when lyrics exceed max_duration",
    )
    roomy = lyrics.check("[Intro]\nhi\n[Outro]\nbye", target_duration=300.0)
    check(
        any("instrumental stretches" in w for w in roomy["warnings"]),
        "warns when lyrics underfill max_duration",
    )


def main() -> int:
    test_presets()
    test_split_roundtrip()
    test_lyrics()

    print()
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}): {'; '.join(FAILURES)}")
        return 1
    print("Companion nodes OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
