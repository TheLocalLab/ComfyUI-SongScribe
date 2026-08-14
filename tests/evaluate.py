"""Score SongScribe against a folder of labelled songs.

Expects each track as an audio file plus a sibling .txt holding a "Style - ..."
line with the true BPM, key and genre. Reports per-axis accuracy rather than a
single number, because the DSP and the descriptor layer fail in very different
ways and should be judged separately.

    python_embeded\\python.exe ComfyUI-SongScribe\\tests\\evaluate.py <folder>
"""

from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from songscribe import audio_io, compose, descriptors, features, tags  # noqa: E402

_BPM = re.compile(r"(\d{2,3})\s*BPM", re.IGNORECASE)
_KEY_OF = re.compile(
    r"key of\s+([A-G][#b♯♭]?)\s*(major|minor)?", re.IGNORECASE
)
_KEY_LOOSE = re.compile(r"\b([A-G][#b]?)\s+(major|minor)\b", re.IGNORECASE)
_MODE_ONLY = re.compile(r"\b(major|minor)\s+key", re.IGNORECASE)

_FEMALE = re.compile(r"\bfemale\b", re.IGNORECASE)
_MALE = re.compile(r"\b(male|tenor|baritone)\b", re.IGNORECASE)
_RAP = re.compile(r"\b(rap|hip ?hop|mc|flow|cadence)\b", re.IGNORECASE)
_INSTRUMENTAL = re.compile(r"\binstrumental\b", re.IGNORECASE)


def parse_truth(path: str) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    # Only the style header carries the labels; the lyrics below it are full of
    # words like "major" and "flow" that would produce false matches.
    head = text.split("Lyrics -")[0] if "Lyrics -" in text else text[:1500]

    truth: dict = {"style_text": head.strip()}

    bpm = _BPM.search(head)
    if bpm:
        truth["bpm"] = int(bpm.group(1))

    key_match = _KEY_OF.search(head) or _KEY_LOOSE.search(head)
    if key_match:
        root = key_match.group(1).replace("♯", "#").replace("♭", "b")
        mode = (key_match.group(2) or "major").lower()
        truth["key"] = f"{root.upper()[0]}{root[1:]} {mode}"
        truth["mode"] = mode
    else:
        mode_only = _MODE_ONLY.search(head)
        if mode_only:
            truth["mode"] = mode_only.group(1).lower()

    # "instrumental" is checked last and only when nothing names a singer:
    # style descriptions routinely say things like "a synth lead enters during
    # the instrumental sections" about a track that very much has vocals.
    if _RAP.search(head):
        truth["vocal"] = "rap"
    elif _FEMALE.search(head):
        truth["vocal"] = "female"
    elif _MALE.search(head):
        truth["vocal"] = "male"
    elif _INSTRUMENTAL.search(head):
        truth["vocal"] = "instrumental"

    truth["female"] = bool(_FEMALE.search(head))
    truth["male"] = bool(_MALE.search(head))
    return truth


def bpm_verdict(measured: float, expected: int) -> tuple[str, str]:
    if expected is None:
        return "-", "no label"
    for multiplier, note in ((1.0, "exact"), (2.0, "half-time"), (0.5, "double-time")):
        if abs(measured - expected * multiplier) <= max(1.5, expected * 0.02):
            return ("OK" if multiplier == 1.0 else "OCTAVE"), note
    return "MISS", f"off by {measured - expected:+.1f}"


def key_verdict(measured: dict, truth: dict) -> tuple[str, str]:
    expected = truth.get("key")
    if not expected:
        mode = truth.get("mode")
        if mode:
            return ("OK" if measured.get("mode") == mode else "MISS"), f"mode only ({mode})"
        return "-", "no label"

    got = measured.get("name", "")
    if got.lower() == expected.lower():
        return "OK", "exact"
    # A key and its relative share every pitch class, so this is the expected
    # near-miss rather than a random error, and worth counting separately.
    if measured.get("relative", "").lower() == expected.lower():
        return "RELATIVE", f"got {got}, relative of {expected}"
    if measured.get("mode") == truth.get("mode"):
        return "MISS", f"got {got}, right mode"
    return "MISS", f"got {got}"


def main() -> int:
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    if not os.path.isdir(folder):
        print(f"not a folder: {folder}")
        return 2

    audio_files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(audio_io.SUPPORTED_EXTENSIONS)
    )
    if not audio_files:
        print(f"no audio found in {folder}")
        return 2

    print(f"Evaluating {len(audio_files)} track(s) from {folder}\n")

    rows = []
    for path in audio_files:
        name = os.path.splitext(os.path.basename(path))[0]
        truth_path = os.path.splitext(path)[0] + ".txt"
        truth = parse_truth(truth_path) if os.path.isfile(truth_path) else {}

        print(f"--- {name} ---")
        started = time.perf_counter()
        loaded = audio_io.load_from_path(path)
        analysis = features.analyse(loaded)
        dsp_time = time.perf_counter() - started

        scored = None
        clap_time = 0.0
        try:
            started = time.perf_counter()
            scored = descriptors.describe(loaded.samples_at(descriptors.CLAP_SR))
            clap_time = time.perf_counter() - started
        except Exception as exc:
            print(f"  descriptor scoring failed: {exc}")

        bpm_status, bpm_note = bpm_verdict(
            analysis["tempo"]["bpm"], truth.get("bpm")
        )
        key_status, key_note = key_verdict(analysis["key"], truth)

        genre = [g["label"] for g in (scored or {}).get("genre", [])]
        presence = (scored or {}).get("vocal_presence")
        timbre = [t["label"] for t in (scored or {}).get("vocal_timbre", [])]
        instruments = [i["label"] for i in (scored or {}).get("instruments", [])]

        print(f"  duration   {analysis['duration_str']}  ({dsp_time:.1f}s DSP, {clap_time:.1f}s CLAP)")
        print(f"  BPM        {analysis['tempo']['bpm']:6.1f}  vs {truth.get('bpm', '?'):>5}   [{bpm_status}] {bpm_note}")
        print(f"  key        {analysis['key']['name']:<10} vs {truth.get('key', truth.get('mode', '?')):<10} [{key_status}] {key_note}")
        print(f"  conf       {analysis['key']['confidence']}  (rel margin {analysis['key'].get('relative_margin')})")
        print(f"  vocal      {presence}  vs label '{truth.get('vocal', '?')}'")
        if timbre:
            print(f"  timbre     {', '.join(timbre)}")
        print(f"  genre      {', '.join(genre) if genre else '(none above threshold)'}")
        print(f"  instr      {', '.join(instruments[:4])}")
        print(f"  truth      {truth.get('style_text', '')[:150].replace(chr(10), ' ')}")
        print()

        rows.append(
            {
                "name": name,
                "bpm_status": bpm_status,
                "key_status": key_status,
                "presence": presence,
                "truth_vocal": truth.get("vocal"),
                "genre": genre,
                "truth": truth,
            }
        )

    # ---- summary
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    def tally(field, good):
        n = sum(1 for r in rows if r[field] in good)
        return n, len(rows)

    bpm_ok, total = tally("bpm_status", ("OK",))
    bpm_any, _ = tally("bpm_status", ("OK", "OCTAVE"))
    key_ok, _ = tally("key_status", ("OK",))
    key_rel, _ = tally("key_status", ("OK", "RELATIVE"))

    print(f"BPM exact            {bpm_ok}/{total}")
    print(f"BPM incl. octave     {bpm_any}/{total}")
    print(f"Key exact            {key_ok}/{total}")
    print(f"Key incl. relative   {key_rel}/{total}")

    # Vocal presence: the label says whether a human voice is present at all.
    voiced_correct = 0
    voiced_total = 0
    for row in rows:
        if not row["truth_vocal"]:
            continue
        voiced_total += 1
        expected_instrumental = row["truth_vocal"] == "instrumental"
        got_instrumental = row["presence"] == "instrumental"
        if expected_instrumental == got_instrumental:
            voiced_correct += 1
    print(f"Vocal presence       {voiced_correct}/{voiced_total}")

    print("\nPer-track BPM/key:")
    for row in rows:
        print(f"  {row['name'][:38]:<40} BPM {row['bpm_status']:<7} key {row['key_status']}")

    print("\nGenre (needs human judgement):")
    for row in rows:
        print(f"  {row['name'][:38]:<40} {', '.join(row['genre']) or '(none)'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
