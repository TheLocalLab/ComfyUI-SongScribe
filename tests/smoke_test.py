"""Standalone smoke test - no ComfyUI required.

Synthesises a track with known BPM and key, runs the full analysis pipeline and
checks the measured values land close to the truth.

    python_embeded\\python.exe ComfyUI-SongScribe\\tests\\smoke_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from songscribe import audio_io, compose, features, tags  # noqa: E402

TRUE_BPM = 78.0
TRUE_KEY = "Db major"
SR = 44100
DURATION = 40.0


def _note(freq: float, length: float, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, length, int(sr * length), endpoint=False)
    # Rhodes-ish: fundamental plus a decaying odd harmonic, soft attack.
    wave = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * freq * 2 * t)
    env = np.exp(-2.5 * t) * (1 - np.exp(-60 * t))
    return wave * env


def _kick(length: float, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, length, int(sr * length), endpoint=False)
    sweep = np.sin(2 * np.pi * (110 * np.exp(-30 * t) + 45) * t)
    return sweep * np.exp(-12 * t)


def _snare(length: float, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, length, int(sr * length), endpoint=False)
    noise = np.random.default_rng(0).normal(0, 1, len(t))
    return (noise * 0.7 + np.sin(2 * np.pi * 190 * t) * 0.3) * np.exp(-22 * t)


def _hat(length: float, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, length, int(sr * length), endpoint=False)
    noise = np.random.default_rng(1).normal(0, 1, len(t))
    return noise * np.exp(-70 * t)


def synthesise() -> np.ndarray:
    rng = np.random.default_rng(42)
    total = int(SR * DURATION)
    track = np.zeros(total)

    beat = 60.0 / TRUE_BPM
    bar = beat * 4

    # Db major: Dbmaj7 - Bbm7 - Ebm7 - Ab7, the sort of loop the caption format
    # is meant to describe.
    chords = [
        [277.18, 349.23, 415.30, 523.25],  # Dbmaj7
        [233.08, 277.18, 349.23, 415.30],  # Bbm7
        [311.13, 369.99, 466.16, 554.37],  # Ebm7
        [415.30, 523.25, 622.25, 739.99],  # Ab7
    ]

    def place(signal: np.ndarray, at: float, gain: float):
        start = int(at * SR)
        end = min(start + len(signal), total)
        if start < total:
            track[start:end] += signal[: end - start] * gain

    bar_index = 0
    position = 0.0
    while position < DURATION:
        chord = chords[bar_index % len(chords)]
        for freq in chord:
            place(_note(freq, bar * 0.95), position, 0.16)

        for b in range(4):
            beat_at = position + b * beat
            if beat_at >= DURATION:
                break
            if b in (0, 2):
                place(_kick(0.35), beat_at, 0.8)
            if b in (1, 3):
                place(_snare(0.25), beat_at, 0.45)
            # Swung eighths on the hats.
            place(_hat(0.08), beat_at, 0.18)
            place(_hat(0.08), beat_at + beat * 0.62, 0.12)

        position += bar
        bar_index += 1

    # Sub bass following the chord roots.
    t = np.linspace(0, DURATION, total, endpoint=False)
    track += 0.22 * np.sin(2 * np.pi * 69.3 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * t / bar))

    # Vinyl crackle + tape hiss, so the noisiness detector has something to find.
    track += rng.normal(0, 0.004, total)
    crackle_idx = rng.integers(0, total, size=int(DURATION * 60))
    track[crackle_idx] += rng.normal(0, 0.08, len(crackle_idx))

    peak = np.max(np.abs(track))
    return (track / peak * 0.7).astype(np.float32)


def main() -> int:
    print("Synthesising test track...")
    signal = synthesise()

    tmp = os.path.join(tempfile.gettempdir(), "songscribe_smoke.wav")
    import soundfile as sf

    sf.write(tmp, signal, SR)
    print(f"Wrote {tmp} ({os.path.getsize(tmp) / 1e6:.1f} MB)")

    loaded = audio_io.load_from_path(tmp)
    print(f"Loaded: {loaded.duration:.2f}s @ {loaded.sample_rate} Hz")

    analysis = features.analyse(loaded)
    file_tags = tags.read_tags(tmp)
    composed = compose.compose_caption(analysis, file_tags, None, seed=0)

    print("\n--- MEASURED ---")
    print(f"duration     {analysis['duration']:.2f}s ({analysis['duration_str']})")
    print(f"bpm          {analysis['tempo']['bpm']}  (true {TRUE_BPM})")
    print(f"feel         {analysis['tempo']['feel']} / swing {analysis['tempo']['swing']}")
    print(f"key          {analysis['key']['name']}  (true {TRUE_KEY}) "
          f"conf {analysis['key']['confidence']}")
    print(f"harmony      {analysis['key']['harmonic_complexity']}")
    print(f"brightness   {analysis['timbre']['brightness']} "
          f"({analysis['timbre']['centroid_hz']} Hz)")
    print(f"noisiness    {analysis['timbre']['noisiness']}")
    print(f"dynamics     {analysis['loudness']['descriptor']} "
          f"(crest {analysis['loudness']['crest_db']} dB)")
    print(f"balance      {analysis['balance']['drum_presence']} / "
          f"{analysis['balance']['low_end']}")
    print(f"sections     {analysis['structure']['section_count']} "
          f"({analysis['structure']['arc']})")

    print("\n--- CAPTION ---")
    print(composed["caption"])

    print("\n--- CHECKS ---")
    failures = []

    bpm = analysis["tempo"]["bpm"]
    # Octave errors (half/double time) are a normal and acceptable outcome for
    # any beat tracker, so accept those as passes.
    bpm_ok = any(abs(bpm - TRUE_BPM * m) < 3.0 for m in (0.5, 1.0, 2.0))
    print(f"{'PASS' if bpm_ok else 'FAIL'}  bpm {bpm} vs {TRUE_BPM} (+/- octave)")
    if not bpm_ok:
        failures.append("bpm")

    key_ok = analysis["key"]["name"] == TRUE_KEY
    relative_ok = analysis["key"]["name"] == "Bb minor"  # relative minor of Db
    status = "PASS" if key_ok else ("WARN" if relative_ok else "FAIL")
    print(f"{status}  key {analysis['key']['name']} vs {TRUE_KEY}")
    if not (key_ok or relative_ok):
        failures.append("key")

    dur_ok = abs(analysis["duration"] - DURATION) < 0.5
    print(f"{'PASS' if dur_ok else 'FAIL'}  duration {analysis['duration']:.2f} vs {DURATION}")
    if not dur_ok:
        failures.append("duration")

    sect_ok = analysis["structure"]["section_count"] >= 2
    print(f"{'PASS' if sect_ok else 'FAIL'}  sections {analysis['structure']['section_count']} >= 2")
    if not sect_ok:
        failures.append("sections")

    cap_ok = all(h in composed["caption"] for h in compose.SECTION_HEADERS)
    print(f"{'PASS' if cap_ok else 'FAIL'}  caption has all three sections")
    if not cap_ok:
        failures.append("caption")

    print()
    if failures:
        print(f"FAILURES: {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
