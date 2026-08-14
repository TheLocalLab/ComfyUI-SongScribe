"""Verify the loader really does accept "a variety of different formats".

Transcodes the smoke-test tone into every container we claim to support, then
loads each one back and checks the duration survives the round trip.

    python_embeded\\python.exe ComfyUI-SongScribe\\tests\\format_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from songscribe import audio_io  # noqa: E402

SR = 44100
DURATION = 12.0

# (extension, codec). None means "let the muxer pick its default".
TARGETS = [
    ("wav", "pcm_s16le"),
    ("flac", "flac"),
    ("mp3", "libmp3lame"),
    ("m4a", "aac"),
    ("ogg", "libvorbis"),
    ("opus", "libopus"),
    ("aiff", "pcm_s16be"),
    ("wma", "wmav2"),
]


def make_source() -> str:
    import soundfile as sf

    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)
    path = os.path.join(tempfile.gettempdir(), "songscribe_fmt_src.wav")
    sf.write(path, signal.astype(np.float32), SR)
    return path


def transcode(source: str, ext: str, codec: str) -> str | None:
    import av

    out = os.path.join(tempfile.gettempdir(), f"songscribe_fmt.{ext}")
    try:
        with av.open(source) as src, av.open(out, "w") as dst:
            in_stream = src.streams.audio[0]
            out_stream = dst.add_stream(codec, rate=SR)
            for frame in src.decode(in_stream):
                frame.pts = None
                for packet in out_stream.encode(frame):
                    dst.mux(packet)
            for packet in out_stream.encode(None):
                dst.mux(packet)
        return out
    except Exception as exc:
        print(f"  SKIP  {ext:5s} - encoder unavailable ({type(exc).__name__}: {exc})")
        return None


def main() -> int:
    source = make_source()
    print(f"Source: {source} ({DURATION}s @ {SR} Hz)\n")

    tested = 0
    failures = []

    for ext, codec in TARGETS:
        path = transcode(source, ext, codec)
        if path is None:
            continue
        tested += 1
        try:
            loaded = audio_io.load_from_path(path)
        except Exception as exc:
            print(f"  FAIL  {ext:5s} - {type(exc).__name__}: {exc}")
            failures.append(ext)
            continue

        drift = abs(loaded.duration - DURATION)
        # Lossy codecs pad with encoder delay/priming samples, so an exact
        # duration match is not achievable; a quarter second is generous
        # enough for those and still catches a truncated decode.
        ok = drift < 0.25
        size_kb = os.path.getsize(path) / 1024
        print(
            f"  {'PASS' if ok else 'FAIL'}  {ext:5s} "
            f"{loaded.duration:6.2f}s (drift {drift:+.3f}s) "
            f"native_sr={loaded.native_sample_rate} {size_kb:7.1f} KB"
        )
        if not ok:
            failures.append(ext)

    print(f"\n{tested - len(failures)}/{tested} formats loaded correctly.")
    if failures:
        print(f"FAILURES: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
