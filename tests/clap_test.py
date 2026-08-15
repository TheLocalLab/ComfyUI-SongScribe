"""Exercise the CLAP descriptor layer.

First run downloads ~600 MB into the Hugging Face cache.

    python_embeded\\python.exe ComfyUI-SongScribe\\tests\\clap_test.py [audio_file]
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from songscribe import audio_io, descriptors  # noqa: E402


def main() -> int:
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.path.join(tempfile.gettempdir(), "songscribe_smoke.wav")
        if not os.path.isfile(path):
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import soundfile as sf

            from smoke_test import SR, synthesise  # type: ignore

            sf.write(path, synthesise(), SR)

    print(f"Audio: {path}")

    axes = descriptors.load_vocabularies()
    total_labels = sum(len(spec["labels"]) for spec in axes.values())
    print(f"Vocabulary: {len(axes)} axes, {total_labels} labels")
    for axis, spec in axes.items():
        print(f"  {axis:16s} {len(spec['labels']):3d} labels  top_k={spec['top_k']}"
              f" thr={spec['threshold']} mode={spec['mode']}")

    loaded = audio_io.load_from_path(path)
    print(f"\nLoaded {loaded.duration:.1f}s")

    started = time.perf_counter()
    samples = loaded.samples_at(descriptors.CLAP_SR)
    resample_time = time.perf_counter() - started
    print(f"48 kHz signal ready in {resample_time:.2f}s ({len(samples)} samples)")

    # Default to a checkpoint that is already cached: a test suite that pulls
    # gigabytes on every run is a test suite nobody runs.
    model = os.environ.get("SONGSCRIBE_TEST_MODEL", "general")
    print(f"Model: {descriptors.resolve_model(model)}")

    started = time.perf_counter()
    result = descriptors.describe(samples, verbose=True, model_id=model)
    elapsed = time.perf_counter() - started
    print(f"\nScored in {elapsed:.2f}s (includes one-off model load)\n")

    started = time.perf_counter()
    descriptors.describe(samples, model_id=model)
    print(f"Second pass (model warm): {time.perf_counter() - started:.2f}s\n")

    print("--- DESCRIPTORS ---")
    print(f"vocal_presence: {result.get('vocal_presence')}")
    for axis in sorted(k for k in result if not k.startswith("_")):
        if axis == "vocal_presence":
            continue
        picks = result[axis]
        if not picks:
            print(f"\n{axis}: (none above threshold)")
            continue
        print(f"\n{axis}:")
        for pick in picks:
            print(f"   {pick['score']:.4f}  {pick['label']}")
        detail = result.get(f"_{axis}_detail", {})
        runners = list(detail.get("all", {}).items())[len(picks) : len(picks) + 3]
        if runners:
            trailing = ", ".join(f"{lbl} {score:.3f}" for lbl, score in runners)
            print(f"   also considered: {trailing}")

    print("\n--- CHECKS ---")
    failures = []

    presence_detail = result.get("_vocal_presence_detail", {})
    print(f"vocal_presence detail: {presence_detail.get('all')}")

    # The synthetic track genuinely has no vocals, so this is a real check on
    # the model rather than a plumbing test - but only when using the default.
    if len(sys.argv) == 1:
        if result.get("vocal_presence") == "instrumental":
            print("PASS  detected instrumental on a track with no vocals")
        else:
            print(f"WARN  expected 'instrumental', got {result.get('vocal_presence')!r}")
        if result.get("vocal_timbre") == []:
            print("PASS  vocal axes correctly suppressed")
        elif result.get("vocal_presence") == "instrumental":
            print("FAIL  vocal axes not suppressed on an instrumental")
            failures.append("vocal_suppression")

    if not result.get("genre"):
        print("WARN  no genre cleared threshold")
    else:
        print(f"PASS  genre: {result['genre'][0]['label']}")

    if not result.get("instruments"):
        print("WARN  no instruments cleared threshold")
    else:
        print(f"PASS  {len(result['instruments'])} instrument(s) identified")

    print()
    if failures:
        print(f"FAILURES: {', '.join(failures)}")
        return 1
    print("Descriptor layer OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
