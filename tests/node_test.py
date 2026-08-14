"""Load the pack exactly the way ComfyUI does, with folder_paths stubbed out.

Catches the failure mode that unit tests miss entirely: the package imports
fine from a shell but blows up under ComfyUI's importlib-based loader, or the
node's INPUT_TYPES/RETURN_TYPES contract is malformed.

    python_embeded\\python.exe ComfyUI-SongScribe\\tests\\node_test.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_NAME = os.path.basename(PACK_DIR)


def stub_folder_paths(input_dir: str):
    """Minimal stand-in for ComfyUI's folder_paths module."""
    module = types.ModuleType("folder_paths")
    module.get_input_directory = lambda: input_dir
    module.get_temp_directory = lambda: tempfile.gettempdir()
    module.get_annotated_filepath = lambda name: os.path.join(input_dir, name)
    sys.modules["folder_paths"] = module


def load_pack():
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME, os.path.join(PACK_DIR, "__init__.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    failures = []

    # Put the smoke-test wav where the upload widget would find it.
    input_dir = os.path.join(tempfile.gettempdir(), "songscribe_input")
    os.makedirs(input_dir, exist_ok=True)
    sample = os.path.join(input_dir, "test_track.wav")
    if not os.path.isfile(sample):
        sys.path.insert(0, PACK_DIR)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import soundfile as sf

        from smoke_test import SR, synthesise  # type: ignore

        sf.write(sample, synthesise(), SR)

    stub_folder_paths(input_dir)

    print(f"Loading '{MODULE_NAME}' via importlib...")
    pack = load_pack()

    mappings = getattr(pack, "NODE_CLASS_MAPPINGS", {})
    print(f"  registered nodes: {list(mappings)}")
    if not mappings:
        print("  FAIL  no nodes registered")
        return 1

    for name, cls in mappings.items():
        print(f"\nChecking {name}:")
        for attr in ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION", "CATEGORY"):
            if not hasattr(cls, attr):
                print(f"  FAIL  missing {attr}")
                failures.append(f"{name}.{attr}")
        if failures:
            continue

        spec = cls.INPUT_TYPES()
        print(f"  required: {list(spec.get('required', {}))}")
        print(f"  optional: {list(spec.get('optional', {}))}")

        file_widget = spec["required"]["audio_file"][0]
        if "test_track.wav" not in file_widget:
            print("  FAIL  upload widget did not list the input-dir file")
            failures.append(f"{name}.audio_file")
        else:
            print(f"  PASS  upload widget sees {len(file_widget) - 1} audio file(s)")

        if len(cls.RETURN_TYPES) != len(cls.RETURN_NAMES):
            print("  FAIL  RETURN_TYPES/RETURN_NAMES length mismatch")
            failures.append(f"{name}.returns")
        else:
            print(f"  PASS  {len(cls.RETURN_TYPES)} outputs: {cls.RETURN_NAMES}")

        if not hasattr(cls, cls.FUNCTION):
            print(f"  FAIL  FUNCTION '{cls.FUNCTION}' is not a method")
            failures.append(f"{name}.function")

        # Actually execute it, the way the graph would.
        print("  executing...")
        node = cls()
        result = getattr(node, cls.FUNCTION)(
            audio_file="test_track.wav", describe="off", use_cache=False, seed=0
        )
        if len(result) != len(cls.RETURN_TYPES):
            print(f"  FAIL  returned {len(result)} values, declared {len(cls.RETURN_TYPES)}")
            failures.append(f"{name}.arity")
        else:
            print(f"  PASS  returned {len(result)} values")

        caption, lyrics, duration = result[0], result[1], result[2]
        if not isinstance(caption, str) or "Global Metadata" not in caption:
            print("  FAIL  caption output malformed")
            failures.append(f"{name}.caption")
        else:
            print(f"  PASS  caption ({len(caption)} chars)")
        if not isinstance(duration, float) or duration <= 0:
            print(f"  FAIL  duration output was {duration!r}")
            failures.append(f"{name}.duration")
        else:
            print(f"  PASS  duration {duration:.2f}s as FLOAT")
        print(f"  INFO  lyrics: {len(lyrics)} chars")

        # Second run must hit the cache.
        changed = cls.IS_CHANGED(
            audio_file="test_track.wav", describe="off", use_cache=True, seed=0
        )
        print(f"  PASS  IS_CHANGED -> {str(changed)[:24]}")
        getattr(node, cls.FUNCTION)(
            audio_file="test_track.wav", describe="off", use_cache=True, seed=0
        )

    print()
    if failures:
        print(f"FAILURES: {', '.join(failures)}")
        return 1
    print("Node contract OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
