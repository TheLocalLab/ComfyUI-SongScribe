"""Sidecar analysis cache.

ComfyUI re-executes a node whenever anything upstream changes, and a full
analysis pass is seconds not milliseconds. Without this the node is unusable in
practice, so the cache is core infrastructure rather than an optimisation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

# Bump when the analysis output shape changes, so stale sidecars are ignored
# instead of being deserialised into something the composer no longer expects.
SCHEMA_VERSION = 1

SIDECAR_SUFFIX = ".songscribe.json"


def fingerprint(path: str, extra: dict | None = None) -> str:
    """Cheap but reliable content fingerprint.

    Hashing a whole 60 MB FLAC on every execution would cost more than the
    analysis it is meant to save, so we hash size, mtime and the head/tail of
    the file. Distinct songs collide only if they share all four.
    """
    stat = os.stat(path)
    hasher = hashlib.sha256()
    hasher.update(str(stat.st_size).encode())
    hasher.update(str(int(stat.st_mtime)).encode())

    chunk = 64 * 1024
    with open(path, "rb") as fh:
        hasher.update(fh.read(chunk))
        if stat.st_size > chunk * 2:
            fh.seek(-chunk, os.SEEK_END)
            hasher.update(fh.read(chunk))

    if extra:
        hasher.update(json.dumps(extra, sort_keys=True, default=str).encode())

    return hasher.hexdigest()[:32]


def sidecar_path(audio_path: str) -> str:
    return os.path.splitext(audio_path)[0] + SIDECAR_SUFFIX


def _fallback_dir() -> str:
    try:
        import folder_paths

        base = os.path.join(folder_paths.get_temp_directory(), "songscribe")
    except Exception:
        base = os.path.join(tempfile.gettempdir(), "songscribe")
    os.makedirs(base, exist_ok=True)
    return base


def load(audio_path: str, key: str) -> dict | None:
    """Return cached payload if it matches `key`, else None."""
    for candidate in _candidates(audio_path, key):
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("schema_version") == SCHEMA_VERSION
            and payload.get("fingerprint") == key
        ):
            return payload
    return None


def save(audio_path: str, key: str, payload: dict) -> str | None:
    """Persist payload. Returns the path written, or None if nowhere was writable."""
    record = dict(payload)
    record["schema_version"] = SCHEMA_VERSION
    record["fingerprint"] = key

    for candidate in _candidates(audio_path, key):
        try:
            os.makedirs(os.path.dirname(candidate), exist_ok=True)
            # Write-then-rename so an interrupted run never leaves a truncated
            # sidecar that would be read back as valid-looking JSON.
            tmp = candidate + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, candidate)
            return candidate
        except OSError:
            continue
    return None


def _candidates(audio_path: str, key: str) -> list[str]:
    """Preferred sidecar location first, then a writable fallback.

    Users often keep music on read-only shares or in ComfyUI's input dir; the
    fallback keeps caching working there instead of silently disabling it.
    """
    paths = []
    if audio_path:
        paths.append(sidecar_path(audio_path))
        stem = os.path.splitext(os.path.basename(audio_path))[0]
    else:
        stem = "audio"
    paths.append(os.path.join(_fallback_dir(), f"{stem}.{key}{SIDECAR_SUFFIX}"))
    return paths
