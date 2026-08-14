"""Embedded file metadata and lyrics.

This is the cheapest and by far the most accurate source of information we
have - if the file already carries an artist, a genre or a full lyric sheet,
that beats anything we could infer from the waveform.
"""

from __future__ import annotations

import os
import re

# Frame/field names vary by container; map them all onto one flat vocabulary.
_FIELD_ALIASES = {
    "title": ("TIT2", "title", "\xa9nam", "Title"),
    "artist": ("TPE1", "artist", "\xa9ART", "Artist"),
    "album": ("TALB", "album", "\xa9alb", "Album"),
    "genre": ("TCON", "genre", "\xa9gen", "Genre"),
    "date": ("TDRC", "date", "\xa9day", "Year", "originaldate"),
    "bpm": ("TBPM", "bpm", "tmpo", "BPM"),
    "key": ("TKEY", "initialkey", "key", "Initial key"),
    "comment": ("COMM", "comment", "\xa9cmt"),
}

_LYRIC_KEYS = ("USLT", "unsyncedlyrics", "lyrics", "\xa9lyr", "LYRICS", "Lyrics")

# [mm:ss.xx] or <mm:ss.xx> timestamps used by LRC / synced lyric formats.
_TIMESTAMP = re.compile(r"[\[<]\d{1,3}:\d{2}(?:[.:]\d{1,3})?[\]>]")


def _stringify(value) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(_stringify(v) for v in value if v is not None).strip()
    text = getattr(value, "text", value)
    if isinstance(text, (list, tuple)):
        return ", ".join(str(t) for t in text).strip()
    return str(text).strip()


def read_tags(path: str) -> dict:
    """Read embedded tags. Never raises - missing metadata is normal."""
    result: dict = {"available": False}
    try:
        import mutagen
    except ImportError:
        result["error"] = "mutagen not installed"
        return result

    try:
        handle = mutagen.File(path)
    except Exception as exc:
        result["error"] = f"could not read tags: {exc}"
        return result

    if handle is None:
        return result

    result["available"] = True

    info = getattr(handle, "info", None)
    if info is not None:
        for attr, key in (
            ("length", "duration"),
            ("bitrate", "bitrate"),
            ("sample_rate", "sample_rate"),
            ("channels", "channels"),
        ):
            value = getattr(info, attr, None)
            if value is not None:
                result[key] = value
        result["codec"] = type(handle).__name__

    raw = dict(handle.tags or {})
    # Normalise keys: ID3 frames appear as "COMM::eng", "TXXX:KEY" etc.
    flat = {}
    for key, value in raw.items():
        flat[key] = value
        base = str(key).split(":")[0]
        flat.setdefault(base, value)

    for field, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            if alias in flat:
                text = _stringify(flat[alias])
                if text:
                    result[field] = text
                break

    lyrics = _extract_lyrics(flat)
    if lyrics:
        result["lyrics"] = lyrics
        result["lyrics_source"] = "embedded tag"

    return result


def _extract_lyrics(flat: dict) -> str | None:
    for key in _LYRIC_KEYS:
        for candidate in (key,) + tuple(k for k in flat if str(k).startswith(key)):
            if candidate in flat:
                text = _stringify(flat[candidate])
                if text and len(text) > 20:
                    return text
    return None


def find_sidecar_lyrics(audio_path: str) -> tuple[str, str] | None:
    """Look for a .lrc / .txt lyric file sitting next to the audio.

    Returns (text, source_description) or None.
    """
    stem = os.path.splitext(audio_path)[0]
    for ext in (".lrc", ".txt"):
        candidate = stem + ext
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read().strip()
            except OSError:
                continue
            if len(text) > 20:
                return text, f"sidecar {ext} file"
    return None


def strip_lrc_timestamps(text: str) -> str:
    """Turn a timestamped LRC sheet into plain lyric lines."""
    lines = []
    for line in text.splitlines():
        cleaned = _TIMESTAMP.sub("", line).strip()
        # Drop LRC metadata directives like [ar:Artist] / [ti:Title].
        if re.fullmatch(r"\[[a-z]{2,}:.*\]", cleaned, flags=re.IGNORECASE):
            continue
        lines.append(cleaned)

    # Collapse runs of blank lines left behind by stripping.
    out: list[str] = []
    for line in lines:
        if not line and out and not out[-1]:
            continue
        out.append(line)
    return "\n".join(out).strip()
