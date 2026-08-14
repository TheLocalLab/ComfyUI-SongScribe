"""The Song Analyzer node - the pack's main entry point."""

from __future__ import annotations

import os
import time
import traceback

from ..songscribe import (
    audio_io,
    cache,
    compose,
    descriptors as descriptor_engine,
    features,
    tags as tag_reader,
)

# Sentinel shown in the file dropdown when the user is feeding the AUDIO socket
# instead of uploading a file.
USE_SOCKET = "(use AUDIO input)"


class SongScribeAnalyzer:
    """Analyse a song and emit a MiniMax-style caption, its lyrics and duration."""

    @classmethod
    def INPUT_TYPES(cls):
        files = [USE_SOCKET] + audio_io.list_input_audio_files()
        return {
            "required": {
                "audio_file": (files, {"audio_upload": True}),
                "describe": (
                    ["clap", "off"],
                    {
                        "default": "clap",
                        "tooltip": "Score genre, mood, instruments and vocal "
                        "character with CLAP (CPU, no GPU needed). The first "
                        "run downloads a ~600 MB model. 'off' emits measured "
                        "facts only.",
                    },
                ),
                "use_cache": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Reuse a previous analysis of the same file "
                        "instead of recomputing it on every queue.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Varies caption phrasing without re-analysing "
                        "the audio.",
                    },
                ),
            },
            "optional": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": "Analyse audio from an upstream node. Takes "
                        "priority over the file dropdown when connected.",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "FLOAT",
        "INT",
        "STRING",
        "SONGSCRIBE_ANALYSIS",
    )
    RETURN_NAMES = (
        "caption",
        "lyrics",
        "duration",
        "duration_int",
        "duration_str",
        "analysis",
    )
    OUTPUT_TOOLTIPS = (
        "Three-section caption (Global Metadata / Vocal Details / Arrangement).",
        "Lyrics from embedded tags or a sidecar .lrc/.txt, if present.",
        "Duration in seconds - wire straight into max_duration.",
        "Duration rounded to whole seconds.",
        "Duration formatted as m:ss.",
        "Full analysis payload for downstream SongScribe nodes.",
    )
    FUNCTION = "analyze"
    CATEGORY = "SongScribe"
    DESCRIPTION = (
        "Extract a structured music caption, lyrics and duration from an audio "
        "file. Measured values (BPM, key, dynamics, section map) come from DSP "
        "analysis and are never guessed."
    )

    @classmethod
    def IS_CHANGED(cls, audio_file, describe="clap", use_cache=True, seed=0, audio=None, **kwargs):
        # Seed changes phrasing, so it must invalidate. File content changes
        # must too - hence the fingerprint rather than the filename.
        if audio_file and audio_file != USE_SOCKET:
            path = audio_io.resolve_input_path(audio_file)
            if os.path.isfile(path):
                try:
                    return f"{cache.fingerprint(path)}:{seed}:{use_cache}:{describe}"
                except OSError:
                    pass
        return float("nan")

    def analyze(self, audio_file, describe="clap", use_cache=True, seed=0, audio=None):
        started = time.perf_counter()

        loaded = self._load(audio_file, audio)

        cache_key = None
        payload = None
        if loaded.has_file:
            # The descriptor mode is part of the key: a cached facts-only run
            # must not satisfy a request that also wants CLAP scoring.
            cache_key = cache.fingerprint(
                loaded.source_path,
                extra={"analysis": cache.SCHEMA_VERSION, "describe": describe},
            )
            if use_cache:
                payload = cache.load(loaded.source_path, cache_key)

        if payload is None:
            analysis = features.analyse(loaded)
            file_tags = (
                tag_reader.read_tags(loaded.source_path) if loaded.has_file else {}
            )
            descriptors = self._describe(loaded, describe)
            payload = {
                "analysis": analysis,
                "tags": file_tags,
                "descriptors": descriptors,
            }
            if cache_key and loaded.has_file:
                cache.save(loaded.source_path, cache_key, payload)
        else:
            analysis = payload["analysis"]
            file_tags = payload.get("tags", {})
            descriptors = payload.get("descriptors")

        lyrics, lyrics_source = self._lyrics(loaded, file_tags)

        composed = compose.compose_caption(
            analysis, tags=file_tags, descriptors=descriptors, seed=seed
        )

        duration = float(analysis.get("duration") or loaded.duration)
        elapsed = time.perf_counter() - started
        if descriptors:
            genre = descriptors.get("genre") or []
            described = (
                genre[0]["label"] if genre else descriptors.get("vocal_presence", "?")
            )
        else:
            described = "facts only"
        print(
            f"[SongScribe] analysed {os.path.basename(loaded.source_path or 'AUDIO input')} "
            f"in {elapsed:.2f}s "
            f"({analysis['tempo']['bpm_int']} BPM, {analysis['key']['name']}, "
            f"{analysis['structure']['section_count']} sections, "
            f"{described}, lyrics: {lyrics_source})"
        )

        analysis_out = {
            "analysis": analysis,
            "tags": file_tags,
            "descriptors": descriptors,
            "caption_parts": composed,
            "lyrics_source": lyrics_source,
            "source_path": loaded.source_path,
        }

        return (
            composed["caption"],
            lyrics,
            duration,
            int(round(duration)),
            analysis.get("duration_str", ""),
            analysis_out,
        )

    def _describe(self, loaded, mode):
        """Run CLAP scoring. A failure here degrades the caption but must never
        take the workflow down - the measured half is still worth having."""
        if mode == "off":
            return None

        try:
            samples = loaded.samples_at(descriptor_engine.CLAP_SR)
            return descriptor_engine.describe(samples)
        except descriptor_engine.DescriptorError as exc:
            print(f"[SongScribe] descriptor scoring unavailable: {exc}")
        except Exception as exc:
            print(f"[SongScribe] descriptor scoring failed: {exc}")
            traceback.print_exc()
        return None

    def _load(self, audio_file, audio):
        if audio is not None:
            return audio_io.load_from_comfy_audio(audio)

        if not audio_file or audio_file == USE_SOCKET:
            raise ValueError(
                "SongScribe: no audio provided. Upload a file with the widget, "
                "or connect an AUDIO input."
            )

        path = audio_io.resolve_input_path(audio_file)
        return audio_io.load_from_path(path)

    def _lyrics(self, loaded, file_tags) -> tuple[str, str]:
        """Embedded tags first, then a sidecar file. Both are exact; neither is
        guaranteed to exist. ASR fallback arrives in a later phase."""
        embedded = file_tags.get("lyrics")
        if embedded:
            return tag_reader.strip_lrc_timestamps(embedded), "embedded tag"

        if loaded.has_file:
            found = tag_reader.find_sidecar_lyrics(loaded.source_path)
            if found:
                text, source = found
                return tag_reader.strip_lrc_timestamps(text), source

        return "", "none found"


NODE_CLASS_MAPPINGS = {"SongScribeAnalyzer": SongScribeAnalyzer}
NODE_DISPLAY_NAME_MAPPINGS = {"SongScribeAnalyzer": "Song Analyzer (SongScribe)"}
