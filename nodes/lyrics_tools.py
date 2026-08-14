"""Lyrics Structure node - validate and repair section tags before rendering."""

from __future__ import annotations

from ..songscribe import lyrics as lyrics_engine


class SongScribeLyricsStructure:
    """Normalise lyric section tags and check they fit the target duration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "normalise_tags": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Rewrite (Chorus), {hook}, 'Verse 2:' etc "
                        "into the [Chorus] / [Verse] form MiniMax expects.",
                    },
                ),
            },
            "optional": {
                "max_duration": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 600.0,
                        "step": 1.0,
                        "tooltip": "Target song length to check the lyrics "
                        "against. 0 disables the fit check.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("lyrics", "report", "section_count", "has_warnings")
    OUTPUT_TOOLTIPS = (
        "Lyrics with section tags normalised.",
        "Human-readable structure and fit report.",
        "Number of section tags found.",
        "True if anything would likely degrade the render.",
    )
    FUNCTION = "check"
    CATEGORY = "SongScribe"
    DESCRIPTION = (
        "Validate and normalise lyric section tags. MiniMax treats tags as the "
        "only structural instruction, so a malformed tag silently drops "
        "structure from a render rather than failing loudly."
    )

    def check(self, lyrics, normalise_tags, max_duration=0.0):
        result = lyrics_engine.check(
            lyrics, target_duration=max_duration if max_duration > 0 else None
        )

        output = result["lyrics"] if normalise_tags else (lyrics or "")
        report = lyrics_engine.format_report(result)

        if result["warnings"]:
            print(f"[SongScribe] lyrics: {len(result['warnings'])} warning(s)")
            for warning in result["warnings"]:
                print(f"  ! {warning}")

        return (
            output,
            report,
            result["section_count"],
            bool(result["warnings"]),
        )


NODE_CLASS_MAPPINGS = {"SongScribeLyricsStructure": SongScribeLyricsStructure}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SongScribeLyricsStructure": "Lyrics Structure (SongScribe)"
}
