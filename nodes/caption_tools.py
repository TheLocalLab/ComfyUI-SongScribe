"""Caption Splitter and Composer - the round-trip pair.

Analysing a track gives you a whole caption, but the useful edit is almost
always to one section: keep the measured arrangement, replace the vocal
description entirely. These two nodes make that a graph operation instead of a
copy-paste.
"""

from __future__ import annotations

from ..songscribe import compose


class SongScribeCaptionSplitter:
    """Break a caption into its three editable sections."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("global_metadata", "vocal_details", "arrangement")
    OUTPUT_TOOLTIPS = (
        "Genre, tempo, key, mood, production.",
        "Voice description, or the instrumental note.",
        "Instrumentation, groove and section map.",
    )
    FUNCTION = "split"
    CATEGORY = "SongScribe"
    DESCRIPTION = (
        "Split a three-section caption into separate strings so one section "
        "can be edited or replaced without touching the others."
    )

    def split(self, caption):
        parts = compose.split_caption(caption)
        return (parts["global"], parts["vocal"], parts["arrangement"])


class SongScribeCaptionComposer:
    """Reassemble three sections into a caption."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_metadata": ("STRING", {"multiline": True, "default": ""}),
                "vocal_details": ("STRING", {"multiline": True, "default": ""}),
                "arrangement": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "headers": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Emit the 'Global Metadata:' / 'Vocal "
                        "Details:' / 'Arrangement:' labels. MiniMax expects "
                        "them; turn off only for other models.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "build"
    CATEGORY = "SongScribe"
    DESCRIPTION = "Reassemble three caption sections into one caption string."

    def build(self, global_metadata, vocal_details, arrangement, headers=True):
        bodies = [
            (global_metadata or "").strip(),
            (vocal_details or "").strip(),
            (arrangement or "").strip(),
        ]

        if headers:
            # Strip any header the user left in place, so enabling this option
            # cannot produce "Arrangement: Arrangement: ...".
            cleaned = []
            for body in bodies:
                cleaned.append(compose.HEADER_PATTERN.sub("", body, count=1).strip())
            bodies = cleaned

            caption = "\n\n".join(
                f"{header}: {body}"
                for header, body in zip(compose.SECTION_HEADERS, bodies)
                if body
            )
        else:
            caption = "\n\n".join(body for body in bodies if body)

        return (caption,)


NODE_CLASS_MAPPINGS = {
    "SongScribeCaptionSplitter": SongScribeCaptionSplitter,
    "SongScribeCaptionComposer": SongScribeCaptionComposer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SongScribeCaptionSplitter": "Caption Splitter (SongScribe)",
    "SongScribeCaptionComposer": "Caption Composer (SongScribe)",
}
