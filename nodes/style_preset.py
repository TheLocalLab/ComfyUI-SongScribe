"""Style Preset node - a caption without needing a reference track.

Emits either MiniMax's three-section caption or YuE2's flat descriptor string.
The two models want genuinely different shapes - YuE2 has no section headers,
states vocal gender outright, and names the language - so the format is a
choice rather than a post-processing step.
"""

from __future__ import annotations

from ..songscribe import compose, presets


class SongScribeStylePreset:
    """Build a caption from a curated preset, with optional blending."""

    @classmethod
    def INPUT_TYPES(cls):
        available = presets.list_choices() or ["(no presets found)"]
        return {
            "required": {
                "preset": (available, {"default": available[0]}),
                "format": (
                    ["minimax", "yue2"],
                    {
                        "default": "minimax",
                        "tooltip": "minimax: three labelled sections (Global "
                        "Metadata / Vocal Details / Arrangement). yue2: one "
                        "flat comma-separated descriptor string, which is what "
                        "YuE2's style input expects.",
                    },
                ),
                "detail": (
                    ["tags", "full", "rich"],
                    {
                        "default": "full",
                        "tooltip": "YuE2 only. tags: genre, tempo, mood and "
                        "vocal. full: adds instrumentation and production. "
                        "rich: adds scene and extra texture. YuE2's own "
                        "prompts range across all three.",
                    },
                ),
                "vocal": (
                    list(presets.YUE2_VOCALS.keys()),
                    {
                        "default": "auto",
                        "tooltip": "YuE2 states vocal type explicitly in most "
                        "prompts. 'auto' uses whatever the preset declares.",
                    },
                ),
                "language": (
                    presets.YUE2_LANGUAGES,
                    {
                        "default": "auto",
                        "tooltip": "Sung language. YuE2 supports English, "
                        "Mandarin, Japanese, Korean, Spanish and Russian; "
                        "naming it steers pronunciation.",
                    },
                ),
                "style": (
                    list(compose.STYLES.keys()),
                    {
                        "default": compose.DEFAULT_STYLE,
                        "tooltip": "MiniMax only: how much measured detail the "
                        "caption carries.",
                    },
                ),
                "era": (
                    list(presets.MODIFIERS["era"]),
                    {"default": "none", "tooltip": "Layer a production era on top."},
                ),
                "texture": (
                    list(presets.MODIFIERS["texture"]),
                    {"default": "none", "tooltip": "Layer a recording texture on top."},
                ),
                "mood_shift": (
                    list(presets.MODIFIERS["mood_shift"]),
                    {"default": "none", "tooltip": "Push the mood in a direction."},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "blend_with": (
                    ["none"] + (presets.list_choices() or []),
                    {"default": "none", "tooltip": "Optional second preset to mix in."},
                ),
                "blend": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "How much of the second preset to admit. "
                        "0 = first only, 1 = second only.",
                    },
                ),
                "extra": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Appended verbatim. Use for anything the "
                        "preset cannot know - a reference artist, a negative "
                        "instruction like 'avoid heavy distortion', or a "
                        "specific structure.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "global_metadata", "vocal_details", "arrangement")
    OUTPUT_TOOLTIPS = (
        "The style prompt, in the selected format. Wire to YuE2's style input "
        "or MiniMax's caption input.",
        "MiniMax section 1 (empty detail in yue2 format is still filled).",
        "MiniMax section 2.",
        "MiniMax section 3.",
    )
    FUNCTION = "build"
    CATEGORY = "SongScribe"
    DESCRIPTION = (
        "Generate a style prompt from a curated preset, for MiniMax Music 3 or "
        "YuE2. Presets are YAML files in songscribe/presets - drop in your own "
        "and they appear here on the next restart."
    )

    def build(
        self,
        preset,
        format="minimax",
        detail="full",
        vocal="auto",
        language="auto",
        style=compose.DEFAULT_STYLE,
        era="none",
        texture="none",
        mood_shift="none",
        seed=0,
        blend_with="none",
        blend=0.5,
        extra="",
    ):
        try:
            data = presets.load_choice(preset)
        except presets.PresetError as exc:
            return (f"[SongScribe] {exc}", "", "", "")

        if blend_with and blend_with != "none" and blend_with != preset:
            try:
                data = presets.blend(data, presets.load_choice(blend_with), blend)
            except presets.PresetError as exc:
                print(f"[SongScribe] blend skipped: {exc}")

        data = presets.apply_modifiers(
            data, era=era, texture=texture, mood_shift=mood_shift
        )

        # The three-section form is always computed: it costs nothing and keeps
        # the companion Splitter/Composer nodes usable in either format.
        composed = presets.to_caption(data, seed=seed, style=style)

        if format == "yue2":
            prompt = presets.to_yue2(
                data, vocal=vocal, language=language, detail=detail
            )
        else:
            prompt = composed["caption"]

        addition = (extra or "").strip()
        if addition:
            separator = ", " if format == "yue2" else "\n\n"
            prompt = f"{prompt}{separator}{addition}" if prompt else addition

        print(
            f"[SongScribe] preset '{data.get('name', preset)}' -> "
            f"{format} ({len(prompt)} chars)"
        )

        return (
            prompt,
            composed["global"],
            composed["vocal"],
            composed["arrangement"],
        )


NODE_CLASS_MAPPINGS = {"SongScribeStylePreset": SongScribeStylePreset}
NODE_DISPLAY_NAME_MAPPINGS = {"SongScribeStylePreset": "Style Preset (SongScribe)"}
