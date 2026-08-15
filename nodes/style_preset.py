"""Style Preset node - a caption without needing a reference track."""

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
                "style": (
                    list(compose.STYLES.keys()),
                    {"default": compose.DEFAULT_STYLE},
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
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("caption", "global_metadata", "vocal_details", "arrangement")
    FUNCTION = "build"
    CATEGORY = "SongScribe"
    DESCRIPTION = (
        "Generate a structured caption from a curated style preset. Presets "
        "are YAML files in songscribe/presets - drop in your own and they "
        "appear here on the next restart."
    )

    def build(
        self,
        preset,
        style,
        era,
        texture,
        mood_shift,
        seed,
        blend_with="none",
        blend=0.5,
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

        composed = presets.to_caption(data, seed=seed, style=style)
        print(f"[SongScribe] preset '{data.get('name', preset)}' -> {len(composed['caption'])} chars")

        return (
            composed["caption"],
            composed["global"],
            composed["vocal"],
            composed["arrangement"],
        )


NODE_CLASS_MAPPINGS = {"SongScribeStylePreset": SongScribeStylePreset}
NODE_DISPLAY_NAME_MAPPINGS = {"SongScribeStylePreset": "Style Preset (SongScribe)"}
