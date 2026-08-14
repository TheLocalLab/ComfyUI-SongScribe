"""SongScribe - structured music captions, lyrics and duration from audio files.

Built for MiniMax Music 3's three-section caption format, but the outputs are
plain STRING/FLOAT so the nodes drop into any audio workflow.
"""

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}


def _register(module):
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(
        getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    )


# Import failures here would take the whole pack down and hide every node, so
# each module is registered independently and a failure is reported loudly
# rather than silently swallowed.
_MODULES = ("analyzer", "caption_tools", "style_preset", "lyrics_tools")

for _name in _MODULES:
    try:
        _module = __import__(f"{__name__}.nodes.{_name}", fromlist=["*"])
        _register(_module)
    except Exception as _exc:  # pragma: no cover - import-time diagnostics
        import traceback

        print(f"[SongScribe] failed to load node module '{_name}': {_exc}")
        traceback.print_exc()

print(f"[SongScribe] loaded {len(NODE_CLASS_MAPPINGS)} node(s)")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
