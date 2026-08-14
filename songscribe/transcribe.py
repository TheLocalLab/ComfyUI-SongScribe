"""Phase 5: optional lyric transcription with faster-whisper.

Default-off, and deliberately so. Whisper was trained on speech, not singing;
on a full mix with instruments competing for the same frequencies it produces
usable-but-imperfect text that generally needs hand-fixing. Embedded tags and
.lrc sidecars are exact, so they are always tried first - this is the fallback
for songs that carry no lyrics at all.

Section tags are *not* taken from the ASR output. Whisper emits words and
timings, nothing about song structure. They are derived from two things it does
give us honestly: silence between sung phrases, and repetition of the lyric text
itself.
"""

from __future__ import annotations

import re
import threading

import numpy as np

# Whisper resamples internally to 16 kHz; feeding it that directly avoids a
# redundant resample pass.
WHISPER_SR = 16000

MODEL_SIZES = ("tiny", "base", "small", "medium")
DEFAULT_MODEL = "base"

# A vocal gap longer than this is treated as an instrumental passage rather
# than a pause for breath.
INSTRUMENTAL_GAP = 8.0

# Gap that ends a lyrical block (a verse, a chorus) without necessarily
# implying a full instrumental section.
BLOCK_GAP = 2.5

_lock = threading.Lock()
_models: dict[str, object] = {}


class TranscriptionError(RuntimeError):
    pass


def is_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def _get_model(size: str, compute_type: str = "int8"):
    """Load and cache a Whisper model. int8 on CPU by design - this must not
    compete with the music model for VRAM in the same workflow."""
    key = f"{size}:{compute_type}"
    with _lock:
        if key in _models:
            return _models[key]

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError(
                "Lyric transcription needs faster-whisper:\n"
                "  python_embeded\\python.exe -m pip install faster-whisper"
            ) from exc

        print(f"[SongScribe] loading Whisper '{size}' ({compute_type}, CPU)...")
        try:
            model = WhisperModel(size, device="cpu", compute_type=compute_type)
        except Exception as exc:
            raise TranscriptionError(f"could not load Whisper '{size}': {exc}") from exc

        _models[key] = model
        return model


def free_models() -> None:
    with _lock:
        _models.clear()


def _normalise_line(text: str) -> str:
    """Collapse a lyric line for repetition comparison."""
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def transcribe(
    samples_16k: np.ndarray,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
    compute_type: str = "int8",
) -> dict:
    """Transcribe sung audio. `samples_16k` must be mono float32 at 16 kHz."""
    model = _get_model(model_size, compute_type)

    segments, info = model.transcribe(
        samples_16k,
        language=language,
        beam_size=5,
        # Music is continuous; Whisper's VAD is tuned for speech and cuts sung
        # phrases short, so gaps are found from segment timings instead.
        vad_filter=False,
        condition_on_previous_text=False,
    )

    collected = []
    for segment in segments:
        text = (segment.text or "").strip()
        if text:
            collected.append(
                {"start": float(segment.start), "end": float(segment.end), "text": text}
            )

    return {
        "segments": collected,
        "language": getattr(info, "language", None),
        "language_probability": round(float(getattr(info, "language_probability", 0.0)), 3),
    }


def structure(segments: list[dict], duration: float | None = None) -> str:
    """Turn timed segments into tagged lyrics.

    Repetition is the only structural claim made here, and it is made from the
    lyric text rather than the audio: a block of lines that occurs more than
    once is a chorus by definition of the word. Blocks that do not repeat are
    labelled [Verse], which is the honest default - not a claim that they are
    verses in a musicologist's sense, but that they are sung, non-repeating
    material.
    """
    if not segments:
        return ""

    # Group segments into blocks separated by vocal gaps.
    blocks: list[dict] = []
    current: dict | None = None
    previous_end = 0.0

    for segment in segments:
        gap = segment["start"] - previous_end
        if current is None or gap >= BLOCK_GAP:
            if current is not None:
                blocks.append(current)
            current = {
                "start": segment["start"],
                "end": segment["end"],
                "lines": [],
                "gap_before": gap if current is not None else segment["start"],
            }
        current["lines"].append(segment["text"])
        current["end"] = segment["end"]
        previous_end = segment["end"]

    if current is not None:
        blocks.append(current)

    # Identify repeated blocks by normalised text.
    signatures = [
        " ".join(_normalise_line(line) for line in block["lines"]) for block in blocks
    ]
    counts: dict[str, int] = {}
    for signature in signatures:
        if signature:
            counts[signature] = counts.get(signature, 0) + 1

    output: list[str] = []

    if blocks and blocks[0]["start"] > 3.0:
        output.append("[Intro]")
        output.append("")

    for index, block in enumerate(blocks):
        if block["gap_before"] >= INSTRUMENTAL_GAP and index > 0:
            output.append("[Instrumental]")
            output.append("")

        signature = signatures[index]
        repeated = counts.get(signature, 0) > 1
        is_last = index == len(blocks) - 1

        if repeated:
            tag = "[Chorus]"
        elif (
            is_last
            # A song transcribed as a single block is not an outro. The closing
            # section can only be identified relative to something before it.
            and index > 0
            and duration
            and block["end"] > duration * 0.85
        ):
            tag = "[Outro]"
        else:
            tag = "[Verse]"

        output.append(tag)
        output.extend(block["lines"])
        output.append("")

    # Trailing instrumental run-out.
    if duration and blocks and duration - blocks[-1]["end"] >= INSTRUMENTAL_GAP:
        output.append("[Outro]")

    return "\n".join(output).strip()


def transcribe_to_lyrics(
    samples_16k: np.ndarray,
    duration: float | None = None,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
) -> dict:
    """Full path: audio in, tagged lyrics out."""
    result = transcribe(samples_16k, model_size=model_size, language=language)
    result["lyrics"] = structure(result["segments"], duration)
    result["line_count"] = sum(1 for s in result["segments"])
    return result
