"""Supervised style tagging with MAEST.

CLAP guesses genre by embedding text and audio near each other; it has never
seen "reggae" as a label. MAEST is trained on 400 Discogs styles, so it has.
Measured on labelled tracks, the difference is not marginal: MAEST returned
Trap / Cloud Rap / Hardcore Hip-Hop for a trap track that CLAP called "bossa
nova", and five reggae sub-styles for a reggae track CLAP called "classic
soul". Both cost about two seconds per track on CPU.

SECURITY
--------
MAEST ships a custom feature extractor and therefore needs
`trust_remote_code=True`, which executes Python from the model repository.
Three things make that acceptable to ship rather than merely tolerable:

1. It is opt-in. The node defaults to CLAP; nothing here runs unless asked.
2. REVISION pins an exact commit, so a later change to the repository cannot
   silently execute on anyone who installed this. Loading "main" would mean
   running whatever the repo contains on the day the user hits queue.
3. The pinned file is a 242-line mel-spectrogram extractor importing only
   numpy, torch and transformers' own audio utilities - no network, no
   subprocess, no eval/exec, no file access.

Anyone re-pointing REVISION at a newer commit should re-read the file first.
"""

from __future__ import annotations

import threading

import numpy as np

MODEL_ID = "mtg-upf/discogs-maest-10s-pw-129e"

# Pinned deliberately - see the security note above.
REVISION = "54b3b0aa49ab26bc86d973c53d41aa6b28597b06"

MAEST_SR = 16000

# MAEST spreads probability over 400 styles, so the winning score is small in
# absolute terms; 0.03 sits well below every correct call observed while still
# excluding the long tail of near-zero labels.
MIN_SCORE = 0.03

# A label must reach this fraction of the winner's score to be a second
# opinion rather than noise.
RELATIVE_FLOOR = 0.35

MAX_STYLES = 2

# Discogs writes styles as "Family---Style". The style half is already good
# caption language; these fix the few whose casing matters.
_CASING = {
    "rnb/swing": "R&B/swing",
    "contemporary r&b": "contemporary R&B",
    "rhythm & blues": "rhythm & blues",
    "idm": "IDM",
    "edm": "EDM",
    "uk garage": "UK garage",
    "j-pop": "J-pop",
    "k-pop": "K-pop",
    "dj battle tool": "DJ battle tool",
}

_lock = threading.Lock()
_pipeline = None


class TaggerError(RuntimeError):
    pass


def is_available() -> bool:
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _get_pipeline():
    global _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise TaggerError(
                "Style tagging needs transformers, which ships with ComfyUI. "
                f"Import failed: {exc}"
            ) from exc

        print(
            f"[SongScribe] loading {MODEL_ID} @ {REVISION[:8]} "
            "(trust_remote_code, pinned revision, ~330 MB on first use)..."
        )
        try:
            _pipeline = pipeline(
                "audio-classification",
                model=MODEL_ID,
                revision=REVISION,
                trust_remote_code=True,
                device=-1,
                top_k=8,
            )
        except Exception as exc:
            raise TaggerError(f"could not load {MODEL_ID}: {exc}") from exc

        print("[SongScribe] style tagger ready")
    return _pipeline


def free_model() -> None:
    global _pipeline
    with _lock:
        _pipeline = None


def _split(label: str) -> tuple[str, str]:
    """'Hip Hop---Trap' -> ('Hip Hop', 'trap')."""
    family, _, style = label.partition("---")
    style = (style or family).strip()
    key = style.lower()
    return family.strip(), _CASING.get(key, key)


def tag(samples_16k: np.ndarray) -> dict:
    """Predict Discogs styles. `samples_16k` must be mono float32 at 16 kHz."""
    classifier = _get_pipeline()

    try:
        predictions = classifier(
            {"raw": samples_16k.astype(np.float32), "sampling_rate": MAEST_SR}
        )
    except Exception as exc:
        raise TaggerError(f"style tagging failed: {exc}") from exc

    if not predictions:
        return {"genre": [], "raw": []}

    top_score = float(predictions[0]["score"])

    # One entry per family. The top five for a reggae track are all reggae
    # sub-styles; listing them as five genres would pad the caption without
    # adding information, whereas a second *family* is genuinely new.
    best_by_family: dict[str, dict] = {}
    for prediction in predictions:
        score = float(prediction["score"])
        if score < MIN_SCORE or score < top_score * RELATIVE_FLOOR:
            continue
        family, style = _split(prediction["label"])
        if family not in best_by_family:
            best_by_family[family] = {"label": style, "score": round(score, 4)}

    ordered = sorted(best_by_family.values(), key=lambda x: -x["score"])

    return {
        "genre": ordered[:MAX_STYLES],
        "raw": [
            {"label": p["label"], "score": round(float(p["score"]), 4)}
            for p in predictions[:8]
        ],
        "top_score": round(top_score, 4),
    }
