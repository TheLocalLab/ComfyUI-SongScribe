"""Stage 2: zero-shot descriptor scoring with CLAP.

The model never writes prose and never emits a number. It only ranks a fixed,
hand-authored vocabulary (see vocab/*.yaml) against the audio. Every phrase it
can possibly return is therefore already valid caption language, which removes
the hallucination surface entirely - the failure mode is "picked a less apt
word", never "invented a fact".

Runs on CPU. laion/clap-htsat-unfused is ~150M parameters and needs no GPU.
"""

from __future__ import annotations

import hashlib
import os
import threading

import numpy as np

MODEL_ID = "laion/clap-htsat-unfused"

# CLAP was trained at 48 kHz on 10 s excerpts.
CLAP_SR = 48000
WINDOW_SECONDS = 10.0

# How many excerpts to score across the track. More windows track the
# arrangement's evolution better but cost linearly more; 6 covers a 5-minute
# song at one window per 50 s.
MAX_WINDOWS = 6

VOCAB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab")

_lock = threading.Lock()
_model = None
_processor = None
_text_cache: dict[str, np.ndarray] = {}


class DescriptorError(RuntimeError):
    pass


def is_available() -> bool:
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def load_vocabularies(vocab_dir: str = VOCAB_DIR) -> dict:
    """Read every vocab/*.yaml. Users can drop new axes in without code changes."""
    import yaml

    axes = {}
    if not os.path.isdir(vocab_dir):
        raise DescriptorError(f"vocabulary directory missing: {vocab_dir}")

    for name in sorted(os.listdir(vocab_dir)):
        if not name.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(vocab_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                spec = yaml.safe_load(fh)
        except Exception as exc:
            print(f"[SongScribe] skipping malformed vocabulary {name}: {exc}")
            continue

        if not spec or not spec.get("labels"):
            continue

        axis = spec.get("axis") or os.path.splitext(name)[0]
        spec.setdefault("prompt", "{label}")
        spec.setdefault("top_k", 3)
        spec.setdefault("threshold", 0.05)
        spec.setdefault("temperature", 0.05)
        spec.setdefault("mode", "multi")
        axes[axis] = spec

    if not axes:
        raise DescriptorError(f"no usable vocabulary files in {vocab_dir}")
    return axes


def _get_model():
    """Load CLAP once per process. The first call downloads ~600 MB."""
    global _model, _processor

    with _lock:
        if _model is not None:
            return _model, _processor

        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise DescriptorError(
                "Descriptor scoring needs transformers and torch, which ship "
                "with ComfyUI. Import failed: " + str(exc)
            ) from exc

        print(f"[SongScribe] loading {MODEL_ID} (first run downloads ~600 MB)...")
        try:
            _model = ClapModel.from_pretrained(MODEL_ID)
            _processor = ClapProcessor.from_pretrained(MODEL_ID)
        except Exception as exc:
            raise DescriptorError(
                f"Could not load {MODEL_ID}: {exc}\n"
                "Check the network connection, or pre-download the model into "
                "the Hugging Face cache."
            ) from exc

        _model.eval()
        # CPU by design - this stays off the GPU so it never competes with the
        # music model for VRAM in the same workflow.
        _model.to("cpu")
        torch.set_grad_enabled(False)
        print("[SongScribe] CLAP ready")

    return _model, _processor


def free_model() -> None:
    """Drop the model so a long-running ComfyUI session can reclaim the RAM."""
    global _model, _processor
    with _lock:
        _model = None
        _processor = None
        _text_cache.clear()


def _windows(samples: np.ndarray, sr: int) -> list[np.ndarray]:
    """Evenly spaced excerpts spanning the track."""
    window_length = int(WINDOW_SECONDS * sr)

    if len(samples) <= window_length:
        return [samples]

    count = int(min(MAX_WINDOWS, max(1, len(samples) // window_length)))
    starts = np.linspace(0, len(samples) - window_length, count).astype(int)
    return [samples[s : s + window_length] for s in starts]


def _to_embedding_array(output) -> np.ndarray:
    """Normalise what transformers hands back into a plain (n, d) array.

    transformers 5 returns a BaseModelOutputWithPooling from get_*_features
    where older versions returned the projected tensor directly, so both shapes
    have to be accepted.
    """
    tensor = output
    for attribute in ("pooler_output", "last_hidden_state"):
        candidate = getattr(output, attribute, None)
        if candidate is not None:
            tensor = candidate
            break
    else:
        if isinstance(output, (tuple, list)):
            tensor = output[0]

    array = tensor.detach().cpu().numpy()
    if array.ndim > 2:
        # Pool away any sequence dimension so every axis is (batch, features).
        array = array.mean(axis=tuple(range(1, array.ndim - 1)))
    return array


def _normalise(array: np.ndarray) -> np.ndarray:
    return array / (np.linalg.norm(array, axis=1, keepdims=True) + 1e-9)


def _embed_audio(windows: list[np.ndarray]) -> np.ndarray:
    import torch

    model, processor = _get_model()

    inputs = processor(
        audio=[w.astype(np.float32) for w in windows],
        sampling_rate=CLAP_SR,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        features = model.get_audio_features(**inputs)

    return _normalise(_to_embedding_array(features))


def _embed_text(prompts: list[str], cache_key: str) -> np.ndarray:
    """Text embeddings are fixed for a given vocabulary, so they are computed
    once per process rather than once per song."""
    import torch

    if cache_key in _text_cache:
        return _text_cache[cache_key]

    model, processor = _get_model()

    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)

    embeddings = _normalise(_to_embedding_array(features))
    _text_cache[cache_key] = embeddings
    return embeddings


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / max(temperature, 1e-6)
    scaled = scaled - scaled.max()
    exponentiated = np.exp(scaled)
    return exponentiated / (exponentiated.sum() + 1e-9)


def score_axis(audio_embeddings: np.ndarray, spec: dict, axis: str) -> dict:
    """Score one vocabulary axis against every window.

    Similarities are softmaxed *within the axis*, which is what makes the
    numbers comparable: raw CLAP cosine similarities sit in a narrow band and
    are not interpretable on their own.
    """
    labels = list(spec["labels"])
    prompts = [spec["prompt"].format(label=label) for label in labels]
    # Key on the prompt text itself. Keying on the label count would silently
    # reuse stale embeddings when a vocabulary is edited without changing
    # length - which is exactly what happens while tuning wording.
    digest = hashlib.sha1("\n".join(prompts).encode("utf-8")).hexdigest()[:16]
    text_embeddings = _embed_text(prompts, cache_key=f"{axis}:{digest}")

    # (n_windows, n_labels)
    similarity = audio_embeddings @ text_embeddings.T

    per_window = np.stack(
        [_softmax(row, spec["temperature"]) for row in similarity], axis=0
    )
    mean_probability = per_window.mean(axis=0)

    order = np.argsort(mean_probability)[::-1]

    if spec.get("mode") == "exclusive":
        winner = int(order[0])
        outcome = (spec.get("outcomes") or {}).get(labels[winner])
        return {
            "top": [
                {
                    "label": labels[winner],
                    "score": round(float(mean_probability[winner]), 4),
                }
            ],
            "outcome": outcome,
            "all": {
                labels[i]: round(float(mean_probability[i]), 4) for i in order[:6]
            },
        }

    selected = []
    for index in order[: spec["top_k"]]:
        score = float(mean_probability[index])
        if score < spec["threshold"]:
            break
        selected.append({"label": labels[index], "score": round(score, 4)})

    return {
        "top": selected,
        "all": {labels[i]: round(float(mean_probability[i]), 4) for i in order[:8]},
        # Per-window winners let the composer talk about how the arrangement
        # changes rather than describing the track as one static block.
        "per_window": [labels[int(np.argmax(row))] for row in per_window],
    }


def describe(
    samples_48k: np.ndarray, vocab_dir: str = VOCAB_DIR, verbose: bool = False
) -> dict:
    """Score every axis. `samples_48k` must be mono float32 at 48 kHz."""
    axes = load_vocabularies(vocab_dir)
    windows = _windows(samples_48k, CLAP_SR)

    if verbose:
        print(f"[SongScribe] scoring {len(windows)} window(s) across {len(axes)} axes")

    audio_embeddings = _embed_audio(windows)

    results: dict = {"_windows": len(windows)}

    presence_spec = axes.pop("vocal_presence", None)
    vocal_state = None
    if presence_spec is not None:
        presence = score_axis(audio_embeddings, presence_spec, "vocal_presence")
        vocal_state = presence.get("outcome")
        results["vocal_presence"] = vocal_state
        results["_vocal_presence_detail"] = presence

    for axis, spec in axes.items():
        # Describing a voice that isn't there is the single most damaging thing
        # this layer could do to a caption, so the vocal axes are skipped
        # outright on an instrumental rather than reported with low scores.
        if axis.startswith("vocal_") and vocal_state == "instrumental":
            results[axis] = []
            continue

        scored = score_axis(audio_embeddings, spec, axis)
        results[axis] = scored["top"]
        results[f"_{axis}_detail"] = scored

    return results
