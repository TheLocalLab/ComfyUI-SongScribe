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

# Selectable CLAP checkpoints. The general model was the original default, but
# it was trained on environmental sound and speech alongside music; the
# music-specialised checkpoints are the same architecture trained on a music-
# heavy corpus, and score better on genre.
# laion/larger_clap_music is deliberately absent. Measured on five labelled
# tracks it returned no genre above threshold and called every one of them
# instrumental - not weak accuracy but a pipeline mismatch, probably around the
# fused-model input handling. Offering a checkpoint that silently returns
# nothing is worse than not offering it. Pass the full id to try it anyway.
MODELS = {
    "music_and_speech": "laion/larger_clap_music_and_speech",
    "general": "laion/clap-htsat-unfused",
}
DEFAULT_MODEL = "music_and_speech"

MODEL_ID = MODELS[DEFAULT_MODEL]


def resolve_model(name: str | None) -> str:
    """Accept a short key or a full Hugging Face id."""
    if not name:
        return MODEL_ID
    return MODELS.get(name, name)

# CLAP was trained at 48 kHz on 10 s excerpts.
CLAP_SR = 48000
WINDOW_SECONDS = 10.0

# How many excerpts to score across the track. More windows track the
# arrangement's evolution better but cost linearly more; 6 covers a 5-minute
# song at one window per 50 s.
MAX_WINDOWS = 6

# Minimum standout (0-1, fraction of the maximum possible for the axis) before
# a label may be stated at all. Per-axis `min_z` in a vocabulary file
# overrides it. Calibrated against labelled tracks - see tools/calibrate.py.
#
# Set low on purpose. Measured on six labelled tracks, standout does not
# separate correct genre calls from wrong ones - the two worst calls scored
# highest - so a high bar here buys silence, not accuracy. It exists to catch
# the genuinely undecided case, not to fix genre; that needs a better model,
# not a stricter gate.
DEFAULT_MIN_Z = 0.25

VOCAB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab")

_lock = threading.Lock()
# Keyed by model id: switching checkpoints must not reuse another model's
# weights or, more subtly, another model's text embeddings.
_loaded: dict[str, tuple] = {}
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
        # An axis may declare several phrasings. Zero-shot scores move
        # noticeably with wording, so averaging a few templates per label is
        # more stable than betting on one - the same trick CLIP uses for its
        # zero-shot benchmarks. A single "prompt" remains valid.
        if not spec.get("prompts"):
            spec["prompts"] = [spec["prompt"]]
        spec.setdefault("top_k", 3)
        spec.setdefault("threshold", 0.05)
        spec.setdefault("temperature", 0.05)
        spec.setdefault("mode", "multi")
        axes[axis] = spec

    if not axes:
        raise DescriptorError(f"no usable vocabulary files in {vocab_dir}")
    return axes


def _get_model(model_id: str | None = None):
    """Load a CLAP checkpoint once per process. First call downloads weights."""
    model_id = resolve_model(model_id)

    with _lock:
        if model_id in _loaded:
            return _loaded[model_id]

        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise DescriptorError(
                "Descriptor scoring needs transformers and torch, which ship "
                "with ComfyUI. Import failed: " + str(exc)
            ) from exc

        print(f"[SongScribe] loading {model_id} (first run downloads weights)...")
        try:
            model = ClapModel.from_pretrained(model_id)
            processor = ClapProcessor.from_pretrained(model_id)
        except Exception as exc:
            raise DescriptorError(
                f"Could not load {model_id}: {exc}\n"
                "Check the network connection, or pre-download the model into "
                "the Hugging Face cache."
            ) from exc

        model.eval()
        # CPU by design - this stays off the GPU so it never competes with the
        # music model for VRAM in the same workflow.
        model.to("cpu")
        torch.set_grad_enabled(False)
        print(f"[SongScribe] CLAP ready ({model_id})")

        _loaded[model_id] = (model, processor)

    return _loaded[model_id]


def free_model() -> None:
    """Drop loaded models so a long-running session can reclaim the RAM."""
    with _lock:
        _loaded.clear()
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


def _embed_audio(windows: list[np.ndarray], model_id: str | None = None) -> np.ndarray:
    import torch

    model, processor = _get_model(model_id)

    inputs = processor(
        audio=[w.astype(np.float32) for w in windows],
        sampling_rate=CLAP_SR,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        features = model.get_audio_features(**inputs)

    return _normalise(_to_embedding_array(features))


def _embed_text(
    prompts: list[str], cache_key: str, model_id: str | None = None
) -> np.ndarray:
    """Text embeddings are fixed for a given vocabulary, so they are computed
    once per process rather than once per song."""
    import torch

    cache_key = f"{resolve_model(model_id)}|{cache_key}"
    if cache_key in _text_cache:
        return _text_cache[cache_key]

    model, processor = _get_model(model_id)

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


def _standout(similarity: np.ndarray) -> np.ndarray:
    """How far each label stands out from its axis, on a 0-1 scale.

    This - not the softmax probability - is the confidence signal.

    A softmax probability cannot measure confidence here because it is
    dominated by `temperature`: at the 0.04-0.07 settings these axes use, the
    winner takes 0.15-0.4 of the mass whether or not the model can tell the
    labels apart. Measured across six tracks and eight axes, every softmax
    threshold fired on every track - the gates were dead code, which is why a
    wrong call was never suppressed.

    A raw z-score fixes the temperature dependence but introduces a subtler
    one: z is bounded by sqrt(n_labels - 1), so the same z means very different
    things on a 4-label axis (ceiling 1.73) and a 64-label axis (ceiling 7.94).
    Comparing those directly repeats the original mistake in a new coordinate
    system. Dividing by the ceiling gives a fraction-of-maximum-possible
    standout that is comparable everywhere.
    """
    mean = similarity.mean(axis=1, keepdims=True)
    std = similarity.std(axis=1, keepdims=True) + 1e-9
    z = (similarity - mean) / std
    ceiling = max(np.sqrt(max(similarity.shape[1] - 1, 1)), 1e-9)
    return z / ceiling


def score_axis(
    audio_embeddings: np.ndarray, spec: dict, axis: str, model_id: str | None = None
) -> dict:
    """Score one vocabulary axis against every window.

    Similarities are softmaxed *within the axis*, which is what makes the
    numbers comparable: raw CLAP cosine similarities sit in a narrow band and
    are not interpretable on their own.
    """
    labels = list(spec["labels"])
    templates = list(spec.get("prompts") or [spec["prompt"]])

    # Every template for every label, flattened into one batch.
    prompts = [
        template.format(label=label) for label in labels for template in templates
    ]
    # Key on the prompt text itself. Keying on the label count would silently
    # reuse stale embeddings when a vocabulary is edited without changing
    # length - which is exactly what happens while tuning wording.
    digest = hashlib.sha1("\n".join(prompts).encode("utf-8")).hexdigest()[:16]
    flat = _embed_text(prompts, cache_key=f"{axis}:{digest}", model_id=model_id)

    if len(templates) > 1:
        # Mean of the unit vectors for each label's templates, renormalised.
        # Averaging before normalising would let a longer template dominate.
        grouped = flat.reshape(len(labels), len(templates), -1)
        text_embeddings = _normalise(grouped.mean(axis=1))
    else:
        text_embeddings = flat

    # (n_windows, n_labels)
    similarity = audio_embeddings @ text_embeddings.T

    per_window = np.stack(
        [_softmax(row, spec["temperature"]) for row in similarity], axis=0
    )
    mean_probability = per_window.mean(axis=0)
    mean_z = _standout(similarity).mean(axis=0)

    order = np.argsort(mean_z)[::-1]
    min_z = float(spec.get("min_z", DEFAULT_MIN_Z))

    def entry(index: int) -> dict:
        return {
            "label": labels[index],
            "score": round(float(mean_probability[index]), 4),
            "z": round(float(mean_z[index]), 3),
        }

    if spec.get("mode") == "exclusive":
        winner = int(order[0])
        confident = float(mean_z[winner]) >= min_z
        outcome = (
            (spec.get("outcomes") or {}).get(labels[winner]) if confident else None
        )
        return {
            "top": [entry(winner)] if confident else [],
            "outcome": outcome,
            "confident": confident,
            "all": {labels[i]: round(float(mean_z[i]), 3) for i in order[:6]},
        }

    selected = []
    for index in order[: spec["top_k"]]:
        if float(mean_z[index]) < min_z:
            break
        # The softmax gate is kept as a secondary filter so existing tuned
        # thresholds still apply, but z is what actually decides.
        if float(mean_probability[index]) < spec.get("threshold", 0.0):
            break
        selected.append(entry(index))

    return {
        "top": selected,
        "all": {labels[i]: round(float(mean_z[i]), 3) for i in order[:8]},
        # Per-window winners let the composer talk about how the arrangement
        # changes rather than describing the track as one static block.
        "per_window": [labels[int(np.argmax(row))] for row in similarity],
    }


def describe(
    samples_48k: np.ndarray,
    vocab_dir: str = VOCAB_DIR,
    verbose: bool = False,
    model_id: str | None = None,
) -> dict:
    """Score every axis. `samples_48k` must be mono float32 at 48 kHz."""
    axes = load_vocabularies(vocab_dir)
    windows = _windows(samples_48k, CLAP_SR)

    if verbose:
        print(f"[SongScribe] scoring {len(windows)} window(s) across {len(axes)} axes")

    audio_embeddings = _embed_audio(windows, model_id)

    results: dict = {"_windows": len(windows), "_model": resolve_model(model_id)}

    presence_spec = axes.pop("vocal_presence", None)
    vocal_state = None
    if presence_spec is not None:
        presence = score_axis(
            audio_embeddings, presence_spec, "vocal_presence", model_id
        )
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

        scored = score_axis(audio_embeddings, spec, axis, model_id)
        results[axis] = scored["top"]
        results[f"_{axis}_detail"] = scored

    return results
