"""Stage 1: deterministic DSP measurement.

Everything in here is measured, not guessed. These values are treated as ground
truth downstream - the CLAP layer in stage 2 is never allowed to override a
number that was actually measured from the signal.

Performance note: the expensive transforms (STFT, HPSS, CQT chroma) are
computed exactly once in `analyse` and passed down. Letting each extractor call
librosa directly is far more readable, but it recomputed HPSS three separate
times and made a 40 s track take 20 s to analyse.
"""

from __future__ import annotations

import math

import numpy as np

from .audio_io import LoadedAudio
from .keys import estimate_key

HOP_LENGTH = 512
N_FFT = 2048

# Key estimation averages chroma over the whole track, so it gains nothing from
# 512-sample resolution. A 4x coarser hop cuts CQT cost by the same factor.
CHROMA_HOP = 2048

# Section boundaries are quoted to the nearest second in the caption, so
# analysing the similarity matrix at ~93 ms resolution is ample.
SEGMENT_DECIMATION = 4

# Global spectral averages are unchanged by looking at every 4th frame.
SPECTRAL_DECIMATION = 4

# HPSS median filtering is by far the most expensive operation in the pipeline
# and scales with total frames. The percussive/harmonic ratio is a global
# statistic, so it is estimated from evenly spaced excerpts instead of the
# whole track: measured error is ~0.002 against the full-resolution result for
# roughly a sixth of the cost.
HPSS_WINDOWS = 6
HPSS_WINDOW_FRAMES = 172  # ~4 s at HOP_LENGTH / 22050 Hz (~24 s sampled total)


def _safe_db(value: float, floor: float = -80.0) -> float:
    if value <= 1e-10:
        return floor
    return float(max(floor, 20.0 * math.log10(value)))


def _tempo_descriptor(bpm: float) -> str:
    for limit, word in (
        (60, "very slow"),
        (76, "slow"),
        (96, "relaxed"),
        (120, "mid-tempo"),
        (144, "upbeat"),
        (176, "fast"),
    ):
        if bpm < limit:
            return word
    return "very fast"


def _brightness_descriptor(centroid_hz: float) -> str:
    # Phrased to read correctly after "tonally ..." in the caption.
    for limit, word in (
        (900, "dark and muffled"),
        (1600, "warm and rounded"),
        (2600, "balanced"),
        (4000, "bright"),
    ):
        if centroid_hz < limit:
            return word
    return "very bright and airy"


def _dynamics_descriptor(crest_db: float) -> str:
    if crest_db < 6:
        return "heavily compressed and loud"
    if crest_db < 10:
        return "compressed"
    if crest_db < 15:
        return "moderately dynamic"
    return "wide open dynamics"


def _percussive_ratio(stft: np.ndarray) -> float:
    """Fraction of spectral energy that is percussive rather than harmonic.

    Estimated from evenly spaced excerpts - see HPSS_WINDOWS above for why.
    """
    import librosa

    total_frames = stft.shape[1]
    budget = HPSS_WINDOWS * HPSS_WINDOW_FRAMES

    if total_frames <= budget:
        sample = stft
    else:
        starts = np.linspace(0, total_frames - HPSS_WINDOW_FRAMES, HPSS_WINDOWS)
        sample = np.concatenate(
            [stft[:, int(s) : int(s) + HPSS_WINDOW_FRAMES] for s in starts], axis=1
        )

    harm_spec, perc_spec = librosa.decompose.hpss(sample)
    h_energy = float(np.sum(np.abs(harm_spec) ** 2))
    p_energy = float(np.sum(np.abs(perc_spec) ** 2))
    return p_energy / (h_energy + p_energy + 1e-9)


def _shared_transforms(y: np.ndarray, sr: int) -> dict:
    """Compute every expensive representation once, at the coarsest resolution
    each consumer can actually make use of."""
    import librosa

    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    mel_power = librosa.feature.melspectrogram(S=magnitude**2, sr=sr)
    mel_db = librosa.power_to_db(mel_power)

    # Key wants CQT chroma (better pitch-class resolution); segmentation only
    # needs relative harmonic change, which STFT chroma captures for a tenth of
    # the cost. Using the right one for each job rather than one for both.
    try:
        chroma_key = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=CHROMA_HOP)
    except Exception:
        chroma_key = librosa.feature.chroma_stft(S=magnitude, sr=sr)
    chroma_struct = librosa.feature.chroma_stft(S=magnitude, sr=sr)

    onset_env = librosa.onset.onset_strength(S=mel_db, sr=sr, hop_length=HOP_LENGTH)

    return {
        "magnitude": magnitude,
        "magnitude_decimated": magnitude[:, ::SPECTRAL_DECIMATION],
        "mel_db": mel_db,
        "chroma_key": chroma_key,
        "chroma_struct": chroma_struct,
        "onset_env": onset_env,
        "freqs": librosa.fft_frequencies(sr=sr, n_fft=N_FFT),
        "percussive_ratio": _percussive_ratio(stft),
    }


def analyse(audio: LoadedAudio) -> dict:
    """Extract the full deterministic feature set from a loaded signal."""
    y = audio.samples
    sr = audio.sample_rate

    ctx = _shared_transforms(y, sr)

    tempo_info = _tempo(ctx, sr)
    key_info = _key(ctx)
    loudness_info = _loudness(y)
    timbre_info = _timbre(y, sr, ctx)
    balance_info = _balance(ctx)
    structure_info = _structure(y, sr, ctx)

    return {
        "duration": round(audio.duration, 3),
        "duration_str": _format_duration(audio.duration),
        "native_sample_rate": audio.native_sample_rate,
        "native_channels": audio.native_channels,
        "tempo": tempo_info,
        "key": key_info,
        "loudness": loudness_info,
        "timbre": timbre_info,
        "balance": balance_info,
        "structure": structure_info,
    }


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _tempo(ctx: dict, sr: int) -> dict:
    import librosa

    tempo, beats = librosa.beat.beat_track(
        onset_envelope=ctx["onset_env"], sr=sr, hop_length=HOP_LENGTH, units="time"
    )
    bpm = float(np.atleast_1d(tempo)[0])

    # Tempo stability: how consistent are the inter-beat intervals? A tight
    # distribution means a programmed/quantised grid; a loose one means live
    # playing or rubato. This is what separates "steady" from "loose and human".
    stability = None
    swing = None
    if len(beats) > 4:
        intervals = np.diff(beats)
        intervals = intervals[intervals > 0]
        if intervals.size > 2:
            stability = float(np.std(intervals) / (np.mean(intervals) + 1e-9))
            # Swing: alternating long/short eighth subdivisions show up as a
            # systematic difference between odd and even inter-onset gaps.
            odd, even = intervals[::2], intervals[1::2]
            n = min(len(odd), len(even))
            if n >= 2:
                ratio = float(np.mean(odd[:n]) / (np.mean(even[:n]) + 1e-9))
                swing = round(abs(ratio - 1.0), 3)

    if stability is None:
        feel = "unknown"
    elif stability < 0.04:
        feel = "machine-tight"
    elif stability < 0.10:
        feel = "steady"
    else:
        feel = "loose and human"

    return {
        "bpm": round(bpm, 1),
        "bpm_int": int(round(bpm)),
        "beat_count": int(len(beats)),
        "stability": None if stability is None else round(stability, 4),
        "feel": feel,
        "swing": swing,
        "descriptor": _tempo_descriptor(bpm),
    }


def _key(ctx: dict) -> dict:
    chroma = ctx["chroma_key"]
    info = estimate_key(chroma)

    # Chroma entropy hints at harmonic complexity: a diatonic pop song
    # concentrates energy in 7 pitch classes, jazz smears it across 12.
    profile = chroma.mean(axis=1)
    profile = profile / (profile.sum() + 1e-9)
    entropy = float(-(profile * np.log2(profile + 1e-12)).sum())
    info["chroma_entropy"] = round(entropy, 3)
    info["harmonic_complexity"] = (
        "simple diatonic" if entropy < 3.2
        else "moderately extended" if entropy < 3.6
        else "chromatic / extended harmony"
    )
    return info


def _loudness(y: np.ndarray) -> dict:
    # Sliding-window RMS via a cumulative sum. np.convolve with a 2048-tap
    # kernel is O(n*k) and cost seconds on a five-minute track; this is O(n).
    window = 2048
    squared = y.astype(np.float64) ** 2
    if squared.size > window:
        cumulative = np.concatenate([[0.0], np.cumsum(squared)])
        rms_frames = np.sqrt(
            (cumulative[window:] - cumulative[:-window]) / window
        )
    else:
        rms_frames = np.sqrt(squared)

    rms = float(np.sqrt(np.mean(squared)))
    peak = float(np.max(np.abs(y)))
    crest_db = _safe_db(peak) - _safe_db(rms)

    # Loudness range: spread between quiet and loud passages.
    if rms_frames.size:
        quiet = _safe_db(float(np.percentile(rms_frames, 10)))
        loud = _safe_db(float(np.percentile(rms_frames, 95)))
        loudness_range = round(loud - quiet, 2)
    else:
        loudness_range = None

    return {
        "rms_db": round(_safe_db(rms), 2),
        "peak_db": round(_safe_db(peak), 2),
        "crest_db": round(crest_db, 2),
        "loudness_range_db": loudness_range,
        "clipping": bool(peak >= 0.999),
        "descriptor": _dynamics_descriptor(crest_db),
    }


def _timbre(y: np.ndarray, sr: int, ctx: dict) -> dict:
    import librosa

    magnitude = ctx["magnitude_decimated"]
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=magnitude, sr=sr)))
    rolloff = float(
        np.mean(librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=0.95))
    )
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=magnitude)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)))

    return {
        "centroid_hz": round(centroid, 1),
        "rolloff_hz": round(rolloff, 1),
        "bandwidth_hz": round(bandwidth, 1),
        "flatness": round(flatness, 5),
        "zero_crossing_rate": round(zcr, 5),
        "brightness": _brightness_descriptor(centroid),
        # High flatness = noise-like content (cymbals, vinyl crackle, tape hiss).
        "noisiness": "noisy / textured" if flatness > 0.02 else "tonal",
    }


def _balance(ctx: dict) -> dict:
    """Harmonic vs percussive energy split, plus the spectral band profile."""
    percussive_ratio = ctx["percussive_ratio"]

    # Band split on the power spectrum. Fractions of magnitude are misleading
    # here: the sub band spans well under 1% of the bins, so even a dominant
    # bass line looks like a small number. Power fractions reflect what is
    # actually audible as weight in the mix.
    power = np.sum(ctx["magnitude"] ** 2, axis=1)
    freqs = ctx["freqs"]
    band_total = float(np.sum(power) + 1e-9)

    def band(low_hz, high_hz=None):
        mask = freqs >= low_hz
        if high_hz is not None:
            mask &= freqs < high_hz
        return float(np.sum(power[mask])) / band_total

    sub = band(0, 80)
    bass = band(80, 250)
    low_mid = band(250, 800)
    mid = band(800, 2500)
    high = band(6000)
    low_total = sub + bass

    return {
        "percussive_ratio": round(percussive_ratio, 3),
        "drum_presence": (
            "drum-forward" if percussive_ratio > 0.50
            else "balanced" if percussive_ratio > 0.30
            else "harmonically led"
        ),
        "sub_energy": round(sub, 4),
        "bass_energy": round(bass, 4),
        "low_mid_energy": round(low_mid, 4),
        "mid_energy": round(mid, 4),
        "high_energy": round(high, 4),
        "low_end": (
            "deep sub-heavy low end" if sub > 0.35
            else "solid bass weight" if low_total > 0.35
            else "moderate low end" if low_total > 0.15
            else "light low end"
        ),
    }


def _structure(y: np.ndarray, sr: int, ctx: dict) -> dict:
    """Segment the track into sections by timbral/harmonic self-similarity.

    This is what later lets the composer say things like 'drums slip in halfway'
    and 'the bridge drops away' rather than describing the song as one flat block.
    """
    import librosa

    duration = len(y) / sr
    # Aim for sections of roughly 20 s, clamped to something sane.
    n_segments = int(np.clip(round(duration / 20.0), 2, 12))

    try:
        mfcc = librosa.feature.mfcc(S=ctx["mel_db"], n_mfcc=13)
        chroma = ctx["chroma_struct"]
        n = min(mfcc.shape[1], chroma.shape[1])
        stacked = np.vstack(
            [
                librosa.util.normalize(mfcc[:, :n], axis=0),
                librosa.util.normalize(chroma[:, :n], axis=0),
            ]
        )
        stacked = np.nan_to_num(stacked)[:, ::SEGMENT_DECIMATION]
        bounds = librosa.segment.agglomerative(stacked, n_segments)
        bound_times = librosa.frames_to_time(
            bounds, sr=sr, hop_length=HOP_LENGTH * SEGMENT_DECIMATION
        )
    except Exception:
        bound_times = np.linspace(0, duration, n_segments + 1)[:-1]

    edges = list(np.unique(np.concatenate([[0.0], bound_times, [duration]])))
    sections = []
    for start, end in zip(edges[:-1], edges[1:]):
        # Sub-5s slivers are clustering artefacts, not musical sections, and
        # they produce nonsense entries in the arrangement's section map.
        if end - start < 5.0:
            continue
        seg = y[int(start * sr) : int(end * sr)]
        if seg.size < sr // 2:
            continue
        seg_rms = float(np.sqrt(np.mean(seg**2)))
        sections.append(
            {
                "start": round(float(start), 2),
                "end": round(float(end), 2),
                "duration": round(float(end - start), 2),
                "rms_db": round(_safe_db(seg_rms), 2),
            }
        )

    if sections:
        levels = np.array([s["rms_db"] for s in sections])
        quietest = int(np.argmin(levels))
        loudest = int(np.argmax(levels))
        median = float(np.median(levels))
        for s in sections:
            rel = s["rms_db"] - median
            s["relative_energy"] = round(rel, 2)
            s["energy_label"] = (
                "sparse" if rel < -3 else "full" if rel > 3 else "steady"
            )
        arc = "quiet-open" if quietest == 0 else "loud-open"
        arc += ", peaks late" if loudest > len(sections) // 2 else ", peaks early"
    else:
        arc = "unknown"

    return {
        "section_count": len(sections),
        "sections": sections,
        "arc": arc,
        "mean_section_length": (
            round(float(np.mean([s["duration"] for s in sections])), 2)
            if sections
            else None
        ),
    }
