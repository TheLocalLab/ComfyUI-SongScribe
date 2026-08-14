"""Audio loading for both of the node's input paths: an uploaded file on disk,
or an AUDIO tensor arriving from an upstream node."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# Everything librosa/soundfile/audioread can plausibly open. The upload widget
# filters on this list, so being generous here is what makes the node accept
# "a variety of different formats".
SUPPORTED_EXTENSIONS = (
    ".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".aac",
    ".wma", ".aif", ".aiff", ".aifc", ".alac", ".ape", ".wv", ".mka",
    ".webm", ".oga", ".spx", ".caf", ".au", ".snd",
)

# Analysis sample rate. 22050 is the librosa default and is plenty for tempo,
# key and timbre features while keeping CPU cost down; we are not doing
# anything that needs content above ~11 kHz.
ANALYSIS_SR = 22050


@dataclass
class LoadedAudio:
    """Mono float32 signal at ANALYSIS_SR, plus provenance."""

    samples: np.ndarray
    sample_rate: int
    duration: float
    source_path: str | None = None
    native_sample_rate: int | None = None
    native_channels: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def has_file(self) -> bool:
        return bool(self.source_path) and os.path.isfile(self.source_path)

    def samples_at(self, target_sr: int) -> np.ndarray:
        """Mono float32 at an arbitrary rate, for consumers that need more
        bandwidth than the analysis rate carries (CLAP wants 48 kHz).

        Re-decodes from the original file when one exists, because upsampling
        the 22.05 kHz analysis signal cannot recover the content above 11 kHz
        that a 48 kHz model was trained to use.
        """
        if target_sr == self.sample_rate:
            return self.samples

        cached = self.meta.get(f"samples_{target_sr}")
        if cached is not None:
            return cached

        result = None
        if self.has_file:
            try:
                import librosa

                result, _ = librosa.load(self.source_path, sr=target_sr, mono=True)
            except Exception:
                try:
                    result = _decode_with_av(self.source_path, target_sr)[0]
                except Exception:
                    result = None

        if result is None or result.size == 0:
            import librosa

            result = librosa.resample(
                self.samples, orig_sr=self.sample_rate, target_sr=target_sr
            )

        result = result.astype(np.float32, copy=False)
        self.meta[f"samples_{target_sr}"] = result
        return result


class AudioLoadError(RuntimeError):
    pass


def _require_librosa():
    try:
        import librosa  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise AudioLoadError(
            "SongScribe needs librosa. Install it into the ComfyUI python:\n"
            "  python_embeded\\python.exe -m pip install librosa mutagen pyyaml"
        ) from exc
    import librosa

    return librosa


def load_from_path(path: str) -> LoadedAudio:
    librosa = _require_librosa()

    if not os.path.isfile(path):
        raise AudioLoadError(f"Audio file not found: {path}")

    native_sr = None
    native_ch = None
    try:
        import soundfile as sf

        info = sf.info(path)
        native_sr = int(info.samplerate)
        native_ch = int(info.channels)
    except Exception:
        # soundfile can't probe every container (m4a/wma go through audioread).
        # Not fatal - these fields are descriptive only.
        pass

    # librosa 1.0 removed the audioread fallback, so it can only open what
    # libsndfile understands - which excludes m4a/aac/wma/opus. PyAV ships with
    # ComfyUI and decodes essentially anything, so it covers the difference.
    samples = None
    sr = ANALYSIS_SR
    first_error: Exception | None = None
    try:
        samples, sr = librosa.load(path, sr=ANALYSIS_SR, mono=True)
    except Exception as exc:
        first_error = exc

    if samples is None or samples.size == 0:
        try:
            samples, native = _decode_with_av(path)
            sr = ANALYSIS_SR
            if native_sr is None:
                native_sr = native
        except Exception as av_error:
            raise AudioLoadError(
                f"Could not decode {os.path.basename(path)}.\n"
                f"  libsndfile: {first_error}\n"
                f"  PyAV: {av_error}"
            ) from (first_error or av_error)

    if samples.size == 0:
        raise AudioLoadError(f"{os.path.basename(path)} decoded to zero samples.")

    return LoadedAudio(
        samples=samples.astype(np.float32, copy=False),
        sample_rate=sr,
        duration=float(len(samples) / sr),
        source_path=path,
        native_sample_rate=native_sr,
        native_channels=native_ch,
    )


def _decode_with_av(
    path: str, target_sr: int = ANALYSIS_SR
) -> tuple[np.ndarray, int | None]:
    """Decode any container PyAV/ffmpeg supports to mono float32 at target_sr.

    Returns (samples, native_sample_rate).
    """
    import av
    from av.audio.resampler import AudioResampler

    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError("file contains no audio stream")

        native_sr = int(stream.rate) if stream.rate else None
        stream.thread_type = "AUTO"

        resampler = AudioResampler(format="fltp", layout="mono", rate=target_sr)

        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().ravel())

        # Flush whatever the resampler is still holding, or the tail of the
        # track goes missing and the reported duration comes up short.
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().ravel())

    if not chunks:
        raise RuntimeError("decoded zero frames")

    return np.concatenate(chunks).astype(np.float32), native_sr


def load_from_comfy_audio(audio: dict) -> LoadedAudio:
    """Convert ComfyUI's AUDIO dict ({'waveform': [B,C,T], 'sample_rate': int})."""
    librosa = _require_librosa()

    if not isinstance(audio, dict) or "waveform" not in audio:
        raise AudioLoadError("AUDIO input was not a valid ComfyUI audio dict.")

    waveform = audio["waveform"]
    sr = int(audio.get("sample_rate") or ANALYSIS_SR)

    array = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform)

    # [B, C, T] -> take the first batch item, average channels to mono.
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 2:
        native_ch = int(array.shape[0])
        array = array.mean(axis=0)
    else:
        native_ch = 1
    array = np.asarray(array, dtype=np.float32).ravel()

    if array.size == 0:
        raise AudioLoadError("AUDIO input contained zero samples.")

    if sr != ANALYSIS_SR:
        array = librosa.resample(array, orig_sr=sr, target_sr=ANALYSIS_SR)

    return LoadedAudio(
        samples=array.astype(np.float32, copy=False),
        sample_rate=ANALYSIS_SR,
        duration=float(len(array) / ANALYSIS_SR),
        source_path=None,
        native_sample_rate=sr,
        native_channels=native_ch,
    )


def list_input_audio_files() -> list[str]:
    """Filenames in ComfyUI's input dir that we can plausibly open."""
    try:
        import folder_paths
    except ImportError:  # running outside ComfyUI (tests)
        return []

    input_dir = folder_paths.get_input_directory()
    if not os.path.isdir(input_dir):
        return []

    files = []
    for name in os.listdir(input_dir):
        full = os.path.join(input_dir, name)
        if os.path.isfile(full) and name.lower().endswith(SUPPORTED_EXTENSIONS):
            files.append(name)
    return sorted(files)


def resolve_input_path(filename: str) -> str:
    try:
        import folder_paths

        return folder_paths.get_annotated_filepath(filename)
    except ImportError:
        return filename
