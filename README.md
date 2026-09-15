# SongScribe

ComfyUI nodes for driving music-generation models: **style prompts** from a
curated preset library, and **caption / lyrics / duration** extracted from an
audio file.

Targets two prompt formats:

- **[MiniMax Music 3](https://github.com/MiniMax-AI/MiniMax-Music3)** — the
  three-section caption (`Global Metadata` / `Vocal Details` / `Arrangement`)
- **[YuE2](https://map-yue2.github.io/)** — a flat comma-separated descriptor
  string

Outputs are plain `STRING` / `FLOAT`, so they drop into any audio workflow.
Everything runs on CPU.

---

## Install

Clone into `ComfyUI/custom_nodes/`, then install the dependencies into the same
Python that runs ComfyUI. For a Windows portable install:

```bash
python_embeded/python.exe -m pip install librosa mutagen pyyaml
```

Purely additive — they don't upgrade or downgrade numpy, torch or anything else
ComfyUI depends on. ComfyUI Manager installs `requirements.txt` for you.

Optional, only for lyric transcription:

```bash
python_embeded/python.exe -m pip install faster-whisper
```

## Nodes

| Node | Does |
|---|---|
| **Style Preset** | Curated style → prompt, in MiniMax or YuE2 format. No audio needed. |
| **Song Analyzer** | Audio → caption, lyrics, duration |
| **Caption Splitter** | Caption → three editable sections |
| **Caption Composer** | Three sections → caption |
| **Lyrics Structure** | Normalise section tags, check they fit the duration |

---

# Style Preset

73 presets across 14 categories, covering the genre taxonomy YuE2 publishes.
The dropdown shows readable names (`World / Reggae`), and any `.yaml` you drop
into `songscribe/presets/` appears there on the next restart.

| Input | Notes |
|---|---|
| `preset` | The style |
| `format` | `minimax` (three sections) or `yue2` (flat descriptors) |
| `detail` | *YuE2 only.* `tags` / `full` / `rich` — their own prompts span all three lengths |
| `vocal` | *YuE2 only.* Voice type; `auto` uses whatever the preset declares |
| `language` | *YuE2 only.* English, Mandarin, Japanese, Korean, Spanish, Russian |
| `style` | *MiniMax only.* `verbatim` / `balanced` / `loose` |
| `era`, `texture`, `mood_shift` | Modifier axes; they **add**, never replace |
| `blend_with` + `blend` | Mix a second preset |
| `extra` | Appended verbatim |

`seed` varies MiniMax phrasing only; it does nothing in `yue2` format, which is
deterministic. `style` likewise does nothing in `yue2`, and `detail`/`vocal`/
`language` do nothing in `minimax`. They're harmless, just inert.

Example, `World / Reggae` in `yue2` format:

```
reggae, roots reggae, dub, rocksteady, jamaican, 76 BPM, laid-back and dreamy,
offbeat guitar skank on the upstroke, one-drop drums with the kick on the three,
deep melodic round bassline, Hammond organ bubble, delay throws on the end of
phrases, warm mid-range voice, sung lazily behind the beat, spring reverb and
dub delay throws, English
```

### How the presets are written

**Name the signature, not the ingredients.** Every genre has drums, bass and a
harmony instrument — listing those describes nothing. Drill is *sliding 808s and
skittering hi-hat triplets*. Reggae is the *offbeat skank and the one-drop kick
on the three*. Bluegrass is the *banjo roll, the mandolin chop, and no drums at
all*.

**Stack related genre terms.** A broad term like "hip hop" is far better
represented in training data than "drill", so presets lead with the specific
term and back it with its family (`drill, hip hop, rap, trap`). This mirrors
YuE2's own prompts, which routinely name four or five related styles. Genres
are named generally rather than regionally — "drill", not "UK drill" — since the
signature elements are shared across scenes. Regional flavour goes in `extra`.

**Presets are structured, not prose**, which is what makes blending meaningful:
merging two structured presets is a list operation, where blending two
paragraphs isn't well defined. Scalars like BPM cross over at the halfway point
rather than averaging — the mean of 78 and 132 BPM is a tempo neither preset
asked for.

Regenerate the shipped set with `python tools/make_presets.py`; edit the YAML
directly for one-offs.

### Wiring to YuE2

`style` is a widget on `YuE2GenerateMusic`, so right-click the node → **convert
`style` to input**, then connect the Style Preset's `prompt` output.

---

# Song Analyzer

**Nothing in the caption is guessed.** BPM, key, dynamics, spectral balance and
the section map are measured by DSP. Where a value can't be measured or scored
confidently, the clause is *omitted* rather than filled with something
plausible — every sentence in a caption becomes an instruction to the music
model, so saying less beats saying something wrong.

| Input | Notes |
|---|---|
| `audio_file` | Upload widget. Set to `(use AUDIO input)` when driving from a socket. |
| `audio` *(optional)* | `AUDIO` from an upstream node; takes priority when connected |
| `describe` | `clap` scores mood/instruments/vocals; `off` emits measured facts only |
| `genre_source` | `clap`, `maest` (supervised, more accurate), or `off` |
| `clap_model` | `music_and_speech` (default) or `general` |
| `transcribe_lyrics` | `off` / `if missing` / `always` |
| `whisper_model` | `tiny` / `base` / `small` / `medium` |
| `style` | How literally the caption reproduces the track |
| `use_cache`, `seed` | |

| Output | Type | Wire to |
|---|---|---|
| `caption` | `STRING` | MiniMax `caption` |
| `lyrics` | `STRING` | MiniMax `lyrics` |
| `duration` | `FLOAT` | MiniMax `max_duration` |
| `duration_int` / `duration_str` | `INT` / `STRING` | `3:47` for filenames |
| `analysis` | `SONGSCRIBE_ANALYSIS` | Downstream SongScribe nodes |

### The `style` dial

Feeding a verbatim analysis back into a generator produces a clone of the
source. This is the knob that stops it.

| Style | Keeps |
|---|---|
| `verbatim` | Exact BPM, key, second-level section timings |
| `balanced` *(default)* | BPM and key; drops exact timings |
| `loose` | Genre, mood, texture only |

Style affects composition only, so switching recomposes instantly from cached
measurements.

### Formats

Anything libsndfile or ffmpeg can decode: wav, flac, mp3, m4a/aac, ogg, opus,
aiff, wma, alac, ape. librosa 1.0 dropped its audioread fallback, so formats
libsndfile can't open are decoded through PyAV, which ships with ComfyUI.

### Lyrics

Three sources, in order of how much they can be trusted:

1. **Embedded tags** (`USLT`/`SYLT`/Vorbis/MP4) — exact
2. **A sibling `.lrc`/`.txt`** — exact. `.lrc` is trusted as-is; a `.txt` must
   actually look like lyrics (short lines, no prose paragraphs), since it could
   be credits or liner notes
3. **Whisper transcription** — an estimate, off by default

Section tags are **not** taken from the ASR. Whisper emits words and timings and
knows nothing about song structure. Tags come from silence between sung phrases
(→ `[Instrumental]`) and repetition of the lyric text — a block occurring more
than once is a `[Chorus]` by definition. Everything else is `[Verse]`.

### Caching

The first analysis writes a `<name>.songscribe.json` sidecar; later runs reuse
it. ComfyUI re-executes a node whenever anything upstream changes, and analysis
takes seconds — cached runs are ~100× faster. Falls back to ComfyUI's temp
directory if the audio's folder isn't writable.

### Performance

CPU only, warm process: ~5s for a 40s track, ~14s for 5 minutes, ~0.15s cached.
Add ~10s once per session for numba's JIT warm-up, and ~2s per track for CLAP
or MAEST scoring.

---

# Lyrics Structure

MiniMax treats bracketed section tags as the **only** executable structural
instruction — the lyric text itself just conveys mood. A malformed tag doesn't
produce a slightly-off song, it silently drops structure from a render that may
take minutes.

`normalise_tags` rewrites the tags in place; turn it off to validate without
changing the text. `max_duration` enables the fit check (0 disables it).

Normalises `(intro)`, `Verse 1:`, `{HOOK}`, `[middle 8]`, `ending:` into
`[Intro]` `[Verse]` `[Chorus]` `[Bridge]` `[Outro]`, and estimates whether the
lyrics fit `max_duration` — warning in both directions. The estimate is a
**range**, since delivery speed differs enormously between a ballad and a rap
verse. A lyric line that merely ends in a colon is not mistaken for a tag.

---

# Caption Splitter / Composer

The round-trip pair. The useful edit is almost always to *one* section — keep
the measured arrangement, replace the vocal description entirely — so these make
that a graph operation instead of copy-paste.

**Splitter** takes a `caption` and emits `global_metadata`, `vocal_details` and
`arrangement`. Degenerate input is preserved rather than dropped: a caption with
no headers comes back whole in `global_metadata`, text before the first header
survives, and markdown-bold headers (`**Arrangement:**`) are handled.

**Composer** takes those three back and rebuilds the caption. `headers` emits
the `Global Metadata:` / `Vocal Details:` / `Arrangement:` labels — MiniMax
expects them; turn it off for other models. Any header left in an input is
stripped first, so enabling it can't produce `Arrangement: Arrangement: ...`.

Both work with the Style Preset node, which always computes the three sections
regardless of the selected `format`.

---

## Measured accuracy, and where it's weak

Scored against six labelled tracks with `tests/evaluate.py`.

| Axis | Result |
|---|---|
| Vocal presence (voice / no voice) | 6/6 |
| Sung vs rapped | 5/6 |
| Genre via `maest` | trap, reggae, heavy metal, contemporary R&B all correct |
| Genre via `clap` | ~3/6, and confidently wrong when wrong |
| Key — mode only | 2/2 |
| Key — exact tonic | 0/3 |
| BPM vs label | 1/5 — see caveat |

**The BPM caveat:** those labels are generation *prompts*, not measurements of
the finished audio. Where the label and the analyzer disagreed on tempo, the
audio's own onset autocorrelation backed the analyzer in 4 of 5 cases — on one
track the correlation at the labelled 96 BPM was *negative* versus 0.421 at the
detected tempo. So that figure substantially measures how closely a generator
honoured its own prompt, not this analyzer's accuracy.

**Known weak spots, stated plainly:**

- **`mood` and `vocal_timbre` barely discriminate.** Across six unrelated
  tracks, `mood` returned the same top label on five of them and `vocal_timbre`
  on four. Treat them as decoration; a preset supplies both far more reliably.
- **`genre` via CLAP is a coin flip.** Confidence does *not* predict
  correctness there — the two worst calls scored highest. Use `maest`.
- **Vocal gender was removed from the vocabulary.** CLAP scored 2/5 on a binary
  male/female question — worse than chance — and answered "female" at 0.80–0.90
  confidence on three male tracks. A caption is an instruction, so a wrong
  gender claim generates the wrong voice.
- **Descriptor thresholds are calibrated on six tracks.** Enough to catch a
  systematic failure, not enough to be settled. Re-run `tools/calibrate.py`
  with more labelled audio.

### Accuracy notes

- **Key** is Krumhansl-Schmuckler profile correlation. Confidence is scored
  against the best *non-relative* alternative, since a key and its relative
  minor share all seven pitch classes. Where the margin is tight the caption
  names both: *"B flat major (or its relative G minor)"*.
- **BPM** can land on half or double time. Inherent to beat tracking.
- **Section boundaries** come from timbral self-similarity. They mark where the
  music changes, not *what* a section is — verse/chorus labelling isn't
  something this can honestly claim, so sections are named positionally.

## Genre: supervised tagging

CLAP guesses genre by embedding text and audio near each other — it never saw
"reggae" as a training label.
[MAEST](https://huggingface.co/mtg-upf/discogs-maest-10s-pw-129e) is
*supervised* on 400 Discogs styles, so it did. Same ~2s per track on CPU.

Set `genre_source` to `maest`. Only genre changes; mood, instruments and vocal
character stay with CLAP, which is what MAEST doesn't predict.

**Why it isn't the default — `trust_remote_code`.** MAEST ships a custom feature
extractor, so loading it executes Python from the model repository. Mitigated
rather than dismissed:

- **Opt-in** — nothing loads unless selected
- **Pinned revision** — `songscribe/tagger.py` pins commit `54b3b0a`, so a later
  change to that repo can't silently execute on installed users
- **Audited** — the pinned file is a 242-line mel-spectrogram extractor
  importing only numpy, torch and transformers' audio utilities; no network, no
  subprocess, no `eval`/`exec`, no file access

Re-point `REVISION` at a newer commit only after reading that file.

## Tests

No ComfyUI required — they stub out `folder_paths`:

```bash
python_embeded/python.exe custom_nodes/ComfyUI-SongScribe/tests/smoke_test.py
```

| Suite | Checks |
|---|---|
| `smoke_test` | Synthesises a 78 BPM D♭ major track, verifies measurements land on it |
| `format_test` | Transcodes to every supported container, verifies each loads |
| `node_test` | Loads the pack through ComfyUI's importlib path, executes end to end |
| `compose_test` | Style dial reduces specificity monotonically; seeds reproduce |
| `companion_test` | Presets, splitter round-trip, lyric tag normalisation |
| `clap_test` / `transcribe_test` | Descriptor and ASR layers |

`SONGSCRIBE_TEST_MODEL=general` keeps the descriptor tests on an already-cached
checkpoint so the suite doesn't pull gigabytes.

## Tools

| Tool | For |
|---|---|
| `tools/make_presets.py` | Regenerate the shipped preset library |
| `tools/calibrate.py` | Re-tune per-axis thresholds against labelled audio |
| `tools/evaluate.py` *(tests/)* | Score the analyzer per-axis against labels |
| `tools/compare_models.py` *(tests/)* | A/B CLAP checkpoints |
| `tools/try_maest.py` | Compare MAEST against CLAP on your own files |

## License

MIT — see [LICENSE](LICENSE).
