# SongScribe

ComfyUI nodes that take an audio file and produce a **structured music caption**,
**lyrics**, and **duration** — shaped for [MiniMax Music 3](https://github.com/MiniMax-AI/MiniMax-Music3)'s
three-section caption format, but emitted as plain `STRING`/`FLOAT` so they drop
into any audio workflow.

```
Global Metadata: 78 BPM, D flat major, moderately extended. Machine-tight timing.
Energy arc: quiet-open, peaks late. Production: tonally warm and rounded, ...

Vocal Details: ...

Arrangement: Mix sits drum-forward, with a solid bass weight. Section map — ...
```

## Design principle

**Nothing in the caption is guessed.** BPM, key, dynamics, spectral balance and
the section map are measured by DSP. Where a value can't be measured or scored,
the clause is *omitted* rather than filled with something plausible — a caption
that says less beats one that says something wrong, because every sentence in it
becomes an instruction to the music model.

This is also why there's no large language model in the pipeline. The caption
format is a fixed three-section template drawn from a bounded vocabulary, so
filling it is a measurement-and-retrieval problem, not a text-generation one.
The whole pack runs on CPU.

## Status

| Phase | What | State |
|---|---|---|
| 1 | DSP analyzer, multi-format loading, embedded lyrics, sidecar cache | **Done** |
| 2 | CLAP zero-shot descriptors (genre, mood, instruments, vocal character) | **Done** |
| 3 | Full composer grammar + style-abstraction dial | **Done** |
| 4 | Caption Splitter/Composer, Style Presets, Lyrics tools | **Done** |
| 5 | Optional `faster-whisper` lyric transcription | **Done** |

## How the descriptors work

The `describe` widget switches on CLAP scoring. There is no language model
involved: [`laion/clap-htsat-unfused`](https://huggingface.co/laion/clap-htsat-unfused)
(~150M params, CPU) embeds audio and text into a shared space, and SongScribe
ranks a **hand-authored vocabulary** against the track.

Every phrase the system can emit lives in `songscribe/vocab/*.yaml`. That's the
whole point: the model picks *from* a list rather than writing free text, so the
worst failure is a less apt word — never an invented fact. Each axis is one file:

| Axis | Picks | Feeds |
|---|---|---|
| `genre` | top 2 | Global Metadata |
| `mood` | top 3 | Global Metadata |
| `production` | top 3 | Global Metadata |
| `scene` | top 2 | Global Metadata |
| `instruments` | top 6 | Arrangement |
| `vocal_timbre` | top 2 | Vocal Details |
| `vocal_delivery` | top 2 | Vocal Details |
| `vocal_presence` | exclusive | gates the whole Vocal Details section |

Scores are softmaxed **within each axis** — raw CLAP cosine similarities sit in
a narrow band and aren't interpretable on their own. Labels below the axis
threshold are dropped rather than padded out to `top_k`.

`vocal_presence` is scored first and, on an instrumental, the vocal axes are
skipped entirely rather than reported with low scores. Describing a voice that
isn't there is the most damaging thing this layer could do to a caption.

### Editing the vocabulary

Drop a new `.yaml` into `songscribe/vocab/` and it becomes an axis on the next
run — no code changes. Tune `threshold` if an axis is too eager or too shy,
`top_k` for how many labels it may contribute, and `temperature` for how sharply
it discriminates.

**The vocabulary is the tuning surface.** If captions come out generic for the
music you work with, add the specific language you want to that genre's file
rather than reaching for a bigger model.

## Install

Clone into `ComfyUI/custom_nodes/`, then install the dependencies into the same
Python that runs ComfyUI. For a Windows portable install:

```bash
python_embeded/python.exe -m pip install librosa mutagen pyyaml
```

These are purely additive — they don't upgrade or downgrade numpy, torch or
anything else ComfyUI depends on.

## Nodes

| Node | Does |
|---|---|
| **Song Analyzer** | Audio → caption, lyrics, duration |
| **Style Preset** | Curated style → caption, no reference track needed |
| **Caption Splitter** | Caption → three editable sections |
| **Caption Composer** | Three sections → caption |
| **Lyrics Structure** | Normalise section tags, check they fit the duration |

## Node: Song Analyzer

**Inputs**

| Input | Notes |
|---|---|
| `audio_file` | Upload widget. Set to `(use AUDIO input)` when driving it from a socket. |
| `audio` *(optional)* | `AUDIO` from an upstream node. Takes priority over the file when connected. |
| `describe` | `clap` scores genre/mood/instruments/vocals; `off` emits measured facts only. |
| `style` | How literally the caption reproduces the track — see below. |
| `use_cache` | Reuse a previous analysis of the same file instead of recomputing. |
| `seed` | Varies caption phrasing without re-analysing the audio. |

### The `style` dial

Feeding a verbatim analysis of a song back into a generator produces a clone of
that song. This is the knob that stops it:

| Style | Keeps | Use for |
|---|---|---|
| `verbatim` | Exact BPM, key, and second-level section timings | Reproducing a reference as closely as possible |
| `balanced` *(default)* | BPM and key; drops exact timings | Steering the model while letting it write a new song |
| `loose` | Genre, mood, texture. No tempo, key or structure | Vibe transfer only |

Style affects composition only, never analysis — so switching it recomposes
instantly from the cached measurements rather than re-analysing the audio.

Section names come from position and energy only (`Intro`, `Section 3`,
`Outro`). The segmentation finds *where* the music changes, not *what* a
section is; labelling something a chorus would be a claim the analysis can't
support. Consecutive sections that behave identically are collapsed into one
entry (`Sections 2-6: the core groove running`).

**Outputs**

| Output | Type | Wire to |
|---|---|---|
| `caption` | `STRING` | MiniMax `caption` |
| `lyrics` | `STRING` | MiniMax `lyrics` |
| `duration` | `FLOAT` | MiniMax `max_duration` |
| `duration_int` | `INT` | — |
| `duration_str` | `STRING` | `3:47`, for filenames and notes |
| `analysis` | `SONGSCRIBE_ANALYSIS` | Downstream SongScribe nodes |

### Formats

Anything libsndfile or ffmpeg can decode: wav, flac, mp3, m4a/aac, ogg, opus,
aiff, wma, alac, ape and more. librosa 1.0 dropped its audioread fallback, so
formats libsndfile can't open are decoded through PyAV, which ships with ComfyUI.

### Lyrics

Three sources, tried in order of how much they can be trusted:

1. **Embedded tags** (`USLT`/`SYLT`/Vorbis/MP4) — exact, someone typed them.
2. **A sibling `.lrc`/`.txt`** — exact, timestamps stripped. `.lrc` is a
   dedicated lyric format and is trusted as-is; a `.txt` could be credits or
   liner notes, so it must actually look like lyrics (short lines, no prose
   paragraphs) before it's accepted.
3. **Whisper transcription** — an estimate, and off by default.

The estimate is never preferred over an exact source unless you set
`transcribe_lyrics` to `always`.

### Transcription quality

`transcribe_lyrics` is `off` by default because sung ASR is markedly worse than
speech. Measured on real tracks with `base` on CPU:

| Track | Speed | Word overlap with true lyrics |
|---|---|---|
| English R&B, clear lead vocal | 0.61× realtime | 90% |
| English reggae, dense mix | 0.12× realtime | 78% |
| Korean/English rap | 0.71× realtime | 52% |

Good enough to save typing, not good enough to ship unread — expect to fix
names, run-together lines, and hummed passages, where Whisper tends to repeat
itself. Bigger models help; `medium` is roughly 4× slower than `base`.

**Section tags are not taken from the ASR.** Whisper emits words and timings and
knows nothing about song structure. Tags come from two things it does report
honestly: silence between sung phrases (→ `[Instrumental]`), and repetition of
the lyric text itself — a block that occurs more than once is a `[Chorus]` by
definition of the word. Everything else is `[Verse]`, which claims only that it
is sung, non-repeating material.

### Caching

The first analysis of a file writes a `<name>.songscribe.json` sidecar; later
runs reuse it. This matters more than it sounds: ComfyUI re-executes a node
whenever anything upstream changes, and analysis takes seconds, not
milliseconds. If the audio's directory isn't writable, the cache falls back to
ComfyUI's temp directory. Cached runs are ~100× faster.

## Style Presets

Presets are **structured YAML**, not prose — they declare the same fields the
analyzer produces (genre, mood, instruments, production…) and are rendered
through the *same* composer. One grammar, one set of tests, and presets and
analysed tracks come out speaking the same language.

It also makes blending well defined: merging two structured presets is a list
operation, where blending two paragraphs of prose is not. `blend_with` plus a
`blend` weight interleaves each field proportionally; scalars like BPM cross
over at the halfway point rather than averaging, since the mean of 78 and 132
BPM is a tempo neither preset asked for.

Three modifier axes (`era`, `texture`, `mood_shift`) layer on top. Modifiers
always *add* — they never replace what the preset declared.

Ships with: lo-fi hip-hop, neo-soul, indie folk, synthwave, dark techno, pop
anthem, cinematic epic, ambient drift. Drop your own `.yaml` into
`songscribe/presets/` and it appears in the dropdown on the next restart.

## Lyrics Structure

MiniMax treats bracketed section tags as the **only** executable structural
instruction — the lyric text itself just conveys mood. So a malformed tag
doesn't produce a slightly-off song, it silently drops structure from a render
that may take minutes.

This node normalises `(intro)`, `Verse 1:`, `{HOOK}`, `[middle 8]` and `ending:`
into `[Intro]` `[Verse]` `[Chorus]` `[Bridge]` `[Outro]`, and estimates whether
the lyrics fit your `max_duration` — warning in both directions (words cut short,
or long instrumental stretches). The estimate is reported as a **range**, since
delivery speed differs enormously between a ballad and a rap verse.

A lyric line that merely ends in a colon is not mistaken for a tag.

## Performance

Measured on a portable Windows install, CPU only, warm process:

| Track length | Time |
|---|---|
| 40 s | ~5 s |
| 5 min | ~14 s |
| any, cached | ~0.15 s |

Add roughly 10 s once per ComfyUI session for numba's JIT warm-up on the first
analysis.

## Tests

No ComfyUI required — they stub out `folder_paths`:

```bash
python_embeded/python.exe custom_nodes/ComfyUI-SongScribe/tests/smoke_test.py
```

- `smoke_test.py` — synthesises a track at a known 78 BPM in D♭ major and checks the measured values land on it.
- `format_test.py` — transcodes to every supported container and verifies each loads back.
- `node_test.py` — loads the pack through ComfyUI's importlib path and executes the node end to end.

## Measured accuracy

Scored against five labelled tracks with `tests/evaluate.py`. Read the caveat
below before trusting the numbers.

| Axis | Result |
|---|---|
| Vocal presence (voice / no voice) | 5/5 |
| Sung vs rapped | 4/5 |
| Key — mode only | 2/2 |
| Key — exact tonic | 0/3 |
| BPM vs label | 1/5 |
| Vocal **gender** | **2/5 — removed from the vocabulary** |

**The caveat:** those labels are *generation prompts*, not measurements of the
finished audio. Where the label and the analyzer disagreed on tempo, the audio's
own onset autocorrelation backed the analyzer in 4 of 5 cases — on one track the
correlation at the labelled 96 BPM was *negative* (−0.031) versus 0.421 at the
detected tempo. So "BPM 1/5" is not 1/5 accuracy against real ground truth; it
substantially measures how closely a music generator honoured its own prompt.
Proper calibration needs tracks with tempo and key measured from the audio.

Two changes came directly out of this run:

- **Gender claims were removed from the vocal vocabulary.** CLAP scored 2/5 on a
  *binary* male/female question — worse than chance — and answered "female" with
  0.80–0.90 confidence on three tracks that were male. A caption is an
  instruction, so a wrong gender claim doesn't just misdescribe the source, it
  generates the wrong voice.
- **The key phrase now names the relative when the margin is tight.** One track
  led its relative by 0.023 while beating every other candidate decisively —
  reported, correctly but uselessly, as confidence 1.00. It now reads
  "B flat major (or its relative G minor)" rather than picking a side.

## Accuracy notes

- **Key** is Krumhansl-Schmuckler profile correlation. Confidence is scored against the best *non-relative* alternative, since a key and its relative minor share all seven pitch classes and would otherwise always look ambiguous. Relative-key confusion is reported in `analysis.key.relative_margin` rather than hidden.
- **BPM** can land on half or double time. That's inherent to beat tracking, not a bug.
- **Percussive ratio** is estimated from ~24 s of evenly spaced excerpts rather than the whole track; measured error against a full-resolution HPSS is ~0.002, for roughly a sixth of the cost.
- **Section boundaries** come from timbral/harmonic self-similarity clustering. They mark where the music changes — they do not identify *what* a section is. Verse/chorus labelling is not something this can honestly claim.
