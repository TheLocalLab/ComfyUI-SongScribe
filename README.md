# SongScribe — AI Music Prompt Nodes for ComfyUI

<img src="icon.png" width="96" align="right" alt="SongScribe logo: an eighth note over an audio waveform">

**Style prompts and song captions for AI music generation in ComfyUI.** Pick a
style from a 73-preset library, or analyse any song to get a prompt, its lyrics
and its duration.

Works with both major open music models:

- **[MiniMax Music 3](https://github.com/MiniMax-AI/MiniMax-Music3)** — three-section caption format
- **[YuE2](https://map-yue2.github.io/)** — flat comma-separated style prompt

Outputs plain `STRING` / `FLOAT`, so they fit any audio workflow. **Runs on CPU** —
no extra VRAM, nothing competing with your music model.

## Install

**One-click (Windows):** [The Local Lab installer](https://www.patreon.com/TheLocalLab/posts/yue2-song-one-ai-169585892)
sets up the latest ComfyUI portable, ComfyUI Manager, this node and its
dependencies in a single run — no manual Python or git steps. Handy for a clean
machine or a second install.

**ComfyUI Manager:** search for *SongScribe* and install. Dependencies are
handled for you.

**Manual:** clone into `ComfyUI/custom_nodes/`, then install into the same
Python that runs ComfyUI:

```bash
python_embeded\python.exe -m pip install librosa mutagen pyyaml
```

Optional, only if you want lyric transcription:

```bash
python_embeded\python.exe -m pip install faster-whisper
```

Restart ComfyUI. The nodes appear under the **SongScribe** category.

## Nodes

| Node | Does |
|---|---|
| **Style Preset** | Pick a style → prompt. No audio needed. |
| **Song Analyzer** | Audio → caption, lyrics, duration |
| **Caption Splitter** | Caption → three editable sections |
| **Caption Composer** | Three sections → caption |
| **Lyrics Structure** | Fix section tags, check they fit the duration |

---

## Style Preset

73 presets across 14 categories. Set **`format`** to match your model, then wire
the **`prompt`** output to it.

| Setting | Use |
|---|---|
| `format` | `minimax` or `yue2` — **set this first** |
| `detail` | *YuE2.* `tags` / `full` / `rich`. Start with `full`. |
| `vocal` | *YuE2.* Voice type. `auto` uses the preset's own. |
| `language` | *YuE2.* Set this if your lyrics aren't English. |
| `style` | *MiniMax.* `verbatim` / `balanced` / `loose` |
| `era`, `texture`, `mood_shift` | Optional flavour; they add, never replace |
| `blend_with` + `blend` | Mix in a second preset |
| `extra` | Free text, appended as-is |

Settings for the format you're *not* using are ignored — harmless, just inert.

**Wiring to YuE2:** `style` on `YuE2GenerateMusic` is a widget, so right-click
the node → **convert `style` to input** before you can connect to it.

Example output (`World / Reggae`, `yue2`):

```
reggae, roots reggae, dub, rocksteady, jamaican, 76 BPM, laid-back and dreamy,
offbeat guitar skank on the upstroke, one-drop drums with the kick on the three,
deep melodic round bassline, Hammond organ bubble, warm mid-range voice, sung
lazily behind the beat, spring reverb and dub delay throws, English
```

**Add your own:** drop a `.yaml` into `songscribe/presets/` — it appears in the
dropdown after a restart. Copy an existing one as a template.

---

## Song Analyzer

Feed it a song, get a prompt back. Nothing is guessed: BPM, key, dynamics and
song structure are measured, and anything that can't be measured confidently is
left out rather than invented.

| Setting | Use |
|---|---|
| `audio_file` | Upload, or set `(use AUDIO input)` to drive it from a socket |
| `describe` | `clap` adds instruments/production/vocals; `off` for measured facts only |
| `genre_source` | **`maest` is much more accurate** than `clap` — see note below |
| `style` | How closely the caption copies the source: `verbatim` / `balanced` / `loose` |
| `transcribe_lyrics` | `off` by default. Only for songs with no lyrics embedded. |
| `use_cache` | Leave on. Re-runs are ~100× faster. |

Outputs `caption`, `lyrics`, `duration` — wire straight into MiniMax's
`caption` / `lyrics` / `max_duration`.

Reads wav, flac, mp3, m4a, ogg, opus, aiff, wma and more. Lyrics come from
embedded tags or a sibling `.lrc`/`.txt` if present; Whisper transcription is a
last resort and off by default.

Speed (CPU): ~5s for a 40s track, ~14s for 5 minutes, instant when cached.

### `genre_source: maest`

CLAP guesses genre by text similarity and gets it wrong often. **MAEST** is
trained on 400 real genre labels and is far better — it correctly identified
trap, reggae, heavy metal and R&B where CLAP said "bossa nova" and "dream pop".

It's **not** the default because it runs custom code from its model repo
(`trust_remote_code`). That's pinned to one audited commit and only loads if you
select it, but it's your call to opt in.

---

## Caption Splitter / Composer

Split a caption into its three sections, edit one, rebuild it. Useful for
keeping an analysed arrangement while replacing the vocal description entirely.

## Lyrics Structure

Rewrites `(intro)`, `Verse 1:`, `{HOOK}` into the `[Intro]` `[Verse]` `[Chorus]`
tags MiniMax expects, and warns if your lyrics are too long or too short for
`max_duration`. Worth putting in front of the lyrics input — MiniMax treats
those tags as the only structural instruction it gets.

---

## Known limits

Honesty about what to trust, measured against labelled tracks:

- **Reliable:** BPM, key, duration, song structure, vocal-presence detection,
  and genre via `maest`.
- **Genre via `clap` is roughly a coin flip.** Use `maest`.
- **Mood and vocal-timbre detection are disabled** — they returned nearly the
  same answer for every song. Presets supply both far more reliably.
- Thresholds are tuned on a small sample. If a preset or reading looks wrong,
  it probably is — please open an issue.

## Contributing

Issues and pull requests welcome, especially corrections to presets. If a genre
doesn't sound right to you, say so — that feedback is what fixed several of
them.

Tests need no ComfyUI:

```bash
python_embeded\python.exe custom_nodes\ComfyUI-SongScribe	ests\smoke_test.py
```

---

## About — The Local Lab

Maintained by **The Local Lab**. Tune in for the best available AI tools, made easy.

- 🎥 **YouTube** — [youtube.com/@TheLocalLab](https://www.youtube.com/@TheLocalLab)
- 💬 **Discord** — [discord.gg/5hmB4N4JFc](https://discord.gg/5hmB4N4JFc) — community chat, support, and feedback
- 💖 **Patreon** — [patreon.com/cw/TheLocalLab](https://www.patreon.com/cw/TheLocalLab) — exclusive one-click Windows installers, ComfyUI workflows, and AI resources. Skip the manual node + model setup and get straight to creating.
- 🐦 **X** — [@TheLocalLab_](https://x.com/TheLocalLab_)

## License

MIT — see [LICENSE](LICENSE).
