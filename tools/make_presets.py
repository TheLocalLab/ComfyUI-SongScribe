"""Generate the shipped preset library.

Presets are plain YAML and meant to be hand-edited; this script exists so the
*shipped* set stays internally consistent - same field names, same vocabulary
phrasing, same level of detail - rather than drifting as entries are added by
hand over time. Editing a preset afterwards is entirely expected.

Vocabulary note: production/instrument/mood/scene values are drawn from
songscribe/vocab/*.yaml wherever possible, so a preset and an analysed track
describe the same thing with the same words.

    python tools/make_presets.py
"""

from __future__ import annotations

import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "songscribe", "presets")

# category, slug, display name, bpm, key, harmony, feel, drums, low_end,
# genre, mood, scene, production, instruments, vocals
P = [
    # ---------------------------------------------------------------- hip-hop
    ("hiphop", "boom_bap", "Hip-Hop / Boom Bap", 92, "E minor", "simple diatonic", "loose and human", "drum-forward", "solid bass weight",
     ["boom bap hip-hop"], ["confident and swaggering", "gritty and raw"], ["a long night drive"],
     ["a dusty low-passed mix", "warm analog tape saturation"],
     ["dusty boom-bap drums", "deep sub bass", "grand piano", "muted trumpet", "vinyl crackle texture"],
     ("spoken", ["a warm mid-range voice"], ["rapped in tight rhythmic flow", "ad-libbed loosely over the beat"])),
    ("hiphop", "lofi", "Hip-Hop / Lo-Fi Chillhop", 78, "Db major", "moderately extended", "loose and human", "drum-forward", "solid bass weight",
     ["lo-fi hip-hop", "chillhop"], ["laid-back and dreamy", "warm and nostalgic"], ["studying and late-night focus", "rainy-day headphones listening"],
     ["a muddy lo-fi bedroom mix", "heavy vinyl crackle and surface noise", "tape hiss and wow-flutter pitch wobble"],
     ["dusty boom-bap drums", "deep sub bass", "warm Rhodes electric piano", "jazzy electric guitar", "vinyl crackle texture"],
     ("wordless", ["a soft androgynous voice"], ["hushed and half-spoken", "sitting low in the mix like another instrument"])),
    ("hiphop", "trap", "Hip-Hop / Trap", 140, "F minor", "simple diatonic", "machine-tight", "drum-forward", "deep sub-heavy low end",
     ["trap"], ["menacing and aggressive", "dark and brooding"], ["an after-hours club"],
     ["a deep sub-heavy club mix", "a loud heavily compressed master"],
     ["808 drum machine", "punchy electronic drum machine", "bright synth lead", "arpeggiated synthesizer"],
     ("spoken", ["a heavily autotuned voice"], ["rapped in a slow drawling flow", "layered in thick stacked harmonies"])),
    ("hiphop", "drill", "Hip-Hop / Drill", 144, "G minor", "simple diatonic", "machine-tight", "drum-forward", "deep sub-heavy low end",
     ["drill"], ["menacing and aggressive", "tense and anxious"], ["a long night drive"],
     ["a deep sub-heavy club mix", "gritty distortion and fuzz"],
     ["808 drum machine", "punchy electronic drum machine", "solo violin", "analog synth pads"],
     ("spoken", ["a deep resonant low-register voice"], ["rapped in tight rhythmic flow"])),
    # ------------------------------------------------------------- soul / r&b
    ("soul", "neo_soul", "Soul / Neo-Soul", 86, "Eb major", "chromatic / extended harmony", "loose and human", "balanced", "solid bass weight",
     ["neo-soul", "contemporary R&B"], ["sensual and smouldering", "romantic and longing"], ["a romantic dinner", "a cocktail lounge evening"],
     ["a crisp studio multitrack production", "warm analog tape saturation"],
     ["warm Rhodes electric piano", "electric bass guitar", "acoustic drum kit", "clean electric guitar", "saxophone"],
     ("sung", ["a soulful gospel-trained voice", "a warm husky voice"], ["crooned smoothly", "layered in thick stacked harmonies"])),
    ("soul", "classic_soul", "Soul / Classic Motown", 112, "C major", "simple diatonic", "loose and human", "balanced", "solid bass weight",
     ["classic soul", "funk"], ["bright and joyful", "hopeful and uplifting"], ["a wedding first dance"],
     ["a raw live-room recording", "warm analog tape saturation"],
     ["acoustic drum kit", "electric bass guitar", "Hammond organ", "brass section", "tambourine and shaker"],
     ("sung", ["a powerful belting voice", "a soulful gospel-trained voice"], ["belted out powerfully", "answered by a call-and-response choir"])),
    ("soul", "funk", "Soul / Funk", 108, "D minor", "moderately extended", "loose and human", "drum-forward", "solid bass weight",
     ["funk", "disco"], ["playful and carefree", "confident and swaggering"], ["a crowded dancefloor at peak hour"],
     ["a crisp studio multitrack production", "a wide spacious stereo image"],
     ["acoustic drum kit", "slap bass", "clean electric guitar", "Hammond organ", "brass section"],
     ("sung", ["a powerful belting voice"], ["belted out powerfully", "answered by a call-and-response choir"])),
    # -------------------------------------------------------------------- pop
    ("pop", "anthem", "Pop / Modern Anthem", 124, "C major", "simple diatonic", "machine-tight", "drum-forward", "solid bass weight",
     ["mainstream pop"], ["bright and joyful", "euphoric and rushing"], ["a festival main stage", "a gym workout"],
     ["a clean modern polished mix", "a loud heavily compressed master", "sidechain pumping compression"],
     ["punchy electronic drum machine", "warm analog synth bass", "bright synth lead", "clean electric guitar", "handclaps"],
     ("sung", ["a clear bright voice", "a smooth polished pop voice"], ["belted out powerfully", "layered in thick stacked harmonies"])),
    ("pop", "synth_pop", "Pop / Synth-Pop", 118, "A minor", "simple diatonic", "machine-tight", "balanced", "solid bass weight",
     ["synth-pop", "indie pop"], ["bittersweet", "hopeful and uplifting"], ["a long night drive"],
     ["a wide spacious stereo image", "long cavernous hall reverb"],
     ["punchy electronic drum machine", "warm analog synth bass", "arpeggiated synthesizer", "analog synth pads"],
     ("sung", ["a clear bright voice"], ["sung softly and gently", "doubled with a close harmony"])),
    ("pop", "dream_pop", "Pop / Dream Pop", 96, "E major", "moderately extended", "loose and human", "harmonically led", "moderate low end",
     ["dream pop", "shoegaze"], ["hazy and drifting", "spacious and weightless"], ["falling asleep and winding down"],
     ["heavy reverb drenching everything", "gentle detuning and chorus wobble", "a shimmering high-end sheen"],
     ["clean electric guitar", "analog synth pads", "electric bass guitar", "acoustic drum kit", "reversed ambient swells"],
     ("sung", ["a light breathy voice", "a distant heavily reverbed voice"], ["sung softly and gently", "drenched in delay and reverb"])),
    ("pop", "kpop", "Pop / K-Pop", 110, "B minor", "moderately extended", "machine-tight", "drum-forward", "deep sub-heavy low end",
     ["k-pop", "mainstream pop"], ["confident and swaggering", "euphoric and rushing"], ["a festival main stage"],
     ["a clean modern polished mix", "a loud heavily compressed master", "sidechain pumping compression"],
     ["punchy electronic drum machine", "808 drum machine", "bright synth lead", "arpeggiated synthesizer", "handclaps"],
     ("sung", ["a smooth polished pop voice", "a clear bright voice"], ["layered in thick stacked harmonies", "rapped in tight rhythmic flow"])),
    # ------------------------------------------------------------------- rock
    ("rock", "indie", "Rock / Indie", 132, "G major", "simple diatonic", "loose and human", "drum-forward", "moderate low end",
     ["indie rock", "alternative rock"], ["bittersweet", "energetic and driving"], ["a summer road trip"],
     ["a raw live-room recording", "a crisp studio multitrack production"],
     ["acoustic drum kit", "electric bass guitar", "clean electric guitar", "crunchy distorted electric guitar"],
     ("sung", ["a nasal indie voice"], ["sung with heavy vibrato", "doubled with a close harmony"])),
    ("rock", "classic", "Rock / Classic 70s", 124, "A major", "simple diatonic", "loose and human", "drum-forward", "solid bass weight",
     ["classic rock", "hard rock"], ["confident and swaggering", "triumphant and soaring"], ["a summer road trip"],
     ["a raw live-room recording", "warm analog tape saturation"],
     ["acoustic drum kit", "electric bass guitar", "crunchy distorted electric guitar", "Hammond organ"],
     ("sung", ["a gritty rasping voice"], ["belted out powerfully"])),
    ("rock", "grunge", "Rock / Grunge", 118, "E minor", "simple diatonic", "loose and human", "drum-forward", "solid bass weight",
     ["grunge", "alternative rock"], ["angry and defiant", "gritty and raw"], ["a film's emotional climax"],
     ["a raw live-room recording", "gritty distortion and fuzz"],
     ["acoustic drum kit", "electric bass guitar", "heavily distorted guitar", "crunchy distorted electric guitar"],
     ("sung", ["a rough untrained raw voice", "a strained emotional voice"], ["shouted and screamed", "sung in a flat deadpan monotone"])),
    ("rock", "punk", "Rock / Punk", 168, "D major", "simple diatonic", "loose and human", "drum-forward", "moderate low end",
     ["punk rock"], ["angry and defiant", "energetic and driving"], ["a festival main stage"],
     ["a raw live-room recording", "a rough demo quality recording"],
     ["acoustic drum kit", "electric bass guitar", "crunchy distorted electric guitar"],
     ("sung", ["a rough untrained raw voice"], ["shouted and screamed"])),
    ("rock", "post_rock", "Rock / Post-Rock", 84, "D major", "simple diatonic", "steady", "balanced", "moderate low end",
     ["post-rock"], ["epic and cinematic", "spacious and weightless", "hopeful and uplifting"], ["a film's emotional climax"],
     ["heavy reverb drenching everything", "a wide spacious stereo image"],
     ["clean electric guitar", "crunchy distorted electric guitar", "acoustic drum kit", "string section", "reversed ambient swells"],
     ("instrumental", [], [])),
    ("rock", "metal", "Rock / Heavy Metal", 150, "E minor", "simple diatonic", "machine-tight", "drum-forward", "deep sub-heavy low end",
     ["heavy metal", "hard rock"], ["menacing and aggressive", "epic and cinematic"], ["a video game boss fight"],
     ["a loud heavily compressed master", "gritty distortion and fuzz"],
     ["acoustic drum kit", "heavily distorted guitar", "electric bass guitar", "orchestral timpani"],
     ("sung", ["a gritty rasping voice", "a strained emotional voice"], ["shouted and screamed", "belted out powerfully"])),
    # ------------------------------------------------------------- electronic
    ("electronic", "house", "Electronic / House", 124, "A minor", "simple diatonic", "machine-tight", "drum-forward", "deep sub-heavy low end",
     ["house", "deep house"], ["hypnotic and trance-like", "euphoric and rushing"], ["a crowded dancefloor at peak hour"],
     ["a deep sub-heavy club mix", "sidechain pumping compression"],
     ["punchy electronic drum machine", "warm analog synth bass", "warm Rhodes electric piano", "analog synth pads"],
     ("wordless", ["a soulful gospel-trained voice"], ["wordless humming and ooh-ahh textures", "chopped and stuttered as a sample"])),
    ("electronic", "techno", "Electronic / Dark Techno", 132, "A minor", "simple diatonic", "machine-tight", "drum-forward", "deep sub-heavy low end",
     ["techno"], ["dark and brooding", "hypnotic and trance-like"], ["an after-hours club"],
     ["a deep sub-heavy club mix", "sidechain pumping compression", "gritty distortion and fuzz"],
     ["punchy electronic drum machine", "808 drum machine", "warm analog synth bass", "reversed ambient swells"],
     ("instrumental", [], [])),
    ("electronic", "dnb", "Electronic / Drum & Bass", 174, "F minor", "simple diatonic", "machine-tight", "drum-forward", "deep sub-heavy low end",
     ["drum and bass"], ["urgent and restless", "energetic and driving"], ["a crowded dancefloor at peak hour", "running and cardio"],
     ["a deep sub-heavy club mix", "a loud heavily compressed master"],
     ["punchy electronic drum machine", "deep sub bass", "analog synth pads", "bright synth lead"],
     ("instrumental", [], [])),
    ("electronic", "synthwave", "Electronic / Synthwave", 112, "F# minor", "simple diatonic", "machine-tight", "drum-forward", "solid bass weight",
     ["synthwave", "synth-pop"], ["warm and nostalgic", "mysterious and shadowy"], ["a long night drive"],
     ["a wide spacious stereo image", "long cavernous hall reverb", "warm analog tape saturation"],
     ["punchy electronic drum machine", "warm analog synth bass", "bright synth lead", "arpeggiated synthesizer", "analog synth pads"],
     ("instrumental", [], [])),
    ("electronic", "ambient", "Electronic / Ambient Drift", 62, "D major", "simple diatonic", "loose and human", "harmonically led", "light low end",
     ["ambient electronic", "new age"], ["calm and meditative", "spacious and weightless"], ["falling asleep and winding down", "meditation and yoga"],
     ["heavy reverb drenching everything", "a wide spacious stereo image", "a shimmering high-end sheen"],
     ["analog synth pads", "reversed ambient swells", "field recording ambience", "grand piano", "solo cello"],
     ("instrumental", [], [])),
    ("electronic", "trip_hop", "Electronic / Trip-Hop", 88, "C minor", "moderately extended", "loose and human", "balanced", "deep sub-heavy low end",
     ["trip-hop", "downtempo"], ["dark and brooding", "sensual and smouldering"], ["an after-hours club", "rainy-day headphones listening"],
     ["a dusty low-passed mix", "heavy vinyl crackle and surface noise", "long cavernous hall reverb"],
     ["dusty boom-bap drums", "deep sub bass", "warm Rhodes electric piano", "string section", "vinyl crackle texture"],
     ("sung", ["a light breathy voice", "a distant heavily reverbed voice"], ["sung softly and gently", "drenched in delay and reverb"])),
    # -------------------------------------------------------------- acoustic
    ("acoustic", "indie_folk", "Acoustic / Indie Folk", 96, "G major", "simple diatonic", "loose and human", "harmonically led", "moderate low end",
     ["indie folk", "singer-songwriter acoustic"], ["tender and intimate", "bittersweet"], ["a campfire singalong", "a slow lazy Sunday morning"],
     ["a dry close-mic'd intimate sound", "a raw live-room recording"],
     ["fingerpicked acoustic guitar", "upright piano", "upright double bass", "mandolin", "brushed jazz drums"],
     ("sung", ["a warm husky voice", "a rough untrained raw voice"], ["sung softly and gently", "doubled with a close harmony"])),
    ("acoustic", "country", "Acoustic / Country", 104, "D major", "simple diatonic", "loose and human", "balanced", "solid bass weight",
     ["country", "americana"], ["warm and nostalgic", "bittersweet"], ["a summer road trip", "a campfire singalong"],
     ["a crisp studio multitrack production", "a raw live-room recording"],
     ["acoustic guitar", "slide guitar", "electric bass guitar", "acoustic drum kit", "upright piano"],
     ("sung", ["a warm husky voice"], ["crooned smoothly", "doubled with a close harmony"])),
    ("acoustic", "bluegrass", "Acoustic / Bluegrass", 140, "A major", "simple diatonic", "loose and human", "harmonically led", "moderate low end",
     ["bluegrass", "folk"], ["playful and carefree", "bright and joyful"], ["a campfire singalong"],
     ["a raw live-room recording", "a dry close-mic'd intimate sound"],
     ["banjo", "mandolin", "fingerpicked acoustic guitar", "upright double bass", "solo violin"],
     ("sung", ["a clear bright voice"], ["layered in thick stacked harmonies", "answered by a call-and-response choir"])),
    ("acoustic", "solo_piano", "Acoustic / Solo Piano", 68, "F major", "moderately extended", "loose and human", "harmonically led", "light low end",
     ["solo piano", "classical orchestral"], ["reflective and quiet", "melancholy and wistful"], ["falling asleep and winding down", "a funeral or memorial"],
     ["a dry close-mic'd intimate sound", "long cavernous hall reverb"],
     ["grand piano"],
     ("instrumental", [], [])),
    # ---------------------------------------------------------- jazz / blues
    ("jazz", "jazz_trio", "Jazz / Late-Night Trio", 120, "Bb major", "chromatic / extended harmony", "loose and human", "balanced", "solid bass weight",
     ["jazz", "bebop jazz"], ["reflective and quiet", "sensual and smouldering"], ["a cocktail lounge evening"],
     ["a raw live-room recording", "warm analog tape saturation"],
     ["grand piano", "upright double bass", "brushed jazz drums", "solo trumpet", "saxophone"],
     ("instrumental", [], [])),
    ("jazz", "bossa_nova", "Jazz / Bossa Nova", 130, "A minor", "chromatic / extended harmony", "loose and human", "harmonically led", "moderate low end",
     ["bossa nova", "latin pop"], ["calm and meditative", "romantic and longing"], ["a beach party", "a romantic dinner"],
     ["a dry close-mic'd intimate sound", "warm analog tape saturation"],
     ["fingerpicked acoustic guitar", "upright double bass", "brushed jazz drums", "flute", "vibraphone"],
     ("sung", ["a light breathy voice"], ["sung softly and gently", "phrased lazily behind the beat"])),
    ("jazz", "blues", "Jazz / Slow Blues", 72, "E major", "moderately extended", "loose and human", "balanced", "solid bass weight",
     ["blues", "classic soul"], ["melancholy and wistful", "gritty and raw"], ["a cocktail lounge evening"],
     ["a raw live-room recording", "warm analog tape saturation"],
     ["clean electric guitar", "Hammond organ", "electric bass guitar", "acoustic drum kit", "saxophone"],
     ("sung", ["a gritty rasping voice", "a smoky jazz-club voice"], ["belted out powerfully", "phrased lazily behind the beat"])),
    # ------------------------------------------------------------------ world
    ("world", "reggae", "World / Reggae", 76, "G major", "simple diatonic", "loose and human", "balanced", "solid bass weight",
     ["reggae", "dub"], ["laid-back and dreamy", "hopeful and uplifting"], ["a beach party", "a slow lazy Sunday morning"],
     ["a wide spacious stereo image", "heavy tape delay throws", "warm analog tape saturation"],
     ["acoustic drum kit", "electric bass guitar", "clean electric guitar", "Hammond organ", "hand percussion"],
     ("sung", ["a warm mid-range voice"], ["phrased lazily behind the beat", "answered by a call-and-response choir"])),
    ("world", "afrobeats", "World / Afrobeats", 104, "F major", "simple diatonic", "loose and human", "drum-forward", "deep sub-heavy low end",
     ["afrobeats", "reggaeton"], ["playful and carefree", "sensual and smouldering"], ["a beach party", "a crowded dancefloor at peak hour"],
     ["a clean modern polished mix", "a deep sub-heavy club mix"],
     ["congas and bongos", "punchy electronic drum machine", "warm analog synth bass", "clean electric guitar", "tambourine and shaker"],
     ("sung", ["a warm mid-range voice"], ["crooned smoothly", "layered in thick stacked harmonies"])),
    ("world", "latin_pop", "World / Latin Pop", 96, "A minor", "moderately extended", "loose and human", "drum-forward", "solid bass weight",
     ["latin pop", "reggaeton"], ["romantic and longing", "playful and carefree"], ["a beach party", "a crowded dancefloor at peak hour"],
     ["a clean modern polished mix", "a wide spacious stereo image"],
     ["congas and bongos", "acoustic guitar", "brass section", "electric bass guitar", "hand percussion"],
     ("sung", ["a warm husky voice"], ["crooned smoothly", "belted out powerfully"])),
    # -------------------------------------------------------------- cinematic
    ("cinematic", "epic_trailer", "Cinematic / Epic Trailer", 92, "D minor", "simple diatonic", "steady", "drum-forward", "deep sub-heavy low end",
     ["epic trailer", "cinematic score"], ["epic and cinematic", "triumphant and soaring", "tense and anxious"], ["a film's emotional climax", "a video game boss fight"],
     ["a wide spacious stereo image", "long cavernous hall reverb", "a loud heavily compressed master"],
     ["orchestral timpani", "string section", "brass section", "solo cello", "acoustic drum kit"],
     ("wordless", ["a choir of many voices"], ["layered in thick stacked harmonies", "drenched in delay and reverb"])),
    ("cinematic", "orchestral", "Cinematic / Orchestral Score", 76, "C major", "moderately extended", "loose and human", "harmonically led", "moderate low end",
     ["classical orchestral", "cinematic score"], ["reflective and quiet", "hopeful and uplifting"], ["a film's emotional climax"],
     ["long cavernous hall reverb", "a wide spacious stereo image"],
     ["string section", "solo violin", "flute", "clarinet", "harp"],
     ("instrumental", [], [])),
    ("cinematic", "horror", "Cinematic / Horror Tension", 60, "C minor", "chromatic / extended harmony", "loose and human", "harmonically led", "deep sub-heavy low end",
     ["cinematic score", "ambient electronic"], ["haunting and eerie", "tense and anxious", "menacing and aggressive"], ["a tense chase scene"],
     ["heavy reverb drenching everything", "gritty distortion and fuzz", "lo-fi bitcrushed digital artefacts"],
     ["solo violin", "string section", "reversed ambient swells", "orchestral timpani", "music box"],
     ("instrumental", [], [])),
]


def quote(value: str) -> str:
    return f'"{value}"' if any(c in value for c in ":#") else value


def emit(entry) -> str:
    (category, slug, name, bpm, key, harmony, feel, drums, low_end,
     genre, mood, scene, production, instruments, vocals) = entry
    presence, timbre, delivery = vocals

    lines = [
        f"name: {name}",
        f"category: {category}",
        f"bpm: {bpm}",
        f"key: {key}",
        f"harmony: {harmony}",
        f"feel: {feel}",
        f"drums: {drums}",
        f"low_end: {low_end}",
    ]
    for field, values in (
        ("genre", genre), ("mood", mood), ("scene", scene),
        ("production", production), ("instruments", instruments),
    ):
        lines.append(f"{field}:")
        lines.extend(f"  - {quote(v)}" for v in values)

    lines.append(f"vocal_presence: {presence}")
    if presence != "instrumental":
        for field, values in (("vocal_timbre", timbre), ("vocal_delivery", delivery)):
            if values:
                lines.append(f"{field}:")
                lines.extend(f"  - {quote(v)}" for v in values)

    return "\n".join(lines) + "\n"


def main() -> int:
    os.makedirs(OUT, exist_ok=True)

    # Remove the previous generation so renamed entries do not linger.
    for existing in os.listdir(OUT):
        if existing.endswith((".yaml", ".yml")):
            os.remove(os.path.join(OUT, existing))

    for entry in P:
        category, slug = entry[0], entry[1]
        path = os.path.join(OUT, f"{category}-{slug}.yaml")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(emit(entry))

    print(f"wrote {len(P)} presets to {OUT}")
    categories: dict[str, int] = {}
    for entry in P:
        categories[entry[0]] = categories.get(entry[0], 0) + 1
    for category, count in sorted(categories.items()):
        print(f"  {category:<12} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
