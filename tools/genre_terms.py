"""Related-genre expansion.

YuE2's own prompts stack several genre words rather than naming one:
"reggaeton, salsa, bachata, latin pop, cumbia". There is a reason beyond
style - a broad term like "hip hop" is far more strongly represented in
training data than a narrow one like "drill", so leading with the specific
term and backing it with its family gives the model both precision and a
reliable anchor.

The preset's own category word belongs here too. A preset named
"Hip-Hop / Drill" that emits only "drill" throws away the most recognisable
word it has.

Ordered specific -> general, because the first term carries the most weight.
"""

RELATED = {
    "boom bap hip-hop": ["hip hop", "rap", "golden age hip hop"],
    "lo-fi hip-hop": ["chillhop", "hip hop", "downtempo"],
    "trap": ["hip hop", "rap", "southern rap"],
    "drill": ["hip hop", "rap", "trap"],
    "grime": ["hip hop", "rap", "uk rap", "garage"],
    "industrial hip hop": ["hip hop", "rap", "industrial", "experimental"],
    "neo-soul": ["r&b", "soul", "contemporary r&b"],
    "classic soul": ["soul", "r&b", "motown", "1960s soul"],
    "funk": ["soul", "r&b", "funk rock"],
    "southern gospel": ["gospel", "country gospel", "christian"],
    "mainstream pop": ["pop", "dance-pop", "radio pop"],
    "synth-pop": ["pop", "new wave", "electronic"],
    "dream pop": ["shoegaze", "indie pop", "alternative"],
    "k-pop": ["pop", "dance-pop", "korean pop"],
    "city pop": ["japanese city pop", "pop", "funk", "disco"],
    "j-pop": ["japanese pop", "pop", "anime"],
    "boy band pop": ["pop", "vocal pop", "teen pop"],
    "easy listening": ["lounge", "adult contemporary", "soft pop"],
    "indie rock": ["alternative rock", "rock", "indie"],
    "classic rock": ["rock", "hard rock", "1970s rock"],
    "grunge": ["alternative rock", "rock", "1990s"],
    "punk rock": ["punk", "rock", "hardcore"],
    "post-rock": ["instrumental rock", "ambient rock", "experimental"],
    "heavy metal": ["metal", "rock", "hard rock"],
    "emo": ["emo rock", "pop punk", "alternative rock"],
    "heartland rock": ["rock", "americana", "classic rock"],
    "soft rock": ["rock", "adult contemporary", "pop rock"],
    "yacht rock": ["soft rock", "smooth jazz", "aor"],
    "space rock": ["psychedelic rock", "krautrock", "rock"],
    "industrial metal": ["metal", "industrial", "hard rock"],
    "thrash metal": ["metal", "speed metal", "heavy metal"],
    "cyber metal": ["metal", "synthwave", "electronic rock"],
    "house": ["electronic", "dance", "edm", "club"],
    "techno": ["electronic", "dance", "edm"],
    "drum and bass": ["electronic", "jungle", "breakbeat", "edm"],
    "synthwave": ["retrowave", "electronic", "1980s synth"],
    "ambient electronic": ["ambient", "electronic", "new age"],
    "trip-hop": ["downtempo", "electronic", "hip hop"],
    "italo-disco": ["disco", "electronic", "1980s"],
    "nu-disco": ["disco", "house", "funk", "electronic"],
    "eurobeat": ["dance", "electronic", "hi-nrg", "eurodance"],
    "electropop": ["synth-pop", "pop", "electronic"],
    "folktronica": ["electronic", "folk", "downtempo"],
    "dark ambient": ["ambient", "drone", "cinematic"],
    "alternative dance": ["indie dance", "new wave", "electronic"],
    "dance-punk": ["post-punk", "indie dance", "punk"],
    "indie folk": ["folk", "acoustic", "singer-songwriter"],
    "country": ["country pop", "americana", "nashville"],
    "bluegrass": ["country", "folk", "americana", "appalachian"],
    "solo piano": ["classical", "neoclassical", "instrumental"],
    "jazz": ["straight-ahead jazz", "bebop", "swing"],
    "cool jazz": ["jazz", "west coast jazz", "modal jazz"],
    "bossa nova": ["jazz", "latin jazz", "brazilian"],
    "blues": ["slow blues", "electric blues", "soul blues"],
    "jazz-funk": ["funk", "jazz fusion", "soul jazz"],
    "afro-jazz": ["jazz", "afrobeat", "world"],
    "big band swing": ["swing", "jazz", "big band"],
    "dixieland": ["traditional jazz", "new orleans jazz", "ragtime"],
    "boogie woogie": ["blues", "piano blues", "jump blues"],
    "jump blues": ["blues", "swing", "rhythm and blues"],
    # ska sits near 140 BPM; pairing it with a 76 BPM reggae preset
    # gives the model two contradictory tempo anchors.
    "reggae": ["roots reggae", "dub", "rocksteady", "jamaican"],
    "afrobeats": ["afropop", "world", "dancehall"],
    "latin pop": ["salsa", "latin", "reggaeton"],
    "bachata": ["latin", "dominican", "latin pop"],
    "bhangra": ["punjabi", "world", "indian"],
    "flamenco": ["spanish guitar", "world", "latin"],
    "arabic pop": ["middle eastern", "world", "arabic"],
    "barbershop": ["a cappella", "vocal harmony", "close harmony"],
    "honky tonk": ["country", "classic country", "western swing"],
    "epic trailer": ["cinematic", "orchestral", "trailer music"],
    "orchestral score": ["classical", "film score", "cinematic"],
    "horror score": ["cinematic", "dark ambient", "film score"],
    "chamber music": ["classical", "string quartet", "chamber"],
}


def expand(genres):
    """Primary genres -> primary plus family, deduped, order preserved."""
    out = []
    for genre in genres:
        for term in [genre] + RELATED.get(genre.lower(), []):
            if term.lower() not in {o.lower() for o in out}:
                out.append(term)
    return out
