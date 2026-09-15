"""Draw the registry icon: an eighth note over a waveform.

The Comfy Registry caps icons at 800x400, so this renders 400x400 (square reads
better in a grid of listing cards) at 4x and downsamples, which is the cheapest
way to get clean antialiased edges out of PIL without a vector rasteriser.

    python tools/make_icon.py
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw

SIZE = 400
SS = 4  # supersample factor
W = SIZE * SS

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "icon.png"
)

BG_TOP = (24, 26, 38)
BG_BOTTOM = (13, 14, 22)
ACCENT = (94, 234, 212)      # teal
ACCENT_DIM = (45, 122, 115)
NOTE = (240, 246, 252)


def rounded_background(draw: ImageDraw.ImageDraw) -> None:
    # Vertical gradient, drawn as rows since PIL has no gradient primitive.
    for y in range(W):
        t = y / W
        colour = tuple(
            int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3)
        )
        draw.line([(0, y), (W, y)], fill=colour)


def waveform(draw: ImageDraw.ImageDraw) -> None:
    """Symmetric bars along the lower third - the 'analysis' half of the idea."""
    centre = int(W * 0.70)
    count = 19
    span = int(W * 0.78)
    left = (W - span) // 2
    gap = span / count
    bar = gap * 0.42

    for i in range(count):
        # Deterministic pseudo-waveform: two sines beating against each other,
        # so it reads as audio rather than as a bar chart.
        phase = i / (count - 1)
        amplitude = (
            0.34 * math.sin(phase * math.pi * 3.1)
            + 0.22 * math.sin(phase * math.pi * 7.3 + 1.1)
        )
        height = abs(amplitude) * W * 0.16 + W * 0.012
        x = left + gap * i + gap / 2
        colour = ACCENT if i % 3 else ACCENT_DIM
        draw.rounded_rectangle(
            [x - bar / 2, centre - height, x + bar / 2, centre + height],
            radius=bar / 2,
            fill=colour,
        )


def eighth_note(image: Image.Image) -> None:
    """Head, stem and flag, drawn on a layer so the head can be rotated."""
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    stem_x = int(W * 0.60)
    stem_top = int(W * 0.17)
    stem_bottom = int(W * 0.50)
    stem_w = int(W * 0.035)

    draw.rounded_rectangle(
        [stem_x - stem_w / 2, stem_top, stem_x + stem_w / 2, stem_bottom],
        radius=stem_w / 2,
        fill=NOTE,
    )

    # Flag: a tapering curve off the top of the stem, approximated by a
    # sequence of shortening horizontal strokes.
    steps = 46
    for i in range(steps):
        t = i / (steps - 1)
        y = stem_top + t * (W * 0.16)
        reach = (W * 0.135) * (1 - t) ** 0.65
        thickness = max(2, int(W * 0.030 * (1 - t * 0.45)))
        curve = math.sin(t * math.pi * 0.5) * W * 0.035
        draw.line(
            [(stem_x, y), (stem_x + reach + curve, y + reach * 0.55)],
            fill=NOTE,
            width=thickness,
        )

    layer_head = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    head_draw = ImageDraw.Draw(layer_head)
    head_rx, head_ry = W * 0.105, W * 0.078
    cx, cy = stem_x - head_rx * 0.86, stem_bottom
    head_draw.ellipse(
        [cx - head_rx, cy - head_ry, cx + head_rx, cy + head_ry], fill=NOTE
    )
    # Real note heads sit at an angle; upright ones look like a cartoon.
    layer_head = layer_head.rotate(20, center=(cx, cy), resample=Image.BICUBIC)

    layer = Image.alpha_composite(layer, layer_head)
    image.alpha_composite(layer)


def main() -> int:
    image = Image.new("RGBA", (W, W), BG_TOP + (255,))
    draw = ImageDraw.Draw(image)

    rounded_background(draw)
    waveform(draw)
    eighth_note(image)

    # Rounded corners, so the card does not look like a pasted screenshot.
    mask = Image.new("L", (W, W), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W, W], radius=int(W * 0.18), fill=255)
    image.putalpha(mask)

    image = image.resize((SIZE, SIZE), Image.LANCZOS)
    image.save(OUT, "PNG", optimize=True)

    print(f"wrote {OUT} ({SIZE}x{SIZE}, {os.path.getsize(OUT) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
