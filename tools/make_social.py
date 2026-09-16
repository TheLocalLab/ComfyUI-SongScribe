"""Draw the GitHub social preview card.

GitHub wants 1280x640 (2:1) for the Open Graph image - the square icon.png is
the wrong shape and gets letterboxed or cropped. This is a separate asset, not
a resize: a 2:1 card has room for the name and a one-line pitch, which is the
whole point of the slot.

Rendered at 2x and downsampled for clean edges.

    python tools/make_social.py
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
SS = 2

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "social-preview.png"
)

BG_TOP = (26, 28, 42)
BG_BOTTOM = (12, 13, 20)
ACCENT = (94, 234, 212)
ACCENT_DIM = (48, 128, 120)
WHITE = (244, 248, 252)
MUTED = (150, 162, 178)

FONTS = "C:/Windows/Fonts"


def font(name: str, size: int):
    try:
        return ImageFont.truetype(os.path.join(FONTS, name), size)
    except OSError:
        return ImageFont.load_default()


def gradient(draw, w, h):
    for y in range(h):
        t = y / h
        draw.line(
            [(0, y), (w, y)],
            fill=tuple(int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3)),
        )


def eighth_note(image, cx, cy, scale):
    """Same mark as the icon, positioned and scaled for the card."""
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    stem_x = cx + scale * 0.16
    stem_top = cy - scale * 0.62
    stem_bottom = cy + scale * 0.30
    stem_w = scale * 0.085

    draw.rounded_rectangle(
        [stem_x - stem_w / 2, stem_top, stem_x + stem_w / 2, stem_bottom],
        radius=stem_w / 2,
        fill=WHITE,
    )

    steps = 46
    for i in range(steps):
        t = i / (steps - 1)
        y = stem_top + t * scale * 0.40
        reach = scale * 0.34 * (1 - t) ** 0.65
        thickness = max(2, int(scale * 0.075 * (1 - t * 0.45)))
        curve = math.sin(t * math.pi * 0.5) * scale * 0.09
        draw.line(
            [(stem_x, y), (stem_x + reach + curve, y + reach * 0.55)],
            fill=WHITE,
            width=thickness,
        )

    head = Image.new("RGBA", image.size, (0, 0, 0, 0))
    hd = ImageDraw.Draw(head)
    rx, ry = scale * 0.26, scale * 0.195
    hx, hy = stem_x - rx * 0.86, stem_bottom
    hd.ellipse([hx - rx, hy - ry, hx + rx, hy + ry], fill=WHITE)
    head = head.rotate(20, center=(hx, hy), resample=Image.BICUBIC)

    image.alpha_composite(Image.alpha_composite(layer, head))


def waveform(draw, x0, x1, y, height):
    count = 58
    gap = (x1 - x0) / count
    bar = gap * 0.40
    for i in range(count):
        phase = i / (count - 1)
        amp = (
            0.36 * math.sin(phase * math.pi * 5.3)
            + 0.24 * math.sin(phase * math.pi * 11.1 + 0.8)
            + 0.14 * math.sin(phase * math.pi * 2.1 + 2.2)
        )
        h = abs(amp) * height + height * 0.06
        x = x0 + gap * i + gap / 2
        draw.rounded_rectangle(
            [x - bar / 2, y - h, x + bar / 2, y + h],
            radius=bar / 2,
            fill=ACCENT if i % 3 else ACCENT_DIM,
        )


def main() -> int:
    w, h = W * SS, H * SS
    image = Image.new("RGBA", (w, h), BG_TOP + (255,))
    draw = ImageDraw.Draw(image)
    gradient(draw, w, h)

    # Content sits inside a safe margin: some surfaces crop the edges.
    note_scale = 190 * SS
    eighth_note(image, 210 * SS, 265 * SS, note_scale)

    draw = ImageDraw.Draw(image)
    title = font("segoeuib.ttf", 92 * SS)
    sub = font("segoeui.ttf", 40 * SS)
    small = font("segoeui.ttf", 31 * SS)

    x = 400 * SS
    draw.text((x, 176 * SS), "SongScribe", font=title, fill=WHITE)
    draw.text((x, 292 * SS), "AI Music Prompt Nodes for ComfyUI", font=sub, fill=ACCENT)
    draw.text(
        (x, 356 * SS),
        "73 style presets  ·  song analysis  ·  runs on CPU",
        font=small,
        fill=MUTED,
    )

    waveform(draw, 96 * SS, 1184 * SS, 520 * SS, 62 * SS)

    draw.text(
        (96 * SS, 578 * SS),
        "MiniMax Music 3  ·  YuE2",
        font=small,
        fill=MUTED,
    )

    image = image.convert("RGB").resize((W, H), Image.LANCZOS)
    image.save(OUT, "PNG", optimize=True)
    print(f"wrote {OUT} ({W}x{H}, {os.path.getsize(OUT) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
