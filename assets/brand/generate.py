"""Generates the Ragdoll mark, wordmark, lockups and social card as outlined SVG.

Usage: python3 generate.py <out-dir> <path/to/Geist-SemiBold.ttf>
Requires fontTools. The Geist TTFs ship in site/node_modules/geist/dist/fonts/geist-sans/.
"""
import sys
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

OUT = sys.argv[1]
FONT = sys.argv[2]
INK, PAPER, TALLY = "#0E0F12", "#F4F2EC", "#FF4F2E"

# Glyph on a 64-unit grid: a 3x3 lattice of 12-unit blocks with 3-unit gutters.
def glyph(fg, accent):
    r = 1.5
    blocks = [(11, 26), (11, 41), (26, 11), (26, 26)]
    s = "".join(f'<rect x="{x}" y="{y}" width="12" height="12" rx="{r}"/>' for x, y in blocks)
    s += '<path d="M41 11h.5a13.5 13.5 0 0 1 0 27H41Z"/>'
    s += '<path d="M30 41h12l11 12H41Z"/>'
    return (f'<g fill="{fg}">{s}</g>'
            f'<rect x="11" y="11" width="12" height="12" rx="{r}" fill="{accent}"/>')

def svg(w, h, body, vb=None):
    vb = vb or f"0 0 {w} {h}"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="{w}" height="{h}" '
            f'role="img" aria-label="Ragdoll">{body}</svg>\n')

def tile(bg, fg):
    return svg(64, 64, f'<rect width="64" height="64" rx="14" fill="{bg}"/>' + glyph(fg, TALLY))

font = TTFont(FONT)
gs = font.getGlyphSet()
cmap = font.getBestCmap()
upm = font["head"].unitsPerEm

def text_path(text, size, x0, baseline, tracking=-0.035):
    scale = size / upm
    x = 0.0
    pen = SVGPathPen(gs)
    for ch in text:
        name = cmap[ord(ch)]
        tp = TransformPen(pen, (scale, 0, 0, -scale, x0 + x * scale, baseline))
        gs[name].draw(tp)
        x += gs[name].width + tracking * upm
    return pen.getCommands(), (x - tracking * upm) * scale

def wordmark(fg, name):
    # Mark at 64 units tall, wordmark cap-aligned beside it.
    d, width = text_path("ragdoll", 46, 78, 47)
    w = round(78 + width + 6)
    body = f'<g transform="translate(-6 0)">{glyph(fg, TALLY)}</g><path fill="{fg}" d="{d}"/>'
    open(f"{OUT}/{name}", "w").write(svg(w, 64, body))

open(f"{OUT}/mark.svg", "w").write(tile(INK, PAPER))
open(f"{OUT}/mark-light.svg", "w").write(tile(PAPER, INK))
open(f"{OUT}/glyph.svg", "w").write(svg(64, 64, glyph("currentColor", TALLY)))
wordmark(INK, "wordmark-light-bg.svg")
wordmark(PAPER, "wordmark-dark-bg.svg")

# Social card (1200x630): ink field, faint macroblock lattice, lockup and tagline.
def social_card():
    W, H = 1200, 630
    grid = "".join(f'<path d="M{x} 0V{H}"/>' for x in range(0, W + 1, 48))
    grid += "".join(f'<path d="M0 {y}H{W}"/>' for y in range(0, H + 1, 48))
    name, _ = text_path("ragdoll", 120, 236, 290)
    tag, _ = text_path("Media, from first principles.", 52, 96, 446, tracking=-0.02)
    sub, _ = text_path("Containers, codecs and pixels in pure Rust. No C libraries.", 30, 96, 506, tracking=-0.01)
    body = (
        f'<rect width="{W}" height="{H}" fill="{INK}"/>'
        f'<g stroke="#1d2026" stroke-width="1">{grid}</g>'
        f'<g transform="translate(70 168) scale(2.4)">{glyph(PAPER, TALLY)}</g>'
        f'<path fill="{PAPER}" d="{name}"/>'
        f'<path fill="{PAPER}" d="{tag}"/>'
        f'<path fill="#8F8E88" d="{sub}"/>'
    )
    open(f"{OUT}/og.svg", "w").write(svg(W, H, body))

social_card()
