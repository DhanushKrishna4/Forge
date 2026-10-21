"""Render Forge's static sharing artwork; no browser or live service required."""

from __future__ import annotations

import argparse
import base64
from functools import lru_cache
from html import escape
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCALE = 4
WIDTH, HEIGHT = 1200, 630
BG = "#191c1b"
CREAM = "#e9ebe4"
COPPER = "#e6a27a"
BLUE = "#8fb6c8"
MUTED = "#a0a69f"
LINE = "#383e38"
PANEL = "#242a26"


@lru_cache(maxsize=100)
def font(size, weight=450, mono=False):
    path = HERE / "fonts" / ("IBMPlexMono-Regular.ttf" if mono else "DMSans.ttf")
    face = ImageFont.truetype(str(path), round(size * SCALE))
    if not mono:
        face.set_variation_by_axes([40, weight])
    return face


def rgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


class Poster:
    """One geometry source produces an editable SVG and a supersampled PNG."""

    def __init__(self):
        self.im = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), BG)
        self.draw = ImageDraw.Draw(self.im)
        self.svg = [f'<rect width="1200" height="630" fill="{BG}"/>']

    def line(self, points, color=LINE, width=1):
        self.draw.line([(round(x * SCALE), round(y * SCALE)) for x, y in points],
                       fill=color, width=max(1, round(width * SCALE)), joint="curve")
        coords = " ".join(f"{x},{y}" for x, y in points)
        self.svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"/>')

    def rect(self, x, y, w, h, color, outline=None, width=1):
        self.draw.rectangle([round(x*SCALE), round(y*SCALE), round((x+w)*SCALE)-1, round((y+h)*SCALE)-1],
                            fill=color, outline=outline, width=round(width*SCALE))
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" stroke="{outline or "none"}" stroke-width="{width}"/>')

    def circle(self, x, y, radius, color, outline=None, width=1):
        self.draw.ellipse([(x-radius)*SCALE, (y-radius)*SCALE, (x+radius)*SCALE, (y+radius)*SCALE],
                          fill=color, outline=outline, width=round(width*SCALE))
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="{outline or "none"}" stroke-width="{width}"/>')

    def text(self, x, baseline, text, size, color=CREAM, weight=450, mono=False):
        face = font(size, weight, mono)
        self.draw.text((x*SCALE, baseline*SCALE), text, font=face, fill=color, anchor="ls")
        family = "IBM Plex Mono" if mono else "DM Sans"
        self.svg.append(f'<text x="{x}" y="{baseline}" fill="{color}" font-family="{family}" font-size="{size}" font-weight="{400 if mono else weight}" font-optical-sizing="none" style="font-variation-settings: &quot;opsz&quot; 40, &quot;wght&quot; {weight}">{escape(text)}</text>')
        return face.getlength(text) / SCALE

    def centered_text(self, cx, cy, text, size, color=CREAM, weight=450, mono=False):
        face = font(size, weight, mono)
        left, top, right, bottom = face.getbbox(text, anchor="ls")
        self.text(cx - (left+right)/(2*SCALE), cy - (top+bottom)/(2*SCALE),
                  text, size, color, weight, mono)

    def image(self, image, x, y, w, h):
        image = image.convert("RGB")
        self.im.paste(image.resize((round(w*SCALE), round(h*SCALE)), Image.Resampling.BICUBIC), (round(x*SCALE), round(y*SCALE)))
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        data = base64.b64encode(buf.getvalue()).decode()
        self.svg.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/png;base64,{data}"/>')

    def save(self, path, svg_path=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = self.im.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        save_png(image, path)
        css = []
        for name, file in [("DM Sans", "DMSans.ttf"), ("IBM Plex Mono", "IBMPlexMono-Regular.ttf")]:
            data = base64.b64encode((HERE / "fonts" / file).read_bytes()).decode()
            css.append(f'@font-face{{font-family:"{name}";src:url(data:font/ttf;base64,{data}) format("truetype");font-weight:100 1000;}}')
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">'
        svg += '<title>Forge — the inspectable learning machine</title><defs><style>' + "".join(css) + '</style></defs>'
        (svg_path or path.with_suffix(".svg")).write_text(svg + "".join(self.svg) + "</svg>\n")
        return image


def save_png(image, path):
    info = PngImagePlugin.PngInfo()
    info.add(b"sRGB", b"\x00")
    image.convert("RGB").save(path, format="PNG", optimize=True, pnginfo=info)


def mark(p, x, y, s=1):
    # Existing Forge F mark, translated from favicon geometry.
    p.rect(x, y, 34*s, 9*s, COPPER)
    p.rect(x, y, 10*s, 43*s, COPPER)
    p.rect(x, y+18*s, 29*s, 9*s, COPPER)
    p.rect(x+22*s, y+33*s, 12*s, 10*s, COPPER)


def learned_surface(fixture):
    p = Poster()
    p.text(70, 91, "THE INSPECTABLE LEARNING MACHINE", 15, MUTED, mono=True)
    p.text(62, 275, "forge.", 166, CREAM, 550)
    p.text(73, 346, "Deep learning,", 35)
    p.text(73, 391, "open to inspection.", 35, COPPER)
    p.line([(73, 483), (475, 483)])
    p.text(73, 521, "PYTHON → GRAPH → WGSL", 17, MUTED, mono=True)
    grid = fixture["grid"]
    colors = []
    for v in grid["probabilities"]:
        a, b = rgb(COPPER), rgb(BLUE)
        colors.append(tuple(round((a[i]*(1-v)+b[i]*v)*0.72 + rgb(BG)[i]*0.28) for i in range(3)))
    field = Image.new("RGB", (grid["width"], grid["height"]))
    field.putdata(colors)
    field = field.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    x, y, size = 614, 76, 510
    p.image(field, x, y, size, size)
    for sample in fixture["samples"]:
        px = x + (sample["x"]+2.5)/5*size
        py = y + (2.5-sample["y"])/5*size
        p.circle(px, py, 4.2, CREAM if sample["label"] == 0 else BG, BG, .75)
    p.text(633, 112, "LEARNED IN FORGE", 13, BG, mono=True)
    p.text(633, 566, "MOONS / SEED 42 / STEP 400", 13, BG, mono=True)
    return p


def proofs(image):
    folder = HERE / "proofs"
    folder.mkdir(exist_ok=True)
    for w, h in [(600, 315), (300, 158)]:
        save_png(image.resize((w, h), Image.Resampling.LANCZOS), folder / f"preview-{w}.png")
    # Crops stress-test edges; these are not alternate recommended OG layouts.
    save_png(image.crop((40, 0, 1160, 630)), folder / "crop-16x9.png")
    save_png(image.crop((180, 0, 1020, 630)), folder / "crop-4x3.png")
    save_png(image.crop((285, 0, 915, 630)), folder / "crop-square.png")


def compilation(fixture, refined=False):
    f = fixture["fusionExample"]
    assert f["unfusedKernelCount"] == 3 and f["fusedKernelCount"] == 1
    p = Poster()
    p.text(66, 244, "forge.", 222, CREAM, 550)
    p.text(767, 113, "The inspectable", 31)
    p.text(767, 154, "learning machine.", 31, COPPER)
    p.text(771, 214, "PYTHON → GRAPH → WGSL", 14, MUTED, mono=True)
    p.line([(76, 287), (1124, 287)])
    p.text(77, 327, "THREE OPERATIONS", 14, BLUE, mono=True)
    p.text(782, 327, "ONE COMPILED KERNEL", 14, COPPER, mono=True)
    for i, (op, code) in enumerate([("add", "x + bias"), ("mul", "× scale"), ("tanh", "tanh(·)")]):
        x = 76 + i*183
        p.rect(x, 355, 154, 132, PANEL, LINE)
        p.rect(x, 355, 154, 4, BLUE)
        p.text(x+18, 412, op, 39, CREAM, mono=True)
        p.text(x+18, 461, code, 14, MUTED, mono=True)
        if i < 2:
            p.line([(x+154, 421), (x+183, 421)], BLUE, 2)
    p.line([(596, 421), (744, 421)], COPPER, 2)
    p.line([(732, 411), (744, 421), (732, 431)], COPPER, 2)
    p.rect(781, 355, 343, 132, COPPER)
    p.text(800, 410, "tanh((x + b) * s)", 23, BG, mono=True)
    p.line([(802, 442), (1099, 442)], "#ae795a", 1)
    p.text(802, 469, "WGSL", 15, BG, mono=True)
    p.text(1032, 469, "float32", 13, BG, mono=True)
    p.text(78, 570, "A MODEL BECOMES A PROGRAM.", 15, MUTED, mono=True)
    p.text(822, 570, "dhanushkrishna4.github.io/Forge", 12, MUTED, mono=True)
    return p


def polished_compilation(fixture, final=False):
    f = fixture["fusionExample"]
    assert [op["op"] for op in f["sourceOperations"]] == ["add", "mul", "tanh"]
    assert (f["unfusedKernelCount"], f["fusedKernelCount"]) == (3, 1)
    p = Poster()
    end = p.text(65, 238, "forge", 211, CREAM, 550)
    p.text(65 + end - 3, 238, ".", 211, COPPER, 550)
    p.text(770, 117, "The inspectable", 31)
    p.text(770, 158, "learning machine.", 31, COPPER)
    if final:
        p.text(773, 214, "Deep learning framework", 23, MUTED)
        p.text(773, 245, "+ compiler", 23, MUTED)
    else:
        p.text(773, 222, "PYTHON FRAMEWORK + COMPILER", 14, MUTED, mono=True)
    p.text(78, 323, "SOURCE OPERATIONS", 14, BLUE, mono=True)
    p.text(782, 323, "FUSED KERNEL", 14, COPPER, mono=True)
    for i, op in enumerate(["+", "×", "tanh"]):
        x = 76 + i*183
        p.rect(x, 349, 154, 151, PANEL, LINE)
        p.rect(x, 349, 154, 4, BLUE)
        if final:
            p.centered_text(x+77, 427, op, 78 if i < 2 else 46, CREAM, mono=True)
        else:
            p.text(x + (40 if i < 2 else 20), 448 if i < 2 else 444, op, 78 if i < 2 else 46, CREAM, mono=True)
        if i < 2:
            p.line([(x+154, 425), (x+183, 425)], BLUE, 2)
    p.line([(596, 425), (744, 425)], COPPER, 2)
    p.line([(732, 415), (744, 425), (732, 435)], COPPER, 2)
    p.rect(781, 349, 343, 151, COPPER)
    p.text(803, 419, "WGSL", 55, BG, mono=True)
    p.text(805, 468, "tanh((x + b) * s)", 21, BG, mono=True)
    p.text(78, 574, "A model becomes a program.", 24, CREAM if final else MUTED)
    if not final:
        p.text(994, 574, "3 ops → 1", 14, MUTED, mono=True)
    return p


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--studies", action="store_true")
    args.add_argument("--polish", action="store_true")
    options = args.parse_args()
    fixture = json.loads((HERE / "fixtures" / "moons.json").read_text())
    if options.studies:
        learned_surface(fixture).save(HERE / "studies" / "01-learned-surface.png")
        compilation(fixture).save(HERE / "studies" / "02-compilation.png")
        return
    if options.polish:
        image = polished_compilation(fixture).save(HERE / "studies" / "03-first-polished.png")
        proofs(image)
        return
    card = polished_compilation(fixture, final=True)
    image = card.save(ROOT / "web" / "public" / "og-v1.png", HERE / "card.svg")
    proofs(image)


if __name__ == "__main__":
    main()
