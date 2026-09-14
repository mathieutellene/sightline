"""Render the figures the README embeds, from the data the web page already uses.

Nothing here is a screenshot. Every figure is drawn from `docs/viz/` — the same
chips, the same activation mosaics and the same metrics the page loads — so the
README cannot drift away from the result: re-run this and the pictures follow
whatever the model now does.

Drawing is Pillow only, which the project already depends on for the chips. It
has no antialiasing of its own, so everything is drawn at SS times the final
size and resampled down, which gives clean polygon edges and round dots for
free. Run it after scripts/export_visuals.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
VIZ = ROOT / "docs" / "viz"
OUT = ROOT / "docs" / "figures"

SS = 3                                   # supersampling factor
LAND = (233, 235, 238)
WATER = (170, 218, 255)
PAPER = (255, 255, 255)
INK = (31, 33, 38)
MUTED = (95, 99, 104)
DIM = (133, 139, 146)
LINE = (226, 229, 233)
BLUE = (26, 115, 232)
DEEP = (23, 78, 166)
RED = (217, 48, 37)

# Same three stops as docs/app.js and the exported mosaics.
COLD, MID, HOT = np.array([214, 227, 252]), np.array([66, 133, 244]), np.array([217, 48, 37])

FONTS = [
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
FONTS_BOLD = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
_warned = False


def font(size: int, bold: bool = False):
    """A real typeface if one can be found, the bitmap default if not."""
    global _warned
    for path in (FONTS_BOLD if bold else FONTS):
        if Path(path).exists():
            return ImageFont.truetype(path, size * SS)
    if not _warned:
        print("  aviso: sin fuentes TrueType, las etiquetas saldran en la fuente por defecto")
        _warned = True
    return ImageFont.load_default()


def ramp(t: float) -> tuple[int, int, int]:
    t = min(max(t, 0.0), 1.0)
    a, b, k = (COLD, MID, t / 0.5) if t < 0.5 else (MID, HOT, (t - 0.5) / 0.5)
    return tuple(int(round(v)) for v in a + (b - a) * k)


def fmt(n: float) -> str:
    return f"{round(n):,}" if n >= 1000 else f"{round(n)}" if n >= 10 else f"{n:.1f}"


class Fig:
    """A figure drawn at SS scale. Every coordinate passed in is in final pixels."""

    def __init__(self, w: int, h: int, bg=PAPER):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w * SS, h * SS), bg)
        self.d = ImageDraw.Draw(self.img)

    def text(self, xy, s, size=13, bold=False, fill=INK, anchor="la"):
        self.d.text((xy[0] * SS, xy[1] * SS), s, font=font(size, bold), fill=fill, anchor=anchor)

    def width_of(self, s, size=13, bold=False) -> float:
        return self.d.textlength(s, font=font(size, bold)) / SS

    def line(self, pts, fill, width=1):
        self.d.line([(x * SS, y * SS) for x, y in pts], fill=fill, width=max(1, int(width * SS)))

    def dot(self, x, y, r, fill, outline=None, ow=1):
        self.d.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS],
                       fill=fill, outline=outline, width=max(1, int(ow * SS)))

    def box(self, xy, radius=0, fill=None, outline=None, ow=1):
        x0, y0, x1, y1 = [v * SS for v in xy]
        if radius:
            self.d.rounded_rectangle([x0, y0, x1, y1], radius=radius * SS, fill=fill,
                                     outline=outline, width=max(1, int(ow * SS)))
        else:
            self.d.rectangle([x0, y0, x1, y1], fill=fill, outline=outline,
                             width=max(1, int(ow * SS)))

    def paste(self, im: Image.Image, x, y, side):
        n = int(round(side * SS))
        # NEAREST keeps the mosaics' whole pixels whole; the chip is photographic.
        mode = Image.NEAREST if im.width <= 600 and im.width % 4 else Image.LANCZOS
        self.img.paste(im.resize((n, n), mode), (int(x * SS), int(y * SS)))

    def chevron(self, x, y, size=9):
        self.line([(x - size * .35, y - size), (x + size * .45, y), (x - size * .35, y + size)],
                  fill=(205, 210, 216), width=2.2)

    def save(self, name: str):
        OUT.mkdir(parents=True, exist_ok=True)
        self.img.resize((self.w, self.h), Image.LANCZOS).save(OUT / name, optimize=True)
        kb = (OUT / name).stat().st_size / 1000
        print(f"  {name:18} {self.w}x{self.h}  {kb:.0f} KB")


# --------------------------------------------------------------------- data
def load():
    data = json.loads((VIZ / "zones.json").read_text())
    geo = json.loads((VIZ / "chicago.geojson").read_text())
    water = json.loads((VIZ / "water.json").read_text())
    trues = sorted(z["true"] for z in data["zones"].values())
    lo = np.log10(max(trues[int(round(0.05 * (len(trues) - 1)))], 1))
    hi = np.log10(max(trues[int(round(0.95 * (len(trues) - 1)))], 10))
    return data, geo, water, float(lo), float(hi)


def merc(lat: float) -> float:
    return float(np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * (180 / np.pi))


# ---------------------------------------------------------------- the map
def figure_map(data, geo, water, lo, hi):
    W, H = 1180, 700
    f = Fig(W, H, LAND)
    rings = [(ft["properties"]["key"],
              [r for poly in ([ft["geometry"]["coordinates"]]
                              if ft["geometry"]["type"] == "Polygon"
                              else ft["geometry"]["coordinates"]) for r in poly])
             for ft in geo["features"] if ft["properties"]["key"] in data["zones"]]

    pts = [(lon, merc(lat)) for _, rs in rings for r in rs for lon, lat in r]
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)

    pad, left = 34, 330            # room for the caption panel on the left
    s = min((W - left - pad) / (x1 - x0), (H - pad * 2) / (y1 - y0))
    ox = left + ((W - left - pad) - (x1 - x0) * s) / 2
    oy = pad + ((H - pad * 2) - (y1 - y0) * s) / 2
    px = lambda lon: ox + (lon - x0) * s                       # noqa: E731
    py = lambda lat: oy + (y1 - merc(lat)) * s                 # noqa: E731

    f.d.polygon([(px(lon) * SS, py(lat) * SS) for lon, lat in water], fill=WATER)
    for key, rs in rings:
        t = (np.log10(max(data["zones"][key]["true"], 1)) - lo) / (hi - lo)
        for r in rs:
            f.d.polygon([(px(lon) * SS, py(lat) * SS) for lon, lat in r],
                        fill=ramp(t), outline=PAPER, width=max(1, int(1.1 * SS)))

    # Two landmarks, so the shape means something to someone who has never been.
    for name, label, dx, dy in [("Loop", "The Loop", 26, -6), ("Ohare", "O'Hare", -8, -26)]:
        z = next((v for v in data["zones"].values()
                  if v["name"].replace("'", "").replace("-", "").startswith(name)), None)
        if not z:
            continue
        cx, cy = px(z["lon"]), py(z["lat"])
        tw = f.width_of(label, 12, True)
        bx, by = cx + dx, cy + dy
        f.line([(cx, cy), (bx + (0 if dx > 0 else tw), by)], fill=(70, 74, 80), width=1.2)
        f.box((bx - 6, by - 11, bx + tw + 6, by + 11), 5, fill=PAPER)
        f.text((bx, by), label, 12, True, INK, anchor="lm")
        f.dot(cx, cy, 3.2, (70, 74, 80), PAPER, 1.4)

    f.box((0, 0, left - 26, H), fill=PAPER)
    f.text((34, 46), "Chicago", 30, True)
    f.text((34, 88), "77 community areas", 15, False, MUTED)
    f.text((34, 150), "MEASURED RIDE DEMAND", 11, True, DIM)
    f.text((34, 172), "trips per km² per day, June 2022", 12.5, False, MUTED)

    bx0, bx1, by = 34, left - 60, 224
    for i in range(int((bx1 - bx0) * SS)):
        c = ramp(i / ((bx1 - bx0) * SS - 1))
        f.d.rectangle([bx0 * SS + i, by * SS, bx0 * SS + i + 1, (by + 9) * SS], fill=c)
    f.text((bx0, by + 16), fmt(10 ** lo), 11.5, False, DIM)
    f.text((bx1, by + 16), fmt(10 ** hi) + "+", 11.5, False, DIM, anchor="ra")

    f.text((34, 286), "This is the answer the network\nis not allowed to see.",
           14.5, True, INK)
    f.text((34, 340),
           "It is trained on New York only. Chicago\n"
           "is scored cold: every zone above is one\n"
           "it has never encountered.", 13, False, MUTED)

    m = data["metrics"]
    f.line([(34, 424), (left - 60, 424)], fill=LINE, width=1)
    f.text((34, 444), "SCORED COLD ON THIS MAP", 11, True, DIM)
    for i, (label, value, col) in enumerate([
            ("rank correlation", f"{m['spearman']:.2f}", DEEP),
            ("R²", f"{m['r2']:+.2f}", MUTED),
            ("typical level error", f"×{m['median_ratio_error']:.2f}", RED)]):
        y = 474 + i * 46
        f.text((34, y), value, 21, True, col)
        f.text((34 + max(f.width_of(value, 21, True), 62) + 14, y + 8), label,
               12.5, False, MUTED)

    f.text((34, H - 54), "Sentinel-2 © ESA/Copernicus\nBoundaries © City of Chicago",
           10.5, False, DIM)
    f.save("map.png")


# ------------------------------------------------------- the network path
def stages(data, key):
    out = [(Image.open(VIZ / "chips" / f"{key}.png"), "INPUT", "128² · 1.28 km", BLUE, None)]
    for i, layer in enumerate(data["layers"]):
        r = layer.get("response_r")
        out.append((Image.open(VIZ / "layers" / f"{key}_L{i + 1}.png"),
                    f"BLOCK {i + 1}", f"{layer['channels']} maps · {layer['shape'][1]}²",
                    INK, r))
    return out


def sig(v: float) -> str:
    return ("+" if v >= 0 else "−") + f"{abs(v):.2f}"


def figure_flow(data, key):
    tiles = stages(data, key)
    z = data["zones"][key]
    side, gap, pad = 196, 40, 34
    W = pad * 2 + len(tiles) * side + len(tiles) * gap + 210
    f = Fig(W, side + 150)

    f.text((pad, 30), "One square of ground, through the network", 19, True)
    f.text((pad, 58), f"{z['name']}, Chicago — a zone the model was never trained on",
           13.5, False, MUTED)

    x, y = pad, 96
    for im, label, sub, col, _ in tiles:
        f.box((x - 1, y - 1, x + side + 1, y + side + 1), 6, outline=LINE, ow=1)
        f.paste(im, x, y, side)
        f.text((x, y + side + 14), label, 12, True, col)
        f.text((x, y + side + 33), sub, 11.5, False, DIM)
        x += side
        f.chevron(x + gap / 2, y + side / 2)
        x += gap

    f.line([(x + 6, y + 20), (x + 6, y + side - 20)], fill=BLUE, width=3)
    f.text((x + 22, y + side / 2 - 26), fmt(z["pred"]), 38, True, DEEP)
    f.text((x + 22, y + side / 2 + 22), "predicted trips/km²/day", 12.5, False, MUTED)
    f.text((x + 22, y + side / 2 + 44), f"measured: {fmt(z['true'])}", 12.5, False, DIM)
    f.save("flow.png")


def figure_compare(data, busy, quiet, ranks):
    side, gap, labw, pad = 150, 30, 240, 34
    rows = [busy, quiet]
    ncol = 1 + len(data["layers"])
    W = pad * 2 + labw + ncol * side + (ncol - 1) * gap
    f = Fig(W, 196 + len(rows) * (side + 78))

    f.text((pad, 30), "The same filters, a busy zone and a quiet one", 19, True)
    f.text((pad, 58),
           "Every tile holds the same channel in both rows and is scaled against the same "
           "anchor, so a fainter tile really is a fainter response.\n"
           "Read the bottom row of numbers left to right: what the network responds to "
           "inverts on its way down.", 13.5, False, MUTED)

    y = 128
    for key in rows:
        z, tiles = data["zones"][key], stages(data, key)
        f.text((pad, y + 4), z["name"], 17, True)
        f.text((pad, y + 30), f"measured {fmt(z['true'])}   ·   predicted {fmt(z['pred'])}",
               12.5, False, MUTED)
        f.text((pad, y + 52), f"rank #{ranks[0][key]} of 77 measured, "
               f"#{ranks[1][key]} predicted", 12.5, False, DIM)
        x = pad + labw
        for im, label, sub, col, r in tiles:
            f.box((x - 1, y - 1, x + side + 1, y + side + 1), 5, outline=LINE, ow=1)
            f.paste(im, x, y, side)
            if key == rows[-1]:
                f.text((x, y + side + 12), label, 11.5, True, col)
                f.text((x, y + side + 29), sub, 11, False, DIM)
                if r is not None:
                    f.text((x, y + side + 50), f"response to demand  {sig(r)}", 11.5,
                           True, DEEP if r >= 0 else RED)
            x += side + gap
        y += side + 78
    f.save("compare.png")


# ------------------------------------------------------------- the result
def figure_results(data, lo, hi):
    W, H = 1180, 470
    f = Fig(W, H)
    pw = (W - 34 * 3) / 2

    # ---- left: measured against predicted, both log10
    x0, y0 = 34, 78
    m = dict(l=62, r=20, t=16, b=46)
    f.text((x0, 30), "Where it is right, and how it is wrong", 17, True)
    f.text((x0, 54), "every Chicago zone, log scale on both axes", 12.5, False, MUTED)
    px0, py0 = x0 + m["l"], y0 + m["t"]
    px1, py1 = x0 + pw - m["r"], y0 + H - 78 - m["b"]
    X = lambda v: px0 + v / 4 * (px1 - px0)                     # noqa: E731
    Y = lambda v: py1 - v / 4 * (py1 - py0)                     # noqa: E731
    for d in range(5):
        f.line([(px0, Y(d)), (px1, Y(d))], fill=(238, 240, 243), width=1)
        f.line([(X(d), py0), (X(d), py1)], fill=(238, 240, 243), width=1)
        f.text((px0 - 8, Y(d)), f"10{'⁰¹²³⁴'[d]}", 11.5, False, DIM, anchor="rm")
        f.text((X(d), py1 + 10), f"10{'⁰¹²³⁴'[d]}", 11.5, False, DIM, anchor="ma")
    for i in range(0, 44, 4):                                   # dashed diagonal
        a, b = i / 44, min((i + 2) / 44, 1)
        f.line([(X(a * 4), Y(a * 4)), (X(b * 4), Y(b * 4))], fill=(195, 201, 207), width=1.4)
    for z in data["zones"].values():
        t = (np.log10(max(z["true"], 1)) - lo) / (hi - lo)
        f.dot(X(np.log10(max(z["true"], 1))), Y(np.log10(max(z["pred"], 1))),
              4.6, ramp(t), PAPER, 1)
    # Pillow cannot rotate a text run, so the y label sits above its axis —
    # which reads as well and costs no extra render pass.
    f.text((px0 - 42, py0 - 20), "predicted", 12.5, False, MUTED)
    f.text(((px0 + px1) / 2, py1 + 32), "measured", 12.5, False, MUTED, anchor="ma")

    # ---- right: how much local truth fixes the level
    rx = 34 * 2 + pw
    f.text((rx, 30), "How much local truth fixes it", 17, True)
    f.text((rx, 54), "one scalar offset on k known zones, 400 draws each", 12.5, False, MUTED)
    rows = data["calibration"]
    maxe = max(r["median_ratio_error"] for r in rows)
    qx0, qx1 = rx + m["l"], rx + pw - m["r"]
    KX = lambda i: qx0 + i / (len(rows) - 1) * (qx1 - qx0)      # noqa: E731
    KY = lambda e: py1 - (e - 1) / (maxe - 1) * (py1 - py0)     # noqa: E731
    e = 1.0
    while e <= maxe + 1e-9:
        f.line([(qx0, KY(e)), (qx1, KY(e))], fill=(238, 240, 243), width=1)
        f.text((qx0 - 8, KY(e)), f"×{e:.1f}", 11.5, False, DIM, anchor="rm")
        e += 0.5
    f.line([(KX(i), KY(r["median_ratio_error"])) for i, r in enumerate(rows)],
           fill=BLUE, width=2.6)
    for i, r in enumerate(rows):
        first = i == 1
        f.dot(KX(i), KY(r["median_ratio_error"]), 6.5 if first else 4.4,
              PAPER if first else BLUE, BLUE, 2.6 if first else 1.5)
        f.text((KX(i), py1 + 10), str(r["k"]), 11.5, False, DIM, anchor="ma")
    f.text((KX(1) + 14, KY(rows[1]["median_ratio_error"]) - 22),
           f"one zone → ×{rows[1]['median_ratio_error']:.2f}", 13, True, INK)
    f.text((qx0 - 42, py0 - 20), "median error factor", 12.5, False, MUTED)
    f.text(((qx0 + qx1) / 2, py1 + 32), "zones where real demand is known",
           12.5, False, MUTED, anchor="ma")
    f.save("results.png")


# --------------------------------------------------- the city as streets
def figure_streets(data, water, lo, hi):
    """Chicago drawn only as its street network, tinted by what the model says.

    Needs data/osm/chicago_roads.json, which sightline/baseline.py fetches. The
    point of the picture is that none of the colour came from a ride record:
    every tint is the network's estimate for the zone the street sits in, and
    the streets outside the city are left grey because there is no estimate for
    them. It is the map a city gets before it has any data of its own.
    """
    path = ROOT / "data" / "osm" / "chicago_roads.json"
    geo_path = VIZ / "chicago.geojson"
    if not path.exists():
        print("  streets.png omitido (falta data/osm/chicago_roads.json)")
        return
    net = json.loads(path.read_text())
    geo = json.loads(geo_path.read_text())

    W, H = 1100, 1500
    f = Fig(W, H, PAPER)

    zones = []
    for ft in geo["features"]:
        key = ft["properties"]["key"]
        if key not in data["zones"]:
            continue
        rings = ([ft["geometry"]["coordinates"]] if ft["geometry"]["type"] == "Polygon"
                 else ft["geometry"]["coordinates"])
        flat = [r for poly in rings for r in poly]
        xs = [p[0] for r in flat for p in r]
        ys = [p[1] for r in flat for p in r]
        zones.append({"key": key, "rings": flat,
                      "bb": (min(xs), min(ys), max(xs), max(ys))})

    pts = [p for z in zones for r in z["rings"] for p in r]
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    my0 = min(merc(p[1]) for p in pts); my1 = max(merc(p[1]) for p in pts)
    pad = 30
    s = min((W - pad * 2) / (x1 - x0), (H - pad * 2) / (my1 - my0))
    ox = pad + ((W - pad * 2) - (x1 - x0) * s) / 2
    oy = pad + ((H - pad * 2) - (my1 - my0) * s) / 2
    px = lambda lon: ox + (lon - x0) * s                        # noqa: E731
    py = lambda lat: oy + (my1 - merc(lat)) * s                 # noqa: E731

    if water:
        f.d.polygon([(px(a) * SS, py(b) * SS) for a, b in water], fill=(233, 243, 252))

    def inside(ring, x, y):
        """Ray casting, on the ring as given."""
        c = False
        for (ax, ay), (bx, by) in zip(ring, ring[1:] + ring[:1]):
            if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
                c = not c
        return c

    # A coarse grid over zone bounding boxes, so each street segment is tested
    # against two or three zones instead of all seventy-seven.
    cell = 0.01
    grid: dict[tuple, list] = {}
    for z in zones:
        a, b, c, d = z["bb"]
        for gx in range(int(a / cell), int(c / cell) + 1):
            for gy in range(int(b / cell), int(d / cell) + 1):
                grid.setdefault((gx, gy), []).append(z)

    def zone_at(x, y):
        for z in grid.get((int(x / cell), int(y / cell)), ()):
            a, b, c, d = z["bb"]
            if a <= x <= c and b <= y <= d and any(inside(r, x, y) for r in z["rings"]):
                return z["key"]
        return None

    drawn = outside = 0
    for r in net:
        for a, b in zip(r["pts"], r["pts"][1:]):
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            key = zone_at(mx, my)
            if key:
                t = (np.log10(max(data["zones"][key]["pred"], 1)) - lo) / (hi - lo)
                col, wdt = ramp(t), 0.75
                drawn += 1
            else:
                col, wdt = (223, 226, 230), 0.5      # beyond the city: no estimate
                outside += 1
            f.line([(px(a[0]), py(a[1])), (px(b[0]), py(b[1]))], fill=col, width=wdt)

    f.box((30, 30, 434, 228), 10, fill=PAPER)
    f.text((50, 50), "Chicago, with nothing but its streets", 21, True)
    f.text((50, 84),
           "Every street tinted by the demand the network\n"
           "predicts for the zone it runs through — an estimate\n"
           "made from satellite imagery alone, with not one\n"
           "ride record from this city behind it.", 13, False, MUTED)
    f.text((50, 168), "Grey: outside the 77 community areas, where there is no estimate.",
           11.5, False, DIM)
    bx0, bx1, by = 50, 410, 190
    for i in range(int((bx1 - bx0) * SS)):
        f.d.rectangle([bx0 * SS + i, by * SS, bx0 * SS + i + 1, (by + 7) * SS],
                      fill=ramp(i / ((bx1 - bx0) * SS - 1)))
    f.text((bx0, by + 12), "quiet", 11, False, DIM)
    f.text((bx1, by + 12), "busy", 11, False, DIM, anchor="ra")
    f.text((30, H - 26), "Streets © OpenStreetMap contributors", 10.5, False, DIM)
    print(f"  ({drawn:,} segmentos en zonas, {outside:,} fuera)")
    f.save("streets.png")


# ----------------------------------------------------- how it learns
def figure_learning(data):
    """Three things measured every epoch, stacked on one time axis.

    The point is the contrast between the second panel and the third, so they
    share an x axis and sit directly on top of one another: a flat line above a
    thrashing one says it faster than either says alone.
    """
    h = data["history"]
    if not h:
        return
    W, PH, top = 1180, 132, 128
    f = Fig(W, top + 3 * (PH + 46) + 30)
    ep = [e["epoch"] for e in h]
    best = data["training"].get("best_epoch")

    f.text((34, 30), "What transfers is learned in three epochs. What does not, never settles.",
           19, True)
    f.text((34, 58),
           "Measured after every epoch while fitting on New York. Chicago is scored here for "
           "the record only — the epoch that ships is chosen\non New York's validation split "
           "and nothing else, which is why the vertical line does not sit at Chicago's best "
           "moment.", 13.5, False, MUTED)

    def panel(i, title, series, ylo, yhi, ticks, note=None):
        y0 = top + i * (PH + 46)
        x0, x1 = 34 + 64, W - 34
        X = lambda e: x0 + (e - ep[0]) / (ep[-1] - ep[0]) * (x1 - x0)     # noqa: E731
        Y = lambda v: y0 + PH - (min(max(v, ylo), yhi) - ylo) / (yhi - ylo) * PH  # noqa: E731
        f.text((34, y0 - 22), title, 14, True)
        for t in ticks:
            f.line([(x0, Y(t)), (x1, Y(t))], fill=(238, 240, 243), width=1)
            f.text((x0 - 8, Y(t)), f"{t:g}", 11.5, False, DIM, anchor="rm")
        if best:
            f.line([(X(best), y0), (X(best), y0 + PH)], fill=(205, 210, 216), width=1.4)
        for label, vals, col in series:
            # Clamping is visible as a flat run against the frame, and the note
            # says where the curve really went.
            f.line([(X(e), Y(v)) for e, v in zip(ep, vals)], fill=col, width=2.2)
        # The legend lives on the title line rather than beside the curves: two
        # entries right-aligned in the plot drew straight over each other, and
        # anywhere inside the frame is somewhere a curve can reach.
        lx = x1
        for label, _, col in reversed(series):
            w = f.width_of(label, 12, True)
            f.text((lx - w, y0 - 22), label, 12, True, col)
            f.line([(lx - w - 18, y0 - 14), (lx - w - 6, y0 - 14)], fill=col, width=2.4)
            lx -= w + 34
        if note:
            # Backed in white: the note sits inside the frame, and inside the
            # frame is exactly where a curve is free to run through it.
            nw = f.width_of(note, 11.5)
            f.box((x0 + 6, y0 + 4, x0 + 14 + nw, y0 + 22), 4, fill=PAPER)
            f.text((x0 + 10, y0 + 6), note, 11.5, False, DIM)
        if i == 2:
            for e in (1, 10, 20, 30, 40, 50, 60):
                f.text((X(e), y0 + PH + 26), str(e), 11.5, False, DIM, anchor="ma")
            f.text(((x0 + x1) / 2, y0 + PH + 44), "epoch", 12.5, False, MUTED, anchor="ma")
        if best and i == 0:
            f.text((X(best) + 8, y0 + 4), f"epoch {best} ships", 11.5, True, MUTED)

    panel(0, "Training loss on New York — the network is fitting",
          [("train loss", [e["train_loss"] for e in h], (95, 99, 104))],
          0, 0.30, [0, 0.1, 0.2, 0.3])
    panel(1, "Rank correlation on Chicago — the ordering, on a city never seen",
          [("Chicago  rank ρ", [e["chi_rho"] for e in h], BLUE)],
          0.70, 0.95, [0.7, 0.8, 0.9],
          note="flat from epoch 3 onward: σ = 0.005 after epoch 5")
    panel(2, "R² on Chicago — the absolute level, same epochs, same model",
          [("Chicago  R²", [e["chi_r2"] for e in h], RED),
           ("New York validation  R²", [e["val_r2"] for e in h], (150, 156, 163))],
          -1, 1, [-1, -0.5, 0, 0.5, 1],
          note="clamped to ±1; epoch 2 really reaches −37.5.  σ = 0.219 after epoch 5")
    f.save("learning.png")


def figure_transfer(data):
    """The same four epochs, on the city it is learning and the city it is not."""
    snaps = data.get("snapshots") or {}
    picks = [e for e in ("1", "4", "16", "60") if e in snaps]
    if len(picks) < 2:
        return
    side, gap, labw, pad = 230, 26, 158, 34
    W = pad * 2 + labw + len(picks) * side + (len(picks) - 1) * gap
    f = Fig(W, 150 + 2 * (side + 40))

    f.text((pad, 30), "Supervised learning, and then the part with no answer key", 19, True)
    f.text((pad, 58),
           "Top row: New York, where every dot has a published answer the network is "
           "corrected against. Bottom row: Chicago, run through the\nsame weights at the same "
           "epochs, with nothing to correct against. Both axes are log trips/km²/day; the "
           "diagonal is a perfect call.", 13.5, False, MUTED)

    vals = [v for k in picks for key in ("val_true", "val_pred", "chi_true", "chi_pred")
            for v in snaps[k][key]]
    lo, hi = min(vals + [0.0]), max(vals)
    lo, hi = np.floor(lo), np.ceil(hi)

    for row, (tk, pk, title, sub) in enumerate([
            ("val_true", "val_pred", "NEW YORK", "held-out zones,\nwith published answers"),
            ("chi_true", "chi_pred", "CHICAGO", "never seen,\nno answers at all")]):
        y = 128 + row * (side + 40)
        f.text((pad, y + 4), title, 13, True, BLUE if row else INK)
        f.text((pad, y + 24), sub, 11.5, False, DIM)
        x = pad + labw
        for k in picks:
            f.box((x, y, x + side, y + side), 8, fill=(250, 251, 252), outline=LINE, ow=1)
            m = 16
            X = lambda v: x + m + (v - lo) / (hi - lo) * (side - 2 * m)      # noqa: E731
            Y = lambda v: y + side - m - (v - lo) / (hi - lo) * (side - 2 * m)  # noqa: E731
            for i in range(0, 40, 4):                       # dashed diagonal
                a, b = lo + (hi - lo) * i / 40, lo + (hi - lo) * min((i + 2) / 40, 1)
                f.line([(X(a), Y(a)), (X(b), Y(b))], fill=(205, 210, 216), width=1.2)
            for t, p in zip(snaps[k][tk], snaps[k][pk]):
                f.dot(X(t), Y(min(max(p, lo), hi)), 3.6,
                      ramp((t - lo) / (hi - lo)), PAPER, .8)
            if row == 1:
                f.text((x + side / 2, y + side + 12), f"epoch {k}", 12, True, MUTED,
                       anchor="ma")
            x += side + gap
    f.save("transfer.png")


def main():
    data, geo, water, lo, hi = load()
    keys = list(data["zones"])
    by_true = sorted(keys, key=lambda k: -data["zones"][k]["true"])
    by_pred = sorted(keys, key=lambda k: -data["zones"][k]["pred"])
    ranks = ({k: i + 1 for i, k in enumerate(by_true)},
             {k: i + 1 for i, k in enumerate(by_pred)})

    print(f"figuras desde docs/viz ({len(keys)} zonas):")
    figure_map(data, geo, water, lo, hi)
    figure_streets(data, water, lo, hi)
    figure_learning(data)
    figure_transfer(data)
    figure_flow(data, by_true[0])
    figure_compare(data, by_true[0], by_true[-1], ranks)
    figure_results(data, lo, hi)


if __name__ == "__main__":
    main()
