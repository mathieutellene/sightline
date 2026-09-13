"""Render what the network actually sees, layer by layer, for every Chicago zone.

The numbers in the README say the ordering transfers and the level does not.
They do not show anybody *how* a satellite chip becomes a demand estimate. This
runs the trained network over each held-out zone and captures the activation
after every convolutional block, so the page can show the whole path: 1.28 km of
ground -> 32 feature maps -> 64 -> 128 -> 128 -> one number.

Channels within a layer are ordered by variance across the whole test set, not
per zone. Ordering them per zone would put a different filter in the same tile
for every zone, and the grid would stop being comparable between them — the
thing a viewer most wants to do is flick between a busy zone and a quiet one and
see the same filter respond differently.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sightline import demand, tiles, zones as Z          # noqa: E402
from sightline.model import DemandNet, normalise, to_target  # noqa: E402

OUT = ROOT / "docs" / "viz"
TILE_GRID = 4          # 4x4 = 16 channels shown per layer
MIN_LAYER_PX = 320     # mosaics are upscaled by a whole factor to at least this
CHIP_PX = 384          # the satellite chip, rendered for a 2x display


def ramp(x: np.ndarray) -> np.ndarray:
    """Activation -> the project's pale-blue -> blue -> red ramp, as RGB uint8.

    The same ramp the map uses, so a bright patch in a feature map and a busy
    zone on the map read as the same kind of "high" without a legend. It runs
    from near-white so that the mosaics sit on a light page without turning
    into dark rectangles.
    """
    cold = np.array([214, 227, 252]); warm = np.array([66, 133, 244])
    hot = np.array([217, 48, 37])
    x = np.clip(x, 0, 1)[..., None]
    lo = cold + (warm - cold) * np.clip(x / 0.5, 0, 1)
    hi = warm + (hot - warm) * np.clip((x - 0.5) / 0.5, 0, 1)
    return np.where(x < 0.5, lo, hi).astype(np.uint8)


def mosaic(act: np.ndarray, order: np.ndarray, scale: np.ndarray) -> Image.Image:
    """(C,H,W) activation -> a TILE_GRID² mosaic of its most varied channels.

    `scale` is one display anchor per channel, measured across the whole test
    set. Stretching each tile to its own range instead would be brighter, and
    would be a lie: a quiet zone's faint response would be pushed up to look
    exactly as strong as a busy zone's, and comparing the two is the only
    reason the tiles are arranged this way.
    """
    n = TILE_GRID * TILE_GRID
    chans = act[order[:n]]
    tops = scale[order[:n]]
    h, w = chans.shape[1:]
    pad = 1
    canvas = np.zeros((TILE_GRID * (h + pad) - pad,
                       TILE_GRID * (w + pad) - pad, 3), dtype=np.uint8)
    canvas[:] = 255                                  # white gutters
    for i, (c, top) in enumerate(zip(chans, tops)):
        # The anchor is the 99th percentile rather than the maximum, because one
        # hot pixel anywhere in the set would otherwise flatten every tile; the
        # gamma lifts mid-tones a ramp starting near white would else lose.
        norm = np.clip(c / top, 0, 1) ** 0.7 if top > 0 else np.zeros_like(c)
        r, q = divmod(i, TILE_GRID)
        y0, x0 = r * (h + pad), q * (w + pad)
        canvas[y0:y0 + h, x0:x0 + w] = ramp(norm)
    # Whole-number upscaling only: resampling a 35 px mosaic to a fixed width
    # aliases the very structure the tile exists to show.
    side = canvas.shape[0]
    factor = max(1, -(-MIN_LAYER_PX // side))
    return Image.fromarray(canvas).resize((side * factor,) * 2, Image.NEAREST)


# Chicago's community-area layer stops at the shoreline, so the water east of it
# has to be reconstructed if the map is to read as Chicago at a glance. For each
# latitude the lake is everything between the city's easternmost point and the
# edge of the frame. The scan stops where the shore reaches the Indiana line:
# below that the city's eastern neighbour is Hammond and Gary, not water.
SHORE_MIN_LAT = 41.705
# South of that the real shore keeps going, east-south-east along the Indiana
# dunes, dropping roughly this much latitude across the width of the frame.
# Squaring the water off at SHORE_MIN_LAT instead leaves a ruler-straight edge
# that reads as a rendering fault rather than as a coastline.
SHORE_FALL = 0.085


def lake_ring(union, steps: int = 260) -> list[list[float]]:
    """The Lake Michigan side of the city boundary, closed off to the east."""
    from shapely.geometry import LineString

    minx, miny, maxx, maxy = union.bounds
    east = maxx + 0.35
    lo = max(miny, SHORE_MIN_LAT)
    shore = []
    for i in range(steps + 1):
        lat = lo + (maxy - lo) * i / steps
        hit = union.intersection(LineString([(minx - 0.1, lat), (east, lat)]))
        if hit.is_empty:
            continue
        shore.append([round(hit.bounds[2], 5), round(lat, 5)])
    if not shore:
        return []
    return shore + [[east, shore[-1][1]],
                    [east, round(shore[0][1] - SHORE_FALL, 5)],
                    shore[0]]


def main():
    run = json.loads((ROOT / "data" / "runs" / "result.json").read_text())
    state = torch.load(ROOT / "data" / "runs" / "demandnet.pt", weights_only=True)
    net = DemandNet(); net.load_state_dict(state); net.eval()
    mu, sd = run["target_mu"], run["target_sd"]

    zs = Z.load("chicago")
    chips = tiles.build("chicago", zs, verbose=False)
    dens = demand.density("chicago", zs)
    keys = [z.key for z in zs if z.key in chips and z.key in dens]
    by = {z.key: z for z in zs}

    X = torch.from_numpy(normalise(np.stack([chips[k] for k in keys])))

    # Forward pass, capturing the output of each of the four blocks.
    acts: list[np.ndarray] = []
    with torch.no_grad():
        h = X
        for block in net.features[:-1]:
            h = block(h)
            acts.append(h.numpy())
        pooled = net.features[-1](h)
        # head() keeps a trailing dim that forward() squeezes; do it here too.
        pred = (net.head(pooled).squeeze(-1).numpy() * sd + mu)
    embed = pooled.squeeze(-1).squeeze(-1).numpy()

    # One channel ordering per layer, and one display scale per channel, both
    # measured across every zone so that a tile means the same thing in all of
    # them — which is the whole point of laying them out this way.
    orders = [a.var(axis=(0, 2, 3)).argsort()[::-1] for a in acts]
    scales = [np.percentile(a, 99, axis=(0, 2, 3)) for a in acts]

    # How hard the channels on display respond, against the demand the network
    # is being asked about. The sign flips deep in the stack — the early blocks
    # answer to texture, the last ones to emptiness — and the page is only
    # entitled to say so because this measures it. Hard-coding the numbers into
    # the prose is how a page ends up describing a model it no longer has.
    lgd = np.log10(np.maximum(np.array([dens[k] for k in keys]), 1))
    shown = TILE_GRID * TILE_GRID
    response = [round(float(np.corrcoef(
        a[:, orders[i][:shown]].mean(axis=(1, 2, 3)), lgd)[0, 1]), 3)
        for i, a in enumerate(acts)]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "layers").mkdir(exist_ok=True)
    (OUT / "chips").mkdir(exist_ok=True)

    meta = {}
    for i, k in enumerate(keys):
        Image.fromarray(np.transpose(chips[k], (1, 2, 0))).resize(
            (CHIP_PX, CHIP_PX), Image.LANCZOS).save(OUT / "chips" / f"{k}.jpg", quality=88)
        for li, a in enumerate(acts):
            mosaic(a[i], orders[li], scales[li]).save(OUT / "layers" / f"{k}_L{li + 1}.png")
        truth = dens[k]
        meta[k] = {
            "name": by[k].name.title(),
            "zone_id": by[k].zone_id,
            "lon": round(by[k].lon, 5), "lat": round(by[k].lat, 5),
            "area_km2": round(by[k].area_km2, 2),
            "pred": round(float(10 ** pred[i]), 1),
            "true": round(float(truth), 1),
            "embed": [round(float(v), 3) for v in embed[i]],
        }

    (OUT / "zones.json").write_text(json.dumps({
        "zones": meta,
        "layers": [{"name": f"block {i+1}",
                    "shape": list(a.shape[1:]),
                    "channels": int(a.shape[1]),
                    "response_r": response[i]} for i, a in enumerate(acts)],
        "metrics": run["chicago"],
        "nyc_metrics": run["nyc_val"],
        "calibration": json.loads((ROOT / "data" / "runs" / "calibration.json").read_text()),
    }, indent=1))

    # Simplified geometry for the map: full precision is 2 MB of coastline
    # nobody will look at at this zoom.
    import geopandas as gpd
    gdf = gpd.read_file(ROOT / "data" / "zones" / "chicago_community_areas.geojson")
    gdf = gdf.rename(columns={"area_numbe": "zone_id"})
    gdf["zone_id"] = gdf["zone_id"].astype(str)
    gdf["key"] = "chicago-" + gdf["zone_id"]
    gdf = gdf[gdf["key"].isin(meta)][["key", "geometry"]]
    gdf["geometry"] = gdf.geometry.simplify(0.0004)
    gdf.to_file(OUT / "chicago.geojson", driver="GeoJSON")

    (OUT / "water.json").write_text(json.dumps(lake_ring(gdf.geometry.union_all())))

    size = sum(f.stat().st_size for f in OUT.rglob("*")) / 1e6
    print(f"{len(meta)} zonas | {len(acts)} capas por zona | {size:.1f} MB en docs/viz")
    for i, a in enumerate(acts):
        print(f"  bloque {i+1}: {a.shape[1]:>3} canales  {a.shape[2]}x{a.shape[3]}")


if __name__ == "__main__":
    main()
