"""One satellite chip per zone, cut from free Sentinel-2 imagery.

Three decisions here that the model's validity rests on.

FIXED GROUND EXTENT, not fixed zone. Every chip covers the same 1.28 km on the
ground, so 128 px always means the same thing. A convolutional network has no
way to recover scale from pixels alone: feed it whole zones and it would see a
Manhattan block and a Chicago neighbourhood at the same pixel size and conclude
they are the same kind of place. The cost is that for a large zone the chip sees
only the core -- recorded in the README as a limitation, not hidden.

SAME SEASON AS THE DEMAND. Imagery is drawn from the summer of the study month.
Predicting June 2022 demand from 2026 imagery would leak four years of
construction into the features.

ONE SCENE PER MGRS TILE. A single Sentinel-2 scene covers 110 km, so a whole
city usually fits in one or two. Searching once per city and reading every zone
window out of those few COGs turns hundreds of HTTP searches into a handful.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")

import rasterio                                          # noqa: E402
from rasterio.warp import transform_bounds                # noqa: E402
from rasterio.windows import from_bounds                  # noqa: E402
from pystac_client import Client                          # noqa: E402

STAC = "https://earth-search.aws.element84.com/v1/"
CACHE = Path(__file__).resolve().parents[1] / "data" / "chips"

CHIP_PX = 128
CHIP_M = 1280.0            # 1.28 km at 10 m/px
MAX_CLOUD = 10

CITY_BBOX = {
    "nyc": (-74.30, 40.47, -73.68, 40.93),
    "chicago": (-87.95, 41.62, -87.50, 42.06),
}
# Imagery window per study month: a season, so there is something cloud-free.
SEASON = {"2022-06": "2022-05-01/2022-09-15"}


def _scenes(city: str, month: str) -> dict[str, object]:
    """Least-cloudy scene per MGRS tile covering the city."""
    cat = Client.open(STAC)
    items = list(cat.search(
        collections=["sentinel-2-l2a"],
        bbox=CITY_BBOX[city],
        datetime=SEASON[month],
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        max_items=200,
    ).items())
    best: dict[str, object] = {}
    for it in items:
        tile = it.id.split("_")[1]                        # e.g. 18TWL
        cur = best.get(tile)
        if cur is None or it.properties["eo:cloud_cover"] < cur.properties["eo:cloud_cover"]:
            best[tile] = it
    return best


def _read_chip(href: str, lon: float, lat: float) -> np.ndarray | None:
    """A CHIP_PX square centred on (lon, lat) covering CHIP_M metres."""
    with rasterio.open(href) as src:
        x, y = transform_bounds("EPSG:4326", src.crs, lon, lat, lon, lat)[:2]
        half = CHIP_M / 2
        win = from_bounds(x - half, y - half, x + half, y + half, src.transform)
        # A zone on the edge of a scene would come back part-empty; skip it and
        # let another scene covering the same ground supply it.
        if win.col_off < 0 or win.row_off < 0:
            return None
        if (win.col_off + win.width > src.width
                or win.row_off + win.height > src.height):
            return None
        chip = src.read(window=win, out_shape=(3, CHIP_PX, CHIP_PX),
                        boundless=False)
    if chip.shape != (3, CHIP_PX, CHIP_PX):
        return None
    # An all-black chip means nodata, not a dark city.
    if chip.std() < 2:
        return None
    return chip


def build(city: str, zones, month: str = "2022-06", verbose: bool = True):
    """Fetch (and cache) one chip per zone. Returns {zone.key: uint8 CHW}."""
    CACHE.mkdir(parents=True, exist_ok=True)
    npz = CACHE / f"{city}_{month}.npz"
    if npz.exists():
        with np.load(npz) as z:
            return {k: z[k] for k in z.files}

    scenes = _scenes(city, month)
    if verbose:
        print(f"{city}: {len(scenes)} escenas "
              + ", ".join(f"{t} ({s.properties['eo:cloud_cover']:.1f}% nube)"
                          for t, s in scenes.items()))

    out: dict[str, np.ndarray] = {}
    for i, z in enumerate(zones, 1):
        for item in scenes.values():
            chip = _read_chip(item.assets["visual"].href, z.lon, z.lat)
            if chip is not None:
                out[z.key] = chip
                break
        if verbose and i % 25 == 0:
            print(f"  {i}/{len(zones)} zonas -> {len(out)} chips")
    np.savez_compressed(npz, **out)
    if verbose:
        print(f"{city}: {len(out)}/{len(zones)} chips -> {npz.name}")
    return out


if __name__ == "__main__":
    from . import zones as Z
    for city in ("nyc", "chicago"):
        zs = Z.load(city)
        chips = build(city, zs)
        a = np.stack(list(chips.values()))
        print(f"  {city}: {a.shape}  media {a.mean():.1f}  std {a.std():.1f}")
