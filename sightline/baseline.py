"""The baseline that has to be beaten: what OpenStreetMap already knows.

The obvious objection to predicting demand from satellite imagery is that the
imagery is a roundabout way of measuring urban density, and OSM hands you that
directly and for free. That objection deserves a number rather than a rebuttal,
so this builds an OSM feature set and the same evaluation runs on both.

The features are road ones: metres of street, junctions, and how much of that
street is arterial rather than residential, measured inside the same 1.28 km
square the satellite chip covers. Road network is the canonical OSM stand-in for
urban density, it is the part of OSM that is complete earliest in any city, and
it is the fairest thing to put against imagery.

HOW IT IS FETCHED. Per-zone Overpass queries were measured at 22-30 seconds each
— nearly three hours for the two cities, and unkind to a free service. Instead
each city is fetched once as a grid of nine tiles, and every zone's features are
computed locally from that. Nine requests instead of three hundred and thirty
two, and the same network then draws the street map in scripts/make_figures.py.

WHY THE LENGTHS ARE CLIPPED. Overpass returns the whole geometry of any way that
so much as touches the bounding box, so summing what comes back counts arterials
that merely pass nearby, most of their length kilometres away. Measured that way
a residential zone in Staten Island came out at 60 km of road per km², roughly
three times the densest real figure anywhere. Only the segments whose midpoint
falls inside the square are counted.

Where this baseline is expected to win: mature cities with complete mapping.
Where it is expected to lose: everywhere the map is thin — which is most of the
world, and the whole reason to ask the satellite instead.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import requests

CACHE = Path(__file__).resolve().parents[1] / "data" / "osm"
OVERPASS = "https://overpass-api.de/api/interpreter"
# Overpass answers a request with no User-Agent with 406 Not Acceptable, which
# reads like a malformed query and is not: it is the server declining to serve
# anonymous traffic. Identify the client and the same query returns 200.
HEADERS = {"User-Agent": "sightline/0.1 (+https://github.com/mathieutellene/sightline)"}
CHIP_M = 1280.0
GRID = 3                    # 3x3 tiles per city; one request for a whole city times out

# Streets, not every line OSM calls a highway. Footpaths, cycleways, service
# drives and alleys are excluded: they would triple the download, and Chicago's
# alley grid alone would swamp the density signal the feature is meant to carry.
ROAD_CLASSES = ("motorway|trunk|primary|secondary|tertiary"
                "|residential|unclassified|living_street")
ARTERIAL = {"motorway", "trunk", "primary", "secondary"}
FEATURES = ["road_m", "arterial_m", "junctions", "segments"]


def _bbox(lon: float, lat: float):
    """The same 1.28 km square the satellite chip covers."""
    dlat = (CHIP_M / 2) / 111_320.0
    dlon = dlat / max(0.2, abs(math.cos(math.radians(lat))))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def _metres(a, b):
    dy = (b[1] - a[1]) * 111_320
    dx = (b[0] - a[0]) * 111_320 * math.cos(math.radians(a[1]))
    return math.hypot(dx, dy)


def _fetch_tile(s, w, n, e) -> list[dict]:
    q = (f'[out:json][timeout:180];way["highway"~"^({ROAD_CLASSES})$"]'
         f"({s},{w},{n},{e});out geom;")
    for attempt in range(4):
        try:
            r = requests.post(OVERPASS, data={"data": q}, headers=HEADERS, timeout=300)
            r.raise_for_status()
            return r.json().get("elements", [])
        except Exception:
            if attempt == 3:
                raise
            time.sleep(15 * (attempt + 1))       # slots free up, they do not vanish
    return []


def roads(city: str, bbox, verbose: bool = True) -> list[dict]:
    """Every street in the city, as {cls, pts:[[lon,lat],...]}. Cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{city}_roads.json"
    if path.exists():
        return json.loads(path.read_text())

    w0, s0, e0, n0 = bbox
    out, seen = [], set()
    for i in range(GRID):
        for j in range(GRID):
            s = s0 + (n0 - s0) * j / GRID
            n = s0 + (n0 - s0) * (j + 1) / GRID
            w = w0 + (e0 - w0) * i / GRID
            e = w0 + (e0 - w0) * (i + 1) / GRID
            els = _fetch_tile(s, w, n, e)
            for el in els:
                # A way crossing a tile edge is returned by both tiles.
                if el["id"] in seen:
                    continue
                seen.add(el["id"])
                g = el.get("geometry") or []
                if len(g) < 2:
                    continue
                out.append({"cls": el.get("tags", {}).get("highway", ""),
                            "pts": [[round(p["lon"], 5), round(p["lat"], 5)] for p in g]})
            if verbose:
                print(f"  {city} celda {i * GRID + j + 1}/{GRID * GRID}: "
                      f"{len(els):,} vías -> {len(out):,} únicas", flush=True)
            time.sleep(1.0)
    path.write_text(json.dumps(out))
    if verbose:
        km = sum(_metres(a, b) for r in out for a, b in zip(r["pts"], r["pts"][1:])) / 1000
        print(f"{city}: {len(out):,} vías, {km:,.0f} km de calle -> {path.name}")
    return out


def features(city: str, zone_list, bbox, verbose: bool = True) -> dict[str, dict]:
    """Per-zone road features, measured inside each zone's chip square."""
    net = roads(city, bbox, verbose)

    # One pass over the network per city, bucketed onto a coarse grid, so each
    # zone only looks at segments that could possibly be near it.
    cell = 0.02                                   # about 2 km
    buckets: dict[tuple, list] = {}
    for r in net:
        arterial = r["cls"] in ARTERIAL
        for a, b in zip(r["pts"], r["pts"][1:]):
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            buckets.setdefault((int(mx / cell), int(my / cell)), []).append(
                (mx, my, _metres(a, b), arterial, tuple(a), tuple(b)))

    out: dict[str, dict] = {}
    for z in zone_list:
        s, w, n, e = _bbox(z.lon, z.lat)
        f = dict.fromkeys(FEATURES, 0.0)
        ends: dict[tuple, int] = {}
        for gx in range(int(w / cell) - 1, int(e / cell) + 2):
            for gy in range(int(s / cell) - 1, int(n / cell) + 2):
                for mx, my, m, arterial, a, b in buckets.get((gx, gy), ()):
                    if not (w <= mx <= e and s <= my <= n):
                        continue
                    f["road_m"] += m
                    f["segments"] += 1
                    if arterial:
                        f["arterial_m"] += m
                    for p in (a, b):
                        ends[p] = ends.get(p, 0) + 1
        f["junctions"] = float(sum(1 for v in ends.values() if v >= 3))
        out[z.key] = f
    if verbose:
        km2 = (CHIP_M / 1000) ** 2
        d = sorted(v["road_m"] / 1000 / km2 for v in out.values())
        print(f"{city}: {len(out)} zonas | densidad de calle km/km²  "
              f"mín {d[0]:.1f}  mediana {d[len(d) // 2]:.1f}  máx {d[-1]:.1f}")
    return out
