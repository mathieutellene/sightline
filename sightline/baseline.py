"""The baseline that has to be beaten: what OpenStreetMap already knows.

The obvious objection to predicting demand from satellite imagery is that the
imagery is a roundabout way of measuring urban density, and OSM hands you that
directly and for free. That objection deserves a number rather than a rebuttal,
so this builds the OSM feature set and the same evaluation runs on both.

The features are the ones a planner would reach for: how much road, how many
buildings, how many places worth going to. Fetched per zone from Overpass with
an identical bounding box to the satellite chip, so the two views see exactly
the same ground.

Where this baseline is expected to win: mature cities with complete mapping.
Where it is expected to lose: everywhere the map is thin — which is most of the
world, and the whole reason to ask the satellite instead.
"""
from __future__ import annotations

import json, time
from pathlib import Path

import requests

CACHE = Path(__file__).resolve().parents[1] / "data" / "osm"
OVERPASS = "https://overpass-api.de/api/interpreter"
CHIP_M = 1280.0
FEATURES = ["road_m", "buildings", "building_m2", "amenities", "shops", "intersections"]


def _bbox(lon: float, lat: float):
    """The same 1.28 km square the satellite chip covers."""
    dlat = (CHIP_M / 2) / 111_320.0
    dlon = dlat / max(0.2, abs(__import__("math").cos(__import__("math").radians(lat))))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def _query(lon: float, lat: float) -> dict[str, float]:
    s, w, n, e = _bbox(lon, lat)
    q = f"""[out:json][timeout:90];
(
  way["highway"]({s},{w},{n},{e});
  way["building"]({s},{w},{n},{e});
  node["amenity"]({s},{w},{n},{e});
  node["shop"]({s},{w},{n},{e});
);
out geom;"""
    r = requests.post(OVERPASS, data={"data": q}, timeout=180)
    r.raise_for_status()
    els = r.json().get("elements", [])

    import math
    f = dict.fromkeys(FEATURES, 0.0)
    node_uses: dict[tuple, int] = {}
    for el in els:
        tags = el.get("tags", {})
        if el["type"] == "node":
            if "amenity" in tags: f["amenities"] += 1
            if "shop" in tags: f["shops"] += 1
            continue
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        if "highway" in tags:
            for a, b in zip(geom, geom[1:]):
                dy = (b["lat"] - a["lat"]) * 111_320
                dx = (b["lon"] - a["lon"]) * 111_320 * math.cos(math.radians(a["lat"]))
                f["road_m"] += math.hypot(dx, dy)
            for p in (geom[0], geom[-1]):
                k = (round(p["lat"], 5), round(p["lon"], 5))
                node_uses[k] = node_uses.get(k, 0) + 1
        elif "building" in tags:
            f["buildings"] += 1
            # shoelace in local metres
            latm = 111_320; lonm = latm * math.cos(math.radians(geom[0]["lat"]))
            a = 0.0
            for p, q2 in zip(geom, geom[1:] + geom[:1]):
                a += (p["lon"] * lonm) * (q2["lat"] * latm) - (q2["lon"] * lonm) * (p["lat"] * latm)
            f["building_m2"] += abs(a) / 2
    f["intersections"] = float(sum(1 for v in node_uses.values() if v >= 3))
    return f


def build(city: str, zone_list, verbose: bool = True) -> dict[str, dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{city}.json"
    have = json.loads(path.read_text()) if path.exists() else {}
    todo = [z for z in zone_list if z.key not in have]
    for i, z in enumerate(todo, 1):
        try:
            have[z.key] = _query(z.lon, z.lat)
        except Exception as exc:                       # Overpass rate-limits hard
            if verbose:
                print(f"  {z.key}: {type(exc).__name__} - reintento luego")
            time.sleep(5)
            continue
        if i % 10 == 0:
            path.write_text(json.dumps(have))
            if verbose:
                print(f"  {i}/{len(todo)} zonas")
        time.sleep(1.0)                                # be a good citizen
    path.write_text(json.dumps(have))
    return have
