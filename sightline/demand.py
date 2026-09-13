"""The target: ride-hailing pickups per zone, from each city's own open data.

Both cities publish every trip. Neither publishes them the same way:

  New York   one parquet per month, 17.8 M rows for June 2022 alone, with the
             pickup zone as a column. Only that column is ever read -- parquet
             is columnar, so a range request pulls 18 MB out of a 437 MB file.
  Chicago    a Socrata endpoint that will do the GROUP BY server-side, so the
             aggregate arrives as a few kilobytes of JSON.

STUDY PERIOD is fixed at June 2022 for both, because that is the most recent
month Chicago publishes. Predicting 2022 demand from 2026 imagery would be
sloppy in a way that is easy to miss and hard to defend.

WHAT IS MODELLED is trips per km² per day, not trips. A Manhattan taxi zone is
a few blocks and a Chicago community area is a neighbourhood; raw counts would
mostly measure how big the polygon is.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
import requests

CACHE = Path(__file__).resolve().parents[1] / "data" / "demand"
STUDY_MONTH = "2022-06"
DAYS_IN_MONTH = 30

NYC_PARQUET = ("https://d37ci6vzurychx.cloudfront.net/trip-data/"
               "fhvhv_tripdata_{month}.parquet")
CHI_RESOURCE = "https://data.cityofchicago.org/resource/2tdj-ffvb.json"


def _cache(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / name


def nyc_pickups(month: str = STUDY_MONTH) -> dict[str, int]:
    """Pickups per taxi zone. High-volume for-hire only: Uber and Lyft.

    Deliberately NOT the yellow cab feed. Yellow taxis are concentrated in
    Manhattan and the airports, so a model trained on them would learn
    "is this Manhattan" and fall over anywhere else. The high-volume for-hire
    feed covers all five boroughs and is the direct analogue of what Chicago
    publishes, which is the only reason the two cities can be compared at all.
    """
    path = _cache(f"nyc_{month}.json")
    if path.exists():
        return json.loads(path.read_text())

    url = NYC_PARQUET.format(month=month)
    with fsspec.filesystem("https").open(url) as f:
        table = pq.ParquetFile(f).read(columns=["PULocationID"])
    counts = Counter(table.column("PULocationID").to_pylist())
    out = {str(k): int(v) for k, v in counts.items() if k is not None}
    path.write_text(json.dumps(out))
    return out


def chicago_pickups(month: str = STUDY_MONTH) -> dict[str, int]:
    """Pickups per community area, aggregated server-side by Socrata."""
    path = _cache(f"chicago_{month}.json")
    if path.exists():
        return json.loads(path.read_text())

    y, m = month.split("-")
    nxt = f"{int(y) + 1}-01-01" if m == "12" else f"{y}-{int(m) + 1:02d}-01"
    r = requests.get(CHI_RESOURCE, params={
        "$select": "pickup_community_area, count(trip_id) as n",
        "$where": (f"trip_start_timestamp >= '{y}-{m}-01' "
                   f"AND trip_start_timestamp < '{nxt}' "
                   f"AND pickup_community_area IS NOT NULL"),
        "$group": "pickup_community_area",
        "$limit": 500,
    }, timeout=300)
    r.raise_for_status()
    out = {str(int(float(row["pickup_community_area"]))): int(row["n"])
           for row in r.json()}
    path.write_text(json.dumps(out))
    return out


def load(city: str, month: str = STUDY_MONTH) -> dict[str, int]:
    return {"nyc": nyc_pickups, "chicago": chicago_pickups}[city](month)


def density(city: str, zones, month: str = STUDY_MONTH) -> dict[str, float]:
    """Trips per km² per day — the quantity the model actually predicts."""
    counts = load(city, month)
    out = {}
    for z in zones:
        n = counts.get(z.zone_id)
        if n is None or z.area_km2 <= 0:
            continue
        out[z.key] = n / z.area_km2 / DAYS_IN_MONTH
    return out


if __name__ == "__main__":
    from . import zones as Z
    for city in ("nyc", "chicago"):
        zs = Z.load(city)
        d = density(city, zs)
        vals = sorted(d.values())
        by = {z.key: z for z in zs}
        top = sorted(d.items(), key=lambda kv: -kv[1])[:3]
        print(f"{city:<8} {len(d):>4} zonas con demanda   "
              f"viajes/km²/día: min {vals[0]:.1f}  mediana {vals[len(vals)//2]:.0f}  "
              f"max {vals[-1]:,.0f}")
        for k, v in top:
            print(f"          {by[k].name[:34]:<36} {v:>9,.0f}")
