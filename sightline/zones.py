"""The geography both cities are measured on.

Two cities that publish ride data publish it on their own administrative units:
New York on 263 taxi zones, Chicago on 77 community areas. Those units are not
comparable — a Manhattan taxi zone is a few blocks, a Chicago community area is
a neighbourhood — so nothing downstream is allowed to compare raw trip counts.
Everything is normalised to **trips per square kilometre per day**, which is a
property of the ground rather than of whoever drew the boundary.

Geometries are fetched once and cached. Both sources are open and need no key.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import requests

CACHE = Path(__file__).resolve().parents[1] / "data" / "zones"

# NYC publishes taxi zones as a shapefile; Chicago publishes community areas as
# GeoJSON straight off its Socrata portal.
NYC_ZONES = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
# The commonly-cited geospatial export id for community areas now returns an
# empty FeatureCollection; the Socrata resource endpoint is the live one.
CHI_AREAS = "https://data.cityofchicago.org/resource/igwz-8jzy.geojson?$limit=100"

# Metres-based CRS per city, so areas and distances are in real units rather
# than degrees. Using the local state-plane/UTM zone keeps distortion tiny.
UTM = {"nyc": "EPSG:32618", "chicago": "EPSG:32616"}


@dataclass(frozen=True)
class Zone:
    """One administrative unit, with everything the pipeline needs about it."""
    city: str
    zone_id: str
    name: str
    lon: float          # centroid, WGS84
    lat: float
    area_km2: float

    @property
    def key(self) -> str:
        return f"{self.city}-{self.zone_id}"


def _cached(name: str, url: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if path.exists() and path.stat().st_size > 0:
        return path
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def _load_nyc() -> gpd.GeoDataFrame:
    path = _cached("nyc_taxi_zones.zip", NYC_ZONES)
    with zipfile.ZipFile(path) as z:
        shp = next(n for n in z.namelist() if n.endswith(".shp"))
        gdf = gpd.read_file(f"zip://{path}!{shp}")
    gdf = gdf.rename(columns={"LocationID": "zone_id", "zone": "name"})
    return gdf[["zone_id", "name", "geometry"]]


def _load_chicago() -> gpd.GeoDataFrame:
    path = _cached("chicago_community_areas.geojson", CHI_AREAS)
    gdf = gpd.read_file(path)
    gdf = gdf.rename(columns={"area_numbe": "zone_id", "community": "name"})
    if "zone_id" not in gdf.columns:                 # portal renames this field
        gdf = gdf.rename(columns={"area_num_1": "zone_id"})
    return gdf[["zone_id", "name", "geometry"]]


def load(city: str) -> list[Zone]:
    """Zones for a city, with centroid in WGS84 and area in real km²."""
    gdf = {"nyc": _load_nyc, "chicago": _load_chicago}[city]()
    # NYC ships its shapefile in state-plane feet, Chicago its GeoJSON in WGS84.
    # Reproject rather than relabel: forcing a CRS on data that already has one
    # silently moves every polygon a few hundred kilometres.
    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")

    # Area has to be computed in a metric CRS; centroids too, or they drift.
    metric = gdf.to_crs(UTM[city])
    areas = metric.area / 1e6
    cents = metric.geometry.centroid.to_crs("EPSG:4326")

    out = []
    for (_, row), area, c in zip(gdf.iterrows(), areas, cents):
        if not area or area <= 0 or c.is_empty:
            continue
        out.append(Zone(
            city=city,
            zone_id=str(row["zone_id"]),
            name=str(row["name"]),
            lon=float(c.x), lat=float(c.y),
            area_km2=float(area),
        ))
    return out


if __name__ == "__main__":
    for city in ("nyc", "chicago"):
        zs = load(city)
        a = sorted(z.area_km2 for z in zs)
        print(f"{city:<8} {len(zs):>4} zonas   "
              f"área km²: min {a[0]:.2f}  mediana {a[len(a)//2]:.2f}  max {a[-1]:.2f}")
        print(f"         ejemplo: {zs[0].name} ({zs[0].lat:.4f}, {zs[0].lon:.4f})")
