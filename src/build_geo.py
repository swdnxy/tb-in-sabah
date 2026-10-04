"""Filter geoBoundaries MYS ADM2 to Sabah and dissolve to the paper's 25 districts.

raw/mys_adm2_simplified.geojson + raw/district_names.csv -> processed/districts.geojson
"""
import json
from pathlib import Path

import pandas as pd
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"


def main():
    names = pd.read_csv(RAW / "district_names.csv")
    geo_to_canon = dict(zip(names.geo_name, names.canonical_name))
    src = json.load(open(RAW / "mys_adm2_simplified.geojson"))

    parts = {}
    for f in src["features"]:
        canon = geo_to_canon.get(f["properties"]["shapeName"])
        if canon:
            parts.setdefault(canon, []).append(shape(f["geometry"]).buffer(0))

    expected = set(names.canonical_name)
    missing = expected - set(parts)
    assert not missing, f"districts missing from boundary file: {missing}"

    feats = []
    for canon, geoms in sorted(parts.items()):
        g = unary_union(geoms)
        pt = g.representative_point()
        feats.append({
            "type": "Feature",
            "id": canon,
            "properties": {"district": canon, "rep_lon": pt.x, "rep_lat": pt.y},
            "geometry": mapping(g),
        })
    PROC.mkdir(parents=True, exist_ok=True)
    json.dump({"type": "FeatureCollection", "features": feats},
              open(PROC / "districts.geojson", "w"))
    print(f"wrote {len(feats)} districts -> data/processed/districts.geojson")


if __name__ == "__main__":
    main()
