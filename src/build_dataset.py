"""Assemble the district x year analysis table.

data/raw/* -> data/processed/district_year.csv

Sources (see README "Data sources"):
  notif_rate        Goroh et al. 2020 Fig. 2, digitised (src/digitise.py), approximate
  noncitizen_share  Goroh et al. 2020 Fig. 5, digitised, share of TB *cases* 2012-2018
  population        DOSM Census 2010 & 2020, linearly interpolated
  poverty_rate      DOSM HIES 2022 district absolute poverty (%), earliest district release
  log_pop_density   census population / boundary area (rurality proxy; DOSM does not
                    publish district urban/rural share as open data)
  dist_to_hospital_km  haversine from district's main town to the nearest of 5
                    specialist hospitals (hand-coded proxy for access)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"
YEARS = range(2012, 2019)
# Paper text: 33 193 cases notified in Sabah 2012-2018.
PAPER_TOTAL_CASES = 33193
PAPER_STATE_RATE = 128  # per 100 000 per year, 2012-2018
# Exact counts quoted in the Results text.
TEXT_CASE_OVERRIDES = {("Kota Kinabalu", 2018): 904}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def population(names):
    census = pd.read_csv(RAW / "census_population_sabah.csv")
    census["canonical_name"] = census.geo_name.map(dict(zip(names.geo_name, names.canonical_name)))
    census = census.groupby("canonical_name")[["pop_census_2010", "pop_census_2020"]].sum()
    rows = []
    for d, r in census.iterrows():
        for y in YEARS:
            # Census reference dates are mid-year 2010 and 2020.
            pop = r.pop_census_2010 + (r.pop_census_2020 - r.pop_census_2010) * (y - 2010) / 10
            rows.append(dict(district=d, year=y, population=round(pop)))
    return pd.DataFrame(rows), census


def poverty(names, census):
    hies = pd.read_csv(RAW / "hies_district.csv")
    hies = hies[(hies.state == "Sabah") & (hies.date == "2022-01-01")]
    m = hies.merge(names, left_on="district", right_on="paper_name", how="left")
    # HIES uses short names ("Nabawan"); boundary/census use "Nabawan / Persiangan".
    m["canonical_name"] = m.canonical_name.fillna(
        m.district.map(lambda n: {"Telupid": "Beluran", "Kalabakan": "Tawau"}.get(n, n)))
    cpop = pd.read_csv(RAW / "census_population_sabah.csv").set_index("geo_name").pop_census_2020
    cpop.index = cpop.index.str.replace(" / Persiangan", "")
    m["w"] = m.district.map(cpop)
    out = m.groupby("canonical_name").apply(
        lambda g: np.average(g.poverty, weights=g.w), include_groups=False)
    return out.rename("poverty_rate").div(100)


def geography():
    gj = json.load(open(PROC / "districts.geojson"))
    towns = pd.read_csv(RAW / "towns.csv").set_index("canonical_name")
    hosp = pd.read_csv(RAW / "hospitals.csv")
    rows = []
    for f in gj["features"]:
        d = f["id"]
        g = shape(f["geometry"])
        # area in km2 (equirectangular approximation, fine at Sabah's latitude)
        lat0 = np.radians(g.centroid.y)
        area_km2 = g.area * (111.32 ** 2) * np.cos(lat0)
        t = towns.loc[d]
        dists = haversine_km(t.lat, t.lon, hosp.lat.values, hosp.lon.values)
        # Robustness variant: mean distance from a grid of points across the district.
        minx, miny, maxx, maxy = g.bounds
        xs, ys = np.meshgrid(np.linspace(minx, maxx, 40), np.linspace(miny, maxy, 40))
        pts = [(x, y) for x, y in zip(xs.ravel(), ys.ravel()) if g.contains(Point(x, y))]
        if pts:
            p = np.array(pts)
            area_d = np.min([haversine_km(p[:, 1], p[:, 0], h.lat, h.lon)
                             for h in hosp.itertuples()], axis=0).mean()
        else:
            area_d = dists.min()
        rows.append(dict(district=d, area_km2=area_km2,
                         dist_to_hospital_km=round(float(dists.min()), 1),
                         nearest_hospital=hosp.hospital[int(dists.argmin())],
                         dist_area_mean_km=round(float(area_d), 1)))
    return pd.DataFrame(rows).set_index("district")


def main():
    names = pd.read_csv(RAW / "district_names.csv")
    pop, census = population(names)
    rate = pd.read_csv(RAW / "digitised_notif_rate.csv")
    nc = pd.read_csv(RAW / "digitised_noncitizen.csv").set_index("district")
    geo = geography()
    pov = poverty(names, census)

    df = pop.merge(rate[["district", "year", "notif_rate", "rate_bin", "source", "vote_share", "n_pixels"]],
                   on=["district", "year"], how="left", validate="1:1")
    assert df.notif_rate.notna().all(), "unmatched district-year in digitised rates"

    # Calibrate denominators: census-interpolated populations are smaller than the
    # DOSM intercensal estimates the paper used, so cases implied by the digitised
    # rates undershoot the paper's statewide total. One statewide factor fixes the
    # level without changing any district's rate or ranking.
    raw_cases = (df.notif_rate * df.population / 1e5).sum()
    calib = PAPER_TOTAL_CASES / raw_cases
    df["population"] = (df.population * calib).round().astype(int)
    df["cases_notified"] = (df.notif_rate * df.population / 1e5).round().astype(int)
    for (d, y), n in TEXT_CASE_OVERRIDES.items():
        m = (df.district == d) & (df.year == y)
        df.loc[m, "cases_notified"] = n
        df.loc[m, "notif_rate"] = round(n / df.loc[m, "population"].iloc[0] * 1e5, 1)
        df.loc[m, "source"] = "paper_text_exact"
    df["noncitizen_share"] = df.district.map(nc.noncitizen_pct) / 100
    df["poverty_rate"] = df.district.map(pov)
    df["log_pop_density"] = np.log(df.population / df.district.map(geo.area_km2))
    for c in ["dist_to_hospital_km", "dist_area_mean_km", "nearest_hospital"]:
        df[c] = df.district.map(geo[c])
    # Low-confidence digitisation: tiny on the figure or a split colour vote.
    df["digitise_low_conf"] = (df.n_pixels < 20) | (df.vote_share < 0.6)
    df["data_source"] = "digitised"

    raw_rate = raw_cases / (df.population / calib).sum() * 1e5
    print(f"digitised statewide rate {raw_rate:.1f}/100k vs paper {PAPER_STATE_RATE} "
          f"({raw_rate / PAPER_STATE_RATE - 1:+.1%}); population calibration x{calib:.3f}")

    cols = ["district", "year", "population", "cases_notified", "notif_rate",
            "noncitizen_share", "poverty_rate", "log_pop_density",
            "dist_to_hospital_km", "dist_area_mean_km", "nearest_hospital",
            "rate_bin", "source", "digitise_low_conf", "data_source"]
    df = df[cols].sort_values(["district", "year"])
    assert df.isna().sum().sum() == 0, df.isna().sum()
    df.to_csv(PROC / "district_year.csv", index=False)
    json.dump({"digitised_state_rate": round(raw_rate, 1), "paper_state_rate": PAPER_STATE_RATE,
               "raw_implied_cases": int(round(raw_cases)), "paper_cases": PAPER_TOTAL_CASES,
               "population_calibration": round(calib, 4)},
              open(PROC / "data_check.json", "w"), indent=2)
    print(f"wrote {len(df)} rows -> data/processed/district_year.csv")


if __name__ == "__main__":
    main()
