"""Digitise the district choropleths in Goroh et al. 2020 (Fig. 2 and Fig. 5).

The paper publishes district-level TB notification rates (Fig. 2, one map per
year) and non-citizen share of cases (Fig. 5) only as binned choropleth maps.
Rather than eyeballing 175 cells, we:

  1. rasterise our own Sabah district boundaries onto each map panel, fitting
     scale + offset by maximising overlap with the coloured pixels;
  2. for each district, take the pixels inside its (eroded) polygon and assign
     the legend bin whose swatch colour is nearest (majority vote);
  3. convert the bin to a numeric value (bin midpoint) and override with the
     exact values quoted in the paper text where they exist.

Outputs are *approximate* (bin resolution 40 per 100k / 10 percentage points)
and are labelled as such everywhere downstream.

raw/goroh_Fig2.jpg, raw/goroh_Fig5.jpg, processed/districts.geojson
  -> raw/digitised_notif_rate.csv, raw/digitised_noncitizen.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"

# Fig. 2 legend (case notification rate per 100k). Colours sampled from the
# legend swatches. Midpoint for the bottom bin uses the lowest rate the paper
# reports anywhere (53, Tongod 2018) as the floor rather than 0.
FIG2_BINS = [  # (rgb, label, value)
    ((15, 253, 33), "0-79", 66.0),
    ((254, 251, 2), "80-119", 100.0),
    ((248, 142, 9), "120-159", 140.0),
    ((252, 5, 1), "160-199", 180.0),
    ((167, 19, 15), "200-229", 215.0),
]
# Panel boxes (x0, y0, x1, y1) found from column/row projections of
# saturated pixels; 2012-2015 top row, 2016-2018 bottom row.
FIG2_PANELS = {
    2012: (5, 13, 161, 146), 2013: (207, 13, 363, 146),
    2014: (408, 13, 565, 146), 2015: (612, 13, 767, 146),
    2016: (5, 192, 161, 323), 2017: (207, 192, 363, 323),
    2018: (408, 192, 565, 323),
}
# Exact 2018 values quoted in the Results text.
FIG2_TEXT_OVERRIDES = {
    ("Semporna", 2018): 228.0, ("Pitas", 2018): 228.0,
    ("Kinabatangan", 2018): 56.0, ("Tongod", 2018): 53.0,
}

# Fig. 5 legend (% of cases who are non-citizens), blue ramp.
FIG5_BIN_EDGES = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 100)]
# Exact / bounded values quoted in the Results text.
FIG5_TEXT_OVERRIDES = {
    "Kinabatangan": 62.0,          # "62% of cases in Kinabatangan"
    "Nabawan": 2.5, "Kota Belud": 2.5, "Pitas": 2.5,   # "< 5%"
}


def load_districts():
    gj = json.load(open(PROC / "districts.geojson"))
    return {f["id"]: shape(f["geometry"]) for f in gj["features"]}


def polys(geom):
    return list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]


def rasterise(geoms, size, tf, erode=0):
    """Draw geometries into a boolean mask using tf=(sx, sy, ox, oy)."""
    sx, sy, ox, oy = tf
    img = Image.new("L", size, 0)
    d = ImageDraw.Draw(img)
    for g in geoms:
        for p in polys(g):
            pts = [(ox + sx * x, oy - sy * y) for x, y in p.exterior.coords]
            if len(pts) > 2:
                d.polygon(pts, fill=255)
    if erode:
        img = img.filter(ImageFilter.MinFilter(2 * erode + 1))
    return np.asarray(img) > 0


def fit_transform(target, geoms, box):
    """Grid-search scale/offset so rasterised districts best overlap `target`."""
    x0, y0, x1, y1 = box
    minx = min(g.bounds[0] for g in geoms); maxx = max(g.bounds[2] for g in geoms)
    miny = min(g.bounds[1] for g in geoms); maxy = max(g.bounds[3] for g in geoms)
    size = (target.shape[1], target.shape[0])

    def iou(tf):
        m = rasterise(geoms, size, tf)
        return (m & target).sum() / max((m | target).sum(), 1)

    # Initial guess from bounding boxes, then coordinate-descent refinement.
    sx = (x1 - x0) / (maxx - minx); sy = (y1 - y0) / (maxy - miny)
    tf = [sx, sy, x0 - sx * minx, y0 + sy * maxy]
    best = iou(tf)
    steps = [0.03 * sx, 0.03 * sy, 2.0, 2.0]
    for _ in range(6):
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for sgn in (-1, 1):
                    cand = list(tf)
                    cand[i] += sgn * steps[i]
                    if i == 0:  # keep the panel anchored when rescaling
                        cand[2] -= sgn * steps[i] * (minx + maxx) / 2
                    if i == 1:
                        cand[3] += sgn * steps[i] * (miny + maxy) / 2
                    s = iou(cand)
                    if s > best:
                        tf, best, improved = cand, s, True
        steps = [s / 2 for s in steps]
    return tf, best


def classify(pixels, palette):
    pal = np.array(palette, dtype=float)
    d = ((pixels[:, None, :].astype(float) - pal[None, :, :]) ** 2).sum(-1)
    return d.argmin(1), np.sqrt(d.min(1))


def read_districts(im, mask_colour, districts, box, palette):
    x0, y0, x1, y1 = box
    sub = np.zeros(mask_colour.shape, bool)
    sub[y0:y1, x0:x1] = mask_colour[y0:y1, x0:x1]
    geoms = list(districts.values())
    tf, score = fit_transform(sub, geoms, box)
    size = (im.shape[1], im.shape[0])
    out = {}
    for name, g in districts.items():
        m = None
        for erode in (1, 0):  # erode to dodge borders; fall back for tiny districts
            m = rasterise([g], size, tf, erode=erode) & sub
            if m.sum() >= 3:
                break
        px = im[m]
        if len(px) == 0:
            out[name] = (None, 0.0, 0)
            continue
        idx, dist = classify(px, palette)
        ok = dist < 90  # ignore border/label pixels far from every swatch
        idx = idx[ok] if ok.any() else idx
        counts = np.bincount(idx, minlength=len(palette))
        out[name] = (int(counts.argmax()), counts.max() / counts.sum(), int(m.sum()))
    return out, score


def digitise_fig2(districts):
    im = np.asarray(Image.open(RAW / "goroh_Fig2.jpg").convert("RGB"))
    colourful = (im.max(2).astype(int) - im.min(2)) > 60
    palette = [b[0] for b in FIG2_BINS]
    rows = []
    for year, box in FIG2_PANELS.items():
        res, score = read_districts(im, colourful, districts, box, palette)
        print(f"  Fig2 {year}: panel fit IoU={score:.2f}")
        for name, (k, conf, npx) in res.items():
            label, value = FIG2_BINS[k][1], FIG2_BINS[k][2]
            src = "fig2_bin_midpoint"
            if (name, year) in FIG2_TEXT_OVERRIDES:
                value, src = FIG2_TEXT_OVERRIDES[(name, year)], "paper_text_exact"
            rows.append(dict(district=name, year=year, notif_rate=value,
                             rate_bin=label, source=src,
                             vote_share=round(conf, 2), n_pixels=npx))
    return pd.DataFrame(rows)


# Fig. 5 legend swatches sampled from the image (two legend rows).
FIG5_PALETTE = [
    (247, 250, 255), (215, 231, 246), (177, 209, 233), (115, 179, 215),
    (62, 142, 195), (21, 99, 171), (9, 49, 108),
]


def digitise_fig5(districts):
    im = np.asarray(Image.open(RAW / "goroh_Fig5.jpg").convert("RGB"))
    box = (0, 0, im.shape[1], 375)  # map area, above the legend
    # The lowest bin is near-white, so fit against the filled-in outline
    # (outline pixels + everything they enclose) rather than coloured pixels.
    drawn = im.min(2) < 230
    drawn[375:] = False
    filled = ndimage.binary_fill_holes(ndimage.binary_closing(drawn, iterations=1))
    res, score = read_districts(im, filled, districts, box, FIG5_PALETTE)
    print(f"  Fig5: panel fit IoU={score:.2f}")
    rows = []
    for name, (k, conf, npx) in res.items():
        lo, hi = FIG5_BIN_EDGES[k]
        val, src = (lo + hi) / 2, "fig5_bin_midpoint"
        if name in FIG5_TEXT_OVERRIDES:
            val, src = FIG5_TEXT_OVERRIDES[name], "paper_text"
        rows.append(dict(district=name, noncitizen_pct=val, pct_bin=f"{lo}-{hi}",
                         source=src, vote_share=round(conf, 2), n_pixels=npx))
    return pd.DataFrame(rows)


def main():
    districts = load_districts()
    f2 = digitise_fig2(districts)
    f2.to_csv(RAW / "digitised_notif_rate.csv", index=False)
    f5 = digitise_fig5(districts)
    f5.to_csv(RAW / "digitised_noncitizen.csv", index=False)
    print(f"wrote {len(f2)} notification cells, {len(f5)} non-citizen cells")


if __name__ == "__main__":
    main()
