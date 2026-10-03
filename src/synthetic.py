"""Synthetic Sabah with known undetected cases.

Real undetected cases are unknown by definition, so we build worlds where we
know them, using the real 25 districts' populations and covariates:

  true incidence:  log rate = base + b . z(risk covariates) + trend
  detection:       p = logistic, ~0.9 next to a specialist hospital,
                   ~0.5 in the most remote district
  notified:        Poisson(true * p)

Scenario "detection"   remote districts are under-detected (our hypothesis).
Scenario "transmission" remote districts truly have less TB and detection is
                        uniform: the confounded world the real data can't rule
                        out. A good method should NOT find big gaps here.

Also used as the PLAN 4.5 fallback: `python src/synthetic.py --write` writes a
fully synthetic district_year.csv with the same schema (clearly flagged).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"

BASE_RATE = 160 / 1e5           # true incidence per person-year at average covariates
B = {"poverty_rate": 0.20, "noncitizen_share": 0.10, "log_pop_density": 0.10}
TREND = 0.01


def _z(s):
    return (s - s.mean()) / s.std(ddof=0)


def detection_prob(dist_km, p_near=0.9, p_far=0.5):
    """Logistic in distance, scaled so p(0) ~ p_near and p(max) ~ p_far."""
    x = dist_km / dist_km.max()
    logit = lambda p: np.log(p / (1 - p))
    k0, k1 = logit(p_near), logit(p_far)
    return 1 / (1 + np.exp(-(k0 + (k1 - k0) * x ** 1.5)))


def generate(template, scenario="detection", seed=0, remote_effect=-0.35):
    """template: real district_year frame (covariates + population are reused)."""
    rng = np.random.default_rng(seed)
    d = template.copy()
    first = d.groupby("district").first()
    lin = sum(b * _z(first[c]) for c, b in B.items())
    # Small unexplained district effects so the model isn't handed a perfect world.
    lin = lin + rng.normal(0, 0.10, len(lin))
    dist = first.dist_to_hospital_km
    if scenario == "detection":
        p = detection_prob(dist)
    elif scenario == "transmission":
        lin = lin + remote_effect * _z(dist)
        p = pd.Series(0.8, index=first.index)
    else:
        raise ValueError(scenario)
    d["true_rate"] = BASE_RATE * np.exp(d.district.map(lin) + TREND * (d.year - 2012))
    d["true_cases"] = rng.poisson(d.true_rate * d.population)
    d["p_detect"] = d.district.map(p)
    d["cases_notified"] = rng.binomial(d.true_cases, d.p_detect)
    d["notif_rate"] = d.cases_notified / d.population * 1e5
    d["true_gap"] = d.true_cases - d.cases_notified
    d["data_source"] = "synthetic"
    return d


def true_gap_table(syn, recent=3):
    last = syn.year.max()
    r = syn[syn.year > last - recent]
    return r.groupby("district").true_gap.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="overwrite district_year.csv with a synthetic dataset (fallback mode)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    real = pd.read_csv(PROC / "district_year.csv")
    syn = generate(real, seed=args.seed)
    out = PROC / ("district_year.csv" if args.write else "synthetic_district_year.csv")
    syn.to_csv(out, index=False)
    print(f"wrote {len(syn)} synthetic rows -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
