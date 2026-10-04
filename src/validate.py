"""Validation: temporal backtest on real data + synthetic ground-truth recovery.

-> data/processed/validation.json, data/processed/backtest_detail.csv

1. Temporal backtest: train 2012-2016, predict 2017-2018 notified cases per
   district. Compare MAE against naive baselines (last value carried forward,
   3-year mean). Also report 90% prediction-interval coverage.
2. Synthetic recovery: in worlds where the undetected cases are known, run the
   same gap pipeline and score precision@5 (top-5 estimated vs top-5 true gap
   rate) and Spearman correlation between estimated and true gap rates.
   Run under both the "detection" and "transmission" scenarios, 200 seeds each.
3. Planner value in the synthetic world: cases actually found by the plan
   (scored against the TRUE gap) vs random allocation.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")  # statsmodels convergence chatter on synthetic refits
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M  # noqa: E402
import planner  # noqa: E402
import synthetic as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
N_SEEDS = 200
K = 5


def backtest(df, train_end=2016, test_years=(2017, 2018)):
    train = df[df.year <= train_end]
    test = df[df.year.isin(test_years)].set_index(["district", "year"])
    fc = M.forecast(train, list(test_years)).set_index(["district", "year"])
    # forecast uses projected population; rescale to actual population for a fair count comparison
    fc["forecast"] = fc.forecast / fc.population * test.population.reindex(fc.index)
    last = train[train.year == train_end].set_index("district").notif_rate
    mean3 = train[train.year > train_end - 3].groupby("district").notif_rate.mean()
    glm = M.fit_glm(train)
    rows = []
    for (d, y), r in test.iterrows():
        X = M._design(glm, M.district_table(train).loc[[d]], y)
        rows.append({
            "district": d, "year": y, "actual": r.cases_notified,
            "fe_trend": fc.loc[(d, y), "forecast"],
            "lo": fc.loc[(d, y), "forecast_lo"], "hi": fc.loc[(d, y), "forecast_hi"],
            "naive_last": last[d] * r.population / 1e5,
            "naive_mean3": mean3[d] * r.population / 1e5,
            "covariate_glm": float(np.exp(X.values @ glm["res"].params.values)[0] * r.population),
        })
    b = pd.DataFrame(rows)
    mae = {m: float((b[m] - b.actual).abs().mean())
           for m in ["fe_trend", "naive_last", "naive_mean3", "covariate_glm"]}
    mape = {m: float(((b[m] - b.actual).abs() / b.actual).mean())
            for m in ["fe_trend", "naive_last", "naive_mean3", "covariate_glm"]}
    coverage = float(((b.actual >= b.lo) & (b.actual <= b.hi)).mean())
    return b, {"mae": mae, "mape": mape, "pi90_coverage": coverage,
               "train": f"2012-{train_end}", "test": list(test_years), "n": len(b)}


def recovery(real, scenario, seeds=N_SEEDS):
    res = {"glm": {"p_at_k": [], "spearman": []}, "index": {"p_at_k": [], "spearman": []},
           "plan_found": [], "random_found": [], "oracle_found": [], "glm_access_sig": []}
    for s in range(seeds):
        syn = S.generate(real, scenario=scenario, seed=s)
        tab = M.district_table(syn)
        true_rate = S.true_gap_table(syn) / tab.population * 1e5
        top_true = set(true_rate.nlargest(K).index)
        model = M.fit_glm(syn)
        res["glm_access_sig"].append(M.access_is_supported(model))
        g_glm = M.glm_gap(syn, model, rng=np.random.default_rng(s))
        g_idx = M.index_gap(syn, rng=np.random.default_rng(s))
        for name, g in (("glm", g_glm), ("index", g_idx)):
            est = g.detection_gap / tab.population * 1e5
            key = g.priority_score if name == "index" else est
            res[name]["p_at_k"].append(len(set(key.nlargest(K).index) & top_true) / K)
            res[name]["spearman"].append(spearmanr(key, true_rate).statistic)
        # Planner: allocate on the index estimate, score against the truth.
        truth = pd.DataFrame({"population": tab.population,
                              "detection_gap": S.true_gap_table(syn)})
        plan = planner.allocate(g_idx.assign(population=tab.population), 4, 4, 300, targeting=5)
        found = 0.0
        rem = truth.detection_gap.clip(lower=0).to_dict()
        for d in plan.district:
            f = min(rem[d], 1200 * 5 * rem[d] / truth.population[d]); rem[d] -= f; found += f
        res["plan_found"].append(found)
        res["random_found"].append(planner.random_baseline(truth, 4, 4, 300, targeting=5, n_draws=300, seed=s))
        res["oracle_found"].append(planner.allocate(truth, 4, 4, 300, targeting=5,
                                                    favour_sensitivity=False).expected_cases_found.sum())

    def agg(v):
        v = np.asarray(v, float)
        return {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v))}

    return {
        "glm": {k: agg(v) for k, v in res["glm"].items()},
        "index": {k: agg(v) for k, v in res["index"].items()},
        "glm_access_detected_rate": float(np.mean(res["glm_access_sig"])),
        "planner_found": agg(res["plan_found"]),
        "random_found": agg(res["random_found"]),
        "oracle_found": agg(res["oracle_found"]),
        "seeds": seeds,
        "chance_p_at_k": K / 25,
    }


def main():
    real = pd.read_csv(PROC / "district_year.csv")
    detail, bt = backtest(real)
    detail.round(1).to_csv(PROC / "backtest_detail.csv", index=False)
    print("Backtest MAE (cases/district-year):", {k: round(v, 1) for k, v in bt["mae"].items()},
          f"| 90% PI coverage {bt['pi90_coverage']:.0%}")
    out = {"backtest": bt, "synthetic": {}}
    for scen in ("detection", "transmission"):
        r = recovery(real, scen)
        out["synthetic"][scen] = r
        print(f"[{scen}] P@5 glm={r['glm']['p_at_k']['mean']:.2f} index={r['index']['p_at_k']['mean']:.2f} "
              f"(chance {r['chance_p_at_k']:.2f}) | rho glm={r['glm']['spearman']['mean']:.2f} "
              f"index={r['index']['spearman']['mean']:.2f} | GLM access sig in "
              f"{r['glm_access_detected_rate']:.0%} | planner {r['planner_found']['mean']:.1f} vs random "
              f"{r['random_found']['mean']:.1f} vs oracle {r['oracle_found']['mean']:.1f}")
    json.dump(out, open(PROC / "validation.json", "w"), indent=2)


if __name__ == "__main__":
    main()
