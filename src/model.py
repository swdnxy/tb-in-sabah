"""Detection-gap model, scoring index and next-year forecast.

data/processed/district_year.csv
  -> data/processed/model_output.csv   (one row per district per method)
  -> data/processed/model_summary.json (coefficients, chosen method)

Two ways to estimate where TB is being missed:

glm    Negative-binomial GLM of notified cases on risk covariates + distance to
       a specialist hospital (offset log population). Counterfactual: set
       distance to the 10th-percentile value ("good access") and predict again.
       gap = counterfactual expected - notified.

index  Transparent z-score index: high poverty, high non-citizen share and long
       distance raise priority; high notification lowers it (low notification
       despite high risk = likely missed cases). The index ranks districts; to
       turn rank into cases for the planner we assume a detection probability
       that falls linearly from P_MAX (lowest priority) to P_MIN (highest
       priority), applied to the statewide mean notification rate so the gap is
       monotone in the score: gap = statewide_rate * pop * (1/p - 1). Both P
       values are dashboard sliders.

Rule from PLAN 5.2: if the access coefficient isn't negative and significant,
the index is the headline method and the GLM is shown as exploratory.

Forecast: NB GLM with district fixed effects + common linear trend, predicting
the next year with simulation-based 90% prediction intervals.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"

RISK = ["poverty_rate", "noncitizen_share", "log_pop_density"]
ACCESS = "dist_to_hospital_km"
BASE_YEAR = 2012
RECENT = 3            # years averaged for "notified latest" (smooths digitisation bins)
N_SIM = 2000
DEFAULT_WEIGHTS = {"poverty": 1.0, "noncitizen": 1.0, "distance": 1.0, "notif": 1.0}
P_MAX, P_MIN = 0.90, 0.60


def _z(s):
    return (s - s.mean()) / s.std(ddof=0)


def district_table(df):
    """One row per district: time-invariant covariates + recent notifications."""
    last = df.year.max()
    recent = df[df.year > last - RECENT]
    g = df.groupby("district")
    out = g[RISK + [ACCESS, "noncitizen_share"]].first().loc[:, lambda x: ~x.columns.duplicated()]
    out["population"] = g.apply(lambda x: x.loc[x.year.idxmax(), "population"], include_groups=False)
    out["notified_latest"] = recent.groupby("district").cases_notified.mean()
    out["notif_rate_latest"] = out.notified_latest / out.population * 1e5
    return out


# ---------------------------------------------------------------- GLM method
def fit_glm(df, covars=None):
    covars = covars or RISK + [ACCESS]
    d = df.copy()
    d["t"] = d.year - BASE_YEAR
    stats = {c: (d[c].mean(), d[c].std(ddof=0)) for c in covars}
    for c in covars:
        d[c + "_z"] = (d[c] - stats[c][0]) / stats[c][1]
    formula = "cases_notified ~ " + " + ".join(c + "_z" for c in covars) + " + t"
    off = np.log(d.population)
    alpha = smf.negativebinomial(formula, d, offset=off).fit(disp=0, maxiter=200).params["alpha"]
    res = smf.glm(formula, d, family=sm.families.NegativeBinomial(alpha=alpha), offset=off).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.district)[0]})
    return {"res": res, "alpha": float(alpha), "stats": stats, "covars": covars}


def _design(model, tab, year, override=None):
    X = pd.DataFrame({"Intercept": 1.0}, index=tab.index)
    for c in model["covars"]:
        v = tab[c] if not override or c not in override else override[c]
        X[c + "_z"] = (v - model["stats"][c][0]) / model["stats"][c][1]
    X["t"] = year - BASE_YEAR
    return X[model["res"].params.index]


def glm_gap(df, model, rng=None):
    rng = rng or np.random.default_rng(0)
    tab = district_table(df)
    years = sorted(df.year.unique())[-RECENT:]
    good = df[ACCESS].quantile(0.10)
    res = model["res"]
    sims = rng.multivariate_normal(res.params.values, res.cov_params().values, N_SIM)
    exp_cf = np.zeros((N_SIM, len(tab)))
    for y in years:
        X = _design(model, tab, y, {ACCESS: pd.Series(np.minimum(tab[ACCESS], good), index=tab.index)})
        pop = df[df.year == y].set_index("district").population.reindex(tab.index).values
        exp_cf += np.exp(sims @ X.values.T) * pop
    exp_cf /= len(years)
    point = exp_cf.mean(0)
    gaps = np.clip(exp_cf - tab.notified_latest.values, 0, None)
    out = pd.DataFrame(index=tab.index)
    out["notified_latest"] = tab.notified_latest
    out["expected_good_access"] = point
    out["detection_gap"] = np.clip(point - tab.notified_latest, 0, None)
    out["gap_lo"] = np.percentile(gaps, 10, axis=0)
    out["gap_hi"] = np.percentile(gaps, 90, axis=0)
    out["method"] = "glm"
    return out


# -------------------------------------------------------------- Index method
def index_scores(tab, w):
    return (w["poverty"] * _z(tab.poverty_rate)
            + w["noncitizen"] * _z(tab.noncitizen_share)
            + w["distance"] * _z(tab[ACCESS])
            - w["notif"] * _z(tab.notif_rate_latest))


def _gap_from_score(tab, score, p_max, p_min):
    pct = score.rank(pct=True, method="average")
    pct = (pct - pct.min()) / (pct.max() - pct.min())
    p_detect = p_max - (p_max - p_min) * pct
    state_rate = tab.notified_latest.sum() / tab.population.sum()
    return state_rate * tab.population * (1 / p_detect - 1), p_detect


def index_gap(df, weights=None, p_max=P_MAX, p_min=P_MIN, rng=None):
    rng = rng or np.random.default_rng(0)
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    tab = district_table(df)
    score = index_scores(tab, w)
    gap, p_detect = _gap_from_score(tab, score, p_max, p_min)
    # Uncertainty: re-draw the weights (Dirichlet around equal weighting) and
    # see how much each district's gap moves.
    keys = list(DEFAULT_WEIGHTS)
    draws = rng.dirichlet(np.ones(len(keys)) * 4, 500) * len(keys)
    sims = np.array([_gap_from_score(tab, index_scores(tab, dict(zip(keys, d))), p_max, p_min)[0].values
                     for d in draws])
    out = pd.DataFrame(index=tab.index)
    out["notified_latest"] = tab.notified_latest
    out["expected_good_access"] = tab.notified_latest + gap
    out["detection_gap"] = gap
    out["gap_lo"] = np.percentile(sims, 10, axis=0)
    out["gap_hi"] = np.percentile(sims, 90, axis=0)
    out["priority_score"] = score
    out["assumed_detection"] = p_detect
    out["method"] = "index"
    return out


# ----------------------------------------------------------------- Forecast
def fit_forecast(df):
    d = df.copy()
    d["t"] = d.year - BASE_YEAR
    off = np.log(d.population)
    f = "cases_notified ~ C(district) + t"
    alpha = smf.negativebinomial(f, d, offset=off).fit(disp=0, maxiter=300).params["alpha"]
    res = smf.glm(f, d, family=sm.families.NegativeBinomial(alpha=alpha), offset=off).fit()
    return res, float(alpha)


def project_population(df, year):
    out = {}
    for dist, g in df.groupby("district"):
        b = np.polyfit(g.year, g.population, 1)
        out[dist] = np.polyval(b, year)
    return pd.Series(out)


def forecast(df, years, rng=None, level=0.90, alpha_mult=1.0):
    rng = rng or np.random.default_rng(0)
    res, alpha = fit_forecast(df)
    alpha = alpha * alpha_mult
    sims = rng.multivariate_normal(res.params.values, res.cov_params().values, N_SIM)
    rows = []
    for y in years:
        pop = project_population(df, y)
        new = pd.DataFrame({"district": pop.index, "t": y - BASE_YEAR})
        X = res.model.data.orig_exog.iloc[:0].reindex(range(len(new))).fillna(0)
        X["Intercept"] = 1.0
        for i, dname in enumerate(new.district):
            col = f"C(district)[T.{dname}]"
            if col in X:
                X.loc[i, col] = 1.0
        X["t"] = new.t.values
        mu = np.exp(sims @ X[res.params.index].values.T) * pop.values
        n = 1 / alpha
        draws = rng.negative_binomial(n, n / (n + mu))
        lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
        rows.append(pd.DataFrame({
            "district": new.district, "year": y,
            "forecast": np.exp(X[res.params.index].values @ res.params.values) * pop.values,
            "forecast_lo": np.percentile(draws, lo, axis=0),
            "forecast_hi": np.percentile(draws, hi, axis=0),
            "population": pop.values}))
    return pd.concat(rows, ignore_index=True)


def calibrate_dispersion(df, train_end, test_years, level=0.90):
    """Smallest NB dispersion multiplier giving >= `level` interval coverage on a
    held-out window. Raw NB intervals under-cover (74% in the 2017-18 backtest)
    because digitised bins flip year to year, so the next-year forecast widens
    them by the multiplier that restores 90% coverage on that holdout."""
    train = df[df.year <= train_end]
    actual = df[df.year.isin(test_years)].set_index(["district", "year"]).cases_notified
    for mult in [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
        fc = forecast(train, list(test_years), alpha_mult=mult).set_index(["district", "year"])
        a = actual.reindex(fc.index)
        if ((a >= fc.forecast_lo) & (a <= fc.forecast_hi)).mean() >= level:
            return float(mult)
    return 32.0


# --------------------------------------------------------------------- Main
def access_is_supported(model):
    res = model["res"]
    k = ACCESS + "_z"
    return bool(res.params[k] < 0 and res.pvalues[k] < 0.05)


def build_output(df):
    model = fit_glm(df)
    chosen = "glm" if access_is_supported(model) else "index"
    mult = calibrate_dispersion(df, df.year.max() - 2, [df.year.max() - 1, df.year.max()])
    fc = forecast(df, [df.year.max() + 1], alpha_mult=mult).set_index("district")
    parts = []
    for out in (glm_gap(df, model), index_gap(df)):
        pop = district_table(df).population
        out["gap_rate_per_100k"] = out.detection_gap / pop * 1e5
        key = out.priority_score if "priority_score" in out else out.gap_rate_per_100k
        out["priority_rank"] = key.rank(ascending=False, method="first").astype(int)
        out["forecast_next_year"] = fc.forecast
        out["forecast_lo"] = fc.forecast_lo
        out["forecast_hi"] = fc.forecast_hi
        out["population"] = pop
        parts.append(out.reset_index().rename(columns={"index": "district"}))
    output = pd.concat(parts, ignore_index=True)
    return output, model, chosen, mult


def summarise(model, chosen):
    res = model["res"]
    coefs = {k: {"coef": float(res.params[k]), "se": float(res.bse[k]),
                 "p": float(res.pvalues[k]), "irr": float(np.exp(res.params[k]))}
             for k in res.params.index}
    k = ACCESS + "_z"
    return {
        "chosen_method": chosen,
        "access_coef_per_sd": float(res.params[k]),
        "access_p": float(res.pvalues[k]),
        "alpha": model["alpha"],
        "n_obs": int(res.nobs),
        "n_districts": 25,
        "se_type": "cluster-robust by district",
        "coefficients": coefs,
        "reason": ("Access coefficient negative and significant: GLM counterfactual used."
                   if chosen == "glm" else
                   f"Access coefficient {res.params[k]:+.3f}/SD (p={res.pvalues[k]:.2f}) is not "
                   "significant with only 25 districts, so the data cannot separate missed cases "
                   "from genuinely lower transmission in remote areas. Following our pre-registered "
                   "rule we lead with the transparent scoring index and show the GLM as exploratory."),
    }


def main():
    df = pd.read_csv(PROC / "district_year.csv")
    output, model, chosen, mult = build_output(df)
    cols = ["district", "method", "population", "notified_latest", "expected_good_access",
            "detection_gap", "gap_lo", "gap_hi", "gap_rate_per_100k", "forecast_next_year",
            "forecast_lo", "forecast_hi", "priority_rank", "priority_score", "assumed_detection"]
    output[cols].round(3).to_csv(PROC / "model_output.csv", index=False)
    summary = summarise(model, chosen)
    summary["forecast_dispersion_mult"] = mult
    json.dump(summary, open(PROC / "model_summary.json", "w"), indent=2)
    print(model["res"].summary().tables[1])
    print(summary["reason"])
    top = output[output.method == chosen].nsmallest(8, "priority_rank")
    print(top[["district", "notified_latest", "detection_gap", "gap_rate_per_100k", "priority_rank"]].round(1).to_string(index=False))
    full_fc = forecast(df, [df.year.max() + 1], alpha_mult=mult)
    full_fc.round(1).to_csv(PROC / "forecast.csv", index=False)


if __name__ == "__main__":
    main()
