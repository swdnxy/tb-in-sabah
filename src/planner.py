"""Allocate mobile TB screening units to districts.

Greedy: each unit, in turn, goes to the district with the highest marginal
expected cases found,
    cases_found = screened * targeting * (remaining_gap / population)
capped so a district's total can't exceed its gap (diminishing returns: once a
unit has screened a district, its remaining gap, and so the yield for the next
unit, falls).

`targeting` = 1 means the unit screens a random sample of residents; >1 means
it concentrates on higher-risk people (symptomatic, household contacts), which
real active-case-finding programmes do. It is an explicit assumption.

Sensitivity preference (PLAN 6.3): with `favour_sensitivity`, units are
allocated on the upper (90th-percentile) gap estimate, so a district that
*might* hold many hidden cases isn't dropped for one that is merely more
certain. Reported yields always use the central estimate.
"""
import numpy as np
import pandas as pd


def _yield(screened, remaining, pop, targeting):
    return min(remaining, screened * targeting * remaining / pop)


def allocate(gaps, n_units, weeks, per_week, targeting=1.0, favour_sensitivity=True):
    """gaps: DataFrame indexed by district with population, detection_gap, gap_hi."""
    screened = weeks * per_week
    plan_col = "gap_hi" if favour_sensitivity and "gap_hi" in gaps else "detection_gap"
    plan_rem = gaps[plan_col].clip(lower=0).astype(float).to_dict()
    true_rem = gaps["detection_gap"].clip(lower=0).astype(float).to_dict()
    pop = gaps["population"].to_dict()
    rows = []
    for unit in range(1, n_units + 1):
        best = max(plan_rem, key=lambda d: _yield(screened, plan_rem[d], pop[d], targeting))
        found = _yield(screened, true_rem[best], pop[best], targeting)
        plan_rem[best] -= _yield(screened, plan_rem[best], pop[best], targeting)
        true_rem[best] -= found
        rows.append({"unit": unit, "district": best, "weeks": weeks,
                     "people_screened": screened, "expected_cases_found": found})
    return pd.DataFrame(rows)


def random_baseline(gaps, n_units, weeks, per_week, targeting=1.0, n_draws=2000, seed=0):
    """Expected cases found if each unit goes to a uniformly random district."""
    rng = np.random.default_rng(seed)
    screened = weeks * per_week
    names = gaps.index.to_numpy()
    gap0 = gaps["detection_gap"].clip(lower=0).to_numpy(float)
    pop = gaps["population"].to_numpy(float)
    totals = np.empty(n_draws)
    for i in range(n_draws):
        rem = gap0.copy()
        tot = 0.0
        for j in rng.integers(0, len(names), n_units):
            f = min(rem[j], screened * targeting * rem[j] / pop[j])
            rem[j] -= f
            tot += f
        totals[i] = tot
    return float(totals.mean())


def summary(gaps, n_units, weeks, per_week, targeting=1.0, favour_sensitivity=True):
    plan = allocate(gaps, n_units, weeks, per_week, targeting, favour_sensitivity)
    found = plan.expected_cases_found.sum()
    rand = random_baseline(gaps, n_units, weeks, per_week, targeting)
    return plan, {
        "cases_found": found,
        "random_cases_found": rand,
        "uplift": found / rand - 1 if rand > 0 else np.nan,
        "people_screened": int(plan.people_screened.sum()),
        "nns": plan.people_screened.sum() / found if found > 0 else np.nan,
    }
