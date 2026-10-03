"""TB Detection-Gap Dashboard for Sabah.

Run:  streamlit run app/streamlit_app.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import model as M  # noqa: E402
import planner  # noqa: E402

PROC = ROOT / "data/processed"
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
C_OBS, C_FC, C_MUTED = "#2a78d6", "#eb6834", "#8a8984"

st.set_page_config(page_title="Sabah TB Detection Gap", page_icon="🫁", layout="wide")


# ------------------------------------------------------------------ loaders
@st.cache_data
def load():
    dy = pd.read_csv(PROC / "district_year.csv")
    mo = pd.read_csv(PROC / "model_output.csv")
    gj = json.load(open(PROC / "districts.geojson"))
    summ = json.load(open(PROC / "model_summary.json"))
    val = json.load(open(PROC / "validation.json"))
    check = json.load(open(PROC / "data_check.json"))
    fc = pd.read_csv(PROC / "forecast.csv")
    return dy, mo, gj, summ, val, check, fc


@st.cache_data
def index_gaps(weights_items, p_max, p_min):
    dy = load()[0]
    g = M.index_gap(dy, dict(weights_items), p_max=p_max, p_min=p_min)
    g["population"] = M.district_table(dy).population
    return g


@st.cache_data
def run_planner(gaps, n_units, weeks, per_week, targeting, sens):
    return planner.summary(gaps, n_units, weeks, per_week, targeting, sens)


dy, mo, gj, summ, val, check, fc = load()
tab_info = M.district_table(dy)
latest_year = int(dy.year.max())

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Controls")
st.sidebar.subheader("Mobile screening units")
n_units = st.sidebar.slider("Number of units", 1, 10, 4)
weeks = st.sidebar.slider("Weeks per deployment", 1, 8, 4)
per_week = st.sidebar.number_input("People screened per unit per week (assumption)",
                                   50, 2000, 300, step=50)
targeting = st.sidebar.slider(
    "Targeting multiplier (assumption)", 1.0, 10.0, 1.0, 0.5,
    help="1 = unit screens a random sample of residents. >1 = screening is focused on "
         "symptomatic people and contacts, so each person screened is more likely to have TB.")
sens = st.sidebar.toggle("Favour sensitivity", True,
                         help="Allocate on the optimistic (90th-percentile) gap, so a district that "
                              "might hold many hidden cases isn't dropped for one that is merely "
                              "more certain. A missed district costs more than a wasted week.")

st.sidebar.subheader("Gap method")
chosen = summ["chosen_method"]
labels = {"index": "Scoring index", "glm": "GLM counterfactual"}
method = st.sidebar.radio(
    "Method", ["index", "glm"], index=0 if chosen == "index" else 1,
    format_func=lambda m: labels[m] + (" (headline)" if m == chosen else " (exploratory)"))

if method == "index":
    w = {
        "poverty": st.sidebar.slider("Weight: poverty", 0.0, 2.0, 1.0, 0.1),
        "noncitizen": st.sidebar.slider("Weight: non-citizen share of cases", 0.0, 2.0, 1.0, 0.1),
        "distance": st.sidebar.slider("Weight: distance to specialist hospital", 0.0, 2.0, 1.0, 0.1),
        "notif": st.sidebar.slider("Weight: low notification rate", 0.0, 2.0, 1.0, 0.1),
    }
    p_max, p_min = st.sidebar.slider(
        "Assumed detection: highest → lowest priority district", 0.30, 0.99, (0.60, 0.90), 0.01,
        help="Turns the index rank into hidden-case counts. The lowest-priority district is "
             "assumed to detect the upper value of its cases, the highest-priority the lower value.")[::-1]
    gaps = index_gaps(tuple(w.items()), p_max, p_min)
else:
    gaps = mo[mo.method == "glm"].set_index("district")
gaps = gaps.copy()
gaps["gap_rate_per_100k"] = gaps.detection_gap / gaps.population * 1e5
gaps["rank"] = (gaps.priority_score if method == "index" else gaps.gap_rate_per_100k) \
    .rank(ascending=False, method="first").astype(int)

# ------------------------------------------------------------------ header + banner
st.title("Where is TB going undetected in Sabah?")
st.caption("District detection-gap estimates → a deployment plan for mobile TB screening units.")
src = dy.data_source.iloc[0]
if src == "synthetic":
    st.error("**Data: SYNTHETIC.** Every number on this dashboard comes from simulated data.")
else:
    st.info(
        f"**Data: REAL, digitised (approximate).** District notification rates 2012–{latest_year} and "
        "non-citizen share digitised from the binned maps in Goroh et al. 2020 (rates within "
        f"{abs(check['digitised_state_rate'] / check['paper_state_rate'] - 1):.0%} of the paper's statewide "
        "128/100k); populations from DOSM Census 2010/2020; poverty from DOSM HIES 2022. "
        "Aggregate district data only. No individual records.")
if method != chosen:
    st.warning(f"**Exploratory method.** {summ['reason']}")

t_map, t_plan, t_fc, t_meth = st.tabs(["🗺️ Map", "🚐 Deployment plan", "📈 Forecast", "📋 Method & limitations"])

# ------------------------------------------------------------------ Tab 1: map
with t_map:
    tab = tab_info.join(gaps[["expected_good_access", "detection_gap", "gap_lo", "gap_hi",
                              "gap_rate_per_100k", "rank"]])
    tab["exp_rate"] = tab.expected_good_access / tab.population * 1e5
    hover = [
        f"<b>{d}</b> · priority #{r['rank']}<br>"
        f"Notified: {r.notif_rate_latest:.0f} /100k (≈{r.notified_latest:.0f} cases/yr)<br>"
        f"Expected with good access: {r.exp_rate:.0f} /100k<br>"
        f"<b>Detection gap: ≈{r.detection_gap:.0f} cases/yr ({r.gap_rate_per_100k:.0f} /100k)</b><br>"
        f"Range (10–90%): {r.gap_lo:.0f}–{r.gap_hi:.0f} cases<br>"
        f"Non-citizen share of cases: {r.noncitizen_share:.0%}<br>"
        f"Distance to specialist hospital: {r.dist_to_hospital_km:.0f} km"
        for d, r in tab.iterrows()]
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure(go.Choropleth(
            geojson=gj, locations=tab.index, z=tab.gap_rate_per_100k, featureidkey="id",
            colorscale=[[i / (len(BLUE) - 1), c] for i, c in enumerate(BLUE)],
            marker_line_color="white", marker_line_width=1,
            colorbar=dict(title=dict(text="Gap<br>/100k"), thickness=12, len=0.7),
            hovertext=hover, hoverinfo="text"))
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=480,
                          title=dict(text=f"Estimated undetected TB cases per 100k per year "
                                          f"({latest_year - 2}–{latest_year} avg)", font_size=14))
        st.plotly_chart(fig, width="stretch")
    with c2:
        top = tab.sort_values("rank").head(10).iloc[::-1]
        per = 1e5 / top.population
        fig = go.Figure(go.Bar(
            x=top.gap_rate_per_100k, y=top.index, orientation="h", marker_color=C_OBS,
            marker_cornerradius=4, customdata=top.detection_gap,
            error_x=dict(type="data", symmetric=False, array=(top.gap_hi - top.detection_gap) * per,
                         arrayminus=(top.detection_gap - top.gap_lo) * per, color=C_MUTED, thickness=1.5),
            hovertemplate="%{y}: %{x:.0f} /100k (≈%{customdata:.0f} hidden cases/yr)<extra></extra>"))
        fig.update_layout(height=480, margin=dict(l=0, r=10, t=30, b=0),
                          title=dict(text="Top 10 priority districts: hidden cases per 100k/yr (10–90% range)",
                                     font_size=14),
                          xaxis=dict(gridcolor="rgba(128,128,128,0.15)"), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
    st.caption(f"Statewide estimated gap: **≈{tab.detection_gap.sum():,.0f} undetected cases per year** "
               f"on top of ≈{tab.notified_latest.sum():,.0f} notified. Hover a district for details.")
    with st.expander("Table view"):
        show = tab.sort_values("rank")[["rank", "population", "notified_latest", "notif_rate_latest",
                                        "detection_gap", "gap_lo", "gap_hi", "gap_rate_per_100k",
                                        "poverty_rate", "noncitizen_share", "dist_to_hospital_km"]]
        st.dataframe(show.round(2), width="stretch")

# ------------------------------------------------------------------ Tab 2: plan
with t_plan:
    plan, s = run_planner(gaps[["population", "detection_gap", "gap_hi"]], n_units, weeks,
                          per_week, targeting, sens)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cases found with this plan", f"{s['cases_found']:.1f}")
    k2.metric("Random allocation", f"{s['random_cases_found']:.1f}",
              f"{s['uplift']:+.0%} with plan" if np.isfinite(s["uplift"]) else None)
    k3.metric("People screened", f"{s['people_screened']:,}")
    k4.metric("Number needed to screen", f"{s['nns']:,.0f}" if np.isfinite(s["nns"]) else "–")
    st.markdown(
        f"**Plan:** {n_units} unit(s) × {weeks} week(s) × {per_week} people/week. Each unit is sent to "
        "the district where it is expected to find the most hidden cases, given what earlier units "
        "already found there.")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(plan.rename(columns={"unit": "Unit #", "district": "District", "weeks": "Weeks",
                                          "people_screened": "People screened",
                                          "expected_cases_found": "Expected cases found"}).round(2),
                     hide_index=True, width="stretch")
    with c2:
        dep = plan.groupby("district").agg(units=("unit", "count"), found=("expected_cases_found", "sum"))
        z = gaps.index.map(lambda d: dep.units.get(d, 0))
        fig = go.Figure(go.Choropleth(
            geojson=gj, locations=gaps.index, z=z, featureidkey="id",
            colorscale=[[0, "#ecebe7"], [0.001, BLUE[2]], [1, BLUE[6]]], zmin=0,
            zmax=max(1, int(z.max())), marker_line_color="white", marker_line_width=1,
            colorbar=dict(title=dict(text="Units"), thickness=12, len=0.6, dtick=1),
            hovertext=[f"<b>{d}</b><br>{dep.units.get(d, 0)} unit(s)<br>"
                       f"≈{dep.found.get(d, 0):.1f} cases found" for d in gaps.index],
            hoverinfo="text"))
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=380,
                          title=dict(text="Where the units go", font_size=14))
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "Yield model: cases found = people screened × targeting multiplier × (remaining gap ÷ population), "
        "capped at the district's gap. Random baseline = average over 2,000 random assignments. "
        "Reported yields always use the central gap estimate, even when allocation favours sensitivity.")

# ------------------------------------------------------------------ Tab 3: forecast
with t_fc:
    order = gaps.sort_values("rank").index.tolist()
    sel = st.selectbox("District", order, index=0)
    hist = dy[dy.district == sel].sort_values("year")
    f = fc[fc.district == sel].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.year, y=hist.cases_notified, mode="lines+markers", name="Notified",
                             line=dict(color=C_OBS, width=2), marker=dict(size=8),
                             hovertemplate="%{x}: %{y:.0f} cases<extra>Notified</extra>"))
    fig.add_trace(go.Scatter(
        x=[hist.year.iloc[-1], f.year], y=[hist.cases_notified.iloc[-1], f.forecast], mode="lines+markers",
        name=f"{int(f.year)} forecast (90% interval)", line=dict(color=C_FC, width=2, dash="dot"),
        marker=dict(size=[0, 9]),
        error_y=dict(type="data", symmetric=False, array=[0, f.forecast_hi - f.forecast],
                     arrayminus=[0, f.forecast - f.forecast_lo], color=C_FC, thickness=1.5),
        hovertemplate="%{x}: %{y:.0f} cases<extra>Forecast</extra>"))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor="rgba(0,0,0,0)",
                      title=dict(text=f"{sel}: notified TB cases per year", font_size=14),
                      yaxis=dict(rangemode="tozero", gridcolor="rgba(128,128,128,0.15)", title="Cases"),
                      xaxis=dict(dtick=1), legend=dict(orientation="h", y=-0.15), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    bt = val["backtest"]
    st.markdown(
        f"**{int(f.year)} forecast for {sel}: {f.forecast:.0f} cases** (90% interval {f.forecast_lo:.0f}–"
        f"{f.forecast_hi:.0f}). Model: negative binomial with district effects and a common trend; intervals "
        f"widened ×{summ['forecast_dispersion_mult']:.0f} in dispersion because raw intervals covered only "
        f"{bt['pi90_coverage']:.0%} in the backtest.")
    st.caption(
        f"Honest note: in the 2017–18 backtest this model (MAE {bt['mae']['fe_trend']:.1f} cases/district-year) did "
        f"**not** beat the naive 'same as last year' baseline (MAE {bt['mae']['naive_last']:.1f}). TB "
        "notifications are very persistent and our digitised rates move in 40-per-100k bins. Use the forecast "
        "for scale and uncertainty, not to call year-on-year changes. TB is endemic: we forecast workload, "
        "not outbreaks.")

# ------------------------------------------------------------------ Tab 4: method
with t_meth:
    syn = val["synthetic"]
    st.subheader("Validation")
    v1, v2, v3 = st.columns(3)
    det = syn["detection"]
    v1.metric("Synthetic precision@5 (index)", f"{det['index']['p_at_k']['mean']:.2f}",
              f"chance = {det['chance_p_at_k']:.2f}", delta_color="off")
    v2.metric("Synthetic rank correlation (index)", f"{det['index']['spearman']['mean']:.2f}")
    v3.metric("Synthetic: planner vs random", f"{det['planner_found']['mean']:.1f} vs "
              f"{det['random_found']['mean']:.1f}", f"oracle {det['oracle_found']['mean']:.1f}",
              delta_color="off")
    tr = syn["transmission"]
    st.markdown(f"""
**1. Synthetic ground-truth recovery ({det['seeds']} simulated Sabahs, real populations and covariates).**
In a world where remote districts are *under-detected* (detection ≈ 90% near a hospital, ≈ 50% in the most
remote district), the index finds {det['index']['p_at_k']['mean'] * 5:.1f} of the true top-5 gap districts
(GLM: {det['glm']['p_at_k']['mean'] * 5:.1f}; chance: 1). A 4-unit plan built on it finds
{det['planner_found']['mean']:.1f} hidden cases vs {det['random_found']['mean']:.1f} for random allocation
(perfect knowledge: {det['oracle_found']['mean']:.1f}).

**…and where it fails.** In a world where remote districts truly have *less* TB and detection is uniform,
both methods point at the wrong districts (precision@5 = {tr['index']['p_at_k']['mean']:.2f}) and the
plan finds fewer cases than random ({tr['planner_found']['mean']:.1f} vs {tr['random_found']['mean']:.1f}).
The GLM even reports a "significant" access effect in {tr['glm_access_detected_rate']:.0%} of those worlds.
**Notification data alone cannot tell these two worlds apart.** That is why the roadmap starts with a
one-unit pilot: its screening yield is the measurement that decides which world Sabah is in.

**2. Temporal backtest (real data, train 2012–16 → predict 2017–18, {bt['n']} district-years).**
MAE in cases per district-year: district-trend NB model {bt['mae']['fe_trend']:.1f}; naive last-year
{bt['mae']['naive_last']:.1f}; 3-year mean {bt['mae']['naive_mean3']:.1f}; covariates-only GLM
{bt['mae']['covariate_glm']:.1f}. We do not beat naive persistence and we say so.

**3. Data check.** Digitised rates give a statewide average of {check['digitised_state_rate']} /100k vs
{check['paper_state_rate']} in the paper. Populations scaled ×{check['population_calibration']:.3f} so case
totals match the paper's 33,193; Kota Kinabalu 2018 set to the paper's exact 904.
""")
    st.subheader("Model")
    coef = pd.DataFrame(summ["coefficients"]).T
    coef.index = coef.index.str.replace("_z", " (per SD)")
    st.markdown(f"""
Negative binomial GLM, offset log(population), {summ['n_obs']} district-years, standard errors clustered by
district (covariates don't vary by year, so there are effectively **25** independent observations).
**Decision rule (set in advance):** use the GLM counterfactual only if the access coefficient is negative
and significant. {summ['reason']}
""")
    st.dataframe(coef[["irr", "coef", "se", "p"]].rename(columns={"irr": "rate ratio"}).round(3),
                 width="stretch")
    st.markdown("""
**Scoring index:** priority = z(poverty) + z(non-citizen share) + z(distance to hospital) − z(notification
rate), with weights adjustable in the sidebar. Low notification *despite* high risk raises priority. To
estimate cases for the planner, the index rank is mapped to an assumed detection probability
(default 90% → 60%) applied to the statewide notification rate. Uncertainty ranges come from re-drawing
the weights 500 times.

**Error trade-off.** A *false positive* (unit sent where few cases are hidden) wastes deployment weeks:
costly but recoverable. A *false negative* (a high-gap district missed) leaves infectious people
undiagnosed for months to years, transmitting all the while. So the planner **favours sensitivity**: it
allocates on the upper gap estimate when districts are close.
""")
    st.subheader("Who is missing from this data")
    st.markdown("""
- **Undocumented and stateless people** (including Sabah's large populations of migrant workers and
  sea-nomadic communities) are under-counted in the census denominators *and* less likely to reach a
  clinic. The non-citizen share term partly adjusts; it cannot fully correct.
- **Remote interior communities** (e.g. Tongod, Kinabatangan) are exactly where low notifications are
  hardest to interpret.
- **Private-sector diagnoses** not notified to myTB are invisible here.
""")
    st.subheader("Safeguards")
    st.markdown("""
- **District-level aggregates only.** No individual records, no village- or household-level targeting in this tool.
- Screening is delivered by **health services with community and NGO partners**, kept strictly separate from
  immigration enforcement. Screening data is never shared for enforcement. Without that firewall, people
  avoid screening and the whole approach fails.
- Free diagnosis and treatment regardless of documentation status. Messaging avoids naming communities.
- The non-citizen variable is used only to *send more services*, never to profile individuals.
""")
    st.subheader("Data sources")
    st.markdown("""
| Variable | Source |
|---|---|
| Notification rate by district × year 2012–18 | Goroh MMD et al. *Epidemiology of tuberculosis in Sabah, Malaysia, 2012–2018.* Infect Dis Poverty 2020;9:119 (Fig. 2, digitised by colour sampling against district boundaries; text values used exactly where quoted) |
| Non-citizen share of TB cases by district | Same paper, Fig. 5 (digitised) |
| Population | DOSM Census 2010 & 2020 (via citypopulation.de), linear interpolation |
| Poverty rate | DOSM Household Income Survey 2022, district level (OpenDOSM) |
| Rurality | log population density (census population ÷ boundary area); DOSM urban/rural share by district not openly available |
| Access | straight-line km from district town to nearest of 5 specialist hospitals (hand-coded proxy) |
| Boundaries | geoBoundaries MYS ADM2; Telupid merged into Beluran and Kalabakan into Tawau to match the paper's 25 districts |
""")
