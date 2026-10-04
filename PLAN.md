# TB Detection-Gap Dashboard for Sabah — Build Plan

> Put this file at the root of your repo. If you use an AI assistant inside VS Code (Copilot Chat, Claude, etc.), point it at this file and work **one phase at a time**. Don't ask it to "build everything."

---

## 0. The project in one paragraph

Sabah carries a disproportionate share of Malaysia's TB burden. The literature attributes this mainly to **delayed care-seeking and poor access to TB services**, not HIV or drug resistance. That means low notification numbers in a remote district can mean *missed* cases rather than *absent* cases. We build a model that estimates, for each Sabah district, **how many TB cases are likely going undetected** (the *detection gap*). A dashboard then turns that estimate into a **deployment plan for mobile screening units**: which districts to send them to first, and roughly how many cases each deployment could find.

**What we are NOT claiming:** that we can forecast "outbreaks." TB is endemic and slow. We prioritise case-finding.

**Key evidence for the pitch (verify the wording in the paper before quoting):**
Goroh MMD et al., *Epidemiology of tuberculosis in Sabah, Malaysia, 2012–2018*, Infectious Diseases of Poverty, 2020. Open access: https://idpjournal.biomedcentral.com/articles/10.1186/s40249-020-00739-7
- Sabah ≈ 20% of Malaysia's TB notifications vs ≈ 10% of the population (2012–2018)
- Notification rates varied widely across the state's 25 districts. Semporna and Pitas were highest in 2018, and Kinabatangan and Tongod (remote interior) were lowest.
- A high share of cases had advanced disease at diagnosis, which suggests late detection
- The paper has figures for **notification rate by district and year** and **non-citizen share by district**. This is our main real dataset.

---

## 1. Time budget

Adjust to your actual submission deadline. Hard rule: **feature freeze 3 hours before the pitch.**

| Phase | Time | Output |
|---|---|---|
| 0. Setup | 30 min | Repo, venv, packages, folder structure |
| 1. Data | 2 h (time-boxed) | `district_year.csv` + `districts.geojson` |
| 2. Model | 2 h | Gap estimates, next-year forecast, validation numbers |
| 3. Dashboard | 3 h | Streamlit app with map, ranking, deployment planner |
| 4. Pitch + rehearsal | 2–3 h | 3–4 min deck, demo script, Q&A prep |

---

## 2. Team split (assumes 4; merge roles if fewer)

| Role | Owns | Phases |
|---|---|---|
| **Data lead** | Data extraction, cleaning, synthetic generator | 1 |
| **Model lead** | GLM, counterfactual gap, validation | 2 |
| **Dashboard lead** | Streamlit app, map, planner | 3 (starts with mock data in Phase 1) |
| **Pitch lead** | Deck, narrative, ethics slide, Q&A doc, rehearsal timing | 4 (starts immediately) |

The dashboard lead should **not wait** for the model. Build against `data/processed/mock_output.csv` with the same columns as the real output (see §5), then swap in real output later.

---

## 3. Phase 0 — Setup (VS Code)

**Extensions:** Python, Jupyter, Pylance, (optional) Rainbow CSV, GitLens.

```bash
mkdir sabah-tb && cd sabah-tb
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install pandas numpy statsmodels plotly streamlit shapely
pip freeze > requirements.txt
git init
```

Skip `geopandas`. It's painful to install on Windows and not needed here. Plotly reads GeoJSON directly.

**Folder structure:**

```
sabah-tb/
├── PLAN.md
├── requirements.txt
├── data/
│   ├── raw/                 # anything downloaded or hand-entered, never edited by code
│   └── processed/           # outputs of scripts
├── src/
│   ├── build_dataset.py     # raw → data/processed/district_year.csv
│   ├── synthetic.py         # generates synthetic ground-truth data
│   ├── model.py             # fit GLM, compute gap + forecast
│   ├── validate.py          # backtest + synthetic recovery test
│   └── planner.py           # allocate mobile units
├── app/
│   └── streamlit_app.py
└── notebooks/
    └── explore.ipynb        # scratch work only
```

Run the app with `streamlit run app/streamlit_app.py`.

---

## 4. Phase 1 — Data (time-box: 2 hours)

### 4.1 Unit of analysis
**District × year**, 2012–2018 (matches the paper). ~25 districts × 7 years ≈ 175 rows.

### 4.2 Sources

| Variable | Source | Notes |
|---|---|---|
| TB notification rate by district & year | Goroh et al. 2020, district-by-year figure + any additional files | Check the article's supplementary files first. If values exist only in a chart, digitise them (WebPlotDigitizer, free) and label as *approximate*. |
| Non-citizen share of TB cases by district | Same paper, non-citizen-by-district figure | Proxy for migrant/undocumented burden |
| District population | DOSM census 2010 and 2020 (OpenDOSM / data.gov.my) | Linearly interpolate for 2012–2018. Use to back out case counts: `cases = rate × pop / 100000` |
| Poverty / income by district | DOSM Household Income & Basic Amenities report (district level) | One value per district is fine |
| Rural share | DOSM census urban/rural by district | One value per district |
| Access proxy | Hand-coded: straight-line km from each district's main town to the nearest **major/specialist hospital** | Enter ~25 town coordinates + 4–6 hospital coordinates by hand and compute haversine distance. Document it as a proxy. |
| District boundaries | geoBoundaries (MYS, ADM2) or GADM level 2 | Filter to Sabah |

### 4.3 District-name reconciliation (do this early, it always breaks)
- The paper uses **25 districts**. Boundary files may have **27** (Kalabakan and Telupid are newer). Dissolve any extra districts into their parent so map and data match. Check which parent each belongs to; don't guess.
- Create `data/raw/district_names.csv` mapping: `geo_name, paper_name, canonical_name`. Every script joins on `canonical_name`.

### 4.4 Output schema — `data/processed/district_year.csv`

```
district, year, population, cases_notified, notif_rate,
noncitizen_share, poverty_rate, rural_share, dist_to_hospital_km
```

### 4.5 Fallback if extraction stalls past the time box
Use the 2018 values available in the paper's text plus `synthetic.py` (see §6.2) to produce a dataset with the same schema. **Say so on the dashboard and in the pitch.** Judges penalise hidden synthetic data, not disclosed synthetic data.

---

## 5. Phase 2 — Model

### 5.1 Core idea: counterfactual "good-access" prediction
Fit a negative binomial GLM on notified cases:

```
log E[cases] = log(population)            # offset
             + b1 * poverty_rate
             + b2 * noncitizen_share
             + b3 * rural_share
             + b4 * dist_to_hospital_km   # access term
             + b5 * (year - 2012)
```

`statsmodels.formula.api.glm(..., family=sm.families.NegativeBinomial(), offset=np.log(df.population))`

Then:
1. **Counterfactual expected cases:** predict again with `dist_to_hospital_km` set to the 10th-percentile value (i.e. "if this district had good access").
2. **Detection gap** = `counterfactual_expected − cases_notified` (floor at 0).
3. **Gap rate** = gap per 100,000. Rank districts on this.

### 5.2 Sanity check on the access coefficient
- If `b4 < 0` (more remote → fewer notifications), this is consistent with under-detection. Use the model.
- If `b4 ≥ 0` or isn't significant, **switch to the transparent scoring index** below, and say why in the pitch. Remote districts may genuinely have lower transmission, and the data can't fully separate the two. Being upfront about this is a technical-execution point, not a weakness.

**Fallback scoring index** (z-scores, weights adjustable on the dashboard):
```
priority = w1*z(poverty) + w2*z(noncitizen_share) + w3*z(dist_to_hospital)
         - w4*z(notif_rate)    # low notifications despite high risk → higher priority
```

### 5.3 Next-year forecast
Using the fitted GLM, predict 2019 notifications per district (extrapolating the year term). Show it with uncertainty (prediction intervals from the NB model or bootstrapped over districts). This satisfies the "anticipate" part of the problem statement.

### 5.4 Model output schema — `data/processed/model_output.csv`

```
district, notified_latest, expected_good_access, detection_gap,
gap_rate_per_100k, forecast_next_year, forecast_lo, forecast_hi,
priority_rank, method   # "glm" or "index"
```

---

## 6. Phase 2b — Validation (this is what wins "Technical Execution")

### 6.1 Temporal backtest (real data)
- Train on 2012–2016, predict 2017–2018 notifications per district.
- Report MAE and compare against a **naive baseline** (last year's value carried forward). If you don't beat naive, say so and explain.

### 6.2 Synthetic ground-truth recovery test
Real undetected cases are unknown by definition, so build a world where we know them:
1. `synthetic.py`: for each district, set a **true incidence** from the risk covariates using chosen coefficients.
2. Apply a **detection probability** that falls with distance to hospital (e.g. logistic, ~0.9 near, ~0.5 remote).
3. Draw notified cases ~ Poisson(true × detection).
4. Run the exact same pipeline and check: do the top-5 gap districts match the true top-5? Report **precision@5** and the correlation between estimated and true gaps.

### 6.3 Error trade-off (state it explicitly)
- **False positive** (send a unit to a district with few hidden cases): wasted deployment weeks. Costly but recoverable.
- **False negative** (miss a high-gap district): undetected cases keep transmitting for months to years.
- We therefore **favour sensitivity**: when ranks are close, the planner includes the uncertain district rather than dropping it.

---

## 7. Phase 3 — Dashboard (Streamlit)

### 7.1 Layout (single page with tabs is enough)

**Sidebar controls**
- Number of mobile screening units (1–10)
- Weeks per deployment (1–8)
- People screened per unit per week (default 300, labelled *assumption*)
- Method toggle: GLM gap vs scoring index (with weight sliders when index is selected)
- Banner: "Data: real (digitised) / synthetic". Always visible.

**Tab 1 — Map**
- Plotly choropleth of Sabah districts coloured by detection-gap rate
- Hover: notified rate, expected-with-good-access rate, gap, non-citizen share, distance to hospital

**Tab 2 — Deployment plan**
- `planner.py`: greedy allocation. Each unit goes to the district with the highest **marginal expected cases found**, where
  `cases_found = screened × (remaining_gap / population)`, and a district's total can't exceed its gap (diminishing returns).
- Output table: unit #, district, weeks, expected cases found
- Headline metric: "Estimated cases found with this plan vs. random allocation"

**Tab 3 — Forecast**
- Line chart per selected district: 2012–2018 notified + 2019 forecast with interval

**Tab 4 — Method & limitations**
- Plain-language model description, validation numbers, data sources
- **Who's missing:** undocumented and stateless people are likely under-notified; the model partly corrects for this but can't fully
- **Safeguards:** district-level aggregates only, no individual records; screening delivered through health services and community/NGO partners, kept separate from immigration enforcement

### 7.2 Build order
1. Load mock output → table renders
2. Map renders with GeoJSON (fix name joins here)
3. Sidebar + planner
4. Forecast tab
5. Swap mock for real `model_output.csv`
6. Limitations tab (copy from pitch lead's notes)

Use `@st.cache_data` on all loaders so the live demo doesn't lag.

---

## 8. Phase 4 — Pitch (3–4 minutes, hard limit)

| Time | Section | Content |
|---|---|---|
| 0:00–0:30 | Hook | Sabah ≈ 10% of Malaysia's population, ≈ 20% of its TB notifications |
| 0:30–1:00 | Reframe | The lowest notification rates are in remote districts. Low numbers may mean missed cases. TB is a detection problem. |
| 1:00–2:15 | **Live demo** | Map → gap ranking → set 4 units → plan + cases found vs random |
| 2:15–2:45 | Validation | Backtest MAE vs naive; synthetic precision@5 |
| 2:45–3:15 | Ethics | Missing populations, stigma/enforcement safeguards, why we favour sensitivity |
| 3:15–3:45 | Roadmap | Real myTB access via Sabah State Health Dept, sub-district resolution, travel-time access, pilot with one mobile unit |

**Demo safety:** record a 60-second screen capture of the working demo as backup in case wifi or the laptop fails.

### Q&A prep (pitch lead writes 2–3 sentence answers for each)
- How do you know low notifications mean under-detection rather than low transmission?
- Your data ends in 2018. Is it still relevant?
- What if the data is digitised from a figure? How accurate is it?
- How did you pick the screening-yield assumption?
- Couldn't this be used to target undocumented communities?
- What does a mobile screening unit actually cost, and who pays?
- Why district level and not finer?
- How would the State Health Department actually use this?

---

## 9. Definition of done (check before feature freeze)

- [ ] `streamlit run app/streamlit_app.py` works from a fresh terminal
- [ ] Map shows all districts, with no grey "missing join" areas
- [ ] Planner output changes when sliders move
- [ ] Data-source banner (real/synthetic) visible on every tab
- [ ] Validation numbers on the dashboard match the slide
- [ ] Backup demo video recorded
- [ ] Pitch rehearsed under 4:00 at least twice
- [ ] Repo pushed; README has the run command

---

## 10. Scope cuts if behind schedule (cut in this order)

1. Forecast intervals → point forecast only
2. Forecast tab → drop entirely
3. GLM → scoring index only (keep validation on synthetic data)
4. Real data → synthetic only, clearly disclosed

**Never cut:** the map, the deployment planner, the limitations/ethics content, the validation number. Those map directly onto the rubric.
