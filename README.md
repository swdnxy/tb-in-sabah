# Sabah TB Detection-Gap Dashboard

**Public Health & Epidemiology track.** Sabah has ~10% of Malaysia's population but ~20% of its TB
notifications (Goroh et al. 2020). The literature points to delayed care-seeking and poor access, so a
*low* notification rate in a remote district may mean **missed** cases, not absent ones. This project
estimates where TB is likely going undetected in each of Sabah's 25 districts and turns that into a
**deployment plan for mobile screening units**.

We do **not** claim to forecast outbreaks. TB is endemic and slow, so we prioritise case-finding.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

All processed outputs are committed, so the app runs without rebuilding. To rebuild from raw data
(about 3 minutes):

```bash
./run_pipeline.sh
```

## What's in it

| Path | What it does |
|---|---|
| `src/build_geo.py` | geoBoundaries ADM2 → 25 Sabah districts (Telupid→Beluran, Kalabakan→Tawau) |
| `src/digitise.py` | Reads the paper's binned district maps (Fig. 2 rates × 7 years, Fig. 5 non-citizen share) by fitting our boundaries onto each panel and sampling the colour per district. Uses exact values where the text quotes them. |
| `src/build_dataset.py` | → `data/processed/district_year.csv` (175 district-years) |
| `src/model.py` | NB GLM + counterfactual gap; transparent scoring index; NB district-trend forecast |
| `src/validate.py` | Temporal backtest + synthetic ground-truth recovery (2 scenarios × 200 worlds) |
| `src/synthetic.py` | Synthetic worlds with known undetected cases (also the PLAN §4.5 fallback: `--write`) |
| `src/planner.py` | Greedy allocation of screening units with diminishing returns + random baseline |
| `app/streamlit_app.py` | Dashboard: Map · Deployment plan · Forecast · Method & limitations |
| `PROPOSAL.md` | One-page project proposal |
| `PITCH.md` | 3:45 pitch script, slide content, demo script, Q&A answers |

## Headline results (all reproduced by `run_pipeline.sh`)

| | |
|---|---|
| Data check | Digitised rates average **129.7/100k** vs the paper's **128** (+1.3%) |
| Access effect (GLM) | −0.055 log-rate per SD of distance, **p = 0.51** (clustered by district) → not significant |
| Method used | **Scoring index** (pre-set rule: use the GLM counterfactual only if the access effect is negative and significant) |
| Top-5 priority districts | Kinabatangan, Tongod, Beluran, Kudat, Kunak (the GLM's top 5 shares 4 of them) |
| Synthetic recovery, under-detection world | precision@5 **0.79** (chance 0.20), Spearman ρ 0.75 |
| Synthetic planner value | 4 units find **18.1** hidden cases vs **7.5** random (perfect knowledge 23.8) |
| Synthetic, low-transmission world | Methods **fail** (precision@5 0.00). Notification data can't tell the two worlds apart, so the pilot unit's yield is what decides |
| Backtest 2017–18 | District-trend model MAE **27.6** cases/district-year vs naive last-year **20.7**. We don't beat persistence and say so |

## Data sources

- Goroh MMD et al. *Epidemiology of tuberculosis in Sabah, Malaysia, 2012–2018.* Infect Dis Poverty
  2020;9:119. doi:10.1186/s40249-020-00739-7 (CC BY 4.0). Full text via Europe PMC (PMC7447595).
- DOSM Census 2010 & 2020 district populations (via citypopulation.de); DOSM HIES 2022 district poverty
  (OpenDOSM `hies_district`).
- geoBoundaries MYS ADM2 (gbOpen).
- Hand-coded: district town and 5 specialist-hospital coordinates (`data/raw/towns.csv`,
  `data/raw/hospitals.csv`).

## Known limitations

- Rates are **digitised from binned maps** (40-per-100k bins, bin midpoints). Small districts
  (Putatan, Penampang) cover only a few pixels and are flagged `digitise_low_conf`.
- Poverty is from 2022 (the earliest district-level release), after the 2012–18 study period.
- Rurality uses log population density; DOSM doesn't publish district urban/rural share as open data.
- Access is straight-line distance, not travel time.
- Non-citizen share is the share of TB *cases*, which is itself affected by detection.
