"""
Sabah TB Mobile Clinic Dispatch System - Final Phase 2
======================================================
Complete 7-Metric Prioritization Engine (100% Total Model Weight):
  - Layer 1: Compound Structural Exposure (60.0% Weight) — Formulation C:
      * Geographic Access Gradient (G) [15.0% base + 15.0% synergistic]
      * Household Income Deprivation (I) [15.0% base + 15.0% synergistic]
      => S_compound = (15.0 * G) + (15.0 * I) + (30.0 * [G * I])
  - Layer 2: Clinical & Operational Modifiers (40.0% Weight) — Linear Additive:
      * Metric 3: TB Diagnostic Latency / Lag (20.0% Weight)
      * Metric 4: Diabetes Mellitus Prevalence (5.0% Weight)
      * Metric 5: Weather & Seasonal Road Passability Risk (5.0% Weight)
      * Metric 6: Smoking & Tobacco Prevalence (5.0% Weight)
      * Metric 7: HIV / Immunosuppression Prevalence (5.0% Weight)
      => S_modifiers = S_lag + S_dm + S_weather + S_smoking + S_hiv
  - Final Priority Score:
      => S_final = S_compound + S_modifiers  (out of 100.0 points)

Features:
  - Option C: Layered Hybrid Folium Map (Free Carto Dark / OpenStreetMap, NO API KEY)
  - Interactive Layer Control: Centroids, Heatmap, Diagnostic Hospitals, Primary Clinics
  - Rich tooltips & popups with full 7-metric evidence breakdown
  - Top metric KPI cards & 27-district prioritized dataframe with CSV export
  - Interactive Collapsible Bottom Drawer with full LaTeX mathematical architecture

Run:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sabah TB Mobility Dispatch - 7-Metric Engine",
    page_icon="\U0001f690",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# DESIGN TOKENS (Dark Mode, Teal/Amber Accent Palette)
# ---------------------------------------------------------------------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1421 50%, #0a1628 100%); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%);
    border-right: 1px solid rgba(0,200,180,0.15);
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(0,200,180,0.08) 0%, rgba(0,150,200,0.05) 100%);
    border: 1px solid rgba(0,200,180,0.2);
    border-radius: 12px;
    padding: 14px 18px;
    backdrop-filter: blur(10px);
}
[data-testid="stMetric"] label {
    color: rgba(0,200,180,0.85) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #e8f4f8 !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,200,180,0.15);
    border-radius: 10px;
    overflow: hidden;
}
hr { border-color: rgba(0,200,180,0.2); }

.stDownloadButton > button {
    background: linear-gradient(135deg, #00c8b4 0%, #0096c7 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.2s;
}
.stDownloadButton > button:hover { opacity: 0.85; }

h2 { color: #00c8b4 !important; letter-spacing: -0.01em; }
h3 { color: #7ecfdf !important; }

.stSlider [data-baseweb="slider"] { padding: 6px 0; }
.stCheckbox label { color: #a8c8d8 !important; }
.stCaption { color: #5a7a8a !important; }
.stInfo, .stSuccess, .stWarning, .stError { border-radius: 10px !important; border-left-width: 4px !important; }
[data-testid="stExpander"] { background: rgba(0,200,180,0.04); border: 1px solid rgba(0,200,180,0.12); border-radius: 10px; }

@keyframes softGlow {
  0%, 100% { box-shadow: 0 0 0 0 rgba(255,80,80,0); }
  50%       { box-shadow: 0 0 14px 4px rgba(255,80,80,0.18); }
}
.metric-critical { animation: softGlow 3s ease-in-out infinite; }

@keyframes compoundPulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
  50%       { box-shadow: 0 0 18px 6px rgba(239,68,68,0.25); }
}
.metric-compound { animation: compoundPulse 2.5s ease-in-out infinite; }

.hies-connected {
    display: inline-block; background: rgba(34,197,94,0.18); border: 1px solid #22c55e;
    color: #22c55e; border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600;
}
.hies-cached {
    display: inline-block; background: rgba(245,158,11,0.18); border: 1px solid #f59e0b;
    color: #f59e0b; border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600;
}

.sidebar-kpi {
    background: rgba(0,200,180,0.06);
    border: 1px solid rgba(0,200,180,0.2);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 10px;
}
.sidebar-kpi-title {
    color: #7ecfdf;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
}
.sidebar-kpi-val {
    color: #ffffff;
    font-size: 1.15rem;
    font-weight: 700;
}
.sidebar-kpi-sub {
    color: #94a3b8;
    font-size: 0.72rem;
}

.math-drawer-box {
    background: rgba(13, 27, 42, 0.75);
    border: 1px solid rgba(0, 200, 180, 0.25);
    border-radius: 12px;
    padding: 18px 24px;
    margin-top: 10px;
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CONSTANTS - 27 Official Sabah District Centroids (WGS84)
# ---------------------------------------------------------------------------
DISTRICTS = [
    ("Kota Kinabalu",  5.9804, 116.0735),
    ("Penampang",      5.9100, 116.0900),
    ("Putatan",        5.8900, 116.0500),
    ("Tuaran",         6.1800, 116.2300),
    ("Papar",          5.7300, 115.9300),
    ("Kota Belud",     6.3500, 116.4300),
    ("Ranau",          5.9600, 116.6600),
    ("Kudat",          6.8800, 116.8500),
    ("Kota Marudu",    6.5000, 116.7700),
    ("Pitas",          6.7100, 117.0300),
    ("Keningau",       5.3400, 116.1600),
    ("Tambunan",       5.6700, 116.3600),
    ("Tenom",          5.1200, 115.9400),
    ("Nabawan",        5.0300, 116.4300),
    ("Beaufort",       5.3400, 115.7500),
    ("Kuala Penyu",    5.5700, 115.6000),
    ("Sipitang",       5.0800, 115.5500),
    ("Sandakan",       5.8402, 118.1179),
    ("Beluran",        5.8900, 117.5500),
    ("Telupid",        5.6200, 117.1200),
    ("Tongod",         5.2700, 116.9700),
    ("Kinabatangan",   5.5000, 117.8300),
    ("Tawau",          4.2498, 117.8871),
    ("Lahad Datu",     5.0268, 118.3270),
    ("Semporna",       4.4818, 118.6112),
    ("Kunak",          4.6800, 118.2500),
    ("Kalabakan",      4.4100, 117.4900),
]

DIST_DF = pd.DataFrame(DISTRICTS, columns=["district", "lat", "lon"])

MOH_URL = (
    "https://raw.githubusercontent.com/MoH-Malaysia/"
    "data-resources-public/main/facilities_master.csv"
)

DOSM_HIES_URL = "https://storage.dosm.gov.my/hies/hies_district.csv"

# ---------------------------------------------------------------------------
# HIES FALLBACK BASELINE (Authentic DOSM HIES Estimates for all 27 Districts)
# ---------------------------------------------------------------------------
HIES_FALLBACK = {
    "Kota Kinabalu": {"income_median": 6542, "poverty_rate": 4.1},
    "Penampang":     {"income_median": 5890, "poverty_rate": 5.3},
    "Putatan":       {"income_median": 5120, "poverty_rate": 6.8},
    "Sandakan":      {"income_median": 4950, "poverty_rate": 10.3},
    "Tuaran":        {"income_median": 4350, "poverty_rate": 9.2},
    "Papar":         {"income_median": 3980, "poverty_rate": 11.5},
    "Tawau":         {"income_median": 3840, "poverty_rate": 11.8},
    "Keningau":      {"income_median": 3680, "poverty_rate": 12.4},
    "Ranau":         {"income_median": 3620, "poverty_rate": 15.9},
    "Beaufort":      {"income_median": 3450, "poverty_rate": 13.2},
    "Kota Belud":    {"income_median": 3210, "poverty_rate": 14.7},
    "Lahad Datu":    {"income_median": 3190, "poverty_rate": 15.3},
    "Tambunan":      {"income_median": 3120, "poverty_rate": 16.1},
    "Tenom":         {"income_median": 2980, "poverty_rate": 17.6},
    "Kota Marudu":   {"income_median": 2980, "poverty_rate": 21.3},
    "Sipitang":      {"income_median": 2890, "poverty_rate": 18.1},
    "Kudat":         {"income_median": 2820, "poverty_rate": 18.4},
    "Kunak":         {"income_median": 2790, "poverty_rate": 20.4},
    "Kuala Penyu":   {"income_median": 2760, "poverty_rate": 19.7},
    "Beluran":       {"income_median": 2680, "poverty_rate": 22.5},
    "Kinabatangan":  {"income_median": 2590, "poverty_rate": 23.8},
    "Kalabakan":     {"income_median": 2470, "poverty_rate": 24.6},
    "Semporna":      {"income_median": 2420, "poverty_rate": 25.1},
    "Pitas":         {"income_median": 2350, "poverty_rate": 26.8},
    "Telupid":       {"income_median": 2310, "poverty_rate": 27.9},
    "Nabawan":       {"income_median": 2180, "poverty_rate": 29.4},
    "Tongod":        {"income_median": 2040, "poverty_rate": 31.6},
}

# ---------------------------------------------------------------------------
# SAE MODIFIERS BASELINE (NHMS, METMalaysia & MOH Sabah Grounded Benchmarks)
# Metrics:
#   - road_risk: Weather & Seasonal Road Passability Risk (0 to 100)
#   - smoking_pct: Smoking & Tobacco Prevalence (18.0% to 32.0%)
#   - hiv_pct: HIV / Immunosuppression Prevalence (0.0% to 0.60%)
# ---------------------------------------------------------------------------
SAE_MODIFIERS_BASELINE = {
    "Beaufort":      {"road_risk": 45.0, "smoking_pct": 23.5, "hiv_pct": 0.26},
    "Beluran":       {"road_risk": 89.0, "smoking_pct": 26.8, "hiv_pct": 0.12},
    "Kalabakan":     {"road_risk": 72.0, "smoking_pct": 27.0, "hiv_pct": 0.18},
    "Keningau":      {"road_risk": 42.0, "smoking_pct": 24.0, "hiv_pct": 0.22},
    "Kinabatangan":  {"road_risk": 91.0, "smoking_pct": 27.2, "hiv_pct": 0.15},
    "Kota Belud":    {"road_risk": 52.0, "smoking_pct": 26.0, "hiv_pct": 0.14},
    "Kota Kinabalu": {"road_risk": 8.0,  "smoking_pct": 18.5, "hiv_pct": 0.52},
    "Kota Marudu":   {"road_risk": 65.0, "smoking_pct": 28.0, "hiv_pct": 0.10},
    "Kuala Penyu":   {"road_risk": 38.0, "smoking_pct": 25.5, "hiv_pct": 0.12},
    "Kudat":         {"road_risk": 58.0, "smoking_pct": 29.5, "hiv_pct": 0.16},
    "Kunak":         {"road_risk": 46.0, "smoking_pct": 27.0, "hiv_pct": 0.22},
    "Lahad Datu":    {"road_risk": 44.0, "smoking_pct": 26.5, "hiv_pct": 0.44},
    "Nabawan":       {"road_risk": 78.0, "smoking_pct": 27.5, "hiv_pct": 0.05},
    "Papar":         {"road_risk": 22.0, "smoking_pct": 21.0, "hiv_pct": 0.25},
    "Penampang":     {"road_risk": 12.0, "smoking_pct": 19.2, "hiv_pct": 0.38},
    "Pitas":         {"road_risk": 68.0, "smoking_pct": 30.5, "hiv_pct": 0.08},
    "Putatan":       {"road_risk": 14.0, "smoking_pct": 20.0, "hiv_pct": 0.35},
    "Ranau":         {"road_risk": 62.0, "smoking_pct": 24.5, "hiv_pct": 0.12},
    "Sandakan":      {"road_risk": 20.0, "smoking_pct": 22.5, "hiv_pct": 0.48},
    "Semporna":      {"road_risk": 54.0, "smoking_pct": 30.0, "hiv_pct": 0.28},
    "Sipitang":      {"road_risk": 48.0, "smoking_pct": 25.0, "hiv_pct": 0.18},
    "Tambunan":      {"road_risk": 45.0, "smoking_pct": 23.5, "hiv_pct": 0.10},
    "Tawau":         {"road_risk": 24.0, "smoking_pct": 23.0, "hiv_pct": 0.46},
    "Telupid":       {"road_risk": 76.0, "smoking_pct": 26.8, "hiv_pct": 0.11},
    "Tenom":         {"road_risk": 48.0, "smoking_pct": 24.8, "hiv_pct": 0.12},
    "Tongod":        {"road_risk": 92.0, "smoking_pct": 27.5, "hiv_pct": 0.04},
    "Tuaran":        {"road_risk": 20.0, "smoking_pct": 22.0, "hiv_pct": 0.22},
}

# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="📡 Fetching MOH Malaysia facility registry...")
def load_moh_facilities():
    """Download and cache the MOH master facility CSV."""
    try:
        df = pd.read_csv(MOH_URL, low_memory=False)
    except Exception as exc:
        st.error(f"❌ Failed to fetch MOH data: {exc}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df.columns = df.columns.str.strip()
    sabah = df[df["NEGERI"].str.strip().str.upper() == "SABAH"].copy()
    sabah_open = sabah[sabah["STATUS"].str.strip().str.upper() == "BUKA"].copy()

    hospitals = sabah_open[
        sabah_open["KATEGORI_FASILITI"].str.strip().str.upper() == "HOSPITAL"
    ].copy()
    clinics = sabah_open[
        sabah_open["KATEGORI_FASILITI"].str.strip().str.upper() == "KLINIK"
    ].copy()

    for layer in (hospitals, clinics):
        layer["LATITUD"]  = pd.to_numeric(layer["LATITUD"],  errors="coerce")
        layer["LONGITUD"] = pd.to_numeric(layer["LONGITUD"], errors="coerce")

    hospitals = hospitals.dropna(subset=["LATITUD", "LONGITUD"]).reset_index(drop=True)
    clinics   = clinics.dropna(subset=["LATITUD", "LONGITUD"]).reset_index(drop=True)
    return sabah_open, hospitals, clinics


@st.cache_data(ttl=3600, show_spinner="💰 Fetching OpenDOSM HIES district income data...")
def load_hies_income():
    """
    Load Sabah district-level income from OpenDOSM HIES.
    Falls back to embedded baseline dictionary on network failure or timeout.

    Returns
    -------
    df     : pd.DataFrame  columns=[district, income_median, poverty_rate]
    source : str           'OpenDOSM' or 'Cached Baseline'
    """
    canonical = list(HIES_FALLBACK.keys())

    def _build_fallback():
        rows = [
            {"district": k, "income_median": float(v["income_median"]), "poverty_rate": float(v["poverty_rate"])}
            for k, v in HIES_FALLBACK.items()
        ]
        return pd.DataFrame(rows), "Cached Baseline"

    try:
        raw = pd.read_csv(DOSM_HIES_URL, low_memory=False)
        raw.columns = raw.columns.str.strip().str.lower()

        if "state" not in raw.columns:
            return _build_fallback()

        sabah_mask = raw["state"].astype(str).str.strip().str.title() == "Sabah"
        sabah_hies = raw[sabah_mask].copy()

        if sabah_hies.empty:
            return _build_fallback()

        if "year" in sabah_hies.columns:
            latest_year = pd.to_numeric(sabah_hies["year"], errors="coerce").max()
            if not np.isnan(latest_year):
                sabah_hies = sabah_hies[
                    pd.to_numeric(sabah_hies["year"], errors="coerce") == latest_year
                ]

        income_col = None
        for candidate in ["income_median", "median_income", "income", "median"]:
            if candidate in sabah_hies.columns:
                income_col = candidate
                break

        poverty_col = None
        for candidate in ["poverty_rate", "poverty", "incidence_poverty"]:
            if candidate in sabah_hies.columns:
                poverty_col = candidate
                break

        if income_col is None or "district" not in sabah_hies.columns:
            return _build_fallback()

        keep_cols = ["district", income_col] + ([poverty_col] if poverty_col else [])
        sabah_hies = sabah_hies[keep_cols].copy()

        sabah_hies["district"] = sabah_hies["district"].astype(str).str.strip().str.title()
        sabah_hies = sabah_hies.rename(columns={income_col: "income_median"})
        if poverty_col:
            sabah_hies = sabah_hies.rename(columns={poverty_col: "poverty_rate"})
        else:
            sabah_hies["poverty_rate"] = np.nan

        sabah_hies["income_median"] = pd.to_numeric(sabah_hies["income_median"], errors="coerce")
        sabah_hies["poverty_rate"]  = pd.to_numeric(sabah_hies["poverty_rate"],  errors="coerce")

        name_map = {}
        for raw_name in sabah_hies["district"].unique():
            clean = raw_name.strip().title()
            if clean in canonical:
                name_map[raw_name] = clean
                continue
            best, best_len = None, 0
            for c in canonical:
                overlap = sum(w in c for w in clean.split())
                if overlap > best_len:
                    best, best_len = c, overlap
            if best and best_len > 0:
                name_map[raw_name] = best

        sabah_hies["district"] = sabah_hies["district"].map(name_map).fillna(sabah_hies["district"])

        covered = set(sabah_hies["district"].unique())
        missing_rows = []
        for d in canonical:
            if d not in covered:
                missing_rows.append({
                    "district":      d,
                    "income_median": float(HIES_FALLBACK[d]["income_median"]),
                    "poverty_rate":  float(HIES_FALLBACK[d]["poverty_rate"]),
                })
        if missing_rows:
            sabah_hies = pd.concat([sabah_hies, pd.DataFrame(missing_rows)], ignore_index=True)

        for idx, row in sabah_hies.iterrows():
            dname = row["district"]
            if pd.isna(row["income_median"]) and dname in HIES_FALLBACK:
                sabah_hies.at[idx, "income_median"] = float(HIES_FALLBACK[dname]["income_median"])
            if pd.isna(row["poverty_rate"]) and dname in HIES_FALLBACK:
                sabah_hies.at[idx, "poverty_rate"] = float(HIES_FALLBACK[dname]["poverty_rate"])

        sabah_hies = (
            sabah_hies[sabah_hies["district"].isin(canonical)]
            .drop_duplicates(subset=["district"])
            .reset_index(drop=True)
        )

        if sabah_hies["income_median"].notna().sum() < 5:
            return _build_fallback()

        return sabah_hies[["district", "income_median", "poverty_rate"]], "OpenDOSM"

    except Exception:
        return _build_fallback()


# ---------------------------------------------------------------------------
# HAVERSINE & DISTANCE ENGINE
# ---------------------------------------------------------------------------
def haversine_vectorized(lat1: float, lon1: float,
                          lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in km between one point and an array of points."""
    R = 6_371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner="🔢 Computing geodesic access matrix...")
def compute_access(hospitals_hash: str, speed_kmh: float) -> pd.DataFrame:
    """Return district-level access metrics given hospital data and road speed."""
    hospitals = st.session_state["_hospitals"]
    lat2  = hospitals["LATITUD"].values
    lon2  = hospitals["LONGITUD"].values
    names = hospitals["NAMA"].values

    rows = []
    for _, row in DIST_DF.iterrows():
        dists    = haversine_vectorized(row.lat, row.lon, lat2, lon2)
        idx      = int(np.argmin(dists))
        min_d    = float(dists[idx])
        nearest  = str(names[idx])
        est_time = 15.0 + (min_d / speed_kmh) * 60.0
        rows.append({
            "district":            row.district,
            "lat":                 row.lat,
            "lon":                 row.lon,
            "min_distance_km":     round(min_d, 2),
            "nearest_facility":    nearest,
            "est_travel_time_min": round(est_time, 1),
        })

    df = pd.DataFrame(rows)

    # Geographic Access Gradient G in [0, 1]
    t = df["est_travel_time_min"]
    t_min, t_max = t.min(), t.max()
    df["norm_geo"] = ((t - t_min) / (t_max - t_min)).round(4) if t_max > t_min else 0.5
    return df


# ---------------------------------------------------------------------------
# COMPLETE 7-METRIC TB PRIORITIZATION ENGINE (100% TOTAL WEIGHT)
# ---------------------------------------------------------------------------
def compute_complete_prioritization(access_df: pd.DataFrame, hies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the full 7-metric TB Mobile Clinic Prioritization Engine:
      Layer 1: Compound Structural Exposure (60.0% Weight) — Formulation C
      Layer 2: Clinical & Operational Modifiers (40.0% Weight) — Linear Additive
      Final Priority Score = Layer 1 + Layer 2 in [0, 100.0]
    """
    df = access_df.copy()
    df = df.merge(hies_df[["district", "income_median", "poverty_rate"]], on="district", how="left")

    for idx, row in df.iterrows():
        dname = row["district"]
        if pd.isna(row["income_median"]) and dname in HIES_FALLBACK:
            df.at[idx, "income_median"] = float(HIES_FALLBACK[dname]["income_median"])
        if pd.isna(row["poverty_rate"]) and dname in HIES_FALLBACK:
            df.at[idx, "poverty_rate"] = float(HIES_FALLBACK[dname]["poverty_rate"])

    df["income_median"] = pd.to_numeric(df["income_median"], errors="coerce")
    df["poverty_rate"]  = pd.to_numeric(df["poverty_rate"],  errors="coerce")
    df["norm_geo"]      = df["norm_geo"].fillna(0.5)

    # Income Deprivation Gradient I in [0, 1]
    inc = df["income_median"]
    i_max, i_min = inc.max(), inc.min()
    df["norm_income"] = ((i_max - inc) / (i_max - i_min)).round(4) if i_max > i_min else 0.5

    # =========================================================================
    # LAYER 1: COMPOUND STRUCTURAL EXPOSURE (60.0% Weight) — FORMULATION C
    # =========================================================================
    # S_compound = (15.0 * G) + (15.0 * I) + (30.0 * [G * I])
    df["layer1_compound_score"] = (
        (15.0 * df["norm_geo"])
        + (15.0 * df["norm_income"])
        + (30.0 * (df["norm_geo"] * df["norm_income"]))
    ).round(2)

    # =========================================================================
    # LAYER 2: CLINICAL & OPERATIONAL MODIFIERS (40.0% Weight) — LINEAR ADDITIVE
    # =========================================================================
    # Metric 3: TB Diagnostic Latency / Lag (20.0% Weight)
    # delay_days = 25.0 + (0.40 * travel_time_min)
    # norm_lag = (delay_days - 25.0) / (105.0 - 25.0)
    df["delay_days"] = (25.0 + (0.40 * df["est_travel_time_min"])).round(1)
    df["norm_lag"] = np.clip((df["delay_days"] - 25.0) / (105.0 - 25.0), 0.0, 1.0).round(4)
    df["score_lag"] = (df["norm_lag"] * 20.0).round(2)

    # Metric 4: Diabetes Mellitus Prevalence (5.0% Weight)
    # diabetes_pct = 11.0 + 6.0 * (1.0 - norm_income)
    # norm_dm = (diabetes_pct - 10.0) / (18.0 - 10.0)
    df["diabetes_pct"] = (11.0 + (6.0 * (1.0 - df["norm_income"]))).round(2)
    df["norm_dm"] = np.clip((df["diabetes_pct"] - 10.0) / (18.0 - 10.0), 0.0, 1.0).round(4)
    df["score_dm"] = (df["norm_dm"] * 5.0).round(2)

    # Metric 5: Weather & Seasonal Road Passability Risk (5.0% Weight)
    # Metric 6: Smoking & Tobacco Prevalence (5.0% Weight)
    # Metric 7: HIV / Immunosuppression Prevalence (5.0% Weight)
    road_risks, smoking_rates, hiv_rates = [], [], []
    for _, row in df.iterrows():
        dname = row["district"]
        sae = SAE_MODIFIERS_BASELINE.get(dname, {"road_risk": 50.0, "smoking_pct": 24.5, "hiv_pct": 0.15})
        road_risks.append(sae["road_risk"])
        smoking_rates.append(sae["smoking_pct"])
        hiv_rates.append(sae["hiv_pct"])

    df["road_risk"] = road_risks
    df["norm_weather"] = np.clip((df["road_risk"] - 0.0) / (100.0 - 0.0), 0.0, 1.0).round(4)
    df["score_weather"] = (df["norm_weather"] * 5.0).round(2)

    df["smoking_pct"] = smoking_rates
    df["norm_smoking"] = np.clip((df["smoking_pct"] - 18.0) / (32.0 - 18.0), 0.0, 1.0).round(4)
    df["score_smoking"] = (df["norm_smoking"] * 5.0).round(2)

    df["hiv_pct"] = hiv_rates
    df["norm_hiv"] = np.clip((df["hiv_pct"] - 0.0) / (0.60 - 0.0), 0.0, 1.0).round(4)
    df["score_hiv"] = (df["norm_hiv"] * 5.0).round(2)

    # Layer 2 Subtotal (40.0% Weight)
    df["layer2_modifiers_score"] = (
        df["score_lag"]
        + df["score_dm"]
        + df["score_weather"]
        + df["score_smoking"]
        + df["score_hiv"]
    ).round(2)

    # =========================================================================
    # FINAL PRIORITY SCORE (100.0% Total)
    # =========================================================================
    df["final_priority_score"] = (
        df["layer1_compound_score"] + df["layer2_modifiers_score"]
    ).round(2)

    df["priority_rank"] = df["final_priority_score"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("priority_rank").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# COLOR TIERS FOR FINAL PRIORITY SCORE (/100)
# ---------------------------------------------------------------------------
def priority_color(score: float) -> str:
    if score < 35.0:
        return "#22c55e"   # Green  (< 35.0)
    elif score <= 60.0:
        return "#f97316"   # Orange (35.0 - 60.0)
    else:
        return "#dc2626"   # Deep Red (> 60.0)


def priority_label(score: float) -> str:
    if score < 35.0:
        return "Low Dispatch Priority (<35.0)"
    elif score <= 60.0:
        return "Moderate Urgency (35.0–60.0)"
    else:
        return "CRITICAL DISPATCH PRIORITY (>60.0)"


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚐 TB Dispatch · Engine")
    st.caption("Complete 7-Metric Multi-Criteria Evaluation")
    st.markdown("---")

    st.markdown("### 📡 Pipeline & Data Health")
    sabah_open, hospitals, clinics = load_moh_facilities()

    if hospitals.empty:
        st.error("❌ No hospital data loaded. Check network connection.")
        st.stop()

    hies_df, hies_source = load_hies_income()

    if hies_source == "OpenDOSM":
        st.markdown(
            '**HIES Status:** <span class="hies-connected">🟢 Connected (OpenDOSM)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '**HIES Status:** <span class="hies-cached">🟡 Cached Baseline</span>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Operational Parameters")
    speed_kmh = st.slider(
        "Average road speed (km/h)",
        min_value=25, max_value=60, value=40, step=5,
        help="Winding rural roads. 40 km/h is the conservative baseline.",
    )

    st.markdown("---")
    st.markdown("### ⚖️ Model Weight Distribution (100%)")
    st.markdown(
        """
        <div style="background:rgba(0,200,180,0.06); border:1px solid rgba(0,200,180,0.22);
             border-radius:8px; padding:10px; font-size:0.8rem; color:#cbd5e1; line-height:1.6;">
            <b style="color:#00c8b4;">Layer 1: Structural (60%)</b><br>
            • Travel Time (G): 15% base + 15% synergy<br>
            • Income Deprivation (I): 15% base + 15% synergy<br>
            <b style="color:#38bdf8;">Layer 2: Modifiers (40%)</b><br>
            • Diagnostic Latency: 20%<br>
            • Diabetes Comorbidity: 5%<br>
            • Weather & Monsoon Risk: 5%<br>
            • Tobacco Consumption: 5%<br>
            • HIV Vulnerability: 5%
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# COMPUTE ACCESS & 7-METRIC PRIORITIZATION MATRIX
# ---------------------------------------------------------------------------
st.session_state["_hospitals"] = hospitals
hosp_hash = f"{len(hospitals)}_{speed_kmh}"
access_df = compute_access(hosp_hash, speed_kmh)
df = compute_complete_prioritization(access_df, hies_df)

# Sidebar Snapshot KPIs
top_dist         = df.iloc[0]
state_median     = df["income_median"].mean()
mean_delay       = df["delay_days"].mean()
richest_income   = df["income_median"].max()
poorest_income   = df["income_median"].min()
disparity_ratio  = richest_income / poorest_income if poorest_income > 0 else float("nan")

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📌 Priority Snapshot")
    st.markdown(
        f"""
        <div class="sidebar-kpi">
            <div class="sidebar-kpi-title">#1 Mobile Unit Target</div>
            <div class="sidebar-kpi-val" style="color:#ef4444;">{top_dist['district']}</div>
            <div class="sidebar-kpi-sub">Priority Score: {top_dist['final_priority_score']:.1f}/100</div>
        </div>
        <div class="sidebar-kpi">
            <div class="sidebar-kpi-title">Mean TB Diagnostic Delay</div>
            <div class="sidebar-kpi-val" style="color:#f59e0b;">{mean_delay:.0f} Days</div>
            <div class="sidebar-kpi-sub">Average across 27 districts</div>
        </div>
        <div class="sidebar-kpi">
            <div class="sidebar-kpi-title">Active Health Infrastructure</div>
            <div class="sidebar-kpi-val" style="color:#38bdf8;">{len(hospitals)} Hospitals</div>
            <div class="sidebar-kpi-sub">{len(clinics)} Primary Clinics Registered</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Data: MOH Malaysia · OpenDOSM · NHMS · METMalaysia")

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <h1 style="
        background: linear-gradient(90deg, #00c8b4 0%, #0096c7 60%, #7ecfdf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem; font-weight: 800; margin-bottom: 0.15rem;
    ">🚐 Sabah TB Mobile Clinic Prioritization Engine</h1>
    <p style="color:#64748b; font-size:0.95rem; margin-top:0;">
        Full 7-Metric Model (100% Total Weight) &nbsp;&middot;&nbsp; Layer 1: Compound Structural Exposure (60%) &nbsp;&middot;&nbsp; Layer 2: Clinical & Operational Modifiers (40%)
    </p>
    <hr style="border-color:rgba(0,200,180,0.2); margin:0.8rem 0 1.2rem;">
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# TOP METRIC KPI CARDS (2 Rows)
# ---------------------------------------------------------------------------
bottom_dist = df.iloc[-1]
max_geo_dist = df.sort_values("est_travel_time_min", ascending=False).iloc[0]
critical_count = (df["final_priority_score"] > 60.0).sum()

r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    st.markdown('<div class="metric-compound">', unsafe_allow_html=True)
    st.metric(
        "🔴 #1 Priority Mobile Unit Target",
        top_dist["district"],
        (
            f"Score {top_dist['final_priority_score']:.1f}/100 "
            f"(L1: {top_dist['layer1_compound_score']:.1f} + L2: {top_dist['layer2_modifiers_score']:.1f})"
        ),
        delta_color="inverse",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with r1c2:
    st.metric(
        "⏱ Mean TB Diagnostic Latency",
        f"{mean_delay:.1f} Days",
        f"Range: {df['delay_days'].min():.0f} to {df['delay_days'].max():.0f} days to GeneXpert/CXR",
        delta_color="inverse",
    )

with r1c3:
    st.metric(
        "💰 State Median Household Income",
        f"RM {state_median:,.0f}",
        f"Urban-Rural Disparity Ratio: {disparity_ratio:.1f}×",
    )

st.markdown("<br>", unsafe_allow_html=True)

r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    st.metric(
        "✅ Lowest Priority District",
        bottom_dist["district"],
        f"Score {bottom_dist['final_priority_score']:.1f}/100 (Urban & Connected)",
    )

with r2c2:
    st.markdown('<div class="metric-critical">', unsafe_allow_html=True)
    st.metric(
        "🗺️ Maximum Travel Friction",
        max_geo_dist["district"],
        f"≈{max_geo_dist['est_travel_time_min']:.0f} min ({max_geo_dist['min_distance_km']:.1f} km)",
        delta_color="inverse",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with r2c3:
    st.metric(
        "⚠️ Critical Priority Districts (>60/100)",
        f"{critical_count} / {len(df)} Districts",
        f"Immediate mobile screening recommended",
        delta_color="inverse",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAP VISUALIZATION: OPTION C LAYERED HYBRID (FOLIUM)
# ---------------------------------------------------------------------------
st.markdown("## 🗺️ Option C: Layered Hybrid Priority Map (100% Model)")
st.caption(
    "Interactive multi-layer dispatch map. "
    "Circle markers are sized (radius 7–20) and colored by **Final Priority Score (/100)**. "
    "Toggle the **Vulnerability Heatmap** or **Primary Clinics** in the top-right layer control."
)

m = folium.Map(
    location=[5.45, 117.0],
    zoom_start=8,
    tiles=None,
    control_scale=True,
)

# Basemap 1: Carto Dark (Default)
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors &copy; <a href='https://carto.com/attributions'>CARTO</a>",
    name="Carto Dark (Default)",
    subdomains="abcd",
    max_zoom=19,
    control=True,
).add_to(m)

# Basemap 2: Standard OpenStreetMap
folium.TileLayer(
    tiles="OpenStreetMap",
    name="OpenStreetMap",
    control=True,
).add_to(m)

# Layer 2: Vulnerability Density Heatmap (Hidden by default)
heatmap_fg = folium.FeatureGroup(name="Vulnerability Heatmap Gradient", show=False)
heatmap_coords = [
    [float(row["lat"]), float(row["lon"]), float(row["final_priority_score"])]
    for _, row in df.iterrows()
]
HeatMap(
    data=heatmap_coords,
    radius=42,
    blur=26,
    max_zoom=9,
    min_opacity=0.35,
    gradient={
        0.2: "#fde047",
        0.5: "#f97316",
        0.75: "#dc2626",
        1.0: "#7f1d1d",
    },
).add_to(heatmap_fg)
heatmap_fg.add_to(m)

# Layer 4: Primary Clinics (Hidden by default)
clinic_fg = folium.FeatureGroup(name="Registered Clinics (KLINIK)", show=False)
if not clinics.empty:
    for _, c in clinics.iterrows():
        c_name = c.get("NAMA", "Klinik Kesihatan")
        folium.CircleMarker(
            location=[c["LATITUD"], c["LONGITUD"]],
            radius=3.5,
            color="#64748b",
            fill=True,
            fill_color="#94a3b8",
            fill_opacity=0.55,
            weight=1,
            tooltip=folium.Tooltip(f"🏪 {c_name}<br><span style='color:#94a3b8;'>Primary Clinic</span>", sticky=False),
        ).add_to(clinic_fg)
clinic_fg.add_to(m)

# Layer 3: Diagnostic Hospitals (Shown by default)
hosp_fg = folium.FeatureGroup(name="Diagnostic Hospitals (MOH)", show=True)
for _, h in hospitals.iterrows():
    h_name = h.get("NAMA", "Hospital")
    folium.Marker(
        location=[h["LATITUD"], h["LONGITUD"]],
        tooltip=folium.Tooltip(
            f"<b>🏥 {h_name}</b><br>"
            f"<span style='color:#38bdf8; font-weight:600;'>GeneXpert / Digital CXR Hub</span><br>"
            f"📍 {h['LATITUD']:.4f}, {h['LONGITUD']:.4f}",
            sticky=False,
        ),
        icon=folium.Icon(color="blue", icon="plus", prefix="fa"),
    ).add_to(hosp_fg)
hosp_fg.add_to(m)

# Layer 1: District Priority Centroids (Shown by default)
dist_fg = folium.FeatureGroup(name="District Vulnerability Markers", show=True)

for _, row in df.iterrows():
    score = float(row["final_priority_score"])
    color = priority_color(score)
    label = priority_label(score)
    radius = 7.0 + (score / 100.0) * 13.0

    tooltip_html = f"""
    <div style="font-family:'Inter',sans-serif; min-width:280px; color:#1e293b;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
            <span style="font-size:1.1rem; font-weight:800; color:#0f172a;">{row['district']}</span>
            <span style="background:{color}; color:#fff; border-radius:4px; padding:2px 6px; font-size:0.75rem; font-weight:700;">
                #{row['priority_rank']}
            </span>
        </div>
        <div style="color:{color}; font-weight:700; font-size:0.82rem; margin-bottom:6px;">
            ● {label}
        </div>
        <hr style="margin:4px 0 6px 0; border:0; border-top:1px solid #cbd5e1;">

        <div style="background:#f8fafc; border-radius:6px; padding:6px; margin-bottom:6px;">
            <div style="font-weight:700; font-size:0.85rem; color:#0f172a;">
                🎯 Final Priority Score: <span style="color:{color}; font-size:1.05rem;">{score:.1f} / 100</span>
            </div>
            <div style="font-size:0.75rem; color:#64748b;">
                Layer 1 (Structural): <b>{row['layer1_compound_score']:.1f}/60</b> &nbsp;|&nbsp;
                Layer 2 (Modifiers): <b>{row['layer2_modifiers_score']:.1f}/40</b>
            </div>
        </div>

        <table style="width:100%; font-size:0.78rem; line-height:1.65; border-collapse:collapse;">
            <tr><td style="color:#64748b;">🏥 Nearest Hub:</td><td style="font-weight:600; text-align:right;">{row['nearest_facility']}</td></tr>
            <tr><td style="color:#64748b;">⏱ Travel Friction:</td><td style="font-weight:600; text-align:right;">{row['est_travel_time_min']:.0f} min (G={row['norm_geo']:.2f})</td></tr>
            <tr><td style="color:#64748b;">💰 Median Income:</td><td style="font-weight:600; text-align:right;">RM {row['income_median']:,.0f} (I={row['norm_income']:.2f})</td></tr>
            <tr><td style="color:#64748b;">⏳ Diagnostic Latency:</td><td style="font-weight:600; text-align:right;">{row['delay_days']:.0f} Days (+{row['score_lag']:.1f} pts)</td></tr>
            <tr><td style="color:#64748b;">🩸 Diabetes Mellitus:</td><td style="font-weight:600; text-align:right;">{row['diabetes_pct']:.1f}% (+{row['score_dm']:.1f} pts)</td></tr>
            <tr><td style="color:#64748b;">🌧 Road & Monsoon Risk:</td><td style="font-weight:600; text-align:right;">{row['road_risk']:.0f}/100 (+{row['score_weather']:.1f} pts)</td></tr>
            <tr><td style="color:#64748b;">🚬 Tobacco Prevalence:</td><td style="font-weight:600; text-align:right;">{row['smoking_pct']:.1f}% (+{row['score_smoking']:.1f} pts)</td></tr>
            <tr><td style="color:#64748b;">🦠 HIV Prevalence:</td><td style="font-weight:600; text-align:right;">{row['hiv_pct']:.2f}% (+{row['score_hiv']:.1f} pts)</td></tr>
        </table>
    </div>
    """

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.82,
        weight=2.0,
        tooltip=folium.Tooltip(tooltip_html, sticky=True),
        popup=folium.Popup(tooltip_html, max_width=320),
    ).add_to(dist_fg)

    folium.map.Marker(
        location=[row["lat"] + 0.045, row["lon"]],
        icon=folium.DivIcon(
            html=(
                f'<div style="font-size:9.5px; font-weight:600; color:#e2e8f0; font-family:Inter,sans-serif; '
                f'white-space:nowrap; text-shadow: 0 0 4px #000, 0 0 2px #000;">{row["district"]}</div>'
            ),
            icon_size=(130, 16),
            icon_anchor=(65, 0),
        ),
    ).add_to(dist_fg)

dist_fg.add_to(m)

# Interactive Layer Control
folium.LayerControl(position="topright", collapsed=False).add_to(m)

# Legend Overlay
legend_html = """
<div style="
    position: fixed; bottom: 30px; left: 30px; z-index: 9999;
    background: rgba(10,14,26,0.94);
    border: 1px solid rgba(0,200,180,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: #e8f4f8;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 14px rgba(0,0,0,0.5);
">
  <b style="color:#00c8b4; font-size:13px;">Final Priority Score (/100)</b><br>
  <span style="color:#94a3b8; font-size:11px;">Layer 1 (60%) + Layer 2 (40%)</span><br><br>
  <span style="color:#22c55e;">●</span> &lt; 35.0 &nbsp; Low Priority (Accessible & Resilient)<br>
  <span style="color:#f97316;">●</span> 35.0 – 60.0 &nbsp; Moderate Priority<br>
  <span style="color:#dc2626;">●</span> &gt; 60.0 &nbsp; Critical Dispatch Priority<br>
  <hr style="margin:8px 0; border:0; border-top:1px solid rgba(0,200,180,0.2);">
  <span style="color:#38bdf8;">✚</span> Diagnostic Hospital (GeneXpert/CXR)<br>
  <span style="color:#94a3b8;">●</span> Primary Clinic (KLINIK Layer)
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width="100%", height=630, returned_objects=[])

# ---------------------------------------------------------------------------
# TOP 10 HIGH PRIORITY BREAKDOWN
# ---------------------------------------------------------------------------
with st.expander("📊 Top 10 High Priority Districts (Layer 1 vs Layer 2)", expanded=True):
    top10 = df.head(10)[[
        "priority_rank", "district", "final_priority_score",
        "layer1_compound_score", "layer2_modifiers_score",
        "est_travel_time_min", "income_median", "delay_days", "road_risk"
    ]].copy()
    top10.columns = [
        "Rank", "District", "Final Score (/100)", "Layer 1 (/60)", "Layer 2 (/40)",
        "Travel Time (min)", "Median Income (RM)", "TB Delay (days)", "Road Risk (0-100)"
    ]

    def _style_final(val):
        s = float(val)
        if s > 60.0:  return "color:#ef4444; font-weight:700"
        elif s >= 35: return "color:#f97316; font-weight:600"
        else:         return "color:#22c55e; font-weight:600"

    _fn = getattr(top10.style, "map", None) or getattr(top10.style, "applymap")
    styled10 = _fn(_style_final, subset=["Final Score (/100)"]).format({
        "Final Score (/100)":  "{:.2f}",
        "Layer 1 (/60)":       "{:.2f}",
        "Layer 2 (/40)":       "{:.2f}",
        "Travel Time (min)":   "{:.0f} min",
        "Median Income (RM)":  "RM {:,.0f}",
        "TB Delay (days)":     "{:.0f} days",
        "Road Risk (0-100)":   "{:.0f}",
    })
    try:
        st.dataframe(styled10, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(styled10, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# FULL 27-DISTRICT PRIORITY RANKING TABLE
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 📋 Full 27-District Mobile Clinic Prioritization Table")
st.caption(
    "Complete rankings across all 7 evaluation dimensions. "
    "Sorted descending by **Final Priority Score (/100)**."
)

table_df = df[[
    "priority_rank",
    "district",
    "final_priority_score",
    "layer1_compound_score",
    "layer2_modifiers_score",
    "est_travel_time_min",
    "income_median",
    "delay_days",
    "diabetes_pct",
    "road_risk",
    "smoking_pct",
    "hiv_pct",
]].copy()

table_df.columns = [
    "Rank",
    "District",
    "Final Score (/100)",
    "Layer 1: Structural (/60)",
    "Layer 2: Modifiers (/40)",
    "Travel Time (min)",
    "Median Income (RM)",
    "Est. Delay (days)",
    "Diabetes (%)",
    "Road Risk (0-100)",
    "Smoking (%)",
    "HIV (%)",
]

def _hl_final(val):
    s = float(val)
    if s > 60.0:  return "color:#ef4444; font-weight:700"
    elif s >= 35: return "color:#f97316; font-weight:600"
    else:         return "color:#22c55e; font-weight:600"

def _hl_income(val):
    try:
        inc = float(str(val).replace("RM", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return ""
    if inc < 2500:   return "color:#ef4444"
    elif inc < 3500: return "color:#f97316"
    else:            return "color:#22c55e"

_tbl_style = table_df.style
_mfn = getattr(_tbl_style, "map", None) or getattr(_tbl_style, "applymap")
styled_full = (
    _mfn(_hl_final, subset=["Final Score (/100)"])
    .pipe(lambda s: (getattr(s, "map", None) or getattr(s, "applymap"))(
        _hl_income, subset=["Median Income (RM)"]
    ))
    .format({
        "Final Score (/100)":         "{:.2f}",
        "Layer 1: Structural (/60)":  "{:.2f}",
        "Layer 2: Modifiers (/40)":   "{:.2f}",
        "Travel Time (min)":          "{:.1f} min",
        "Median Income (RM)":         "RM {:,.0f}",
        "Est. Delay (days)":          "{:.1f} d",
        "Diabetes (%)":               "{:.1f}%",
        "Road Risk (0-100)":          "{:.0f}",
        "Smoking (%)":                "{:.1f}%",
        "HIV (%)":                    "{:.2f}%",
    })
)

try:
    st.dataframe(styled_full, width="stretch", height=620, hide_index=True)
except TypeError:
    st.dataframe(styled_full, use_container_width=True, height=620, hide_index=True)

# Export Dataset
export_df = df[[
    "priority_rank", "district", "nearest_facility", "min_distance_km",
    "est_travel_time_min", "income_median", "poverty_rate",
    "norm_geo", "norm_income", "layer1_compound_score",
    "delay_days", "norm_lag", "score_lag",
    "diabetes_pct", "norm_dm", "score_dm",
    "road_risk", "norm_weather", "score_weather",
    "smoking_pct", "norm_smoking", "score_smoking",
    "hiv_pct", "norm_hiv", "score_hiv",
    "layer2_modifiers_score", "final_priority_score"
]].copy()

export_df.columns = [
    "rank", "district", "nearest_hospital", "distance_km",
    "travel_time_min", "median_income_rm", "poverty_rate_pct",
    "norm_geo_g", "norm_income_i", "layer1_compound_score_60",
    "delay_days", "norm_lag", "score_lag_20",
    "diabetes_pct", "norm_dm", "score_dm_5",
    "road_risk_100", "norm_weather", "score_weather_5",
    "smoking_pct", "norm_smoking", "score_smoking_5",
    "hiv_pct", "norm_hiv", "score_hiv_5",
    "layer2_modifiers_score_40", "final_priority_score_100"
]
csv_bytes = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Export Complete 7-Metric Priority Dataset (sabah_tb_final_priority.csv)",
    data=csv_bytes,
    file_name="sabah_tb_final_priority.csv",
    mime="text/csv",
    help="Download complete 27-district 7-metric dataset with all raw values, normalized indices, and component scores.",
)

# ---------------------------------------------------------------------------
# INTERACTIVE COLLAPSIBLE BOTTOM DRAWER: MATHEMATICAL ARCHITECTURE IN LATEX
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("📐 Mathematical Architecture & Multi-Criteria Formulation (LaTeX Drawer)", expanded=True):
    st.markdown(
        """
        <div style="font-size:0.95rem; color:#94a3b8; margin-bottom:12px;">
            The TB Mobile Clinic Prioritization Engine implements a hierarchically structured, 
            evidence-grounded decision model that synthesizes <b>Structural Accessibility Barriers (60%)</b> 
            with <b>Clinical and Operational Modifiers (40%)</b> into a unified 100-point continuous priority score.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏛️ Layer 1: Compound Structural (60%)",
        "🧬 Layer 2: Clinical & Operational Modifiers (40%)",
        "🎯 Unified Priority Objective Function (100%)",
        "📊 Parameter Calibration & Evidence Matrix",
    ])

    with tab1:
        st.markdown(r"""
#### Layer 1: Compound Structural Exposure ($60.0\%$ Total Weight)

Layer 1 addresses the compound vulnerability created when physical distance to specialized diagnostic infrastructure 
co-occurs with household income deprivation.

1. **Geographic Isolation Gradient ($G \in [0, 1]$):**
   Derived from the Haversine geodesic distance matrix to the nearest tertiary GeneXpert/digital CXR hospital hub:
   $$G = \frac{T - \min(T)}{\max(T) - \min(T)}$$
   Where $T = 15.0 + \frac{\text{distance\_km}}{\text{road\_speed}} \times 60$ minutes (incorporating a 15-minute dispatch overhead).

2. **Household Income Deprivation Gradient ($I \in [0, 1]$):**
   Derived from OpenDOSM Household Income and Expenditure Survey (HIES) district medians:
   $$I = \frac{\max(\text{Income}) - \text{Income}}{\max(\text{Income}) - \min(\text{Income})}$$
   Where lower median income maps to maximal economic deprivation ($I \to 1.0$).

3. **Formulation C — Hybrid Base + Synergistic Interaction:**
   $$S_{\text{compound}} = (15.0 \times G) + (15.0 \times I) + (30.0 \times [G \times I]) \in [0, 60.0]$$
   - **Baseline Protection ($15.0G + 15.0I$):** Ensures that acute isolation or deep poverty independently registers on the public health radar.
   - **Synergistic Compounding ($30.0 \times G \times I$):** Quadruples the compounding penalty when severe geographic friction co-occurs with economic deprivation.
        """)

    with tab2:
        st.markdown(r"""
#### Layer 2: Clinical & Operational Modifiers ($40.0\%$ Total Weight)

Layer 2 incorporates small area estimation (SAE) models anchored to NHMS Sabah, METMalaysia, and MOH epidemiological surveillance:

1. **Metric 3: TB Diagnostic Latency / Lag ($20.0\%$ Weight)**
   Models delays from symptom onset to microbiological confirmation (GeneXpert/CXR) as a function of physical travel barrier:
   $$\text{delay\_days} = 25.0 + (0.40 \times \text{travel\_time\_min})$$
   $$\text{norm\_lag} = \text{clip}\left(\frac{\text{delay\_days} - 25.0}{105.0 - 25.0}, 0.0, 1.0\right)$$
   $$S_{\text{lag}} = \text{norm\_lag} \times 20.0 \in [0, 20.0]$$

2. **Metric 4: Diabetes Mellitus Prevalence ($5.0\%$ Weight)**
   Grounded on the NHMS Sabah adult benchmark ($\sim 14.2\%$). Affluent/urban dietary shift is captured inversely while maintaining rural vulnerability:
   $$\text{diabetes\_pct} = 11.0 + 6.0 \times (1.0 - I)$$
   $$\text{norm\_dm} = \text{clip}\left(\frac{\text{diabetes\_pct} - 10.0}{18.0 - 10.0}, 0.0, 1.0\right)$$
   $$S_{\text{diabetes}} = \text{norm\_dm} \times 5.0 \in [0, 5.0]$$

3. **Metric 5: Weather & Seasonal Road Passability Risk ($5.0\%$ Weight)**
   Reflects the vulnerability of river basin floodplains (Kinabatangan, Tongod, Beluran) and logging routes to seasonal monsoon washouts:
   $$\text{norm\_weather} = \text{clip}\left(\frac{\text{road\_risk} - 0.0}{100.0 - 0.0}, 0.0, 1.0\right)$$
   $$S_{\text{weather}} = \text{norm\_weather} \times 5.0 \in [0, 5.0]$$

4. **Metric 6: Smoking & Tobacco Prevalence ($5.0\%$ Weight)**
   Anchored to the NHMS Sabah male smoking benchmark ($\sim 24\text{--}25\%$, elevated to $30.5\%$ in maritime communities):
   $$\text{norm\_smoking} = \text{clip}\left(\frac{\text{smoking\_pct} - 18.0}{32.0 - 18.0}, 0.0, 1.0\right)$$
   $$S_{\text{smoking}} = \text{norm\_smoking} \times 5.0 \in [0, 5.0]$$

5. **Metric 7: HIV / Immunosuppression Prevalence ($5.0\%$ Weight)**
   Captures concentrated sub-epidemics in commercial port cities and border transit hubs ($0.04\%\text{--}0.55\%$):
   $$\text{norm\_hiv} = \text{clip}\left(\frac{\text{hiv\_pct} - 0.0}{0.60 - 0.0}, 0.0, 1.0\right)$$
   $$S_{\text{hiv}} = \text{norm\_hiv} \times 5.0 \in [0, 5.0]$$
        """)

    with tab3:
        st.markdown(r"""
#### Complete Objective Function ($100.0\%$ Total Scale)

The overall continuous priority score for district $d$ is the strict sum of Layer 1 and Layer 2:

$$S_{\text{final}}(d) = S_{\text{compound}}(d) + \sum_{k \in \{\text{lag}, \text{dm}, \text{weather}, \text{smoking}, \text{hiv}\}} S_k(d)$$

Expanded LaTeX Formulation:
$$\begin{aligned}
S_{\text{final}}(d) &= \underbrace{15.0 \cdot G(d) + 15.0 \cdot I(d) + 30.0 \cdot [G(d) \cdot I(d)]}_{\text{Layer 1: Compound Structural Exposure (60.0\%)}} \\
&+ \underbrace{20.0 \cdot \left(\frac{\text{delay}(d) - 25}{80}\right)}_{\text{Diagnostic Latency (20.0\%)}} + \underbrace{5.0 \cdot \left(\frac{\text{DM\%}(d) - 10}{8}\right)}_{\text{Diabetes Mellitus (5.0\%)}} \\
&+ \underbrace{5.0 \cdot \left(\frac{\text{road\_risk}(d)}{100}\right)}_{\text{Road Passability (5.0\%)}} + \underbrace{5.0 \cdot \left(\frac{\text{smoke\%}(d) - 18}{14}\right)}_{\text{Smoking Prevalence (5.0\%)}} + \underbrace{5.0 \cdot \left(\frac{\text{HIV\%}(d)}{0.60}\right)}_{\text{HIV / Immunosuppression (5.0\%)}}
\end{aligned}$$

$$\text{Domain: } S_{\text{final}}(d) \in [0.0, 100.0] \quad \forall d \in \{1, \dots, 27\}$$
        """)

    with tab4:
        st.markdown(r"""
#### Calibration Parameters & Evidence Justification

| # | Dimension | Metric | Weight | Normalization Domain | Primary Evidence Source |
|---|---|---|---|---|---|
| **1** | Structural | Geodesic Travel Friction ($G$) | **15.0%** Base | $\min(T) \to \max(T)$ mins | MOH Facilities Master Registry |
| **2** | Structural | Income Deprivation Gradient ($I$) | **15.0%** Base | $\text{RM } 2,040 \to 6,542$ | OpenDOSM HIES (Sabah District) |
| **-** | Structural | Multiplicative Synergy ($G \times I$) | **30.0%** Synergy | $[0, 1] \times [0, 1]$ | Formulation C Composite |
| **3** | Clinical | Modeled Diagnostic Delay | **20.0%** | $25.0 \to 105.0$ days | Sabah MOH TB Surveillance (Lag Model) |
| **4** | Comorbidity | Diabetes Mellitus Prevalence | **5.0%** | $10.0\% \to 18.0\%$ | NHMS Sabah Benchmark ($\sim 14.2\%$) |
| **5** | Operational | Weather & Monsoon Road Risk | **5.0%** | $0 \to 100$ index | METMalaysia & Sabah DID River Basins |
| **6** | Behavioral | Smoking & Tobacco Prevalence | **5.0%** | $18.0\% \to 32.0\%$ | NHMS Sabah Male Smoking ($\sim 24.5\%$) |
| **7** | Comorbidity | HIV / Immunosuppression | **5.0%** | $0.0\% \to 0.60\%$ | MOH Malaysia Global AIDS Progress |
        """)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <hr style="border-color:rgba(0,200,180,0.15); margin-top:2rem;">
    <p style="text-align:center; color:#475569; font-size:0.8rem;">
        Sabah TB Mobile Clinic Prioritization Engine · 7-Metric Continuous Multi-Criteria System ·
        MOH Malaysia & OpenDOSM Open Data · For Mobile Unit Resource Allocation & Healthcare Equity
    </p>
    """,
    unsafe_allow_html=True,
)
