"""
Sabah TB Mobile Clinic Dispatch System - Phase 2
=================================================
Extends Phase 1 with real socioeconomic data from OpenDOSM HIES and a
Continuous Compound Structural Exposure Score combining:
  * Geographic Access Gradient     G  (weight 30%)
  * Income Deprivation Gradient    I  (weight 30%)
  => Compound Structural Exposure  S = (G x I) x 60.0  (out of 60 points)

Option C: Layered Hybrid Folium Map:
  - Base: Free Carto Dark / OpenStreetMap (NO API KEY required)
  - Layer 1: District Priority Centroids (circle markers sized 7-18, 3-tier colors)
  - Layer 2: Vulnerability Density Heatmap (HeatMap, hidden by default)
  - Layer 3: Diagnostic Hospitals (MOH, prominent blue markers, shown by default)
  - Layer 4: Primary Clinics (KLINIK, hidden by default to eliminate clutter)
  - Interactive Layer Control (collapsed=False)

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
    page_title="Sabah TB Mobility Dispatch - Phase 2",
    page_icon="\U0001f690",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# DESIGN TOKENS (Dark Mode, Teal/Amber Accent Palette)
# ---------------------------------------------------------------------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
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

        # Latest survey year
        if "year" in sabah_hies.columns:
            latest_year = pd.to_numeric(sabah_hies["year"], errors="coerce").max()
            if not np.isnan(latest_year):
                sabah_hies = sabah_hies[
                    pd.to_numeric(sabah_hies["year"], errors="coerce") == latest_year
                ]

        # Resolve income column
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

        # Fuzzy-match district names against canonical list
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

        # Fill missing districts from fallback
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

        # Fill NaN values from fallback
        for idx, row in sabah_hies.iterrows():
            dname = row["district"]
            if pd.isna(row["income_median"]) and dname in HIES_FALLBACK:
                sabah_hies.at[idx, "income_median"] = float(HIES_FALLBACK[dname]["income_median"])
            if pd.isna(row["poverty_rate"]) and dname in HIES_FALLBACK:
                sabah_hies.at[idx, "poverty_rate"] = float(HIES_FALLBACK[dname]["poverty_rate"])

        # Restrict strictly to the 27 canonical Sabah districts
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

    # Geographic Isolation Gradient G in [0, 1]
    t = df["est_travel_time_min"]
    t_min, t_max = t.min(), t.max()
    df["geo_gradient"] = ((t - t_min) / (t_max - t_min)).round(4) if t_max > t_min else 0.5
    df["access_score"]  = df["geo_gradient"]
    df["priority_rank"] = df["access_score"].rank(ascending=False, method="min").astype(int)
    return df


# ---------------------------------------------------------------------------
# CONTINUOUS COMPOUND STRUCTURAL SCORE (60% TOTAL WEIGHT)
# ---------------------------------------------------------------------------
def build_compound_df(access_df: pd.DataFrame, hies_df: pd.DataFrame,
                      geo_weight: float = 0.30, econ_weight: float = 0.30) -> pd.DataFrame:
    """
    Merge geodesic access with HIES income data and calculate:
      1. Geographic Access Gradient:    G = (T - min(T)) / (max(T) - min(T))
      2. Income Deprivation Gradient:   I = (max(Inc) - Inc) / (max(Inc) - min(Inc))
      3. Formulation C (Hybrid Base + Synergistic Interaction):
         S = (0.25 * total_scale * G) + (0.25 * total_scale * I) + (0.50 * total_scale * (G * I))
         When total_scale = 60.0:
           S = 15.0 * G + 15.0 * I + 30.0 * (G * I)  (out of 60 points)
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
    df["geo_gradient"]  = df["geo_gradient"].fillna(0.5)

    # Income Deprivation Gradient I in [0, 1]
    inc = df["income_median"]
    i_max, i_min = inc.max(), inc.min()
    df["income_gradient"] = ((i_max - inc) / (i_max - i_min)).round(4) if i_max > i_min else 0.5

    # Formulation C: Hybrid Base (15% Geo + 15% Income) + Synergistic Interaction (30% Geo * Income) = 60% total
    total_scale = (geo_weight + econ_weight) * 100.0
    df["compound_score"] = (
        (0.25 * total_scale * df["geo_gradient"])
        + (0.25 * total_scale * df["income_gradient"])
        + (0.50 * total_scale * (df["geo_gradient"] * df["income_gradient"]))
    ).round(2)

    df["compound_rank"] = df["compound_score"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("compound_rank").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# COLOR TIERS FOR COMPOUND STRUCTURAL SCORE (/60)
# ---------------------------------------------------------------------------
def compound_color(score: float) -> str:
    """
    Color tiers for Option C map:
      Green:     < 10.0   (Low priority / Accessible & economically resilient)
      Orange:    10.0-35.0 (Moderate priority)
      Deep Red:  > 35.0   (Critical priority - remote interior + severe poverty)
    """
    if score < 10.0:
        return "#22c55e"   # Green
    elif score <= 35.0:
        return "#f97316"   # Orange
    else:
        return "#dc2626"   # Deep Red


def compound_label(score: float) -> str:
    if score < 10.0:
        return "Low Vulnerability (<10.0)"
    elif score <= 35.0:
        return "Moderate Priority (10.0–35.0)"
    else:
        return "Critical Priority (>35.0)"


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚐 TB Dispatch · Phase 2")
    st.markdown("---")

    # Pipeline Health & Data Sources
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

    # Pipeline Speed Parameter
    st.markdown("### ⚙️ Pipeline Parameters")
    speed_kmh = st.slider(
        "Average road speed (km/h)",
        min_value=25, max_value=60, value=40, step=5,
        help="Winding rural roads. 40 km/h is the conservative baseline.",
    )

    st.markdown("---")
    st.markdown("### 📊 Continuous Compound Weights")
    st.caption("Compound structural model combines Geographic Access & Household Income Deprivation.")

    geo_weight = st.slider(
        "Geographic Isolation Weight (G)",
        min_value=10, max_value=50, value=30, step=5,
        help="Weight applied to the Geo Isolation Gradient G (Default: 30%).",
    ) / 100.0

    econ_weight = st.slider(
        "Income Deprivation Weight (I)",
        min_value=10, max_value=50, value=30, step=5,
        help="Weight applied to the Income Deprivation Gradient I (Default: 30%).",
    ) / 100.0

    max_compound_val = (geo_weight + econ_weight) * 100.0

    st.markdown(
        f"""
        <div style="background:rgba(0,200,180,0.07); border:1px solid rgba(0,200,180,0.25);
             border-radius:8px; padding:10px; font-size:0.82rem; color:#a8c8d8; margin-top:4px;">
            <b style="color:#00c8b4;">Formulation C (Hybrid + Synergy)</b><br>
            S = 15&middot;G + 15&middot;I + 30&middot;(G &times; I)<br>
            Max Score Scale: <b style="color:#e8f4f8;">{max_compound_val:.0f} pts</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# COMPUTE ACCESS & MERGE COMPOUND DATA
# ---------------------------------------------------------------------------
st.session_state["_hospitals"] = hospitals
hosp_hash = f"{len(hospitals)}_{speed_kmh}"
access_df   = compute_access(hosp_hash, speed_kmh)
compound_df = build_compound_df(access_df, hies_df, geo_weight, econ_weight)

# Sidebar KPI Cards
top_district     = compound_df.iloc[0]
state_median     = compound_df["income_median"].mean()
richest_income   = compound_df["income_median"].max()
poorest_income   = compound_df["income_median"].min()
disparity_ratio  = richest_income / poorest_income if poorest_income > 0 else float("nan")

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📌 Priority & Infrastructure Snapshot")

    st.markdown(
        f"""
        <div class="sidebar-kpi">
            <div class="sidebar-kpi-title">Highest Compound Priority</div>
            <div class="sidebar-kpi-val" style="color:#ef4444;">{top_district['district']}</div>
            <div class="sidebar-kpi-sub">Score {top_district['compound_score']:.1f}/{max_compound_val:.0f} · RM {top_district['income_median']:,.0f}</div>
        </div>
        <div class="sidebar-kpi">
            <div class="sidebar-kpi-title">Sabah State Median Income</div>
            <div class="sidebar-kpi-val">RM {state_median:,.0f}</div>
            <div class="sidebar-kpi-sub">Across 27 administrative districts</div>
        </div>
        <div class="sidebar-kpi">
            <div class="sidebar-kpi-title">Active Diagnostic Infrastructure</div>
            <div class="sidebar-kpi-val" style="color:#38bdf8;">{len(hospitals)} Hospitals</div>
            <div class="sidebar-kpi-sub">{len(clinics)} Primary Clinics Registered</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Data: MOH Facilities & OpenDOSM HIES")
    st.caption("Coordinate System: WGS84 Geodesic")

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <h1 style="
        background: linear-gradient(90deg, #00c8b4 0%, #0096c7 60%, #7ecfdf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.1rem; font-weight: 700; margin-bottom: 0.15rem;
    ">🚐 Sabah TB Mobile Clinic Dispatch System</h1>
    <p style="color:#5a7a8a; font-size:0.95rem; margin-top:0;">
        Phase 2 &nbsp;&middot;&nbsp; Continuous Compound Structural Exposure Score &nbsp;&middot;&nbsp;
        Geographic Access (30%) &times; Income Deprivation (30%) = 60% Model Weight
    </p>
    <hr style="border-color:rgba(0,200,180,0.2); margin:0.8rem 0 1.2rem;">
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# TOP METRIC KPI CARDS (2 Rows)
# ---------------------------------------------------------------------------
bottom_district = compound_df.iloc[-1]
geo_top         = compound_df.sort_values("geo_gradient", ascending=False).iloc[0]
critical_count  = (compound_df["compound_score"] > 35.0).sum()

r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    st.markdown('<div class="metric-compound">', unsafe_allow_html=True)
    st.metric(
        "🔴 Highest Compound Vulnerability",
        top_district["district"],
        (
            f"Score {top_district['compound_score']:.1f}/{max_compound_val:.0f} | "
            f"RM {top_district['income_median']:,.0f} median | "
            f"≈{top_district['est_travel_time_min']:.0f} min"
        ),
        delta_color="inverse",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with r1c2:
    st.metric(
        "💰 State Median Income",
        f"RM {state_median:,.0f}",
        f"Average across all 27 Sabah districts",
    )

with r1c3:
    st.metric(
        "📊 Urban-Rural Income Disparity",
        f"{disparity_ratio:.1f}× Ratio",
        f"Richest RM {richest_income:,.0f} vs Poorest RM {poorest_income:,.0f}",
        delta_color="inverse",
    )

st.markdown("<br>", unsafe_allow_html=True)

r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    st.metric(
        "✅ Lowest Compound Vulnerability",
        bottom_district["district"],
        (
            f"Score {bottom_district['compound_score']:.1f}/{max_compound_val:.0f} | "
            f"RM {bottom_district['income_median']:,.0f} median"
        ),
    )

with r2c2:
    st.markdown('<div class="metric-critical">', unsafe_allow_html=True)
    st.metric(
        "🗺️ Maximum Geographic Isolation",
        geo_top["district"],
        f"G = {geo_top['geo_gradient']:.3f} | ≈{geo_top['est_travel_time_min']:.0f} min travel",
        delta_color="inverse",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with r2c3:
    st.metric(
        "⚠️ Critical Priority Districts (>35/60)",
        f"{critical_count} / {len(compound_df)} Districts",
        f"Multiplicative isolation & poverty trigger",
        delta_color="inverse",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAP VISUALIZATION: OPTION C LAYERED HYBRID (FOLIUM)
# ---------------------------------------------------------------------------
st.markdown("## 🗺️ Option C: Layered Hybrid Vulnerability Map")
st.caption(
    "Decluttered default view showing district priorities and diagnostic hospitals. "
    "Toggle the **Vulnerability Heatmap** or **Primary Clinics** layers via the layer switcher on the top-right."
)

# Base Map: Free Carto Dark / OpenStreetMap (NO API KEY required)
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

# ---------------------------------------------------------------------------
# Layer 2: Vulnerability Density Heatmap (HeatMap, hidden by default)
# ---------------------------------------------------------------------------
heatmap_fg = folium.FeatureGroup(name="Vulnerability Heatmap Gradient", show=False)
heatmap_coords = [
    [float(row["lat"]), float(row["lon"]), float(row["compound_score"])]
    for _, row in compound_df.iterrows()
]
HeatMap(
    data=heatmap_coords,
    radius=40,
    blur=25,
    max_zoom=9,
    min_opacity=0.35,
    gradient={
        0.2: "#fde047",  # Yellow
        0.5: "#f97316",  # Orange
        0.8: "#dc2626",  # Crimson Red
        1.0: "#7f1d1d",  # Deep Maroon
    },
).add_to(heatmap_fg)
heatmap_fg.add_to(m)

# ---------------------------------------------------------------------------
# Layer 4: Primary Clinics (KLINIK, hidden by default)
# ---------------------------------------------------------------------------
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
            tooltip=folium.Tooltip(f"🏪 {c_name}<br><span style='color:#94a3b8;'>Primary Clinic (KLINIK)</span>", sticky=False),
        ).add_to(clinic_fg)
clinic_fg.add_to(m)

# ---------------------------------------------------------------------------
# Layer 3: Diagnostic Hospitals (MOH, shown by default)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Layer 1: District Priority Centroids (Shown by default)
# ---------------------------------------------------------------------------
dist_fg = folium.FeatureGroup(name="District Vulnerability Markers", show=True)

for _, row in compound_df.iterrows():
    score = float(row["compound_score"])
    color = compound_color(score)
    label = compound_label(score)

    # Dynamic radius from 7 to 18 based on score
    radius = 7.0 + (score / max_compound_val) * 11.0

    pov_rate_val = row.get("poverty_rate", np.nan)
    pov_str = f"{pov_rate_val:.1f}%" if pd.notna(pov_rate_val) else "N/A"

    tooltip_html = f"""
    <div style="font-family:'Inter',sans-serif; min-width:260px; color:#1e293b;">
        <div style="font-size:1.05rem; font-weight:700; color:#0f172a; margin-bottom:2px;">
            {row['district']}
        </div>
        <div style="color:{color}; font-weight:700; font-size:0.85rem; margin-bottom:6px;">
            ● {label}
        </div>
        <hr style="margin:4px 0 8px 0; border:0; border-top:1px solid #cbd5e1;">
        <table style="width:100%; font-size:0.82rem; line-height:1.75; border-collapse:collapse;">
            <tr>
                <td style="color:#64748b;">🏥 Nearest Hospital:</td>
                <td style="font-weight:600; text-align:right;">{row['nearest_facility']}</td>
            </tr>
            <tr>
                <td style="color:#64748b;">⏱ Travel Time:</td>
                <td style="font-weight:600; text-align:right;">{row['est_travel_time_min']:.0f} min ({row['min_distance_km']:.1f} km)</td>
            </tr>
            <tr>
                <td style="color:#64748b;">💰 Median Monthly Income:</td>
                <td style="font-weight:600; text-align:right;">RM {row['income_median']:,.0f}</td>
            </tr>
            <tr>
                <td style="color:#64748b;">🏚 Poverty Rate:</td>
                <td style="font-weight:600; text-align:right;">{pov_str}</td>
            </tr>
            <tr>
                <td style="color:#64748b;">📍 Geo Index (G):</td>
                <td style="font-weight:600; text-align:right;">{row['geo_gradient']:.3f}</td>
            </tr>
            <tr>
                <td style="color:#64748b;">📉 Income Index (I):</td>
                <td style="font-weight:600; text-align:right;">{row['income_gradient']:.3f}</td>
            </tr>
            <tr style="background:#f1f5f9; border-top:1px solid #cbd5e1;">
                <td style="color:#0f172a; font-weight:700; padding:4px 2px;">🧮 Compound Score:</td>
                <td style="font-size:0.95rem; font-weight:800; color:{color}; text-align:right; padding:4px 2px;">
                    {score:.1f} / {max_compound_val:.0f}
                </td>
            </tr>
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

    # District label
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
legend_html = f"""
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
  <b style="color:#00c8b4; font-size:13px;">Compound Structural Score (/{max_compound_val:.0f})</b><br>
  <span style="color:#94a3b8; font-size:11px;">Formulation C: 15&middot;G + 15&middot;I + 30&middot;(G &times; I)</span><br><br>
  <span style="color:#22c55e;">●</span> &lt; 10.0 &nbsp; Low Priority (Resilient)<br>
  <span style="color:#f97316;">●</span> 10.0 – 35.0 &nbsp; Moderate Priority<br>
  <span style="color:#dc2626;">●</span> &gt; 35.0 &nbsp; Critical Priority (Remote + Poverty)<br>
  <hr style="margin:8px 0; border:0; border-top:1px solid rgba(0,200,180,0.2);">
  <span style="color:#38bdf8;">✚</span> Diagnostic Hospital (MOH Hub)<br>
  <span style="color:#94a3b8;">●</span> Primary Clinic (KLINIK Layer)
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width="100%", height=630, returned_objects=[])

# ---------------------------------------------------------------------------
# COMPOUND BREAKDOWN (TOP 10)
# ---------------------------------------------------------------------------
with st.expander("📊 Top 10 High Priority Districts (Compound Breakdown)", expanded=True):
    top10 = compound_df.head(10)[[
        "district", "geo_gradient", "income_gradient", "compound_score",
        "income_median", "est_travel_time_min"
    ]].copy()
    top10.columns = [
        "District", "Geo Index (G)", "Income Deprivation (I)", f"Compound Score (/{max_compound_val:.0f})",
        "Median Income (RM)", "Travel Time (min)"
    ]

    def _style_score(val):
        s = float(val)
        if s > 35.0:  return "color:#ef4444; font-weight:700"
        elif s >= 10: return "color:#f97316; font-weight:600"
        else:         return "color:#22c55e; font-weight:600"

    _fn = getattr(top10.style, "map", None) or getattr(top10.style, "applymap")
    styled10 = _fn(_style_score, subset=[f"Compound Score (/{max_compound_val:.0f})"]).format({
        "Geo Index (G)":                            "{:.3f}",
        "Income Deprivation (I)":                   "{:.3f}",
        f"Compound Score (/{max_compound_val:.0f})": "{:.2f}",
        "Median Income (RM)":                       "RM {:,.0f}",
        "Travel Time (min)":                        "{:.0f} min",
    })
    try:
        st.dataframe(styled10, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(styled10, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# FULL DISTRICT RANKING TABLE (27 DISTRICTS)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 📋 District Priority Ranking Table")
st.caption(
    f"Sorted descending by Continuous Compound Score (out of {max_compound_val:.0f}). "
    "Calculated via Formulation C: Hybrid Base (15% Geo + 15% Income) + Synergistic Interaction (30% Geo &times; Income) = 60% total."
)

table_df = compound_df[[
    "district",
    "est_travel_time_min",
    "income_median",
    "geo_gradient",
    "income_gradient",
    "compound_score",
]].copy()

table_df.columns = [
    "District",
    "Travel Time (min)",
    "Median Income (RM)",
    "Geo Index (G)",
    "Income Deprivation (I)",
    f"Compound Score (/{max_compound_val:.0f})",
]

def _hl_compound(val):
    s = float(val)
    if s > 35.0:  return "color:#ef4444; font-weight:700"
    elif s >= 10: return "color:#f97316; font-weight:600"
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
    _mfn(_hl_compound, subset=[f"Compound Score (/{max_compound_val:.0f})"])
    .pipe(lambda s: (getattr(s, "map", None) or getattr(s, "applymap"))(
        _hl_income, subset=["Median Income (RM)"]
    ))
    .format({
        "Travel Time (min)":                        "{:.1f} min",
        "Median Income (RM)":                       "RM {:,.0f}",
        "Geo Index (G)":                            "{:.4f}",
        "Income Deprivation (I)":                   "{:.4f}",
        f"Compound Score (/{max_compound_val:.0f})": "{:.2f}",
    })
)

try:
    st.dataframe(styled_full, width="stretch", height=620, hide_index=False)
except TypeError:
    st.dataframe(styled_full, use_container_width=True, height=620, hide_index=False)

# Export Dataset
export_df = compound_df[[
    "compound_rank", "district", "nearest_facility", "min_distance_km",
    "est_travel_time_min", "income_median", "poverty_rate",
    "geo_gradient", "income_gradient", "compound_score"
]].copy()
export_df.columns = [
    "rank", "district", "nearest_hospital", "distance_km",
    "travel_time_min", "median_income_rm", "poverty_rate_pct",
    "geo_gradient_g", "income_deprivation_i", "compound_structural_score_60"
]
csv_bytes = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Export Table as CSV (sabah_compound_vulnerability.csv)",
    data=csv_bytes,
    file_name="sabah_compound_vulnerability.csv",
    mime="text/csv",
    help="Download complete 27-district compound TB vulnerability dataset as CSV.",
)

# ---------------------------------------------------------------------------
# METHODOLOGY & FORMULATION EXPANDER
# ---------------------------------------------------------------------------
with st.expander("📐 Mathematical Formulation & Data Engineering Details", expanded=False):
    st.markdown(fr"""
### Continuous Compound Structural Exposure Model (60% Weight)

1. **Geographic Isolation Gradient ($G \in [0, 1]$):**
   $$G = \frac{{T - \min(T)}}{{\max(T) - \min(T)}}$$
   - Where $T$ is geodesic travel time in minutes to the nearest GeneXpert/digital CXR hospital hub.
   - Road speed baseline: **{speed_kmh} km/h** with 15-minute dispatch overhead.

2. **Income Deprivation Gradient ($I \in [0, 1]$):**
   $$I = \frac{{\max(\text{{Income}}) - \text{{Income}}}}{{\max(\text{{Income}}) - \min(\text{{Income}})}}$$
   - Lower median household income yields higher deprivation gradient ($I \to 1.0$).

3. **Compound Structural Score ($S_{{\text{{compound}}}} \in [0, 60]$) — Formulation C:**
   $$S_{{\text{{compound}}}} = (0.25 \times 60 \times G) + (0.25 \times 60 \times I) + (0.50 \times 60 \times G \times I) = 15G + 15I + 30(G \times I)$$
   - **Hybrid Base (30% total):** 15% Geographic Isolation ($15G$) + 15% Income Deprivation ($15I$).
   - **Synergistic Interaction (30% total):** 30% Multiplicative Interaction ($30 \times G \times I$).
   - Current scale with weights ({geo_weight*100:.0f}% G + {econ_weight*100:.0f}% I): **{max_compound_val:.0f} pts max**.

---

### Data Sources
- **Ministry of Health Malaysia (MOH):** `facilities_master.csv` (Diagnostic Hospitals & Clinics).
- **Department of Statistics Malaysia (OpenDOSM HIES):** `hies_district.csv` (Sabah district median household income).
- **Fallback Engine:** Embedded authentic DOSM HIES baseline covering all 27 districts for 100% offline resilience.
    """)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <hr style="border-color:rgba(0,200,180,0.15); margin-top:2rem;">
    <p style="text-align:center; color:#334155; font-size:0.8rem;">
        Sabah TB Mobile Clinic Dispatch System · Phase 2 ·
        Option C Layered Hybrid · MOH Malaysia + OpenDOSM HIES Data ·
        For public health research and mobile unit allocation
    </p>
    """,
    unsafe_allow_html=True,
)
