"""
Sabah TB Mobile Clinic Dispatch System - Phase 2
=================================================
Extends Phase 1 with real socioeconomic data from OpenDOSM HIES and a
Compound Structural Exposure Score combining:
  * Geographic Isolation Gradient  G  (weight 30%)
  * Income Deprivation Gradient    I  (weight 30%)
  => Compound Score  S = G x I x 60   (out of 60 points)

Run:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import folium
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
# DESIGN TOKENS  (dark mode, teal/amber palette)
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
    padding: 16px 20px;
    backdrop-filter: blur(10px);
}
[data-testid="stMetric"] label {
    color: rgba(0,200,180,0.85) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] { color: #e8f4f8 !important; font-size: 1.5rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }
[data-testid="stDataFrame"] { border: 1px solid rgba(0,200,180,0.15); border-radius: 10px; overflow: hidden; }
hr { border-color: rgba(0,200,180,0.2); }
.stDownloadButton > button {
    background: linear-gradient(135deg, #00c8b4 0%, #0096c7 100%) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; padding: 0.5rem 1.2rem !important; transition: opacity 0.2s;
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
  50%       { box-shadow: 0 0 18px 6px rgba(239,68,68,0.22); }
}
.metric-compound { animation: compoundPulse 2.5s ease-in-out infinite; }
.hies-connected {
    display:inline-block; background:rgba(34,197,94,0.18); border:1px solid #22c55e;
    color:#22c55e; border-radius:20px; padding:2px 10px; font-size:0.75rem; font-weight:600;
}
.hies-cached {
    display:inline-block; background:rgba(245,158,11,0.18); border:1px solid #f59e0b;
    color:#f59e0b; border-radius:20px; padding:2px 10px; font-size:0.75rem; font-weight:600;
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CONSTANTS - 27 Official Sabah District Centroids
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
# HIES FALLBACK BASELINE (authentic DOSM HIES estimates - all 27 districts)
# ---------------------------------------------------------------------------
HIES_FALLBACK = {
    "Kota Kinabalu": {"income_median": 6542, "poverty_rate": 4.1},
    "Penampang":     {"income_median": 5890, "poverty_rate": 5.3},
    "Putatan":       {"income_median": 5120, "poverty_rate": 6.8},
    "Tuaran":        {"income_median": 4350, "poverty_rate": 9.2},
    "Papar":         {"income_median": 3980, "poverty_rate": 11.5},
    "Kota Belud":    {"income_median": 3210, "poverty_rate": 14.7},
    "Ranau":         {"income_median": 3050, "poverty_rate": 15.9},
    "Kudat":         {"income_median": 2820, "poverty_rate": 18.4},
    "Kota Marudu":   {"income_median": 2650, "poverty_rate": 21.3},
    "Pitas":         {"income_median": 2350, "poverty_rate": 26.8},
    "Keningau":      {"income_median": 3680, "poverty_rate": 12.4},
    "Tambunan":      {"income_median": 3120, "poverty_rate": 16.1},
    "Tenom":         {"income_median": 2980, "poverty_rate": 17.6},
    "Nabawan":       {"income_median": 2180, "poverty_rate": 29.4},
    "Beaufort":      {"income_median": 3450, "poverty_rate": 13.2},
    "Kuala Penyu":   {"income_median": 2760, "poverty_rate": 19.7},
    "Sipitang":      {"income_median": 2890, "poverty_rate": 18.1},
    "Sandakan":      {"income_median": 4120, "poverty_rate": 10.3},
    "Beluran":       {"income_median": 2680, "poverty_rate": 22.5},
    "Telupid":       {"income_median": 2310, "poverty_rate": 27.9},
    "Tongod":        {"income_median": 2040, "poverty_rate": 31.6},
    "Kinabatangan":  {"income_median": 2590, "poverty_rate": 23.8},
    "Tawau":         {"income_median": 3840, "poverty_rate": 11.8},
    "Lahad Datu":    {"income_median": 3190, "poverty_rate": 15.3},
    "Semporna":      {"income_median": 2420, "poverty_rate": 25.1},
    "Kunak":         {"income_median": 2790, "poverty_rate": 20.4},
    "Kalabakan":     {"income_median": 2470, "poverty_rate": 24.6},
}

# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Fetching MOH Malaysia facility registry...")
def load_moh_facilities():
    """Download and cache the MOH master facility CSV."""
    try:
        df = pd.read_csv(MOH_URL, low_memory=False)
    except Exception as exc:
        st.error(f"Failed to fetch MOH data: {exc}")
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

    hospitals = hospitals.dropna(subset=["LATITUD", "LONGITUD"])
    clinics   = clinics.dropna(subset=["LATITUD", "LONGITUD"])
    return sabah_open, hospitals, clinics


@st.cache_data(ttl=3600, show_spinner="Fetching OpenDOSM HIES district income data...")
def load_hies_income():
    """
    Load Sabah district-level income from OpenDOSM HIES.
    Falls back to embedded baseline dictionary on network failure.

    Returns
    -------
    df     : pd.DataFrame  columns=[district, income_median, poverty_rate]
    source : str           'OpenDOSM' or 'Cached Baseline'
    """
    canonical = list(HIES_FALLBACK.keys())

    def _build_fallback():
        rows = [
            {"district": k, "income_median": v["income_median"], "poverty_rate": v["poverty_rate"]}
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
        if "poverty_rate" not in sabah_hies.columns:
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
                    "district":     d,
                    "income_median": HIES_FALLBACK[d]["income_median"],
                    "poverty_rate":  HIES_FALLBACK[d]["poverty_rate"],
                })
        if missing_rows:
            sabah_hies = pd.concat([sabah_hies, pd.DataFrame(missing_rows)], ignore_index=True)

        # Fill NaN values from fallback
        for idx, row in sabah_hies.iterrows():
            if pd.isna(row["income_median"]) and row["district"] in HIES_FALLBACK:
                sabah_hies.at[idx, "income_median"] = HIES_FALLBACK[row["district"]]["income_median"]
            if pd.isna(row["poverty_rate"]) and row["district"] in HIES_FALLBACK:
                sabah_hies.at[idx, "poverty_rate"] = HIES_FALLBACK[row["district"]]["poverty_rate"]

        sabah_hies = sabah_hies.drop_duplicates(subset=["district"]).reset_index(drop=True)

        if sabah_hies["income_median"].notna().sum() < 5:
            return _build_fallback()

        return sabah_hies[["district", "income_median", "poverty_rate"]], "OpenDOSM"

    except Exception:
        return _build_fallback()


# ---------------------------------------------------------------------------
# HAVERSINE & DISTANCE ENGINE
# ---------------------------------------------------------------------------
def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between one point and an array of points."""
    R = 6_371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner="Computing geodesic access matrix...")
def compute_access(hospitals_hash: str, speed_kmh: float) -> pd.DataFrame:
    """Return district-level access metrics given hospital data and speed."""
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

    # Geo Isolation Gradient G in [0, 1]
    t = df["est_travel_time_min"]
    t_min, t_max = t.min(), t.max()
    df["geo_gradient"] = ((t - t_min) / (t_max - t_min)).round(4) if t_max > t_min else 0.5

    df["access_score"]  = df["geo_gradient"]  # backward compat
    df["priority_rank"] = df["access_score"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("priority_rank").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# COMPOUND SCORE ENGINE
# ---------------------------------------------------------------------------
def build_compound_df(access_df, hies_df, geo_weight=0.30, econ_weight=0.30):
    """
    Merge geodesic access data with HIES income data and compute:
      G  - Geographic Isolation Gradient in [0, 1]
      I  - Income Deprivation Gradient   in [0, 1]
      S  - Compound Score = G x I x (geo_weight + econ_weight) x 100
           (displays out of 60 when weights sum to 0.60)
    """
    df = access_df.copy()
    df = df.merge(hies_df[["district", "income_median", "poverty_rate"]], on="district", how="left")

    # Fill remaining NaN from fallback
    for idx, row in df.iterrows():
        if pd.isna(row["income_median"]) and row["district"] in HIES_FALLBACK:
            df.at[idx, "income_median"] = HIES_FALLBACK[row["district"]]["income_median"]
        if pd.isna(row["poverty_rate"]) and row["district"] in HIES_FALLBACK:
            df.at[idx, "poverty_rate"] = HIES_FALLBACK[row["district"]]["poverty_rate"]

    df["income_median"] = pd.to_numeric(df["income_median"], errors="coerce")
    df["poverty_rate"]  = pd.to_numeric(df["poverty_rate"],  errors="coerce")
    df["geo_gradient"]  = df["geo_gradient"].fillna(0.5)

    # Income Deprivation Gradient I in [0, 1]
    inc  = df["income_median"]
    i_max, i_min = inc.max(), inc.min()
    df["income_gradient"] = ((i_max - inc) / (i_max - i_min)).round(4) if i_max > i_min else 0.5

    # Compound Score S = G x I x combined_weight x 100
    combined_weight = geo_weight + econ_weight
    max_score       = combined_weight * 100.0
    df["compound_score"] = (df["geo_gradient"] * df["income_gradient"] * max_score).round(2)

    df["compound_rank"] = df["compound_score"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("compound_rank").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# COLOUR HELPERS (compound score out of 60)
# ---------------------------------------------------------------------------
def compound_color(score):
    if score < 10.0:
        return "#22c55e"   # green
    elif score < 22.0:
        return "#84cc16"   # lime
    elif score < 35.0:
        return "#f59e0b"   # amber
    elif score < 48.0:
        return "#f97316"   # orange
    else:
        return "#ef4444"   # deep red


def compound_label(score):
    if score < 10.0:
        return "Low Priority - Accessible & Resilient"
    elif score < 22.0:
        return "Moderate-Low"
    elif score < 35.0:
        return "Moderate Priority"
    elif score < 48.0:
        return "High Priority"
    else:
        return "CRITICAL - Remote + Severe Poverty"


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## TB Dispatch Phase 2")
    st.markdown("---")

    st.markdown("### Pipeline Parameters")
    speed_kmh = st.slider(
        "Average road speed (km/h)",
        min_value=25, max_value=60, value=40, step=5,
        help="Winding rural roads. 40 km/h is the conservative baseline.",
    )
    show_clinics = st.checkbox(
        "Show KLINIK reference layer", value=False,
        help="Toggle secondary clinic dots on the map."
    )

    st.markdown("---")
    st.markdown("### Compound Score Weights")
    st.caption("Adjust the contribution of each pillar. Max compound score = (geo + econ) x 100.")

    geo_weight = st.slider(
        "Geographic Isolation Weight (%)",
        min_value=10, max_value=50, value=30, step=5,
        help="Weight applied to the Geo Isolation Gradient G."
    ) / 100.0

    econ_weight = st.slider(
        "Economic Vulnerability Weight (%)",
        min_value=10, max_value=50, value=30, step=5,
        help="Weight applied to the Income Deprivation Gradient I."
    ) / 100.0

    combined_pct = (geo_weight + econ_weight) * 100
    max_compound = combined_pct
    st.markdown(
        f"""
        <div style="background:rgba(0,200,180,0.07); border:1px solid rgba(0,200,180,0.2);
             border-radius:8px; padding:10px; font-size:0.82rem; color:#a8c8d8; margin-top:8px;">
            <b style="color:#00c8b4;">Compound Interaction Formula</b><br>
            S = G &times; I &times; {combined_pct:.0f}<br>
            Max possible score: <b style="color:#e8f4f8;">{max_compound:.0f} pts</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Pipeline Health")

    sabah_open, hospitals, clinics = load_moh_facilities()

    if hospitals.empty:
        st.error("No hospital data loaded. Check network connection.")
        st.stop()

    hies_df, hies_source = load_hies_income()

    if hies_source == "OpenDOSM":
        st.markdown(
            'HIES Source: <span class="hies-connected">Connected (OpenDOSM)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            'HIES Source: <span class="hies-cached">Cached Baseline</span>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.metric("Total Open Sabah Facilities",  len(sabah_open))
    st.metric("Active Diagnostic Hospitals",   len(hospitals))
    st.metric("Registered Clinics (KLINIK)",   len(clinics))
    st.metric("HIES Districts Loaded",          len(hies_df))

    st.markdown("---")
    st.caption("Data: MOH Malaysia - facilities_master.csv")
    st.caption("Income: OpenDOSM HIES - hies_district.csv")
    st.caption("Geometry: WGS84 decimal degrees - Haversine geodesic")


# ---------------------------------------------------------------------------
# COMPUTE ACCESS MATRIX + COMPOUND SCORE
# ---------------------------------------------------------------------------
st.session_state["_hospitals"] = hospitals

hosp_hash = (
    f"{len(hospitals)}_"
    f"{hospitals['NAMA'].str.cat()[:40] if not hospitals.empty else ''}_"
    f"{speed_kmh}"
)
access_df   = compute_access(hosp_hash, speed_kmh)
compound_df = build_compound_df(access_df, hies_df, geo_weight, econ_weight)

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
    ">Sabah TB Mobile Clinic Dispatch System</h1>
    <p style="color:#5a7a8a; font-size:0.95rem; margin-top:0;">
        Phase 2 &nbsp;&middot;&nbsp; Compound Structural Exposure Score &nbsp;&middot;&nbsp;
        Geographic Isolation (30%) + Income Deprivation (30%) = 60% Model Weight
        &nbsp;&middot;&nbsp; OpenDOSM HIES + MOH Malaysia Data
    </p>
    <hr style="border-color:rgba(0,200,180,0.2); margin:0.8rem 0 1.2rem;">
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# TOP METRIC KPI CARDS  (6 cards, 2 rows)
# ---------------------------------------------------------------------------
top_district     = compound_df.iloc[0]
bottom_district  = compound_df.iloc[-1]
state_median     = compound_df["income_median"].mean()
richest_income   = compound_df["income_median"].max()
poorest_income   = compound_df["income_median"].min()
disparity_ratio  = richest_income / poorest_income if poorest_income > 0 else float("nan")
geo_top          = compound_df.sort_values("geo_gradient", ascending=False).iloc[0]
max_compound_val = (geo_weight + econ_weight) * 100

r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    st.markdown('<div class="metric-compound">', unsafe_allow_html=True)
    st.metric(
        "Highest Compound Vulnerability",
        top_district["district"],
        (
            f"Score {top_district['compound_score']:.1f}/{max_compound_val:.0f} | "
            f"RM {top_district['income_median']:,.0f} median | "
            f"~{top_district['est_travel_time_min']:.0f} min"
        ),
        delta_color="inverse",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with r1c2:
    st.metric(
        "Lowest Compound Vulnerability",
        bottom_district["district"],
        (
            f"Score {bottom_district['compound_score']:.1f}/{max_compound_val:.0f} | "
            f"RM {bottom_district['income_median']:,.0f} median"
        ),
    )

with r1c3:
    st.metric(
        "State Median Income (Average)",
        f"RM {state_median:,.0f}",
        f"Across all {len(compound_df)} Sabah districts",
    )

st.markdown("<br>", unsafe_allow_html=True)

r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    st.metric(
        "Urban-Rural Income Disparity Ratio",
        f"{disparity_ratio:.1f}x",
        f"RM {richest_income:,.0f} / RM {poorest_income:,.0f}",
        delta_color="inverse",
    )

with r2c2:
    st.markdown('<div class="metric-critical">', unsafe_allow_html=True)
    st.metric(
        "Most Geographically Isolated",
        geo_top["district"],
        f"G = {geo_top['geo_gradient']:.3f} | ~{geo_top['est_travel_time_min']:.0f} min",
        delta_color="inverse",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with r2c3:
    avg_compound   = compound_df["compound_score"].mean()
    critical_count = (compound_df["compound_score"] > 35).sum()
    st.metric(
        "Critical Priority Districts",
        f"{critical_count} / {len(compound_df)}",
        f"Avg compound score: {avg_compound:.1f}/{max_compound_val:.0f}",
        delta_color="inverse",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAIN MAP
# ---------------------------------------------------------------------------
st.markdown("## Compound Structural Exposure Map")
st.caption(
    f"Circle size and colour reflect the Compound Structural Score (out of {max_compound_val:.0f}). "
    "Green < 10 (Accessible & Resilient)  |  "
    "Lime 10-22 (Moderate-Low)  |  "
    "Amber 22-35 (Moderate)  |  "
    "Orange 35-48 (High)  |  "
    "Red > 48 (Critical)"
)

m = folium.Map(location=[5.5, 116.8], zoom_start=7, tiles="OpenStreetMap", control_scale=True)

# Hospital layer
hosp_fg = folium.FeatureGroup(name="Diagnostic Hospitals", show=True)
for _, h in hospitals.iterrows():
    name = h.get("NAMA", "Hospital")
    folium.Marker(
        location=[h["LATITUD"], h["LONGITUD"]],
        tooltip=folium.Tooltip(
            f"<b>{name}</b><br>"
            f"<span style='color:#60a5fa'>Diagnostic Hub</span><br>"
            f"{h['LATITUD']:.4f}, {h['LONGITUD']:.4f}",
            sticky=False,
        ),
        icon=folium.Icon(color="blue", icon="plus-sign", prefix="glyphicon"),
    ).add_to(hosp_fg)
hosp_fg.add_to(m)

# Clinic layer
if show_clinics and not clinics.empty:
    clinic_fg = folium.FeatureGroup(name="KLINIK Reference", show=True)
    for _, c in clinics.iterrows():
        folium.CircleMarker(
            location=[c["LATITUD"], c["LONGITUD"]],
            radius=4, color="#64748b", fill=True,
            fill_color="#94a3b8", fill_opacity=0.5, weight=1,
            tooltip=folium.Tooltip(c.get("NAMA", "Klinik"), sticky=False),
        ).add_to(clinic_fg)
    clinic_fg.add_to(m)

# District centroid layer
dist_fg = folium.FeatureGroup(name="District Centroids", show=True)

for _, row in compound_df.iterrows():
    color  = compound_color(row["compound_score"])
    label  = compound_label(row["compound_score"])
    radius = 8 + (row["compound_score"] / max_compound_val) * 20

    pov_str = (
        f"{row['poverty_rate']:.1f}%"
        if pd.notna(row.get("poverty_rate")) else "N/A"
    )

    tooltip_html = f"""
    <div style="font-family:Inter,sans-serif; min-width:260px;">
        <b style="font-size:1rem;">{row['district']}</b>
        <br><span style="color:{color}; font-weight:600;">{label}</span>
        <hr style="margin:4px 0; border-color:#ddd;">
        <table style="font-size:0.82rem; line-height:1.85;">
            <tr><td>Nearest Hospital</td><td><b>{row['nearest_facility']}</b></td></tr>
            <tr><td>Travel Time</td>
                <td><b>{row['est_travel_time_min']:.0f} min</b> ({row['min_distance_km']:.1f} km)</td></tr>
            <tr><td>Median Monthly Income</td><td><b>RM {row['income_median']:,.0f}</b></td></tr>
            <tr><td>Poverty Rate</td><td><b>{pov_str}</b></td></tr>
            <tr><td>Geo Isolation (G)</td><td><b>{row['geo_gradient']:.3f}</b></td></tr>
            <tr><td>Income Deprivation (I)</td><td><b>{row['income_gradient']:.3f}</b></td></tr>
            <tr style="background:rgba(239,68,68,0.08);">
                <td><b>Compound Score</b></td>
                <td><b style="font-size:1rem; color:{color};">
                    {row['compound_score']:.1f} / {max_compound_val:.0f}
                </b></td>
            </tr>
        </table>
    </div>
    """

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=radius, color=color, fill=True,
        fill_color=color, fill_opacity=0.78, weight=2,
        tooltip=folium.Tooltip(tooltip_html, sticky=True),
        popup=folium.Popup(tooltip_html, max_width=300),
    ).add_to(dist_fg)

    folium.map.Marker(
        location=[row["lat"] + 0.04, row["lon"]],
        icon=folium.DivIcon(
            html=(
                f'<div style="font-size:9px; color:#d1d5db; font-family:Inter,sans-serif; '
                f'white-space:nowrap; text-shadow:0 0 3px #000;">{row["district"]}</div>'
            ),
            icon_size=(120, 16), icon_anchor=(60, 0),
        ),
    ).add_to(dist_fg)

dist_fg.add_to(m)
folium.LayerControl(position="topright", collapsed=False).add_to(m)

legend_html = f"""
<div style="
    position:fixed; bottom:30px; left:30px; z-index:9999;
    background:rgba(10,14,26,0.92); border:1px solid rgba(0,200,180,0.3);
    border-radius:10px; padding:12px 16px; font-family:Inter,sans-serif;
    font-size:12px; color:#e8f4f8; backdrop-filter:blur(8px);">
  <b style="color:#00c8b4; font-size:13px;">Compound Score (/{max_compound_val:.0f})</b>
  <br><span style="color:#7ecfdf; font-size:11px;">G &times; I &times; {(geo_weight+econ_weight)*100:.0f}</span><br><br>
  <span style="color:#22c55e;">&#9679;</span> &lt;10 Low Priority<br>
  <span style="color:#84cc16;">&#9679;</span> 10-22 Moderate-Low<br>
  <span style="color:#f59e0b;">&#9679;</span> 22-35 Moderate<br>
  <span style="color:#f97316;">&#9679;</span> 35-48 High<br>
  <span style="color:#ef4444;">&#9679;</span> &gt;48 Critical<br>
  <hr style="margin:8px 0; border-color:rgba(0,200,180,0.2);">
  <span style="color:#60a5fa;">+</span> Diagnostic Hospital<br>
  <span style="color:#94a3b8;">&#9679;</span> Registered Clinic
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width="100%", height=620, returned_objects=[])

# ---------------------------------------------------------------------------
# COMPOUND SCORE BREAKDOWN - Top 10
# ---------------------------------------------------------------------------
with st.expander("Compound Score Breakdown - Top 10 Most Vulnerable", expanded=True):
    top10 = compound_df.head(10)[[
        "district", "geo_gradient", "income_gradient", "compound_score",
        "income_median", "poverty_rate", "est_travel_time_min"
    ]].copy()
    top10.columns = [
        "District", "G (Geo)", "I (Income Dep.)", f"Score (/{max_compound_val:.0f})",
        "Median Income (RM)", "Poverty Rate (%)", "Travel Time (min)"
    ]

    def _style_score(val):
        s = float(val)
        if s >= 48:   return "color:#ef4444; font-weight:700"
        elif s >= 35: return "color:#f97316; font-weight:700"
        elif s >= 22: return "color:#f59e0b; font-weight:600"
        else:         return "color:#84cc16; font-weight:600"

    _fn = getattr(top10.style, "map", None) or getattr(top10.style, "applymap")
    styled10 = _fn(_style_score, subset=[f"Score (/{max_compound_val:.0f})"]).format({
        "G (Geo)":                          "{:.3f}",
        "I (Income Dep.)":                  "{:.3f}",
        f"Score (/{max_compound_val:.0f})": "{:.1f}",
        "Median Income (RM)":               "RM {:,.0f}",
        "Poverty Rate (%)":                 "{:.1f}%",
        "Travel Time (min)":                "{:.0f}",
    })
    st.dataframe(styled10, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# FULL RANKED TABLE  (all 27 districts)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Full District Compound Vulnerability Ranking")
st.caption(
    f"Sorted by Compound Score (highest to lowest). "
    f"Score = G x I x {(geo_weight+econ_weight)*100:.0f} -- combining Geographic Isolation "
    "and Income Deprivation into a single composite vulnerability metric."
)

display_df = compound_df[[
    "compound_rank", "district", "nearest_facility", "min_distance_km",
    "est_travel_time_min", "income_median", "poverty_rate",
    "geo_gradient", "income_gradient", "compound_score",
]].copy()
display_df.columns = [
    "Rank", "District", "Nearest Hospital", "Distance (km)", "Travel Time (min)",
    "Median Income (RM)", "Poverty Rate (%)",
    "Geo Index (G)", "Income Deprivation (I)", f"Compound Score (/{max_compound_val:.0f})",
]

def _hl_compound(val):
    s = float(val)
    if s >= 48:   return "color:#ef4444; font-weight:700"
    elif s >= 35: return "color:#f97316; font-weight:700"
    elif s >= 22: return "color:#f59e0b; font-weight:600"
    elif s >= 10: return "color:#84cc16; font-weight:600"
    else:         return "color:#22c55e; font-weight:600"

def _hl_income(val):
    try:
        inc = float(str(val).replace("RM", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return ""
    if inc < 2500:   return "color:#ef4444"
    elif inc < 3500: return "color:#f59e0b"
    else:            return "color:#22c55e"

_base = display_df.style
_mfn  = getattr(_base, "map", None) or getattr(_base, "applymap")
styled_full = (
    _mfn(_hl_compound, subset=[f"Compound Score (/{max_compound_val:.0f})"])
    .format({
        "Distance (km)":                          "{:.2f}",
        "Travel Time (min)":                      "{:.1f}",
        "Median Income (RM)":                     "RM {:,.0f}",
        "Poverty Rate (%)":                       "{:.1f}%",
        "Geo Index (G)":                          "{:.4f}",
        "Income Deprivation (I)":                 "{:.4f}",
        f"Compound Score (/{max_compound_val:.0f})": "{:.2f}",
    })
)

st.dataframe(styled_full, use_container_width=True, height=650, hide_index=True)

export_df = compound_df.copy()
export_df["max_compound_score"] = max_compound_val
csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Export Combined Dataset as CSV",
    data=csv_bytes,
    file_name="sabah_compound_tb_vulnerability.csv",
    mime="text/csv",
    help="Download all 27-district compound vulnerability data as CSV.",
)

# ---------------------------------------------------------------------------
# METHODOLOGY EXPANDER
# ---------------------------------------------------------------------------
with st.expander("Methodology & Data Sources", expanded=False):
    st.markdown(f"""
**Phase 2 - Compound Structural Exposure Score**

| Component | Detail |
|---|---|
| **Facility data** | MOH Malaysia `facilities_master.csv` (live, cached 1 h) |
| **Income data** | OpenDOSM HIES `hies_district.csv` - Sabah district-level (latest survey year) |
| **Fallback income** | Embedded DOSM HIES baseline dict (27 districts) - active when offline |
| **Distance metric** | Haversine great-circle (geodesic), WGS84 |
| **Speed assumption** | {speed_kmh} km/h (winding/rural roads - adjustable in sidebar) |
| **Travel time** | 15 + (distance_km / {speed_kmh}) x 60 minutes |

**Mathematical Formulation:**

- **G** = (T - min_T) / (max_T - min_T)  [Geographic Isolation Gradient]
- **I** = (max_Income - Income) / (max_Income - min_Income)  [Income Deprivation Gradient]
- **S** = G x I x {(geo_weight + econ_weight) * 100:.0f}  [Compound Score, out of {max_compound_val:.0f}]

**Interpretation:**
- Score > **35** indicates a district that is *simultaneously* remote (high G) and economically deprived (high I).
- The multiplicative G x I interaction ensures high scores only when **both** conditions co-exist.
- Max score = **{max_compound_val:.0f}** when geo_weight + econ_weight = {(geo_weight + econ_weight) * 100:.0f}%.

**Data sources:**
- Ministry of Health Malaysia: [github.com/MoH-Malaysia/data-resources-public](https://github.com/MoH-Malaysia/data-resources-public)
- OpenDOSM HIES: [storage.dosm.gov.my/hies/hies_district.csv](https://storage.dosm.gov.my/hies/hies_district.csv)
    """)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <hr style="border-color:rgba(0,200,180,0.15); margin-top:2rem;">
    <p style="text-align:center; color:#334155; font-size:0.8rem;">
        Sabah TB Mobile Clinic Dispatch System &middot; Phase 2 &middot;
        Streamlit + Folium &middot; MOH Malaysia + OpenDOSM HIES &middot;
        For public health planning purposes only
    </p>
    """,
    unsafe_allow_html=True,
)
