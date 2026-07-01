"""
OceanExplorer_TS.py
Versione corretta + Salinity + Hovmöller diagrams
"""

from __future__ import annotations

import io
import warnings
from datetime import datetime

import folium
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CS-MACH1 Ocean Explorer", page_icon="🌊", layout="wide")

st.markdown("""
<style>
.main-title {font-size:2.2rem; font-weight:800; color:#00A6D6;}
.section-hdr {font-size:1.3rem; font-weight:700; color:#00A6D6; border-bottom:2px solid #00A6D6; padding-bottom:6px; margin:1.5rem 0 0.8rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌊 CS-MACH1 — Ocean T+S Explorer</div>", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DEFAULT_LAT, DEFAULT_LON = 44.38, 9.07

CORA_TEMP_SURF = "https://erddap.emodnet-physics.eu/erddap/griddap/INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP.csv?TEMP%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D%5B(1.0):1:(1)%5D%5B({lat}):1:({lat})%5D%5B({lon}):1:({lon})%5D"
CORA_TEMP_DEPTH = "https://erddap.emodnet-physics.eu/erddap/griddap/INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP.csv?TEMP%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D%5B(1.0):1:({depth})%5D%5B({lat}):1:({lat})%5D%5B({lon}):1:({lon})%5D"

CORA_PSAL_SURF = "https://erddap.emodnet-physics.eu/erddap/griddap/INSITU_GLO_PHY_TS_OA_MY_013_052_PSAL.csv?PSAL%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D%5B(1.0):1:(1)%5D%5B({lat}):1:({lat})%5D%5B({lon}):1:({lon})%5D"
CORA_PSAL_DEPTH = "https://erddap.emodnet-physics.eu/erddap/griddap/INSITU_GLO_PHY_TS_OA_MY_013_052_PSAL.csv?PSAL%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D%5B(1.0):1:({depth})%5D%5B({lat}):1:({lat})%5D%5B({lon}):1:({lon})%5D"

# ── Helpers ───────────────────────────────────────────────────────────────────
def _normalize_cora(df: pd.DataFrame, var: str) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    for possible in [var, var.upper(), "SEA_WATER_TEMPERATURE", "SEA_WATER_SALINITY"]:
        if possible in df.columns:
            if possible != var:
                df = df.rename(columns={possible: var})
            break
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if "depth" not in df.columns and "z" in df.columns:
        df = df.rename(columns={"z": "depth"})
    return df


@st.cache_data(ttl=3600)
def fetch_wod_all(lat: float, lon: float):
    try:
        from beacon_api import Client
        client = Client("https://beacon-wod.maris.nl",proxy_headers={"User-Agent": "my-app/1.0 (antonio.novellino@dedagroup.it)"})
        qb = client.query()
        qb.add_select_column("Temperature", "TEMPERATURE")
        qb.add_select_column("Salinity", "PSAL")
        qb.add_select_column("z", "DEPTH")
        qb.add_select_column("time", "TIME")
        raw = qb.to_pandas_dataframe()
        for c in ["TEMPERATURE", "PSAL", "DEPTH"]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw["TIME"] = pd.to_datetime(raw["TIME"], errors="coerce")
        return raw.dropna(subset=["DEPTH"])
    except Exception as e:
        st.warning(f"WOD: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def fetch_cora_surface(lat: float, lon: float, is_salinity=False):
    url = (CORA_PSAL_SURF if is_salinity else CORA_TEMP_SURF).format(lat=round(lat,4), lon=round(lon,4))
    var = "PSAL" if is_salinity else "TEMP"
    try:
        r = requests.get(url, verify=False, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), skiprows=[1])
        df = _normalize_cora(df, var)
        df[var] = pd.to_numeric(df[var], errors="coerce").round(3)
        return df.dropna(subset=["time", var])
    except Exception as e:
        st.warning(f"CORA surface: {e}")
        return None


@st.cache_data(ttl=86400)
def fetch_cora_depth(lat: float, lon: float, max_depth: float, is_salinity=False):
    url = (CORA_PSAL_DEPTH if is_salinity else CORA_TEMP_DEPTH).format(lat=round(lat,4), lon=round(lon,4), depth=float(max_depth))
    var = "PSAL" if is_salinity else "TEMP"
    try:
        r = requests.get(url, verify=False, timeout=90)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), skiprows=[1])
        df = _normalize_cora(df, var)
        df[var] = pd.to_numeric(df[var], errors="coerce").round(3)
        if "depth" in df.columns:
            df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
        return df.dropna(subset=["time", var])
    except Exception as e:
        st.warning(f"CORA depth: {e}")
        return None


# ── Plot functions ────────────────────────────────────────────────────────────
def plot_monthly(df, var, title, color="steelblue"):
    if df is None or df.empty:
        fig, ax = plt.subplots(); ax.text(0.5,0.5,"No data"); return fig
    df = df.copy()
    df["m"] = df["time"].dt.month
    monthly = df.groupby("m")[var].agg(["mean","std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8,5))
    ax.fill_between(monthly["m"], monthly["mean"]-monthly["std"], monthly["mean"]+monthly["std"], alpha=0.25, color=color)
    ax.plot(monthly["m"], monthly["mean"], "o-", color=color, lw=2)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel("Temperature (°C)" if var=="TEMP" else "Salinity (PSU)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_depth_profile(df, var, max_depth, title, color="steelblue"):
    fig, ax = plt.subplots(figsize=(6,8))
    if df is None or df.empty or "depth" not in df.columns:
        ax.text(0.5,0.5,"No depth data", ha="center", va="center")
        fig.tight_layout()
        return fig
    profile = df.groupby("depth")[var].agg(["mean","std","median"]).reset_index()
    ax.fill_betweenx(profile["depth"], profile["mean"]-profile["std"], profile["mean"]+profile["std"], alpha=0.2, color=color)
    ax.plot(profile["mean"], profile["depth"], color=color, lw=2.5)
    ax.set_xlabel(var)
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_ylim(max_depth, 0)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_hovmoller(df, var, max_depth, title, is_cora=True):
    """Hovmöller diagram (month vs depth)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data for Hovmöller", ha="center", va="center")
        fig.tight_layout()
        return fig

    df_plot = df.copy()
    if "depth" not in df_plot.columns and "DEPTH" in df_plot.columns:
        df_plot = df_plot.rename(columns={"DEPTH": "depth"})
    if "time" not in df_plot.columns and "TIME" in df_plot.columns:
        df_plot["time"] = pd.to_datetime(df_plot["TIME"])

    df_plot = df_plot[df_plot["depth"] <= max_depth].dropna(subset=["time", "depth", var])
    df_plot["month"] = df_plot["time"].dt.month
    depth_bin = 10
    df_plot["DEPTH_BIN"] = np.round(df_plot["depth"] / depth_bin) * depth_bin

    monthly = df_plot.groupby(["month", "DEPTH_BIN"])[var].mean().reset_index()
    if monthly.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return fig

    hov = monthly.pivot(index="DEPTH_BIN", columns="month", values=var).sort_index()
    cf = ax.contourf(hov.columns, hov.index, hov.values, levels=30, cmap="RdYlBu_r", extend="both")
    ax.contour(hov.columns, hov.index, hov.values, levels=10, colors="k", linewidths=0.3, alpha=0.4)
    fig.colorbar(cf, ax=ax, label=var)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel("Month")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📍 Location")
    lat = st.number_input("Latitude (°N)", -90.0, 90.0, DEFAULT_LAT, step=0.01, format="%.4f")
    lon = st.number_input("Longitude (°E)", -180.0, 180.0, DEFAULT_LON, step=0.01, format="%.4f")
    max_depth = st.slider("Max depth (m)", 10, 5000, 300, step=10)
    run_btn = st.button("▶️ Run Analysis", type="primary", use_container_width=True)

# Mappa
m = folium.Map(location=[lat, lon], zoom_start=6, tiles="CartoDB positron")
folium.Marker([lat, lon]).add_to(m)
map_data = st_folium(m, width=700, height=420, returned_objects=["last_clicked"])

if map_data and map_data.get("last_clicked"):
    cl = map_data["last_clicked"]
    st.session_state.lat = round(cl["lat"],4)
    st.session_state.lon = round(cl["lng"],4)
    st.rerun()

# ── Analysis ──────────────────────────────────────────────────────────────────
if run_btn or st.session_state.get("results"):
    if run_btn:
        with st.spinner("Downloading data..."):
            res = {
                "cora_t_surf": fetch_cora_surface(lat, lon, False),
                "cora_s_surf": fetch_cora_surface(lat, lon, True),
                "cora_t_dep": fetch_cora_depth(lat, lon, max_depth, False),
                "cora_s_dep": fetch_cora_depth(lat, lon, max_depth, True),
                "lat": lat, "lon": lon
            }
            st.session_state.results = res

    res = st.session_state.results

    # Temperature
    st.markdown("<div class='section-hdr'>🌡️ TEMPERATURE</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_monthly(res["cora_t_surf"], "TEMP", "CORA Temperature Monthly Mean"))
        st.pyplot(plot_hovmoller(res["cora_t_dep"], "TEMP", max_depth, "CORA Temperature Hovmöller", True))
    with c2:
        st.pyplot(plot_depth_profile(res["cora_t_dep"], "TEMP", max_depth, "CORA Temperature Depth Profile"))

    # Salinity
    st.markdown("<div class='section-hdr'>🌊 SALINITY (PSAL)</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_monthly(res["cora_s_surf"], "PSAL", "CORA Salinity Monthly Mean", "teal"))
        st.pyplot(plot_hovmoller(res["cora_s_dep"], "PSAL", max_depth, "CORA Salinity Hovmöller", True))
    with c2:
        st.pyplot(plot_depth_profile(res["cora_s_dep"], "PSAL", max_depth, "CORA Salinity Depth Profile", "teal"))

    st.success("✅ Analisi completata")

st.caption("CORA (EMODnet) + WOD")
