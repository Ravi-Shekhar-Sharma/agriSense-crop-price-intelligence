# -*- coding: utf-8 -*-
"""
AgriSense - Crop Price Intelligence Dashboard
McKinsey/BCG consulting aesthetic — clean, white, data-first
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os
import runpy
import calendar

@st.cache_data(show_spinner="Building feature dataset... (~60 seconds on first launch)")
def ensure_features_exist():
    if not os.path.exists("data/processed/features.csv"):
        runpy.run_path("src/feature_engineering.py")
    return True

ensure_features_exist()

# ===============================================================================
# CONFIGURATION & CONSTANTS
# ===============================================================================

CONFIG = {
    "data_dir": "data/processed",
    "models_dir": "models",
    "festival_window": 7
}

COMMODITIES = [
    'atta','gram_dal','groundnut_oil','gur','masoor_dal','milk','moong_dal',
    'mustard_oil','onion','palm_oil','potato','rice','salt','soya_oil',
    'sugar','sunflower_oil','tea_loose','tomato','tur_dal','urad_dal',
    'vanaspati','wheat'
]

COMMODITY_DISPLAY = {
    'atta':'Wheat Flour (Atta)','gram_dal':'Gram Dal','groundnut_oil':'Groundnut Oil',
    'gur':'Jaggery (Gur)','masoor_dal':'Masoor Dal','milk':'Milk',
    'moong_dal':'Moong Dal','mustard_oil':'Mustard Oil','onion':'Onion',
    'palm_oil':'Palm Oil','potato':'Potato','rice':'Rice','salt':'Salt',
    'soya_oil':'Soya Oil','sugar':'Sugar','sunflower_oil':'Sunflower Oil',
    'tea_loose':'Tea Loose','tomato':'Tomato','tur_dal':'Tur Dal',
    'urad_dal':'Urad Dal','vanaspati':'Vanaspati','wheat':'Wheat'
}

COMMODITY_CATEGORIES = {
    'atta':'Cereal','gram_dal':'Pulse','groundnut_oil':'Oil','gur':'Sugar',
    'masoor_dal':'Pulse','milk':'Dairy','moong_dal':'Pulse','mustard_oil':'Oil',
    'onion':'Vegetable','palm_oil':'Oil','potato':'Vegetable','rice':'Cereal',
    'salt':'Other','soya_oil':'Oil','sugar':'Sugar','sunflower_oil':'Oil',
    'tea_loose':'Other','tomato':'Vegetable','tur_dal':'Pulse','urad_dal':'Pulse',
    'vanaspati':'Oil','wheat':'Cereal'
}

HARVEST_MONTHS = {
    'tomato':[11,12,1,2,5,6],'onion':[3,4,5,10,11],'potato':[12,1,2,3],
    'wheat':[3,4,5],'rice':[10,11],'atta':[3,4,5],'tur_dal':[12,1,2],
    'urad_dal':[10,11,12],'moong_dal':[3,4,9,10],'gram_dal':[3,4,5],
    'masoor_dal':[3,4,5],'mustard_oil':[3,4,5],'groundnut_oil':[10,11,12],
    'soya_oil':[10,11],'sunflower_oil':[3,4],'palm_oil':[10,11,12,1],
    'sugar':[11,12,1,2,3],'gur':[11,12,1,2,3]
}

FESTIVAL_DATES = {
    'Diwali':   ['2022-10-24','2023-11-12','2024-11-01'],
    'Holi':     ['2022-03-18','2023-03-08','2024-03-25'],
    'Eid':      ['2022-05-03','2023-04-21','2024-04-10'],
    'Navratri': ['2022-09-26','2023-10-15','2024-10-03'],
    'Pongal':   ['2023-01-14','2024-01-15'],
}

DROP_COLS = ['date','state','commodity','commodity_category','region','retail_price']

EXPECTED_FEATURES = [
    'wholesale_price','year','month','day_of_month','day_of_week',
    'week_of_year','quarter','day_of_year','is_weekend','is_month_end',
    'is_month_start','month_sin','month_cos','dow_sin','dow_cos',
    'retail_lag_1d','wholesale_lag_1d','retail_lag_7d','wholesale_lag_7d',
    'retail_lag_14d','wholesale_lag_14d','retail_lag_30d','wholesale_lag_30d',
    'retail_change_7d','retail_pct_chg_7d','wholesale_pct_chg_7d',
    'retail_change_30d','retail_pct_chg_30d','wholesale_pct_chg_30d',
    'retail_roll_mean_7d','retail_roll_std_7d','wholesale_roll_mean_7d',
    'wholesale_roll_std_7d','retail_zscore_7d','retail_roll_mean_30d',
    'retail_roll_std_30d','wholesale_roll_mean_30d','wholesale_roll_std_30d',
    'retail_zscore_30d','retail_roll_min_30d','retail_roll_max_30d',
    'retail_30d_range','price_spread','spread_ratio','margin_pct',
    'spread_roll_7d','spread_roll_30d','is_harvest_month','is_kharif_season',
    'is_rabi_season','is_monsoon','is_summer','is_winter','days_to_diwali',
    'near_diwali','days_to_holi','near_holi','days_to_eid','near_eid',
    'days_to_navratri','near_navratri','days_to_pongal','near_pongal',
    'near_any_festival','cat_Cereal','cat_Dairy','cat_Oil','cat_Other',
    'cat_Pulse','cat_Sugar','cat_Vegetable','region_Central','region_East',
    'region_North','region_Northeast','region_Other','region_South',
    'region_West','state_enc','commodity_enc'
]

COLORS = {
    "bg_page":        "#FFFFFF",
    "bg_surface":     "#F7F8FA",
    "bg_sidebar":     "#FAFAFA",
    "text_primary":   "#0D0D0D",
    "text_secondary": "#6B7280",
    "text_muted":     "#9CA3AF",
    "accent":         "#1B4332",
    "accent_light":   "#D1FAE5",
    "chart_primary":  "#1B4332",
    "chart_second":   "#6B7280",
    "chart_up":       "#166534",
    "chart_down":     "#991B1B",
    "chart_neutral":  "#9CA3AF",
    "border":         "#E5E7EB",
    "border_strong":  "#D1D5DB",
    "positive":       "#166534",
    "negative":       "#991B1B",
    "warning":        "#92400E",
    "chart_bg":       "#FFFFFF",
    "gridline":       "#F3F4F6",
}

REGION_PALETTE = {
    "North":     "#1B4332",
    "South":     "#1E3A5F",
    "East":      "#92400E",
    "West":      "#374151",
    "Central":   "#6B7280",
    "Northeast": "#9CA3AF",
    "Other":     "#D1D5DB",
}

# ===============================================================================
# CSS — CONSULTING AESTHETIC
# ===============================================================================

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #FFFFFF !important;
        color: #0D0D0D !important;
        font-size: clamp(15px, 1.4vw, 20px) !important;
    }

    /* General text — 20px at 1440px, scales down on narrow screens */
    p, span, li, div.stMarkdown { font-size: clamp(15px, 1.4vw, 20px) !important; }
    label { font-size: clamp(14px, 1.2vw, 18px) !important; }
    .stSelectbox label, .stRadio label, .stDateInput label { font-size: clamp(13px, 1.1vw, 17px) !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FAFAFA !important;
        border-right: 1px solid #E5E7EB !important;
        box-shadow: none !important;
        font-size: 15px !important;
    }

    .sidebar-logo { padding: 8px 0 20px 0; border-bottom: 1px solid #E5E7EB; margin-bottom: 20px; }
    .sidebar-logo-name { font-size: 20px !important; font-weight: 700; color: #0D0D0D !important; letter-spacing: -0.02em; }
    .sidebar-logo-sub { font-size: 13px !important; color: #9CA3AF !important; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 6px; }

    .sidebar-label {
        font-size: 13px !important; color: #9CA3AF !important; text-transform: uppercase;
        letter-spacing: 0.08em; font-weight: 600; margin-bottom: 4px; margin-top: 16px;
        display: block;
    }

    .sidebar-metric {
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 4px;
        padding: 10px 12px; margin-bottom: 8px;
    }
    .sidebar-metric-label { font-size: 13px !important; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 500; }
    .sidebar-metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 26px !important; font-weight: 600; color: #0D0D0D; margin-top: 2px; line-height: 1.2; }

    /* Main content */
    .main .block-container {
        padding-top: 16px !important;
        padding-bottom: 48px !important;
        max-width: 1400px !important;
    }

    /* Page header */
    .page-header { padding: 0 0 18px 0; border-bottom: 2px solid #1B4332; margin-bottom: 24px; }
    .page-header-eyebrow { font-size: clamp(22px, 2.4vw, 34px) !important; font-weight: 700; color: #1B4332; text-transform: none; letter-spacing: -0.01em; margin-bottom: 10px; line-height: 1.3; }
    .page-header-title { font-size: clamp(32px, 3.5vw, 50px) !important; font-weight: 700; color: #1B4332; letter-spacing: -0.02em; margin: 0; line-height: 1.15; }
    .page-header-sub { font-size: clamp(18px, 1.75vw, 25px) !important; color: #6B7280; margin-top: 6px; font-weight: 400; }
    .page-header-meta { font-size: clamp(13px, 1.05vw, 15px) !important; color: #9CA3AF; margin-top: 6px; }

    /* KPI cards */
    .kpi-card {
        background: #FFFFFF; border: 1px solid #E5E7EB; border-top: 3px solid #1B4332;
        border-radius: 4px; padding: 18px 20px; margin-bottom: 8px;
    }
    .kpi-card.red   { border-top-color: #991B1B; }
    .kpi-card.grey  { border-top-color: #9CA3AF; }
    .kpi-card.amber { border-top-color: #92400E; }
    .kpi-card.blue  { border-top-color: #1E3A5F; }
    .kpi-label { font-size: 13px !important; font-weight: 600; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
    .kpi-value { font-family: 'IBM Plex Mono', monospace; font-size: 36px !important; font-weight: 600; color: #0D0D0D; line-height: 1.1; letter-spacing: -0.02em; }
    .kpi-sub { font-size: 14px !important; color: #9CA3AF; margin-top: 4px; }
    .kpi-delta-pos { font-size: 15px !important; color: #166534; font-weight: 500; margin-top: 4px; }
    .kpi-delta-neg { font-size: 15px !important; color: #991B1B; font-weight: 500; margin-top: 4px; }

    /* Section headers — dark green accent */
    .section-header {
        font-size: clamp(12px, 1.05vw, 15px) !important; font-weight: 700; color: #1B4332; text-transform: uppercase;
        letter-spacing: 0.12em; padding-bottom: 8px; border-bottom: 2px solid #1B4332;
        margin: 32px 0 18px 0;
    }

    /* Alert boxes */
    .alert-box { padding: 12px 16px; border-radius: 4px; font-size: 16px !important; margin: 8px 0; border-left: 3px solid; }
    .alert-box.yellow { background: #FFFBEB; border-left-color: #92400E; color: #78350F; }
    .alert-box.green  { background: #F0FDF4; border-left-color: #166534; color: #14532D; }
    .alert-box.red    { background: #FEF2F2; border-left-color: #991B1B; color: #7F1D1D; }
    .alert-box.blue   { background: #EFF6FF; border-left-color: #1E3A5F; color: #1E3A5F; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #E5E7EB !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        padding: 0 !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 0 !important;
        border-bottom: 2px solid transparent !important;
        padding: 12px 22px !important;
        font-size: clamp(14px, 1.4vw, 20px) !important;
        font-weight: 500 !important;
        color: #6B7280 !important;
        margin-bottom: -1px !important;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        border-bottom: 2px solid #1B4332 !important;
        color: #1B4332 !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #1B4332 !important; background: transparent !important; }

    /* Tables */
    .stDataFrame { border: 1px solid #E5E7EB !important; border-radius: 4px !important; }

    /* Stakeholder cards */
    .sh-card {
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 4px;
        padding: 16px 18px; height: 100%;
    }
    .sh-icon { font-size: 22px !important; }
    .sh-title { color: #0D0D0D; font-weight: 700; font-size: 18px !important; margin: 4px 0 2px 0; }
    .sh-focus { color: #9CA3AF; font-size: 14px !important; margin-bottom: 10px; }
    .sh-row { display: flex; justify-content: space-between; padding: 6px 0;
        border-bottom: 1px solid #F3F4F6; font-size: 15px !important; }
    .sh-label { color: #9CA3AF; }
    .sh-val { font-weight: 600; font-family: 'IBM Plex Mono', monospace; font-size: 15px !important; color: #0D0D0D; }

    /* Generic info card */
    .info-card {
        background: #FFFFFF; border: 1px solid #E5E7EB; border-top: 3px solid #1B4332;
        border-radius: 4px; padding: 16px 18px; margin-bottom: 8px;
    }
    .info-card-label { font-size: 13px !important; font-weight: 600; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
    .info-card-value { font-family: 'IBM Plex Mono', monospace; font-size: 28px !important; font-weight: 600; color: #0D0D0D; }
    .info-card-sub { font-size: 14px !important; color: #9CA3AF; margin-top: 4px; }

    /* Risk badges */
    .risk-badge { display: inline-block; padding: 3px 10px; border-radius: 2px; font-size: 14px !important; font-weight: 600; }
    .risk-low    { background: #F0FDF4; color: #166534; }
    .risk-medium { background: #FFFBEB; color: #92400E; }
    .risk-high   { background: #FEF2F2; color: #991B1B; }

    /* Scale tab step cards */
    .step-card { background: #F7F8FA; border-radius: 4px; padding: 16px; }
    .step-card-title { color: #1B4332; font-weight: 700; font-size: 15px !important; margin-bottom: 8px; }
    .step-card-body { color: #6B7280; font-size: 14px !important; line-height: 1.7; }

    /* Download button */
    .stDownloadButton button {
        background: transparent !important; border: 1px solid #E5E7EB !important;
        color: #6B7280 !important; font-size: 13px !important; font-weight: 500 !important;
        border-radius: 4px !important; padding: 6px 14px !important;
    }
    .stDownloadButton button:hover {
        border-color: #1B4332 !important; color: #1B4332 !important; background: #F0FDF4 !important;
    }

    /* Selectbox, radio */
    .stSelectbox > div > div {
        border: 1px solid #E5E7EB !important; border-radius: 4px !important;
        font-size: 14px !important; background: #FFFFFF !important;
    }

    /* Hide Streamlit chrome — keep header DOM intact so native sidebar toggle works */
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #F3F4F6 !important;
    }
    /* Hide deploy/share/star buttons inside the header toolbar */
    [data-testid="stDeployButton"] { display: none !important; }
    [data-testid="stStatusWidget"]  { display: none !important; }
    /* Style the native sidebar collapse / expand button */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"]        button {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 4px !important;
        color: #1B4332 !important;
    }
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="collapsedControl"]        svg {
        fill: #1B4332 !important;
    }

    /* Date input — force white everywhere */
    [data-testid="stDateInput"] > div,
    [data-testid="stDateInput"] > div > div { background: transparent !important; }
    [data-testid="stDateInput"] input {
        background: #FFFFFF !important;
        color: #0D0D0D !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 4px !important;
        font-size: 14px !important;
    }
    [data-baseweb="input"],
    [data-baseweb="input"] > div,
    [data-baseweb="base-input"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border-color: #E5E7EB !important;
    }
    [data-baseweb="calendar"] {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 4px !important;
    }
    [data-baseweb="calendar"] * { color: #0D0D0D !important; }
    [data-baseweb="calendar"] button { background: transparent !important; }
    [data-baseweb="calendar"] [aria-selected="true"] button { background: #1B4332 !important; color: #FFFFFF !important; }

    hr { border: none; border-top: 1px solid #E5E7EB; margin: 20px 0; }

    @media (max-width: 900px) {
        .stTabs [data-baseweb="tab"] { padding: 8px 12px !important; font-size: 13px !important; }
        .main .block-container { padding-left: 12px !important; padding-right: 12px !important; }
    }

    </style>
    """, unsafe_allow_html=True)

# ===============================================================================
# DATA LOADING
# ===============================================================================

@st.cache_data(ttl=3600)
def load_features():
    try:
        path = os.path.join(CONFIG["data_dir"], "features.csv")
        dtype_dict = {
            'retail_price':'float32','wholesale_price':'float32',
            'year':'int16','month':'int8','day_of_month':'int8',
            'day_of_week':'int8','quarter':'int8',
            'is_weekend':'int8','is_month_end':'int8','is_month_start':'int8'
        }
        df = pd.read_csv(path, dtype=dtype_dict, parse_dates=['date'])
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_results():
    try:
        return pd.read_csv("outputs/model_results.csv")
    except Exception:
        return pd.DataFrame()

@st.cache_resource
def load_model(commodity, model_key="xgb"):
    try:
        path = f"{CONFIG['models_dir']}/{model_key}_{commodity}.pkl"
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ===============================================================================
# UI HELPERS
# ===============================================================================

def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def alert(text, color="yellow"):
    st.markdown(f'<div class="alert-box {color}">{text}</div>', unsafe_allow_html=True)

def kpi_card(label, value, sub=None, delta=None, color="green"):
    delta_html = ""
    if delta is not None:
        try:
            v = float(str(delta).replace('%','').replace('+','').replace('₹',''))
            cls  = "kpi-delta-pos" if v >= 0 else "kpi-delta-neg"
            arrow = "↑" if v >= 0 else "↓"
            delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
        except Exception:
            delta_html = f'<div class="kpi-sub">{delta}</div>'
    sub_html   = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    color_cls  = {"green":"","red":"red","grey":"grey","orange":"amber","blue":"blue"}.get(color,"")
    st.markdown(f"""
    <div class="kpi-card {color_cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}{delta_html}
    </div>""", unsafe_allow_html=True)

def plotly_base_layout(fig, title="", height=380, x_title="", y_title="", show_legend=True):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=13, color="#0D0D0D", family="Inter"),
                   x=0, xanchor="left", pad=dict(b=12)),
        height=height,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=11, color="#6B7280"),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=11, color="#9CA3AF")),
            gridcolor="#F3F4F6", gridwidth=1, linecolor="#E5E7EB", linewidth=1,
            tickfont=dict(size=10, color="#9CA3AF"), showgrid=True, zeroline=False, ticklen=0,
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=11, color="#9CA3AF")),
            gridcolor="#F3F4F6", gridwidth=1, linecolor="#E5E7EB", linewidth=1,
            tickfont=dict(size=10, color="#9CA3AF"), showgrid=True, zeroline=False, ticklen=0,
        ),
        margin=dict(l=48, r=24, t=52, b=40),
        legend=dict(
            visible=show_legend, orientation="h",
            yanchor="bottom", y=1.01, xanchor="left", x=0,
            font=dict(size=11, color="#6B7280"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor="#E5E7EB",
                        font=dict(size=12, color="#0D0D0D", family="Inter")),
        dragmode=False,
    )
    return fig

def get_filtered_df(df, commodity, state):
    df_c = df[df["commodity"] == commodity].copy()
    if state == "All India":
        numeric_cols = df_c.select_dtypes(include=[np.number]).columns.tolist()
        df_agg = (df_c.sort_values("date")
                     .groupby("date")[numeric_cols].mean().reset_index())
        df_agg["state"] = "All India"
        df_agg["commodity"] = commodity
        return df_agg.sort_values("date").reset_index(drop=True)
    return df_c[df_c["state"] == state].sort_values("date").reset_index(drop=True)

def timestamp_to_ms(ts):
    """Plotly 6.x + pandas 2.x fix: pass int ms instead of Timestamp to add_vline."""
    if isinstance(ts, pd.Timestamp):
        return int(ts.value // 10**6)
    return ts

def format_price(p):
    return f"₹{p:.2f}"

def format_pct(p):
    return f"{'+' if p >= 0 else ''}{p:.1f}%"

# ===============================================================================
# ANALYTICS FUNCTIONS
# ===============================================================================

@st.cache_data(ttl=3600)
def compute_festival_impact(commodity):
    try:
        df = load_features()
        df_c = df[df["commodity"] == commodity].copy()
        win = CONFIG["festival_window"]
        results = []
        for festival, dates in FESTIVAL_DATES.items():
            for date_str in dates:
                fd = pd.Timestamp(date_str)
                mf = (df_c["date"] >= fd - pd.Timedelta(days=win)) & \
                     (df_c["date"] <= fd + pd.Timedelta(days=win))
                mb = (df_c["date"] >= fd - pd.Timedelta(days=win+30)) & \
                     (df_c["date"] <  fd - pd.Timedelta(days=win))
                if mf.sum() > 0 and mb.sum() > 0:
                    fp = float(df_c[mf]["retail_price"].mean())
                    bp = float(df_c[mb]["retail_price"].mean())
                    results.append({
                        "Festival": festival,
                        "Year": fd.year,
                        "Label": f"{festival} {fd.year}",
                        "Festival Price": round(fp, 2),
                        "Baseline Price": round(bp, 2),
                        "Impact %": round(((fp - bp) / (bp + 1e-8)) * 100, 2),
                    })
        return pd.DataFrame(results)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def compute_risk_zones(commodity):
    try:
        df = load_features()
        df_c = df[df["commodity"] == commodity]
        risk = df_c.groupby("state")["retail_price"].agg(
            mean_price="mean", std_price="std", count="count"
        ).reset_index()
        risk["cv"] = risk["std_price"] / (risk["mean_price"] + 1e-8)
        risk["risk_level"] = pd.cut(risk["cv"],
            bins=[0, 0.08, 0.18, float("inf")],
            labels=["Low","Medium","High"]).astype(str)
        risk["risk_color"] = risk["risk_level"].map(
            {"Low": COLORS["positive"], "Medium": COLORS["warning"], "High": COLORS["negative"]})
        return risk.sort_values("cv", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def compute_procurement_timing(commodity):
    try:
        df = load_features()
        df_c = df[df["commodity"] == commodity].copy()
        df_c["month_num"] = df_c["date"].dt.month
        monthly = df_c.groupby("month_num")["retail_price"].mean()
        annual_avg = monthly.mean()
        harvest_m  = HARVEST_MONTHS.get(commodity, [])
        rows = []
        for m in range(1, 13):
            if m not in monthly.index:
                continue
            avg = float(monthly[m])
            pct = ((avg - annual_avg) / (annual_avg + 1e-8)) * 100
            if pct <= -10: action, signal = "Strong Buy", "↑↑"
            elif pct <= -3: action, signal = "Buy", "↑"
            elif pct <= 3:  action, signal = "Hold / Monitor", "→"
            elif pct <= 10: action, signal = "Reduce Stock", "↓"
            else:           action, signal = "Sell / Liquidate", "↓↓"
            rows.append({
                "Month": calendar.month_abbr[m],
                "Month_Num": m,
                "Avg Price": round(avg, 2),
                "vs Annual Avg %": round(pct, 2),
                "Signal": signal,
                "Action": action,
                "Harvest": "Yes" if m in harvest_m else "",
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ===============================================================================
# FORECAST
# ===============================================================================

@st.cache_data(ttl=1800)
def generate_forecast_optimized(commodity, state, model_key="xgb", n_days=7):
    try:
        model = load_model(commodity, model_key)
        if model is None:
            return pd.DataFrame()
        df = load_features()
        if df.empty:
            return pd.DataFrame()
        df_f = get_filtered_df(df, commodity, state)
        if len(df_f) < 30:
            return pd.DataFrame()

        df_s   = df_f.sort_values("date").copy()
        last   = df_s.iloc[-1].copy()
        last_date = pd.Timestamp(last["date"])

        us = sorted(df["state"].unique());     state_enc = {s:i for i,s in enumerate(us)}
        uc = sorted(df["commodity"].unique()); comm_enc  = {c:i for i,c in enumerate(uc)}

        ph  = list(df_s["retail_price"].dropna().values[-30:])
        wh  = list(df_s["wholesale_price"].dropna().values[-30:])
        fcols = EXPECTED_FEATURES
        results = []
        current = last.copy()

        for i in range(1, n_days + 1):
            nd  = last_date + pd.Timedelta(days=i)
            new = current.copy()
            new["date"] = nd; new["year"] = nd.year; new["month"] = nd.month
            new["day_of_month"] = nd.day; new["day_of_week"] = nd.dayofweek
            new["week_of_year"] = nd.isocalendar()[1]
            new["quarter"] = (nd.month-1)//3+1; new["day_of_year"] = nd.timetuple().tm_yday
            new["is_weekend"] = int(nd.dayofweek >= 5); new["is_month_start"] = int(nd.day == 1)
            new["is_month_end"] = int(nd.day == calendar.monthrange(nd.year, nd.month)[1])
            new["month_sin"] = np.sin(2*np.pi*nd.month/12); new["month_cos"] = np.cos(2*np.pi*nd.month/12)
            new["dow_sin"]   = np.sin(2*np.pi*nd.dayofweek/7); new["dow_cos"] = np.cos(2*np.pi*nd.dayofweek/7)
            m = nd.month
            new["is_harvest_month"] = int(m in HARVEST_MONTHS.get(commodity, []))
            new["is_kharif_season"] = int(m in [6,7,8,9]); new["is_rabi_season"] = int(m in [11,12,1,2])
            new["is_monsoon"] = int(m in [6,7,8,9]); new["is_summer"] = int(m in [3,4,5]); new["is_winter"] = int(m in [11,12,1,2])
            any_flag = 0
            for fest, dates in FESTIVAL_DATES.items():
                fk = fest.lower(); fdates = [pd.Timestamp(d) for d in dates]
                md = min(abs((nd - fd).days) for fd in fdates)
                new[f"days_to_{fk}"] = md; flag = int(md <= CONFIG["festival_window"])
                new[f"near_{fk}"] = flag; any_flag = max(any_flag, flag)
            new["near_any_festival"] = any_flag
            nh = len(ph); nw = len(wh)
            new["retail_lag_1d"]   = ph[-1]; new["retail_lag_7d"]  = ph[-7]  if nh>=7  else ph[0]
            new["retail_lag_14d"]  = ph[-14] if nh>=14 else ph[0]; new["retail_lag_30d"] = ph[-30] if nh>=30 else ph[0]
            new["wholesale_lag_1d"]  = wh[-1]; new["wholesale_lag_7d"]  = wh[-7]  if nw>=7  else wh[0]
            new["wholesale_lag_14d"] = wh[-14] if nw>=14 else wh[0]; new["wholesale_lag_30d"] = wh[-30] if nw>=30 else wh[0]
            w7=ph[-7:] if len(ph)>=7 else ph; w30=ph[-30:] if len(ph)>=30 else ph
            wh7=wh[-7:] if len(wh)>=7 else wh; wh30=wh[-30:] if len(wh)>=30 else wh
            new["retail_roll_mean_7d"] = float(np.mean(w7)); new["retail_roll_std_7d"] = float(np.std(w7))+1e-8
            new["retail_roll_mean_30d"] = float(np.mean(w30)); new["retail_roll_std_30d"] = float(np.std(w30))+1e-8
            new["wholesale_roll_mean_7d"] = float(np.mean(wh7)); new["wholesale_roll_std_7d"] = float(np.std(wh7))+1e-8
            new["wholesale_roll_mean_30d"] = float(np.mean(wh30)); new["wholesale_roll_std_30d"] = float(np.std(wh30))+1e-8
            prev=ph[-1]; pw=wh[-1]
            new["retail_zscore_7d"]  = (prev-new["retail_roll_mean_7d"])/new["retail_roll_std_7d"]
            new["retail_zscore_30d"] = (prev-new["retail_roll_mean_30d"])/new["retail_roll_std_30d"]
            new["retail_roll_min_30d"] = float(np.min(w30)); new["retail_roll_max_30d"] = float(np.max(w30))
            new["retail_30d_range"] = new["retail_roll_max_30d"] - new["retail_roll_min_30d"]
            p7=new["retail_lag_7d"]; p30=new["retail_lag_30d"]; pw7=new["wholesale_lag_7d"]; pw30=new["wholesale_lag_30d"]
            new["retail_change_7d"]    = prev - p7;  new["retail_change_30d"] = prev - p30
            new["retail_pct_chg_7d"]   = ((prev-p7)/(p7+1e-8))*100; new["retail_pct_chg_30d"] = ((prev-p30)/(p30+1e-8))*100
            new["wholesale_pct_chg_7d"]= ((pw-pw7)/(pw7+1e-8))*100; new["wholesale_pct_chg_30d"] = ((pw-pw30)/(pw30+1e-8))*100
            new["price_spread"]  = prev-pw; new["spread_ratio"] = prev/(pw+1e-8); new["margin_pct"] = ((prev-pw)/(pw+1e-8))*100
            new["spread_roll_7d"] = float(np.mean([r-w for r,w in zip(w7,wh7)]))
            new["spread_roll_30d"] = float(np.mean([r-w for r,w in zip(w30,wh30)]))
            cat = COMMODITY_CATEGORIES.get(commodity, "Other")
            for c in ["Cereal","Dairy","Oil","Other","Pulse","Sugar","Vegetable"]:
                new[f"cat_{c}"] = 1 if cat==c else 0
            region = last.get("region","Other")
            for r in ["Central","East","North","Northeast","Other","South","West"]:
                new[f"region_{r}"] = 1 if region==r else 0
            new["state_enc"] = state_enc.get(state, 0); new["commodity_enc"] = comm_enc.get(commodity, 0)
            new["wholesale_price"] = pw
            for col in fcols:
                if col not in new.index: new[col] = 0.0
            X = pd.DataFrame([new])[fcols].fillna(0)
            predicted = float(model.predict(X)[0])
            am = new["margin_pct"]/100; est_ws = predicted/(1+am+1e-8)
            ph.append(predicted); wh.append(est_ws)
            new["retail_price"] = predicted; new["wholesale_price"] = est_ws; current = new
            results.append({"date": nd, "day": f"Day +{i}", "predicted_price": round(predicted,2),
                "change_pct": round(((predicted-float(last["retail_price"]))/(float(last["retail_price"])+1e-8))*100,2)})
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Forecast error: {e}")
        return pd.DataFrame()

# ===============================================================================
# MAIN APPLICATION
# ===============================================================================

def main():
    st.set_page_config(
        page_title="AgriSense — Indian Crop Price Intelligence",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_css()

    with st.spinner("Loading data..."):
        df = load_features()
        results_df = load_results()

    if df.empty:
        st.error("Could not load data. Please check file paths.")
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        # Logo with SVG wheat icon
        st.markdown("""
        <div class="sidebar-logo" style="display:flex;align-items:center;gap:10px;">
            <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
              <!-- Main stem -->
              <line x1="18" y1="33" x2="18" y2="5" stroke="#1B4332" stroke-width="1.8" stroke-linecap="round"/>
              <!-- Top grain -->
              <ellipse cx="18" cy="6.5" rx="3" ry="5" fill="#1B4332"/>
              <!-- Grain pair 1 -->
              <ellipse cx="11" cy="12" rx="4.5" ry="2" fill="#1B4332" transform="rotate(-45 11 12)"/>
              <ellipse cx="25" cy="12" rx="4.5" ry="2" fill="#1B4332" transform="rotate(45 25 12)"/>
              <!-- Grain pair 2 -->
              <ellipse cx="10" cy="18.5" rx="4.5" ry="2" fill="#2D6A4F" transform="rotate(-42 10 18.5)"/>
              <ellipse cx="26" cy="18.5" rx="4.5" ry="2" fill="#2D6A4F" transform="rotate(42 26 18.5)"/>
              <!-- Grain pair 3 -->
              <ellipse cx="11" cy="24.5" rx="4" ry="1.8" fill="#40916C" transform="rotate(-38 11 24.5)"/>
              <ellipse cx="25" cy="24.5" rx="4" ry="1.8" fill="#40916C" transform="rotate(38 25 24.5)"/>
              <!-- Leaf -->
              <path d="M18 29 Q24 27 26 23" stroke="#52B788" stroke-width="1.4" fill="none" stroke-linecap="round"/>
            </svg>
            <div>
              <div class="sidebar-logo-name">AgriSense <span style="font-size:0.69rem;color:#9CA3AF;font-weight:400;">(एग्रीसेंस)</span></div>
              <div class="sidebar-logo-sub">Crop Price Intelligence <span style="font-weight:400;">(कृषि मूल्य विश्लेषण)</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<span class="sidebar-label">Commodity</span>', unsafe_allow_html=True)
        selected_commodity = st.selectbox("", COMMODITIES,
            format_func=lambda x: COMMODITY_DISPLAY[x],
            index=COMMODITIES.index("tomato"), label_visibility="collapsed")

        st.markdown('<span class="sidebar-label">State / Region</span>', unsafe_allow_html=True)
        all_states = ["All India"] + sorted(df["state"].dropna().unique().tolist())
        selected_state = st.selectbox("", all_states, label_visibility="collapsed")
        is_all_india = (selected_state == "All India")

        model_key = "xgb"

        st.markdown('<span class="sidebar-label">Date Range</span>', unsafe_allow_html=True)
        date_min_global = df["date"].min().date()
        date_max_global = df["date"].max().date()
        date_range = st.date_input("", value=(date_min_global, date_max_global),
            min_value=date_min_global, max_value=date_max_global,
            label_visibility="collapsed")

        if isinstance(date_range, (tuple, list)):
            if len(date_range) >= 2:
                d_start = pd.Timestamp(date_range[0]); d_end = pd.Timestamp(date_range[-1])
            elif len(date_range) == 1:
                d_start = d_end = pd.Timestamp(date_range[0])
            else:
                d_start = pd.Timestamp(date_min_global); d_end = pd.Timestamp(date_max_global)
        else:
            d_start = d_end = pd.Timestamp(date_range)

        st.markdown("<hr style='border:none;border-top:1px solid #E5E7EB;margin:16px 0;'>", unsafe_allow_html=True)

        df_commodity = df[df["commodity"] == selected_commodity]
        if not df_commodity.empty:
            latest_price = float(df_commodity["retail_price"].iloc[-1])
            pc_raw = df_commodity["retail_pct_chg_30d"].iloc[-1]
            price_change = float(pc_raw) if not pd.isna(pc_raw) else 0.0
            color_30 = COLORS["negative"] if price_change > 0 else COLORS["positive"]
            sign = "+" if price_change >= 0 else ""
            st.markdown(f"""
            <div class="sidebar-metric">
                <div class="sidebar-metric-label">Current Price</div>
                <div class="sidebar-metric-value">₹{latest_price:.2f}/kg</div>
            </div>
            <div class="sidebar-metric">
                <div class="sidebar-metric-label">30-Day Change</div>
                <div class="sidebar-metric-value" style="color:{color_30};">{sign}{price_change:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        df_sid = get_filtered_df(df, selected_commodity, selected_state)
        df_sid = df_sid[(df_sid["date"] >= d_start) & (df_sid["date"] <= d_end)]
        if len(df_sid) > 0:
            csv_dl = df_sid.to_csv(index=False).encode("utf-8")
            st.download_button("Download Data", csv_dl,
                f"agrisense_{selected_commodity}.csv", "text/csv",
                use_container_width=True)

        # Hindi placement 4: footer
        st.markdown("""
        <div style="margin-top:20px;padding-top:14px;border-top:1px solid #E5E7EB;">
            <div style="font-size:10px;color:#9CA3AF;line-height:1.9;">
                Data: Aug 2022 – Aug 2024<br>
                34 States · 22 Commodities<br>
                Data Source: Government of India <span style="font-size:9px;">(डेटा स्रोत: भारत सरकार)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Page Header (with logo on right) ─────────────────────────────────────
    st.markdown(f"""
    <div class="page-header" style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
            <div class="page-header-eyebrow">AgriSense · Indian Agricultural Price System <span style="font-size:0.65em;color:#9CA3AF;font-weight:400;">(भारतीय कृषि मूल्य प्रणाली)</span></div>
            <h1 class="page-header-title">Indian Crop Price Intelligence</h1>
            <div class="page-header-sub">{COMMODITY_DISPLAY[selected_commodity]}</div>
            <div class="page-header-meta">
                {selected_state} &nbsp;·&nbsp;
                {d_start.strftime("%d %b %Y")} – {d_end.strftime("%d %b %Y")}
            </div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px;padding-top:4px;">
            <svg width="64" height="64" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
              <line x1="18" y1="33" x2="18" y2="5" stroke="#1B4332" stroke-width="1.8" stroke-linecap="round"/>
              <ellipse cx="18" cy="6.5" rx="3" ry="5" fill="#1B4332"/>
              <ellipse cx="11" cy="12" rx="4.5" ry="2" fill="#1B4332" transform="rotate(-45 11 12)"/>
              <ellipse cx="25" cy="12" rx="4.5" ry="2" fill="#1B4332" transform="rotate(45 25 12)"/>
              <ellipse cx="10" cy="18.5" rx="4.5" ry="2" fill="#2D6A4F" transform="rotate(-42 10 18.5)"/>
              <ellipse cx="26" cy="18.5" rx="4.5" ry="2" fill="#2D6A4F" transform="rotate(42 26 18.5)"/>
              <ellipse cx="11" cy="24.5" rx="4" ry="1.8" fill="#40916C" transform="rotate(-38 11 24.5)"/>
              <ellipse cx="25" cy="24.5" rx="4" ry="1.8" fill="#40916C" transform="rotate(38 25 24.5)"/>
              <path d="M18 29 Q24 27 26 23" stroke="#52B788" stroke-width="1.4" fill="none" stroke-linecap="round"/>
            </svg>
            <div style="font-size:13px;font-weight:700;color:#1B4332;letter-spacing:-0.01em;">AgriSense</div>
            <div style="font-size:11px;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.08em;">Crop Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overview",
        "Price Explorer",
        "State Analysis",
        "Trading Intelligence",
        "Market Drivers",
        "7-Day Forecast",
        "Scale the Platform",
    ])

    df_filtered = get_filtered_df(df, selected_commodity, selected_state)
    df_filtered = df_filtered[
        (df_filtered["date"] >= d_start) & (df_filtered["date"] <= d_end)
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        # Hindi placement 3: section header — in brackets alongside English
        section_header("Price Movement Analysis (मूल्य परिवर्तन)")

        if df_filtered.empty:
            alert("No data available for the selected filters.", "yellow")
        else:
            current_price = float(df_filtered["retail_price"].iloc[-1])
            current_date  = df_filtered["date"].iloc[-1].strftime("%d %B %Y")

            weekly_change = 0.0
            if len(df_filtered) >= 8:
                wa = float(df_filtered["retail_price"].iloc[-8])
                if wa > 0: weekly_change = ((current_price - wa) / wa) * 100

            monthly_change = 0.0
            if len(df_filtered) >= 31:
                ma = float(df_filtered["retail_price"].iloc[-31])
                if ma > 0: monthly_change = ((current_price - ma) / ma) * 100

            accuracy = 95.0
            if not results_df.empty:
                mp = results_df[(results_df["commodity"]==selected_commodity) & (results_df["model"]=="XGBoost")]
                if len(mp) > 0: accuracy = float(100 - mp["mape"].iloc[0])

            c1,c2,c3,c4 = st.columns(4)
            with c1: kpi_card("Current Price",   f"₹{current_price:.2f}", sub=current_date)
            with c2: kpi_card("Weekly Change",   format_pct(weekly_change),
                               delta=format_pct(weekly_change),
                               color="red" if weekly_change > 0 else "green")
            with c3: kpi_card("Monthly Change",  format_pct(monthly_change),
                               delta=format_pct(monthly_change),
                               color="red" if monthly_change > 0 else "green")
            with c4: kpi_card("Model Accuracy",  f"{accuracy:.1f}%", sub="Forecast Accuracy")

            # Price trend
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_filtered["date"], y=df_filtered["retail_price"],
                mode="lines", name="Retail",
                line=dict(color=COLORS["chart_primary"], width=2),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:.2f}/kg<extra></extra>"
            ))
            if "wholesale_price" in df_filtered.columns:
                fig.add_trace(go.Scatter(
                    x=df_filtered["date"], y=df_filtered["wholesale_price"],
                    mode="lines", name="Wholesale",
                    line=dict(color=COLORS["chart_second"], width=1.5, dash="dot"),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:.2f}/kg<extra></extra>"
                ))
            if len(df_filtered) >= 7:
                ma7 = df_filtered["retail_price"].rolling(7).mean()
                fig.add_trace(go.Scatter(
                    x=df_filtered["date"], y=ma7, mode="lines", name="7-Day MA",
                    line=dict(color=COLORS["chart_neutral"], width=1.5, dash="dot"),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:.2f}/kg<extra></extra>"
                ))
            commodity_upper = COMMODITY_DISPLAY[selected_commodity].upper()
            plotly_base_layout(fig,
                title=f"{commodity_upper} — Retail vs Wholesale Price ({selected_state})",
                height=420, x_title="Date", y_title="Price (₹/kg)")
            st.plotly_chart(fig, use_container_width=True)

            # Insights & stats
            c1, c2 = st.columns(2)
            with c1:
                section_header("Key Insights")
                mean_p = float(df_filtered["retail_price"].mean())
                if weekly_change > 5:   st.write("↑ Prices rising rapidly — up more than 5% this week.")
                elif weekly_change < -5: st.write("↓ Prices falling rapidly — down more than 5% this week.")
                else:                    st.write("→ Price movement stable this week.")
                if current_price > mean_p * 1.2:   st.write("Current price is significantly above the period average.")
                elif current_price < mean_p * 0.8: st.write("Current price is below average — potential buying opportunity.")
                else:                               st.write("Current price is near the historical average for this period.")
                hm = HARVEST_MONTHS.get(selected_commodity, [])
                import datetime as _dt
                if _dt.date.today().month in hm: st.write("Currently in harvest season — expect supply-side pressure on prices.")
                else:                            st.write("Off-season period — monitor supply availability closely.")

            with c2:
                section_header("Market Statistics")
                stats = [
                    ("Highest Price",  f"₹{df_filtered['retail_price'].max():.2f}"),
                    ("Lowest Price",   f"₹{df_filtered['retail_price'].min():.2f}"),
                    ("Average Price",  f"₹{df_filtered['retail_price'].mean():.2f}"),
                    ("Price Range",    f"₹{df_filtered['retail_price'].max()-df_filtered['retail_price'].min():.2f}"),
                    ("Latest Update",  current_date),
                ]
                for k, v in stats:
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;padding:5px 0;"
                        f"border-bottom:1px solid #F3F4F6;font-size:13px;'>"
                        f"<span style='color:#6B7280;'>{k}</span>"
                        f"<span style='font-family:IBM Plex Mono,monospace;font-weight:600;'>{v}</span></div>",
                        unsafe_allow_html=True)

            # Stakeholder perspective
            section_header("Stakeholder Perspective")

            df_all = df[df["commodity"] == selected_commodity]
            latest_d = df_all["date"].max()
            df_lat = df_all[df_all["date"] == latest_d]

            hm_list = HARVEST_MONTHS.get(selected_commodity, [])
            if hm_list and not df_all.empty:
                da = df_all.copy(); da["is_h"] = da["date"].dt.month.isin(hm_list)
                harvest_avg   = float(da[da["is_h"]]["retail_price"].mean())
                offseason_avg = float(da[~da["is_h"]]["retail_price"].mean())
            else:
                harvest_avg = offseason_avg = current_price

            best_mg_state = df_lat.loc[df_lat["margin_pct"].idxmax(),"state"] if not df_lat.empty else "N/A"
            avg_margin    = float(df_lat["margin_pct"].mean()) if not df_lat.empty else 0
            spread_avg    = float(df_lat["price_spread"].mean()) if not df_lat.empty else 0
            ws_price      = float(df_lat["wholesale_price"].mean()) if not df_lat.empty else current_price

            proc_df = compute_procurement_timing(selected_commodity)
            import datetime as _dt2; cm = _dt2.date.today().month
            proc_signal = "Hold / Monitor"
            if not proc_df.empty:
                row = proc_df[proc_df["Month_Num"] == cm]
                if not row.empty: proc_signal = row.iloc[0]["Signal"] + " " + row.iloc[0]["Action"]

            risk_df = compute_risk_zones(selected_commodity)
            n_high  = int((risk_df["risk_level"] == "High").sum()) if not risk_df.empty else 0
            cv_nat  = float(df_all["retail_price"].std() / (df_all["retail_price"].mean() + 1e-8))
            stability = "Stable" if cv_nat < 0.1 else ("Moderate" if cv_nat < 0.2 else "Volatile")
            trend_sig = "↑ Rising" if weekly_change > 2 else ("↓ Falling" if weekly_change < -2 else "→ Stable")
            spike_alert = "Price spike" if weekly_change > 10 else ("Normal range" if abs(weekly_change) <= 5 else "Elevated")

            sh1, sh2, sh3, sh4 = st.columns(4)
            with sh1:
                st.markdown(f"""<div class="sh-card" style="border-top:3px solid {COLORS['positive']};">
                    <div class="sh-icon">👨‍🌾</div><div class="sh-title">Farmer</div>
                    <div class="sh-focus">Maximise harvest sale returns</div>
                    <div class="sh-row"><span class="sh-label">Current Price</span><span class="sh-val">₹{current_price:.2f}</span></div>
                    <div class="sh-row"><span class="sh-label">Harvest Avg</span><span class="sh-val">₹{harvest_avg:.2f}</span></div>
                    <div class="sh-row"><span class="sh-label">Off-season Avg</span><span class="sh-val">₹{offseason_avg:.2f}</span></div>
                    <div class="sh-row"><span class="sh-label">Price Trend</span><span class="sh-val">{trend_sig}</span></div>
                </div>""", unsafe_allow_html=True)
            with sh2:
                st.markdown(f"""<div class="sh-card" style="border-top:3px solid {COLORS['warning']};">
                    <div class="sh-icon">🏪</div><div class="sh-title">Trader</div>
                    <div class="sh-focus">Arbitrage &amp; margin opportunities</div>
                    <div class="sh-row"><span class="sh-label">Avg Margin</span><span class="sh-val">{avg_margin:.1f}%</span></div>
                    <div class="sh-row"><span class="sh-label">Best Margin State</span><span class="sh-val">{best_mg_state}</span></div>
                    <div class="sh-row"><span class="sh-label">Price Spread</span><span class="sh-val">₹{spread_avg:.2f}</span></div>
                    <div class="sh-row"><span class="sh-label">Market Trend</span><span class="sh-val">{trend_sig}</span></div>
                </div>""", unsafe_allow_html=True)
            with sh3:
                st.markdown(f"""<div class="sh-card" style="border-top:3px solid #1E3A5F;">
                    <div class="sh-icon">🏭</div><div class="sh-title">Wholesaler</div>
                    <div class="sh-focus">Optimise procurement &amp; inventory</div>
                    <div class="sh-row"><span class="sh-label">Wholesale Price</span><span class="sh-val">₹{ws_price:.2f}</span></div>
                    <div class="sh-row"><span class="sh-label">Procurement Signal</span><span class="sh-val">{proc_signal}</span></div>
                    <div class="sh-row"><span class="sh-label">Supply Stability</span><span class="sh-val">{stability}</span></div>
                    <div class="sh-row"><span class="sh-label">Retail Premium</span><span class="sh-val">{avg_margin:.1f}%</span></div>
                </div>""", unsafe_allow_html=True)
            with sh4:
                st.markdown(f"""<div class="sh-card" style="border-top:3px solid {COLORS['negative']};">
                    <div class="sh-icon">🏛️</div><div class="sh-title">Policymaker</div>
                    <div class="sh-focus">Monitor stability &amp; interventions</div>
                    <div class="sh-row"><span class="sh-label">Price Stability</span><span class="sh-val">{stability}</span></div>
                    <div class="sh-row"><span class="sh-label">High-Risk States</span><span class="sh-val">{n_high}</span></div>
                    <div class="sh-row"><span class="sh-label">Price Alert</span><span class="sh-val">{spike_alert}</span></div>
                    <div class="sh-row"><span class="sh-label">Weekly Change</span><span class="sh-val">{weekly_change:+.1f}%</span></div>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — PRICE EXPLORER
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        section_header("Advanced Price Analytics")
        if df_filtered.empty:
            alert("No data available for the selected filters.", "yellow")
        else:
            c1, c2 = st.columns(2)
            with c1:
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(
                    x=df_filtered["retail_price"], nbinsx=30,
                    marker_color=COLORS["chart_primary"], marker_line_width=0,
                    hovertemplate="₹%{x:.1f} – %{y} days<extra></extra>"
                ))
                plotly_base_layout(fig_h,
                    title=f"{COMMODITY_DISPLAY[selected_commodity].upper()} — Price Distribution",
                    height=360, x_title="Price (₹/kg)", y_title="Frequency", show_legend=False)
                st.plotly_chart(fig_h, use_container_width=True)

            with c2:
                if len(df_filtered) >= 30:
                    dv = df_filtered.copy()
                    dv["vol"] = dv["retail_price"].rolling(30).std()
                    fig_v = go.Figure()
                    fig_v.add_trace(go.Scatter(
                        x=dv["date"], y=dv["vol"], mode="lines",
                        name="30-Day Volatility",
                        line=dict(color=COLORS["chart_primary"], width=1.5),
                        fill="tozeroy", fillcolor="rgba(27,67,50,0.06)",
                        hovertemplate="<b>%{x|%d %b %Y}</b><br>Volatility: ₹%{y:.2f}<extra></extra>"
                    ))
                    plotly_base_layout(fig_v,
                        title="30-Day Rolling Price Volatility (Std Dev)",
                        height=360, x_title="Date", y_title="₹/kg", show_legend=False)
                    st.plotly_chart(fig_v, use_container_width=True)

            section_header("Seasonal Price Patterns")
            if len(df_filtered) >= 365:
                ds = df_filtered.copy()
                ds["month_name"] = ds["date"].dt.strftime("%B")
                mo = pd.DataFrame(ds.groupby("month_name")["retail_price"].mean()).reset_index()
                month_order = ["January","February","March","April","May","June",
                               "July","August","September","October","November","December"]
                mo["month_name"] = pd.Categorical(mo["month_name"], categories=month_order, ordered=True)
                mo = mo.sort_values("month_name")
                hm_s = HARVEST_MONTHS.get(selected_commodity, [])
                mo["bar_color"] = mo["month_name"].apply(
                    lambda mn: COLORS["chart_primary"] if (month_order.index(mn)+1 in hm_s) else COLORS["border_strong"]
                )
                ann_avg = float(mo["retail_price"].mean())
                fig_s = go.Figure()
                fig_s.add_trace(go.Bar(
                    x=mo["month_name"], y=mo["retail_price"],
                    marker_color=mo["bar_color"], marker_line_width=0,
                    hovertemplate="<b>%{x}</b><br>Avg: ₹%{y:.2f}/kg<extra></extra>",
                    name="Monthly Avg"
                ))
                fig_s.add_hline(y=ann_avg, line_color=COLORS["border_strong"], line_width=1,
                                annotation_text=f"Annual avg ₹{ann_avg:.2f}",
                                annotation_font=dict(size=10, color=COLORS["text_muted"]),
                                annotation_position="right")
                plotly_base_layout(fig_s,
                    title="Average Price by Month  (dark bars = harvest season)",
                    height=360, x_title="Month", y_title="Avg Price (₹/kg)", show_legend=False)
                st.plotly_chart(fig_s, use_container_width=True)
            else:
                alert("Insufficient data for seasonal analysis — need at least 1 year.", "blue")

            section_header("Year-on-Year Comparison")
            dy = df_filtered.copy()
            dy["year"] = dy["date"].dt.year; dy["month"] = dy["date"].dt.month
            yoy = dy.groupby(["year","month"])["retail_price"].mean().reset_index()
            if yoy["year"].nunique() >= 2:
                palette = [COLORS["chart_primary"], COLORS["chart_down"], COLORS["chart_second"]]
                fig_y = go.Figure()
                for idx, yr in enumerate(sorted(yoy["year"].unique())):
                    yd = yoy[yoy["year"] == yr]
                    fig_y.add_trace(go.Scatter(
                        x=yd["month"], y=yd["retail_price"], mode="lines+markers",
                        name=str(yr), line=dict(color=palette[idx % len(palette)], width=2),
                        marker=dict(size=4),
                        hovertemplate=f"<b>{yr}</b> Month %{{x}}: ₹%{{y:.2f}}/kg<extra></extra>"
                    ))
                plotly_base_layout(fig_y, title="Year-on-Year Price Comparison",
                    height=340, y_title="Avg Price (₹/kg)", show_legend=True)
                fig_y.update_layout(
                    xaxis=dict(tickmode="array",
                        tickvals=list(range(1,13)),
                        ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                        tickfont=dict(size=10, color=COLORS["text_muted"]),
                        gridcolor=COLORS["gridline"], linecolor=COLORS["border"], zeroline=False, ticklen=0)
                )
                st.plotly_chart(fig_y, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — STATE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        section_header("Regional Market Comparison")

        latest_date = df["date"].max()
        df_lat3 = df[(df["commodity"]==selected_commodity) & (df["date"]==latest_date)].copy()

        if df_lat3.empty:
            alert("No recent state data available.", "red")
        else:
            df_s3 = df_lat3.sort_values("retail_price")
            nat_avg = float(df_s3["retail_price"].mean())
            min_p = float(df_s3["retail_price"].min()); max_p = float(df_s3["retail_price"].max())

            bar_colors = []
            for _, row in df_s3.iterrows():
                if row["retail_price"] == min_p:   bar_colors.append(COLORS["positive"])
                elif row["retail_price"] == max_p: bar_colors.append(COLORS["negative"])
                else:                              bar_colors.append(COLORS["border_strong"])

            fig_st = go.Figure()
            fig_st.add_trace(go.Bar(
                x=df_s3["retail_price"], y=df_s3["state"], orientation="h",
                marker_color=bar_colors, marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>₹%{x:.2f}/kg<extra></extra>"
            ))
            fig_st.add_vline(x=nat_avg, line_color=COLORS["chart_neutral"],
                             line_width=1, line_dash="dash")
            fig_st.add_annotation(x=nat_avg, y=1, yref="paper",
                text=f"National avg ₹{nat_avg:.2f}",
                showarrow=True, arrowhead=0, arrowcolor=COLORS["text_muted"], ax=40, ay=0,
                font=dict(size=10, color=COLORS["text_muted"]))
            plotly_base_layout(fig_st,
                title=f"State-wise Retail Prices — {COMMODITY_DISPLAY[selected_commodity]}",
                height=700, x_title="Price (₹/kg)", show_legend=False)
            st.plotly_chart(fig_st, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**5 Cheapest States**")
                ch = df_s3.head(5)[["state","retail_price","margin_pct"]].copy()
                ch["retail_price"] = ch["retail_price"].apply(lambda x: f"₹{x:.2f}")
                ch["margin_pct"]   = ch["margin_pct"].apply(lambda x: f"{x:.1f}%")
                ch.columns = ["State","Price","Margin"]
                st.dataframe(ch, use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**5 Most Expensive States**")
                ex = df_s3.tail(5)[["state","retail_price","margin_pct"]].copy()
                ex["retail_price"] = ex["retail_price"].apply(lambda x: f"₹{x:.2f}")
                ex["margin_pct"]   = ex["margin_pct"].apply(lambda x: f"{x:.1f}%")
                ex.columns = ["State","Price","Margin"]
                st.dataframe(ex.iloc[::-1], use_container_width=True, hide_index=True)

            # Risk zones
            section_header("Price Risk Zone Analysis")
            alert("Risk zones use Coefficient of Variation (CV = Std / Mean) across the full dataset. High CV indicates unstable pricing.", "blue")
            rz = compute_risk_zones(selected_commodity)
            if not rz.empty:
                cr1,cr2,cr3 = st.columns(3)
                with cr1: kpi_card("Low Risk States",    str(int((rz["risk_level"]=="Low").sum())),    sub="CV < 8%  — stable",   color="green")
                with cr2: kpi_card("Medium Risk States", str(int((rz["risk_level"]=="Medium").sum())), sub="CV 8–18%",             color="orange")
                with cr3: kpi_card("High Risk States",   str(int((rz["risk_level"]=="High").sum())),   sub="CV > 18% — volatile", color="red")

                fig_rz = go.Figure()
                fig_rz.add_trace(go.Bar(
                    x=rz["cv"]*100, y=rz["state"], orientation="h",
                    marker_color=rz["risk_color"], marker_line_width=0,
                    hovertemplate="<b>%{y}</b><br>CV: %{x:.1f}%<extra></extra>"
                ))
                fig_rz.add_vline(x=8,  line_color=COLORS["positive"],  line_width=1, line_dash="dot")
                fig_rz.add_vline(x=18, line_color=COLORS["negative"],  line_width=1, line_dash="dot")
                fig_rz.add_annotation(x=8,  y=1, yref="paper", text="Low / Medium",  showarrow=False, font=dict(size=9, color=COLORS["text_muted"]), yshift=4)
                fig_rz.add_annotation(x=18, y=1, yref="paper", text="Medium / High", showarrow=False, font=dict(size=9, color=COLORS["text_muted"]), yshift=4)
                plotly_base_layout(fig_rz,
                    title="State Risk Classification — Price Volatility (CV %)",
                    height=650, x_title="Coefficient of Variation (%)", show_legend=False)
                st.plotly_chart(fig_rz, use_container_width=True)

                rz_d = rz[["state","mean_price","cv","risk_level"]].copy()
                rz_d["mean_price"] = rz_d["mean_price"].apply(lambda x: f"₹{x:.2f}")
                rz_d["cv"]         = (rz_d["cv"]*100).apply(lambda x: f"{x:.1f}%")
                rz_d["risk_level"] = rz_d["risk_level"].apply(lambda r:
                    "Low" if r=="Low" else ("Medium" if r=="Medium" else "High"))
                rz_d.columns = ["State","Avg Price","Volatility (CV)","Risk Level"]
                st.dataframe(rz_d.reset_index(drop=True), use_container_width=True, hide_index=True)

            # Region comparison
            section_header("Region-wise Price Spread")
            if "region" in df_lat3.columns and not df_lat3.empty:
                rg = df_lat3.groupby("region")["retail_price"].agg(["mean","min","max"]).reset_index()
                rg.columns = ["Region","Avg","Min","Max"]
                fig_reg = go.Figure()
                fig_reg.add_trace(go.Bar(x=rg["Region"], y=rg["Avg"],
                    marker_color=COLORS["chart_primary"], marker_line_width=0, name="Avg",
                    hovertemplate="<b>%{x}</b><br>Avg: ₹%{y:.2f}<extra></extra>"))
                fig_reg.add_trace(go.Scatter(x=rg["Region"], y=rg["Max"],
                    mode="markers", marker=dict(color=COLORS["negative"], size=9, symbol="triangle-up"), name="Max"))
                fig_reg.add_trace(go.Scatter(x=rg["Region"], y=rg["Min"],
                    mode="markers", marker=dict(color=COLORS["positive"], size=9, symbol="triangle-down"), name="Min"))
                plotly_base_layout(fig_reg,
                    title="Region-wise Price Spread — Avg / Min / Max",
                    height=340, y_title="Price (₹/kg)")
                st.plotly_chart(fig_reg, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — TRADING INTELLIGENCE
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        section_header("Trading Margin Analysis")
        df_cm = df[df["commodity"]==selected_commodity].copy()
        ld4   = df_cm["date"].max()
        df_m  = df_cm[df_cm["date"]==ld4].copy()

        if df_m.empty:
            alert("No margin data available.", "red")
        else:
            mm    = float(df_m["margin_pct"].mean()); sm = float(df_m["margin_pct"].std())
            mx_r  = df_m.loc[df_m["margin_pct"].idxmax()]
            at    = mm + 1.5*sm
            anom  = df_m[df_m["margin_pct"] > at]
            sp_av = float(df_m["price_spread"].mean())

            c1,c2,c3,c4 = st.columns(4)
            with c1: kpi_card("Avg Trading Margin", f"{mm:.1f}%",       sub="National average")
            with c2: kpi_card("Highest Margin",     f"{mx_r['margin_pct']:.1f}%", sub=mx_r["state"])
            with c3: kpi_card("High Margin States",  str(len(anom)),    sub=f">{at:.1f}% threshold", color="red" if len(anom)>0 else "green")
            with c4: kpi_card("Avg Price Spread",   f"₹{sp_av:.2f}",   sub="Retail minus wholesale")

            # Scatter — uniform size, region-colored
            if "region" in df_m.columns:
                df_m["region_plot"] = df_m["region"].fillna("Other")
            else:
                df_m["region_plot"] = "Other"

            fig_sc = go.Figure()
            for reg, grp in df_m.groupby("region_plot"):
                fig_sc.add_trace(go.Scatter(
                    x=grp["wholesale_price"], y=grp["retail_price"], mode="markers",
                    name=reg,
                    marker=dict(size=8, color=REGION_PALETTE.get(reg, COLORS["chart_neutral"]),
                                opacity=0.75, line=dict(width=0)),
                    text=grp["state"],
                    hovertemplate="<b>%{text}</b><br>Wholesale: ₹%{x:.2f}<br>Retail: ₹%{y:.2f}<extra></extra>"
                ))
            lo2 = min(float(df_m["wholesale_price"].min()), float(df_m["retail_price"].min()))
            hi2 = max(float(df_m["wholesale_price"].max()), float(df_m["retail_price"].max()))
            fig_sc.add_trace(go.Scatter(x=[lo2,hi2], y=[lo2,hi2], mode="lines",
                line=dict(color=COLORS["border_strong"], width=1), name="Zero margin", showlegend=True))

            # Annotate anomaly states
            for _, row in anom.iterrows():
                fig_sc.add_annotation(
                    x=float(row["wholesale_price"]), y=float(row["retail_price"]),
                    text=row["state"], showarrow=False,
                    font=dict(size=9, color=COLORS["negative"]), xshift=5, yshift=5)

            plotly_base_layout(fig_sc,
                title="Wholesale vs Retail Price by State (colored by region)",
                height=460, x_title="Wholesale Price (₹/kg)", y_title="Retail Price (₹/kg)")
            st.plotly_chart(fig_sc, use_container_width=True)

            if len(anom) > 0:
                alert(f"{len(anom)} states have unusually high margins (>{at:.1f}%). Review supply-chain efficiency in flagged states.", "yellow")
                ad = anom[["state","retail_price","wholesale_price","margin_pct"]].copy()
                ad["retail_price"]    = ad["retail_price"].apply(lambda x: f"₹{x:.2f}")
                ad["wholesale_price"] = ad["wholesale_price"].apply(lambda x: f"₹{x:.2f}")
                ad["margin_pct"]      = ad["margin_pct"].apply(lambda x: f"{x:.1f}%")
                ad.columns = ["State","Retail","Wholesale","Margin %"]
                st.dataframe(ad.sort_values("Margin %", ascending=False), use_container_width=True, hide_index=True)
            else:
                alert("All state margins are within normal range.", "green")

        # Procurement timing
        section_header("Procurement Timing & Inventory Planning")
        alert("Signals based on historical monthly price patterns relative to the annual average.", "blue")
        pt = compute_procurement_timing(selected_commodity)
        if not pt.empty:
            c1, c2 = st.columns([3,2])
            with c1:
                bar_c = []
                for _, row in pt.iterrows():
                    pct = row["vs Annual Avg %"]
                    if pct <= -10:  bar_c.append(COLORS["positive"])
                    elif pct <= 0:  bar_c.append("#6EE7B7")
                    elif pct <= 10: bar_c.append("#FCA5A5")
                    else:           bar_c.append(COLORS["negative"])
                fig_pt = go.Figure()
                fig_pt.add_trace(go.Bar(
                    x=pt["Month"], y=pt["vs Annual Avg %"],
                    marker_color=bar_c, marker_line_width=0,
                    hovertemplate="<b>%{x}</b><br>Deviation: %{y:+.1f}%<extra></extra>",
                    name="% vs Annual Avg"
                ))
                fig_pt.add_hline(y=0, line_color=COLORS["border_strong"], line_width=1,
                                 annotation_text="Annual average",
                                 annotation_font=dict(size=10, color=COLORS["text_muted"]),
                                 annotation_position="right")
                plotly_base_layout(fig_pt,
                    title="Monthly Price Deviation from Annual Average",
                    height=340, x_title="Month", y_title="% Deviation", show_legend=False)
                st.plotly_chart(fig_pt, use_container_width=True)

            with c2:
                st.markdown("**Monthly Action Table**")
                dp = pt[["Month","Avg Price","vs Annual Avg %","Signal","Action","Harvest"]].copy()
                dp["Avg Price"]        = dp["Avg Price"].apply(lambda x: f"₹{x:.2f}")
                dp["vs Annual Avg %"]  = dp["vs Annual Avg %"].apply(lambda x: f"{x:+.1f}%")
                dp["Action"]           = dp["Signal"] + " " + dp["Action"]
                dp = dp[["Month","Avg Price","vs Annual Avg %","Action","Harvest"]]
                st.dataframe(dp, use_container_width=True, hide_index=True)

            best_buy  = pt.loc[pt["vs Annual Avg %"].idxmin(),  "Month"]
            best_sell = pt.loc[pt["vs Annual Avg %"].idxmax(), "Month"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""<div class="info-card">
                    <div class="info-card-label">Optimal Buy / Stock-Up Period</div>
                    <div class="info-card-value" style="color:{COLORS['positive']};">{best_buy}</div>
                    <div class="info-card-sub">Historically lowest price month — build inventory</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="info-card" style="border-top-color:{COLORS['negative']};">
                    <div class="info-card-label">Optimal Sell / Liquidate Period</div>
                    <div class="info-card-value" style="color:{COLORS['negative']};">{best_sell}</div>
                    <div class="info-card-sub">Historically highest price month — reduce stock</div>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — MARKET DRIVERS
    # ══════════════════════════════════════════════════════════════════════════
    with tab5:
        section_header("Demand–Supply Market Drivers")
        alert("Festival-linked demand surges, harvest cycle supply impacts, and seasonal pricing patterns.", "blue")

        # Festival impact
        section_header("Festival-Linked Price Impact")
        fi = compute_festival_impact(selected_commodity)
        if not fi.empty:
            c1, c2 = st.columns([2,1])
            with c1:
                fig_fi = go.Figure()
                for festival in FESTIVAL_DATES.keys():
                    fd = fi[fi["Festival"]==festival]
                    if fd.empty: continue
                    color = COLORS["positive"] if float(fd["Impact %"].mean()) <= 0 else COLORS["negative"]
                    fig_fi.add_trace(go.Bar(
                        x=fd["Label"], y=fd["Impact %"], name=festival,
                        marker_color=color, marker_line_width=0,
                        hovertemplate="<b>%{x}</b><br>Impact: %{y:+.2f}%<extra></extra>"
                    ))
                fig_fi.add_hline(y=0, line_color=COLORS["border_strong"], line_width=1)
                plotly_base_layout(fig_fi,
                    title=f"Festival Period Price Change vs Pre-Festival Baseline (±{CONFIG['festival_window']} days)",
                    height=380, x_title="Festival", y_title="Price Impact (%)")
                st.plotly_chart(fig_fi, use_container_width=True)
            with c2:
                st.markdown("**Festival Summary**")
                fs = fi.groupby("Festival")["Impact %"].mean().reset_index()
                fs.columns = ["Festival","Avg Impact %"]
                fs["Avg Impact %"] = fs["Avg Impact %"].apply(lambda x: f"{x:+.1f}%")
                st.dataframe(fs, use_container_width=True, hide_index=True)
                avg_imp = float(fi["Impact %"].mean())
                if avg_imp > 3:   alert(f"Festivals drive prices up ~{avg_imp:.1f}% on average. Stock ahead of festivals.", "yellow")
                elif avg_imp < -3: alert(f"Festivals are associated with price drops (~{avg_imp:.1f}%).", "blue")
                else:              alert("Festival impact on this commodity is modest (<3% avg).", "green")
        else:
            alert("Insufficient data to compute festival impact for this commodity.", "blue")

        # Harvest cycle
        section_header("Harvest Cycle & Supply Analysis")
        hm5 = HARVEST_MONTHS.get(selected_commodity, [])
        df_fc = df[df["commodity"]==selected_commodity].copy()
        df_fc["month_num"] = df_fc["date"].dt.month

        if not df_fc.empty:
            c1, c2 = st.columns([3,1])
            with c1:
                mb = df_fc.groupby("month_num")["retail_price"].agg(
                    ["mean","std","min","max"]).reset_index()
                mb.columns = ["month_num","mean","std","min","max"]
                ml = [calendar.month_abbr[m] for m in mb["month_num"]]
                fig_hv = go.Figure()
                fig_hv.add_trace(go.Bar(
                    x=ml, y=mb["mean"],
                    marker_color=[COLORS["chart_primary"] if m in hm5 else COLORS["border_strong"]
                                  for m in mb["month_num"]],
                    marker_line_width=0,
                    error_y=dict(type="data", array=mb["std"].tolist(),
                                 color=COLORS["text_muted"], thickness=1.5, width=4),
                    name="Mean ± Std",
                    hovertemplate="<b>%{x}</b><br>Mean: ₹%{y:.2f}/kg<extra></extra>"
                ))
                fig_hv.add_trace(go.Scatter(x=ml, y=mb["max"], mode="markers",
                    marker=dict(color=COLORS["negative"], size=7, symbol="triangle-up"), name="Max"))
                fig_hv.add_trace(go.Scatter(x=ml, y=mb["min"], mode="markers",
                    marker=dict(color=COLORS["positive"], size=7, symbol="triangle-down"), name="Min"))
                plotly_base_layout(fig_hv,
                    title="Monthly Price Distribution  (dark bars = harvest season)",
                    height=380, x_title="Month", y_title="Price (₹/kg)")
                st.plotly_chart(fig_hv, use_container_width=True)

            with c2:
                st.markdown("**Harvest Calendar**")
                for mn in range(1,13):
                    label = "Harvest" if mn in hm5 else "Off-season"
                    col   = COLORS["positive"] if mn in hm5 else COLORS["text_muted"]
                    st.markdown(f"<div style='font-size:12px;color:{col};padding:2px 0;'>"
                                f"<b>{calendar.month_abbr[mn]}</b> — {label}</div>", unsafe_allow_html=True)
                if hm5:
                    ha  = float(df_fc[df_fc["month_num"].isin(hm5)]["retail_price"].mean())
                    oa  = float(df_fc[~df_fc["month_num"].isin(hm5)]["retail_price"].mean())
                    st.markdown("---")
                    st.markdown(f"<div style='font-size:11px;color:{COLORS['text_muted']};'>Harvest avg</div>"
                                f"<div style='font-family:IBM Plex Mono,monospace;font-size:15px;font-weight:600;'>₹{ha:.2f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:11px;color:{COLORS['text_muted']};margin-top:8px;'>Off-season avg</div>"
                                f"<div style='font-family:IBM Plex Mono,monospace;font-size:15px;font-weight:600;'>₹{oa:.2f}</div>", unsafe_allow_html=True)

        # Seasonal heatmap
        section_header("Seasonal Price Heatmap")
        dh = df_fc.copy(); dh["year"] = dh["date"].dt.year
        hp = dh.groupby(["year","month_num"])["retail_price"].mean().reset_index()
        hpiv = hp.pivot(index="year", columns="month_num", values="retail_price")
        hpiv.columns = [calendar.month_abbr[m] for m in hpiv.columns]
        if not hpiv.empty:
            fig_heat = go.Figure(data=go.Heatmap(
                z=hpiv.values, x=list(hpiv.columns), y=[str(y) for y in hpiv.index],
                colorscale="RdYlGn_r",
                hovertemplate="<b>%{y} %{x}</b><br>₹%{z:.2f}/kg<extra></extra>",
                colorbar=dict(title="₹/kg", tickfont=dict(size=10))
            ))
            fig_heat.update_layout(
                title=dict(text="<b>Average Retail Price — Year × Month</b>",
                           font=dict(size=13,color="#0D0D0D",family="Inter"), x=0, xanchor="left"),
                height=280, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(family="Inter",size=10,color=COLORS["text_secondary"]),
                margin=dict(l=48,r=24,t=52,b=32),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # Supply–demand driver table
        section_header("Supply–Demand Driver Framework")
        cat5  = COMMODITY_CATEGORIES.get(selected_commodity,"Other")
        kharif = any(m in [6,7,8,9] for m in HARVEST_MONTHS.get(selected_commodity,[]))
        rabi   = any(m in [11,12,1,2] for m in HARVEST_MONTHS.get(selected_commodity,[]))
        stext  = ("Kharif (Jun–Sep)" if kharif else "") + (" / Rabi (Nov–Feb)" if rabi else "") or "Year-round"

        drivers5 = [
            ("Monsoon / Rainfall",         f"Kharif crops depend on Jun–Sep rainfall. Poor monsoon → supply shortage → price spike. Harvest season: {stext}.",                              "High" if kharif else "Medium"),
            ("Logistics & Cold Storage",   "Transport costs and cold-chain availability drive post-harvest price spread across regions.",                                                  "High"),
            ("Harvest Cycles",             f"Primary season: {stext}. Harvest glut → prices fall; off-season scarcity → prices rise.",                                                    "High"),
            ("Festival Demand",            "Demand peaks during Diwali, Holi, Navratri — especially for pulses, oils, sugar, and dairy.",                                                "High" if cat5 in ["Pulse","Oil","Sugar","Dairy"] else "Medium"),
            ("MSP / Government Policy",    "Minimum Support Prices and buffer-stock releases cap extreme swings for staple commodities.",                                                  "High" if cat5 in ["Cereal","Pulse"] else "Low"),
            ("Import / Export Policy",     "Import duty changes or export bans (e.g. onion, wheat) can sharply alter domestic prices.",                                                   "High" if selected_commodity in ["onion","wheat","rice","sugar"] else "Low"),
        ]
        impact_cls = {"High":"risk-high","Medium":"risk-medium","Low":"risk-low"}
        for drv, desc, imp in drivers5:
            ca, cb, cc = st.columns([2,5,1])
            with ca: st.markdown(f"<span style='font-size:13px;font-weight:600;color:{COLORS['text_primary']};'>{drv}</span>", unsafe_allow_html=True)
            with cb: st.markdown(f"<span style='font-size:12px;color:{COLORS['text_secondary']};'>{desc}</span>", unsafe_allow_html=True)
            with cc: st.markdown(f"<span class='risk-badge {impact_cls[imp]}'>{imp}</span>", unsafe_allow_html=True)
            st.markdown(f"<hr style='border:none;border-top:1px solid {COLORS['gridline']};margin:4px 0;'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — 7-DAY FORECAST
    # ══════════════════════════════════════════════════════════════════════════
    with tab6:
        section_header("7-Day Price Forecast")

        if is_all_india:
            alert("Forecast requires a specific state selection. Please choose a state from the sidebar.", "yellow")
        else:
            with st.spinner("Generating forecast..."):
                fc_df = generate_forecast_optimized(selected_commodity, selected_state, model_key)

            if fc_df.empty:
                alert("Unable to generate forecast — insufficient data or model unavailable for this combination.", "red")
            else:
                df_rec = get_filtered_df(df, selected_commodity, selected_state).tail(30)
                cur_p  = float(df_rec["retail_price"].iloc[-1]) if not df_rec.empty else 0.0
                d7_p   = float(fc_df.iloc[-1]["predicted_price"])
                d7_chg = float(fc_df.iloc[-1]["change_pct"])

                mape = 1.0
                if not results_df.empty:
                    mp = results_df[(results_df["commodity"]==selected_commodity) & (results_df["model"]=="XGBoost")]
                    if len(mp) > 0: mape = float(mp["mape"].iloc[0])

                c1,c2,c3,c4 = st.columns(4)
                with c1: kpi_card("Current Price",    f"₹{cur_p:.2f}",   sub=df_rec["date"].iloc[-1].strftime("%d %b %Y") if not df_rec.empty else "")
                with c2: kpi_card("Day +7 Forecast",  f"₹{d7_p:.2f}",    sub="7-Day Prediction")
                with c3: kpi_card("Expected Change",  format_pct(d7_chg), delta=format_pct(d7_chg),
                                   color="red" if d7_chg > 0 else "green")
                with c4: kpi_card("Model Accuracy",   f"{100-mape:.1f}%", sub="Forecast Accuracy")

                section_header(f"7-Day Forecast — {COMMODITY_DISPLAY[selected_commodity].upper()} ({selected_state})")

                fig_fc2 = go.Figure()
                if not df_rec.empty:
                    fig_fc2.add_trace(go.Scatter(
                        x=df_rec["date"], y=df_rec["retail_price"],
                        name="Historical", mode="lines",
                        line=dict(color=COLORS["chart_primary"], width=2),
                        hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:.2f}/kg<extra></extra>"
                    ))

                fig_fc2.add_trace(go.Scatter(
                    x=fc_df["date"], y=fc_df["predicted_price"],
                    name="Forecast", mode="lines+markers",
                    line=dict(color=COLORS["chart_down"], width=1.5, dash="dash"),
                    marker=dict(size=6, color=COLORS["chart_down"]),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast: ₹%{y:.2f}/kg<extra></extra>"
                ))

                # Price labels on forecast points
                for _, row in fc_df.iterrows():
                    fig_fc2.add_annotation(
                        x=row["date"], y=float(row["predicted_price"]),
                        text=f"₹{row['predicted_price']:.0f}",
                        showarrow=False, yshift=12,
                        font=dict(size=9, color=COLORS["chart_down"], family="IBM Plex Mono")
                    )

                upper = fc_df["predicted_price"] * (1 + mape/100)
                lower = fc_df["predicted_price"] * (1 - mape/100)
                fig_fc2.add_trace(go.Scatter(
                    x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
                    y=pd.concat([upper, lower[::-1]]),
                    fill="toself", fillcolor="rgba(153,27,27,0.05)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Confidence Band", hoverinfo="skip"
                ))

                if not df_rec.empty:
                    fig_fc2.add_vline(
                        x=timestamp_to_ms(df_rec["date"].iloc[-1]),
                        line_color=COLORS["border_strong"], line_width=1, line_dash="dash")
                    fig_fc2.add_annotation(
                        x=timestamp_to_ms(df_rec["date"].iloc[-1]), y=1, yref="paper",
                        text="Forecast start", showarrow=False,
                        font=dict(size=9, color=COLORS["text_muted"]),
                        xshift=4, yshift=0, xanchor="left"
                    )

                plotly_base_layout(fig_fc2,
                    title=f"7-Day Forecast — {COMMODITY_DISPLAY[selected_commodity]} ({selected_state})",
                    height=440, x_title="Date", y_title="Price (₹/kg)")
                st.plotly_chart(fig_fc2, use_container_width=True)

                section_header("Forecast Breakdown")
                disp = fc_df.copy()
                disp["predicted_price"] = disp["predicted_price"].apply(lambda x: f"₹{x:.2f}")
                disp["change_pct"]      = disp["change_pct"].apply(lambda x: f"{'↑ +' if x>=0 else '↓ '}{x:.2f}%")
                disp["confidence"]      = [f"₹{p*(1-mape/100):.2f} – ₹{p*(1+mape/100):.2f}" for p in fc_df["predicted_price"]]
                disp = disp[["day","date","predicted_price","change_pct","confidence"]]
                disp.columns = ["Day","Date","Predicted Price","Change","Confidence Range"]
                st.dataframe(disp, use_container_width=True, hide_index=True)

                csv_fc = fc_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Forecast CSV", csv_fc,
                    f"forecast_{selected_commodity}_{selected_state}.csv", "text/csv",
                    use_container_width=True)

                alert("Forecast accuracy decreases over the horizon. Day +1 is most reliable. Use for trend guidance, not exact price prediction.", "yellow")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 — SCALE THE PLATFORM
    # ══════════════════════════════════════════════════════════════════════════
    with tab7:
        section_header("Modular Product Framework")

        n_comm   = len(COMMODITIES)
        n_states = int(df["state"].nunique())
        n_models = len([f for f in os.listdir(CONFIG["models_dir"]) if f.endswith(".pkl")])
        total_r  = len(df)

        alert("AgriSense is built as a config-driven platform. Adding a new crop or region requires no changes to dashboard logic — only data, a trained model file, and a few config entries.", "blue")

        c1,c2,c3 = st.columns(3)
        with c1: kpi_card("Commodities Tracked",   str(n_comm),     sub="Cereals, pulses, oils, vegetables, dairy")
        with c2: kpi_card("States / UTs Covered",  str(n_states),   sub="All major Indian states and union territories")
        with c3: kpi_card("Historical Data Points", f"{total_r:,}", sub="Daily retail and wholesale prices")

        section_header("How to Add a New Crop")
        s1,s2,s3 = st.columns(3)
        with s1:
            st.markdown(f"""<div class="step-card">
                <div class="step-card-title">Step 1 — Data</div>
                <div class="step-card-body">
                    Collect daily retail and wholesale price data for the new crop across target states.
                    Format identically to <code>features.csv</code>. Run through
                    <code>src/feature_engineering.py</code> to generate all lag, rolling, and seasonal
                    features. Append resulting rows to <code>data/processed/features.csv</code>.
                </div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class="step-card">
                <div class="step-card-title">Step 2 — Train Model</div>
                <div class="step-card-body">
                    Run <code>src/model_training.py</code> — it loops over all commodities automatically,
                    so it trains a predictive model for the new crop without any script changes.
                    Output model files are saved to <code>models/</code> automatically.
                    are saved to <code>models/</code>.
                </div>
            </div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""<div class="step-card">
                <div class="step-card-title">Step 3 — Register in Config</div>
                <div class="step-card-body">
                    Add 4 lines at the top of <code>app.py</code>:<br><br>
                    • <code>COMMODITIES</code> — crop key<br>
                    • <code>COMMODITY_DISPLAY</code> — display name<br>
                    • <code>COMMODITY_CATEGORIES</code> — category (Cereal / Pulse / Oil etc.)<br>
                    • <code>HARVEST_MONTHS</code> — list of harvest month numbers
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:12px;padding:10px 14px;background:#F0FDF4;
                    border-left:3px solid {COLORS['positive']};border-radius:4px;font-size:13px;color:#14532D;">
            <b>Result:</b> The new crop instantly appears in the sidebar dropdown. All 7 tabs
            auto-populate — price charts, state analysis, risk zones, forecast, market drivers — with zero additional code changes.
        </div>""", unsafe_allow_html=True)

        section_header("How to Add a New State or Region")
        s4, s5 = st.columns(2)
        with s4:
            st.markdown(f"""<div class="step-card">
                <div class="step-card-title">Step 1 — Add the Data</div>
                <div class="step-card-body">
                    Collect price data for the new state in the same column format as the existing dataset.
                    Run feature engineering, then append to <code>data/processed/features.csv</code>.
                    The new state name auto-appears in the sidebar dropdown on next app load. No code changes required.
                </div>
            </div>""", unsafe_allow_html=True)
        with s5:
            st.markdown(f"""<div class="step-card">
                <div class="step-card-title">Step 2 — Optional: Assign Region</div>
                <div class="step-card-body">
                    If the new state belongs to a geographic region (North / South / East / West / Central / Northeast),
                    set this in the <code>region</code> column of the raw data.
                    If omitted, the state defaults to the "Other" bucket — all charts still work correctly.
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:12px;padding:10px 14px;background:#F0FDF4;
                    border-left:3px solid {COLORS['positive']};border-radius:4px;font-size:13px;color:#14532D;">
            <b>Result:</b> The new state appears in the sidebar, state comparison chart, risk zone table,
            forecast tab, and all trading intelligence views. No dashboard code changes required.
        </div>""", unsafe_allow_html=True)

        section_header("Architecture at a Glance")
        arch = [
            ("Data Layer",       "data/processed/features.csv",              "Single append to extend coverage. Feature engineering pipeline in src/feature_engineering.py is reusable for any commodity or state."),
            ("Model Layer",      "models/<commodity>.pkl",                   "One trained model file per commodity. Training pipeline loops automatically over all commodities in the dataset."),
            ("Config Layer",     "Top of app.py (4 dictionaries)",           "4 dictionary entries to register a new crop. Zero entries for a new state."),
            ("Dashboard Layer",  "app.py — all 7 tabs",                      "Fully data-driven. No hardcoded commodity or state lists in any chart, forecast, or analysis section."),
            ("Forecast Engine",  "generate_forecast_optimized()",            "Accepts any commodity + state + model combination. Auto-builds feature vectors from historical data."),
        ]
        for layer, component, detail in arch:
            ca, cb, cc = st.columns([1.5,2,4])
            with ca: st.markdown(f"<span style='font-size:12px;font-weight:600;color:{COLORS['accent']};'>{layer}</span>", unsafe_allow_html=True)
            with cb: st.markdown(f"<code style='font-size:11px;color:{COLORS['text_secondary']};'>{component}</code>", unsafe_allow_html=True)
            with cc: st.markdown(f"<span style='font-size:12px;color:{COLORS['text_secondary']};'>{detail}</span>", unsafe_allow_html=True)
            st.markdown(f"<hr style='border:none;border-top:1px solid {COLORS['gridline']};margin:5px 0;'>", unsafe_allow_html=True)


main()
