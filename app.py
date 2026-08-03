
from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from agent import AgentContext
from config import config
from database.sqlite_manager import SQLiteManager
from graph import AnalystGraph
from tools.dataset_loader import DatasetLoader
from tools.pandas_tool import PandasTool
from utils.helpers import dataframe_summary, summary_to_text
from utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="DataPilot AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "DataPilot AI — an autonomous data analyst you can talk to.",
    },
)


# --------------------------------------------------------------------------- #
# Theme (single locked palette)
# --------------------------------------------------------------------------- #
THEME = {
    "neon": "#00e5ff",
    "neon_2": "#7c4dff",
    "accent": "#ff5cf7",
    "bg": "#0a0e1a",
    "bg_soft": "#0f1424",
}


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_css(theme: dict) -> None:
    """Inject the futuristic glassmorphism theme with design tokens."""
    st.markdown(
        f"""
        <style>
        @import url('[fonts.googleapis.com](https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono&display=swap)');

        :root {{
            --neon: {theme['neon']};
            --neon-2: {theme['neon_2']};
            --accent: {theme['accent']};
            --bg: {theme['bg']};
            --bg-soft: {theme['bg_soft']};
            --glass: rgba(255, 255, 255, 0.045);
            --glass-strong: rgba(255, 255, 255, 0.07);
            --glass-border: rgba(255, 255, 255, 0.09);
            --text: #e6edf3;
            --text-dim: #9aa7bd;
            --radius: 16px;
            --radius-lg: 22px;
            --shadow: 0 8px 40px rgba(0, 0, 0, 0.35);
            --glow: 0 0 18px rgba(0, 229, 255, 0.35);
        }}

        html, body, [class*="css"] {{
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text);
        }}

        /* Animated aurora background + fine grain overlay */
        .stApp {{
            background:
                radial-gradient(circle at 15% 20%, rgba(124, 77, 255, 0.16), transparent 42%),
                radial-gradient(circle at 85% 25%, rgba(0, 229, 255, 0.13), transparent 46%),
                radial-gradient(circle at 50% 95%, rgba(255, 92, 247, 0.08), transparent 50%),
                var(--bg);
            background-attachment: fixed;
            animation: auroraDrift 24s ease-in-out infinite alternate;
        }}
        .stApp::before {{
            content: "";
            position: fixed; inset: 0;
            pointer-events: none; z-index: 0;
            opacity: 0.035;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='[w3.org](http://www.w3.org/2000/svg)' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
        }}
        @keyframes auroraDrift {{
            0%   {{ background-position: 0% 0%, 100% 0%, 50% 100%, 0 0; }}
            100% {{ background-position: 6% 4%, 94% 6%, 46% 96%, 0 0; }}
        }}

        /* Hide default header clutter */
        header[data-testid="stHeader"] {{ background: transparent; }}
        #MainMenu, footer {{ visibility: hidden; }}

        /* Hero header with animated gradient border */
        .hero {{
            position: relative;
            padding: 2.6rem 2.4rem;
            border-radius: var(--radius-lg);
            background: linear-gradient(135deg, rgba(0,229,255,0.10), rgba(124,77,255,0.12));
            backdrop-filter: blur(16px);
            margin-bottom: 1.6rem;
            box-shadow: var(--shadow);
            overflow: hidden;
            z-index: 1;
        }}
        .hero::before {{
            content: "";
            position: absolute; inset: 0;
            padding: 1px; border-radius: var(--radius-lg);
            background: linear-gradient(120deg, var(--neon), var(--neon-2), var(--accent), var(--neon));
            background-size: 300% 300%;
            -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
            -webkit-mask-composite: xor; mask-composite: exclude;
            animation: borderFlow 8s linear infinite;
            pointer-events: none;
        }}
        @keyframes borderFlow {{ to {{ background-position: 300% 50%; }} }}
        .hero::after {{
            content: "";
            position: absolute; inset: 0;
            background: radial-gradient(circle at 90% 10%, rgba(0,229,255,0.18), transparent 40%);
            pointer-events: none;
        }}
        .hero h1 {{
            font-size: 2.6rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(90deg, var(--neon), var(--neon-2), var(--accent));
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.6px;
            animation: titleShine 6s linear infinite;
        }}
        @keyframes titleShine {{ to {{ background-position: 200% center; }} }}
        .hero p {{
            color: var(--text-dim);
            margin: 0.55rem 0 0;
            font-size: 1.02rem;
            max-width: 640px;
        }}
        .hero-chips {{ margin-top: 1.1rem; display: flex; gap: 0.55rem; flex-wrap: wrap; }}
        .chip {{
            font-size: 0.78rem;
            padding: 0.32rem 0.8rem;
            border-radius: 999px;
            border: 1px solid var(--glass-border);
            background: var(--glass);
            color: var(--text-dim);
            transition: border-color 0.25s ease, transform 0.25s ease;
        }}
        .chip:hover {{ border-color: rgba(0,229,255,0.4); transform: translateY(-2px); }}
        .chip b {{ color: var(--neon); font-weight: 600; }}

        /* Metric cards with accent top border */
        [data-testid="stMetric"] {{
            position: relative;
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius);
            padding: 1rem 1.2rem;
            backdrop-filter: blur(10px);
            overflow: hidden;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }}
        [data-testid="stMetric"]::before {{
            content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, var(--neon), var(--neon-2));
            opacity: 0.7;
        }}
        [data-testid="stMetric"]:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 34px rgba(0, 229, 255, 0.20);
            border-color: rgba(0, 229, 255, 0.4);
        }}
        [data-testid="stMetricValue"] {{ color: var(--neon); font-weight: 700; }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.4rem;
            background: var(--glass);
            padding: 0.35rem;
            border-radius: 14px;
            border: 1px solid var(--glass-border);
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 10px;
            color: var(--text-dim);
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: color 0.2s ease, background 0.2s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{ color: var(--text); }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, rgba(0,229,255,0.18), rgba(124,77,255,0.18));
            color: var(--text);
            box-shadow: 0 0 14px rgba(0, 229, 255, 0.25);
        }}

        /* Buttons */
        .stButton > button, .stDownloadButton > button {{
            border-radius: 12px;
            border: 1px solid rgba(0, 229, 255, 0.4);
            background: linear-gradient(135deg, rgba(0,229,255,0.12), rgba(124,77,255,0.12));
            color: var(--text);
            font-weight: 500;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: var(--neon);
            box-shadow: var(--glow);
            color: #fff;
            transform: translateY(-1px);
        }}
        .stButton > button:active {{ transform: translateY(0); }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: rgba(10, 14, 26, 0.9);
            border-right: 1px solid var(--glass-border);
            backdrop-filter: blur(22px);
        }}
        [data-testid="stSidebar"] h1 {{ font-size: 1.3rem; color: var(--neon); }}

        /* Chat bubbles with role-tinted left border */
        [data-testid="stChatMessage"] {{
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius);
            backdrop-filter: blur(8px);
            margin-bottom: 0.6rem;
            padding: 0.55rem 0.8rem;
            transition: border-color 0.2s ease;
        }}
        [data-testid="stChatMessage"]:hover {{ border-color: rgba(0,229,255,0.25); }}

        /* Focus glow for inputs */
        [data-testid="stChatInput"] textarea:focus,
        .stTextInput input:focus,
        .stSelectbox [data-baseweb="select"]:focus-within {{
            box-shadow: var(--glow) !important;
            border-color: var(--neon) !important;
        }}

        /* Expanders */
        [data-testid="stExpander"] {{
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            background: var(--glass);
        }}

        /* Dataframe */
        [data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--glass-border);
        }}

        /* File uploader */
        [data-testid="stFileUploaderDropzone"] {{
            background: var(--glass);
            border: 1px dashed rgba(0, 229, 255, 0.35);
            border-radius: 12px;
            transition: border-color 0.2s ease, background 0.2s ease;
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: var(--neon);
            background: var(--glass-strong);
        }}

        /* Chat input */
        [data-testid="stChatInput"] textarea {{
            background: var(--glass) !important;
            border-radius: 12px !important;
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(var(--neon), var(--neon-2));
            border-radius: 8px;
        }}
        ::-webkit-scrollbar-track {{ background: transparent; }}

        /* Section pills with gradient underline */
        .section-pill {{
            position: relative;
            display: inline-block;
            padding: 0.34rem 1rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--neon);
            background: rgba(0, 229, 255, 0.08);
            border: 1px solid rgba(0, 229, 255, 0.25);
            margin-bottom: 0.9rem;
        }}

        /* Status dot */
        .status-dot {{
            height: 9px; width: 9px; border-radius: 50%;
            display: inline-block; margin-right: 6px;
            box-shadow: 0 0 8px currentColor;
        }}
        .online  {{ color: #2ecc71; background: #2ecc71; }}
        .offline {{ color: #ff5c5c; background: #ff5c5c; }}

        /* Quality gauge card */
        .quality-card {{
            display: flex; align-items: center; gap: 1.2rem;
            padding: 1.2rem 1.4rem;
            border-radius: var(--radius);
            background: var(--glass-strong);
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(10px);
        }}
        .gauge {{
            --v: 0;
            width: 92px; height: 92px; border-radius: 50%;
            background: conic-gradient(var(--neon) calc(var(--v) * 1%), rgba(255,255,255,0.06) 0);
            display: grid; place-items: center; position: relative;
            flex-shrink: 0;
            animation: gaugeIn 1s ease-out;
        }}
        @keyframes gaugeIn {{ from {{ filter: grayscale(1); opacity: 0.4; }} to {{ filter: none; opacity: 1; }} }}
        .gauge::before {{
            content: ""; position: absolute; inset: 8px;
            border-radius: 50%; background: var(--bg-soft);
        }}
        .gauge span {{ position: relative; font-weight: 700; font-size: 1.25rem; color: var(--neon); }}

        /* Feature grid on empty state */
        .feature-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem; margin-top: 1.2rem;
        }}
        .feature-card {{
            padding: 1.3rem; border-radius: var(--radius);
            background: var(--glass); border: 1px solid var(--glass-border);
            backdrop-filter: blur(10px);
            transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        }}
        .feature-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(0,229,255,0.4);
            box-shadow: 0 12px 30px rgba(0,229,255,0.15);
        }}
        .feature-card .fc-icon {{ font-size: 1.6rem; }}
        .feature-card .fc-title {{ font-weight: 600; margin: 0.5rem 0 0.3rem; color: var(--text); }}
        .feature-card .fc-desc {{ font-size: 0.86rem; color: var(--text-dim); line-height: 1.4; }}

        /* Suggestion chips */
        .stButton.suggest > button {{
            background: rgba(124,77,255,0.10);
            border: 1px solid rgba(124,77,255,0.3);
            font-size: 0.82rem;
        }}

        /* Typing shimmer */
        .thinking {{
            display: inline-block;
            background: linear-gradient(90deg, var(--text-dim), var(--neon), var(--text-dim));
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shimmer 1.6s linear infinite;
            font-weight: 500;
        }}
        @keyframes shimmer {{ to {{ background-position: 200% center; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_header() -> None:
    """Render the futuristic hero banner with live stats."""
    df = st.session_state.df
    rows = f"{len(df):,}" if df is not None else "—"
    cols = f"{df.shape[1]}" if df is not None else "—"
    msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.markdown(
        f"""
        <div class="hero">
            <h1>🛰️ DataPilot AI</h1>
            <p>Your autonomous data analyst — upload, connect, and converse with your data in natural language.</p>
            <div class="hero-chips">
                <span class="chip">Rows <b>{rows}</b></span>
                <span class="chip">Columns <b>{cols}</b></span>
                <span class="chip">Questions asked <b>{msgs}</b></span>
                <span class="chip">Engine <b>Analyst Graph</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(label: str) -> None:
    """Render a small neon pill above a section."""
    st.markdown(f'<span class="section-pill">{label}</span>', unsafe_allow_html=True)


def empty_state() -> None:
    """Render an engaging empty state with feature highlights."""
    st.info("👈 Upload a CSV/Excel file, connect SQLite, or load the sample dataset to begin.")
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="fc-icon">💬</div>
                <div class="fc-title">Conversational Analysis</div>
                <div class="fc-desc">Ask questions in plain language and get answers, tables, and charts back instantly.</div>
            </div>
            <div class="feature-card">
                <div class="fc-icon">🔬</div>
                <div class="fc-title">Deep Exploration</div>
                <div class="fc-desc">Profile any column, filter rows on the fly, and export exactly what you need.</div>
            </div>
            <div class="feature-card">
                <div class="fc-icon">🩺</div>
                <div class="fc-title">Quality Scoring</div>
                <div class="fc-desc">Automatic completeness, duplicate, and column-health checks with a live score.</div>
            </div>
            <div class="feature-card">
                <div class="fc-icon">🗄️</div>
                <div class="fc-title">SQLite Ready</div>
                <div class="fc-desc">Connect a database, browse tables, and load them straight into the workspace.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# State & caching
# --------------------------------------------------------------------------- #
def init_state() -> None:
    """Initialize Streamlit session state keys."""
    defaults = {
        "df": None,
        "df_name": None,
        "sqlite": None,
        "messages": [],
        "graph": None,
        "pending_question": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner="Initializing agent…")
def get_graph() -> AnalystGraph:
    """Build and cache the analyst graph across reruns."""
    return AnalystGraph()


@st.cache_data(show_spinner=False)
def _sample_dataset() -> pd.DataFrame:
    """Generate a small demo dataset for instant exploration."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame(
        {
            "order_id": range(1, n + 1),
            "region": rng.choice(["North", "South", "East", "West"], n),
            "category": rng.choice(["Tech", "Home", "Sports", "Toys"], n),
            "units": rng.integers(1, 25, n),
            "price": rng.normal(50, 18, n).round(2).clip(5),
            "discount": rng.choice([0, 0.05, 0.1, 0.15, 0.2], n),
            "date": pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 200, n), "D"),
        }
    ).assign(revenue=lambda d: (d.units * d.price * (1 - d.discount)).round(2))


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
def quality_score(df: pd.DataFrame) -> tuple[int, list[str]]:
    """Compute a 0–100 data-quality score and human-readable warnings."""
    warnings: list[str] = []
    cells = df.size or 1

    missing_ratio = df.isna().sum().sum() / cells
    dup_ratio = df.duplicated().sum() / (len(df) or 1)

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    high_card = [
        c for c in df.select_dtypes(include="object").columns
        if df[c].nunique(dropna=True) > 0.9 * len(df) and len(df) > 20
    ]
    high_missing = [c for c in df.columns if df[c].isna().mean() > 0.3]

    score = 100
    score -= min(40, int(missing_ratio * 100))
    score -= min(25, int(dup_ratio * 100))
    score -= len(constant_cols) * 5
    score -= len(high_missing) * 4
    score = max(0, min(100, score))

    if high_missing:
        warnings.append(f"High missingness in: {', '.join(high_missing[:5])}")
    if constant_cols:
        warnings.append(f"Constant (single-value) columns: {', '.join(constant_cols[:5])}")
    if high_card:
        warnings.append(f"Very high cardinality (possible IDs): {', '.join(high_card[:5])}")
    if dup_ratio > 0.02:
        warnings.append(f"{df.duplicated().sum()} duplicate rows detected")
    return score, warnings


def render_quality(df: pd.DataFrame) -> None:
    """Render the data-quality gauge and warnings."""
    score, warnings = quality_score(df)
    st.markdown(
        f"""
        <div class="quality-card">
            <div class="gauge" style="--v:{score}"><span>{score}</span></div>
            <div>
                <div style="font-weight:600; font-size:1.05rem;">Data Quality Score</div>
                <div style="color:var(--text-dim); font-size:0.88rem;">
                    Weighted by completeness, duplicates and column health.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if warnings:
        with st.expander("⚠️ Quality warnings", expanded=score < 75):
            for w in warnings:
                st.markdown(f"- {w}")
    else:
        st.success("No structural issues detected — the dataset looks clean.")


# --------------------------------------------------------------------------- #
# Rendering: Overview
# --------------------------------------------------------------------------- #
def render_dataset_info(df: pd.DataFrame) -> None:
    """Render dataset information and statistics."""
    summary = dataframe_summary(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{summary['rows']:,}")
    c2.metric("Columns", summary["columns"])
    c3.metric("Duplicates", summary["duplicate_rows"])
    c4.metric("Memory (KB)", round(summary["memory_bytes"] / 1024, 1))

    st.write("")
    render_quality(df)
    st.write("")

    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("🔎 Column types & missing values", expanded=True):
            dtypes = pd.DataFrame(
                {
                    "dtype": df.dtypes.astype(str),
                    "missing": df.isna().sum(),
                    "missing %": (df.isna().mean() * 100).round(1),
                    "unique": df.nunique(),
                }
            )
            st.dataframe(dtypes, use_container_width=True)
    with col_b:
        with st.expander("📊 Summary statistics", expanded=True):
            st.dataframe(PandasTool(df).describe(), use_container_width=True)


# --------------------------------------------------------------------------- #
# Rendering: Explore
# --------------------------------------------------------------------------- #
def render_explore(df: pd.DataFrame) -> None:
    """Interactive column profiler and filterable preview."""
    section_title("Column Profiler")
    col = st.selectbox("Choose a column to profile", df.columns)
    series = df[col]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unique", f"{series.nunique():,}")
    m2.metric("Missing", f"{series.isna().sum():,}")
    m3.metric("Missing %", f"{series.isna().mean() * 100:.1f}%")
    m4.metric("Dtype", str(series.dtype))

    st.write("")
    if pd.api.types.is_numeric_dtype(series):
        fig = px.histogram(df, x=col, nbins=40, template="plotly_dark",
                           color_discrete_sequence=[THEME["neon"]])
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        top = series.value_counts().head(15).reset_index()
        top.columns = [col, "count"]
        fig = px.bar(top, x="count", y=col, orientation="h", template="plotly_dark",
                     color_discrete_sequence=[THEME["neon_2"]])
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    section_title("Filter & Export")
    search = st.text_input("Search across all columns", placeholder="Type to filter rows…")
    view = df
    if search:
        mask = df.astype(str).apply(
            lambda r: r.str.contains(search, case=False, na=False)
        ).any(axis=1)
        view = df[mask]
        st.caption(f"{len(view):,} matching rows")
    st.dataframe(view.head(200), use_container_width=True)

    st.download_button(
        "⬇️ Download current view (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=f"{st.session_state.df_name or 'data'}_export.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar() -> None:
    """Render the sidebar controls."""
    st.sidebar.title("⚡ Control Center")

    db_online = st.session_state.sqlite is not None
    data_online = st.session_state.df is not None
    st.sidebar.markdown(
        f"""
        <div style="font-size:0.85rem; margin:0.6rem 0 0.8rem;">
            <span class="status-dot {'online' if data_online else 'offline'}"></span>
            Dataset {'loaded' if data_online else 'none'}<br>
            <span class="status-dot {'online' if db_online else 'offline'}"></span>
            SQLite {'connected' if db_online else 'disconnected'}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.divider()
    st.sidebar.subheader("📁 Data Sources")

    csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    xlsx_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx", "xls"])

    if csv_file is not None:
        st.session_state.df = DatasetLoader.load(csv_file, csv_file.name)
        st.session_state.df_name = csv_file.name
        _register_dataset()
    elif xlsx_file is not None:
        st.session_state.df = DatasetLoader.load(xlsx_file, xlsx_file.name)
        st.session_state.df_name = xlsx_file.name
        _register_dataset()

    if st.sidebar.button("🧪 Load sample dataset", use_container_width=True):
        st.session_state.df = _sample_dataset()
        st.session_state.df_name = "sample_sales"
        _register_dataset()

    st.sidebar.divider()
    st.sidebar.subheader("🗄️ SQLite")
    db_path = st.sidebar.text_input("Database path", value="")
    if st.sidebar.button("🔌 Connect", use_container_width=True) and db_path:
        _connect_sqlite(db_path)

    if st.session_state.sqlite is not None:
        tables = st.session_state.sqlite.list_tables()
        table = st.sidebar.selectbox("Select table", tables)
        if table and st.sidebar.button(
            "📥 Load table into DataFrame", use_container_width=True
        ):
            st.session_state.df = st.session_state.sqlite.load_table(table)
            st.session_state.df_name = table
            _register_dataset()

    st.sidebar.divider()
    if st.sidebar.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []

    if st.session_state.messages:
        st.sidebar.download_button(
            "📤 Export chat (Markdown)",
            data=_transcript_md(),
            file_name="datapilot_conversation.md",
            mime="text/markdown",
            use_container_width=True,
        )


def _transcript_md() -> str:
    """Serialize the conversation to Markdown."""
    lines = [f"# DataPilot AI — Conversation\n\n_Exported {datetime.utcnow():%Y-%m-%d %H:%M} UTC_\n"]
    for m in st.session_state.messages:
        who = "🧑‍💻 **You**" if m["role"] == "user" else "🛰️ **DataPilot**"
        lines.append(f"\n### {who}\n\n{m['content']}\n")
        if m.get("code"):
            lines.append(f"\n```\n{m['code']}\n```\n")
    return "\n".join(lines)


def _register_dataset() -> None:
    """Store dataset metadata into long-term memory."""
    df = st.session_state.df
    name = st.session_state.df_name or "dataset"
    try:
        text = summary_to_text(name, dataframe_summary(df))
        get_graph().agent.rag.store_dataset_metadata(name, text)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to register dataset metadata: %s", exc)


def _connect_sqlite(db_path: str) -> None:
    """Connect to a SQLite database and store its schema."""
    try:
        manager = SQLiteManager(db_path)
        st.session_state.sqlite = manager
        get_graph().agent.rag.store_schema(manager.schema_text())
        st.sidebar.success(f"Connected: {len(manager.list_tables())} tables")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Connection failed: {exc}")


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
SUGGESTIONS = [
    "Give me a summary of this dataset",
    "What are the top trends and correlations?",
    "Show the distribution of the main numeric column",
    "Are there any data quality issues?",
]


def _run_question(question: str) -> None:
    """Execute a single question through the agent graph."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🛰️"):
        placeholder = st.empty()
        placeholder.markdown('<span class="thinking">Analyzing your data…</span>',
                             unsafe_allow_html=True)
        ctx = AgentContext(df=st.session_state.df, sqlite=st.session_state.sqlite)
        resp = get_graph().run(question, ctx)
        placeholder.empty()

        if resp.error:
            content = f"⚠️ {resp.error}"
            st.error(resp.error)
        else:
            content = resp.answer
            st.markdown(resp.answer)
        if resp.dataframe is not None:
            st.dataframe(resp.dataframe, use_container_width=True)
        if resp.figure is not None:
            st.plotly_chart(resp.figure, use_container_width=True)
        if resp.code:
            with st.expander("⌨️ Generated code / SQL"):
                st.code(resp.code)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "dataframe": resp.dataframe,
            "figure": resp.figure,
            "code": resp.code,
        }
    )


def render_chat() -> None:
    """Render the chat interface and handle new questions."""
    # Starter suggestions when the chat is empty
    if not st.session_state.messages and st.session_state.df is not None:
        st.caption("Try a starter question:")
        cols = st.columns(2)
        for i, s in enumerate(SUGGESTIONS):
            with cols[i % 2]:
                if st.button(s, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.pending_question = s
                    st.rerun()

    # Replay history
    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🛰️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("dataframe") is not None:
                st.dataframe(msg["dataframe"], use_container_width=True)
            if msg.get("figure") is not None:
                st.plotly_chart(msg["figure"], use_container_width=True)
            if msg.get("code"):
                with st.expander("⌨️ Generated code / SQL"):
                    st.code(msg["code"])

    # Regenerate control
    if st.session_state.messages and st.session_state.df is not None:
        last_user = next(
            (m["content"] for m in reversed(st.session_state.messages)
             if m["role"] == "user"), None
        )
        if last_user and st.button("🔁 Regenerate last answer"):
            st.session_state.pending_question = last_user
            st.rerun()

    # Handle queued question from a suggestion / regenerate
    if st.session_state.pending_question:
        q = st.session_state.pending_question
        st.session_state.pending_question = None
        _run_question(q)

    question = st.chat_input("Ask about your data…")
    if question:
        _run_question(question)


def render_history() -> None:
    """Searchable conversation history."""
    if not st.session_state.messages:
        st.info("No conversation yet — head to the Chat tab to get started.")
        return
    query = st.text_input("🔍 Search the conversation", placeholder="Filter messages…")
    for m in st.session_state.messages:
        if query and query.lower() not in m["content"].lower():
            continue
        who = "You" if m["role"] == "user" else "DataPilot"
        with st.expander(f"{who}: {m['content'][:60]}…" if len(m['content']) > 60
                         else f"{who}: {m['content']}"):
            st.markdown(m["content"])
            if m.get("code"):
                st.code(m["code"])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Application entry point."""
    init_state()
    inject_css(THEME)
    hero_header()

    problems = config.validate()
    if problems:
        for problem in problems:
            st.warning(problem)

    sidebar()

    if st.session_state.df is None:
        empty_state()
        return

    overview, explore, chat, history = st.tabs(
        ["📊 Overview", "🔬 Explore", "💬 Chat", "🕑 History"]
    )

    with overview:
        section_title(f"Dataset · {st.session_state.df_name}")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        render_dataset_info(st.session_state.df)

    with explore:
        render_explore(st.session_state.df)

    with chat:
        section_title("💬 Conversation")
        render_chat()

    with history:
        section_title("🕑 History")
        render_history()


if __name__ == "__main__":
    main()
