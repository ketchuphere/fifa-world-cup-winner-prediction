"""
app/streamlit/app.py
---------------------
Copa Oracle 2026 — Streamlit App entry point.

Run with:
    streamlit run app/streamlit/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Copa Oracle 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e6f0 !important; }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1b2838 0%, #243447 100%);
        border: 1px solid #2d4a6e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-card h2 { color: #f5c518; margin: 0; font-size: 2.2rem; }
    .metric-card p  { color: #8ba3c7; margin: 4px 0 0; font-size: 0.85rem; }

    /* Probability bar */
    .prob-bar-wrap { margin: 8px 0; }
    .prob-label    { color: #c9d6e3; font-size: 0.9rem; margin-bottom: 3px; }
    .prob-bar-bg   { background: #1b2838; border-radius: 8px; height: 28px; position: relative; }
    .prob-bar-fill { height: 100%; border-radius: 8px; display: flex;
                     align-items: center; padding-left: 10px;
                     font-weight: 700; font-size: 0.9rem; color: #fff; }
    .tag-win  { background: linear-gradient(90deg,#2ecc71,#27ae60); }
    .tag-draw { background: linear-gradient(90deg,#f39c12,#e67e22); }
    .tag-loss { background: linear-gradient(90deg,#e74c3c,#c0392b); }

    /* Section header */
    .section-header {
        border-left: 4px solid #f5c518;
        padding-left: 12px;
        margin: 24px 0 16px;
        font-size: 1.3rem;
        font-weight: 700;
        color: #e0e6f0;
    }

    /* Page title */
    .page-title {
        background: linear-gradient(135deg,#0d1b2a,#1b4b82);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
    .page-title h1 { color: #f5c518; margin: 0; font-size: 2.2rem; }
    .page-title p  { color: #8ba3c7; margin: 6px 0 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Copa Oracle 2026")
    st.markdown("---")
    st.markdown("""
    **Navigation**
    - 🎯 Match Predictor
    - 🏆 Team Rankings
    - 🎲 Bracket Simulator
    - 📊 Market Mispricings
    """)
    st.markdown("---")
    st.markdown("""
    **Model Info**
    - Algorithm: LightGBM
    - Features: 32
    - CV Accuracy: ~59%
    - Data: 1930–2022
    """)
    st.markdown("---")
    st.caption("Built with Copa Oracle 2026 Pipeline")

# ── Home page ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-title'>
    <h1>⚽ Copa Oracle 2026</h1>
    <p>FIFA World Cup 2026 · ML-Powered Match Prediction Engine</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""<div class='metric-card'>
        <h2>964</h2><p>Matches Trained On</p></div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class='metric-card'>
        <h2>32</h2><p>Engineered Features</p></div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class='metric-card'>
        <h2>59%</h2><p>CV Accuracy</p></div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""<div class='metric-card'>
        <h2>10K</h2><p>Monte Carlo Sims</p></div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 👈 Select a page from the sidebar to get started")

c1, c2 = st.columns(2)
with c1:
    st.info("🎯 **Match Predictor** — Head-to-head win probabilities for any two teams")
    st.info("🎲 **Bracket Simulator** — Monte Carlo simulation of the full 2026 bracket")
with c2:
    st.info("🏆 **Team Rankings** — Copa Oracle Score leaderboard for all 32 teams")
    st.info("📊 **Market Mispricings** — Compare model probabilities vs betting market odds")
