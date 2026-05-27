"""
app/streamlit/pages/3_Bracket_Simulator.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from app.streamlit.loader import get_predictor, WC_2026_TEAMS

st.set_page_config(page_title="Bracket Simulator · Copa Oracle", page_icon="🎲", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a,#1b2838)}
[data-testid="stSidebar"] *{color:#e0e6f0!important}
.champion-box{background:linear-gradient(135deg,#2d1f00,#7d4e00);
              border:2px solid #f5c518;border-radius:16px;
              padding:28px;text-align:center;margin:16px 0}
.champion-box h1{color:#f5c518;font-size:3rem;margin:0}
.champion-box p{color:#ffd87e;margin:6px 0 0;font-size:1.1rem}
</style>
""", unsafe_allow_html=True)

pred = get_predictor()
all_teams = sorted(pred.profiles.keys())

st.markdown("## 🎲 Bracket Simulator")
st.markdown("Monte Carlo simulation of the FIFA World Cup 2026 bracket — runs thousands of tournaments to estimate each team's probability of winning.")
st.markdown("---")

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([4, 3, 3])
with col1:
    n_sims = st.select_slider(
        "Number of simulations",
        options=[1_000, 2_000, 5_000, 10_000],
        value=5_000,
    )
with col2:
    host_team = st.selectbox("Host Nation", ["United States", "Mexico", "Canada"] + all_teams,
                              index=0)
with col3:
    seed = st.number_input("Random Seed", value=42, min_value=0, max_value=9999)

# ── Team selection ────────────────────────────────────────────────────────────
st.markdown("### 🌍 Select 32 Teams")
selected = st.multiselect(
    "Teams in the tournament",
    all_teams,
    default=[t for t in WC_2026_TEAMS if t in all_teams][:32],
    max_selections=32,
)

if len(selected) < 4:
    st.warning("Please select at least 4 teams.")
    st.stop()

if len(selected) % 2 != 0:
    selected = selected[:-1]
    st.info(f"Using {len(selected)} teams (even number required).")

# ── Run simulation ────────────────────────────────────────────────────────────
run_btn = st.button(f"🚀 Run {n_sims:,} Simulations", use_container_width=True, type="primary")

if run_btn:
    with st.spinner(f"Running {n_sims:,} Monte Carlo simulations..."):
        sim = pred.simulate_bracket(selected, host=host_team, n_sims=n_sims, seed=seed)

    st.markdown("---")

    # Champion
    champion = sim.iloc[0]
    st.markdown(f"""
    <div class='champion-box'>
        <div style='font-size:2rem'>🏆</div>
        <h1>{champion['team']}</h1>
        <p>Predicted Champion &nbsp;|&nbsp; Win Probability: <strong>{champion['win_prob']:.1f}%</strong></p>
        <p style='color:#c9a84c;font-size:.9rem'>Copa Oracle Score: {champion['copa_score']:.1f}/100</p>
    </div>
    """, unsafe_allow_html=True)

    # Top 8 bar chart
    st.markdown("### 📊 Win Probability — Top 16 Teams")
    top16 = sim.head(16)
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    cmap = cm.get_cmap("YlOrRd")
    colors = [cmap(0.3 + 0.7 * i / len(top16)) for i in range(len(top16) - 1, -1, -1)]
    bars = ax.barh(top16["team"][::-1], top16["win_prob"][::-1],
                   color=colors, height=0.65, edgecolor="none")

    for bar, val in zip(bars, top16["win_prob"][::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="#e0e6f0", fontsize=9, fontweight="bold")

    ax.set_xlabel("Win Probability (%)", color="#8ba3c7")
    ax.tick_params(colors="#c9d6e3")
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Full results table
    st.markdown("---")
    st.markdown("### 📋 Full Simulation Results")
    display = sim[["team", "win_prob", "copa_score"]].copy()
    display.index = range(1, len(display) + 1)
    display.columns = ["Team", "Win Probability (%)", "Copa Oracle Score"]
    st.dataframe(
        display.style.background_gradient(subset=["Win Probability (%)"], cmap="YlOrRd"),
        use_container_width=True,
        height=600,
    )

    # Copa Score vs Win Prob scatter
    st.markdown("---")
    st.markdown("### 🔬 Copa Oracle Score vs Win Probability")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.patch.set_facecolor("#0d1b2a")
    ax2.set_facecolor("#0d1b2a")

    sc = ax2.scatter(sim["copa_score"], sim["win_prob"],
                     c=sim["win_prob"], cmap="YlOrRd",
                     s=100, edgecolors="#0d1b2a", linewidths=0.5, zorder=3)

    for _, row in sim.head(8).iterrows():
        ax2.annotate(row["team"], (row["copa_score"], row["win_prob"]),
                     xytext=(4, 4), textcoords="offset points",
                     color="#c9d6e3", fontsize=7.5)

    corr = np.corrcoef(sim["copa_score"], sim["win_prob"])[0, 1]
    m, b = np.polyfit(sim["copa_score"], sim["win_prob"], 1)
    x_line = np.linspace(sim["copa_score"].min(), sim["copa_score"].max(), 100)
    ax2.plot(x_line, m * x_line + b, color="#f5c518", linestyle="--",
             linewidth=1.5, label=f"r = {corr:.2f}")

    ax2.set_xlabel("Copa Oracle Score", color="#8ba3c7")
    ax2.set_ylabel("Win Probability (%)", color="#8ba3c7")
    ax2.tick_params(colors="#c9d6e3")
    ax2.legend(facecolor="#1b2838", labelcolor="#e0e6f0")
    for spine in ax2.spines.values(): spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    corr_sign = "positive" if corr > 0 else "negative"
    st.caption(f"Correlation between Copa Oracle Score and simulated win probability: r = {corr:.3f} ({corr_sign})")
else:
    st.info("👆 Configure your settings above and click **Run Simulations** to start.")
