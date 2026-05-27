"""
app/streamlit/pages/2_Team_Rankings.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from app.streamlit.loader import get_predictor

st.set_page_config(page_title="Team Rankings · Copa Oracle", page_icon="🏆", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a,#1b2838)}
[data-testid="stSidebar"] *{color:#e0e6f0!important}
.rank-card{background:linear-gradient(135deg,#1b2838,#243447);
           border:1px solid #2d4a6e;border-radius:12px;padding:16px 20px;margin:6px 0}
.rank-num{color:#f5c518;font-size:1.8rem;font-weight:700;margin:0}
.rank-team{color:#e0e6f0;font-size:1.1rem;font-weight:600;margin:2px 0}
.rank-score{color:#2ecc71;font-size:1.4rem;font-weight:700}
</style>
""", unsafe_allow_html=True)

pred = get_predictor()

st.markdown("## 🏆 Copa Oracle Team Rankings")
st.markdown("Teams ranked by their Copa Oracle Score — a weighted composite of Elo, Form, Attack, and Defence.")
st.markdown("---")

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([3, 3, 2])
with col1:
    conf_filter = st.multiselect(
        "Filter by Confederation",
        ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"],
        default=[]
    )
with col2:
    sort_by = st.selectbox("Sort by", ["copa_score", "elo", "form", "gpg", "gcpg"])
with col3:
    top_n = st.slider("Show Top N", 5, 50, 20)

# ── Load rankings ─────────────────────────────────────────────────────────────
scores = pred.all_team_scores()
if conf_filter:
    scores = scores[scores["confederation"].isin(conf_filter)]
scores = scores.sort_values(sort_by, ascending=(sort_by == "gcpg")).reset_index(drop=True)
scores = scores.head(top_n)

# ── Top 3 podium ──────────────────────────────────────────────────────────────
st.markdown("### 🥇 Top 3 Teams")
pod1, pod2, pod3 = st.columns(3)
podium = [
    (pod1, "🥇", "#f5c518"),
    (pod2, "🥈", "#c0c0c0"),
    (pod3, "🥉", "#cd7f32"),
]
for i, (col, medal, color) in enumerate(podium):
    if i < len(scores):
        row = scores.iloc[i]
        with col:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1b2838,#243447);
                        border:2px solid {color};border-radius:14px;
                        padding:20px;text-align:center'>
                <div style='font-size:2.5rem'>{medal}</div>
                <div style='color:{color};font-size:1.2rem;font-weight:700'>{row['team']}</div>
                <div style='color:#2ecc71;font-size:1.8rem;font-weight:700'>{row['copa_score']:.1f}</div>
                <div style='color:#8ba3c7;font-size:.8rem'>Copa Oracle Score</div>
                <hr style='border-color:#2d4a6e;margin:10px 0'>
                <div style='color:#c9d6e3;font-size:.82rem'>
                    Elo: {row['elo']:.0f} &nbsp;|&nbsp;
                    Form: {row['form']:.2f}<br>
                    GPG: {row['gpg']:.2f} &nbsp;|&nbsp;
                    GCPG: {row['gcpg']:.2f}
                </div>
            </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Full rankings chart ───────────────────────────────────────────────────────
st.markdown("### 📊 Full Rankings Chart")
fig, ax = plt.subplots(figsize=(12, max(6, len(scores) * 0.38)))
fig.patch.set_facecolor("#0d1b2a")
ax.set_facecolor("#0d1b2a")

n = len(scores)
cmap = cm.get_cmap("RdYlGn")
norm_vals = (scores[sort_by] - scores[sort_by].min()) / (scores[sort_by].max() - scores[sort_by].min() + 1e-9)
colors = [cmap(v) for v in norm_vals]

y_pos = range(n)
bars  = ax.barh(list(y_pos), scores[sort_by].tolist(),
                color=colors[::-1] if sort_by != "gcpg" else colors,
                height=0.7, edgecolor="none")

ax.set_yticks(list(y_pos))
ax.set_yticklabels(scores["team"].tolist(), color="#c9d6e3", fontsize=9)
ax.invert_yaxis()
ax.set_xlabel(sort_by.replace("_", " ").title(), color="#8ba3c7")
ax.tick_params(axis="x", colors="#8ba3c7")
for spine in ax.spines.values(): spine.set_visible(False)

for bar, val, row in zip(bars, scores[sort_by], scores.itertuples()):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", color="#e0e6f0", fontsize=8)

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

# ── Full data table ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Full Data Table")
display_cols = ["team", "copa_score", "elo", "form", "gpg", "gcpg", "confederation", "wc_exp"]
display_df   = scores[display_cols].copy()
display_df.index = range(1, len(display_df) + 1)
display_df.columns = ["Team", "Copa Score", "Elo", "Form", "Goals/G", "Concede/G", "Confederation", "WC Exp"]

st.dataframe(
    display_df.style.background_gradient(subset=["Copa Score"], cmap="YlGn"),
    use_container_width=True,
    height=500,
)

# ── Copa Oracle score formula explanation ─────────────────────────────────────
with st.expander("ℹ️ How is the Copa Oracle Score calculated?"):
    st.markdown("""
    ```
    Score = (Elo × 40%) + (Form × 25%) + (Attack Rate × 20%) + (Defence Rate × 15%)
    ```
    All four components are normalised 0→1 across all known teams before weighting.

    - **Elo** — Stage-weighted Elo rating computed progressively across all WC matches (1930–2022)
    - **Form** — Exponentially-weighted average of last 5 match results (W=3, D=1, L=0)
    - **Attack Rate** — Goals scored per game
    - **Defence Rate** — Inverted goals conceded per game (lower concede = higher score)
    """)
