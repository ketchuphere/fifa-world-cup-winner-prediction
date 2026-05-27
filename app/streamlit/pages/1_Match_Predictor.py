"""
app/streamlit/pages/1_Match_Predictor.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from app.streamlit.loader import get_predictor, WC_2026_TEAMS

st.set_page_config(page_title="Match Predictor · Copa Oracle", page_icon="🎯", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a,#1b2838)}
[data-testid="stSidebar"] *{color:#e0e6f0!important}
.result-box{border-radius:14px;padding:22px 28px;text-align:center;margin:8px 0}
.result-win {background:linear-gradient(135deg,#1a4731,#27ae60);border:1px solid #2ecc71}
.result-draw{background:linear-gradient(135deg,#4a3200,#e67e22);border:1px solid #f39c12}
.result-lose{background:linear-gradient(135deg,#4a0e0e,#c0392b);border:1px solid #e74c3c}
.result-box h2{color:#fff;margin:0;font-size:1.9rem}
.result-box p {color:rgba(255,255,255,.75);margin:4px 0 0;font-size:.9rem}
.stat-row{display:flex;justify-content:space-between;
          padding:8px 16px;border-radius:8px;background:#1b2838;margin:4px 0}
.stat-label{color:#8ba3c7;font-size:.85rem}
.stat-value{color:#e0e6f0;font-weight:600;font-size:.85rem}
</style>
""", unsafe_allow_html=True)

pred = get_predictor()
all_teams = sorted(pred.profiles.keys())

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🎯 Match Predictor")
st.markdown("Select two teams and a stage to get win probabilities.")
st.markdown("---")

# ── Inputs ────────────────────────────────────────────────────────────────────
col1, col_vs, col2 = st.columns([5, 1, 5])
with col1:
    home = st.selectbox("🏠 Home Team", all_teams,
                        index=all_teams.index("Brazil") if "Brazil" in all_teams else 0)
with col_vs:
    st.markdown("<div style='text-align:center;padding-top:32px;font-size:1.5rem;color:#f5c518'>VS</div>",
                unsafe_allow_html=True)
with col2:
    away_list = [t for t in all_teams if t != home]
    away = st.selectbox("✈️ Away Team", away_list,
                        index=away_list.index("Argentina") if "Argentina" in away_list else 0)

col3, col4, col5 = st.columns([4, 3, 3])
with col3:
    stage = st.selectbox("🏟️ Match Stage", [
        "group", "round_of_16", "quarter_final", "semi_final", "third_place", "final"
    ], format_func=lambda s: s.replace("_", " ").title())
with col4:
    host_home = st.checkbox(f"🏠 {home} is host nation")
with col5:
    host_away = st.checkbox(f"✈️ {away} is host nation")

predict_btn = st.button("⚡ Predict Match", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn or True:
    res = pred.predict(home, away, stage, int(host_home), int(host_away))
    probs = res["probs"]
    p_h = probs.get("H", 0)
    p_d = probs.get("D", 0)
    p_a = probs.get("A", 0)
    predicted = res["predicted"]

    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    # Result boxes
    r1, r2, r3 = st.columns(3)
    with r1:
        cls = "result-win" if predicted == "H" else "result-lose"
        icon = "🏆" if predicted == "H" else "❌"
        st.markdown(f"""<div class='result-box {cls}'>
            <h2>{icon} {p_h:.1f}%</h2>
            <p>{home} Win</p></div>""", unsafe_allow_html=True)
    with r2:
        cls = "result-win" if predicted == "D" else "result-draw"
        icon = "🤝" if predicted == "D" else "➖"
        st.markdown(f"""<div class='result-box {cls}'>
            <h2>{icon} {p_d:.1f}%</h2>
            <p>Draw</p></div>""", unsafe_allow_html=True)
    with r3:
        cls = "result-win" if predicted == "A" else "result-lose"
        icon = "🏆" if predicted == "A" else "❌"
        st.markdown(f"""<div class='result-box {cls}'>
            <h2>{icon} {p_a:.1f}%</h2>
            <p>{away} Win</p></div>""", unsafe_allow_html=True)

    # Probability bar chart
    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 2.2))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    outcomes = [home, "Draw", away]
    values   = [p_h, p_d, p_a]
    colors   = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars     = ax.barh(outcomes, values, color=colors, height=0.55, edgecolor="none")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="#e0e6f0", fontsize=12, fontweight="bold")

    ax.set_xlim(0, 110)
    ax.set_xlabel("Win Probability (%)", color="#8ba3c7")
    ax.tick_params(colors="#c9d6e3")
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Team stats comparison
    st.markdown("---")
    st.markdown("### 📋 Team Comparison")
    hp = pred.get_profile(home)
    ap = pred.get_profile(away)

    comp_data = {
        "Stat": ["Elo Rating", "Form Score", "Goals/Game", "Concede/Game",
                 "WC Experience", "Copa Oracle Score", "Confederation"],
        home: [
            f"{hp['final_elo']:.0f}",
            f"{hp['form']:.2f}",
            f"{hp['gpg']:.2f}",
            f"{hp['gcpg']:.2f}",
            f"{hp['wc_exp']} WCs",
            f"{pred.team_score(home):.1f}/100",
            __import__('src.pipelines.feature_eng_pipeline',
                       fromlist=['confederation']).confederation(home),
        ],
        away: [
            f"{ap['final_elo']:.0f}",
            f"{ap['form']:.2f}",
            f"{ap['gpg']:.2f}",
            f"{ap['gcpg']:.2f}",
            f"{ap['wc_exp']} WCs",
            f"{pred.team_score(away):.1f}/100",
            __import__('src.pipelines.feature_eng_pipeline',
                       fromlist=['confederation']).confederation(away),
        ],
    }
    comp_df = pd.DataFrame(comp_data).set_index("Stat")
    st.dataframe(comp_df, use_container_width=True)

    # H2H history
    st.markdown("### 🔁 Head-to-Head History")
    h2h = pred.h2h_history(home, away)
    if len(h2h) == 0:
        st.info("No previous World Cup meetings between these two teams.")
    else:
        st.dataframe(h2h.reset_index(drop=True), use_container_width=True)
        wins_home = (h2h["match_result"] == "H").sum()
        wins_away = (h2h["match_result"] == "A").sum()
        draws     = (h2h["match_result"] == "D").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home} Wins", wins_home)
        c2.metric("Draws", draws)
        c3.metric(f"{away} Wins", wins_away)
