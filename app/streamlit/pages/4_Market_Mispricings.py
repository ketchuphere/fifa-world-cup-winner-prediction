"""
app/streamlit/pages/4_Market_Mispricings.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from app.streamlit.loader import get_predictor

st.set_page_config(page_title="Market Mispricings · Copa Oracle", page_icon="📊", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a,#1b2838)}
[data-testid="stSidebar"] *{color:#e0e6f0!important}
.buy-tag  {background:#1a4731;color:#2ecc71;border:1px solid #2ecc71;
           padding:3px 10px;border-radius:6px;font-size:.8rem;font-weight:700}
.sell-tag {background:#4a0e0e;color:#e74c3c;border:1px solid #e74c3c;
           padding:3px 10px;border-radius:6px;font-size:.8rem;font-weight:700}
.info-card{background:linear-gradient(135deg,#1b2838,#243447);
           border:1px solid #2d4a6e;border-radius:12px;padding:16px 20px;margin:6px 0}
</style>
""", unsafe_allow_html=True)

pred = get_predictor()
all_teams = sorted(pred.profiles.keys())

st.markdown("## 📊 Market Mispricings")
st.markdown("Compare model probabilities against your market odds to find edges.")
st.markdown("---")

# ── How it works ──────────────────────────────────────────────────────────────
with st.expander("ℹ️ How does this work?"):
    st.markdown("""
    Enter market odds (as implied probabilities %) for upcoming matches.
    The model compares its own predicted probabilities against the market.

    - **BUY signal** → Model gives higher probability than market (team undervalued)
    - **SELL signal** → Model gives lower probability than market (team overvalued)
    - **Edge** = Model% − Market%

    The threshold slider controls the minimum edge required to flag a mispricing.
    """)

# ── Controls ──────────────────────────────────────────────────────────────────
threshold = st.slider("Minimum Edge to Flag (%)", min_value=2, max_value=25, value=8)
st.markdown("---")

# ── Match entry ───────────────────────────────────────────────────────────────
st.markdown("### ➕ Enter Upcoming Matches")
st.caption("Add as many matches as you like. Leave market odds at 33/33/33 if unknown.")

NUM_MATCHES = st.number_input("Number of matches to analyse", min_value=1, max_value=10, value=4)

matchups = []
for i in range(int(NUM_MATCHES)):
    with st.container():
        st.markdown(f"**Match {i+1}**")
        c1, c2, c3, c4, c5, c6, c7 = st.columns([3, 3, 2, 2, 2, 2, 2])
        with c1:
            home = st.selectbox("Home", all_teams, key=f"home_{i}",
                                index=all_teams.index("Brazil") if "Brazil" in all_teams else i % len(all_teams))
        with c2:
            away_opts = [t for t in all_teams if t != home]
            away = st.selectbox("Away", away_opts, key=f"away_{i}",
                                index=i % len(away_opts))
        with c3:
            stage = st.selectbox("Stage", ["group","round_of_16","quarter_final",
                                            "semi_final","final"], key=f"stage_{i}",
                                  format_func=lambda s: s.replace("_"," ").title())
        with c4:
            mkt_h = st.number_input("Mkt H%", 1.0, 98.0, 40.0, key=f"mh_{i}", step=0.5)
        with c5:
            mkt_d = st.number_input("Mkt D%", 1.0, 98.0, 25.0, key=f"md_{i}", step=0.5)
        with c6:
            mkt_a = st.number_input("Mkt A%", 1.0, 98.0, 35.0, key=f"ma_{i}", step=0.5)
        with c7:
            st.markdown("<br>", unsafe_allow_html=True)
            total = mkt_h + mkt_d + mkt_a
            color = "#2ecc71" if 99 <= total <= 110 else "#e74c3c"
            st.markdown(f"<span style='color:{color};font-size:.8rem'>Total: {total:.0f}%</span>",
                        unsafe_allow_html=True)
        matchups.append((home, away, stage, mkt_h, mkt_d, mkt_a))
        st.markdown("---")

# ── Analyse ───────────────────────────────────────────────────────────────────
if st.button("🔍 Analyse Mispricings", use_container_width=True, type="primary"):
    results = pred.find_mispricings(matchups, threshold=threshold)

    if len(results) == 0:
        st.success(f"✅ No mispricings found above the {threshold}% threshold.")
        st.info("Try lowering the threshold or entering more matches.")
    else:
        st.markdown(f"### 🚨 {len(results)} Mispricing(s) Found")

        buy_signals  = results[results["signal"] == "BUY"]
        sell_signals = results[results["signal"] == "SELL"]

        mc1, mc2 = st.columns(2)
        mc1.metric("🟢 BUY Signals",  len(buy_signals))
        mc2.metric("🔴 SELL Signals", len(sell_signals))

        st.markdown("---")

        # ── Detailed table ────────────────────────────────────────────────────
        st.markdown("#### 📋 Full Mispricing Table")
        display = results.copy()

        def highlight_signal(row):
            if row["signal"] == "BUY":
                return ["background-color: #1a3a25"] * len(row)
            return ["background-color: #3a1a1a"] * len(row)

        st.dataframe(
            display.style.apply(highlight_signal, axis=1),
            use_container_width=True
        )

        # ── Bar chart ─────────────────────────────────────────────────────────
        st.markdown("#### 📊 Edge Visualisation")
        fig, ax = plt.subplots(figsize=(11, max(4, len(results) * 0.55)))
        fig.patch.set_facecolor("#0d1b2a")
        ax.set_facecolor("#0d1b2a")

        labels = [f"{r['match']} [{r['outcome']}]" for _, r in results.iterrows()]
        edges  = results["edge"].tolist()
        colors = ["#2ecc71" if e > 0 else "#e74c3c" for e in edges]

        bars = ax.barh(labels, edges, color=colors, height=0.6, edgecolor="none")
        ax.axvline(0, color="#8ba3c7", linewidth=1)
        ax.axvline(threshold,  color="#f5c518", linewidth=1, linestyle="--",
                   label=f"+{threshold}% threshold")
        ax.axvline(-threshold, color="#f5c518", linewidth=1, linestyle="--")

        for bar, val in zip(bars, edges):
            xpos = bar.get_width() + (0.3 if val >= 0 else -0.3)
            ha   = "left" if val >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.1f}%", va="center", ha=ha,
                    color="#e0e6f0", fontsize=8.5, fontweight="bold")

        ax.set_xlabel("Edge (Model% − Market%)", color="#8ba3c7")
        ax.tick_params(colors="#c9d6e3", labelsize=8)
        ax.legend(facecolor="#1b2838", labelcolor="#e0e6f0", fontsize=8)
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── All match predictions ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🎯 Full Model Predictions vs Market")
        for home, away, stage, mkt_h, mkt_d, mkt_a in matchups:
            res = pred.predict(home, away, stage)
            probs = res["probs"]
            with st.expander(f"**{home}** vs **{away}**  [{stage.replace('_',' ').title()}]"):
                rows = {
                    "Outcome":  [home, "Draw", away],
                    "Model %":  [f"{probs.get('H',0):.1f}%", f"{probs.get('D',0):.1f}%", f"{probs.get('A',0):.1f}%"],
                    "Market %": [f"{mkt_h:.1f}%", f"{mkt_d:.1f}%", f"{mkt_a:.1f}%"],
                    "Edge":     [
                        f"{probs.get('H',0)-mkt_h:+.1f}%",
                        f"{probs.get('D',0)-mkt_d:+.1f}%",
                        f"{probs.get('A',0)-mkt_a:+.1f}%",
                    ]
                }
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("👆 Enter your matches above and click **Analyse Mispricings**.")
