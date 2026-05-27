"""
src/pipelines/feature_eng_pipeline.py
--------------------------------------
Stage 1 + 2: Raw data → cleaned combined dataset → feature matrix.

Improvements over the original notebook:
  - Stage-weighted Elo (K varies by match importance)
  - Inter-tournament Elo decay toward 1500
  - Attack / Defense ratings (goals per game, not raw cumulative)
  - Confederation strength encoding
  - Tournament experience (# prior WC appearances)
  - Recency-weighted form (exponential, not linear)
  - All stats computed strictly BEFORE the current match (no leakage)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

#Confederation mapping

CONFEDERATION = {
    # UEFA
    **dict.fromkeys([
        "Germany","France","Spain","England","Italy","Netherlands","Portugal",
        "Belgium","Croatia","Denmark","Switzerland","Sweden","Austria","Poland",
        "Czechia","Czech Republic","Czechoslovakia","Hungary","Romania","Serbia",
        "Yugoslavia","Scotland","Turkey","Ukraine","Russia","Soviet Union",
        "Slovakia","Slovenia","Greece","Norway","Bulgaria","Bosnia and Herzegovina",
        "Northern Ireland","Republic of Ireland","Wales","Albania","North Macedonia",
        "Finland","Georgia","Iceland",
    ], "UEFA"),
    #CONMEBOL
    **dict.fromkeys([
        "Brazil","Argentina","Uruguay","Colombia","Chile","Peru","Paraguay",
        "Ecuador","Bolivia","Venezuela",
    ], "CONMEBOL"),
    #CONCACAF
    **dict.fromkeys([
        "United States","Mexico","Costa Rica","Honduras","Jamaica","Panama",
        "Trinidad and Tobago","Canada","Cuba","El Salvador","Haiti",
        "Guatemala","Curacao",
    ], "CONCACAF"),
    #CAF
    **dict.fromkeys([
        "Cameroon","Nigeria","Senegal","Ghana","Morocco","Egypt","Tunisia",
        "Algeria","Ivory Coast","Cote d'Ivoire","South Africa","Angola",
        "Togo","Congo DR","Zambia","Mali","Burkina Faso","Cabo Verde",
        "Equatorial Guinea",
    ], "CAF"),
    #AFC
    **dict.fromkeys([
        "Japan","South Korea","Iran","Saudi Arabia","Australia","China",
        "Iraq","North Korea","Kuwait","UAE","Qatar","Indonesia",
        "Uzbekistan","Jordan","Bahrain",
    ], "AFC"),
    #OFC
    **dict.fromkeys(["New Zealand","Australia"], "OFC"),
}

CONF_STRENGTH = {
    "UEFA": 0.72, "CONMEBOL": 0.68, "CONCACAF": 0.45,
    "CAF": 0.42,  "AFC": 0.40,      "OFC": 0.30,
}

STAGE_ORDER = {
    "group": 1, "round_of_16": 2, "quarter_final": 3,
    "semi_final": 4, "third_place": 5, "final": 6,
}

ELO_K = {"group": 32, "round_of_16": 40, "quarter_final": 48,
          "semi_final": 56, "third_place": 48, "final": 64}

ELO_START   = 1500
DECAY       = 0.95   # after each tournament, ratings pulled toward mean by 5%
FORM_WINDOW = 5


#Helpers

def normalize_stage(s: str) -> str:
    s = str(s).lower().strip()
    if "group" in s or s in list("abcdefgh"):
        return "group"
    if "round of 16" in s or "first round" in s or "preliminary" in s:
        return "round_of_16"
    if "quarter" in s:
        return "quarter_final"
    if "semi" in s:
        return "semi_final"
    if "third" in s or "play-off" in s:
        return "third_place"
    if "final" in s:
        return "final"
    return "group"


def get_result(hg: int, ag: int, win_cond: str = "",
               home: str = "", away: str = "") -> str:
    wc = str(win_cond).lower()
    if "extra time" in wc or "penalt" in wc:
        if home.lower() in wc:
            return "H"
        if away.lower() in wc:
            return "A"
        return "H" if hg > ag else "A"
    if hg > ag:
        return "H"
    if ag > hg:
        return "A"
    return "D"


def confederation(team: str) -> str:
    return CONFEDERATION.get(team, "UEFA")   # default UEFA (most common)


def conf_strength(team: str) -> float:
    return CONF_STRENGTH.get(confederation(team), 0.45)


#Stage 1: Load & Unify

def load_and_combine(raw_dir: Path) -> pd.DataFrame:
    """Merge all 4 source files into a single match-level DataFrame."""

    x3 = pd.read_csv(raw_dir / "WorldCupMatches.csv")
    x2 = pd.read_csv(raw_dir / "2018_worldcup_v3.csv")
    x1 = pd.read_csv(raw_dir / "FifaWorldcup2022.csv")
    x4 = pd.read_csv(raw_dir / "WorldCups.csv")

    #1930–2014
    x3 = x3.drop_duplicates().dropna(subset=[
        "Year", "Home Team Name", "Away Team Name",
        "Home Team Goals", "Away Team Goals"
    ])
    x3["Year"] = x3["Year"].astype(int)
    x3["Home Team Goals"] = x3["Home Team Goals"].astype(int)
    x3["Away Team Goals"] = x3["Away Team Goals"].astype(int)
    x3["Half-time Home Goals"] = x3["Half-time Home Goals"].fillna(0).astype(int)
    x3["Half-time Away Goals"] = x3["Half-time Away Goals"].fillna(0).astype(int)

    x3["match_result"] = x3.apply(
        lambda r: get_result(r["Home Team Goals"], r["Away Team Goals"],
                             r["Win conditions"], r["Home Team Name"], r["Away Team Name"]),
        axis=1,
    )
    x3["stage"] = x3["Stage"].apply(normalize_stage)

    df1 = pd.DataFrame({
        "year":       x3["Year"],
        "home_team":  x3["Home Team Name"].str.strip(),
        "away_team":  x3["Away Team Name"].str.strip(),
        "home_goals": x3["Home Team Goals"],
        "away_goals": x3["Away Team Goals"],
        "stage":      x3["stage"],
        "match_result": x3["match_result"],
    })

    #2018
    x2["match_result"] = x2.apply(
        lambda r: get_result(r["Home Team Goals"], r["Away Team Goals"]), axis=1
    )
    x2["stage"] = x2["Stage"].apply(normalize_stage)

    df2 = pd.DataFrame({
        "year":       2018,
        "home_team":  x2["Home Team Name"].str.strip(),
        "away_team":  x2["Away Team Name"].str.strip(),
        "home_goals": x2["Home Team Goals"],
        "away_goals": x2["Away Team Goals"],
        "stage":      x2["stage"],
        "match_result": x2["match_result"],
    })

    #2022
    x1 = x1.sort_values(["Match No.", "Sl. No"]).reset_index(drop=True)
    rows_2022 = []
    for _, grp in x1.groupby("Match No."):
        grp = grp.reset_index(drop=True)
        if len(grp) != 2:
            continue
        home, away = grp.iloc[0], grp.iloc[1]
        hg, ag = int(home["Goal"]), int(away["Goal"])
        stage_raw = str(home["Group"]).strip()
        stage = "group" if stage_raw in list("ABCDEFGH") else normalize_stage(stage_raw)
        rows_2022.append({
            "year": 2022,
            "home_team":    home["Team"].strip(),
            "away_team":    away["Team"].strip(),
            "home_goals":   hg,
            "away_goals":   ag,
            "stage":        stage,
            "match_result": get_result(hg, ag),
        })
    df3 = pd.DataFrame(rows_2022)

    #Merge host info
    extra = pd.DataFrame({"Year": [2018, 2022], "Country": ["Russia", "Qatar"]})
    x4_ext = pd.concat([x4[["Year", "Country"]], extra], ignore_index=True)
    host_map = x4_ext.set_index("Year")["Country"].to_dict()

    combined = pd.concat([df1, df2, df3], ignore_index=True)
    combined = combined.sort_values("year").reset_index(drop=True)
    combined["host_country"] = combined["year"].map(host_map)
    combined["home_is_host"] = (
        combined["home_team"].str.lower() == combined["host_country"].str.lower()
    ).astype(int)
    combined["away_is_host"] = (
        combined["away_team"].str.lower() == combined["host_country"].str.lower()
    ).astype(int)
    combined["stage_num"]   = combined["stage"].map(STAGE_ORDER).fillna(1).astype(int)
    combined["is_knockout"] = (combined["stage_num"] >= 2).astype(int)

    return combined


#Stage 2: Feature Engineering

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features strictly BEFORE each match (no leakage).
    Returns enriched DataFrame with 25 feature columns.
    """
    df = df.copy().reset_index(drop=True)

    #Per-team match log (home + away perspective)
    records = []
    for idx, row in df.iterrows():
        r = row["match_result"]
        for side in ("home", "away"):
            is_home  = side == "home"
            team     = row["home_team"] if is_home else row["away_team"]
            opponent = row["away_team"] if is_home else row["home_team"]
            gf       = row["home_goals"] if is_home else row["away_goals"]
            ga       = row["away_goals"] if is_home else row["home_goals"]
            win      = (r == "H") if is_home else (r == "A")
            draw     = r == "D"
            records.append({
                "match_idx": idx,
                "year":      row["year"],
                "team":      team,
                "opponent":  opponent,
                "gf": gf, "ga": ga,
                "win": int(win), "draw": int(draw), "loss": int(not win and not draw),
                "pts": 3 if win else (1 if draw else 0),
                "wdl": "W" if win else ("D" if draw else "L"),
            })
    hist = pd.DataFrame(records)

    #Track WC appearances per team
    wc_years = df.groupby("year").first().reset_index()["year"].tolist()
    team_wc_appearances: dict[str, set] = {}
    for idx, row in df.iterrows():
        for team in (row["home_team"], row["away_team"]):
            team_wc_appearances.setdefault(team, set()).add(row["year"])

    def wc_experience(team: str, current_year: int) -> int:
        years = team_wc_appearances.get(team, set())
        return len([y for y in years if y < current_year])

    #Elo tracker (with stage-weighted K and inter-tournament decay)
    elo: dict[str, float] = {}
    last_year: dict[str, int] = {}

    def get_elo(team: str) -> float:
        return elo.get(team, ELO_START)

    def expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def update_elo(team: str, opp: str, result_score: float,
                   stage: str, year: int) -> None:
        #Decay toward mean if new tournament
        for t in (team, opp):
            if t in last_year and last_year[t] < year:
                old = elo.get(t, ELO_START)
                elo[t] = old + DECAY * (ELO_START - old)
            last_year[t] = year

        ra, rb  = get_elo(team), get_elo(opp)
        k       = ELO_K.get(stage, 32)
        ea      = expected(ra, rb)
        elo[team] = ra + k * (result_score - ea)
        elo[opp]  = rb + k * ((1 - result_score) - (1 - ea))

    #Build per-team cumulative helpers
    def past_stats(team: str, before_idx: int):
        p = hist[(hist["team"] == team) & (hist["match_idx"] < before_idx)]
        if len(p) == 0:
            return dict(n=0, gf=0, ga=0, pts=0, gpg=0.0, gcpg=0.0)
        n = len(p)
        return dict(
            n=n,
            gf=int(p["gf"].sum()),
            ga=int(p["ga"].sum()),
            pts=int(p["pts"].sum()),
            gpg=round(p["gf"].sum() / n, 3),    # goals per game
            gcpg=round(p["ga"].sum() / n, 3),   # goals conceded per game
        )

    def form_score(team: str, before_idx: int) -> float:
        """Exponentially weighted form over last FORM_WINDOW matches."""
        p = hist[(hist["team"] == team) & (hist["match_idx"] < before_idx)].tail(FORM_WINDOW)
        if len(p) == 0:
            return 1.0   #neutral (W=3 * 0.33 ≈ 1)
        wdl_scores = p["pts"].tolist()
        weights    = np.exp(np.linspace(0, 1, len(wdl_scores)))
        return float(np.average(wdl_scores, weights=weights))

    def h2h_rate(home: str, away: str, before_idx: int) -> float:
        mask = (
            (df.index < before_idx) &
            (df["home_team"] == home) &
            (df["away_team"] == away)
        )
        p = df[mask]
        if len(p) == 0:
            return 0.33
        return round((p["match_result"] == "H").mean(), 3)

    #Main loop
    rows_out = []
    for idx, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        stage  = row["stage"]
        year   = row["year"]

        h_elo_pre = get_elo(ht)
        a_elo_pre = get_elo(at)

        h_stats = past_stats(ht, idx)
        a_stats = past_stats(at, idx)
        h_form  = form_score(ht, idx)
        a_form  = form_score(at, idx)

        #Update Elo AFTER capturing pre-match ratings
        result = row["match_result"]
        s_h = 1.0 if result == "H" else (0.5 if result == "D" else 0.0)
        update_elo(ht, at, s_h, stage, year)

        h2h = h2h_rate(ht, at, idx)

        row_feats = {
            #Cumulative raw stats
            "HTGS": h_stats["gf"],   "ATGS": a_stats["gf"],
            "HTGC": h_stats["ga"],   "ATGC": a_stats["ga"],
            "HTP":  h_stats["pts"],  "ATP":  a_stats["pts"],
            #Per-game rates (NEW — normalises for teams with different match counts)
            "home_gpg":  h_stats["gpg"],  "away_gpg":  a_stats["gpg"],
            "home_gcpg": h_stats["gcpg"], "away_gcpg": a_stats["gcpg"],
            #Differentials
            "goal_diff":    h_stats["gf"]   - a_stats["gf"],
            "concede_diff": h_stats["ga"]   - a_stats["ga"],
            "points_diff":  h_stats["pts"]  - a_stats["pts"],
            "gpg_diff":     h_stats["gpg"]  - a_stats["gpg"],
            "gcpg_diff":    h_stats["gcpg"] - a_stats["gcpg"],
            #Form (exponentially weighted)
            "home_form":   h_form,
            "away_form":   a_form,
            "form_diff":   h_form - a_form,
            #Elo (stage-weighted, with decay)
            "home_elo":    h_elo_pre,
            "away_elo":    a_elo_pre,
            "elo_diff":    h_elo_pre - a_elo_pre,
            #H2H
            "h2h_home_win_rate": h2h,
            #Confederation strength
            "home_conf_strength": conf_strength(ht),
            "away_conf_strength": conf_strength(at),
            "conf_strength_diff": conf_strength(ht) - conf_strength(at),
            #Tournament experience
            "home_wc_exp": wc_experience(ht, year),
            "away_wc_exp": wc_experience(at, year),
            "wc_exp_diff": wc_experience(ht, year) - wc_experience(at, year),
            #Match context
            "stage_num":    row["stage_num"],
            "is_knockout":  row["is_knockout"],
            "home_is_host": row["home_is_host"],
            "away_is_host": row["away_is_host"],
        }
        rows_out.append(row_feats)

    feat_df = pd.DataFrame(rows_out)
    result_df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    return result_df


FEATURE_COLS = [
    "HTGS", "ATGS", "HTGC", "ATGC", "HTP", "ATP",
    "home_gpg", "away_gpg", "home_gcpg", "away_gcpg",
    "goal_diff", "concede_diff", "points_diff", "gpg_diff", "gcpg_diff",
    "home_form", "away_form", "form_diff",
    "home_elo", "away_elo", "elo_diff",
    "h2h_home_win_rate",
    "home_conf_strength", "away_conf_strength", "conf_strength_diff",
    "home_wc_exp", "away_wc_exp", "wc_exp_diff",
    "stage_num", "is_knockout", "home_is_host", "away_is_host",
]


def run(raw_dir: Path, preprocessed_dir: Path, features_dir: Path) -> pd.DataFrame:
    print("Feature Engineering Pipeline")

    print("  [1/3] Loading & combining raw datasets...")
    combined = load_and_combine(raw_dir)
    combined.to_csv(preprocessed_dir / "wc_combined.csv", index=False)
    print(f"        {len(combined)} matches saved → wc_combined.csv")

    print("  [2/3] Engineering features (no-leakage)...")
    features = build_features(combined)
    features.to_csv(features_dir / "wc_features.csv", index=False)
    print(f"        {len(features)} rows × {len(FEATURE_COLS)} features saved → wc_features.csv")

    print("  [3/3] Result distribution:")
    vc = features["match_result"].value_counts()
    for r, c in vc.items():
        print(f"        {r}: {c} ({c/len(features):.1%})")

    print("Feature Engineering Pipeline completed.\n")
    return features
