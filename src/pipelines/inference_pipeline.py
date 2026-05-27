"""
src/pipelines/inference_pipeline.py
-------------------------------------
Stage 4: Trained model → match predictions, Copa Oracle scores,
         bracket simulation (Monte Carlo), mispricing detection.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .feature_eng_pipeline import (
    FEATURE_COLS, ELO_START, ELO_K, DECAY, FORM_WINDOW,
    CONFEDERATION, CONF_STRENGTH, STAGE_ORDER,
    get_result, conf_strength, confederation,
)


#Copa Oracle Score

def copa_oracle_score(elo: float, form: float, gpg: float, gcpg: float,
                       elo_min: float, elo_max: float,
                       form_min: float, form_max: float,
                       gpg_min: float, gpg_max: float,
                       gcpg_min: float, gcpg_max: float,
                       weights: dict) -> float:
    def norm(v, lo, hi):
        if hi == lo:
            return 0.5
        return max(0.0, min(1.0, (v - lo) / (hi - lo)))

    elo_n    = norm(elo,   elo_min,   elo_max)
    form_n   = norm(form,  form_min,  form_max)
    attack_n = norm(gpg,   gpg_min,   gpg_max)
    def_n    = 1.0 - norm(gcpg, gcpg_min, gcpg_max)  # lower concede = better

    score = (
        elo_n    * weights.get("elo_weight",    0.40) +
        form_n   * weights.get("form_weight",   0.25) +
        attack_n * weights.get("attack_weight", 0.20) +
        def_n    * weights.get("defense_weight",0.15)
    )
    return round(score * 100, 2)


#Predictor class

class Predictor:
    """
    Wraps the trained model + team profiles rebuilt from feature data.
    Thread-safe for concurrent requests.
    """

    def __init__(self, predictions_dir: Path, features_path: Path, cfg: dict):
        self.cfg = cfg
        self._load_model(predictions_dir)
        self._build_profiles(features_path)

    def _load_model(self, predictions_dir: Path) -> None:
        self.model         = joblib.load(predictions_dir / "model.pkl")
        self.label_encoder = joblib.load(predictions_dir / "label_encoder.pkl")
        self.feature_cols  = joblib.load(predictions_dir / "feature_cols.pkl")

    def _build_profiles(self, features_path: Path) -> None:
        """Rebuild team profiles from the feature CSV (last known state per team)."""
        df = pd.read_csv(features_path)

        #Final Elo (recomputed post-last-match for each team)
        elo: dict[str, float] = {}
        def expected(ra, rb): return 1 / (1 + 10 ** ((rb - ra) / 400))
        last_year: dict[str, int] = {}

        for _, row in df.iterrows():
            ht, at = row["home_team"], row["away_team"]
            year   = row["year"]
            stage  = row["stage"]
            for t in (ht, at):
                if t in last_year and last_year[t] < year:
                    old = elo.get(t, ELO_START)
                    elo[t] = old + DECAY * (ELO_START - old)
                last_year[t] = year
            ra, rb = elo.get(ht, ELO_START), elo.get(at, ELO_START)
            r = row["match_result"]
            s_h = 1.0 if r == "H" else (0.5 if r == "D" else 0.0)
            k = ELO_K.get(stage, 32)
            elo[ht] = ra + k * (s_h - expected(ra, rb))
            elo[at] = rb + k * ((1 - s_h) - (1 - expected(ra, rb)))

        #Last-seen profile per team
        profiles: dict[str, dict] = {}
        for _, row in df.iterrows():
            profiles[row["home_team"]] = {
                "elo":    row["home_elo"],
                "form":   row["home_form"],
                "gpg":    row["home_gpg"],
                "gcpg":   row["home_gcpg"],
                "gf":     row["HTGS"],
                "ga":     row["HTGC"],
                "pts":    row["HTP"],
                "wc_exp": row["home_wc_exp"],
            }
            profiles[row["away_team"]] = {
                "elo":    row["away_elo"],
                "form":   row["away_form"],
                "gpg":    row["away_gpg"],
                "gcpg":   row["away_gcpg"],
                "gf":     row["ATGS"],
                "ga":     row["ATGC"],
                "pts":    row["ATP"],
                "wc_exp": row["away_wc_exp"],
            }

        #Override with final Elo
        for team in profiles:
            profiles[team]["final_elo"] = elo.get(team, ELO_START)

        self.profiles = profiles
        self.elo_final = elo
        self._df = df

        #For Copa Oracle normalisation
        elos   = [p["final_elo"] for p in profiles.values()]
        forms  = [p["form"]      for p in profiles.values()]
        gpgs   = [p["gpg"]       for p in profiles.values()]
        gcpgs  = [p["gcpg"]      for p in profiles.values()]
        self._norm_bounds = {
            "elo_min":   min(elos),   "elo_max":   max(elos),
            "form_min":  min(forms),  "form_max":  max(forms),
            "gpg_min":   min(gpgs),   "gpg_max":   max(gpgs),
            "gcpg_min":  min(gcpgs),  "gcpg_max":  max(gcpgs),
        }

    def _default_profile(self) -> dict:
        return {"elo": ELO_START, "final_elo": ELO_START, "form": 1.0,
                "gpg": 0.0, "gcpg": 0.0, "gf": 0, "ga": 0, "pts": 0, "wc_exp": 0}

    def get_profile(self, team: str) -> dict:
        return self.profiles.get(team, self._default_profile())

    def h2h(self, home: str, away: str) -> float:
        df = self._df
        mask = (df["home_team"] == home) & (df["away_team"] == away)
        p = df[mask]
        if len(p) == 0:
            return 0.33
        return round((p["match_result"] == "H").mean(), 3)

    def h2h_history(self, team_a: str, team_b: str) -> pd.DataFrame:
        df = self._df
        mask = (
            ((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
            ((df["home_team"] == team_b) & (df["away_team"] == team_a))
        )
        return df[mask][["year","stage","home_team","away_team",
                          "home_goals","away_goals","match_result"]].copy()

    def predict(self, home: str, away: str,
                stage: str = "group",
                is_host_home: int = 0,
                is_host_away: int = 0) -> dict:

        hp = self.get_profile(home)
        ap = self.get_profile(away)
        h_elo = hp["final_elo"]
        a_elo = ap["final_elo"]
        stage_num  = STAGE_ORDER.get(stage, 1)
        is_knockout = int(stage_num >= 2)

        feats = {
            "HTGS": hp["gf"],  "ATGS": ap["gf"],
            "HTGC": hp["ga"],  "ATGC": ap["ga"],
            "HTP":  hp["pts"], "ATP":  ap["pts"],
            "home_gpg":  hp["gpg"],  "away_gpg":  ap["gpg"],
            "home_gcpg": hp["gcpg"], "away_gcpg": ap["gcpg"],
            "goal_diff":    hp["gf"]  - ap["gf"],
            "concede_diff": hp["ga"]  - ap["ga"],
            "points_diff":  hp["pts"] - ap["pts"],
            "gpg_diff":     hp["gpg"] - ap["gpg"],
            "gcpg_diff":    hp["gcpg"]- ap["gcpg"],
            "home_form":    hp["form"],   "away_form":    ap["form"],
            "form_diff":    hp["form"]  - ap["form"],
            "home_elo":     h_elo,        "away_elo":     a_elo,
            "elo_diff":     h_elo - a_elo,
            "h2h_home_win_rate": self.h2h(home, away),
            "home_conf_strength": conf_strength(home),
            "away_conf_strength": conf_strength(away),
            "conf_strength_diff": conf_strength(home) - conf_strength(away),
            "home_wc_exp": hp["wc_exp"], "away_wc_exp": ap["wc_exp"],
            "wc_exp_diff":  hp["wc_exp"] - ap["wc_exp"],
            "stage_num":    stage_num,   "is_knockout": is_knockout,
            "home_is_host": is_host_home,"away_is_host": is_host_away,
        }

        X = pd.DataFrame([feats])[self.feature_cols]
        proba   = self.model.predict_proba(X)[0]
        classes = self.model.classes_

        prob_map = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}
        predicted = max(prob_map, key=prob_map.get)

        return {
            "home":      home,
            "away":      away,
            "stage":     stage,
            "probs":     prob_map,
            "predicted": predicted,
            "home_elo":  round(h_elo, 1),
            "away_elo":  round(a_elo, 1),
            "home_form": round(hp["form"], 2),
            "away_form": round(ap["form"], 2),
        }

    def team_score(self, team: str) -> float:
        p = self.get_profile(team)
        return copa_oracle_score(
            elo=p["final_elo"], form=p["form"], gpg=p["gpg"], gcpg=p["gcpg"],
            weights=self.cfg.get("copa_oracle_score", {}),
            **self._norm_bounds,
        )

    def all_team_scores(self) -> pd.DataFrame:
        rows = []
        for team in sorted(self.profiles):
            rows.append({
                "team":         team,
                "copa_score":   self.team_score(team),
                "elo":          round(self.profiles[team]["final_elo"], 1),
                "form":         round(self.profiles[team]["form"], 2),
                "gpg":          round(self.profiles[team]["gpg"], 2),
                "gcpg":         round(self.profiles[team]["gcpg"], 2),
                "confederation": confederation(team),
                "wc_exp":       self.profiles[team]["wc_exp"],
            })
        return pd.DataFrame(rows).sort_values("copa_score", ascending=False).reset_index(drop=True)

    def simulate_bracket(self, teams: list[str],
                          host: str = "USA",
                          n_sims: int = 10_000,
                          seed: int = 42) -> pd.DataFrame:
        """
        Monte Carlo simulation for a 32-team World Cup 2026 bracket.
        Returns win counts and probabilities per team.
        """
        rng   = np.random.default_rng(seed)
        wins  = {t: 0 for t in teams}
        finals_appearances = {t: 0 for t in teams}

        if len(teams) % 2 != 0:
            teams = teams[:len(teams) - 1]

        for _ in range(n_sims):
            remaining = list(teams)
            rng.shuffle(remaining)

            stage_names = ["round_of_16", "quarter_final", "semi_final", "final"]
            stage_idx = 0

            while len(remaining) > 1:
                stage = stage_names[min(stage_idx, len(stage_names) - 1)]
                next_round = []
                for i in range(0, len(remaining), 2):
                    ht, at = remaining[i], remaining[i + 1]
                    is_host_h = int(ht.lower() == host.lower())
                    is_host_a = int(at.lower() == host.lower())
                    res = self.predict(ht, at, stage, is_host_h, is_host_a)
                    p = res["probs"]
                    # In knockout, draw = 50/50 tiebreak
                    ph = p.get("H", 33.3) / 100
                    pd_ = p.get("D", 33.3) / 100
                    pa = p.get("A", 33.3) / 100
                    rand = rng.random()
                    if rand < ph:
                        next_round.append(ht)
                    elif rand < ph + pd_:
                        next_round.append(ht if rng.random() < 0.5 else at)
                    else:
                        next_round.append(at)
                remaining = next_round
                stage_idx += 1

            if remaining:
                wins[remaining[0]] += 1
                finals_appearances[remaining[0]] += 1

        rows = [
            {
                "team":      t,
                "wins":      wins[t],
                "win_prob":  round(wins[t] / n_sims * 100, 2),
                "copa_score": self.team_score(t),
            }
            for t in teams
        ]
        return pd.DataFrame(rows).sort_values("win_prob", ascending=False).reset_index(drop=True)

    def find_mispricings(self, matchups: list[tuple],
                          threshold: float = 10.0) -> pd.DataFrame:
        """
        Compare model probabilities against market odds.
        matchups: list of (home, away, stage, market_home%, market_draw%, market_away%)
        """
        rows = []
        for item in matchups:
            home, away, stage = item[0], item[1], item[2]
            market = {"H": float(item[3]), "D": float(item[4]), "A": float(item[5])}
            res = self.predict(home, away, stage)
            model = res["probs"]
            for outcome, label in [("H", home), ("D", "Draw"), ("A", away)]:
                diff = model.get(outcome, 0) - market.get(outcome, 0)
                if abs(diff) >= threshold:
                    rows.append({
                        "match":    f"{home} vs {away}",
                        "outcome":  label,
                        "model_%":  model.get(outcome, 0),
                        "market_%": market.get(outcome, 0),
                        "edge":     round(diff, 1),
                        "signal":   "BUY" if diff > 0 else "SELL",
                    })
        return pd.DataFrame(rows).sort_values("edge", ascending=False).reset_index(drop=True)


#Pretty print helper

def print_prediction(res: dict) -> None:
    home, away = res["home"], res["away"]
    probs = res["probs"]
    pred  = res["predicted"]
    label = {
        "H": f"{home} wins",
        "D": "Draw",
        "A": f"{away} wins",
    }[pred]

    sep = "═" * 54
    print(f"\n{sep}")
    print(f"  {home}  vs  {away}  [{res['stage'].replace('_',' ').title()}]")
    print(sep)
    print(f"  Prediction  →  {label}")
    print()
    print(f"  {'Outcome':<24}  {'Probability':>11}  Bar")
    print(f"  {'─'*24}  {'─'*11}  {'─'*20}")
    for outcome, lbl in [("H", home), ("D", "Draw"), ("A", away)]:
        p   = probs.get(outcome, 0)
        bar = "█" * int(p / 5)
        print(f"  {lbl:<24}  {p:>10.1f}%  {bar}")
    print()
    print(f"  Elo  →  {home}: {res['home_elo']:.0f}  |  {away}: {res['away_elo']:.0f}")
    print(f"  Form →  {home}: {res['home_form']:.2f}  |  {away}: {res['away_form']:.2f}")
    print(sep)


def run(predictions_dir: Path, features_path: Path, cfg: dict,
        matchups: Optional[list] = None) -> Predictor:

    print("Inference Pipeline")
    predictor = Predictor(predictions_dir, features_path, cfg)
    print(f"  Loaded model | {len(predictor.profiles)} teams profiled")

    if matchups:
        print(f"\n  Running {len(matchups)} predictions...\n")
        for m in matchups:
            res = predictor.predict(*m)
            print_prediction(res)

    print("Inference Pipeline completed.\n")
    return predictor
