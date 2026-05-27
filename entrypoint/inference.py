"""
entrypoint/inference.py
------------------------
CLI entrypoint for match predictions, team rankings, and bracket simulation.

Usage:
    python entrypoint/inference.py --predict "Brazil" "Argentina" --stage final
    python entrypoint/inference.py --rankings
    python entrypoint/inference.py --simulate
    python entrypoint/inference.py --all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CFG
from src.pipelines.inference_pipeline import Predictor, print_prediction

FEATURES_PATH   = Path(CFG["paths"]["features"]) / "wc_features.csv"
PREDICTIONS_DIR = Path(CFG["paths"]["predictions"])

WC_2026_TEAMS = [
    "Brazil", "Argentina", "France", "England", "Spain", "Germany",
    "Portugal", "Netherlands", "Belgium", "Croatia", "Denmark", "Switzerland",
    "Uruguay", "Colombia", "Mexico", "United States", "Canada", "Japan",
    "South Korea", "Morocco", "Senegal", "Nigeria", "Cameroon", "Ghana",
    "Saudi Arabia", "Iran", "Australia", "Ecuador", "Serbia", "Poland",
    "Turkey", "Qatar",
]

DEMO_MATCHUPS = [
    ("Brazil",    "Argentina",   "final"),
    ("France",    "England",     "semi_final"),
    ("Spain",     "Germany",     "quarter_final"),
    ("Morocco",   "Netherlands", "round_of_16"),
    ("Japan",     "Colombia",    "group"),
    ("USA",       "Mexico",      "round_of_16"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Copa Oracle 2026 — Inference")
    p.add_argument("--predict",   nargs=2, metavar=("HOME", "AWAY"))
    p.add_argument("--stage",     default="group",
                   choices=["group","round_of_16","quarter_final",
                            "semi_final","third_place","final"])
    p.add_argument("--host-home", action="store_true")
    p.add_argument("--host-away", action="store_true")
    p.add_argument("--rankings",  action="store_true", help="Print Copa Oracle rankings")
    p.add_argument("--simulate",  action="store_true", help="Run Monte Carlo bracket simulation")
    p.add_argument("--n-sims",    type=int, default=10_000)
    p.add_argument("--all",       action="store_true", help="Run demo predictions + rankings")
    return p.parse_args()


def main():
    args   = parse_args()
    pred   = Predictor(PREDICTIONS_DIR, FEATURES_PATH, CFG)

    print("Copa Oracle 2026 — Inference")
    if args.predict:
        home, away = args.predict
        res = pred.predict(home, away, args.stage,
                           int(args.host_home), int(args.host_away))
        print_prediction(res)

    if args.rankings or args.all:
        scores = pred.all_team_scores()
        print("\nCopa Oracle Rankings")
        print(f"  {'#':<4} {'Team':<24} {'Score':>6}  {'Elo':>6}  {'Form':>5}  {'Conf'}")
        print(f"  {'─'*4} {'─'*24} {'─'*6}  {'─'*6}  {'─'*5}  {'─'*8}")
        for i, row in scores.head(20).iterrows():
            print(f"  {i+1:<4} {row['team']:<24} {row['copa_score']:>6.1f}  "
                  f"{row['elo']:>6.0f}  {row['form']:>5.2f}  {row['confederation']}")

    if args.simulate or args.all:
        print(f"\n Monte Carlo Bracket Simulation ({args.n_sims:,} runs)")
        sim = pred.simulate_bracket(WC_2026_TEAMS, host="USA",
                                     n_sims=args.n_sims, seed=42)
        print(f"  {'#':<4} {'Team':<24} {'Win %':>6}  {'Copa Score':>10}")
        print(f"  {'─'*4} {'─'*24} {'─'*6}  {'─'*10}")
        for i, row in sim.head(16).iterrows():
            bar = "" * int(row["win_prob"] / 2)
            print(f"  {i+1:<4} {row['team']:<24} {row['win_prob']:>5.1f}%  "
                  f"{row['copa_score']:>10.1f}  {bar}")

    if args.all:
        print("\nDemo Match Predictions")
        for home, away, stage in DEMO_MATCHUPS:
            res = pred.predict(home, away, stage)
            print_prediction(res)

    print()


if __name__ == "__main__":
    main()
