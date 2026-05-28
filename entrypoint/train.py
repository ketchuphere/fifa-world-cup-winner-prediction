"""
entrypoint/train.py
--------------------
CLI entrypoint: runs feature engineering then training.

Usage:
    python entrypoint/train.py
    python entrypoint/train.py --skip-features   # if wc_features.csv already exists
"""

import argparse
import sys
from pathlib import Path

#Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CFG
from src.pipelines.feature_eng_pipeline import run as run_feature_eng
from src.pipelines.training_pipeline import run as run_training


def parse_args():
    p = argparse.ArgumentParser(description="Copa Oracle 2026 — Training Pipeline")
    p.add_argument("--skip-features", action="store_true",
                   help="Skip feature engineering if wc_features.csv already exists")
    return p.parse_args()


def main():
    args = parse_args()

    raw_dir            = Path(CFG["paths"]["raw"])
    preprocessed_dir   = Path(CFG["paths"]["preprocessed"])
    features_dir       = Path(CFG["paths"]["features"])
    predictions_dir    = Path(CFG["paths"]["predictions"])

    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    features_file = features_dir / "wc_features.csv"

    print("          Copa Oracle 2026 — Training Run             ")
    

    #Stage 1 + 2: Feature Engineering
    if args.skip_features and features_file.exists():
        print(f"  Skipping feature engineering (found {features_file})\n")
    else:
        run_feature_eng(raw_dir, preprocessed_dir, features_dir)

    #Stage 3: Training
    metrics = run_training(features_file, predictions_dir, CFG)

    print("                 Final Metrics Summary                ")
    print(f"  Val  2018 accuracy : {metrics['val_metrics']['accuracy']:.4f}")
    print(f"  Val  2018 log-loss : {metrics['val_metrics']['log_loss']:.4f}")
    print(f"  Test 2022 accuracy : {metrics['test_metrics']['accuracy']:.4f}")
    print(f"  Test 2022 log-loss : {metrics['test_metrics']['log_loss']:.4f}")
    print(f"  CV   accuracy      : {metrics['cv_accuracy']:.4f}")
    print(f"  CV   macro-F1      : {metrics['cv_f1']:.4f}")
    print(f"  CV   log-loss      : {metrics['cv_log_loss']:.4f}")
    print(" \n")


if __name__ == "__main__":
    main()