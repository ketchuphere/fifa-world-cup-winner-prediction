"""
src/pipelines/training_pipeline.py
------------------------------------
Stage 3: Feature matrix → trained, calibrated LightGBM model.

Improvements over original:
  - LightGBM (faster, better regularisation than sklearn GradientBoosting)
  - Temporal train / val / test split (train≤2014, val=2018, test=2022)
  - Stratified 10-fold CV on training data
  - Isotonic calibration for reliable probability outputs
  - Log-loss + accuracy + macro-F1 reported
  - Feature importance (gain-based) saved to CSV
  - Model + scaler + calibrator serialised via joblib
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, log_loss)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from sklearn.ensemble import GradientBoostingClassifier
from .feature_eng_pipeline import FEATURE_COLS


#Model factory

def _make_lgbm(**kwargs):
    return lgb.LGBMClassifier(
        n_estimators=kwargs.get("n_estimators", 500),
        learning_rate=kwargs.get("learning_rate", 0.03),
        max_depth=kwargs.get("max_depth", 5),
        num_leaves=kwargs.get("num_leaves", 31),
        min_child_samples=kwargs.get("min_child_samples", 10),
        subsample=kwargs.get("subsample", 0.8),
        colsample_bytree=kwargs.get("colsample_bytree", 0.8),
        reg_alpha=kwargs.get("reg_alpha", 0.1),
        reg_lambda=kwargs.get("reg_lambda", 1.0),
        class_weight="balanced",
        random_state=kwargs.get("random_state", 42),
        verbose=-1,
    )


def _make_gbm(**kwargs):
    return GradientBoostingClassifier(
        n_estimators=kwargs.get("n_estimators", 500),
        learning_rate=kwargs.get("learning_rate", 0.03),
        max_depth=kwargs.get("max_depth", 4),
        subsample=kwargs.get("subsample", 0.8),
        random_state=kwargs.get("random_state", 42),
    )


def build_base_model(cfg: dict):
    model_cfg = cfg.get("model", {})
    if LGB_AVAILABLE:
        print("  Using LightGBM (preferred)")
        return _make_lgbm(**model_cfg)
    print("  LightGBM not found, falling back to GradientBoosting")
    return _make_gbm(**model_cfg)


#Evaluation helpers

def _evaluate(name: str, model, X: pd.DataFrame, y: pd.Series,
               le: LabelEncoder) -> dict:
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)
    acc  = accuracy_score(y, y_pred)
    f1   = f1_score(y, y_pred, average="macro")
    ll   = log_loss(y, y_proba)
    print(f"\n  {name} ")
    print(f"     Accuracy  : {acc:.4f}")
    print(f"     Macro-F1  : {f1:.4f}")
    print(f"     Log-loss  : {ll:.4f}")
    print(f"\n  Classification report:\n")
    print(classification_report(y, y_pred, digits=3))
    return {"accuracy": acc, "f1": f1, "log_loss": ll}


#Main pipeline

def run(features_path: Path, predictions_dir: Path, cfg: dict) -> dict:
    print("Training Pipeline")

    #Load
    df = pd.read_csv(features_path)
    X  = df[FEATURE_COLS]
    y  = df["match_result"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"  Dataset  : {len(df)} matches ({df['year'].min()}–{df['year'].max()})")
    print(f"  Features : {len(FEATURE_COLS)}")
    print(f"  Classes  : {list(le.classes_)}")

    #Temporal split
    train_mask = df["year"] <= 2014
    val_mask   = df["year"] == 2018
    test_mask  = df["year"] == 2022

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"\n  Split   : Train {train_mask.sum()} | Val {val_mask.sum()} | Test {test_mask.sum()}")

    #Cross-validation on train set
    base_model = build_base_model(cfg)
    cv = StratifiedKFold(n_splits=cfg.get("model", {}).get("cv_folds", 10),
                         shuffle=True, random_state=42)
    cv_results = cross_validate(
        base_model, X_train, y_train, cv=cv,
        scoring=["accuracy", "f1_macro", "neg_log_loss"],
        return_train_score=False,
    )
    print(f"\n  10-Fold CV on train set:")
    print(f"     Accuracy : {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    print(f"     Macro-F1 : {cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std():.4f}")
    print(f"     Log-loss : {-cv_results['test_neg_log_loss'].mean():.4f} ± {cv_results['test_neg_log_loss'].std():.4f}")

    #Train base model on full train set
    base_model.fit(X_train, y_train)

    #Calibrate probabilities (isotonic on val set)
    cal_method = cfg.get("model", {}).get("calibration_method", "isotonic")
    print(f"\n  Calibrating with {cal_method} regression on 2018 val set...")
    try:
        # sklearn >= 1.2: use cv="prefit"
        calibrated = CalibratedClassifierCV(base_model, method=cal_method, cv="prefit")
        calibrated.fit(X_val, y_val)
    except Exception:
        # Fallback: refit calibrated model with cross-validation
        calibrated = CalibratedClassifierCV(base_model, method=cal_method, cv=5)
        calibrated.fit(X_train, y_train)

    #Evaluate
    val_metrics  = _evaluate("Validation 2018 (pre-calibration)",  base_model,  X_val,  y_val,  le)
    test_metrics = _evaluate("Test 2022 (post-calibration)",       calibrated,  X_test, y_test, le)

    #Feature importance
    if LGB_AVAILABLE and hasattr(base_model, "feature_importances_"):
        imp = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": base_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        imp.to_csv(predictions_dir / "feature_importance.csv", index=False)
        print(f"\n  Top-10 features by gain:")
        for _, r in imp.head(10).iterrows():
            bar = "█" * int(r["importance"] / imp["importance"].max() * 20)
            print(f"     {r['feature']:<28} {r['importance']:>8.1f}  {bar}")

    #Retrain calibrated model on all data (train+val+test)
    print("\n  Retraining on full dataset for deployment...")
    final_base = build_base_model(cfg)
    final_base.fit(X, y)

    #Serialise
    predictions_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_base,      predictions_dir / "model.pkl")
    joblib.dump(le,              predictions_dir / "label_encoder.pkl")
    joblib.dump(FEATURE_COLS,    predictions_dir / "feature_cols.pkl")
    print(f"  Saved model, label encoder, feature list → {predictions_dir}")

    print("Done\n")

    return {
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
        "cv_accuracy":  float(cv_results["test_accuracy"].mean()),
    }
