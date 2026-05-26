# Copa Oracle 2026 ⚽

A production-grade ML pipeline for predicting FIFA World Cup 2026 match outcomes.
Built as a complete rewrite of the original Copa Oracle project with improved accuracy,
cleaner architecture, and a full DS project structure.

---

## Results

| Metric | Original (GBM) | This Version (LightGBM) |
|--------|---------------|------------------------|
| CV Accuracy (10-fold) | ~55% | **59.2%** |
| Test Accuracy (2022) | ~56% | **48.4%** |
| Features | 20 | **32** |
| Probability calibration | ✗ | **✓ isotonic** |
| Temporal holdout | ✗ | **✓ 2018+2022** |
| Leakage prevention | ✓ | **✓** |

> Note: 2022 test set is only 64 matches, making accuracy volatile. CV accuracy on 836
> training matches (59.2%) is the more reliable generalisation estimate.

---

## Project Structure

```
copa-oracle-2026/
├── config/
│   ├── config.yml          ← All hyperparameters & paths
│   └── config.py           ← Config loader (exposes CFG dict)
├── data/
│   ├── 01-raw/             ← Original CSV files (never modified)
│   ├── 02-preprocessed/    ← wc_combined.csv (964 matches, 1930–2022)
│   ├── 03-features/        ← wc_features.csv (32 engineered features)
│   └── 04-predictions/     ← model.pkl, label_encoder.pkl, feature_importance.csv
├── entrypoint/
│   ├── train.py            ← Run feature eng + training
│   └── inference.py        ← Predictions, rankings, bracket simulation
├── notebooks/
│   ├── EDA.ipynb           ← Exploratory data analysis
│   └── Baseline.ipynb      ← Baseline model benchmarks
├── src/pipelines/
│   ├── feature_eng_pipeline.py   ← Stages 1 & 2
│   ├── training_pipeline.py      ← Stage 3
│   └── inference_pipeline.py     ← Stage 4
├── tests/
│   └── test_training.py    ← 29 unit + integration tests
├── Dockerfile
├── docker-compose.yml
├── .gitlab-ci.yml
├── Makefile
├── requirements.txt
└── requirements-dev.txt
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train
make train

# 3. Predict
python entrypoint/inference.py --predict "Brazil" "Argentina" --stage final

# 4. Rankings
python entrypoint/inference.py --rankings

# 5. Bracket simulation (10,000 Monte Carlo runs)
python entrypoint/inference.py --simulate

# 6. Tests
make test
```

---

## Key Improvements Over Original

### Features (20 → 32)
| New Feature | Why it helps |
|---|---|
| `home_gpg` / `away_gpg` | Goals per game normalises for teams with fewer matches |
| `home_gcpg` / `away_gcpg` | Concede rate, not raw total |
| `gpg_diff` / `gcpg_diff` | Direct attack vs defence comparison |
| `home_conf_strength` | UEFA/CONMEBOL teams historically stronger |
| `home_wc_exp` / `away_wc_exp` | Experienced teams handle pressure better |
| `wc_exp_diff` | Experience gap between the two sides |

### Elo System
- **Stage-weighted K**: K=32 (group), K=48 (QF), K=64 (final) — big wins in finals count more
- **Inter-tournament decay**: Ratings drift 5% toward 1500 between World Cups, preventing stale ratings from dominating

### Model
- **LightGBM** over sklearn's GradientBoosting: better regularisation, native categorical support, 3–5× faster
- **Isotonic calibration**: raw classifier probabilities are over-confident; calibration makes them reliable
- **Temporal split**: train on 1930–2014, validate on 2018, test on 2022 — no future data leaks into training

### Evaluation
- **Log-loss** alongside accuracy: penalises confident wrong predictions, more informative for probability outputs
- **Macro-F1**: handles class imbalance (draws are rare at 19.5%), unlike accuracy
- **10-fold stratified CV**: more reliable than the original 5-fold

---

## Copa Oracle Score

Each team receives a 0–100 score computed as a weighted combination of normalised metrics:

```
Score = (Elo × 40%) + (Form × 25%) + (Attack rate × 20%) + (Defence rate × 15%)
```

All components are min-max normalised across all known teams before weighting.

---

## Docker

```bash
# Build
docker compose build

# Train
docker compose run train

# Inference
docker compose run inference

# Tests
docker compose run test
```

---

## Configuration

All tunable parameters live in `config/config.yml`. No hardcoded values in source files.

Key sections:
- `model` — LightGBM hyperparameters, CV folds, calibration method
- `elo` — K-factors per stage, decay factor
- `copa_oracle_score` — component weights
- `monte_carlo` — number of simulations, random seed
- `confederation_strength` — historical win-rate weights per confederation
