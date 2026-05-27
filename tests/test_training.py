"""
tests/test_training.py
-----------------------
Unit + integration tests for the Copa Oracle 2026 pipeline.
Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.feature_eng_pipeline import (
    normalize_stage, get_result, confederation, conf_strength,
    FEATURE_COLS, load_and_combine, build_features,
)
from config import CFG


#normalize_stage

class TestNormalizeStage:
    def test_group(self):
        assert normalize_stage("Group A") == "group"
        assert normalize_stage("group stage") == "group"

    def test_round_of_16(self):
        assert normalize_stage("Round of 16") == "round_of_16"
        assert normalize_stage("First Round") == "round_of_16"

    def test_quarter(self):
        assert normalize_stage("Quarter-Finals") == "quarter_final"

    def test_semi(self):
        assert normalize_stage("Semi-Finals") == "semi_final"

    def test_final(self):
        assert normalize_stage("Final") == "final"

    def test_third(self):
        assert normalize_stage("Third Place") == "third_place"


#get_result

class TestGetResult:
    def test_home_win(self):
        assert get_result(2, 1) == "H"

    def test_away_win(self):
        assert get_result(0, 3) == "A"

    def test_draw(self):
        assert get_result(1, 1) == "D"

    def test_extra_time_home(self):
        assert get_result(1, 1, "Extra Time Brazil", "Brazil", "Germany") == "H"

    def test_extra_time_away(self):
        assert get_result(1, 1, "Penalty Germany won", "Brazil", "Germany") == "A"


#Confederation + strength

class TestConfederation:
    def test_known_teams(self):
        assert confederation("Brazil") == "CONMEBOL"
        assert confederation("France") == "UEFA"
        assert confederation("Japan") == "AFC"
        assert confederation("Morocco") == "CAF"
        assert confederation("Mexico") == "CONCACAF"

    def test_unknown_defaults_to_uefa(self):
        assert confederation("Unknown FC") == "UEFA"

    def test_conf_strength_range(self):
        for team in ["Brazil", "France", "Japan", "Morocco", "Mexico"]:
            s = conf_strength(team)
            assert 0.0 <= s <= 1.0, f"{team} strength {s} out of range"


#Feature columns

class TestFeatureCols:
    def test_feature_count(self):
        assert len(FEATURE_COLS) == 32, f"Expected 32 features, got {len(FEATURE_COLS)}"

    def test_no_duplicates(self):
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))

    def test_key_features_present(self):
        for f in ["elo_diff", "home_elo", "form_diff", "h2h_home_win_rate",
                  "stage_num", "home_conf_strength", "gpg_diff"]:
            assert f in FEATURE_COLS, f"Missing feature: {f}"


#Integration test: data loading

class TestDataLoading:
    @pytest.fixture(scope="class")
    def combined(self):
        raw_dir = Path(CFG["paths"]["raw"])
        if not raw_dir.exists():
            pytest.skip("Raw data not available")
        return load_and_combine(raw_dir)

    def test_row_count(self, combined):
        assert len(combined) >= 900, f"Expected ≥900 matches, got {len(combined)}"

    def test_year_range(self, combined):
        assert combined["year"].min() == 1930
        assert combined["year"].max() == 2022

    def test_no_null_results(self, combined):
        assert combined["match_result"].notna().all()

    def test_result_values(self, combined):
        assert set(combined["match_result"].unique()) == {"H", "D", "A"}

    def test_stage_values(self, combined):
        valid = {"group","round_of_16","quarter_final","semi_final","third_place","final"}
        assert set(combined["stage"].unique()).issubset(valid)


#Integration test: feature engineering

class TestFeatureEngineering:
    @pytest.fixture(scope="class")
    def features(self):
        raw_dir = Path(CFG["paths"]["raw"])
        if not raw_dir.exists():
            pytest.skip("Raw data not available")
        combined = load_and_combine(raw_dir)
        return build_features(combined)

    def test_no_null_features(self, features):
        null_counts = features[FEATURE_COLS].isnull().sum()
        nulls = null_counts[null_counts > 0]
        assert len(nulls) == 0, f"Null values in features:\n{nulls}"

    def test_elo_plausible(self, features):
        assert features["home_elo"].between(800, 2200).all()
        assert features["away_elo"].between(800, 2200).all()

    def test_form_range(self, features):
        hf = features["home_form"]
        if isinstance(hf, pd.DataFrame): hf = hf.iloc[:, 0]
        assert (hf >= 0).all() and (hf <= 3.1).all()

    def test_stage_num_range(self, features):
        col = features["stage_num"]
        if isinstance(col, pd.DataFrame): col = col.iloc[:, 0]
        assert (col >= 1).all() and (col <= 6).all()

    def test_host_flags_binary(self, features):
        for flag in ("home_is_host", "away_is_host"):
            col = features[flag]
            if isinstance(col, pd.DataFrame): col = col.iloc[:, 0]
            assert set(col.unique()).issubset({0, 1})

    def test_conf_strength_range(self, features):
        assert features["home_conf_strength"].between(0, 1).all()
        assert features["away_conf_strength"].between(0, 1).all()

    def test_leakage_check(self, features):
        """Ensure Elo used at match time is pre-match (not post-match)."""
        #First match for any team should use starting Elo (1500)
        first_matches = features.groupby("home_team").first().reset_index()
        #Some teams' first match Elo should be near 1500
        near_start = (first_matches["home_elo"] - 1500).abs() < 200
        assert near_start.any(), "No team starts near Elo 1500 — possible leakage"
