"""
app/streamlit/loader.py
------------------------
Cached predictor loader shared across all Streamlit pages.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from config import CFG
from src.pipelines.inference_pipeline import Predictor

FEATURES_PATH   = Path(CFG["paths"]["features"]) / "wc_features.csv"
PREDICTIONS_DIR = Path(CFG["paths"]["predictions"])

WC_2026_TEAMS = [
    "Argentina", "Australia", "Belgium", "Brazil", "Cameroon", "Canada",
    "Chile", "Colombia", "Croatia", "Denmark", "Ecuador", "England",
    "France", "Germany", "Ghana", "Iran", "Japan", "Mexico",
    "Morocco", "Netherlands", "Nigeria", "Poland", "Portugal",
    "Saudi Arabia", "Senegal", "Serbia", "South Korea", "Spain",
    "Switzerland", "Turkey", "United States", "Uruguay",
]

@st.cache_resource(show_spinner="Loading Copa Oracle model...")
def get_predictor() -> Predictor:
    return Predictor(PREDICTIONS_DIR, FEATURES_PATH, CFG)
