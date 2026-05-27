"""
app/api/main.py
----------------
Copa Oracle 2026 — FastAPI Backend

Endpoints:
  GET  /                         → HTML dashboard
  GET  /api/teams                → list of all known teams
  POST /api/predict              → match prediction
  GET  /api/rankings             → Copa Oracle team rankings
  POST /api/simulate             → Monte Carlo bracket simulation
  POST /api/mispricings          → market mispricing detection
  GET  /api/team/{name}          → single team profile
  GET  /health                   → health check

Run with:
    uvicorn app.api.main:app --reload --port 8000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import functools

from config import CFG
from src.pipelines.inference_pipeline import Predictor
from src.pipelines.feature_eng_pipeline import confederation

FEATURES_PATH   = Path(CFG["paths"]["features"]) / "wc_features.csv"
PREDICTIONS_DIR = Path(CFG["paths"]["predictions"])

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Copa Oracle 2026 API",
    description="FIFA World Cup 2026 ML Prediction Engine",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML dashboard)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ── Cached predictor ──────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor(PREDICTIONS_DIR, FEATURES_PATH, CFG)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    home:         str   = Field(..., example="Brazil")
    away:         str   = Field(..., example="Argentina")
    stage:        str   = Field("group", example="final")
    host_home:    int   = Field(0, ge=0, le=1)
    host_away:    int   = Field(0, ge=0, le=1)

class SimulateRequest(BaseModel):
    teams:        list[str] = Field(..., min_length=4)
    host:         str       = Field("United States")
    n_sims:       int       = Field(5000, ge=100, le=50_000)
    seed:         int       = Field(42)

class MispricingEntry(BaseModel):
    home:         str
    away:         str
    stage:        str   = "group"
    market_home:  float = Field(..., ge=0, le=100)
    market_draw:  float = Field(..., ge=0, le=100)
    market_away:  float = Field(..., ge=0, le=100)

class MispricingsRequest(BaseModel):
    matchups:     list[MispricingEntry]
    threshold:    float = Field(8.0, ge=0, le=50)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "LightGBM", "version": "2.0.0"}


@app.get("/api/teams")
def list_teams():
    pred = get_predictor()
    return {
        "teams": sorted(pred.profiles.keys()),
        "count": len(pred.profiles),
    }


@app.get("/api/team/{name}")
def team_profile(name: str):
    pred = get_predictor()
    if name not in pred.profiles:
        raise HTTPException(status_code=404, detail=f"Team '{name}' not found.")
    p = pred.profiles[name]
    return {
        "team":          name,
        "elo":           round(p["final_elo"], 1),
        "form":          round(p["form"], 3),
        "gpg":           round(p["gpg"], 3),
        "gcpg":          round(p["gcpg"], 3),
        "wc_exp":        p["wc_exp"],
        "confederation": confederation(name),
        "copa_score":    pred.team_score(name),
    }


@app.post("/api/predict")
def predict_match(req: PredictRequest):
    pred = get_predictor()
    try:
        res = pred.predict(req.home, req.away, req.stage, req.host_home, req.host_away)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return res


@app.get("/api/rankings")
def rankings(
    confederation: Optional[str] = None,
    top_n: int = 32,
    sort_by: str = "copa_score",
):
    pred = get_predictor()
    df   = pred.all_team_scores()
    if confederation:
        df = df[df["confederation"] == confederation.upper()]
    valid_sorts = {"copa_score", "elo", "form", "gpg", "gcpg", "wc_exp"}
    if sort_by not in valid_sorts:
        sort_by = "copa_score"
    df = df.sort_values(sort_by, ascending=(sort_by == "gcpg")).head(top_n)
    return {"rankings": df.to_dict(orient="records")}


@app.post("/api/simulate")
def simulate_bracket(req: SimulateRequest):
    pred = get_predictor()
    if req.n_sims > 10_000:
        raise HTTPException(status_code=400, detail="Max 10,000 simulations per request.")
    try:
        result = pred.simulate_bracket(req.teams, req.host, req.n_sims, req.seed)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "champion":     result.iloc[0]["team"],
        "win_prob":     result.iloc[0]["win_prob"],
        "n_sims":       req.n_sims,
        "results":      result.to_dict(orient="records"),
    }


@app.post("/api/mispricings")
def find_mispricings(req: MispricingsRequest):
    pred  = get_predictor()
    items = [
        (m.home, m.away, m.stage, m.market_home, m.market_draw, m.market_away)
        for m in req.matchups
    ]
    df = pred.find_mispricings(items, threshold=req.threshold)
    return {
        "count":       len(df),
        "threshold":   req.threshold,
        "mispricings": df.to_dict(orient="records"),
    }


# ── HTML dashboard ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard():
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text())
    return HTMLResponse(content="<h2>Static files not found. Run from project root.</h2>")
