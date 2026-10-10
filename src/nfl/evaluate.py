"""Phase 5.6 — honest betting evaluation (does the model actually make money?).

Prediction accuracy is not profit. This module back-tests the market-anchored model as if
it were betting:
  * Moneyline: bet the side where the model's win prob beats the de-vigged market implied prob
    by a threshold; settle at the posted American odds; report ROI.
  * Against the spread: bet the side the model's margin favors vs. the line; settle at -110;
    compare the hit rate to the 52.38% break-even.

The expected, honest result is near-zero/negative edge against the closing line — that is the
correct finding for a public-data model, and reporting it is the point.
"""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import numpy as np
import polars as pl

from .market_model import (
    FIRST_TEST_SEASON,
    MIN_SIGMA,
    SHRINK,
    _folds,
    _make_regressor,
    _normal_win_prob,
)
from .train import _american_to_prob, _mlflow_uri, feature_cols, load_modeling_df

ML_EDGE_THRESHOLD = 0.03   # only bet moneyline when model edge over the market exceeds this
ATS_PAYOUT = 100 / 110     # profit on a won -110 bet


def oos_predictions(name: str = "lightgbm") -> pl.DataFrame:
    """Walk-forward out-of-sample predictions for every test game."""
    df = load_modeling_df().filter(
        pl.col("spread_line").is_not_null()
        & pl.col("result").is_not_null()
        & (pl.col("result") != 0)
    )
    feats = feature_cols(df)
    last = int(df["season"].max())
    parts = []
    for _season, tr, _cal, te in _folds(df, last):
        sigma = max(float(np.std((tr["result"] - tr["spread_line"]).to_numpy())), MIN_SIGMA)
        model = _make_regressor(name)
        model.fit(tr.select(feats).to_pandas(), (tr["result"] - tr["spread_line"]).to_numpy())
        mu, p = _residual_predict_local(model, te, feats, sigma)
        parts.append(te.select("game_id", "season", "home_win", "result", "spread_line",
                               "home_moneyline", "away_moneyline").with_columns(
            pl.Series("p_home", p), pl.Series("mu_margin", mu)))
    return pl.concat(parts)


def _residual_predict_local(model, part, feats, sigma):
    mu_r = model.predict(part.select(feats).to_pandas()) * SHRINK
    mu = part["spread_line"].to_numpy() + mu_r
    return mu, _normal_win_prob(mu, sigma)


def moneyline_backtest(preds: pl.DataFrame) -> dict:
    d = preds.filter(pl.col("home_moneyline").is_not_null() & pl.col("away_moneyline").is_not_null())
    bets = profit = wins = 0
    for r in d.iter_rows(named=True):
        ih = _american_to_prob(r["home_moneyline"])
        ia = _american_to_prob(r["away_moneyline"])
        if ih is None or ia is None:
            continue
        ih, ia = ih / (ih + ia), ia / (ih + ia)         # de-vig
        edge_home = r["p_home"] - ih
        edge_away = (1 - r["p_home"]) - ia
        if max(edge_home, edge_away) < ML_EDGE_THRESHOLD:
            continue
        bet_home = edge_home >= edge_away
        ml = r["home_moneyline"] if bet_home else r["away_moneyline"]
        won = (r["home_win"] == 1) if bet_home else (r["home_win"] == 0)
        bets += 1
        wins += int(won)
        profit += (ml / 100 if ml > 0 else 100 / -ml) if won else -1
    return {"ml_bets": bets, "ml_win_rate": (wins / bets) if bets else 0.0,
            "ml_roi": (profit / bets) if bets else 0.0, "ml_units": profit}


def ats_backtest(preds: pl.DataFrame) -> dict:
    # Model picks home ATS when its predicted margin beats the line; settle at -110, drop pushes.
    d = preds.filter(pl.col("result") != pl.col("spread_line"))
    pick_home = d["mu_margin"].to_numpy() > d["spread_line"].to_numpy()
    home_covered = (d["result"].to_numpy() > d["spread_line"].to_numpy())
    won = pick_home == home_covered
    n = len(won)
    hit = won.mean() if n else 0.0
    roi = (won * ATS_PAYOUT - (~won) * 1.0).mean() if n else 0.0
    return {"ats_bets": int(n), "ats_hit_rate": float(hit), "ats_roi": float(roi),
            "ats_breakeven": 0.5238}


def run() -> int:
    mlflow.set_tracking_uri(_mlflow_uri())
    mlflow.set_experiment("nfl_market_model")
    preds = oos_predictions("lightgbm")
    ml = moneyline_backtest(preds)
    ats = ats_backtest(preds)
    with mlflow.start_run(run_name="betting_backtest"):
        mlflow.log_metrics({**ml, **ats})
    print(f"=== Betting backtest (test {FIRST_TEST_SEASON}+, {preds.height} games) ===")
    print(f"  MONEYLINE (edge>{ML_EDGE_THRESHOLD:.0%}): {ml['ml_bets']} bets, "
          f"win {ml['ml_win_rate']:.1%}, ROI {ml['ml_roi']:+.1%}, {ml['ml_units']:+.1f} units")
    print(f"  ATS: {ats['ats_bets']} bets, hit {ats['ats_hit_rate']:.1%} "
          f"(break-even {ats['ats_breakeven']:.2%}), ROI {ats['ats_roi']:+.1%}")
    verdict = ("profitable" if ml["ml_roi"] > 0 and ats["ats_hit_rate"] > 0.5238
               else "no sustainable edge vs. the market (expected)")
    print(f"  Verdict: {verdict}.")
    return 0
