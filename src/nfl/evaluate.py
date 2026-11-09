"""Betting backtest for the market-anchored model: does the accuracy turn into money?

Moneyline: bet whichever side's model prob beats the de-vigged implied prob by
ML_EDGE_THRESHOLD, settle at the posted price. ATS: take the side the predicted margin
favours, settle at -110, compare against break-even.

Expect roughly break-even-minus-vig against closing lines. That's the honest answer for
a public-data model; a big positive ROI here would mean a bug.
"""
from __future__ import annotations

import mlflow
import polars as pl

from .market_model import (
    FIRST_TEST_SEASON,
    fit_residual_model,
    folds,
    load_market_df,
    residual_predict,
    residual_sigma,
)
from .train import _american_to_prob, _mlflow_uri, feature_cols

# 3 percentage points. Arbitrary, and ROI moves with it -- always quote the two together.
ML_EDGE_THRESHOLD = 0.03
ATS_PAYOUT = 100 / 110       # profit per unit on a winning -110 bet
ATS_BREAKEVEN = 110 / 210    # 52.38%


def oos_predictions(name: str = "lightgbm") -> pl.DataFrame:
    """Walk-forward out-of-sample predictions for every test game.

    Same folds as market_model, but p_home here is the raw Normal probability -- the
    isotonic step is skipped, so moneyline edges are measured before calibration.
    """
    df = load_market_df()
    feats = feature_cols(df)
    parts = []
    for _season, tr, _cal, te in folds(df, int(df["season"].max())):
        model = fit_residual_model(name, tr, feats)
        mu, p = residual_predict(model, te, feats, residual_sigma(tr))
        parts.append(te.select("game_id", "season", "home_win", "result", "spread_line",
                               "home_moneyline", "away_moneyline").with_columns(
            pl.Series("p_home", p), pl.Series("mu_margin", mu)))
    return pl.concat(parts)


def moneyline_backtest(preds: pl.DataFrame) -> dict:
    d = preds.filter(pl.col("home_moneyline").is_not_null() & pl.col("away_moneyline").is_not_null())
    bets = profit = wins = 0
    for r in d.iter_rows(named=True):
        ih = _american_to_prob(r["home_moneyline"])
        ia = _american_to_prob(r["away_moneyline"])
        if ih is None or ia is None:
            continue
        ih, ia = ih / (ih + ia), ia / (ih + ia)  # de-vig
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
    # Pushes refund the stake, so they're excluded rather than counted as losses.
    d = preds.filter(pl.col("result") != pl.col("spread_line"))
    pick_home = d["mu_margin"].to_numpy() > d["spread_line"].to_numpy()
    home_covered = (d["result"].to_numpy() > d["spread_line"].to_numpy())
    won = pick_home == home_covered
    n = len(won)
    hit = won.mean() if n else 0.0
    roi = (won * ATS_PAYOUT - (~won) * 1.0).mean() if n else 0.0
    return {"ats_bets": int(n), "ats_hit_rate": float(hit), "ats_roi": float(roi),
            "ats_breakeven": ATS_BREAKEVEN}


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
    verdict = ("profitable" if ml["ml_roi"] > 0 and ats["ats_hit_rate"] > ATS_BREAKEVEN
               else "no sustainable edge vs. the market (expected)")
    print(f"  Verdict: {verdict}.")
    return 0
