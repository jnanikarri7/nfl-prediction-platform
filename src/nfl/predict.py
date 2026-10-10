"""Load the registered champion model and run inference.

Demonstrates the serving side of the MLOps loop: pull a model from the MLflow
registry by alias (not a file path), score it, and produce predictions for
upcoming (unplayed) games. Also back-tests on the latest completed season as a
smoke test that the registered artifact loads and performs as logged.
"""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import polars as pl
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .train import _mlflow_uri, feature_cols, load_modeling_df

CHAMPION_URI = "models:/nfl_home_win@champion"


def _load_champion():
    mlflow.set_tracking_uri(_mlflow_uri())
    model = mlflow.sklearn.load_model(CHAMPION_URI)
    return model


def smoke_test(model, df: pl.DataFrame, feats: list[str]) -> None:
    """IN-SAMPLE smoke test: confirms the registered artifact loads and scores.
    The champion is trained on all completed games, so these numbers are optimistic
    (not a fair estimate). The honest out-of-sample performance is the Phase-5
    walk-forward result (~0.67 AUC). Use `scripts/run_train.py` for OOS metrics.
    """
    last = int(df.filter(pl.col("home_win").is_not_null())["season"].max())
    test = df.filter((pl.col("season") == last) & pl.col("home_win").is_not_null())
    p = model.predict_proba(test.select(feats).to_pandas())[:, 1]
    y = test["home_win"].to_numpy()
    print(f"\n=== Smoke test on {last} ({test.height} games) — IN-SAMPLE, optimistic ===")
    print(f"  Accuracy : {accuracy_score(y, (p >= 0.5).astype(int)):.3f}")
    print(f"  ROC-AUC  : {roc_auc_score(y, p):.4f}")
    print(f"  Log-Loss : {log_loss(y, p):.4f}")
    print("  (honest out-of-sample = walk-forward ~0.67 AUC; see Phase 5)")


def predict_upcoming(model, df: pl.DataFrame, feats: list[str], limit: int = 12) -> pl.DataFrame:
    upcoming = df.filter(~pl.col("is_played")).sort("season", "week")
    if upcoming.height == 0:
        print("\nNo unplayed games in the feature table.")
        return upcoming
    p = model.predict_proba(upcoming.select(feats).to_pandas())[:, 1]
    out = upcoming.select("game_id", "season", "week", "away_team", "home_team",
                          "spread_line").with_columns(
        pl.Series("home_win_prob", p).round(3),
        pl.Series("pick", ["HOME" if x >= 0.5 else "AWAY" for x in p]),
    )
    print(f"\n=== Upcoming game predictions (first {min(limit, out.height)} of {out.height}) ===")
    pl.Config.set_tbl_rows(limit)
    print(out.head(limit))
    return out


def run() -> int:
    model = _load_champion()
    print(f"Loaded champion model: {CHAMPION_URI}  ({type(model).__name__})")
    df = load_modeling_df()
    feats = feature_cols(df)
    smoke_test(model, df, feats)
    predict_upcoming(model, df, feats)
    print("\n[DONE] Inference OK.")
    return 0
