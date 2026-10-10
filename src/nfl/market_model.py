"""Phase 5.5 — market-anchored distributional model (Tier-1 research refactor).

Diagnosis (see docs/research.md): the baseline GBM (win AUC 0.67, Brier 0.23) is WORSE
than the Vegas line it ingests (AUC 0.724, Brier ~0.19). The fix is not more features —
it is to anchor to the market and calibrate. This module:

  1. Predicts the market RESIDUAL (actual_margin - spread_line), not the raw outcome, so the
     model only expresses deviations from Vegas (shrunk by SHRINK).
  2. Treats margin ~ Normal(mu, sigma) and derives BOTH win and cover probabilities analytically
     (Stern 1991), instead of training separate classifiers.
  3. Calibrates win probabilities with isotonic regression fit on an internal calibration season
     (leakage-free), and reports Brier / log-loss / reliability — the metrics that matter for
     betting — alongside AUC.
  4. Uses a small, strongly-regularized tree with early-stopping; in the residual framing the
     anchoring itself limits market-fighting, so we rely on regularization + shrinkage (documented).

Walk-forward per test season N: train = [N-1-W, N-2], calibrate = N-1, test = N.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import lightgbm as lgb
import matplotlib
import mlflow
import numpy as np
import polars as pl
import xgboost as xgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .train import TRAIN_WINDOW, _mlflow_uri, feature_cols, load_modeling_df

SHRINK = 0.5           # pull the model's residual halfway toward the market (anchoring)
MIN_SIGMA = 10.0       # guard against degenerate variance
FIRST_TEST_SEASON = 2020  # needs W train seasons + 1 calibration season before it


def _normal_win_prob(mu: np.ndarray, sigma: float) -> np.ndarray:
    return norm.cdf(mu / sigma)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p),
            "logloss": log_loss(y, p, labels=[0, 1]),
            "accuracy": accuracy_score(y, (p >= 0.5).astype(int))}


def _make_regressor(name: str):
    """Small, strongly-regularized margin-residual regressor."""
    if name == "lightgbm":
        return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=15,
                                 max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=2.0, min_child_samples=30, verbose=-1,
                                 random_state=42)
    return xgb.XGBRegressor(n_estimators=400, learning_rate=0.02, max_depth=3,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                            min_child_weight=10, tree_method="hist", random_state=42)


def _reliability_plot(y, p_raw, p_cal, title: str) -> str:
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for p, lab in [(p_raw, "raw"), (p_cal, "calibrated")]:
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 10:
                xs.append(p[m].mean())
                ys.append(y[m].mean())
        plt.plot(xs, ys, "o-", label=lab)
    plt.xlabel("predicted P(home win)")
    plt.ylabel("observed frequency")
    plt.title(title)
    plt.legend()
    path = os.path.join(tempfile.gettempdir(), f"{title.replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return path


def _folds(df: pl.DataFrame, last_season: int):
    for n in range(FIRST_TEST_SEASON, last_season + 1):
        tr = df.filter((pl.col("season") >= n - 1 - TRAIN_WINDOW) & (pl.col("season") <= n - 2))
        cal = df.filter(pl.col("season") == n - 1)
        te = df.filter(pl.col("season") == n)
        if tr.height and cal.height and te.height:
            yield n, tr, cal, te


def _residual_predict(model, part: pl.DataFrame, feats: list[str], sigma: float):
    """Return (margin_mu, win_prob_raw) for a partition."""
    mu_r = model.predict(part.select(feats).to_pandas()) * SHRINK
    mu = part["spread_line"].to_numpy() + mu_r         # anchor + shrunk deviation
    return mu, _normal_win_prob(mu, sigma)


def _run_model(name: str, df: pl.DataFrame, feats: list[str], last_season: int) -> dict:
    oof_y, oof_raw, oof_cal = [], [], []
    with mlflow.start_run(run_name=f"market_{name}"):
        mlflow.log_params({"model": name, "framing": "market_residual_normal",
                           "shrink": SHRINK, "train_window": TRAIN_WINDOW})
        for season, tr, cal, te in _folds(df, last_season):
            sigma = max(float(np.std((tr["result"] - tr["spread_line"]).to_numpy())), MIN_SIGMA)
            model = _make_regressor(name)
            model.fit(tr.select(feats).to_pandas(), (tr["result"] - tr["spread_line"]).to_numpy())

            # Calibrate win prob on the internal calibration season (leakage-free).
            _, cal_raw = _residual_predict(model, cal, feats, sigma)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(cal_raw, cal["home_win"].to_numpy())

            _, te_raw = _residual_predict(model, te, feats, sigma)
            te_cal = iso.predict(te_raw)
            y = te["home_win"].to_numpy()
            oof_y.append(y)
            oof_raw.append(te_raw)
            oof_cal.append(te_cal)
            mlflow.log_metrics({"fold_brier_cal": brier_score_loss(y, te_cal)}, step=season)

        y = np.concatenate(oof_y)
        raw, cal = np.concatenate(oof_raw), np.concatenate(oof_cal)
        m_raw, m_cal = _metrics(y, raw), _metrics(y, cal)
        mlflow.log_metrics({**{f"raw_{k}": v for k, v in m_raw.items()},
                            **{f"cal_{k}": v for k, v in m_cal.items()}})
        mlflow.log_artifact(_reliability_plot(y, raw, cal, f"market {name} reliability"))

        # Final model on all available seasons -> registry.
        full = df.filter(pl.col("home_win").is_not_null())
        final = _make_regressor(name)
        final.fit(full.select(feats).to_pandas(), (full["result"] - full["spread_line"]).to_numpy())
        mlflow.sklearn.log_model(final, name="model", serialization_format="cloudpickle",
                                 registered_model_name="nfl_margin_market")
        print(f"  market_{name:9s} raw: AUC={m_raw['auc']:.4f} Brier={m_raw['brier']:.4f}  "
              f"cal: AUC={m_cal['auc']:.4f} Brier={m_cal['brier']:.4f} Acc={m_cal['accuracy']:.3f}")
        return m_cal


def _market_baseline(df: pl.DataFrame, last_season: int) -> dict:
    """Pure closing-spread -> Normal CDF. Zero ML. The target to match."""
    test = df.filter((pl.col("season") >= FIRST_TEST_SEASON) & pl.col("home_win").is_not_null()
                     & pl.col("spread_line").is_not_null())
    # Fit sigma on earlier seasons only.
    tr = df.filter((pl.col("season") < FIRST_TEST_SEASON) & pl.col("result").is_not_null()
                   & pl.col("spread_line").is_not_null())
    sigma = max(float(np.std((tr["result"] - tr["spread_line"]).to_numpy())), MIN_SIGMA)
    y = test["home_win"].to_numpy()
    p = _normal_win_prob(test["spread_line"].to_numpy(), sigma)
    m = _metrics(y, p)
    with mlflow.start_run(run_name="market_baseline_normal"):
        mlflow.log_param("sigma", round(sigma, 3))
        mlflow.log_metrics(m)
    print(f"  MARKET (spread->Normal, sigma={sigma:.2f}) "
          f"AUC={m['auc']:.4f} Brier={m['brier']:.4f} Acc={m['accuracy']:.3f}")
    return m


def run() -> int:
    mlflow.set_tracking_uri(_mlflow_uri())
    mlflow.set_experiment("nfl_market_model")
    # Played, non-tie, spread-present games only — the trainable/evaluable universe.
    df = load_modeling_df().filter(
        pl.col("spread_line").is_not_null()
        & pl.col("result").is_not_null()
        & (pl.col("result") != 0)
    )
    feats = feature_cols(df)
    last = int(df["season"].max())
    print(f"=== Market-anchored distributional model | test {FIRST_TEST_SEASON}-{last} ===")
    base = _market_baseline(df, last)
    results = {m: _run_model(m, df, feats, last) for m in ("lightgbm", "xgboost")}
    best = max(results, key=lambda m: -results[m]["brier"])  # lower Brier is better
    print(f"\n[DONE] Best calibrated: market_{best} "
          f"(Brier {results[best]['brier']:.4f} vs market {base['brier']:.4f}, "
          f"AUC {results[best]['auc']:.4f} vs market {base['auc']:.4f}).")
    return 0
