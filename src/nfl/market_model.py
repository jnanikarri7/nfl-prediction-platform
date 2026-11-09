"""Market-anchored margin model.

The baseline classifiers in train.py score *worse* than the spread they're given as a
feature -- they spend capacity re-learning the line. So instead:

  - regress the residual (result - spread_line) and shrink it toward zero, i.e. only
    express disagreement with the market;
  - treat margin ~ Normal(spread + residual, sigma) and read win probability off the
    CDF (Stern 1991) rather than training a separate classifier;
  - isotonic-calibrate on a held-out season before scoring the test season.

Folds, per test season N: train [N-1-W, N-2], calibrate N-1, test N.
See docs/research.md for the reasoning and numbers.
"""
from __future__ import annotations

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

from .train import TRAIN_WINDOW, _mlflow_uri, feature_cols, load_modeling_df, save_plot

# Fixed, not tuned. If you sweep it, score on the calibration season, never the test one.
SHRINK = 0.5
# Stern measured ~13.9 pts for margin-vs-spread; the floor only bites on tiny train slices.
MIN_SIGMA = 10.0
FIRST_TEST_SEASON = 2020  # TRAIN_WINDOW train seasons + 1 calibration season after 2015


def load_market_df() -> pl.DataFrame:
    """Played, non-tie games with a closing spread -- the only rows this model can use."""
    return load_modeling_df().filter(
        pl.col("spread_line").is_not_null()
        & pl.col("result").is_not_null()
        & (pl.col("result") != 0)
    )


def residual_sigma(df: pl.DataFrame) -> float:
    return max(float(np.std((df["result"] - df["spread_line"]).to_numpy())), MIN_SIGMA)


def _normal_win_prob(mu: np.ndarray, sigma: float) -> np.ndarray:
    return norm.cdf(mu / sigma)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p),
            "logloss": log_loss(y, p, labels=[0, 1]),
            "accuracy": accuracy_score(y, (p >= 0.5).astype(int))}


def _make_regressor(name: str):
    # Deliberately weak: shallow trees, heavy L2, big leaves. The target is mostly noise
    # around zero and anything expressive just fits that noise.
    if name == "lightgbm":
        return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=15,
                                 max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=2.0, min_child_samples=30, verbose=-1,
                                 random_state=42)
    return xgb.XGBRegressor(n_estimators=400, learning_rate=0.02, max_depth=3,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                            min_child_weight=10, tree_method="hist", random_state=42)


def fit_residual_model(name: str, df: pl.DataFrame, feats: list[str]):
    model = _make_regressor(name)
    model.fit(df.select(feats).to_pandas(), (df["result"] - df["spread_line"]).to_numpy())
    return model


def residual_predict(model, part: pl.DataFrame, feats: list[str], sigma: float):
    """Return (predicted home margin, raw home-win prob)."""
    mu = part["spread_line"].to_numpy() + model.predict(part.select(feats).to_pandas()) * SHRINK
    return mu, _normal_win_prob(mu, sigma)


def _reliability_plot(y, p_raw, p_cal, title: str) -> str:
    bins = np.linspace(0, 1, 11)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for p, lab in [(p_raw, "raw"), (p_cal, "calibrated")]:
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        # Skip bins with <10 games; a 2-game bin at 100% just adds noise to the plot.
        pts = [(p[idx == b].mean(), y[idx == b].mean())
               for b in range(10) if (idx == b).sum() >= 10]
        if pts:
            plt.plot(*zip(*pts, strict=True), "o-", label=lab)
    plt.xlabel("predicted P(home win)")
    plt.ylabel("observed frequency")
    plt.title(title)
    plt.legend()
    return save_plot(title)


def folds(df: pl.DataFrame, last_season: int):
    for n in range(FIRST_TEST_SEASON, last_season + 1):
        tr = df.filter((pl.col("season") >= n - 1 - TRAIN_WINDOW) & (pl.col("season") <= n - 2))
        cal = df.filter(pl.col("season") == n - 1)
        te = df.filter(pl.col("season") == n)
        if tr.height and cal.height and te.height:
            yield n, tr, cal, te


def _run_model(name: str, df: pl.DataFrame, feats: list[str], last_season: int) -> dict:
    oof_y, oof_raw, oof_cal = [], [], []
    with mlflow.start_run(run_name=f"market_{name}"):
        mlflow.log_params({"model": name, "framing": "market_residual_normal",
                           "shrink": SHRINK, "train_window": TRAIN_WINDOW})
        for season, tr, cal, te in folds(df, last_season):
            sigma = residual_sigma(tr)
            model = fit_residual_model(name, tr, feats)

            _, cal_raw = residual_predict(model, cal, feats, sigma)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(cal_raw, cal["home_win"].to_numpy())

            _, te_raw = residual_predict(model, te, feats, sigma)
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

        # TODO: the registered artifact is the bare regressor -- no sigma, no isotonic
        # map. Anyone serving it has to rebuild both. Wrap all three in a pyfunc.
        final = fit_residual_model(name, df.filter(pl.col("home_win").is_not_null()), feats)
        mlflow.sklearn.log_model(final, name="model", serialization_format="cloudpickle",
                                 registered_model_name="nfl_margin_market")
        print(f"  market_{name:9s} raw: AUC={m_raw['auc']:.4f} Brier={m_raw['brier']:.4f}  "
              f"cal: AUC={m_cal['auc']:.4f} Brier={m_cal['brier']:.4f} Acc={m_cal['accuracy']:.3f}")
        return m_cal


def _market_baseline(df: pl.DataFrame) -> dict:
    """Closing spread through the same Normal CDF, no ML. This is the bar."""
    test = df.filter(pl.col("season") >= FIRST_TEST_SEASON)
    sigma = residual_sigma(df.filter(pl.col("season") < FIRST_TEST_SEASON))
    y = test["home_win"].to_numpy()
    m = _metrics(y, _normal_win_prob(test["spread_line"].to_numpy(), sigma))
    with mlflow.start_run(run_name="market_baseline_normal"):
        mlflow.log_param("sigma", round(sigma, 3))
        mlflow.log_metrics(m)
    print(f"  MARKET (spread->Normal, sigma={sigma:.2f}) "
          f"AUC={m['auc']:.4f} Brier={m['brier']:.4f} Acc={m['accuracy']:.3f}")
    return m


def run() -> int:
    mlflow.set_tracking_uri(_mlflow_uri())
    mlflow.set_experiment("nfl_market_model")
    df = load_market_df()
    feats = feature_cols(df)
    last = int(df["season"].max())
    print(f"=== Market-anchored distributional model | test {FIRST_TEST_SEASON}-{last} ===")
    base = _market_baseline(df)
    results = {m: _run_model(m, df, feats, last) for m in ("lightgbm", "xgboost")}
    best = min(results, key=lambda m: results[m]["brier"])
    print(f"\n[DONE] Best calibrated: market_{best} "
          f"(Brier {results[best]['brier']:.4f} vs market {base['brier']:.4f}, "
          f"AUC {results[best]['auc']:.4f} vs market {base['auc']:.4f}).")
    return 0
