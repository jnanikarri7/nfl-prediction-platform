# nfl-prediction-platform

![CI](https://github.com/jnanikarri7/nfl-prediction-platform/actions/workflows/ci.yml/badge.svg?branch=main)

Predicts NFL game outcomes (winner, cover, margin) from nflverse data with a
Bronze/Silver/Gold Delta lakehouse and MLflow-tracked models. The data is small (~550k
plays, ~250 MB), so scale isn't the hard part. The hard part is temporal correctness:
no feature may see the game it's predicting, and every result is compared to the
Vegas line rather than to a coin flip.

It runs end-to-end on a laptop (polars + delta-rs, no Spark or Java). There's also an
Azure Databricks deployment that costs about $4/month on a single-node cluster.

## Results

Walk-forward out-of-sample, test seasons 2019+ (2020+ for the market model):

| Target | Metric | LightGBM | XGBoost | Market-anchored | Vegas |
|---|---|---|---|---|---|
| Home win | ROC-AUC | 0.666 | 0.669 | 0.714 | 0.724 |
| | Brier | — | 0.23 | 0.214 | 0.211 |
| | Accuracy | 61.8% | 61.9% | — | 66.1% |
| Cover | ROC-AUC | 0.497 | 0.497 | — | ~0.50 |
| Margin | MAE (pts) | 10.48 | 10.36 | — | 9.82 |

The Vegas column is the de-vigged moneyline for AUC and accuracy, and the spread for
MAE. The Brier figure is spread -> Normal(sigma), the same framing the market model
uses; on that basis Vegas AUC is 0.726.

- The plain GBMs lose to the spread they're given as a feature. The market-anchored
  model ([`market_model.py`](src/nfl/market_model.py)) regresses the residual against
  the line instead and closes most of that gap. It doesn't beat the market, and nothing
  built on public data realistically will.
- ATS at ~0.50 AUC is the expected result. A cover model that scores well above that is
  almost certainly leaking.
- The betting backtest (`run_evaluate.py`) shows no sustainable edge against these
  lines, which is also expected.

Write-ups: [`docs/research.md`](docs/research.md) (why the market-anchored framing),
[`docs/DESIGN.md`](docs/DESIGN.md) (decision log),
[`docs/week5_postmortem.md`](docs/week5_postmortem.md) (one week's misses, game by game).

## Running it locally

Python 3.10+. The first Bronze run downloads ~250 MB from nflverse.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/run_bronze.py --start 2024 --end 2024   # one season, smoke test
python scripts/run_bronze.py                           # full 2015-present backfill
python scripts/run_silver.py
python scripts/run_gold.py
python scripts/run_train.py          # baseline GBMs + Vegas baseline, sets `champion`
python scripts/run_market_model.py
python scripts/run_evaluate.py       # ROI / ATS backtest
python scripts/run_predict.py        # score unplayed games with the champion

mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Tests and lint (same as CI):

```bash
pip install ruff pytest
ruff check src scripts tests && pytest
```

| Env var | Default | Used for |
|---|---|---|
| `NFL_DATA_DIR` | `./data` | Root of the local Delta lake (`bronze/`, `silver/`, `gold/`) |
| `MLFLOW_TRACKING_URI` | `sqlite:///./mlflow.db` | Set to `databricks` to log to a workspace |

## Running it on Azure Databricks

```bash
az login
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # storage account name must be globally unique
terraform init && terraform apply              # ADLS Gen2, service principal, workspace
```

Create the `nfl-adls` secret scope (`sp-client-id`, `sp-client-secret`,
`sp-tenant-id`; see step 6 of `scripts/az_setup.sh`). Then replace `stnflpredictXX`
in `databricks.yml` and the notebooks with your storage account name and run:

```bash
databricks bundle deploy -t dev
databricks bundle run nfl_pipeline -t dev
```

Set a budget alert before you leave it running. Cost comes almost entirely from
cluster uptime, and the 20-minute auto-terminate is what keeps it near $4/month.
Sizing notes are in [`docs/phase1_architecture.md`](docs/phase1_architecture.md).

## Layout

```
src/nfl/              local pipeline (the reference implementation)
  bronze.py           nflreadpy -> Delta, plus lineage columns
  silver.py           dedupe, team-code conformance, play filtering, DQ gate
  gold.py             EMAs, Elo, travel, injuries, QB value -> one row per game
  train.py            baseline GBMs, walk-forward CV, registry + champion alias
  market_model.py     residual-vs-spread model with isotonic calibration
  evaluate.py         betting backtest
  quality.py          the DQ checks Silver/Gold gate on
scripts/              thin CLI wrappers, one per stage
databricks/notebooks/ PySpark versions of bronze/silver/gold/train
infra/terraform/      ADLS Gen2, service principal, Databricks workspace
databricks.yml        Asset Bundle: the 4-task job
```

## Design notes and trade-offs

**Two engines.** Every stage was written once in polars and once in PySpark. That
let the whole thing be built and tested for $0 and still prove it runs on Databricks.
The cost is that the two drift, and they already have (see below). If this
kept going I'd drop the hand-written notebooks and run `src/nfl` on the cluster
via polars-on-driver, which is fine at this data size.

**Leakage control is structural, not reviewed.** Rolling stats are always
`ewm_mean(...).shift(1).over(team)`, and Elo is snapshotted before the update. There's
a test for the shift pattern (`tests/test_features.py`). New features should reuse
those two patterns rather than invent a third.

**Walk-forward, 4-season window.** Train on N-4..N-1 and test on N. The market model
holds out N-1 for isotonic calibration, so it trains on N-5..N-2 and its first test
season is 2020.

**Homegrown DQ instead of Great Expectations.** Four checks (`unique`, `not_null`,
`in_range`, `in_set`) that raise before a write. They caught a real bug during
development: schedules use `SD`/`STL`/`OAK` for pre-relocation seasons while pbp uses
current codes, so home/away joins broke for those seasons. That's what
[`teams.py`](src/nfl/teams.py) fixes.

**Bronze is overwrite-by-default locally.** Append is the textbook Bronze contract,
but locally you re-run constantly. `--mode append` exists, and Silver dedupes on
natural keys either way.

## Known gaps

- **Upcoming games are scored on thin features.** EMAs and QB value come from
  play-by-play, which unplayed games don't have yet. At inference time the EMA diffs
  are null and both QBs get the replacement-level value, so predictions lean almost
  entirely on Elo and the line. The fix is to carry forward each team's latest
  post-game EMA (TODO in `gold.build_gold`).
- **The champion is still the baseline classifier.** The market model is better, but
  its registered artifact is the bare regressor. Sigma and the isotonic map aren't
  saved with it, so `run_predict.py` can't use it yet. That needs a pyfunc wrapper.
- **Databricks notebooks lag the local pipeline.** They have no injuries ingestion, no
  injury/QB-value/opponent-adjusted features, and no market model. The Databricks job
  reproduces the Phase 5 baseline only.
- **`databricks.yml` declares a `storage_account` variable but doesn't use it.** The
  account name is hardcoded in the spark_conf keys, so you have to edit it by hand.
- **The backtest uses nflverse lines, which are close to closing.** You usually
  can't get those prices early in the week, so read the ROI numbers as optimistic.
- **Travel distance assumes the home team doesn't travel.** That's wrong for
  international games, where only `home_field_advantage = 0` carries the signal.
- **`END_SEASON` is the calendar year.** In January/February it points at a season
  that hasn't started. Pass `--end` explicitly then.

Data: [nflverse](https://github.com/nflverse), via `nflreadpy`.
