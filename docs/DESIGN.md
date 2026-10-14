# NFL Prediction Platform — Design & Decision Record

The complete, end-to-end record of **what** was built, **why**, **what alternatives** were considered,
and **what comes next**, step by step. Per-phase deep dives live in `docs/phase*.md`; the research
basis is in [`research.md`](research.md); the latest results post-mortem is in
[`week5_postmortem.md`](week5_postmortem.md).

---

## 0. Goals & constraints
- **Goal:** a portfolio-grade NFL win/spread prediction platform demonstrating modern Data Engineering
  and MLOps, judged by senior/staff-level craft — not by beating Vegas.
- **Constraints:** ≤ $5/month cloud (Azure for Students); must also run $0 locally; corporate Windows
  machine with no Java and TLS interception.
- **Design thesis:** NFL data is *wide but small* (~550k plays, ~250 MB). The hard problem is
  **temporal correctness** (no leakage) and **honest evaluation against the market**, not scale.

---

## 1. Infrastructure & cost (Phase 1)
- **Decision:** Terraform-provisioned ADLS Gen2 (hierarchical namespace) + a single-node
  `Standard_DS3_v2` Databricks cluster with 20-min auto-termination; service principal with
  `Storage Blob Data Contributor`; direct `abfss://` + OAuth via a secret scope (no mounts).
- **Why:** HNS enables real Delta performance; single-node removes worker DBUs (the main cost lever);
  auto-terminate stops idle burn; SP + secret scope is least-privilege and rotatable; direct ABFSS is
  the modern, Unity-Catalog-friendly pattern. ~$4/mo, storage a rounding error.
- **Alternatives:** plain Blob (no HNS — worse Delta); multi-node cluster (unnecessary at 250 MB);
  account keys (god-mode, un-rotatable); `dbutils.fs.mount` (legacy/global); Portal/CLI click-ops
  (not reproducible). Terraform chosen for the IaC story.
- **Next steps:** Databricks Asset Bundle for the job (done, Phase 6); OIDC federation instead of a
  PAT in CI; a `$5` budget alert; Unity Catalog if the project grows.

## 2. Dual-engine local-first architecture (cross-cutting)
- **Decision:** every transform exists **twice** — a local `polars` + `delta-rs` module in `src/nfl/`
  and a PySpark notebook in `databricks/notebooks/` — from one logic spec.
- **Why:** the machine has no Java (PySpark-on-Windows is painful), so local dev must be Java-free;
  polars/delta-rs writes real Delta tables and runs the full pipeline for $0. The PySpark notebooks
  prove cloud parity. Build/test free, pay only to demonstrate.
- **Alternatives:** PySpark-only (needs JDK + winutils, can't dev locally cheaply); local Spark in
  Docker (heavy); DuckDB (fine, but polars matches the pandas/Spark mental model better).
- **Next steps:** the Databricks notebooks are now **behind** the local modules (missing Tier-2 QB /
  opponent-adjusted / injury features) — **re-sync them** so parity is exact.

## 3. Bronze — raw ingestion (Phase 2)
- **Decision:** `nflreadpy` → polars → pandas (Arrow) → Delta, append-only, with lineage columns
  (`_ingest_ts`, `_source_file`, `_batch_id`, `_source_rows`); season-partitioned; 7 datasets
  (pbp, schedules, player_stats, rosters, injuries, ngs_*).
- **Why:** Bronze is an immutable, auditable landing zone — no cleaning, so history is always
  replayable. Lineage columns make every row traceable. Arrow keeps the polars→Spark hop type-faithful.
- **Alternatives:** write straight to Silver (loses raw audit trail); CSV/Parquet without Delta (no
  ACID/time-travel); overwrite-only (loses history — we support both via `write_mode`).
- **Gotcha handled:** nflverse all-null columns arrive as polars `Null` dtype that Arrow/Delta can't
  materialize → cast to string (plumbing only).
- **Next steps:** incremental/CDC ingestion (only new weeks) instead of full backfill; schema-registry
  checks on ingest.

## 4. Silver — clean / validate / conform (Phase 3)
- **Decision:** dedupe; drop special-teams / garbage-time / null-possession plays; impute dome weather;
  flag neutral-site/HFA; **conform team identity across relocations** (SD→LAC, STL→LAR, OAK→LV);
  a Great-Expectations-style DQ gate that **hard-fails** before any write; Z-ORDER on access keys.
- **Why:** this is where data becomes analysis-ready and trustworthy. The DQ gate caught a real
  franchise-relocation join bug during development (unique-key check). Team conforming is essential —
  otherwise home/away joins break and Elo splits a franchise in two.
- **Alternatives:** Great Expectations proper (heavier dependency; our lightweight suite is enough at
  this scale); keep garbage time (distorts team EPA); hard season reset for weather (dome nulls are
  expected, not missing — impute instead).
- **Next steps:** add row-count drift / freshness expectations; quarantine failing rows instead of
  hard-fail for non-critical checks; Liquid Clustering on Databricks instead of Z-ORDER.

## 5. Gold — leakage-safe features (Phase 4 + Tier 2)
- **Decision:** one row per game, home-vs-away deltas, targets on played games. Features: rolling 4/8-wk
  **EMA of off/def EPA** (pass+rush split), **CPOE**, **turnovers**; **Elo** (K=20, HFA=55, offseason
  regression); **rest days**; **haversine travel**; **weekly injury index** (position×status) + QB-out;
  **continuous QB value** (starter rolling passing-EPA); **opponent-adjusted EPA**.
- **Why / leakage controls:** every rolling feature is `shift(1)` per team (never sees its own game);
  Elo is recorded **pre-update** iterating in date order; injury reports are pre-kickoff. Betting lines
  are legitimate pre-game features; scores are targets only. Opponent-adjustment (subtract opponent
  pre-game strength, then roll) fixes schedule confounding behind the previously-"flat" EPA features.
- **Alternatives:** full ridge/mixed-effects opponent adjustment (heavier; research shows only ~0.5%
  gain and defensive adjustment can hurt — chose the one-pass version); QB Elo (richer but complex vs
  our rolling passing-EPA proxy); drive-level / tracking features (Big Data Bowl — out of scope).
- **Known limitation (important):** rolling features are keyed to a game's own play-by-play, so for
  **unplayed future games they are null** — live forward predictions currently use only
  spread+Elo+injuries. **Next step (high priority):** carry each team's most-recent rolling form
  forward to its next scheduled game. See [`week5_postmortem.md`](week5_postmortem.md).
- **Other next steps:** success rate, early-down EPA, explosive-play rate, pass-rate-over-expected,
  pace; empirical-Bayes shrinkage for weeks 1–4; state-space/Kalman team strength.

## 6. MLOps — training & tracking (Phase 5)
- **Decision:** LightGBM + XGBoost on `home_win`, `home_covered`, `target_margin`; **time-based
  walk-forward CV** (train [N-4,N-1], test N); MLflow logs params/metrics/importance/artifacts; best
  model aliased `champion`; **Vegas logged as a baseline**.
- **Why:** GBDTs dominate wide/small tabular data (no GPU, fast, free importances). Walk-forward is the
  only valid split — random k-fold trains on the future. Benchmarking vs the market keeps claims honest.
- **Alternatives:** random k-fold (leaks); deep learning (loses on this data profile); logistic
  regression (good baseline, less expressive); separate per-target tuning (deferred).
- **Result:** win AUC 0.669 — **worse than the market baseline (0.724)** it ingests. This diagnosis
  drove Phase 5.5.
- **Next steps:** nested CV for hyperparameters; expanding window; Optuna search logged as child runs.

## 7. Market-anchored distributional model (Phase 5.5 — the biggest lever)
- **Decision:** predict the **market residual** (`result − spread_line`), shrink toward the line
  (SHRINK=0.5), treat margin ~ Normal(μ, σ≈12.8), and derive **both** win `Φ(μ/σ)` and cover
  `Φ((μ−spread)/σ)` analytically; isotonic calibration; report Brier/log-loss + reliability.
- **Why:** the baseline was *diluting* an efficient market signal. Anchoring to the line and only
  deviating on signal is how nfelo/538 operate. Lifted win **AUC 0.669 → 0.714**, Brier **0.230 →
  0.214** — essentially matching Vegas (0.726 / 0.211).
- **Alternatives:** predict raw outcome (what failed); separate classifiers for win/cover (incoherent;
  distributional model unifies them); NGBoost/quantile for a learned σ (next step); fixed vs learned σ.
- **Honest findings:** still does **not** beat the pure spread→Normal baseline (the ceiling); isotonic
  calibration on a single ~250-game season slightly **hurt** (small-sample overfit).
- **Next steps:** learned/heteroscedastic σ; pool multiple seasons for calibration (or Platt);
  confidence-weighted SHRINK (deviate more when Elo/EPA strongly disagree — see post-mortem).

## 8. Honest betting evaluation (Phase 5.6)
- **Decision:** back-test as if betting — moneyline (bet when model edge over de-vigged market > 3%)
  and ATS (vs −110, drop pushes); report ROI and hit rate.
- **Why:** accuracy ≠ profit. The real test is money.
- **Result:** ML ROI **−2.2%**, ATS **50.1%** (< 52.38% break-even) → **no sustainable edge**, as
  expected for a public-data model. Reporting this (not a cherry-picked season) is the point.
- **Alternatives / next steps:** Kelly staking; Closing-Line-Value analysis (needs opening lines — not
  in current data); bet only highest-confidence deviations; soft/early-week lines vs closing.

## 9. CI/CD & packaging (Phase 6)
- **Decision:** GitHub Actions (ruff + pytest on every push; gated bundle deploy on main); Databricks
  Asset Bundle defining the bronze→silver→gold→train job; pytest suite on the correctness invariants
  (DQ, team conforming, EMA-shift leakage, odds math, haversine).
- **Why:** tests target the invariants that silently corrupt results if broken (a leaky window, a
  mis-mapped team). IaC-for-the-job + CI is the senior/staff signal.
- **Next steps:** OIDC auth in CI; coverage gate; schedule the Databricks job for weekly inference;
  pre-commit hooks.

---

## 10. Consolidated roadmap (prioritized)
1. **Forward-feature carry-forward** — make live predictions use full features (correctness). *High.*
2. **Re-sync Databricks notebooks** with Tier-2 features — restore exact dual-engine parity. *High.*
3. **Calibration fix** — pool seasons / Platt; add learned σ. *Medium.*
4. **Confidence-weighted deviation from the line** — deviate more when power ratings strongly disagree
   with the spread (the one catchable Week-5 miss pattern). *Medium, validate carefully.*
5. **Feature depth** — success rate, early-down EPA, PROE, pace; empirical-Bayes early-season shrinkage.
   *Medium.*
6. **Validation** — expanding window + nested CV + Optuna. *Medium.*
7. **Live Azure deploy** — Terraform apply + `databricks bundle run` for real cloud parity. *When ready
   to spend ~$4/mo.*
8. **CLV analysis** — ingest opening lines; measure closing-line value. *Low (data dependency).*

## 11. The one-paragraph summary
We built a full medallion lakehouse (local-first, cloud-parity) with leakage-safe features and an
MLOps loop, then used research + feature-importance diagnosis to discover the model was *worse than the
market it ingested*, re-framed it to anchor to the line (matching Vegas), added principled features
(injuries, QB value, opponent-adjusted EPA) with honest diminishing-returns findings, and proved with a
betting backtest that it matches but cannot beat the closing line. The judgment to recognize and report
the ceiling — rather than fake an edge — is the project's real deliverable.
