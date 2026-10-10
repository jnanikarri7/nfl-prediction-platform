# Research — How to Improve the Model (and the honest ceiling)

Two web research passes (top open-source projects + SOTA techniques), cross-referenced
against this model's own feature importances. Primary sources listed at the bottom.

## The key diagnosis
The ML model (win AUC 0.67, Brier 0.23, margin MAE 10.4) is **worse than the Vegas line it
already ingests** (AUC 0.724, Brier ~0.19). A trivial baseline — closing spread → Normal CDF,
`p_home = Φ(−spread_line / σ)`, σ≈13.0–13.5 — reaches **AUC ≈ 0.72 / Brier ≈ 0.19 with zero ML**.

The model's own importances confirm it: **`spread_line` alone = 14.3%** of importance; the ~27
team-form features are all **flat at ~3%** (correlated, EPA-derived, not opponent-adjusted). The
GBM is **diluting an efficient market signal**, not improving on it.

Two hard limits:
- **Margin MAE is saturated.** Irreducible floor ≈ `σ·√(2/π) ≈ 10.4–10.8` for σ≈13–13.5
  (Stern 1991). Our 10.4 is already there — **stop optimizing MAE.**
- **ATS ≈ 0.50 AUC is expected by construction.** Break-even at −110 is 52.38%; no public model
  reliably beats the closing line. ~0.50 is correct, not a bug.

## Prioritized plan
**Tier 1 — one refactor, most of the gain (target 0.67→~0.72 AUC, Brier 0.23→~0.20):**
1. **Model the market residual** (`actual_margin − spread_line`) or shrink toward the line — anchor
   to Vegas, deviate only on signal (nfelo/538's core move).
2. **Distributional margin model** — predict μ, assume margin~Normal(μ,σ), derive **both** win
   `Φ(μ/σ)` and cover `Φ((μ−spread)/σ)` from one coherent model.
3. **Calibration layer** (isotonic) + optimize/report **log-loss & Brier**, not AUC.
4. **Monotonic constraints / strong regularization / early stopping** to stop market-fighting
   overfits. (In the residual framing the anchoring itself limits this; we lean on regularization +
   shrinkage and document the choice.)

**Tier 2 — feature quality (modest, validate each):** opponent-adjusted EPA (ridge/mixed-effects);
continuous **QB value** (vs. the binary QB-out flag); efficiency splits (success rate, early-down
EPA, explosive rate, PROE, pace); empirical-Bayes shrinkage for weeks 1–4.

**Tier 3 — showcase / ATS polish:** state-space/Kalman team strength (uncertainty → σ);
line-movement (open→current, *causal* only) + Closing-Line-Value eval; expanding-window + nested CV;
final stack with the market.

## Tier 1 — implemented (`src/nfl/market_model.py`, Phase 5.5)
Market-residual target + Normal(μ,σ) → analytic win/cover probs + isotonic calibration, walk-forward
with an internal calibration season (train `[N-1-W, N-2]`, calibrate `N-1`, test `N`).

| Model | Win AUC | Brier | Acc |
|---|---|---|---|
| Baseline GBM (Phase 5) | 0.669 | ~0.230 | 61.9% |
| **Market-anchored (raw)** | **0.714** | **0.214** | 65.6% |
| Pure market baseline (spread→Normal, σ=12.8) | 0.726 | 0.211 | 66.5% |

**Outcome:** anchoring closed ~75% of the AUC gap (0.669→0.714) and cut Brier 0.23→0.214 — the model
now essentially *matches* the market. Two honest findings: (a) it still does **not beat** the pure
spread→Normal baseline, confirming the market ceiling; (b) **isotonic calibration slightly hurt**
(cal Brier 0.217 > raw 0.214) because a single ~250-game calibration season overfits — the raw
Normal probabilities are already well-calibrated. Next: pool more calibration data or use Platt.

## Realistic ceiling (the portfolio win condition)
Match the market (~0.72 AUC, ~0.19 Brier) and be **well-calibrated**, with an honest CLV writeup —
not "beat Vegas." NFL outcomes are dominated by game-to-game randomness and the market aggregates
nearly all public signal (Lopez et al. 2018; Stern 1991).

## Top open-source references
| Project | Approach | Metric |
|---|---|---|
| nfelo (greerreNFL) | Elo + QB value + **market-blended prior** | ~51–53% ATS good seasons |
| nflmodel (morelandjs) | MELO (margin-distributional Elo) | spread MAE 10.38, calibrated |
| nfl-analytics (BlairCurrey) | rolling feats → linear, honest Vegas baseline | MAE ~10.3; *"real signal, no edge"* |
| 538 NFL Elo | MOV multiplier + **QB-adjusted Elo** + rest bonuses | 63–67% straight-up |

## Sources
- Opponent-adjusted EPA — https://www.opensourcefootball.com/posts/2020-08-20-adjusting-epa-for-strenght-of-opponent/
- Bayesian team ability (shrinkage, Student-t) — https://www.opensourcefootball.com/posts/2021-06-27-estimating-team-ability-from-epa/
- Rolling-EPA window/EMA weighting — https://www.opensourcefootball.com/posts/2020-12-29-exploring-rolling-averages-of-epa/
- nflfastR WP/EP/CP calibration (LOSO; log-loss 0.52→0.44) — https://www.opensourcefootball.com/posts/2020-09-28-nflfastr-ep-wp-and-cp-models/
- Stern 1991, Normal(spread, σ≈13.86) — https://www.jstor.org/stable/2684286
- Lopez, Matthews, Baumer 2018 (randomness ceiling) — https://arxiv.org/abs/1701.05976
- nfelo market-anchored model — https://www.nfeloapp.com/the-nfelo-model/
- 538 NFL Elo methodology (archived) — https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/
- LightGBM monotone constraints — https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#monotonic-constraints
- NGBoost distributional regression — https://stanfordmlgroup.github.io/projects/ngboost/
- Projects: nfelo https://github.com/greerreNFL/nfelo · nflmodel https://github.com/morelandjs/nflmodel · nfl-analytics https://github.com/BlairCurrey/nfl-analytics
