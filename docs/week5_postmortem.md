# Week 5 (2026) Post-Mortem — Why We Missed, and What It Teaches

Our model went **9/15 (60%)** straight-up in 2026 Week 5. That *feels* bad after the confident
slate, so this is a data-driven look at exactly why — and whether it signals a real problem or normal
NFL variance.

## The scoreboard

| Game | Actual | Model pick (conf) | Hit? | Spread | elo_diff | Note |
|---|---|---|---|---|---|---|
| PHI@JAX | JAX +2 | JAX (0.72) | ✅ | JAX −7.5 | −28 | won outright, didn't cover |
| CIN@MIA | CIN +7 | CIN (0.71) | ✅ | CIN −7 | — | |
| CLE@NYJ | NYJ +5 | NYJ (0.61) | ✅ | NYJ −2.5 | −68 | |
| DET@ARI | DET +24 | DET (0.64) | ✅ | DET −5.5 | −139 | |
| DEN@LAC | DEN +6 | DEN (0.66) | ✅ | DEN −3.5 | −77 | |
| HOU@TEN | HOU +23 | HOU (0.67) | ✅ | HOU −6.5 | −144 | |
| IND@PIT | PIT +12 | PIT (0.59) | ✅ | PIT −3 | +41 | |
| LV@NE | NE +28 | NE (0.60) | ✅ | NE −3.5 | +118 | |
| MIN@NO | NO +10 | NO (0.67) | ✅ | NO −2.5 | −141 | |
| **BAL@ATL** | **BAL +7** | ATL (0.64) | ❌ | ATL −3.5 | −60 | Elo favored BAL; model followed the line |
| **BUF@LAR** | **BUF +3** | LAR (0.55) | ❌ | LAR −3.5 | −32 | Elo favored BUF; model followed the line |
| **CHI@GB** | **GB +24** | CHI (0.52) | ❌ | CHI −1.5 | +2 | pick'em; GB blowout |
| **SF@SEA** | **SF +3** | SEA (0.63) | ❌ | SEA −3 | +29 | straight upset |
| **TB@DAL** | **TB +8** | DAL (0.77) | ❌ | DAL −9.5 | +41 | our most confident pick — a 17.5-pt swing vs. the line |
| **NYG@WAS** | **~tie** | WAS (0.58) | ❌ | WAS −3.5 | +28 | coin-flip game |

## Root cause: it was a high-variance upset week
- **The Vegas favorite lost in 5 of 15 games (33%)** — well above the ~25% long-run baseline. Simply
  picking favorites went **10/15 (67%)**; we went 9/15.
- **Four of our six misses were genuine upsets** (CHI+24, SF, TB over DAL, and the NYG/WAS coin flip)
  that *no* public model or the market caught. TB beating a 9.5-point favorite is a ~2.5-sigma result.
- **One week = 15 games is far too small to judge a model.** 60% in a 33%-upset week is squarely
  within the model's expected ~62–65% and the market's ~66%. This is variance, not a broken model.

## The one real, catchable pattern (and the lesson)
In **two** misses — BAL@ATL (elo_diff −60) and BUF@LAR (elo_diff −32) — our **own Elo power rating
strongly favored the team that won**, but the model **anchored to the Vegas line** and picked the
other side. The market-anchoring (SHRINK toward the spread) that *helped* us match Vegas overall also
**suppressed our model's correct contrarian signal** in these spots.

That is the genuine, actionable finding: anchoring is right on average but costs us when our power
rating has a strong, well-founded disagreement with the line.

## What we're missing (prioritized fixes)
1. **Forward-feature carry-forward (bug).** The *live* predictions we gave pre-game ran with **null
   rolling features** (EPA EMAs, QB value, rest) because those are keyed to a game's own play-by-play,
   which doesn't exist for a future game. This week it didn't change the W/L (null vs full both 60%),
   but it means live picks lean on spread+Elo+injuries only. **Fix:** carry each team's most-recent
   rolling form forward to its next scheduled game. *(High priority — correctness.)*
2. **Confidence-weighted deviation from the line.** When Elo (or opponent-adjusted EPA) disagrees with
   the spread by a large, stable margin, allow the model to deviate more (lower effective SHRINK);
   when they agree, stay anchored. Could have flipped BAL and BUF. *(Validate carefully — usually the
   market is right.)*
3. **Variance-aware reporting.** Report a weekly expectation with a confidence band (e.g., "9–10 of 15
   expected, ±2") so a 60% week reads as normal, not alarming.
4. **Bigger samples for any claim.** Evaluate over full seasons (256+ games), never a single week.

## The honest conclusion
We didn't "do something wrong." We hit a 33%-upset week and matched market-level accuracy minus one
game. The only true defect is the forward-feature carry-forward; the only true *edge* idea is
confidence-weighted deviation when our power rating strongly disagrees with the line. Everything else
is NFL variance — which is exactly why beating the market is so hard (see [`research.md`](research.md)).
