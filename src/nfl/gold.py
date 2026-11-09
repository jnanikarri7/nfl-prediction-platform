"""Gold: one row per scheduled game, home-minus-away feature deltas, targets on played games.

Every feature must be knowable before kickoff. Two places this can go wrong:
  - rolling stats: every EMA is shift(1)'d within the team's timeline, so a game never
    sees its own result;
  - Elo: the rating is snapshotted *before* the post-game update, iterating in date order.
Closing lines are fair game (published pre-kickoff); scores are targets only.
Unplayed games keep null targets so the same table feeds inference.
"""
from __future__ import annotations

import math

import polars as pl

from . import quality as dq
from .config import GOLD_DIR, SILVER_DIR
from .lake import write_table
from .stadiums import TEAM_COORDS

EMA_SPANS = (4, 8)
# Injury index = sum(position weight x status weight) per team-week. Weights are
# hand-tuned priors, not fitted. The final injury report comes out before kickoff,
# so same-week joins are safe.
POS_WEIGHT = {"QB": 4.0, "WR": 1.5, "RB": 1.2, "TE": 1.0, "T": 1.3, "LT": 1.3,
              "RT": 1.3, "G": 1.1, "C": 1.1, "OL": 1.2, "EDGE": 1.3, "DE": 1.2,
              "DT": 1.1, "CB": 1.3, "S": 1.1, "FS": 1.1, "SS": 1.1, "LB": 1.0}
STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.35}
# Conventional NFL Elo constants. Not tuned against this data -- a cheap thing to sweep.
ELO_K = 20.0
ELO_HFA = 55.0
ELO_BASE = 1500.0
ELO_REGRESS = 0.75
MAX_REST_DAYS = 14      # otherwise week 1 "rest" is ~200 days and dominates the feature


def _team_game_stats(pbp: pl.DataFrame) -> pl.DataFrame:
    off = pbp.group_by("game_id", "posteam").agg(
        pl.len().alias("off_plays"),
        pl.col("epa").mean().alias("off_epa"),
        pl.col("epa").filter(pl.col("pass") == 1).mean().alias("off_pass_epa"),
        pl.col("epa").filter(pl.col("rush") == 1).mean().alias("off_rush_epa"),
        pl.col("cpoe").mean().alias("off_cpoe"),
        (pl.col("interception").sum() + pl.col("fumble_lost").sum()).alias("giveaways"),
    ).rename({"posteam": "team"})

    deff = pbp.group_by("game_id", "defteam").agg(
        pl.col("epa").mean().alias("def_epa"),
        pl.col("epa").filter(pl.col("pass") == 1).mean().alias("def_pass_epa"),
        pl.col("epa").filter(pl.col("rush") == 1).mean().alias("def_rush_epa"),
        (pl.col("interception").sum() + pl.col("fumble_lost").sum()).alias("takeaways"),
    ).rename({"defteam": "team"})

    return off.join(deff, on=["game_id", "team"], how="inner")


EMA_BASE_COLS = ["off_epa", "def_epa", "off_pass_epa", "off_rush_epa",
                 "def_pass_epa", "def_rush_epa", "off_cpoe",
                 "giveaways", "takeaways"]


def _team_timeline(team_game: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
    ctx = sched.select("game_id", "season", "week", "gameday",
                        "home_team", "away_team")
    tg = (
        team_game.join(ctx, on="game_id", how="inner")
        .with_columns(
            pl.col("gameday").str.to_date().alias("game_date"),
            (pl.col("team") == pl.col("home_team")).alias("is_home"),
            pl.when(pl.col("team") == pl.col("home_team"))
              .then(pl.col("away_team")).otherwise(pl.col("home_team")).alias("opponent"),
        )
        .sort("team", "game_date", "game_id")
    )

    # shift(1) AFTER ewm: the EMA at row i includes game i, so shift it onto game i+1.
    ema_exprs = [
        pl.col(col).ewm_mean(span=span, adjust=False, min_samples=1)
          .shift(1).over("team").alias(f"{col}_ema{span}")
        for span in EMA_SPANS for col in EMA_BASE_COLS
    ]
    rest = (
        pl.when(pl.col("season") == pl.col("season").shift(1).over("team"))
          .then((pl.col("game_date") - pl.col("game_date").shift(1).over("team")).dt.total_days())
          .otherwise(pl.lit(MAX_REST_DAYS))
          .clip(0, MAX_REST_DAYS)
          .alias("rest_days")
    )
    tg = tg.with_columns(ema_exprs + [rest])

    # Opponent adjustment: raw EPA minus the opponent's pre-game EMA-8, then rolled.
    # Without this, a team that just played three bad defenses looks elite. Opponent
    # EMAs are already shifted, so this stays leakage-safe.
    opp = (tg.select("game_id", "team", "off_epa_ema8", "def_epa_ema8")
             .rename({"team": "opponent", "off_epa_ema8": "opp_off_str",
                      "def_epa_ema8": "opp_def_str"}))
    tg = tg.join(opp, on=["game_id", "opponent"], how="left").with_columns(
        (pl.col("off_epa") - pl.col("opp_def_str")).alias("adj_off_game"),
        (pl.col("def_epa") - pl.col("opp_off_str")).alias("adj_def_game"),
    ).sort("team", "game_date", "game_id")
    adj_exprs = [
        pl.col(f"adj_{side}_game").ewm_mean(span=sp, adjust=False, min_samples=1)
          .shift(1).over("team").alias(f"{side}_epa_adj_ema{sp}")
        for sp in EMA_SPANS for side in ("off", "def")
    ]
    tg = tg.with_columns(adj_exprs)

    feat_cols = ([f"{c}_ema{s}" for s in EMA_SPANS for c in EMA_BASE_COLS]
                 + [f"{side}_epa_adj_ema{sp}" for sp in EMA_SPANS for side in ("off", "def")])
    return tg.select(
        ["game_id", "team", "season", "week", "is_home", "rest_days"] + feat_cols
    )


def _compute_elo(sched: pl.DataFrame) -> pl.DataFrame:
    games = sched.sort("gameday", "game_id").select(
        "game_id", "season", "home_team", "away_team",
        "result", "is_neutral_site", "is_played"
    )
    ratings: dict[str, float] = {}
    cur_season = None
    rows = []

    for r in games.iter_rows(named=True):
        if cur_season is None:
            cur_season = r["season"]
        if r["season"] != cur_season:
            for t in ratings:
                ratings[t] = ELO_REGRESS * ratings[t] + (1 - ELO_REGRESS) * ELO_BASE
            cur_season = r["season"]

        home, away = r["home_team"], r["away_team"]
        eh = ratings.get(home, ELO_BASE)
        ea = ratings.get(away, ELO_BASE)
        rows.append({"game_id": r["game_id"], "home_elo_pre": eh, "away_elo_pre": ea})

        if r["is_played"] and r["result"] is not None:
            hfa = 0.0 if r["is_neutral_site"] else ELO_HFA
            exp_home = 1.0 / (1.0 + 10 ** ((ea - eh - hfa) / 400.0))
            actual = 1.0 if r["result"] > 0 else (0.5 if r["result"] == 0 else 0.0)
            delta = ELO_K * (actual - exp_home)
            ratings[home] = eh + delta
            ratings[away] = ea - delta

    return pl.DataFrame(rows).with_columns(
        (pl.col("home_elo_pre") - pl.col("away_elo_pre")).alias("elo_diff")
    )


def _haversine(lat1, lon1, lat2, lon2) -> float:
    r = 3958.8  # earth radius, miles
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


def _travel(sched: pl.DataFrame) -> pl.DataFrame:
    # Home team is assumed not to travel, and away miles are stadium-to-stadium. Both are
    # wrong for international games; home_field_advantage=0 is the only signal there.
    def miles(home: str, away: str) -> float | None:
        h, a = TEAM_COORDS.get(home), TEAM_COORDS.get(away)
        return _haversine(a[0], a[1], h[0], h[1]) if h and a else None

    out = sched.select("game_id", "home_team", "away_team")
    away_miles = [miles(r["home_team"], r["away_team"]) for r in out.iter_rows(named=True)]
    return out.select("game_id").with_columns(
        pl.Series("away_travel_miles", away_miles),
        pl.lit(0.0).alias("home_travel_miles"),
    )


def _injury_team_week() -> pl.DataFrame:
    inj = pl.read_delta(str(SILVER_DIR / "injuries"))
    # Schedules store season/week as i32; polars won't join on mismatched key dtypes.
    inj = inj.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    inj = inj.with_columns(
        pl.col("position").fill_null("UNK")
          .replace_strict(POS_WEIGHT, default=0.8, return_dtype=pl.Float64).alias("pw"),
        pl.col("report_status").fill_null("None")
          .replace_strict(STATUS_WEIGHT, default=0.0, return_dtype=pl.Float64).alias("sw"),
    ).with_columns(
        (pl.col("pw") * pl.col("sw")).alias("iw"),
        ((pl.col("position") == "QB")
         & pl.col("report_status").is_in(["Out", "Doubtful"])).alias("is_qb_out"),
    )
    return inj.group_by("season", "week", "team").agg(
        pl.col("iw").sum().round(3).alias("injury_index"),
        pl.col("is_qb_out").max().cast(pl.Int8).alias("qb_out"),
    )


QB_EMA_SPAN = 10
QB_MIN_DROPBACKS = 10       # drops RB/WR trick-play "passers"
QB_REPLACEMENT_PCTL = 0.15  # debuts and backups with no history get this


def _qb_pregame_value(pbp: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
    """Starter's rolling passing EPA/play going into each game."""
    qg = (
        pbp.filter((pl.col("pass") == 1) & pl.col("passer_player_id").is_not_null())
        .group_by("passer_player_id", "game_id")
        .agg(pl.col("epa").mean().alias("qb_epa"), pl.len().alias("dropbacks"))
        .filter(pl.col("dropbacks") >= QB_MIN_DROPBACKS)
    )
    dates = sched.select("game_id", "gameday").with_columns(
        pl.col("gameday").str.to_date().alias("game_date"))
    qg = (qg.join(dates, on="game_id", how="inner")
            .sort("passer_player_id", "game_date", "game_id")
            .with_columns(
                pl.col("qb_epa").ewm_mean(span=QB_EMA_SPAN, adjust=False, min_samples=1)
                  .shift(1).over("passer_player_id").alias("qb_value")))
    return qg.select("game_id", "passer_player_id", "qb_value")


def build_gold() -> pl.DataFrame:
    pbp = pl.read_delta(str(SILVER_DIR / "pbp"))
    sched = pl.read_delta(str(SILVER_DIR / "schedules"))

    # TODO: unplayed games have no pbp, so they get no timeline row (all EMA diffs null)
    # and no QB-value row (both QBs fall back to replacement level). Elo and the lines
    # are the only live signal at inference time. Fix: carry each team's/QB's latest
    # post-game EMA forward onto its next scheduled game.
    timeline = _team_timeline(_team_game_stats(pbp), sched)
    elo = _compute_elo(sched)
    travel = _travel(sched)

    feat_cols = [c for c in timeline.columns
                 if c not in ("game_id", "team", "season", "week", "is_home")]
    home_feats = timeline.filter(pl.col("is_home")).drop("team", "is_home", "season", "week")
    away_feats = timeline.filter(~pl.col("is_home")).drop("team", "is_home", "season", "week")
    home_feats = home_feats.rename({c: f"home_{c}" for c in feat_cols})
    away_feats = away_feats.rename({c: f"away_{c}" for c in feat_cols})

    base = sched.select(
        "game_id", "season", "week", "gameday", "home_team", "away_team",
        "spread_line", "total_line", "home_moneyline", "away_moneyline",
        "div_game", "is_dome", "is_neutral_site", "home_field_advantage",
        "temp", "wind", "is_played", "result", "total",
        "home_qb_id", "away_qb_id",
    )

    inj = _injury_team_week()
    home_inj = inj.rename({"team": "home_team", "injury_index": "home_injury_index",
                           "qb_out": "home_qb_out"})
    away_inj = inj.rename({"team": "away_team", "injury_index": "away_injury_index",
                           "qb_out": "away_qb_out"})

    g = (
        base
        .join(home_feats, on="game_id", how="left")
        .join(away_feats, on="game_id", how="left")
        .join(elo, on="game_id", how="left")
        .join(travel, on="game_id", how="left")
        .join(home_inj, on=["season", "week", "home_team"], how="left")
        .join(away_inj, on=["season", "week", "away_team"], how="left")
    )
    # No row in the report means nobody listed, not unknown.
    g = g.with_columns(
        pl.col("home_injury_index").fill_null(0.0),
        pl.col("away_injury_index").fill_null(0.0),
        pl.col("home_qb_out").fill_null(0),
        pl.col("away_qb_out").fill_null(0),
    )

    qb = _qb_pregame_value(pbp, sched)
    repl = float(qb["qb_value"].quantile(QB_REPLACEMENT_PCTL) or 0.0)
    g = (
        g.join(qb.rename({"passer_player_id": "home_qb_id", "qb_value": "home_qb_value"}),
               on=["game_id", "home_qb_id"], how="left")
         .join(qb.rename({"passer_player_id": "away_qb_id", "qb_value": "away_qb_value"}),
               on=["game_id", "away_qb_id"], how="left")
         .with_columns(pl.col("home_qb_value").fill_null(repl),
                       pl.col("away_qb_value").fill_null(repl))
    )

    delta_exprs = [
        (pl.col(f"home_{c}") - pl.col(f"away_{c}")).alias(f"{c}_diff")
        for c in feat_cols
    ]
    g = g.with_columns(
        delta_exprs
        + [(pl.col("away_travel_miles") - pl.col("home_travel_miles")).alias("travel_diff"),
           (pl.col("home_injury_index") - pl.col("away_injury_index")).alias("injury_index_diff"),
           (pl.col("home_qb_out") - pl.col("away_qb_out")).alias("qb_out_diff"),
           (pl.col("home_qb_value") - pl.col("away_qb_value")).alias("qb_value_diff")]
    )

    # Ties and pushes get null targets rather than a third class.
    g = g.with_columns(
        pl.when(pl.col("is_played") & (pl.col("result") != 0))
          .then((pl.col("result") > 0).cast(pl.Int8)).alias("home_win"),
        pl.when(pl.col("is_played") & (pl.col("result") != pl.col("spread_line")))
          .then((pl.col("result") > pl.col("spread_line")).cast(pl.Int8)).alias("home_covered"),
        pl.when(pl.col("is_played")).then(pl.col("result")).alias("target_margin"),
    )
    return g.sort("season", "week", "game_id")


def run() -> int:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Gold build ===")
    g = build_gold()

    report = (
        dq.DataQualityReport("gold_game_features")
        .add(dq.expect_unique(g, ["game_id"]))
        .add(dq.expect_not_null(g, "home_team"))
        .add(dq.expect_in_range(g, "home_elo_pre", 1000, 2000))
        .add(dq.expect_in_range(g, "away_elo_pre", 1000, 2000))
    )
    report.print_summary()
    report.raise_if_failed()

    write_table(g, str(GOLD_DIR / "game_features"), ["week", "game_id"])

    trainable = g.filter(pl.col("home_win").is_not_null()).height
    print(f"  wrote gold/game_features: {g.height:,} games, {g.width} cols")
    print(f"  trainable (played, non-tie): {trainable:,}  |  future: "
          f"{g.filter(~pl.col('is_played')).height:,}")
    print("[DONE] Gold complete.")
    return 0
