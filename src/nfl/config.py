"""Paths, season range, and the Bronze dataset registry."""
from __future__ import annotations

import datetime as dt
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import nflreadpy as nfl
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
# Override to keep the lake off the repo drive.
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR", REPO_ROOT / "data"))
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

START_SEASON = 2015
# TODO: in Jan/Feb this asks for a season that hasn't kicked off yet (playoffs belong to
# the previous year). Untested what nflreadpy does then; pass --end explicitly to be safe.
END_SEASON = dt.date.today().year
NGS_FIRST_SEASON = 2016  # NGS starts in 2016


def season_range(start: int = START_SEASON, end: int = END_SEASON) -> list[int]:
    return list(range(start, end + 1))


@dataclass(frozen=True)
class Dataset:
    name: str                                    # folder under data/bronze/
    loader: Callable[[list[int]], pl.DataFrame]
    source_file: str                             # written to _source_file
    partition_by: tuple[str, ...] = ("season",)
    first_season: int | None = None              # earlier seasons are skipped


def _ngs(stat_type: str) -> Callable[[list[int]], pl.DataFrame]:
    return lambda seasons: nfl.load_nextgen_stats(seasons=seasons, stat_type=stat_type)


DATASETS: list[Dataset] = [
    Dataset("pbp",          lambda s: nfl.load_pbp(seasons=s),            "nflverse/pbp"),
    Dataset("schedules",    lambda s: nfl.load_schedules(seasons=s),      "nflverse/schedules"),
    Dataset("player_stats", lambda s: nfl.load_player_stats(seasons=s),   "nflverse/player_stats"),
    Dataset("rosters",      lambda s: nfl.load_rosters_weekly(seasons=s), "nflverse/rosters_weekly"),
    Dataset("injuries",     lambda s: nfl.load_injuries(seasons=s),       "nflverse/injuries"),
    Dataset("ngs_passing",   _ngs("passing"),   "nflverse/ngs_passing",   first_season=NGS_FIRST_SEASON),
    Dataset("ngs_rushing",   _ngs("rushing"),   "nflverse/ngs_rushing",   first_season=NGS_FIRST_SEASON),
    Dataset("ngs_receiving", _ngs("receiving"), "nflverse/ngs_receiving", first_season=NGS_FIRST_SEASON),
]
