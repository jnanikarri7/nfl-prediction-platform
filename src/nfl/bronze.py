"""Bronze: land nflreadpy output as-is in Delta, plus lineage columns. No cleaning here."""
from __future__ import annotations

import datetime as dt
import time
import uuid

import polars as pl
from deltalake import write_deltalake

from .config import BRONZE_DIR, DATASETS, Dataset, season_range

LINEAGE_COLS = ("_ingest_ts", "_source_file", "_batch_id", "_source_rows")


# TODO: utcnow() is deprecated as of 3.12. Switching to an aware datetime changes the
# _ingest_ts Delta type, so existing append-mode tables would need a rewrite first.
def _new_batch_id() -> str:
    return f"{dt.datetime.utcnow():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"


def _cast_null_columns(df: pl.DataFrame) -> pl.DataFrame:
    # Columns that are entirely null come through as polars Null dtype, which Delta
    # can't store. Utf8 is just a placeholder type; no values change.
    null_cols = [c for c, t in df.schema.items() if t == pl.Null]
    if null_cols:
        df = df.with_columns(pl.col(c).cast(pl.Utf8) for c in null_cols)
    return df


def _add_lineage(df: pl.DataFrame, source_file: str, batch_id: str) -> pl.DataFrame:
    return df.with_columns(
        pl.lit(dt.datetime.utcnow()).alias("_ingest_ts"),
        pl.lit(source_file).alias("_source_file"),
        pl.lit(batch_id).alias("_batch_id"),
        pl.lit(df.height, dtype=pl.Int64).alias("_source_rows"),
    )


def ingest_dataset(ds: Dataset, seasons: list[int], batch_id: str, mode: str) -> int:
    """Load one dataset and write it to its Bronze Delta table. Returns row count."""
    use_seasons = [s for s in seasons if ds.first_season is None or s >= ds.first_season]
    if not use_seasons:
        print(f"  {ds.name:16s} no seasons in range (needs >= {ds.first_season}); skipped")
        return 0

    t0 = time.time()
    df = ds.loader(use_seasons)
    if df.height == 0:
        print(f"  {ds.name:16s} source returned 0 rows; skipped")
        return 0

    df = _cast_null_columns(df)
    df = _add_lineage(df, ds.source_file, batch_id)

    path = str(BRONZE_DIR / ds.name)
    partition_by = [c for c in ds.partition_by if c in df.columns]
    write_deltalake(
        path,
        df.to_arrow(),
        mode=mode,
        partition_by=partition_by or None,
        schema_mode="overwrite" if mode == "overwrite" else "merge",
    )
    print(f"  {ds.name:16s} rows={df.height:>7,} cols={df.width:>4} "
          f"part={partition_by} ({time.time() - t0:.1f}s) -> {path}")
    return df.height


def run(seasons: list[int] | None = None,
        datasets: list[Dataset] | None = None,
        mode: str = "overwrite") -> dict[str, int]:
    # overwrite is the local default because you re-run constantly while developing.
    # append is the "real" Bronze behaviour; the duplicates it creates are dropped in Silver.
    seasons = seasons or season_range()
    datasets = datasets or DATASETS
    batch_id = _new_batch_id()
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Bronze ingest | seasons {seasons[0]}-{seasons[-1]} "
          f"| mode={mode} | batch={batch_id} ===")
    counts = {ds.name: ingest_dataset(ds, seasons, batch_id, mode) for ds in datasets}
    print(f"[DONE] Bronze complete. Total rows: {sum(counts.values()):,}")
    return counts
