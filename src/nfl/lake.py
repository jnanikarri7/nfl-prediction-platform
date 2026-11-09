from __future__ import annotations

import polars as pl
from deltalake import DeltaTable, write_deltalake


def write_table(df: pl.DataFrame, path: str, z_order: list[str]) -> None:
    """Overwrite a Silver/Gold Delta table, partitioned by season, then Z-ORDER it.

    Bronze doesn't use this -- it needs append/merge semantics.
    """
    part = ["season"] if "season" in df.columns else None
    write_deltalake(path, df.to_arrow(), mode="overwrite",
                    partition_by=part, schema_mode="overwrite")
    cols = [c for c in z_order if c in df.columns]
    if not cols:
        return
    # Best-effort: the table is already written and readable at this point, so a failed
    # compaction is a perf problem, not a correctness one. Don't fail the build over it.
    try:
        DeltaTable(path).optimize.z_order(cols)
    except Exception as e:
        print(f"    (z-order skipped for {path}: {type(e).__name__}: {e})")
