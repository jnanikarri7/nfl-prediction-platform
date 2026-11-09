#!/usr/bin/env python
"""CLI entrypoint for the local Silver build.

  python scripts/run_silver.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (sets sys.path + UTF-8 stdout)
from nfl import silver

if __name__ == "__main__":
    counts = silver.run()
    raise SystemExit(0 if all(counts.values()) else 1)
