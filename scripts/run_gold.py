#!/usr/bin/env python
"""CLI entrypoint for the local Gold feature build.

  python scripts/run_gold.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (sets sys.path + UTF-8 stdout)
from nfl import gold

if __name__ == "__main__":
    raise SystemExit(gold.run())
