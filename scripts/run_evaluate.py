#!/usr/bin/env python
"""Phase 5.6 — honest betting backtest (ROI + ATS) of the market-anchored model.

  python scripts/run_evaluate.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (sets sys.path + UTF-8 stdout)
from nfl import evaluate

if __name__ == "__main__":
    raise SystemExit(evaluate.run())
