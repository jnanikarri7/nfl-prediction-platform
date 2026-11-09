#!/usr/bin/env python
"""Phase 5.5 — train the market-anchored distributional model.

  python scripts/run_market_model.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (sets sys.path + UTF-8 stdout)
from nfl import market_model

if __name__ == "__main__":
    raise SystemExit(market_model.run())
