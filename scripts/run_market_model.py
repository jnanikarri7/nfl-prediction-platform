#!/usr/bin/env python
"""Phase 5.5 — train the market-anchored distributional model.

  python scripts/run_market_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl import market_model

if __name__ == "__main__":
    raise SystemExit(market_model.run())
