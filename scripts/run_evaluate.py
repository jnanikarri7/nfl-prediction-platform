#!/usr/bin/env python
"""Phase 5.6 — honest betting backtest (ROI + ATS) of the market-anchored model.

  python scripts/run_evaluate.py
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

from nfl import evaluate

if __name__ == "__main__":
    raise SystemExit(evaluate.run())
