#!/usr/bin/env python
"""Load the champion model from the MLflow registry and run inference.

  python scripts/run_predict.py
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

from nfl import predict

if __name__ == "__main__":
    raise SystemExit(predict.run())
