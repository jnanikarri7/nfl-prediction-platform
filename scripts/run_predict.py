#!/usr/bin/env python
"""Load the champion model from the MLflow registry and run inference.

  python scripts/run_predict.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (sets sys.path + UTF-8 stdout)
from nfl import predict

if __name__ == "__main__":
    raise SystemExit(predict.run())
