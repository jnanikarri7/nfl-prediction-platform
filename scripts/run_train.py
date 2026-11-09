#!/usr/bin/env python
"""CLI entrypoint for Phase 5 training.

  python scripts/run_train.py
Then explore:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (sets sys.path + UTF-8 stdout)
from nfl import train

if __name__ == "__main__":
    raise SystemExit(train.run())
