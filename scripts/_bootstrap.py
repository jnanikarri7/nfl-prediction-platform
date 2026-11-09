"""Imported first by every scripts/run_*.py, so they work without `pip install -e .`."""
import sys
from pathlib import Path

# Windows consoles default to cp1252 and crash on polars' box-drawing table borders.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # already wrapped (pytest, IDE consoles)
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
