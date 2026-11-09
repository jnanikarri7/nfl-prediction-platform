"""Local (polars + delta-rs) implementation of the NFL medallion pipeline.

The PySpark notebooks under databricks/notebooks/ follow the same logic, but they
lag behind this package -- see the README's parity notes.
"""
import os

# MLflow 3 prints an "agent hint" banner on import; it clutters every CLI run.
# Set here so any module importing mlflow after `nfl` gets it.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

__version__ = "0.1.0"
