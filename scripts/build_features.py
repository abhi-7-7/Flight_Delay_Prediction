"""Regenerate the Phase 2 feature store from Datasets/.

Usage:  .venv/bin/python scripts/build_features.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2.features import build_store  # noqa: E402

if __name__ == "__main__":
    t = time.time()
    info = build_store()
    print(f"Feature store built in {time.time() - t:.1f}s: {info}")
