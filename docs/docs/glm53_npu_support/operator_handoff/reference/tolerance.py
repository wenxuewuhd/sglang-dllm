"""Moved to ``layer_check/tolerance.py``.

The two-reference acceptance method outlived the handoff package it was written
for -- every layer check uses it, and this package is withdrawn. Kept as a
shim so the withdrawn tests still import.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "layer_check"))
from tolerance import *  # noqa: F401,F403,E402
from tolerance import ABS_MIN, SLACK, budget, noise_floor, rel_err  # noqa: F401,E402
