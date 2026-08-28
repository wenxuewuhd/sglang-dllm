"""Make ``reference`` importable when pytest is run from anywhere."""

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
if str(PKG_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT.parent))
