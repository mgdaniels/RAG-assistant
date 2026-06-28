"""Pytest configuration.

Adds ``src`` and the project root to ``sys.path`` so the test suite can import
the ``ragassistant`` package and top-level modules (such as ``app``) whether or
not the project has been installed with ``pip install -e .``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
