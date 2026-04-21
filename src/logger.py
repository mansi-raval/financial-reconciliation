"""Centralised logging setup for the reconciliation system.

Call `get_logger(__name__)` in every module. One call to `setup()` in
main.py configures handlers for both console AND a rotating log file so
every run is fully auditable.
"""

import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "outputs"
LOG_FILE = LOG_DIR / "reconciliation.log"


def setup(level: int = logging.INFO) -> None:
    """Configure root logger once at startup (called from main.py)."""
    LOG_DIR.mkdir(exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)

    # File handler — DEBUG and above (captures everything)
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
