import os
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional


def setup_logger(name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Create a console (and optional file) logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class MetricsLogger:
    """Append-mode CSV logger for per-epoch metrics."""

    def __init__(self, log_dir: str, filename: str = "metrics.csv"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)
        self._header_written = os.path.exists(self.filepath)

    def log(self, metrics: Dict[str, Any]) -> None:
        """Append one row of metrics."""
        write_header = not self._header_written
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(metrics)

    def save_config(self, config: Dict[str, Any], filename: str = "config.json") -> None:
        """Save experiment config alongside metrics."""
        path = os.path.join(os.path.dirname(self.filepath), filename)
        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)
