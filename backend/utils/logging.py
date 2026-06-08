"""Logging utilities."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class TrainingLogger:
    """Structured logger for training sessions."""

    def __init__(
        self,
        name: str = "aitrainer",
        log_dir: Optional[str] = None,
        level: int = logging.INFO,
    ) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(log_dir) / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def metric(self, name: str, value: float, step: int) -> None:
        """Log a metric in parseable format."""
        self.logger.info(f"METRIC | {name} | {value:.6f} | step {step}")


def get_logger(name: str = "aitrainer", log_dir: Optional[str] = None) -> TrainingLogger:
    return TrainingLogger(name, log_dir)
