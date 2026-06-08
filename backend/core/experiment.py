"""Experiment tracking system for reproducible training."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Experiment:
    """A single training experiment record."""
    name: str
    model_type: str
    model_name: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    status: str = "created"  # created | running | completed | failed
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    duration: float = 0.0
    artifact_path: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Experiment":
        return cls(**data)


try:
    from backend.core.database import db as _sqlite_db
    _HAS_SQLITE = True
except Exception:
    _HAS_SQLITE = False


class ExperimentTracker:
    """Manage experiments with SQLite (primary) or JSON (fallback) storage."""

    def __init__(self, storage_dir: str = "./experiments") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._current: Optional[Experiment] = None
        self._use_sqlite = _HAS_SQLITE

    def create(
        self,
        name: str,
        model_type: str,
        model_name: str,
        hyperparameters: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> Experiment:
        """Create a new experiment."""
        exp = Experiment(
            name=name,
            model_type=model_type,
            model_name=model_name,
            hyperparameters=hyperparameters or {},
            tags=tags or [],
            status="created",
        )
        self._current = exp
        self._save(exp)
        return exp

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Log metrics for the current experiment."""
        if self._current is None:
            raise RuntimeError("No active experiment")
        for key, value in metrics.items():
            if key not in self._current.metrics:
                self._current.metrics[key] = []
            self._current.metrics[key].append({"step": step, "value": value})
        self._current.updated_at = time.time()
        self._save(self._current)

    def log_final(self, metrics: Dict[str, float]) -> None:
        if self._current:
            self._current.final_metrics = metrics
            self._current.status = "completed"
            self._current.duration = time.time() - self._current.created_at
            self._save(self._current)

    def set_status(self, status: str) -> None:
        if self._current:
            self._current.status = status
            self._current.updated_at = time.time()
            self._save(self._current)

    def set_artifact(self, path: str) -> None:
        if self._current:
            self._current.artifact_path = path
            self._save(self._current)

    def get_current(self) -> Optional[Experiment]:
        return self._current

    def list(self, status: Optional[str] = None) -> List[Experiment]:
        """List all experiments, optionally filtered by status."""
        if self._use_sqlite:
            try:
                rows = _sqlite_db.list_experiments(status)
                if rows:
                    return [Experiment.from_dict(r) for r in rows]
            except Exception:
                pass

        # JSON fallback
        experiments = []
        for f in sorted(self.storage_dir.glob("*.json"), reverse=True):
            exp = self._load(f)
            if exp and (status is None or exp.status == status):
                experiments.append(exp)
        return experiments

    def get(self, name: str) -> Optional[Experiment]:
        if self._use_sqlite:
            try:
                row = _sqlite_db.get_experiment(name)
                if row:
                    return Experiment.from_dict(row)
            except Exception:
                pass
        path = self.storage_dir / f"{name}.json"
        return self._load(path) if path.exists() else None

    def delete(self, name: str) -> None:
        if self._use_sqlite:
            try:
                _sqlite_db.delete_experiment(name)
            except Exception:
                pass
        path = self.storage_dir / f"{name}.json"
        if path.exists():
            path.unlink()

    def compare(self, names: List[str]) -> List[Experiment]:
        """Load multiple experiments for comparison."""
        return [self.get(n) for n in names if self.get(n) is not None]

    def _save(self, exp: Experiment) -> None:
        if self._use_sqlite:
            try:
                _sqlite_db.save_experiment(exp.to_dict())
                return
            except Exception:
                pass
        # JSON fallback
        path = self.storage_dir / f"{exp.name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(exp.to_dict(), f, indent=2, ensure_ascii=False)

    def _load(self, path: Path) -> Optional[Experiment]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return Experiment.from_dict(json.load(f))
        except Exception:
            return None


exp_tracker = ExperimentTracker()
