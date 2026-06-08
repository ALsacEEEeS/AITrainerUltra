"""Training metrics tracking and visualization."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class MetricsTracker:
    """Tracks training metrics over time."""

    def __init__(self) -> None:
        self._metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._current_step = 0

    def log(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value at a given step."""
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append({
            "step": step or self._current_step,
            "value": value,
        })
        self._current_step += 1

    def log_dict(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log(name, value, step)

    def get(self, name: str) -> List[Dict[str, Any]]:
        return self._metrics.get(name, [])

    def latest(self, name: str) -> Optional[float]:
        values = self._metrics.get(name, [])
        return values[-1]["value"] if values else None

    def summary(self) -> Dict[str, Any]:
        result = {}
        for name, values in self._metrics.items():
            vals = [v["value"] for v in values]
            if vals:
                result[name] = {
                    "min": min(vals),
                    "max": max(vals),
                    "avg": sum(vals) / len(vals),
                    "latest": vals[-1],
                    "count": len(vals),
                }
        return result

    def reset(self) -> None:
        """Clear all tracked metrics."""
        self._metrics.clear()
        self._current_step = 0

    def to_chart_data(self, name: str) -> Dict[str, List]:
        values = self._metrics.get(name, [])
        return {
            "steps": [v["step"] for v in values],
            "values": [v["value"] for v in values],
        }
