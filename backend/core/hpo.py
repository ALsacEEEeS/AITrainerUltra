"""Hyperparameter Optimization - Grid Search & Bayesian (Optuna)."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class SearchSpace:
    """Define a hyperparameter search space."""
    learning_rate: List[float] = field(default_factory=lambda: [1e-5, 3e-5, 5e-5])
    batch_size: List[int] = field(default_factory=lambda: [4, 8, 16])
    num_epochs: List[int] = field(default_factory=lambda: [2, 3, 5])
    warmup_steps: List[int] = field(default_factory=lambda: [50, 100, 200])
    weight_decay: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1])
    lora_r: List[int] = field(default_factory=lambda: [8, 16, 32])
    lora_alpha: List[int] = field(default_factory=lambda: [16, 32, 64])
    dropout: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])


@dataclass
class TrialResult:
    """Result of a single HPO trial."""
    trial_id: int
    params: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending | running | completed | failed
    error: Optional[str] = None


class GridSearch:
    """Exhaustive grid search over hyperparameter space."""

    def __init__(self, search_space: SearchSpace) -> None:
        self.search_space = search_space
        self.results: List[TrialResult] = []

    def generate_trials(self) -> List[Dict[str, Any]]:
        """Generate all combinations from search space."""
        param_grid = {
            "learning_rate": self.search_space.learning_rate,
            "batch_size": self.search_space.batch_size,
            "num_epochs": self.search_space.num_epochs,
            "warmup_steps": self.search_space.warmup_steps,
            "weight_decay": self.search_space.weight_decay,
        }
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]


class RandomSearch:
    """Random search over hyperparameter space."""

    def __init__(self, search_space: SearchSpace, n_trials: int = 20) -> None:
        self.search_space = search_space
        self.n_trials = n_trials
        self.results: List[TrialResult] = []

    def generate_trials(self) -> List[Dict[str, Any]]:
        """Generate random hyperparameter combinations."""
        trials = []
        for _ in range(self.n_trials):
            trial = {
                "learning_rate": random.choice(self.search_space.learning_rate),
                "batch_size": random.choice(self.search_space.batch_size),
                "num_epochs": random.choice(self.search_space.num_epochs),
                "warmup_steps": random.choice(self.search_space.warmup_steps),
                "weight_decay": random.choice(self.search_space.weight_decay),
            }
            trials.append(trial)
        return trials


class BayesianOptimizer:
    """Simple Bayesian optimization using Optuna-style approach.

    Falls back to random search if optuna is not installed.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        n_trials: int = 20,
        direction: str = "minimize",
    ) -> None:
        self.search_space = search_space
        self.n_trials = n_trials
        self.direction = direction
        self.results: List[TrialResult] = []
        self._use_optuna = False

        try:
            import optuna  # noqa: F401
            self._use_optuna = True
        except ImportError:
            pass

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        progress_callback: Optional[Callable[[int, Dict[str, Any], float], None]] = None,
    ) -> Dict[str, Any]:
        """Run Bayesian optimization.

        Args:
            objective_fn: Function that takes params dict, returns metric.
            progress_callback: Called after each trial with (trial_id, params, value).

        Returns:
            Best hyperparameters found.
        """
        if self._use_optuna:
            return self._optimize_optuna(objective_fn, progress_callback)
        return self._optimize_random(objective_fn, progress_callback)

    def _optimize_optuna(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Use optuna for Bayesian optimization."""
        import optuna

        def objective(trial: Any) -> float:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
                "num_epochs": trial.suggest_int("num_epochs", 1, 10),
                "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            }
            value = objective_fn(params)
            return value

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials)

        for i, trial in enumerate(study.trials):
            if trial.values is not None:
                self.results.append(TrialResult(
                    trial_id=i,
                    params=trial.params,
                    metrics={"value": trial.values[0]},
                    status="completed",
                ))

        return study.best_params

    def _optimize_random(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Fallback random search."""
        best_value = float("inf") if self.direction == "minimize" else float("-inf")
        best_params = {}

        for i in range(self.n_trials):
            params = {
                "learning_rate": random.choice(self.search_space.learning_rate),
                "batch_size": random.choice(self.search_space.batch_size),
                "num_epochs": random.choice(self.search_space.num_epochs),
                "warmup_steps": random.choice(self.search_space.warmup_steps),
                "weight_decay": random.choice(self.search_space.weight_decay),
            }
            try:
                value = objective_fn(params)
                self.results.append(TrialResult(
                    trial_id=i, params=params,
                    metrics={"value": value}, status="completed",
                ))
                better = value < best_value if self.direction == "minimize" else value > best_value
                if better:
                    best_value = value
                    best_params = params
                if progress_callback:
                    progress_callback(i, params, value)
            except Exception as e:
                self.results.append(TrialResult(
                    trial_id=i, params=params,
                    status="failed", error=str(e),
                ))
                if progress_callback:
                    progress_callback(i, params, None)

        return best_params
