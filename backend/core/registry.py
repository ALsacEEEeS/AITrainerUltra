"""Model type registry for extensible model support."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Type


class ModelRegistry:
    """Registry for model trainers, datasets, and metrics."""

    def __init__(self) -> None:
        self._trainers: Dict[str, str] = {}
        self._model_info: Dict[str, Dict[str, Any]] = {}

    def register_trainer(self, name: str, module_path: str, info: Optional[Dict] = None) -> None:
        """Register a trainer class by module path (lazy loading)."""
        self._trainers[name] = module_path
        if info:
            self._model_info[name] = info

    def get_trainer(self, name: str) -> Type:
        if name not in self._trainers:
            raise KeyError(f"Unknown trainer '{name}'. Available: {list(self._trainers.keys())}")
        full_path = self._trainers[name]
        parts = full_path.split(".")
        module_path = ".".join(parts[:-1])
        class_name = parts[-1]
        module = importlib.import_module(module_path)
        if not hasattr(module, class_name):
            raise ImportError(f"Class {class_name} not found in {module_path}")
        return getattr(module, class_name)

    def list_trainers(self) -> List[str]:
        """List all registered trainers."""
        return list(self._trainers.keys())

    def get_model_info(self, name: str) -> Dict[str, Any]:
        return self._model_info.get(name, {})

    def list_supported_types(self) -> Dict[str, Dict[str, Any]]:
        """Return all supported model types with metadata."""
        return {
            name: self._model_info.get(name, {"description": "No description"})
            for name in self._trainers
        }


# Global registry
registry = ModelRegistry()


def register(name: str, info: Optional[Dict] = None):
    """Decorator to register a trainer class with lazy loading.

    The decorated class's module path is stored; actual import
    happens when the trainer is first requested.
    """
    def decorator(cls):
        module_path = f"{cls.__module__}.{cls.__qualname__}"
        # Register the actual class
        import backend.core.registry as reg_mod
        if name not in reg_mod.registry._trainers:
            reg_mod.registry._trainers[name] = module_path
            if info:
                reg_mod.registry._model_info[name] = info
        return cls
    return decorator
