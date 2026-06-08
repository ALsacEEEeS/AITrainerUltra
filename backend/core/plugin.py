"""Plugin system for hot-pluggable extensions."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type


class PluginBase:
    """Base class for all plugins."""

    name: str = ""
    version: str = "1.0.0"
    description: str = ""

    def on_load(self) -> None:
        """Called when plugin is loaded."""
        pass

    def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        pass

    def get_routes(self) -> List[Dict[str, Any]]:
        """Return custom API routes."""
        return []

    def get_node_definitions(self) -> List[Dict[str, Any]]:
        """Return custom node types for the editor."""
        return []


class PluginManager:
    """Discover, load, and manage plugins."""

    def __init__(self, plugin_dirs: Optional[List[str]] = None) -> None:
        self.plugins: Dict[str, PluginBase] = {}
        self.plugin_dirs = plugin_dirs or ["plugins"]
        self._hooks: Dict[str, List[Callable]] = {}

    def discover(self) -> List[Dict[str, Any]]:
        """Discover available plugins without loading them."""
        discovered = []
        for plugin_dir in self.plugin_dirs:
            path = Path(plugin_dir)
            if not path.exists():
                continue
            for item in path.iterdir():
                if item.is_dir() and (item / "__init__.py").exists():
                    try:
                        spec = importlib.util.spec_from_file_location(
                            item.name, item / "__init__.py"
                        )
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            for _, obj in inspect.getmembers(mod, inspect.isclass):
                                if issubclass(obj, PluginBase) and obj is not PluginBase:
                                    discovered.append({
                                        "name": getattr(obj, "name", item.name),
                                        "version": getattr(obj, "version", "1.0.0"),
                                        "description": getattr(obj, "description", ""),
                                        "class": obj,
                                    })
                    except Exception as e:
                        print(f"Failed to discover plugin {item.name}: {e}")
        return discovered

    def load_plugin(self, plugin_class: Type[PluginBase]) -> bool:
        """Load a plugin by its class."""
        try:
            instance = plugin_class()
            instance.on_load()
            self.plugins[instance.name] = instance
            return True
        except Exception as e:
            print(f"Failed to load plugin {plugin_class.__name__}: {e}")
            return False

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin by name."""
        if name in self.plugins:
            try:
                self.plugins[name].on_unload()
                del self.plugins[name]
                return True
            except Exception as e:
                print(f"Failed to unload plugin {name}: {e}")
        return False

    def load_all(self) -> int:
        """Discover and load all available plugins."""
        count = 0
        for info in self.discover():
            if self.load_plugin(info["class"]):
                count += 1
        return count

    def get_plugin(self, name: str) -> Optional[PluginBase]:
        return self.plugins.get(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "loaded": True,
            }
            for p in self.plugins.values()
        ]

    def register_hook(self, event: str, handler: Callable) -> None:
        """Register a hook handler."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(handler)

    def trigger_hook(self, event: str, data: Any = None) -> List[Any]:
        """Trigger all handlers for a hook event."""
        results = []
        for handler in self._hooks.get(event, []):
            try:
                result = handler(data)
                results.append(result)
            except Exception as e:
                print(f"Hook {event} error: {e}")
        return results

    def get_node_definitions(self) -> List[Dict[str, Any]]:
        """Aggregate node definitions from all plugins."""
        nodes = []
        for plugin in self.plugins.values():
            nodes.extend(plugin.get_node_definitions())
        return nodes


plugin_manager = PluginManager()


def plugin(name: str, version: str = "1.0.0", description: str = ""):
    """Decorator to register a plugin class."""
    def decorator(cls):
        if not hasattr(cls, "name") or not cls.name:
            cls.name = name
        if not hasattr(cls, "version") or not cls.version:
            cls.version = version
        if not hasattr(cls, "description") or not cls.description:
            cls.description = description
        plugin_manager.load_plugin(cls)
        return cls
    return decorator
