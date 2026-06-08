"""Event system for real-time training updates."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List


class EventType(Enum):
    """Training event types."""
    TRAINING_START = "training:start"
    TRAINING_STEP = "training:step"
    TRAINING_EPOCH = "training:epoch"
    TRAINING_END = "training:end"
    TRAINING_ERROR = "training:error"
    METRICS_UPDATE = "metrics:update"
    MODEL_LOADED = "model:loaded"
    MODEL_SAVED = "model:saved"
    NODE_RUN = "node:run"
    NODE_COMPLETE = "node:complete"
    WORKFLOW_START = "workflow:start"
    WORKFLOW_END = "workflow:end"
    LOG_MESSAGE = "log:message"


class EventBus:
    """Simple async event bus for inter-component communication."""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable]] = {}
        self._ws_clients: List[Any] = []

    def subscribe(self, event: str, handler: Callable) -> None:
        """Register an event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable) -> None:
        """Remove an event handler."""
        if event in self._handlers:
            self._handlers[event] = [
                h for h in self._handlers[event] if h is not handler
            ]

    def register_ws(self, ws: Any) -> None:
        """Register a WebSocket client."""
        self._ws_clients.append(ws)

    def unregister_ws(self, ws: Any) -> None:
        """Remove a WebSocket client."""
        self._ws_clients = [c for c in self._ws_clients if c is not ws]

    async def emit(self, event: str, data: Any = None) -> None:
        """Emit an event to all handlers and WS clients."""
        payload = {
            "event": event,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }
        tasks = []
        if event in self._handlers:
            for handler in self._handlers[event]:
                tasks.append(self._safe_call(handler, data))
        for ws in self._ws_clients:
            tasks.append(self._ws_send(ws, payload))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_call(self, handler: Callable, data: Any) -> None:
        """Call a handler safely, supporting both sync and async."""
        try:
            result = handler(data)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            print(f"Event handler error: {e}")

    async def _ws_send(self, ws: Any, payload: Dict) -> None:
        """Send payload over WebSocket."""
        try:
            text = json.dumps(payload, default=str)
            await ws.send_text(text)
        except Exception:
            pass


# Global event bus instance
event_bus = EventBus()
