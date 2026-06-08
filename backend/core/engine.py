"""Main training engine orchestrator."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, Optional

from backend.core.config import TrainingConfig
from backend.core.events import EventBus, EventType, event_bus as _global_event_bus
from backend.core.registry import registry


class TrainingEngine:
    """Orchestrates training jobs across model types."""

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self._current_job: Optional[Dict[str, Any]] = None
        self._is_running = False
        self._event_bus = event_bus or _global_event_bus

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def current_job(self) -> Optional[Dict[str, Any]]:
        return self._current_job

    async def start_training(self, config: TrainingConfig) -> str:
        """Start a training job from config."""
        if self._is_running:
            raise RuntimeError("Training already in progress")

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        config.save(f"{config.output_dir}/config.json")

        trainer_cls = registry.get_trainer(config.model_type)
        trainer = trainer_cls(config, self._event_bus)

        job_id = f"job_{int(time.time())}"
        self._current_job = {"id": job_id, "config": config, "trainer": trainer}
        self._is_running = True

        if self._event_bus:
            await self._event_bus.emit(EventType.TRAINING_START.value, {
                "job_id": job_id,
                "model_type": config.model_type,
                "model_name": config.model_name,
            })

        asyncio.create_task(self._run_training(job_id, trainer))
        return job_id

    async def _run_training(self, job_id: str, trainer: Any) -> None:
        try:
            result = await trainer.train()
            if self._event_bus:
                await self._event_bus.emit(EventType.TRAINING_END.value, {
                    "job_id": job_id,
                    "result": result,
                })
        except Exception as e:
            if self._event_bus:
                await self._event_bus.emit(EventType.TRAINING_ERROR.value, {
                    "job_id": job_id,
                    "error": str(e),
                })
        finally:
            self._is_running = False
            self._current_job = None

    async def stop_training(self) -> None:
        """Stop the current training job."""
        if self._current_job:
            trainer = self._current_job["trainer"]
            if hasattr(trainer, "stop"):
                trainer.stop()
            self._is_running = False

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._is_running,
            "job": self._current_job["id"] if self._current_job else None,
            "model_type": self._current_job["config"].model_type if self._current_job else None,
        }


# Global engine instance
engine = TrainingEngine()
