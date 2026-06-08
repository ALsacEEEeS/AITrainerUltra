"""Job scheduler for queuing and executing training jobs."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """A scheduled training job."""
    id: str = field(default_factory=lambda: f"job_{uuid.uuid4().hex[:8]}")
    name: str = ""
    model_type: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    progress: float = 0.0
    tags: List[str] = field(default_factory=list)


class JobQueue:
    """Async FIFO job queue with priority support."""

    def __init__(self, max_concurrent: int = 2) -> None:
        self.queue: List[Job] = []
        self.active: Dict[str, Job] = {}
        self.history: List[Job] = []
        self.max_concurrent = max_concurrent
        self._handlers: Dict[str, List[Callable]] = {}

    def enqueue(self, job: Job) -> str:
        """Add a job to the queue."""
        self.queue.append(job)
        self.queue.sort(key=lambda j: (-j.priority, j.created_at))
        self._emit("enqueued", job)
        return job.id

    def dequeue(self) -> Optional[Job]:
        if not self.queue or len(self.active) >= self.max_concurrent:
            return None
        job = self.queue.pop(0)
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        self.active[job.id] = job
        self._emit("started", job)
        return job

    def complete(self, job_id: str, result: Optional[Dict] = None) -> None:
        """Mark a job as completed."""
        job = self.active.pop(job_id, None)
        if job:
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.result = result
            job.progress = 1.0
            self.history.append(job)
            self._emit("completed", job)

    def fail(self, job_id: str, error: str) -> None:
        """Mark a job as failed."""
        job = self.active.pop(job_id, None)
        if job:
            job.status = JobStatus.FAILED
            job.completed_at = time.time()
            job.error = error
            self.history.append(job)
            self._emit("failed", job)

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        for job in self.queue:
            if job.id == job_id:
                job.status = JobStatus.CANCELLED
                self.queue.remove(job)
                self.history.append(job)
                self._emit("cancelled", job)
                return True
        if job_id in self.active:
            job = self.active[job_id]
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            self.history.append(job)
            del self.active[job_id]
            self._emit("cancelled", job)
            return True
        return False

    def get_status(self, job_id: str) -> Optional[Job]:
        for job in self.queue:
            if job.id == job_id:
                return job
        if job_id in self.active:
            return self.active[job_id]
        for job in self.history:
            if job.id == job_id:
                return job
        return None

    def list_jobs(self, status: Optional[JobStatus] = None) -> List[Job]:
        """List jobs, optionally filtered."""
        all_jobs = self.queue + list(self.active.values()) + self.history
        if status:
            return [j for j in all_jobs if j.status == status]
        return all_jobs

    def update_progress(self, job_id: str, progress: float) -> None:
        """Update job progress."""
        job = self.active.get(job_id)
        if job:
            job.progress = progress
            self._emit("progress", job)

    def on(self, event: str, handler: Callable) -> None:
        """Register job event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def _emit(self, event: str, job: Job) -> None:
        """Emit job event to handlers."""
        for handler in self._handlers.get(event, []):
            try:
                handler(job)
            except Exception:
                pass

    def stats(self) -> Dict[str, Any]:
        return {
            "queued": len(self.queue),
            "active": len(self.active),
            "completed": sum(1 for j in self.history if j.status == JobStatus.COMPLETED),
            "failed": sum(1 for j in self.history if j.status == JobStatus.FAILED),
            "max_concurrent": self.max_concurrent,
        }


class Scheduler:
    """Background scheduler that processes the job queue."""

    def __init__(self, queue: Optional[JobQueue] = None) -> None:
        self.queue = queue or JobQueue()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the scheduler loop."""
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            job = self.queue.dequeue()
            if job:
                asyncio.create_task(self._execute_job(job))
            await asyncio.sleep(1)

    async def _execute_job(self, job: Job) -> None:
        """Execute a single job."""
        try:
            result = {"status": "completed", "message": f"Job {job.name} finished"}
            self.queue.complete(job.id, result)
        except Exception as e:
            self.queue.fail(job.id, str(e))


job_queue = JobQueue()
scheduler = Scheduler(job_queue)
