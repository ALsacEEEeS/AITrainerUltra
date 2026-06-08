"""Checkpoint management utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class CheckpointManager:
    """Manages training checkpoints with disk space management."""

    def __init__(self, base_dir: str = "./checkpoints", max_checkpoints: int = 5) -> None:
        self.base_dir = Path(base_dir)
        self.max_checkpoints = max_checkpoints
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, data: Dict[str, Any]) -> str:
        """Save a checkpoint."""
        ckpt_dir = self.base_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        meta_path = ckpt_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "name": name,
                "checkpoints": data.get("checkpoints", {}),
                "metrics": data.get("metrics", {}),
            }, f, indent=2)

        self._cleanup_old()
        return str(ckpt_dir)

    def load(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint."""
        ckpt_dir = self.base_dir / name
        meta_path = ckpt_dir / "metadata.json"

        if not meta_path.exists():
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints = []
        for ckpt_dir in self.base_dir.iterdir():
            if ckpt_dir.is_dir():
                meta = self.load(ckpt_dir.name)
                if meta:
                    checkpoints.append(meta)
        return checkpoints

    def delete(self, name: str) -> None:
        """Delete a checkpoint."""
        ckpt_dir = self.base_dir / name
        if ckpt_dir.exists():
            import shutil
            shutil.rmtree(ckpt_dir)

    def _cleanup_old(self) -> None:
        """Remove oldest checkpoints if exceeding max."""
        checkpoints = sorted(
            self.base_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
        )
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest)
