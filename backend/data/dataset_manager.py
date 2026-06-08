"""Dataset management - browse, stats, and versioning."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


class DatasetCard:
    """Metadata for a managed dataset."""

    def __init__(
        self,
        name: str,
        dataset_type: str,
        split: str = "train",
        num_samples: int = 0,
        description: str = "",
        features: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.dataset_type = dataset_type
        self.split = split
        self.num_samples = num_samples
        self.description = description
        self.features = features or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dataset_type": self.dataset_type,
            "split": self.split,
            "num_samples": self.num_samples,
            "description": self.description,
            "features": self.features,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DatasetCard":
        return cls(
            name=data["name"],
            dataset_type=data.get("dataset_type", "text"),
            split=data.get("split", "train"),
            num_samples=data.get("num_samples", 0),
            description=data.get("description", ""),
            features=data.get("features", []),
        )


class DatasetManager:
    """Manage local datasets with statistics."""

    def __init__(self, storage_dir: str = "./dataset_store") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all managed datasets."""
        datasets = []
        for ds_dir in self.storage_dir.iterdir():
            if ds_dir.is_dir():
                card_path = ds_dir / "dataset_card.json"
                if card_path.exists():
                    with open(card_path, "r", encoding="utf-8") as f:
                        datasets.append(json.load(f))
        return datasets

    def get_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        for ds in self.list_datasets():
            if ds["name"] == name:
                return ds
        return None

    def register_dataset(
        self,
        name: str,
        dataset_type: str,
        source_path: str,
        description: str = "",
    ) -> str:
        """Register an existing dataset directory."""
        ds_dir = self.storage_dir / name
        ds_dir.mkdir(parents=True, exist_ok=True)

        src = Path(source_path)
        num_samples = 0
        features = []

        if src.is_dir():
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, ds_dir / f.name)
            if src.suffix == ".json":
                with open(src, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    num_samples = len(data) if isinstance(data, list) else 1
                    features = list(data[0].keys()) if isinstance(data, list) and data else []
        elif src.is_file():
            shutil.copy2(src, ds_dir / src.name)

        card = DatasetCard(
            name=name,
            dataset_type=dataset_type,
            num_samples=num_samples,
            description=description,
            features=features,
        )
        card.save(str(ds_dir / "dataset_card.json"))
        return str(ds_dir)

    def compute_stats(self, name: str) -> Dict[str, Any]:
        """Compute statistics for a dataset."""
        ds_dir = self.storage_dir / name
        if not ds_dir.exists():
            return {"error": f"Dataset '{name}' not found"}

        stats = {
            "name": name,
            "size_mb": 0,
            "num_files": 0,
            "file_types": {},
            "num_samples": 0,
        }

        total_bytes = 0
        for f in ds_dir.rglob("*"):
            if f.is_file() and f.suffix != ".json":
                total_bytes += f.stat().st_size
                stats["num_files"] += 1
                ext = f.suffix or "no_ext"
                stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1

        stats["size_mb"] = round(total_bytes / (1024 * 1024), 2)

        card_path = ds_dir / "dataset_card.json"
        if card_path.exists():
            with open(card_path, "r") as f:
                card = json.load(f)
                stats["num_samples"] = card.get("num_samples", 0)

        return stats

    def delete_dataset(self, name: str) -> None:
        """Delete a managed dataset."""
        path = self.storage_dir / name
        if path.exists():
            shutil.rmtree(path)


dataset_manager = DatasetManager()

# Patch DatasetCard.save if missing
if not hasattr(DatasetCard, "save"):
    def _save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    DatasetCard.save = _save
