"""SQLite database backend for persistent storage.

Replaces JSON file storage for experiments, model store, and dataset store.
Provides async-compatible CRUD operations with connection pooling.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


class Database:
    """Thread-safe SQLite database with auto-migration."""

    def __init__(self, db_path: str = "./data/aitrainer.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = self.conn
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                name TEXT PRIMARY KEY,
                model_type TEXT NOT NULL,
                model_name TEXT DEFAULT '',
                hyperparameters TEXT DEFAULT '{}',
                dataset_info TEXT DEFAULT '{}',
                metrics TEXT DEFAULT '{}',
                final_metrics TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                status TEXT DEFAULT 'created',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                duration REAL DEFAULT 0.0,
                artifact_path TEXT,
                notes TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS models (
                name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                version TEXT DEFAULT '1.0.0',
                description TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                license TEXT DEFAULT 'MIT',
                framework TEXT DEFAULT 'pytorch',
                source_path TEXT DEFAULT '',
                created_at REAL NOT NULL,
                PRIMARY KEY (name, version)
            );

            CREATE TABLE IF NOT EXISTS datasets (
                name TEXT PRIMARY KEY,
                dataset_type TEXT NOT NULL,
                split TEXT DEFAULT 'train',
                num_samples INTEGER DEFAULT 0,
                description TEXT DEFAULT '',
                features TEXT DEFAULT '[]',
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model_type TEXT DEFAULT 'llm',
                status TEXT DEFAULT 'queued',
                progress REAL DEFAULT 0.0,
                config TEXT DEFAULT '{}',
                error TEXT,
                result TEXT,
                priority INTEGER DEFAULT 0,
                tags TEXT DEFAULT '[]',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                name TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                epoch INTEGER DEFAULT 0,
                loss REAL DEFAULT 0.0,
                model_type TEXT DEFAULT '',
                size_bytes INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                PRIMARY KEY (name, output_dir)
            );

            CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
            CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        """)
        conn.commit()

    # ─── Experiments ──────────────────────────────────────────────────

    def save_experiment(self, exp_data: Dict[str, Any]) -> None:
        """Insert or replace an experiment."""
        import time
        now = time.time()
        self.conn.execute("""
            INSERT OR REPLACE INTO experiments
            (name, model_type, model_name, hyperparameters, dataset_info,
             metrics, final_metrics, tags, status, created_at, updated_at,
             duration, artifact_path, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exp_data["name"],
            exp_data.get("model_type", ""),
            exp_data.get("model_name", ""),
            json.dumps(exp_data.get("hyperparameters", {})),
            json.dumps(exp_data.get("dataset_info", {})),
            json.dumps(exp_data.get("metrics", {})),
            json.dumps(exp_data.get("final_metrics", {})),
            json.dumps(exp_data.get("tags", [])),
            exp_data.get("status", "created"),
            exp_data.get("created_at", now),
            exp_data.get("updated_at", now),
            exp_data.get("duration", 0.0),
            exp_data.get("artifact_path"),
            exp_data.get("notes", ""),
        ))
        self.conn.commit()

    def get_experiment(self, name: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE name = ?", (name,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_experiments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List experiments, optionally filtered by status."""
        if status:
            rows = self.conn.execute(
                "SELECT * FROM experiments WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def delete_experiment(self, name: str) -> None:
        """Delete an experiment."""
        self.conn.execute("DELETE FROM experiments WHERE name = ?", (name,))
        self.conn.commit()

    # ─── Models ────────────────────────────────────────────────────────

    def save_model(self, model_data: Dict[str, Any]) -> None:
        """Insert or replace a model."""
        import time
        self.conn.execute("""
            INSERT OR REPLACE INTO models
            (name, model_type, version, description, tags, license, framework, source_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_data["name"],
            model_data.get("model_type", ""),
            model_data.get("version", "1.0.0"),
            model_data.get("description", ""),
            json.dumps(model_data.get("tags", [])),
            model_data.get("license", "MIT"),
            model_data.get("framework", "pytorch"),
            model_data.get("source_path", ""),
            model_data.get("created_at", time.time()),
        ))
        self.conn.commit()

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models."""
        rows = self.conn.execute(
            "SELECT * FROM models ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def delete_model(self, name: str, version: Optional[str] = None) -> None:
        """Delete a model."""
        if version:
            self.conn.execute(
                "DELETE FROM models WHERE name = ? AND version = ?", (name, version)
            )
        else:
            self.conn.execute("DELETE FROM models WHERE name = ?", (name,))
        self.conn.commit()

    # ─── Jobs ──────────────────────────────────────────────────────────

    def save_job(self, job_data: Dict[str, Any]) -> None:
        """Insert or replace a job."""
        import time
        now = time.time()
        self.conn.execute("""
            INSERT OR REPLACE INTO jobs
            (id, name, model_type, status, progress, config, error, result, priority, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_data["id"],
            job_data.get("name", ""),
            job_data.get("model_type", "llm"),
            job_data.get("status", "queued"),
            job_data.get("progress", 0.0),
            json.dumps(job_data.get("config", {})),
            job_data.get("error"),
            job_data.get("result"),
            job_data.get("priority", 0),
            json.dumps(job_data.get("tags", [])),
            job_data.get("created_at", now),
            job_data.get("updated_at", now),
        ))
        self.conn.commit()

    def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List jobs, optionally filtered by status."""
        if status:
            rows = self.conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM jobs ORDER BY priority DESC, created_at DESC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ─── Helpers ───────────────────────────────────────────────────────

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a sqlite3.Row to a dict with JSON field parsing."""
        d = dict(row)
        for key in ("hyperparameters", "dataset_info", "metrics", "final_metrics", "tags", "config", "features"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def get_stats(self) -> Dict[str, Any]:
        result = {"experiments": 0, "models": 0, "datasets": 0, "jobs": 0}
        for table in result:
            row = self.conn.execute(
                f"SELECT COUNT(*) as c FROM {table}"
            ).fetchone()
            result[table] = row["c"] if row else 0
        return result


# Global database instance
db = Database()
