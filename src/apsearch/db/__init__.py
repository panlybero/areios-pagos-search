"""Database access: a thin facade over the SQLite + sqlite-vec backend."""

from __future__ import annotations

from apsearch.db.sqlite import connect, ensure_seed_db, fold_greek, init_sqlite_db

__all__ = ["connect", "ensure_seed_db", "fold_greek", "init_sqlite_db", "migrate"]


def migrate() -> None:
    """Create/refresh the schema. Safe to run repeatedly."""
    init_sqlite_db()
