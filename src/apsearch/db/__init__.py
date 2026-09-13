"""Database access: connection pool, migrations, vector index management."""

from __future__ import annotations

import threading
from importlib import resources
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from apsearch.config import settings

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def pool() -> ConnectionPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ConnectionPool(
                settings.dsn,
                min_size=1,
                max_size=8,
                kwargs={"row_factory": dict_row},
                open=True,
            )
    return _pool


def connect() -> psycopg.Connection:
    """A standalone connection (used for long-running maintenance statements)."""
    return psycopg.connect(settings.dsn, row_factory=dict_row, autocommit=True)


def query(sql: str, params: Any = None) -> list[dict]:
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        if cur.description is None:
            return []
        return cur.fetchall()


def execute(sql: str, params: Any = None) -> int:
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


def _schema_sql() -> str:
    return resources.files("apsearch.db").joinpath("schema.sql").read_text(encoding="utf-8")


def migrate(with_vector_index: bool = False) -> None:
    """Create/refresh the schema. Safe to run repeatedly."""
    dim = settings.embed_dim
    with connect() as conn, conn.cursor() as cur:
        cur.execute(_schema_sql())

        # Pin the embedding column to the configured dimension. Changing the
        # model therefore requires re-embedding, which is the correct behaviour.
        cur.execute(
            """
            SELECT atttypmod AS typmod
            FROM pg_attribute
            WHERE attrelid = 'chunk'::regclass AND attname = 'embedding'
            """
        )
        row = cur.fetchone()
        current = row["typmod"] if row else -1
        if current != dim:
            cur.execute("SELECT count(*) AS n FROM chunk WHERE embedding IS NOT NULL")
            if cur.fetchone()["n"]:
                raise RuntimeError(
                    f"chunk.embedding has dimension {current} but settings request {dim}. "
                    "Run `apsearch index reset` to drop existing embeddings first."
                )
            cur.execute("DROP INDEX IF EXISTS chunk_embedding_idx")
            cur.execute(f"ALTER TABLE chunk ALTER COLUMN embedding TYPE vector({dim})")

        if with_vector_index:
            create_vector_index(cur)


def create_vector_index(cur=None) -> None:
    """Build the HNSW index.

    Deliberately separate from `migrate`: on a cold backfill it is far cheaper
    to bulk-load every embedding first and build the index once at the end.
    """
    sql = """
        CREATE INDEX IF NOT EXISTS chunk_embedding_idx
        ON chunk USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """
    if cur is not None:
        cur.execute(sql)
        return
    with connect() as conn, conn.cursor() as c:
        c.execute("SET maintenance_work_mem = '512MB'")
        c.execute(sql)


def has_vector_index() -> bool:
    rows = query(
        "SELECT 1 FROM pg_class WHERE relname = 'chunk_embedding_idx' AND relkind = 'i'"
    )
    return bool(rows)
