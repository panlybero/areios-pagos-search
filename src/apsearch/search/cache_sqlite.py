"""SQLite query embedding cache."""

from __future__ import annotations

import random
import sqlite3

import sqlite_vec

from apsearch.config import settings
from apsearch.logging import get_logger
from apsearch.search.cache import cache_key

log = get_logger(__name__)


def get_cached_vector(conn: sqlite3.Connection, query: str, model_sig: str) -> list[float] | None:
    if not settings.query_cache_enabled:
        return None

    h = cache_key(query, model_sig)
    cur = conn.execute(
        """
        UPDATE query_cache
           SET last_accessed = datetime('now'),
               access_count = access_count + 1
         WHERE query_hash = ?
        RETURNING embedding
        """,
        (h,),
    )
    row = cur.fetchone()
    if row and row["embedding"] is not None:
        log.debug("sqlite query cache hit for %r", query)
        raw = row["embedding"]
        # Deserialize from sqlite_vec binary format
        # If float32 array, each float is 4 bytes
        import struct
        n_floats = len(raw) // 4
        return list(struct.unpack(f"{n_floats}f", raw))
    return None


def store_cached_vector(
    conn: sqlite3.Connection, query: str, model_sig: str, vector: list[float]
) -> None:
    if not settings.query_cache_enabled:
        return

    h = cache_key(query, model_sig)
    raw = sqlite_vec.serialize_float32(vector)
    conn.execute(
        """
        INSERT INTO query_cache (query_hash, model_sig, query_text, embedding, created_at, last_accessed)
        VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
        ON CONFLICT (query_hash) DO UPDATE SET
            last_accessed = datetime('now'),
            access_count = query_cache.access_count + 1
        """,
        (h, model_sig, query[:500], raw),
    )
    conn.commit()

    if random.random() < 0.02:
        prune_query_cache(conn)


def prune_query_cache(
    conn: sqlite3.Connection,
    max_entries: int | None = None,
    ttl_days: int | None = None,
) -> int:
    max_entries = max_entries if max_entries is not None else settings.query_cache_max_entries
    ttl_days = ttl_days if ttl_days is not None else settings.query_cache_ttl_days
    deleted = 0

    # 1. TTL
    cur = conn.execute(
        """
        DELETE FROM query_cache
         WHERE datetime(last_accessed) < datetime('now', '-' || ? || ' days')
        """,
        (ttl_days,),
    )
    deleted += cur.rowcount

    # 2. LRU cap
    cur = conn.execute("SELECT count(*) AS total FROM query_cache")
    total = cur.fetchone()["total"]
    if total > max_entries:
        excess = total - max_entries
        cur = conn.execute(
            """
            DELETE FROM query_cache
             WHERE query_hash IN (
                 SELECT query_hash FROM query_cache
                  ORDER BY datetime(last_accessed) ASC
                  LIMIT ?
             )
            """,
            (excess,),
        )
        deleted += cur.rowcount
    conn.commit()
    return deleted


def cache_stats(conn: sqlite3.Connection) -> dict:
    cur = conn.execute(
        """
        SELECT count(*) AS total_entries,
               coalesce(sum(access_count), 0) AS total_lookups,
               min(created_at) AS oldest_entry,
               max(last_accessed) AS newest_access
          FROM query_cache
        """
    )
    res = dict(cur.fetchone())
    res["table_size"] = f"{res['total_entries'] * 3} KB (approx)"
    return res
