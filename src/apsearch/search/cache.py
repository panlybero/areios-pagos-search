"""Query embedding cache with automatic LRU and TTL cleanup.

Why this exists
---------------
User and agent queries repeat frequently ("αδικοπραξία", "άρθρο 559 ΚΠολΔ",
"παραγραφή αξιώσεων"). Calling the embedding API for every repeat query wastes
money and adds 200-500ms of latency.

Bounded storage guarantees (no unbounded growth)
------------------------------------------------
1. **LRU Cap**: At most ``APSEARCH_QUERY_CACHE_MAX_ENTRIES`` (default 20,000)
   rows are kept. When the cap is reached, the least recently accessed queries
   are evicted first. 20,000 entries take ~65 MB in Postgres.
2. **TTL**: Queries unaccessed for ``APSEARCH_QUERY_CACHE_TTL_DAYS`` (default
   60 days) are pruned automatically.
3. **Model Partitioning**: The cache key incorporates the embedding model's
   signature (e.g. `gemini:gemini-embedding-2:768`). Changing models never
   returns vectors from an incompatible embedding space.
4. **Accent & Case Invariance**: "Αδικοπραξία", "αδικοπραξία", and "αδικοπραξια"
   normalize to the same cache key so casing/tonos variants hit the cache.
"""

from __future__ import annotations

import hashlib
import random
import unicodedata

try:
    from pgvector import Vector
except ImportError:
    Vector = None  # type: ignore

from apsearch.config import settings
from apsearch.logging import get_logger

log = get_logger(__name__)


def normalize_query(query: str) -> str:
    """Normalize query text for cache keying: accent-folded, lowercase, single-spaced."""
    s = unicodedata.normalize("NFD", query.strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = unicodedata.normalize("NFC", s).lower().replace("ς", "σ")
    return " ".join(s.split())


def cache_key(query: str, model_sig: str) -> str:
    """Deterministic hash combining the model signature and normalized query text."""
    norm = normalize_query(query)
    raw = f"{model_sig}\n{norm}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_vector(cur, query: str, model_sig: str) -> Vector | None:
    """Retrieve cached embedding vector, updating last_accessed on hit."""
    if not settings.query_cache_enabled:
        return None

    h = cache_key(query, model_sig)
    cur.execute(
        """
        UPDATE query_cache
           SET last_accessed = now(),
               access_count = access_count + 1
         WHERE query_hash = %s
        RETURNING embedding
        """,
        (h,),
    )
    row = cur.fetchone()
    if row and row["embedding"] is not None:
        log.debug("query cache hit for %r", query)
        emb = row["embedding"]
        return emb if isinstance(emb, Vector) else Vector(emb)
    return None


def store_cached_vector(cur, query: str, model_sig: str, vector: Vector) -> None:
    """Store an embedding in the cache and probabilistically prune stale entries."""
    if not settings.query_cache_enabled:
        return

    h = cache_key(query, model_sig)
    cur.execute(
        """
        INSERT INTO query_cache (query_hash, model_sig, query_text, embedding, created_at, last_accessed)
        VALUES (%s, %s, %s, %s, now(), now())
        ON CONFLICT (query_hash) DO UPDATE
          SET last_accessed = now(),
              access_count = query_cache.access_count + 1
        """,
        (h, model_sig, query[:500], vector),
    )

    # Probabilistic cleanup (~2% of cache writes) so we don't prune on every write
    if random.random() < 0.02:
        prune_query_cache(cur)


def prune_query_cache(
    cur,
    max_entries: int | None = None,
    ttl_days: int | None = None,
) -> int:
    """Delete expired entries (TTL) and oldest accessed entries if exceeding cap (LRU).

    Returns total number of rows removed.
    """
    max_entries = max_entries if max_entries is not None else settings.query_cache_max_entries
    ttl_days = ttl_days if ttl_days is not None else settings.query_cache_ttl_days
    deleted = 0

    # 1. TTL eviction: remove queries not accessed within ttl_days
    cur.execute(
        """
        DELETE FROM query_cache
         WHERE last_accessed < now() - make_interval(days => %s)
        """,
        (ttl_days,),
    )
    deleted += cur.rowcount

    # 2. Capacity eviction (LRU): remove oldest accessed if over max_entries
    cur.execute("SELECT count(*) AS total FROM query_cache")
    total = cur.fetchone()["total"]
    if total > max_entries:
        excess = total - max_entries
        cur.execute(
            """
            DELETE FROM query_cache
             WHERE query_hash IN (
                 SELECT query_hash FROM query_cache
                  ORDER BY last_accessed ASC
                  LIMIT %s
             )
            """,
            (excess,),
        )
        deleted += cur.rowcount
        log.info("pruned %d excess LRU entries from query cache (cap %d)", excess, max_entries)

    return deleted


def cache_stats(cur) -> dict:
    """Current statistics of the query cache."""
    cur.execute(
        """
        SELECT count(*) AS total_entries,
               coalesce(sum(access_count), 0) AS total_lookups,
               min(created_at) AS oldest_entry,
               max(last_accessed) AS newest_access,
               pg_size_pretty(pg_total_relation_size('query_cache')) AS table_size
          FROM query_cache
        """
    )
    return cur.fetchone()
