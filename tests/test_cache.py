"""Tests for query embedding cache, LRU/TTL cleanup, and anti-spam guards."""

from __future__ import annotations

import pytest
from pgvector import Vector

from apsearch.search.cache import (
    cache_key,
    cache_stats,
    get_cached_vector,
    normalize_query,
    prune_query_cache,
    store_cached_vector,
)


class TestQueryNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("  αδικοπραξία  ", "αδικοπραξια"),
            ("ΑΔΙΚΟΠΡΑΞΊΑ", "αδικοπραξια"),
            ("αγωγή   αδικοπραξίας  ", "αγωγη αδικοπραξιασ"),
            ("λόγος  αναιρέσεως", "λογοσ αναιρεσεωσ"),
        ],
    )
    def test_normalizes_case_accents_and_whitespace(self, raw, expected):
        assert normalize_query(raw) == expected

    def test_cache_keys_converge_for_variants(self):
        sig = "gemini:gemini-embedding-2:768"
        k1 = cache_key("παραγραφή αξιώσεων", sig)
        k2 = cache_key("  ΠΑΡΑΓΡΑΦΉ   ΑΞΙΏΣΕΩΝ  ", sig)
        k3 = cache_key("παραγράφη αξιωσεών", sig)
        assert k1 == k2 == k3

    def test_cache_keys_differ_across_models(self):
        q = "παραγραφή"
        k1 = cache_key(q, "gemini:gemini-embedding-2:768")
        k2 = cache_key(q, "fastembed:multilingual-e5-small:384")
        assert k1 != k2


def _db_available() -> bool:
    try:
        import psycopg

        from apsearch.config import settings

        with psycopg.connect(settings.dsn, connect_timeout=2):
            return True
    except Exception:
        return False


needs_db = pytest.mark.skipif(not _db_available(), reason="no database reachable")


@pytest.mark.integration
@needs_db
class TestQueryCacheDB:
    def test_cache_hit_and_access_counter(self):
        from apsearch.db import pool

        sig = "test:model:768"
        query = "δοκιμή query cache"
        dummy_vec = Vector([0.1] * 768)

        with pool().connection() as conn, conn.cursor() as cur:
            # Clean test slate
            cur.execute("DELETE FROM query_cache WHERE model_sig = %s", (sig,))

            # 1. Miss initially
            assert get_cached_vector(cur, query, sig) is None

            # 2. Store vector
            store_cached_vector(cur, query, sig, dummy_vec)

            # 3. Hit
            hit = get_cached_vector(cur, query, sig)
            assert hit is not None
            assert len(hit.to_list()) == 768

            # 4. Access count increments
            cur.execute(
                "SELECT access_count FROM query_cache WHERE query_hash = %s",
                (cache_key(query, sig),),
            )
            assert cur.fetchone()["access_count"] == 2

            # Clean up
            cur.execute("DELETE FROM query_cache WHERE model_sig = %s", (sig,))

    def test_lru_pruning_removes_oldest_first(self):
        from apsearch.db import pool

        sig = "test:lru:768"
        dummy_vec = Vector([0.05] * 768)

        with pool().connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM query_cache WHERE model_sig = %s", (sig,))

            # Insert 5 queries
            for i in range(5):
                store_cached_vector(cur, f"query {i}", sig, dummy_vec)

            # Prune with cap of 2
            deleted = prune_query_cache(cur, max_entries=2, ttl_days=365)
            assert deleted >= 3

            # Exactly 2 remain
            cur.execute("SELECT count(*) AS n FROM query_cache")
            assert cur.fetchone()["n"] == 2

            stats = cache_stats(cur)
            assert stats["total_entries"] == 2
            assert stats["table_size"]

            cur.execute("DELETE FROM query_cache WHERE model_sig = %s", (sig,))

    def test_query_length_guard_fails_loudly(self):
        from apsearch.search.hybrid import search

        long_query = "α" * 700
        with pytest.raises(ValueError, match="Query is too long"):
            search(long_query)
