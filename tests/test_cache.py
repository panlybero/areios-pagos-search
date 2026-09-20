"""Tests for query embedding cache, LRU/TTL cleanup, and anti-spam guards."""

from __future__ import annotations

import sqlite3

import pytest

from apsearch.db.sqlite import connect, init_sqlite_db
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


@pytest.fixture
def cache_db() -> sqlite3.Connection:
    conn = connect(":memory:", auto_seed=False)
    init_sqlite_db(conn)
    yield conn
    conn.close()


class TestQueryCacheDB:
    def test_cache_hit_and_access_counter(self, cache_db: sqlite3.Connection):
        sig = "test:model:768"
        query = "δοκιμή query cache"
        dummy_vec = [0.1] * 768

        # 1. Miss initially
        assert get_cached_vector(cache_db, query, sig) is None

        # 2. Store vector
        store_cached_vector(cache_db, query, sig, dummy_vec)

        # 3. Hit
        hit = get_cached_vector(cache_db, query, sig)
        assert hit is not None
        assert len(hit) == 768

        # 4. Access count increments
        cur = cache_db.execute(
            "SELECT access_count FROM query_cache WHERE query_hash = ?",
            (cache_key(query, sig),),
        )
        assert cur.fetchone()["access_count"] == 2

    def test_lru_pruning_removes_oldest_first(self, cache_db: sqlite3.Connection):
        sig = "test:lru:768"
        dummy_vec = [0.05] * 768

        for i in range(5):
            store_cached_vector(cache_db, f"query {i}", sig, dummy_vec)

        deleted = prune_query_cache(cache_db, max_entries=2, ttl_days=365)
        assert deleted >= 3

        cur = cache_db.execute("SELECT count(*) AS n FROM query_cache")
        assert cur.fetchone()["n"] == 2

        stats = cache_stats(cache_db)
        assert stats["total_entries"] == 2
        assert stats["table_size"]

    def test_query_length_guard_fails_loudly(self):
        from apsearch.search.hybrid import search

        long_query = "α" * 700
        with pytest.raises(ValueError, match="Query is too long"):
            search(long_query)
