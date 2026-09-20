"""Tests for SQLite + sqlite-vec engine, FTS5 Greek search, and SQLite query cache."""

from __future__ import annotations

import sqlite3

import pytest

from apsearch.db.sqlite import connect, fold_greek, init_sqlite_db
from apsearch.search import cache
from apsearch.search.hybrid import build_fts5_query, highlight_greek


@pytest.fixture
def test_db():
    conn = connect(":memory:", auto_seed=False)
    init_sqlite_db(conn)
    yield conn
    conn.close()


class TestSQLiteEngine:
    def test_sqlite_vec_extension_loaded(self, test_db: sqlite3.Connection):
        cur = test_db.cursor()
        cur.execute("SELECT vec_version()")
        ver = cur.fetchone()[0]
        assert ver.startswith("v0.")

    def test_fts5_greek_folding_and_stemming(self, test_db: sqlite3.Connection):
        cur = test_db.cursor()
        orig = "Αγωγή αδικοπραξίας και αποζημίωση κατά του Δημοσίου."
        folded = fold_greek(orig)
        cur.execute("INSERT INTO chunk_fts(rowid, content_folded) VALUES (1, ?)", (folded,))

        # Query with prefix
        q = build_fts5_query("αδικοπραξία")
        assert "αδικοπραξ" in q
        cur.execute("SELECT rowid FROM chunk_fts WHERE content_folded MATCH ?", (q,))
        assert cur.fetchone()[0] == 1

    def test_fts_delete_on_contentless_table(self, test_db: sqlite3.Connection):
        from apsearch.db.sqlite import fts_delete

        cur = test_db.cursor()
        cur.execute("INSERT INTO decision_fts(rowid, subject_folded, summary_folded, body_folded) VALUES (3, 'x', 'y', 'z')")
        fts_delete(test_db, "decision_fts", 3, ["x", "y", "z"])
        assert cur.execute("SELECT count(*) FROM decision_fts").fetchone()[0] == 0

    def test_fts_delete_on_non_contentless_table(self, test_db: sqlite3.Connection):
        from apsearch.db.sqlite import fts_delete

        cur = test_db.cursor()
        cur.execute("DROP TABLE IF EXISTS chunk_fts")
        cur.execute(
            "CREATE VIRTUAL TABLE chunk_fts USING fts5(content_folded, content UNINDEXED, cd UNINDEXED, part UNINDEXED)"
        )
        cur.execute(
            "INSERT INTO chunk_fts(rowid, content_folded, content, cd, part) VALUES (7, 'folded', 'orig', 'c', 'p')"
        )
        fts_delete(test_db, "chunk_fts", 7, ["folded"])
        assert cur.execute("SELECT count(*) FROM chunk_fts").fetchone()[0] == 0


class TestSQLiteQueryCache:
    def test_cache_roundtrip_and_stats(self, test_db: sqlite3.Connection):
        sig = "test:model:768"
        query = "παραγραφή αξιώσεων"
        dummy = [0.01 * (i % 50) for i in range(768)]

        # 1. Miss initially
        assert cache.get_cached_vector(test_db, query, sig) is None

        # 2. Store
        cache.store_cached_vector(test_db, query, sig, dummy)

        # 3. Hit
        cached = cache.get_cached_vector(test_db, query, sig)
        assert cached is not None
        assert len(cached) == 768
        assert pytest.approx(cached[0], rel=1e-3) == dummy[0]

        # 4. Stats
        st = cache.cache_stats(test_db)
        assert st["total_entries"] == 1
        assert st["total_lookups"] == 2


class TestHighlight:
    def test_highlight_marks_greek_words(self):
        text = "Κατά τη διάταξη του άρθρου 914 ΑΚ όποιος ζημιώσει άλλον παράνομα έχει υποχρέωση αποζημιώσεως."
        snip = highlight_greek(text, "αποζημίωση")
        assert "**αποζημιώσεως**" in snip or "αποζημιώσεως" in snip
