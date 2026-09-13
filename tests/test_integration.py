"""Integration tests that need a live Postgres.

Skipped automatically when no database is reachable, so `pytest` stays green on
a bare checkout. Run `docker compose up -d && apsearch db migrate` to enable.

These exist because two real bugs (an AmbiguousParameter in `list_themes` and a
stale denormalised counter) were invisible to the unit tests and only surfaced
when the MCP server was driven over its actual transport.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _db_available() -> bool:
    # A direct connection with a short timeout, rather than the pool: the pool
    # retries on failure and turns "no database" into a 30-second skip.
    try:
        import psycopg

        from apsearch.config import settings

        with psycopg.connect(settings.dsn, connect_timeout=2):
            return True
    except Exception:
        return False


needs_db = pytest.mark.skipif(not _db_available(), reason="no database reachable")


@needs_db
class TestThemes:
    def test_list_themes_unfiltered(self):
        """Regression: `%(p)s IS NULL` raised AmbiguousParameter."""
        from apsearch.search.hybrid import list_themes

        rows = list_themes(None, 5)
        assert isinstance(rows, list)
        assert all({"code", "label", "n_decisions"} <= set(r) for r in rows)

    def test_list_themes_filtered(self):
        from apsearch.search.hybrid import list_themes

        rows = list_themes("αδικ", 5)
        assert all("αδικ" in r["label"].lower() for r in rows)

    def test_counts_are_live_not_stale(self):
        """theme.n_decisions is only refreshed by a thematic crawl.

        Reading it directly under-reports whenever decisions were linked by any
        other path, and agents rank themes by this number.
        """
        from apsearch.db import query
        from apsearch.search.hybrid import list_themes

        linked = query(
            """
            SELECT t.label, count(*) AS n
              FROM decision_theme dt JOIN theme t ON t.code = dt.theme_code
             GROUP BY t.label ORDER BY n DESC LIMIT 1
            """
        )
        if not linked:
            pytest.skip("no theme links in this corpus")
        top = list_themes(None, 1)[0]
        assert top["n_decisions"] == linked[0]["n"]


@needs_db
class TestQueryPipeline:
    def test_prefix_expansion_reaches_postgres(self):
        from apsearch.db import pool
        from apsearch.search.query import prepare

        with pool().connection() as conn, conn.cursor() as cur:
            expr, params = prepare(cur, "αδικοπραξίας του Δημοσίου")
            assert params["qexpr"] == "αδικοπραξ:* & δημοσ:*"
            # Must be a valid tsquery, not just a plausible string.
            cur.execute(f"SELECT {expr} AS q", params)
            assert cur.fetchone()["q"]

    def test_operator_queries_bypass_expansion(self):
        from apsearch.db import pool
        from apsearch.search.query import prepare

        with pool().connection() as conn, conn.cursor() as cur:
            expr, params = prepare(cur, '"λόγος αναιρέσεως"')
            assert "websearch_to_tsquery" in expr
            cur.execute(f"SELECT {expr} AS q", params)

    def test_search_modes_all_execute(self):
        from apsearch.search.hybrid import search

        for mode in ("keyword", "semantic", "hybrid"):
            results = search("αναίρεση", mode=mode, limit=3)
            assert isinstance(results, list)
            for r in results:
                assert r.cd and r.url.startswith("https://")

    def test_filters_are_applied(self):
        from apsearch.search.hybrid import Filters, search

        results = search("αναίρεση", limit=5, filters=Filters(year_to=2000))
        assert all(r.year is None or r.year <= 2000 for r in results)

    def test_empty_query_browses(self):
        from apsearch.search.hybrid import Filters, search

        results = search("", limit=3, filters=Filters(year_from=1990))
        assert len(results) <= 3


@needs_db
class TestMCPTools:
    """Exercise the MCP tool functions the way a client would call them."""

    @pytest.mark.asyncio
    async def test_all_tools_registered(self):
        from apsearch.mcp.server import build_server

        names = {t.name for t in await build_server().list_tools()}
        assert names == {
            "search_decisions", "get_decision", "list_themes", "corpus_stats",
        }

    @pytest.mark.asyncio
    async def test_get_decision_rejects_bad_citation(self):
        from apsearch.mcp.server import build_server

        server = build_server()
        result = await server.call_tool("get_decision", {"citation": "nonsense"})
        payload = result[1] if isinstance(result, tuple) else result
        assert "error" in str(payload)
