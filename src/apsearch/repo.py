"""Persistence layer: all SQL that writes lives here."""

from __future__ import annotations

from collections.abc import Iterable

from psycopg.rows import dict_row

from apsearch import repo_sqlite
from apsearch.config import settings
from apsearch.crawler.discover import CATEGORIES, CHAMBERS
from apsearch.crawler.fetch import content_hash, decision_url
from apsearch.crawler.parse import Decision, DecisionRef, Theme, fold_greek
from apsearch.db import pool
from apsearch.logging import get_logger

log = get_logger(__name__)

_CATEGORY_BY_LABEL = {fold_greek(v): k for k, v in CATEGORIES.items()}
_CHAMBER_BY_LABEL = {fold_greek(v): k for k, v in CHAMBERS.items()}


def category_id(label: str | None) -> int | None:
    return _CATEGORY_BY_LABEL.get(fold_greek(label or "")) if label else None


def chamber_id(label: str | None) -> int | None:
    return _CHAMBER_BY_LABEL.get(fold_greek(label or "")) if label else None


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------


def upsert_themes(themes: Iterable[Theme]) -> int:
    if settings.db_backend == "sqlite":
        return repo_sqlite.upsert_themes(themes)
    rows = [(t.code, t.label, t.slug) for t in themes]
    if not rows:
        return 0
    with pool().connection() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO theme (code, label, slug) VALUES (%s, %s, %s)
            ON CONFLICT (code) DO UPDATE
              SET label = EXCLUDED.label, slug = EXCLUDED.slug
            """,
            rows,
        )
    return len(rows)


def link_theme(code: int, cds: Iterable[str]) -> int:
    """Attach a subject heading to decisions we already hold."""
    if settings.db_backend == "sqlite":
        return repo_sqlite.link_theme(code, cds)
    cds = list(cds)
    if not cds:
        return 0
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO decision_theme (cd, theme_code)
            SELECT d.cd, %s FROM decision d WHERE d.cd = ANY(%s)
            ON CONFLICT DO NOTHING
            """,
            (code, cds),
        )
        n = cur.rowcount
        cur.execute(
            """
            UPDATE theme
               SET n_decisions = (SELECT count(*) FROM decision_theme WHERE theme_code = %s),
                   crawled_at = now()
             WHERE code = %s
            """,
            (code, code),
        )
    return n


def themes_needing_crawl(max_age_days: int = 30) -> list[dict]:
    if settings.db_backend == "sqlite":
        return repo_sqlite.themes_needing_crawl(max_age_days)
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT code, label FROM theme
             WHERE crawled_at IS NULL
                OR crawled_at < now() - make_interval(days => %s)
             ORDER BY crawled_at NULLS FIRST, code
            """,
            (max_age_days,),
        )
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Discovery queue
# ---------------------------------------------------------------------------


def known_cds(cds: Iterable[str]) -> set[str]:
    if settings.db_backend == "sqlite":
        return repo_sqlite.known_cds(cds)
    cds = list(cds)
    if not cds:
        return set()
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT cd FROM decision WHERE cd = ANY(%s)", (cds,))
        return {r["cd"] for r in cur.fetchall()}


def enqueue(refs: Iterable[DecisionRef]) -> int:
    """Queue newly discovered decisions whose text we do not have yet."""
    if settings.db_backend == "sqlite":
        return repo_sqlite.enqueue(refs)
    refs = list(refs)
    if not refs:
        return 0
    rows = [
        (r.cd, r.number, r.year, r.category, r.chamber)
        for r in refs
    ]
    with pool().connection() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO fetch_queue (cd, number, year, category, chamber)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (cd) DO NOTHING
            """,
            rows,
        )
        # Anything already stored does not need fetching again.
        cur.execute(
            "DELETE FROM fetch_queue q USING decision d WHERE d.cd = q.cd"
        )
    return len(rows)


def take_queue(limit: int) -> list[dict]:
    if settings.db_backend == "sqlite":
        return repo_sqlite.take_queue(limit)
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT cd, number, year, category, chamber, attempts
              FROM fetch_queue
             WHERE attempts < 5
             ORDER BY year DESC NULLS LAST, number DESC NULLS LAST
             LIMIT %s
            """,
            (limit,),
        )
        return cur.fetchall()


def dequeue(cd: str) -> None:
    if settings.db_backend == "sqlite":
        return repo_sqlite.dequeue(cd)
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM fetch_queue WHERE cd = %s", (cd,))


def mark_queue_error(cd: str, error: str) -> None:
    if settings.db_backend == "sqlite":
        return repo_sqlite.mark_queue_error(cd, error)
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE fetch_queue
               SET attempts = attempts + 1, last_error = %s
             WHERE cd = %s
            """,
            (error[:2000], cd),
        )


def queue_depth() -> int:
    if settings.db_backend == "sqlite":
        return repo_sqlite.queue_depth()
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM fetch_queue WHERE attempts < 5")
        return cur.fetchone()["n"]


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def upsert_decision(dec: Decision) -> str:
    """Insert or update a decision. Returns 'new' | 'changed' | 'unchanged'."""
    if settings.db_backend == "sqlite":
        return repo_sqlite.upsert_decision(dec)
    chash = content_hash(dec.body)
    url = decision_url(dec.cd, dec.number, dec.year)
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO decision (
                cd, number, year, category_id, chamber_id, category, chamber,
                subject, summary, body, source_url, content_hash, last_changed
            ) VALUES (
                %(cd)s, %(number)s, %(year)s, %(category_id)s, %(chamber_id)s,
                %(category)s, %(chamber)s, %(subject)s, %(summary)s, %(body)s,
                %(source_url)s, %(content_hash)s, now()
            )
            ON CONFLICT (cd) DO UPDATE SET
                number       = EXCLUDED.number,
                year         = EXCLUDED.year,
                category_id  = EXCLUDED.category_id,
                chamber_id   = EXCLUDED.chamber_id,
                category     = EXCLUDED.category,
                chamber      = EXCLUDED.chamber,
                subject      = EXCLUDED.subject,
                summary      = EXCLUDED.summary,
                body         = EXCLUDED.body,
                source_url   = EXCLUDED.source_url,
                content_hash = EXCLUDED.content_hash,
                last_fetched = now(),
                last_changed = CASE
                    WHEN decision.content_hash IS DISTINCT FROM EXCLUDED.content_hash
                    THEN now() ELSE decision.last_changed END
            RETURNING (xmax = 0) AS inserted,
                      (content_hash = %(content_hash)s) AS same_hash
            """,
            {
                "cd": dec.cd,
                "number": dec.number,
                "year": dec.year,
                "category_id": category_id(dec.category),
                "chamber_id": chamber_id(dec.chamber),
                "category": dec.category,
                "chamber": dec.chamber,
                "subject": dec.subject,
                "summary": dec.summary,
                "body": dec.body,
                "source_url": url,
                "content_hash": chash,
            },
        )
        row = cur.fetchone()

        # Subject headings parsed off the decision page itself; match them to
        # the controlled vocabulary by folded label.
        if dec.subjects:
            cur.execute(
                """
                INSERT INTO decision_theme (cd, theme_code)
                SELECT %s, t.code FROM theme t
                 WHERE lower(translate(t.label,
                        'άέήίόύώϊϋΐΰςΆΈΉΊΌΎΏ',
                        'αεηιουωιυιυσαεηιουω')) = ANY(%s)
                ON CONFLICT DO NOTHING
                """,
                (dec.cd, [fold_greek(s) for s in dec.subjects]),
            )

    if row["inserted"]:
        return "new"
    return "unchanged" if row["same_hash"] else "changed"


def decision_count() -> int:
    if settings.db_backend == "sqlite":
        return repo_sqlite.decision_count()
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM decision")
        return cur.fetchone()["n"]


# ---------------------------------------------------------------------------
# Crawl bookkeeping
# ---------------------------------------------------------------------------


def record_partition(
    year: int, category_id_: int, chamber_id_: int, n_found: int, truncated: bool,
    status: str = "done", error: str | None = None,
) -> None:
    if settings.db_backend == "sqlite":
        return repo_sqlite.record_partition(
            year, category_id_, chamber_id_, n_found, truncated, status, error
        )
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO crawl_partition
                (year, category_id, chamber_id, status, n_found, truncated, error, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (year, category_id, chamber_id) DO UPDATE SET
                status = EXCLUDED.status, n_found = EXCLUDED.n_found,
                truncated = EXCLUDED.truncated, error = EXCLUDED.error,
                updated_at = now()
            """,
            (year, category_id_, chamber_id_, status, n_found, truncated, error),
        )


def year_is_done(year: int) -> bool:
    if settings.db_backend == "sqlite":
        return repo_sqlite.year_is_done(year)
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT status FROM crawl_partition
             WHERE year = %s AND category_id = 6 AND chamber_id = 1
            """,
            (year,),
        )
        row = cur.fetchone()
        return bool(row and row["status"] == "done")


def start_run(kind: str) -> int:
    if settings.db_backend == "sqlite":
        return repo_sqlite.start_run(kind)
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO crawl_run (kind) VALUES (%s) RETURNING id", (kind,))
        return cur.fetchone()["id"]


def finish_run(run_id: int, **counts) -> None:
    if settings.db_backend == "sqlite":
        return repo_sqlite.finish_run(run_id, **counts)
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE crawl_run SET finished_at = now(),
                   n_discovered = %s, n_fetched = %s, n_changed = %s,
                   n_errors = %s, notes = %s
             WHERE id = %s
            """,
            (
                counts.get("n_discovered", 0),
                counts.get("n_fetched", 0),
                counts.get("n_changed", 0),
                counts.get("n_errors", 0),
                counts.get("notes"),
                run_id,
            ),
        )


def stats() -> dict:
    if settings.db_backend == "sqlite":
        return repo_sqlite.stats()
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              (SELECT count(*) FROM decision)                              AS decisions,
              (SELECT count(*) FROM decision WHERE summary IS NOT NULL)    AS with_summary,
              (SELECT count(*) FROM chunk)                                 AS chunks,
              (SELECT count(*) FROM chunk WHERE embedding IS NOT NULL)     AS embedded,
              (SELECT count(*) FROM theme)                                 AS themes,
              (SELECT count(*) FROM decision_theme)                        AS theme_links,
              (SELECT count(*) FROM fetch_queue WHERE attempts < 5)        AS queued,
              (SELECT min(year) FROM decision)                             AS first_year,
              (SELECT max(year) FROM decision)                             AS last_year,
              (SELECT count(*) FROM decision
                WHERE indexed_hash IS DISTINCT FROM content_hash)          AS pending_index
            """
        )
        return cur.fetchone()
