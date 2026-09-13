"""SQLite repository implementation."""

from __future__ import annotations

from collections.abc import Iterable

from apsearch.crawler.discover import CATEGORIES, CHAMBERS
from apsearch.crawler.fetch import content_hash, decision_url
from apsearch.crawler.parse import Decision, DecisionRef, Theme
from apsearch.db.sqlite import connect, fold_greek
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
    rows = [(t.code, t.label, t.slug) for t in themes]
    if not rows:
        return 0
    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO theme (code, label, slug) VALUES (?, ?, ?)
            ON CONFLICT (code) DO UPDATE
              SET label = excluded.label, slug = excluded.slug
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def link_theme(code: int, cds: Iterable[str]) -> int:
    cds = list(cds)
    if not cds:
        return 0
    with connect() as conn:
        placeholders = ",".join("?" * len(cds))
        conn.execute(
            f"""
            INSERT OR IGNORE INTO decision_theme (cd, theme_code)
            SELECT cd, ? FROM decision WHERE cd IN ({placeholders})
            """,
            [code, *cds],
        )
        cur = conn.execute(
            """
            UPDATE theme
               SET n_decisions = (SELECT count(*) FROM decision_theme WHERE theme_code = ?),
                   crawled_at = datetime('now')
             WHERE code = ?
            """,
            (code, code),
        )
        conn.commit()
        return cur.rowcount


def themes_needing_crawl(max_age_days: int = 30) -> list[dict]:
    with connect() as conn:
        cur = conn.execute(
            """
            SELECT code, label FROM theme
             WHERE crawled_at IS NULL
                OR datetime(crawled_at) < datetime('now', '-' || ? || ' days')
             ORDER BY crawled_at IS NOT NULL, code
            """,
            (max_age_days,),
        )
        return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Discovery queue
# ---------------------------------------------------------------------------


def known_cds(cds: Iterable[str]) -> set[str]:
    cds = list(cds)
    if not cds:
        return set()
    with connect() as conn:
        placeholders = ",".join("?" * len(cds))
        cur = conn.execute(f"SELECT cd FROM decision WHERE cd IN ({placeholders})", cds)
        return {r["cd"] for r in cur.fetchall()}


def enqueue(refs: Iterable[DecisionRef]) -> int:
    refs = list(refs)
    if not refs:
        return 0
    rows = [(r.cd, r.number, r.year, r.category, r.chamber) for r in refs]
    with connect() as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO fetch_queue (cd, number, year, category, chamber)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        # Delete anything already stored
        conn.execute("DELETE FROM fetch_queue WHERE cd IN (SELECT cd FROM decision)")
        conn.commit()
    return len(rows)


def take_queue(limit: int) -> list[dict]:
    with connect() as conn:
        cur = conn.execute(
            """
            SELECT cd, number, year, category, chamber, attempts
              FROM fetch_queue
             WHERE attempts < 5
             ORDER BY year DESC, number DESC
             LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def dequeue(cd: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM fetch_queue WHERE cd = ?", (cd,))
        conn.commit()


def mark_queue_error(cd: str, error: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            UPDATE fetch_queue
               SET attempts = attempts + 1, last_error = ?
             WHERE cd = ?
            """,
            (str(error)[:2000], cd),
        )
        conn.commit()


def queue_depth() -> int:
    with connect() as conn:
        cur = conn.execute("SELECT count(*) AS n FROM fetch_queue WHERE attempts < 5")
        return cur.fetchone()["n"]


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def upsert_decision(dec: Decision) -> str:
    chash = content_hash(dec.body)
    url = decision_url(dec.cd, dec.number, dec.year)
    with connect() as conn:
        cur = conn.execute("SELECT content_hash FROM decision WHERE cd = ?", (dec.cd,))
        existing = cur.fetchone()

        if existing is None:
            outcome = "new"
            conn.execute(
                """
                INSERT INTO decision (
                    cd, number, year, category_id, chamber_id, category, chamber,
                    subject, summary, body, body_chars, source_url, content_hash,
                    first_seen, last_fetched, last_changed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), datetime('now'))
                """,
                (
                    dec.cd, dec.number, dec.year, category_id(dec.category),
                    chamber_id(dec.chamber), dec.category, dec.chamber,
                    dec.subject, dec.summary, dec.body, len(dec.body),
                    url, chash,
                ),
            )
        else:
            outcome = "unchanged" if existing["content_hash"] == chash else "changed"
            conn.execute(
                """
                UPDATE decision SET
                    number = ?, year = ?, category_id = ?, chamber_id = ?,
                    category = ?, chamber = ?, subject = ?, summary = ?,
                    body = ?, body_chars = ?, source_url = ?, content_hash = ?,
                    last_fetched = datetime('now'),
                    last_changed = CASE WHEN content_hash != ? THEN datetime('now') ELSE last_changed END
                WHERE cd = ?
                """,
                (
                    dec.number, dec.year, category_id(dec.category),
                    chamber_id(dec.chamber), dec.category, dec.chamber,
                    dec.subject, dec.summary, dec.body, len(dec.body),
                    url, chash, chash, dec.cd,
                ),
            )

        # Update decision FTS table
        conn.execute(
            """
            INSERT OR REPLACE INTO decision_fts (rowid, subject_folded, summary_folded, body_folded, cd)
            VALUES (
                (SELECT rowid FROM decision WHERE cd = ?),
                ?, ?, ?, ?
            )
            """,
            (dec.cd, fold_greek(dec.subject), fold_greek(dec.summary), fold_greek(dec.body), dec.cd),
        )

        # Attach subjects to controlled vocabulary
        if dec.subjects:
            for s in dec.subjects:
                fs = fold_greek(s)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO decision_theme (cd, theme_code)
                    SELECT ?, code FROM theme WHERE lower(label) = ?
                    """,
                    (dec.cd, fs),
                )
        conn.commit()
    return outcome


def decision_count() -> int:
    with connect() as conn:
        cur = conn.execute("SELECT count(*) AS n FROM decision")
        return cur.fetchone()["n"]


# ---------------------------------------------------------------------------
# Crawl Bookkeeping
# ---------------------------------------------------------------------------


def record_partition(
    year: int, category_id_: int, chamber_id_: int, n_found: int, truncated: bool,
    status: str = "done", error: str | None = None,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO crawl_partition
                (year, category_id, chamber_id, status, n_found, truncated, error, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT (year, category_id, chamber_id) DO UPDATE SET
                status = excluded.status, n_found = excluded.n_found,
                truncated = excluded.truncated, error = excluded.error,
                updated_at = datetime('now')
            """,
            (year, category_id_, chamber_id_, status, n_found, 1 if truncated else 0, error),
        )
        conn.commit()


def year_is_done(year: int) -> bool:
    with connect() as conn:
        cur = conn.execute(
            """
            SELECT status FROM crawl_partition
             WHERE year = ? AND category_id = 6 AND chamber_id = 1
            """,
            (year,),
        )
        row = cur.fetchone()
        return bool(row and row["status"] == "done")


def start_run(kind: str) -> int:
    with connect() as conn:
        cur = conn.execute("INSERT INTO crawl_run (kind) VALUES (?)", (kind,))
        conn.commit()
        return cur.lastrowid


def finish_run(run_id: int, **counts) -> None:
    with connect() as conn:
        conn.execute(
            """
            UPDATE crawl_run SET finished_at = datetime('now'),
                   n_discovered = ?, n_fetched = ?, n_changed = ?,
                   n_errors = ?, notes = ?
             WHERE id = ?
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
        conn.commit()


def stats() -> dict:
    with connect() as conn:
        cur = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM decision) AS decisions,
              (SELECT count(*) FROM decision WHERE summary IS NOT NULL AND length(summary) > 0) AS with_summary,
              (SELECT count(*) FROM chunk) AS chunks,
              (SELECT count(*) FROM chunk_vec) AS embedded,
              (SELECT count(*) FROM theme) AS themes,
              (SELECT count(*) FROM decision_theme) AS theme_links,
              (SELECT count(*) FROM fetch_queue WHERE attempts < 5) AS queued,
              (SELECT min(year) FROM decision) AS first_year,
              (SELECT max(year) FROM decision) AS last_year,
              (SELECT count(*) FROM decision WHERE indexed_hash IS NOT content_hash) AS pending_index
            """
        )
        return dict(cur.fetchone())
