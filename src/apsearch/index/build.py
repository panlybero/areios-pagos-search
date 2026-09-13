"""Build the retrieval index: chunk decisions, embed chunks, store vectors.

Resumable by construction. ``decision.indexed_hash`` records the content hash
that was last indexed, so re-running only touches decisions that are new or
whose text actually changed upstream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from pgvector import Vector
from pgvector.psycopg import register_vector

from apsearch.config import settings
from apsearch.db import create_vector_index, pool
from apsearch.index.chunk import chunk_decision
from apsearch.index.embed import EmbeddingBackend, get_backend
from apsearch.logging import get_logger

log = get_logger(__name__)


@dataclass
class IndexStats:
    decisions: int = 0
    chunks: int = 0
    seconds: float = 0.0

    @property
    def rate(self) -> float:
        return self.chunks / self.seconds if self.seconds else 0.0


def embedding_signature(backend: EmbeddingBackend) -> str:
    return f"{settings.embed_backend}:{backend.name}:{backend.dim}"


def assert_embedding_space(backend: EmbeddingBackend) -> None:
    """Refuse to mix vectors from two different models in one index."""
    sig = embedding_signature(backend)
    if settings.db_backend == "sqlite":
        from apsearch.db.sqlite import connect

        with connect() as conn:
            cur = conn.execute("SELECT value FROM index_meta WHERE key = 'embedding'")
            row = cur.fetchone()
            if row and row["value"] != sig:
                cur2 = conn.execute("SELECT count(*) AS n FROM chunk_vec")
                if cur2.fetchone()["n"]:
                    raise RuntimeError(
                        f"index holds vectors from {row['value']!r} but configuration "
                        f"requests {sig!r}. Run `apsearch index reset` to re-embed."
                    )
            conn.execute(
                """
                INSERT INTO index_meta (key, value, updated_at) VALUES (?, ?, datetime('now'))
                ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = datetime('now')
                """,
                (sig, sig),
            )
            conn.commit()
        return

    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT value FROM index_meta WHERE key = 'embedding'")
        row = cur.fetchone()
        if row and row["value"] != sig:
            cur.execute("SELECT count(*) AS n FROM chunk WHERE embedding IS NOT NULL")
            if cur.fetchone()["n"]:
                raise RuntimeError(
                    f"index holds vectors from {row['value']!r} but configuration "
                    f"requests {sig!r}. Run `apsearch index reset` to re-embed."
                )
        cur.execute(
            """
            INSERT INTO index_meta (key, value) VALUES ('embedding', %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
            """,
            (sig,),
        )


def pending_decisions(limit: int) -> list[dict]:
    """Decisions whose current text has not been indexed yet."""
    if settings.db_backend == "sqlite":
        from apsearch.db.sqlite import connect

        with connect() as conn:
            cur = conn.execute(
                """
                SELECT cd, subject, summary, body, content_hash
                  FROM decision
                 WHERE indexed_hash IS NOT content_hash
                   AND body IS NOT NULL
                 ORDER BY year DESC, number DESC
                 LIMIT ?
                """,
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT cd, subject, summary, body, content_hash
              FROM decision
             WHERE indexed_hash IS DISTINCT FROM content_hash
               AND body IS NOT NULL
             ORDER BY year DESC NULLS LAST, number DESC NULLS LAST
             LIMIT %s
            """,
            (limit,),
        )
        return cur.fetchall()


def pending_count() -> int:
    if settings.db_backend == "sqlite":
        from apsearch.db.sqlite import connect

        with connect() as conn:
            cur = conn.execute(
                """
                SELECT count(*) AS n FROM decision
                 WHERE indexed_hash IS NOT content_hash AND body IS NOT NULL
                """
            )
            return cur.fetchone()["n"]

    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) AS n FROM decision
             WHERE indexed_hash IS DISTINCT FROM content_hash AND body IS NOT NULL
            """
        )
        return cur.fetchone()["n"]


def index_batch_sqlite(rows: list[dict], backend: EmbeddingBackend) -> int:
    import sqlite_vec

    from apsearch.db.sqlite import connect, fold_greek

    if not rows:
        return 0

    plans: list[tuple[str, list]] = []
    texts: list[str] = []
    for row in rows:
        chunks = chunk_decision(row["body"], row["summary"], row["subject"])
        plans.append((row["cd"], chunks))
        texts.extend(c.content for c in chunks)

    if not texts:
        _mark_indexed(rows)
        return 0

    vectors = backend.embed_passages(texts)

    with connect() as conn:
        cursor_pos = 0
        for cd, chunks in plans:
            old_ids = [
                r["id"]
                for r in conn.execute("SELECT id FROM chunk WHERE cd = ?", (cd,)).fetchall()
            ]
            if old_ids:
                p_holders = ",".join("?" * len(old_ids))
                conn.execute(f"DELETE FROM chunk_fts WHERE rowid IN ({p_holders})", old_ids)
                conn.execute(f"DELETE FROM chunk_vec WHERE rowid IN ({p_holders})", old_ids)
                conn.execute("DELETE FROM chunk WHERE cd = ?", (cd,))

            for c in chunks:
                vec_bytes = sqlite_vec.serialize_float32(vectors[cursor_pos])
                cursor_pos += 1
                cur = conn.execute(
                    """
                    INSERT INTO chunk (cd, ordinal, part, char_start, char_end, content)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (cd, c.ordinal, c.part, c.char_start, c.char_end, c.content),
                )
                chunk_id = cur.lastrowid
                conn.execute(
                    """
                    INSERT INTO chunk_fts (rowid, content_folded, content, cd, part)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (chunk_id, fold_greek(c.content), c.content, cd, c.part),
                )
                conn.execute(
                    "INSERT INTO chunk_vec (rowid, embedding) VALUES (?, ?)",
                    (chunk_id, vec_bytes),
                )

        conn.executemany(
            "UPDATE decision SET indexed_hash = ? WHERE cd = ?",
            [(r["content_hash"], r["cd"]) for r in rows],
        )
        conn.commit()
    return len(texts)


def index_batch(rows: list[dict], backend: EmbeddingBackend) -> int:
    """Chunk, embed and store one batch of decisions. Returns chunks written."""
    if settings.db_backend == "sqlite":
        return index_batch_sqlite(rows, backend)

    if not rows:
        return 0

    plans: list[tuple[str, list]] = []
    texts: list[str] = []
    for row in rows:
        chunks = chunk_decision(row["body"], row["summary"], row["subject"])
        plans.append((row["cd"], chunks))
        texts.extend(c.content for c in chunks)

    if not texts:
        # Nothing to embed, but still mark as indexed so we don't spin on it.
        _mark_indexed(rows)
        return 0

    vectors = backend.embed_passages(texts)

    with pool().connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cursor_pos = 0
            for cd, chunks in plans:
                # Replace wholesale: simpler and correct when text changes.
                cur.execute("DELETE FROM chunk WHERE cd = %s", (cd,))
                payload = []
                for c in chunks:
                    payload.append(
                        (cd, c.ordinal, c.part, c.char_start, c.char_end,
                         c.content, Vector(vectors[cursor_pos]))
                    )
                    cursor_pos += 1
                if payload:
                    cur.executemany(
                        """
                        INSERT INTO chunk
                            (cd, ordinal, part, char_start, char_end, content, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        payload,
                    )
            cur.executemany(
                "UPDATE decision SET indexed_hash = %s WHERE cd = %s",
                [(r["content_hash"], r["cd"]) for r in rows],
            )
    return len(texts)


def _mark_indexed(rows: list[dict]) -> None:
    if settings.db_backend == "sqlite":
        from apsearch.db.sqlite import connect

        with connect() as conn:
            conn.executemany(
                "UPDATE decision SET indexed_hash = ? WHERE cd = ?",
                [(r["content_hash"], r["cd"]) for r in rows],
            )
            conn.commit()
        return

    with pool().connection() as conn, conn.cursor() as cur:
        cur.executemany(
            "UPDATE decision SET indexed_hash = %s WHERE cd = %s",
            [(r["content_hash"], r["cd"]) for r in rows],
        )


def run_index(
    limit: int | None = None,
    batch_size: int = 16,
    build_index_after: bool = True,
) -> IndexStats:
    """Index pending decisions.

    ``batch_size`` is in *decisions*, not chunks; a decision yields ~20 chunks,
    so 16 decisions is roughly 300 texts per embedding call.
    """
    backend = get_backend()
    assert_embedding_space(backend)
    stats = IndexStats()
    started = time.monotonic()
    total_pending = pending_count()
    target = min(limit, total_pending) if limit else total_pending
    log.info("indexing %d of %d pending decisions with %s",
             target, total_pending, backend.name)

    while True:
        remaining = target - stats.decisions
        if remaining <= 0:
            break
        rows = pending_decisions(min(batch_size, remaining))
        if not rows:
            break
        n = index_batch(rows, backend)
        stats.decisions += len(rows)
        stats.chunks += n
        stats.seconds = time.monotonic() - started
        log.info(
            "indexed %d/%d decisions, %d chunks (%.1f chunks/s)",
            stats.decisions, target, stats.chunks, stats.rate,
        )

    stats.seconds = time.monotonic() - started
    if build_index_after and stats.chunks and settings.db_backend != "sqlite":
        log.info("building HNSW index (this is the slow part on first run)")
        create_vector_index()
    return stats


def reset_index() -> None:
    """Drop all chunks and embeddings so the corpus can be re-indexed."""
    if settings.db_backend == "sqlite":
        from apsearch.db.sqlite import connect

        with connect() as conn:
            conn.execute("DELETE FROM chunk")
            conn.execute("DELETE FROM chunk_fts")
            conn.execute("DELETE FROM chunk_vec")
            conn.execute("DELETE FROM query_cache")
            conn.execute("UPDATE decision SET indexed_hash = NULL")
            conn.execute("DELETE FROM index_meta WHERE key = 'embedding'")
            conn.commit()
        log.info("SQLite index and query cache reset")
        return

    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS chunk_embedding_idx")
        cur.execute("TRUNCATE chunk RESTART IDENTITY")
        cur.execute("TRUNCATE query_cache")
        cur.execute("UPDATE decision SET indexed_hash = NULL")
        cur.execute("DELETE FROM index_meta WHERE key = 'embedding'")
    log.info("index and query cache reset; embedding model/dimension can now be changed")
