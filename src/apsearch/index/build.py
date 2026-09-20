"""Build the retrieval index: chunk decisions, embed chunks, store vectors.

Resumable by construction. ``decision.indexed_hash`` records the content hash
that was last indexed, so re-running only touches decisions that are new or
whose text actually changed upstream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import sqlite_vec

from apsearch.config import settings
from apsearch.db.sqlite import connect, fold_greek, fts_delete
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


def pending_decisions(limit: int) -> list[dict]:
    """Decisions whose current text has not been indexed yet."""
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


def pending_count() -> int:
    with connect() as conn:
        cur = conn.execute(
            """
            SELECT count(*) AS n FROM decision
             WHERE indexed_hash IS NOT content_hash AND body IS NOT NULL
            """
        )
        return cur.fetchone()["n"]


def index_batch(rows: list[dict], backend: EmbeddingBackend) -> int:
    """Chunk, embed and store one batch of decisions. Returns chunks written."""
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
            # Replace wholesale: simpler and correct when text changes. The
            # contentless FTS5 rows must be dropped via the special 'delete'
            # command (they cannot be DELETE'd normally), so we fetch each old
            # chunk's text to reconstruct its indexed value first.
            old = [
                dict(r)
                for r in conn.execute("SELECT id, content FROM chunk WHERE cd = ?", (cd,)).fetchall()
            ]
            for o in old:
                fts_delete(conn, "chunk_fts", o["id"], [fold_greek(o["content"])])
                conn.execute("DELETE FROM chunk_vec WHERE rowid = ?", (o["id"],))
                conn.execute("DELETE FROM chunk WHERE id = ?", (o["id"],))

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
                    "INSERT INTO chunk_fts (rowid, content_folded) VALUES (?, ?)",
                    (chunk_id, fold_greek(c.content)),
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


def _mark_indexed(rows: list[dict]) -> None:
    with connect() as conn:
        conn.executemany(
            "UPDATE decision SET indexed_hash = ? WHERE cd = ?",
            [(r["content_hash"], r["cd"]) for r in rows],
        )
        conn.commit()


def run_index(limit: int | None = None, batch_size: int = 16) -> IndexStats:
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
    return stats


def reset_index() -> None:
    """Drop all chunks and embeddings so the corpus can be re-indexed."""
    with connect() as conn:
        conn.execute("DELETE FROM chunk")
        # Contentless FTS5 tables cannot be emptied with DELETE; drop + recreate.
        conn.execute("DROP TABLE IF EXISTS chunk_fts")
        conn.execute("CREATE VIRTUAL TABLE chunk_fts USING fts5(content_folded, content='')")
        conn.execute("DELETE FROM chunk_vec")
        conn.execute("DELETE FROM query_cache")
        conn.execute("UPDATE decision SET indexed_hash = NULL")
        conn.execute("DELETE FROM index_meta WHERE key = 'embedding'")
        conn.commit()
    log.info("index and query cache reset; embedding model/dimension can now be changed")
