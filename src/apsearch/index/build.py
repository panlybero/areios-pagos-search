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
    """Refuse to mix vectors from two different models in one index.

    Cosine distances between embeddings from different models are meaningless,
    and the failure mode is silent: search just returns nonsense. Better to
    stop and make the operator run `index reset`.
    """
    sig = embedding_signature(backend)
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
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) AS n FROM decision
             WHERE indexed_hash IS DISTINCT FROM content_hash AND body IS NOT NULL
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
    if build_index_after and stats.chunks:
        log.info("building HNSW index (this is the slow part on first run)")
        create_vector_index()
    return stats


def reset_index() -> None:
    """Drop all chunks and embeddings so the corpus can be re-indexed."""
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS chunk_embedding_idx")
        cur.execute("TRUNCATE chunk RESTART IDENTITY")
        cur.execute("TRUNCATE query_cache")
        cur.execute("UPDATE decision SET indexed_hash = NULL")
        cur.execute("DELETE FROM index_meta WHERE key = 'embedding'")
    log.info("index and query cache reset; embedding model/dimension can now be changed")
