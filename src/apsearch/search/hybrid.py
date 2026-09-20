"""Hybrid retrieval over the decision corpus (SQLite FTS5 + sqlite-vec).

Three independent retrievers, fused with Reciprocal Rank Fusion:

``chunk_lexical``   FTS5 over passages. Wins on citations, statute numbers and
                    exact legal terms of art.
``doc_lexical``     FTS5 over the whole decision (subject/summary/body).
                    Catches documents whose relevance is spread thinly rather
                    than concentrated in one passage.
``semantic``        sqlite-vec cosine over passage embeddings. Carries the
                    morphological load that the stemmer drops (it treats
                    "αδικοπραξία" and "αδικοπραξίας" differently), and handles
                    paraphrase.

RRF is used rather than score interpolation because the three scores live on
incomparable scales (FTS5 rank is unbounded, cosine distance is [0,2]) and RRF
needs no per-corpus tuning to stay stable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import sqlite_vec

from apsearch.config import settings
from apsearch.db.sqlite import connect, fold_greek
from apsearch.index.embed import get_backend
from apsearch.logging import get_logger
from apsearch.search import cache
from apsearch.search.query import (
    MIN_PREFIX_LEN,
    content_tokens,
    has_operators,
    trim_unstable_suffix,
)

log = get_logger(__name__)

Mode = Literal["hybrid", "keyword", "semantic"]

#: Relative trust in each retriever. Lexical passage matches are the strongest
#: single signal for legal search; the doc-level signal is a tie-breaker.
DEFAULT_WEIGHTS = {"chunk_lexical": 1.0, "doc_lexical": 0.5, "semantic": 1.0}


@dataclass(slots=True)
class Filters:
    year_from: int | None = None
    year_to: int | None = None
    category: str | None = None
    chamber: str | None = None
    themes: list[str] = field(default_factory=list)
    cds: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Passage:
    chunk_id: int
    ordinal: int
    part: str
    content: str
    highlight: str | None = None
    lexical_score: float | None = None
    semantic_score: float | None = None


@dataclass(slots=True)
class Result:
    cd: str
    number: int | None
    year: int | None
    citation: str
    category: str | None
    chamber: str | None
    subject: str | None
    summary: str | None
    themes: list[str]
    url: str
    score: float
    ranks: dict[str, int]
    passages: list[Passage] = field(default_factory=list)

    def to_dict(self, include_passages: bool = True) -> dict:
        d = {
            "cd": self.cd,
            "citation": self.citation,
            "number": self.number,
            "year": self.year,
            "category": self.category,
            "chamber": self.chamber,
            "subject": self.subject,
            "summary": self.summary,
            "themes": self.themes,
            "url": self.url,
            "score": round(self.score, 6),
            "matched_by": sorted(self.ranks),
        }
        if include_passages:
            d["passages"] = [
                {
                    "part": p.part,
                    "ordinal": p.ordinal,
                    "text": p.highlight or p.content,
                }
                for p in self.passages
            ]
        return d


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def rrf(
    ranked_lists: dict[str, list[str]],
    weights: dict[str, float] | None = None,
    k: int | None = None,
) -> dict[str, tuple[float, dict[str, int]]]:
    """Reciprocal Rank Fusion.

    ``score(d) = Σ_r w_r / (k + rank_r(d))``. ``k`` damps the influence of the
    very top of each list so a single retriever cannot dominate the fusion.
    """
    k = k if k is not None else settings.rrf_k
    weights = weights or DEFAULT_WEIGHTS
    out: dict[str, tuple[float, dict[str, int]]] = {}
    for source, ids in ranked_lists.items():
        w = weights.get(source, 1.0)
        for rank, cd in enumerate(ids, start=1):
            score, ranks = out.get(cd, (0.0, {}))
            if source in ranks:  # keep only the best rank per source
                continue
            out[cd] = (score + w / (k + rank), {**ranks, source: rank})
    return out


# ---------------------------------------------------------------------------
# FTS5 query construction & highlighting
# ---------------------------------------------------------------------------


def build_fts5_query(query: str, conjunctive: bool = True) -> str:
    """Turn a Greek query into SQLite FTS5 syntax."""
    query = (query or "").strip()
    if not query:
        return ""

    if has_operators(query):
        # Escape quotes or clean operators for FTS5
        clean = query.replace('"', '""')
        return f'"{clean}"'

    tokens = content_tokens(query)
    if not tokens:
        tokens = [t for t in query.split() if t]

    parts: list[str] = []
    for tok in tokens:
        clean = re.sub(r"[^\w\u0370-\u03ff\u1f00-\u1fff]", "", tok, flags=re.UNICODE)
        if not clean:
            continue
        folded = fold_greek(clean)
        if len(folded) >= MIN_PREFIX_LEN:
            trimmed = trim_unstable_suffix(folded)
            parts.append(f'"{trimmed}"*')
        else:
            parts.append(f'"{folded}"')

    if not parts:
        return ""
    sep = " AND " if conjunctive else " OR "
    return sep.join(dict.fromkeys(parts))


def highlight_greek(text: str, query: str, max_chars: int = 350) -> str:
    """Highlight query stems in original Greek text and extract a relevant window."""
    if not text:
        return ""
    tokens = content_tokens(query)
    if not tokens:
        tokens = [t for t in query.split() if len(t) >= 3]

    stems = []
    for t in tokens:
        f = fold_greek(re.sub(r"[^\w]", "", t))
        if len(f) >= 4:
            stems.append(trim_unstable_suffix(f))
        elif f:
            stems.append(f)

    if not stems:
        return text[:max_chars].replace("\n", " ").strip()

    # Find the earliest match position to center the snippet
    folded_text = fold_greek(text)
    first_pos = len(text)
    for stem in stems:
        idx = folded_text.find(stem)
        if 0 <= idx < first_pos:
            first_pos = idx

    # Compute excerpt window
    start = max(0, first_pos - 60)
    end = min(len(text), start + max_chars)
    # Align to word boundary
    if start > 0:
        sp = text.find(" ", start)
        if 0 < sp < start + 30:
            start = sp + 1
    snippet = text[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = "… " + snippet
    if end < len(text):
        snippet = snippet + " …"

    # Mark bold tags
    for stem in sorted(stems, key=len, reverse=True):
        pattern = re.compile(f"(?i)\\b({re.escape(stem)}[\\w]*)", re.UNICODE)
        # Approximate match on original text
        snippet = pattern.sub(r"**\1**", snippet)
    return snippet


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def search(
    query: str,
    mode: Mode = "hybrid",
    limit: int | None = None,
    filters: Filters | None = None,
    weights: dict[str, float] | None = None,
    candidate_pool: int | None = None,
    passages_per_result: int = 3,
    highlight: bool = True,
    expand_query: bool = True,
    min_lexical_hits: int = 5,
) -> list[Result]:
    limit = limit or settings.default_limit
    filters = filters or Filters()
    pool_size = candidate_pool or settings.candidate_pool
    query = (query or "").strip()

    if len(query) > settings.max_query_chars:
        raise ValueError(
            f"Query is too long ({len(query)} > {settings.max_query_chars} chars). "
            "Please provide a concise legal search query."
        )

    use_lexical = mode in ("hybrid", "keyword") and bool(query)
    use_semantic = mode in ("hybrid", "semantic") and bool(query)

    where_clauses: list[str] = []
    where_params: list[Any] = []
    if filters.year_from is not None:
        where_clauses.append("d.year >= ?")
        where_params.append(filters.year_from)
    if filters.year_to is not None:
        where_clauses.append("d.year <= ?")
        where_params.append(filters.year_to)
    if filters.category:
        where_clauses.append("d.category LIKE ?")
        where_params.append(f"%{filters.category}%")
    if filters.chamber:
        where_clauses.append("d.chamber = ?")
        where_params.append(filters.chamber)
    if filters.cds:
        placeholders = ",".join("?" * len(filters.cds))
        where_clauses.append(f"d.cd IN ({placeholders})")
        where_params.extend(filters.cds)
    if filters.themes:
        theme_preds = " OR ".join(["t.label LIKE ?" for _ in filters.themes])
        where_clauses.append(
            f"""d.cd IN (
                SELECT dt.cd FROM decision_theme dt JOIN theme t ON t.code = dt.theme_code
                 WHERE {theme_preds}
            )"""
        )
        where_params.extend([f"%{t}%" for t in filters.themes])

    where_sql = (" AND ".join(where_clauses)) if where_clauses else "1=1"

    with connect() as conn:
        qvec = None
        if use_semantic:
            backend = get_backend()
            model_sig = f"{settings.embed_backend}:{backend.name}:{backend.dim}"
            # Check query cache
            cached = cache.get_cached_vector(conn, query, model_sig)
            if cached is not None:
                qvec = cached
            else:
                raw_vec = backend.embed_queries([query])[0]
                cache.store_cached_vector(conn, query, model_sig, raw_vec)
                qvec = raw_vec

        chunk_rows: list[dict] = []
        doc_rows: list[dict] = []
        vec_rows: list[dict] = []
        ranked: dict[str, list[str]] = {}

        # 1. Lexical retrieval via FTS5
        if use_lexical:
            fts_query = build_fts5_query(query, conjunctive=True)
            if fts_query:
                cur = conn.execute(
                    f"""
                    SELECT c.cd, c.id AS chunk_id, c.ordinal, c.part, f.rank AS score
                      FROM chunk_fts f
                      JOIN chunk c ON c.id = f.rowid
                      JOIN decision d ON d.cd = c.cd
                     WHERE {where_sql} AND f.content_folded MATCH ?
                     ORDER BY f.rank
                     LIMIT ?
                    """,
                    [*where_params, fts_query, pool_size],
                )
                chunk_rows = [dict(r) for r in cur.fetchall()]

                cur = conn.execute(
                    f"""
                    SELECT d.cd, f.rank AS score
                      FROM decision_fts f
                      JOIN decision d ON d.rowid = f.rowid
                     WHERE {where_sql} AND f.decision_fts MATCH ?
                     ORDER BY f.rank
                     LIMIT ?
                    """,
                    [*where_params, fts_query, pool_size],
                )
                doc_rows = [dict(r) for r in cur.fetchall()]

            # Fallback to OR if AND had too few hits
            if len(chunk_rows) < min_lexical_hits and not has_operators(query):
                or_query = build_fts5_query(query, conjunctive=False)
                if or_query and or_query != fts_query:
                    cur = conn.execute(
                        f"""
                        SELECT c.cd, c.id AS chunk_id, c.ordinal, c.part, f.rank AS score
                          FROM chunk_fts f
                          JOIN chunk c ON c.id = f.rowid
                          JOIN decision d ON d.cd = c.cd
                         WHERE {where_sql} AND f.content_folded MATCH ?
                         ORDER BY f.rank
                         LIMIT ?
                        """,
                        [*where_params, or_query, pool_size],
                    )
                    chunk_rows = [dict(r) for r in cur.fetchall()]

                    cur = conn.execute(
                        f"""
                        SELECT d.cd, f.rank AS score
                          FROM decision_fts f
                          JOIN decision d ON d.rowid = f.rowid
                         WHERE {where_sql} AND f.decision_fts MATCH ?
                         ORDER BY f.rank
                         LIMIT ?
                        """,
                        [*where_params, or_query, pool_size],
                    )
                    doc_rows = [dict(r) for r in cur.fetchall()]

            ranked["chunk_lexical"] = [r["cd"] for r in chunk_rows]
            ranked["doc_lexical"] = [r["cd"] for r in doc_rows]

        # 2. Semantic retrieval via sqlite-vec
        if use_semantic and qvec is not None:
            raw_blob = sqlite_vec.serialize_float32(qvec)
            cur = conn.execute(
                f"""
                WITH knn AS (
                    SELECT rowid, distance
                      FROM chunk_vec
                     WHERE embedding MATCH ?
                       AND k = ?
                )
                SELECT c.cd, c.id AS chunk_id, c.ordinal, c.part, k.distance AS score
                  FROM knn k
                  JOIN chunk c ON c.id = k.rowid
                  JOIN decision d ON d.cd = c.cd
                 WHERE {where_sql}
                 ORDER BY k.distance
                """,
                [raw_blob, pool_size, *where_params],
            )
            vec_rows = [dict(r) for r in cur.fetchall()]
            ranked["semantic"] = [r["cd"] for r in vec_rows]

        if not query:
            cur = conn.execute(
                f"""
                SELECT cd FROM decision d
                 WHERE {where_sql}
                 ORDER BY d.year DESC, d.number DESC
                 LIMIT ?
                """,
                [*where_params, limit],
            )
            top = [(r["cd"], 0.0, {}) for r in cur.fetchall()]
        else:
            fused = rrf(ranked, weights)
            top = [
                (cd, score, ranks)
                for cd, (score, ranks) in sorted(
                    fused.items(), key=lambda kv: kv[1][0], reverse=True
                )[:limit]
            ]

        if not top:
            return []

        cds = [cd for cd, _, _ in top]
        meta = _load_meta(conn, cds)
        passages = _collect_passages(
            conn, cds, chunk_rows, vec_rows, query, passages_per_result, highlight
        )

    results: list[Result] = []
    for cd, score, ranks in top:
        m = meta.get(cd)
        if not m:
            continue
        results.append(
            Result(
                cd=cd,
                number=m["number"],
                year=m["year"],
                citation=f"{m['number']}/{m['year']}",
                category=m["category"],
                chamber=m["chamber"],
                subject=m["subject"],
                summary=m["summary"],
                themes=list(m["themes"] or []),
                url=m["source_url"],
                score=score,
                ranks=ranks,
                passages=passages.get(cd, []),
            )
        )
    return results


def _load_meta(conn, cds: list[str]) -> dict[str, dict]:
    placeholders = ",".join("?" * len(cds))
    cur = conn.execute(
        f"""
        SELECT d.cd, d.number, d.year, d.category, d.chamber, d.subject, d.summary,
               d.source_url,
               coalesce(group_concat(t.label, ', '), '') AS theme_labels
          FROM decision d
          LEFT JOIN decision_theme dt ON dt.cd = d.cd
          LEFT JOIN theme t ON t.code = dt.theme_code
         WHERE d.cd IN ({placeholders})
         GROUP BY d.cd
        """,
        cds,
    )
    out = {}
    for r in cur.fetchall():
        d = dict(r)
        d["themes"] = [t.strip() for t in d["theme_labels"].split(",") if t.strip()]
        out[d["cd"]] = d
    return out


def _collect_passages(
    conn,
    cds: list[str],
    chunk_rows: list[dict],
    vec_rows: list[dict],
    query: str,
    per_result: int,
    highlight: bool,
) -> dict[str, list[Passage]]:
    best: dict[str, dict[int, Passage]] = {cd: {} for cd in cds}
    wanted = set(cds)

    for row in chunk_rows:
        if row["cd"] not in wanted:
            continue
        slot = best[row["cd"]]
        p = slot.get(row["chunk_id"])
        if p is None:
            p = Passage(row["chunk_id"], row["ordinal"], row["part"], "")
            slot[row["chunk_id"]] = p
        p.lexical_score = float(row["score"])

    for row in vec_rows:
        if row["cd"] not in wanted:
            continue
        slot = best[row["cd"]]
        p = slot.get(row["chunk_id"])
        if p is None:
            p = Passage(row["chunk_id"], row["ordinal"], row["part"], "")
            slot[row["chunk_id"]] = p
        p.semantic_score = float(row["score"])

    chosen: dict[str, list[Passage]] = {}
    chunk_ids: list[int] = []
    for cd in cds:
        ranked = sorted(
            best[cd].values(),
            key=lambda p: (
                p.part != "summary",
                -((1.0 / (abs(p.lexical_score or 1.0) + 0.1)) * 10 + (1.0 - (p.semantic_score or 1.0))),
            ),
        )[:per_result]
        chosen[cd] = ranked
        chunk_ids.extend(p.chunk_id for p in ranked)

    if not chunk_ids:
        return chosen

    placeholders = ",".join("?" * len(chunk_ids))
    cur = conn.execute(
        f"SELECT id, content FROM chunk WHERE id IN ({placeholders})",
        chunk_ids,
    )
    content_map = {r["id"]: r["content"] for r in cur.fetchall()}

    for plist in chosen.values():
        for p in plist:
            raw = content_map.get(p.chunk_id, "")
            p.content = raw
            if highlight and query:
                p.highlight = highlight_greek(raw, query)
            else:
                p.highlight = raw[:350].replace("\n", " ").strip()
    return chosen


# ---------------------------------------------------------------------------
# Direct lookups
# ---------------------------------------------------------------------------


def get_decision(
    cd: str | None = None,
    number: int | None = None,
    year: int | None = None,
    chamber: str | None = None,
    include_body: bool = True,
) -> list[dict]:
    clauses, params = [], []
    if cd:
        clauses.append("d.cd = ?")
        params.append(cd)
    if number is not None:
        clauses.append("d.number = ?")
        params.append(number)
    if year is not None:
        clauses.append("d.year = ?")
        params.append(year)
    if chamber:
        clauses.append("d.chamber = ?")
        params.append(chamber)
    if not clauses:
        raise ValueError("provide cd, or number and year")

    body_col = "d.body," if include_body else ""
    with connect() as conn:
        cur = conn.execute(
            f"""
            SELECT d.cd, d.number, d.year, d.category, d.chamber, d.subject,
                   d.summary, {body_col} d.source_url, d.body_chars,
                   d.first_seen, d.last_fetched,
                   coalesce(group_concat(t.label, ', '), '') AS theme_labels
              FROM decision d
              LEFT JOIN decision_theme dt ON dt.cd = d.cd
              LEFT JOIN theme t ON t.code = dt.theme_code
             WHERE {' AND '.join(clauses)}
             GROUP BY d.cd
             ORDER BY d.year DESC, d.number DESC
            """,
            params,
        )
        out = []
        for r in cur.fetchall():
            row = dict(r)
            row["themes"] = [t.strip() for t in row["theme_labels"].split(",") if t.strip()]
            row.pop("theme_labels", None)
            out.append(row)
        return out


def list_themes(prefix: str | None = None, limit: int = 100) -> list[dict]:
    with connect() as conn:
        cur = conn.execute(
            """
            SELECT t.code, t.label, count(dt.cd) AS n_decisions
              FROM theme t
              LEFT JOIN decision_theme dt ON dt.theme_code = t.code
             WHERE (? IS NULL OR t.label LIKE '%' || ? || '%')
             GROUP BY t.code, t.label
             ORDER BY count(dt.cd) DESC, t.label
             LIMIT ?
            """,
            (prefix, prefix, limit),
        )
        return [dict(r) for r in cur.fetchall()]
