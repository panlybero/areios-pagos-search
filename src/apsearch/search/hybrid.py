"""Hybrid retrieval over the decision corpus.

Three independent retrievers, fused with Reciprocal Rank Fusion:

``chunk_lexical``   Postgres FTS over passages. Wins on citations, statute
                    numbers and exact legal terms of art.
``doc_lexical``     Weighted FTS over the whole decision (subject^A,
                    headnote^B, body^C). Catches documents whose relevance is
                    spread thinly rather than concentrated in one passage.
``semantic``        pgvector cosine over passage embeddings. Carries the
                    morphological load that the Greek snowball stemmer drops
                    (it stems "αδικοπραξία" and "αδικοπραξίας" differently),
                    and handles paraphrase.

RRF is used rather than score interpolation because the three scores live on
incomparable scales (ts_rank_cd is unbounded, cosine is [-1,1]) and RRF needs
no per-corpus tuning to stay stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pgvector import Vector
from pgvector.psycopg import register_vector

from apsearch.config import settings
from apsearch.db import pool
from apsearch.index.embed import get_backend
from apsearch.logging import get_logger
from apsearch.search import query as qbuild

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

    def where(self) -> tuple[str, dict]:
        clauses: list[str] = []
        params: dict = {}
        if self.year_from is not None:
            clauses.append("d.year >= %(year_from)s")
            params["year_from"] = self.year_from
        if self.year_to is not None:
            clauses.append("d.year <= %(year_to)s")
            params["year_to"] = self.year_to
        if self.category:
            clauses.append("d.category ILIKE %(category)s")
            params["category"] = f"%{self.category}%"
        if self.chamber:
            clauses.append("d.chamber = %(chamber)s")
            params["chamber"] = self.chamber
        if self.cds:
            clauses.append("d.cd = ANY(%(cds)s)")
            params["cds"] = self.cds
        if self.themes:
            clauses.append(
                """EXISTS (
                    SELECT 1 FROM decision_theme dt JOIN theme t ON t.code = dt.theme_code
                     WHERE dt.cd = d.cd AND t.label ILIKE ANY(%(themes)s)
                )"""
            )
            params["themes"] = [f"%{t}%" for t in self.themes]
        return (" AND ".join(clauses) if clauses else "TRUE"), params


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
# Retrievers
# ---------------------------------------------------------------------------


def _chunk_lexical(cur, qexpr: str, qparams: dict, where: str,
                   params: dict, pool_size: int):
    cur.execute(
        f"""
        SELECT c.cd, c.id AS chunk_id, c.ordinal, c.part,
               ts_rank_cd(c.tsv_stem, q, 32) AS score
          FROM chunk c
          JOIN decision d ON d.cd = c.cd,
               LATERAL (SELECT {qexpr} AS q) tq
         WHERE {where} AND c.tsv_stem @@ tq.q
         ORDER BY score DESC, c.id
         LIMIT %(pool)s
        """,
        {**params, **qparams, "pool": pool_size},
    )
    return cur.fetchall()


def _doc_lexical(cur, qexpr: str, qparams: dict, where: str,
                 params: dict, pool_size: int):
    cur.execute(
        f"""
        SELECT d.cd, ts_rank_cd(d.tsv_stem, tq.q, 32) AS score
          FROM decision d, LATERAL (SELECT {qexpr} AS q) tq
         WHERE {where} AND d.tsv_stem @@ tq.q
         ORDER BY score DESC, d.cd
         LIMIT %(pool)s
        """,
        {**params, **qparams, "pool": pool_size},
    )
    return cur.fetchall()


def _semantic(cur, qvec, where: str, params: dict, pool_size: int):
    cur.execute(
        f"""
        SELECT c.cd, c.id AS chunk_id, c.ordinal, c.part,
               1 - (c.embedding <=> %(qv)s) AS score
          FROM chunk c
          JOIN decision d ON d.cd = c.cd
         WHERE {where} AND c.embedding IS NOT NULL
         ORDER BY c.embedding <=> %(qv)s
         LIMIT %(pool)s
        """,
        {**params, "qv": qvec, "pool": pool_size},
    )
    return cur.fetchall()


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


def _dedupe_keep_order(rows, key="cd") -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for r in rows:
        if r[key] not in seen:
            seen.add(r[key])
            order.append(r[key])
    return order


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
    where, params = filters.where()
    query = (query or "").strip()

    use_lexical = mode in ("hybrid", "keyword") and bool(query)
    use_semantic = mode in ("hybrid", "semantic") and bool(query)

    qvec = None
    if use_semantic:
        # Wrap in Vector so psycopg binds it as `vector`; a bare list would be
        # adapted to float8[] and the <=> operator would not resolve.
        qvec = Vector(get_backend().embed_queries([query])[0])

    with pool().connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # Widen HNSW search so that post-filtering doesn't starve results.
            # SET LOCAL takes no bind parameters, hence set_config().
            cur.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)",
                (str(max(pool_size, 100)),),
            )

            chunk_rows: list[dict] = []
            vec_rows: list[dict] = []
            ranked: dict[str, list[str]] = {}

            qexpr, qparams = "", {}
            if use_lexical:
                qexpr, qparams = qbuild.prepare(cur, query, expand=expand_query)
                chunk_rows = _chunk_lexical(cur, qexpr, qparams, where,
                                            params, pool_size)
                doc_rows = _doc_lexical(cur, qexpr, qparams, where,
                                        params, pool_size)

                # All query terms ANDed is the right default for precision, but
                # it is brittle for natural-language questions: one rare word
                # zeroes the whole lexical side. When AND is too selective,
                # retry disjunctively -- ts_rank_cd still favours documents
                # matching more terms, and RRF keeps the fusion stable.
                if len(chunk_rows) < min_lexical_hits and qbuild.is_expanded(qparams):
                    or_expr, or_params = qbuild.prepare(
                        cur, query, expand=True, conjunctive=False
                    )
                    if or_expr:
                        chunk_rows = _chunk_lexical(cur, or_expr, or_params,
                                                    where, params, pool_size)
                        doc_rows = _doc_lexical(cur, or_expr, or_params,
                                                where, params, pool_size)
                        qexpr, qparams = or_expr, or_params

                ranked["chunk_lexical"] = _dedupe_keep_order(chunk_rows)
                ranked["doc_lexical"] = _dedupe_keep_order(doc_rows)
            if use_semantic and qvec is not None:
                vec_rows = _semantic(cur, qvec, where, params, pool_size)
                ranked["semantic"] = _dedupe_keep_order(vec_rows)

            if not query:
                # Filter-only browse: most recent first.
                cur.execute(
                    f"""
                    SELECT d.cd FROM decision d
                     WHERE {where}
                     ORDER BY d.year DESC NULLS LAST, d.number DESC NULLS LAST
                     LIMIT %(lim)s
                    """,
                    {**params, "lim": limit},
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
            meta = _load_meta(cur, cds)
            passages = _collect_passages(
                cur, cds, chunk_rows, vec_rows, qexpr, qparams,
                passages_per_result, highlight and use_lexical,
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


def _load_meta(cur, cds: list[str]) -> dict[str, dict]:
    cur.execute(
        """
        SELECT cd, number, year, category, chamber, subject, summary,
               source_url, themes
          FROM decision_meta WHERE cd = ANY(%s)
        """,
        (cds,),
    )
    return {r["cd"]: r for r in cur.fetchall()}


def _collect_passages(
    cur,
    cds: list[str],
    chunk_rows: list[dict],
    vec_rows: list[dict],
    qexpr: str,
    qparams: dict,
    per_result: int,
    highlight: bool,
) -> dict[str, list[Passage]]:
    """Pick the best passages per decision from whichever retriever found them."""
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
    ids: list[int] = []
    for cd in cds:
        ranked = sorted(
            best[cd].values(),
            key=lambda p: (
                # Prefer the editorial headnote, then combined evidence.
                p.part != "summary",
                -((p.lexical_score or 0) * 10 + (p.semantic_score or 0)),
            ),
        )[:per_result]
        chosen[cd] = ranked
        ids.extend(p.chunk_id for p in ranked)

    if not ids:
        return chosen

    if highlight and qexpr:
        cur.execute(
            f"""
            SELECT id, content,
                   ts_headline('el_stem', content, {qexpr},
                               'MaxFragments=2, MaxWords=40, MinWords=18,'
                               'StartSel=**, StopSel=**, FragmentDelimiter= … ')
                   AS highlight
              FROM chunk WHERE id = ANY(%(ids)s)
            """,
            {**qparams, "ids": ids},
        )
    else:
        cur.execute(
            "SELECT id, content, NULL AS highlight FROM chunk WHERE id = ANY(%s)",
            (ids,),
        )
    texts = {r["id"]: r for r in cur.fetchall()}
    for plist in chosen.values():
        for p in plist:
            row = texts.get(p.chunk_id)
            if row:
                p.content = row["content"]
                p.highlight = row["highlight"]
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
    """Fetch decisions by opaque id or by citation (number/year)."""
    clauses, params = [], {}
    if cd:
        clauses.append("d.cd = %(cd)s")
        params["cd"] = cd
    if number is not None:
        clauses.append("d.number = %(number)s")
        params["number"] = number
    if year is not None:
        clauses.append("d.year = %(year)s")
        params["year"] = year
    if chamber:
        clauses.append("d.chamber = %(chamber)s")
        params["chamber"] = chamber
    if not clauses:
        raise ValueError("provide cd, or number and year")

    body_col = "d.body," if include_body else ""
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT d.cd, d.number, d.year, d.category, d.chamber, d.subject,
                   d.summary, {body_col} d.source_url, d.body_chars,
                   d.first_seen, d.last_fetched, m.themes
              FROM decision d
              JOIN decision_meta m ON m.cd = d.cd
             WHERE {' AND '.join(clauses)}
             ORDER BY d.year DESC, d.number DESC
            """,
            params,
        )
        return cur.fetchall()


def list_themes(prefix: str | None = None, limit: int = 100) -> list[dict]:
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.code, t.label,
                   -- Counted live rather than read from theme.n_decisions:
                   -- that column is only refreshed by a thematic crawl, so it
                   -- goes stale as soon as decisions are linked by any other
                   -- path. Agents rank themes by this number, so it has to be
                   -- true. decision_theme_by_theme makes it cheap.
                   count(dt.cd) AS n_decisions
              FROM theme t
              LEFT JOIN decision_theme dt ON dt.theme_code = t.code
             -- Casts are required: Postgres cannot infer a parameter's type
             -- from `$1 IS NULL` alone and raises AmbiguousParameter.
             WHERE (%(p)s::text IS NULL OR t.label ILIKE %(p)s::text)
             GROUP BY t.code, t.label
             ORDER BY count(dt.cd) DESC, t.label
             LIMIT %(lim)s
            """,
            {"p": f"%{prefix}%" if prefix else None, "lim": limit},
        )
        return cur.fetchall()
