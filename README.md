# apsearch — Areios Pagos case-law search

Hybrid keyword + semantic search over the published case law of the **Άρειος
Πάγος** (Supreme Court of Greece), 1994–present, with an MCP server so agents
can query it directly.

Everything in the retrieval stack is open source (Postgres + pgvector + Greek
full-text search). The embedding model is pluggable: a hosted one for speed, or
a fully local ONNX model if you want zero external dependencies.

## Why it is built this way

The source site (`areiospagos.gr`) is a legacy ASP application, and several of
its properties dictated the design:

- **Search results are capped at 3000 rows** with no pagination, and the
  truncation is announced only in a Greek sentence at the bottom of the page.
  Discovery is therefore an *adaptive partition search* (`crawler/discover.py`):
  query by year, and split by category → chamber → decision-number range only
  when a response comes back truncated.
- **`cd` identifiers are stable** across sessions (verified), so they are used
  as primary keys and permanent links.
- **Pages are windows-1253**, frequently without a charset header, and the HTML
  is unbalanced. Parsing is regex-based on raw markup rather than DOM-based,
  because DOM repair differs between pages.
- **Postgres' Greek stemmer is unreliable** for legal Greek, and its stopword
  list is empty. Both are worked around in `search/query.py` — see below.

## Architecture

```
areiospagos.gr
      │  polite crawler (0.5 req/s, archived responses)
      ▼
  raw HTML archive ──► parsers ──► Postgres
                                    ├── decision   (text, headnote, subjects)
                                    ├── chunk      (passages + embeddings)
                                    └── theme      (court's own vocabulary)
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
              chunk lexical FTS     doc-level FTS          pgvector
                    └──────────── RRF fusion ─────────────────┘
                                    │
                        CLI  ·  REST API  ·  MCP server
```

### Retrieval

Three retrievers are fused with Reciprocal Rank Fusion:

| Retriever | Good at |
| --------- | ------- |
| `chunk_lexical` | citations, statute numbers, exact terms of art |
| `doc_lexical`   | relevance spread thinly across a long decision |
| `semantic`      | paraphrase, and the morphology the stemmer drops |

RRF is used instead of score interpolation because `ts_rank_cd` is unbounded
while cosine is `[-1,1]`; RRF needs no per-corpus tuning to stay stable.

### The two Greek-language problems

**1. Unstable stemming.** Postgres stems the same lemma inconsistently:

```
αδικοπραξία → αδικοπραξ      αδικοπραξίας → αδικοπραξι
αναίρεση    → αναιρεσ        αναιρεσείων  → αναιρεσει
```

A nominative query silently misses genitive documents — and the genitive is the
*normal* form in Greek legal prose (`λόγος αναιρέσεως`, `αγωγή αδικοπραξίας`).
Since the unstable part is always a suffix, queries are expanded to prefix
lexemes (`αδικοπραξ:*`), which unifies the variants.

**2. No stopword list.** `to_tsvector('greek', 'του την και στο')` yields
`τ, την, κα, στ` — present in 100% of documents. These are filtered at query
time, on **surface forms, before stemming**, because after stemming the
information needed to do it safely is gone:

```
από (preposition) → απ    |    ΑΠ (Άρειος Πάγος)       → απ
αν  ("if")        → αν    |    ΑΝ (Αναγκαστικός Νόμος) → αν
```

Filtering the stem `απ` would delete the court's own name from queries. Short
all-caps tokens are therefore always treated as legal abbreviations, keeping
`ΑΚ`, `ΠΚ`, `ΚΠολΔ`, `ΚΠΔ`, `ΝΔ`, `ΕΣΔΑ` searchable.

Stopwords are filtered from queries only, not from the index: managed Postgres
(Cloud SQL) gives no filesystem access to install a stopword file.

## Crawling policy

The origin is a single IIS box run by a public institution, and there is **no
`robots.txt`** — no crawl budget has been granted, so the crawler imposes a
conservative one on itself:

- one request in flight at a time, ≥2 s apart (~0.5 req/s), with jitter;
- exponential backoff on 5xx/429, honouring `Retry-After`, and the penalty
  applies globally rather than to the offending call only;
- **every response is archived**, so re-runs, parser fixes and tests never
  re-hit the origin — the single most effective politeness measure available;
- `APSEARCH_MAX_REQUESTS_PER_RUN` as a runaway-job circuit breaker;
- set `APSEARCH_CONTACT_EMAIL`; it goes in the `User-Agent` so the site's
  operators can reach you instead of silently blocking you.

A full backfill is ~66k decisions ≈ 37 hours of wall-clock at this rate. It is
resumable — interrupt it freely. The daily poll costs ~10 requests.

## Quick start

```bash
docker compose up -d                 # Postgres 17 + pgvector
cp .env.example .env                 # add APSEARCH_GEMINI_API_KEY
pip install -e ".[api,mcp]"

apsearch db migrate
apsearch crawl themes --limit 5      # court's subject vocabulary
apsearch crawl backfill --year-from 2024 --fetch-limit 50
apsearch index build
apsearch search "παραγραφή αξιώσεων κατά του Δημοσίου"
```

### Commands

```
apsearch db migrate|stats
apsearch crawl backfill|poll|fetch|themes
apsearch index build|reset|vector-index
apsearch search <query> [--mode hybrid|keyword|semantic] [--year-from ...]
apsearch get 144/2015 [--body]
apsearch themes [prefix]
apsearch serve                       # REST API
apsearch mcp [--transport stdio|http]
```

## MCP server

```jsonc
// claude_desktop_config.json
{
  "mcpServers": {
    "areios-pagos": {
      "command": "apsearch",
      "args": ["mcp"],
      "env": { "APSEARCH_PG_PORT": "5433", "APSEARCH_GEMINI_API_KEY": "..." }
    }
  }
}
```

Tools: `search_decisions`, `get_decision`, `list_themes`, `corpus_stats`.
Verified against protocol `2025-11-25` over a real stdio handshake.

Search returns metadata plus matching passages, never full texts — a single
decision runs to ~37k characters. Agents fetch full text only for the
decisions they actually need.

Use an absolute path to the executable: MCP clients do not run a login shell.
See [`docs/mcp.md`](docs/mcp.md) for remote (streamable-HTTP) setup and the
tool-design rationale.

## Embedding backends

| | `gemini` (default in `.env.example`) | `fastembed` (fully open source) |
| --- | --- | --- |
| Model | `gemini-embedding-2`, 768 dim | `multilingual-e5-small`, 384 dim |
| Context | 8192 tokens | 512 tokens |
| Throughput | ~200–270 chunks/s | ~4 chunks/s per vCPU |
| Cost | ~$100 for a full backfill | free |
| Needs | API key | ~500 MB RAM |

Switching backends invalidates every stored vector. `index_meta` records which
model produced the index and refuses to mix embedding spaces — run
`apsearch index reset` to re-embed.

> Note: `gemini-embedding-001` returns **unnormalised** vectors when
> MRL-truncated below 3072 dims (observed norm ≈ 0.59). `gemini.py`
> renormalises unconditionally.

## Data model

`decision` (one row per judgment, with the editorial `Θέμα` subjects and
`Περίληψη` headnote), `chunk` (passages + vectors), `theme` (the court's own
controlled vocabulary of ~713 legal subjects, which is far better than anything
inferred), plus crawl bookkeeping (`crawl_partition`, `fetch_queue`,
`crawl_run`) that makes every stage resumable.

Chunking splits on the decisions' rhetorical skeleton — `ΣΚΕΦΘΗΚΕ ΣΥΜΦΩΝΑ ΜΕ ΤΟ
ΝΟΜΟ`, `ΓΙΑ ΤΟΥΣ ΛΟΓΟΥΣ ΑΥΤΟΥΣ` — so a chunk rarely straddles two unrelated
legal points. The headnote becomes its own chunk, since it is an editorial
abstract of exactly what the decision holds.

## Tests

```bash
pytest            # 70 tests; 60 need neither network nor database
```

Integration tests auto-skip when no Postgres is reachable
(`docker compose up -d && apsearch db migrate` to enable them).

Parser tests run against saved fixtures. Note `test_no_runaway_duplication`:
the chunker once emitted 757 overlapping spans for one decision (2.1x text
coverage, some spans one character long) because the packing cursor could stall
and creep forward one character at a time. That is a 7x embedding-cost bug that
is invisible without an explicit invariant, so it has a named regression test.

## Deployment

Not deployed. See [`docs/deploy-gcp.md`](docs/deploy-gcp.md) for the intended
Cloud Run + Cloud SQL shape; the config layer already supports Cloud SQL unix
sockets, GCS-backed response archiving, structured Cloud Logging output, and
`PORT`.

## Legal note

Decisions are published by the court and are anonymised at source. This project
only mirrors and indexes them. It is not affiliated with the Άρειος Πάγος, and
the official text on `areiospagos.gr` remains the only authoritative version.

## Licence

MIT
