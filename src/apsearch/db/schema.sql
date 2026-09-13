-- ============================================================================
-- apsearch :: Areios Pagos case-law index
-- Postgres 16+ with pgvector. All components open source.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ----------------------------------------------------------------------------
-- Text search configuration for Greek.
--
-- Postgres ships a Greek snowball stemmer, but it is inconsistent on the
-- inflections that dominate legal Greek (e.g. "αδικοπραξία" -> αδικοπραξ while
-- "αδικοπραξίας" -> αδικοπραξι).  We therefore:
--   1. run `unaccent` first so that accented / unaccented / ALL-CAPS forms
--      (Greek capitals drop accents) collapse to one surface form, and
--   2. keep a second, *unstemmed* configuration so that exact legal terms and
--      citations can still be matched verbatim.
-- Recall gaps left by the stemmer are covered by dense vector search.
-- ----------------------------------------------------------------------------

DROP TEXT SEARCH CONFIGURATION IF EXISTS el_stem CASCADE;
CREATE TEXT SEARCH CONFIGURATION el_stem (COPY = greek);
ALTER TEXT SEARCH CONFIGURATION el_stem
    ALTER MAPPING FOR hword, hword_part, word, asciiword, asciihword
    WITH unaccent, greek_stem;

DROP TEXT SEARCH CONFIGURATION IF EXISTS el_exact CASCADE;
CREATE TEXT SEARCH CONFIGURATION el_exact (COPY = simple);
ALTER TEXT SEARCH CONFIGURATION el_exact
    ALTER MAPPING FOR hword, hword_part, word, asciiword, asciihword
    WITH unaccent, simple;

-- ----------------------------------------------------------------------------
-- Reference data
-- ----------------------------------------------------------------------------

-- Top-level case category (the site's X_TMHMA select).
CREATE TABLE IF NOT EXISTS case_category (
    id    smallint PRIMARY KEY,
    label text NOT NULL
);

INSERT INTO case_category (id, label) VALUES
    (1, 'ΠΟΛΙΤΙΚΕΣ'),
    (2, 'ΠΟΙΝΙΚΕΣ'),
    (3, 'Νόμου 3068/2002'),
    (4, 'Νόμου 4239/2014'),
    (5, 'Πράξεις Νόμου 4842/2021')
ON CONFLICT (id) DO UPDATE SET label = EXCLUDED.label;

-- Chamber / section (the site's X_SUB_TMHMA select).
CREATE TABLE IF NOT EXISTS chamber (
    id    smallint PRIMARY KEY,
    label text NOT NULL
);

INSERT INTO chamber (id, label) VALUES
    (2, 'Α'), (13, 'Α1'), (12, 'Α2'), (14, 'Α3'),
    (3, 'Β'), (4, 'Β1'), (5, 'Β2'),
    (6, 'Γ'), (7, 'Δ'), (8, 'Ε'), (9, 'ΣΤ'), (10, 'Ζ'),
    (11, 'ΟΛΟΜΕΛΕΙΑ'),
    (15, 'Α Ποιν. Διακ.'), (16, 'Β Ποιν. Διακ.')
ON CONFLICT (id) DO UPDATE SET label = EXCLUDED.label;

-- The site's controlled vocabulary of legal subjects (λήμματα).
CREATE TABLE IF NOT EXISTS theme (
    code       integer PRIMARY KEY,          -- site's `code` query param
    label      text NOT NULL,
    slug       text,
    n_decisions integer NOT NULL DEFAULT 0,
    crawled_at timestamptz
);

CREATE INDEX IF NOT EXISTS theme_label_trgm ON theme USING gin (label gin_trgm_ops);

-- ----------------------------------------------------------------------------
-- Decisions
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS decision (
    -- The site's stable opaque id (`cd` query parameter). Verified stable
    -- across sessions, so it is a safe natural primary key.
    cd            text PRIMARY KEY,

    number        integer,
    year          integer,
    category_id   smallint REFERENCES case_category(id),
    chamber_id    smallint REFERENCES chamber(id),
    category      text,      -- denormalised label as published
    chamber       text,

    subject       text,      -- "Θέμα": comma-separated legal subjects
    summary       text,      -- "Περίληψη": editorial headnote
    body          text,      -- full decision text
    body_chars    integer GENERATED ALWAYS AS (length(body)) STORED,

    source_url    text NOT NULL,
    content_hash  text,      -- sha256 of body; detects silent edits

    first_seen    timestamptz NOT NULL DEFAULT now(),
    last_fetched  timestamptz NOT NULL DEFAULT now(),
    last_changed  timestamptz,

    -- Set when the embedding pipeline has processed the current content_hash.
    indexed_hash  text,

    -- Weighted lexical index: subject > summary > body.
    tsv_stem tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('el_stem'::regconfig, coalesce(subject, '')), 'A') ||
        setweight(to_tsvector('el_stem'::regconfig, coalesce(summary, '')), 'B') ||
        setweight(to_tsvector('el_stem'::regconfig, coalesce(body, '')),    'C')
    ) STORED,

    tsv_exact tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('el_exact'::regconfig, coalesce(subject, '')), 'A') ||
        setweight(to_tsvector('el_exact'::regconfig, coalesce(summary, '')), 'B') ||
        setweight(to_tsvector('el_exact'::regconfig, coalesce(body, '')),    'C')
    ) STORED
);

CREATE INDEX IF NOT EXISTS decision_tsv_stem_idx  ON decision USING gin (tsv_stem);
CREATE INDEX IF NOT EXISTS decision_tsv_exact_idx ON decision USING gin (tsv_exact);
CREATE INDEX IF NOT EXISTS decision_year_idx      ON decision (year);
CREATE INDEX IF NOT EXISTS decision_citation_idx  ON decision (number, year);
CREATE INDEX IF NOT EXISTS decision_category_idx  ON decision (category_id, chamber_id);
CREATE INDEX IF NOT EXISTS decision_pending_idx   ON decision (cd)
    WHERE indexed_hash IS DISTINCT FROM content_hash;

-- many-to-many decision <-> theme, populated from the thematic index crawl
CREATE TABLE IF NOT EXISTS decision_theme (
    cd         text    NOT NULL REFERENCES decision(cd) ON DELETE CASCADE,
    theme_code integer NOT NULL REFERENCES theme(code)  ON DELETE CASCADE,
    PRIMARY KEY (cd, theme_code)
);

CREATE INDEX IF NOT EXISTS decision_theme_by_theme ON decision_theme (theme_code);

-- ----------------------------------------------------------------------------
-- Chunks (passage-level retrieval + embeddings)
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS chunk (
    id        bigserial PRIMARY KEY,
    cd        text    NOT NULL REFERENCES decision(cd) ON DELETE CASCADE,
    ordinal   integer NOT NULL,
    -- 'subject' | 'summary' | 'body'
    part      text    NOT NULL DEFAULT 'body',
    char_start integer,
    char_end   integer,
    content   text    NOT NULL,
    embedding vector,          -- dimension enforced by the index, see migrate()

    tsv_stem tsvector GENERATED ALWAYS AS (
        to_tsvector('el_stem'::regconfig, content)
    ) STORED,

    UNIQUE (cd, ordinal)
);

CREATE INDEX IF NOT EXISTS chunk_cd_idx       ON chunk (cd);
CREATE INDEX IF NOT EXISTS chunk_tsv_stem_idx ON chunk USING gin (tsv_stem);

-- Records which model produced the vectors currently stored, so that a config
-- change cannot silently mix two incompatible embedding spaces in one index.
CREATE TABLE IF NOT EXISTS index_meta (
    key        text PRIMARY KEY,
    value      text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- ----------------------------------------------------------------------------
-- Crawl bookkeeping
-- ----------------------------------------------------------------------------

-- One row per (year, category, chamber) search partition, so that a backfill
-- can be interrupted and resumed, and so we can detect the site's 3000-row cap.
CREATE TABLE IF NOT EXISTS crawl_partition (
    year        integer  NOT NULL,
    category_id smallint NOT NULL,
    chamber_id  smallint NOT NULL,
    status      text     NOT NULL DEFAULT 'pending',  -- pending|done|truncated|error
    n_found     integer  NOT NULL DEFAULT 0,
    truncated   boolean  NOT NULL DEFAULT false,
    error       text,
    updated_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (year, category_id, chamber_id)
);

CREATE INDEX IF NOT EXISTS crawl_partition_status ON crawl_partition (status);

-- Decisions discovered by a listing but whose full text is not fetched yet.
CREATE TABLE IF NOT EXISTS fetch_queue (
    cd          text PRIMARY KEY,
    number      integer,
    year        integer,
    category    text,
    chamber     text,
    discovered_at timestamptz NOT NULL DEFAULT now(),
    attempts    integer NOT NULL DEFAULT 0,
    last_error  text
);

CREATE TABLE IF NOT EXISTS crawl_run (
    id         bigserial PRIMARY KEY,
    kind       text NOT NULL,               -- backfill|incremental|themes
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    n_discovered integer NOT NULL DEFAULT 0,
    n_fetched    integer NOT NULL DEFAULT 0,
    n_changed    integer NOT NULL DEFAULT 0,
    n_errors     integer NOT NULL DEFAULT 0,
    notes      text
);

-- ----------------------------------------------------------------------------
-- Convenience view
-- ----------------------------------------------------------------------------

CREATE OR REPLACE VIEW decision_meta AS
SELECT
    d.cd,
    d.number,
    d.year,
    d.number || '/' || d.year AS citation,
    d.category,
    d.chamber,
    d.subject,
    d.summary,
    d.body_chars,
    d.source_url,
    d.first_seen,
    d.last_fetched,
    coalesce(
        array_agg(t.label ORDER BY t.label) FILTER (WHERE t.label IS NOT NULL),
        '{}'
    ) AS themes
FROM decision d
LEFT JOIN decision_theme dt ON dt.cd = d.cd
LEFT JOIN theme t ON t.code = dt.theme_code
GROUP BY d.cd;
