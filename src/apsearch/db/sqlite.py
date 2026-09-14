"""SQLite + sqlite-vec database backend for zero-setup standalone deployments.

Key differences from Postgres
-----------------------------
* Stored in a single file on disk (default: `data/areios_pagos.db`).
* Zero daemon, zero ports, zero installation. Runs in-process inside the app.
* `sqlite-vec` provides C-level vector similarity search (`vec0` virtual table).
* `FTS5` provides fast full-text search. Greek diacritics are pre-folded into
  `content_folded` so accents/casing always match.
* WAL journal mode enables concurrent reads without blocking writes.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import sqlite_vec

from apsearch.config import settings
from apsearch.logging import get_logger

log = get_logger(__name__)

# ----------------------------------------------------------------------------
# sqlite3 module selection
# ----------------------------------------------------------------------------
# The stdlib `sqlite3` module on the official macOS and Windows Python builds
# (including the ones GitHub Actions / PyInstaller use) is compiled WITHOUT
# loadable-extension support: `Connection.enable_load_extension` simply does
# not exist, which crashes `sqlite_vec.load()` with
#   AttributeError: 'sqlite3.Connection' object has no attribute
#   'enable_load_extension'
# `pysqlite3-binary` ships its own statically-linked SQLite with extension
# loading compiled in and is API-compatible with stdlib `sqlite3` (dbapi2), so
# we prefer it whenever it's installed and silently fall back to stdlib
# `sqlite3` for environments where extension loading already works (e.g. the
# Linux system Python used in local dev/CI here).
try:
    import pysqlite3.dbapi2 as sqlite3  # type: ignore[import-not-found]
except ImportError:
    import sqlite3  # type: ignore[no-redef]

if not hasattr(sqlite3.Connection, "enable_load_extension"):
    raise ImportError(
        "No usable sqlite3 module found with loadable-extension support. "
        "Install `pysqlite3-binary` (pip install pysqlite3-binary)."
    )


def fold_greek(s: str | None) -> str:
    """Strip diacritics, lowercase, and normalize final sigma for FTS indexing."""
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", s).lower().replace("ς", "σ")


def ensure_seed_db(db_path: Path) -> None:
    """If the target database does not exist, initialize it from a seed archive if available."""
    if db_path.is_file() and db_path.stat().st_size > 0:
        return

    import gzip
    import shutil

    # Check for split multi-part archives (e.g. areios_pagos_seed.db.gz.part-aa, part-ab)
    for search_dir in [
        Path(__file__).resolve().parents[2] / "data",
        Path.cwd() / "data",
        Path.cwd(),
    ]:
        parts = sorted(search_dir.glob("areios_pagos_seed.db.gz.part-*"))
        target_gz = search_dir / "areios_pagos_seed.db.gz"
        if parts and not target_gz.is_file():
            log.info("Joining %d seed parts into %s...", len(parts), target_gz)
            with open(target_gz, "wb") as f_out:
                for p in parts:
                    with open(p, "rb") as f_in:
                        shutil.copyfileobj(f_in, f_out)

    candidates = [
        Path(__file__).resolve().parents[2] / "data" / "areios_pagos_seed.db.gz",
        Path(__file__).resolve().parents[2] / "data" / "areios_pagos.db",
        Path.cwd() / "data" / "areios_pagos_seed.db.gz",
        Path.cwd() / "areios_pagos_seed.db.gz",
    ]
    for c in candidates:
        if c.is_file():
            log.info("Seeding initial database from %s -> %s", c, db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            if c.name.endswith(".gz"):
                with gzip.open(c, "rb") as f_in, open(db_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copyfile(c, db_path)
            log.info("Database seeded successfully (%d bytes)", db_path.stat().st_size)
            return


def connect(
    db_path: str | Path | None = None, auto_seed: bool = True
) -> sqlite3.Connection:
    """Connect to SQLite and load sqlite-vec extension."""
    path = Path(db_path or settings.sqlite_file)
    if auto_seed and str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
        ensure_seed_db(path)

    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -64000")  # 64MB memory cache
    return conn


SCHEMA_SQL = """
-- ----------------------------------------------------------------------------
-- Decisions
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS decision (
    cd           TEXT PRIMARY KEY,
    number       INTEGER,
    year         INTEGER,
    category_id  INTEGER,
    chamber_id   INTEGER,
    category     TEXT,
    chamber      TEXT,
    subject      TEXT,
    summary      TEXT,
    body         TEXT,
    body_chars   INTEGER,
    source_url   TEXT NOT NULL,
    content_hash TEXT,
    indexed_hash TEXT,
    first_seen   TEXT NOT NULL DEFAULT (datetime('now')),
    last_fetched TEXT NOT NULL DEFAULT (datetime('now')),
    last_changed TEXT
);

CREATE INDEX IF NOT EXISTS decision_year_idx ON decision (year);
CREATE INDEX IF NOT EXISTS decision_citation_idx ON decision (number, year);
CREATE INDEX IF NOT EXISTS decision_pending_idx ON decision (cd) WHERE indexed_hash IS NOT content_hash;

-- ----------------------------------------------------------------------------
-- Thematic Index
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS theme (
    code        INTEGER PRIMARY KEY,
    label       TEXT NOT NULL,
    slug        TEXT,
    n_decisions INTEGER NOT NULL DEFAULT 0,
    crawled_at  TEXT
);

CREATE TABLE IF NOT EXISTS decision_theme (
    cd         TEXT NOT NULL REFERENCES decision(cd) ON DELETE CASCADE,
    theme_code INTEGER NOT NULL REFERENCES theme(code) ON DELETE CASCADE,
    PRIMARY KEY (cd, theme_code)
);

CREATE INDEX IF NOT EXISTS decision_theme_by_theme ON decision_theme (theme_code);

-- ----------------------------------------------------------------------------
-- Chunks
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunk (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    cd         TEXT NOT NULL REFERENCES decision(cd) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    part       TEXT NOT NULL DEFAULT 'body',
    char_start INTEGER,
    char_end   INTEGER,
    content    TEXT NOT NULL,
    UNIQUE (cd, ordinal)
);

CREATE INDEX IF NOT EXISTS chunk_cd_idx ON chunk (cd);

-- ----------------------------------------------------------------------------
-- Query Cache
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_cache (
    query_hash    TEXT PRIMARY KEY,
    model_sig     TEXT NOT NULL,
    query_text    TEXT NOT NULL,
    embedding     BLOB,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    last_accessed TEXT NOT NULL DEFAULT (datetime('now')),
    access_count  INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS query_cache_lru_idx ON query_cache (last_accessed);

-- ----------------------------------------------------------------------------
-- Metadata & Crawl Bookkeeping
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS index_meta (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS crawl_partition (
    year        INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    chamber_id  INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    n_found     INTEGER NOT NULL DEFAULT 0,
    truncated   INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (year, category_id, chamber_id)
);

CREATE TABLE IF NOT EXISTS fetch_queue (
    cd            TEXT PRIMARY KEY,
    number        INTEGER,
    year          INTEGER,
    category      TEXT,
    chamber       TEXT,
    discovered_at TEXT NOT NULL DEFAULT (datetime('now')),
    attempts      INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT
);

CREATE TABLE IF NOT EXISTS crawl_run (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL,
    started_at   TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at  TEXT,
    n_discovered INTEGER NOT NULL DEFAULT 0,
    n_fetched    INTEGER NOT NULL DEFAULT 0,
    n_changed    INTEGER NOT NULL DEFAULT 0,
    n_errors     INTEGER NOT NULL DEFAULT 0,
    notes        TEXT
);
"""


def init_sqlite_db(conn: sqlite3.Connection | None = None) -> None:
    """Initialize SQLite tables, virtual FTS5 tables, and virtual vec0 table."""
    dim = settings.embed_dim
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True

    try:
        conn.executescript(SCHEMA_SQL)

        # FTS5 tables for keyword search
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                content_folded,
                content UNINDEXED,
                cd UNINDEXED,
                part UNINDEXED,
                tokenize = "unicode61"
            )
            """
        )

        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS decision_fts USING fts5(
                subject_folded,
                summary_folded,
                body_folded,
                cd UNINDEXED,
                tokenize = "unicode61"
            )
            """
        )

        # Vector virtual table
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(embedding float[{dim}])"
        )
        conn.commit()
        log.info("SQLite schema initialized successfully (dim=%d)", dim)
    finally:
        if close_after:
            conn.close()
