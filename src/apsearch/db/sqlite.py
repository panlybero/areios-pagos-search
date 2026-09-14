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

import re
import time
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
# `sqlean.py` bundles its own recent SQLite build with extension loading
# compiled in and is dbapi2-API-compatible with stdlib `sqlite3` (verified:
# enable_load_extension/load_extension, Row factory, FTS5, executemany all
# behave identically). It ships real wheels for macOS -- including Apple
# Silicon arm64 -- unlike `pysqlite3-binary`, which is Linux-only. We prefer
# it whenever installed and silently fall back to stdlib `sqlite3` for
# environments where extension loading already works (e.g. the Linux system
# Python used in local dev/CI here).
try:
    import sqlean as sqlite3  # type: ignore[import-not-found]
except ImportError:
    import sqlite3  # type: ignore[no-redef]

if not hasattr(sqlite3.Connection, "enable_load_extension"):
    raise ImportError(
        "No usable sqlite3 module found with loadable-extension support. "
        "Install `sqlean.py` (pip install sqlean.py)."
    )


def fold_greek(s: str | None) -> str:
    """Strip diacritics, lowercase, and normalize final sigma for FTS indexing."""
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", s).lower().replace("ς", "σ")


#: Where the pre-built corpus lives. Bumping this requires a matching release
#: with these two exact asset names (GitHub caps a single release asset at 2GB,
#: hence the split).
SEED_RELEASE_TAG = "v0.3.0"
SEED_RELEASE_BASE = (
    f"https://github.com/panlybero/areios-pagos-search/releases/download/{SEED_RELEASE_TAG}"
)
SEED_PART_NAMES = ["areios_pagos_seed.db.gz.part-aa", "areios_pagos_seed.db.gz.part-ab"]

#: The real corpus is ~7GB uncompressed with 26k+ decisions. Anything smaller
#: or emptier than this is not our data -- most likely an empty schema that
#: got auto-created by `sqlite3.connect()` on a path that was never seeded
#: (e.g. the executable was launched directly rather than via launch.command,
#: so the network fetch never ran).
_MIN_HEALTHY_BYTES = 500_000_000
_MIN_HEALTHY_DECISIONS = 1000


def _db_is_healthy(db_path: Path) -> bool:
    """Sanity-check that `db_path` actually holds the real corpus.

    A SQLite file that merely *exists* proves nothing: `sqlite3.connect()`
    silently creates an empty file for any path that doesn't exist yet, and
    once a virtual table like `chunk_vec` is created inside it (possibly at
    the wrong embedding dimension, if that happened before a config fix), it
    persists forever across relaunches because every later `CREATE VIRTUAL
    TABLE IF NOT EXISTS` is a silent no-op against it.
    """
    if not db_path.is_file() or db_path.stat().st_size < _MIN_HEALTHY_BYTES:
        return False
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        try:
            n = conn.execute("SELECT count(*) FROM decision").fetchone()[0]
            return n >= _MIN_HEALTHY_DECISIONS
        finally:
            conn.close()
    except Exception:
        return False  # missing table, corrupt file, locked, etc.


def _discard_unhealthy_db(db_path: Path) -> None:
    log.warning(
        "Database at %s exists but does not look like the real corpus "
        "(too small, or missing/empty tables) -- discarding and re-seeding.",
        db_path,
    )
    for suffix in ("", "-wal", "-shm"):
        p = Path(f"{db_path}{suffix}")
        if p.exists():
            p.unlink()


def _join_local_parts(search_dir: Path) -> Path | None:
    import shutil

    parts = sorted(search_dir.glob("areios_pagos_seed.db.gz.part-*"))
    if not parts:
        return None
    target_gz = search_dir / "areios_pagos_seed_joined.db.gz"
    log.info("Joining %d local seed parts into %s...", len(parts), target_gz)
    with open(target_gz, "wb") as f_out:
        for p in parts:
            with open(p, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)
    return target_gz


def _download_seed_from_github(dest_dir: Path) -> Path | None:
    """Fetch the split seed archive parts straight from the GitHub release.

    This is the fallback for the common real-world launch path: the user
    double-clicks the compiled executable directly (or runs it from a
    Terminal at an arbitrary cwd) rather than the `launch.command` wrapper
    script, so the shell-level download-and-join logic there never runs.
    Doing the fetch here in Python means the app is self-sufficient
    regardless of how it was started.
    """
    try:
        import httpx
    except ImportError:
        log.error("httpx not available; cannot download the seed database.")
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    joined = dest_dir / "areios_pagos_seed_download.db.gz"
    try:
        with open(joined, "wb") as out:
            for part_name in SEED_PART_NAMES:
                # A cache-busting query param avoids ever getting stuck behind a
                # stale negative (404) response GitHub's edge cached for the bare
                # URL -- observed in practice shortly after a release/repo change.
                url = f"{SEED_RELEASE_BASE}/{part_name}?t={int(time.time())}"
                log.info("Downloading %s (this happens once)...", part_name)
                with httpx.stream("GET", url, follow_redirects=True, timeout=180.0) as resp:
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length", 0))
                    downloaded = 0
                    last_logged = -1
                    for chunk in resp.iter_bytes(chunk_size=4 * 1024 * 1024):
                        out.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            if pct >= last_logged + 10:
                                log.info("  %s: %d%% (%d MB / %d MB)",
                                         part_name, pct, downloaded // (1024 * 1024),
                                         total // (1024 * 1024))
                                last_logged = pct
        return joined
    except Exception as exc:
        log.error("Downloading the seed database failed: %s", exc)
        joined.unlink(missing_ok=True)
        return None


def ensure_seed_db(db_path: Path) -> None:
    """Make sure `db_path` holds the real pre-built corpus, fetching it if not."""
    if _db_is_healthy(db_path):
        return
    if db_path.exists():
        _discard_unhealthy_db(db_path)

    import gzip
    import shutil

    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. A plain, already-extracted database sitting next to the app.
    for search_dir in (Path(__file__).resolve().parents[2] / "data", Path.cwd() / "data", Path.cwd()):
        plain = search_dir / "areios_pagos.db"
        if plain.is_file() and plain.resolve() != db_path.resolve():
            log.info("Seeding from local database %s -> %s", plain, db_path)
            shutil.copyfile(plain, db_path)
            if _db_is_healthy(db_path):
                return
            _discard_unhealthy_db(db_path)

    # 2. A local, already-downloaded .gz (whole or split into .part-*).
    gz_source: Path | None = None
    for search_dir in (Path(__file__).resolve().parents[2] / "data", Path.cwd() / "data", Path.cwd()):
        whole = search_dir / "areios_pagos_seed.db.gz"
        if whole.is_file():
            gz_source = whole
            break
        joined = _join_local_parts(search_dir)
        if joined is not None:
            gz_source = joined
            break

    # 3. Nothing local: fetch it from the GitHub release ourselves.
    if gz_source is None:
        log.info(
            "No local seed archive found -- downloading the pre-built case-law "
            "database from GitHub (~2.5 GB, one-time)."
        )
        gz_source = _download_seed_from_github(db_path.parent)

    if gz_source is None:
        log.error(
            "Could not obtain the seed database. Starting with an empty local "
            "database -- search will find nothing until a backfill is run."
        )
        return

    log.info("Extracting %s -> %s...", gz_source, db_path)
    with gzip.open(gz_source, "rb") as f_in, open(db_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    if gz_source.name == "areios_pagos_seed_download.db.gz":
        gz_source.unlink(missing_ok=True)  # one-time download artifact; the .db is what matters now

    if _db_is_healthy(db_path):
        log.info("Database seeded successfully (%d bytes)", db_path.stat().st_size)
    else:
        log.error("Seeded database still fails the health check -- something is wrong with the archive.")


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

        # Vector virtual table. `CREATE ... IF NOT EXISTS` is a silent no-op
        # against an existing table, so if one already exists at a different
        # dimension (e.g. an empty stub created before a config fix, or a
        # switched embedding model), fail loudly here instead of letting it
        # surface later as a cryptic dimension-mismatch error deep inside a
        # search query.
        existing = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'chunk_vec' AND type = 'table'"
        ).fetchone()
        if existing is not None:
            existing_sql = existing[0] or ""
            m = re.search(r"float\[(\d+)\]", existing_sql)
            if m and int(m.group(1)) != dim:
                raise RuntimeError(
                    f"chunk_vec already exists with {m.group(1)}-dimensional vectors, "
                    f"but the configured embedding backend produces {dim}-dimensional "
                    f"ones. This database is stale (likely created before a config fix, "
                    f"or with a different embedding model). Delete it and relaunch to "
                    f"re-seed automatically:\n  rm {settings.sqlite_file}*"
                )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(embedding float[{dim}])"
        )
        conn.commit()
        log.info("SQLite schema initialized successfully (dim=%d)", dim)
    finally:
        if close_after:
            conn.close()
