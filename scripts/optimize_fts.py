"""Rebuild FTS tables as contentless (content="") to eliminate 3.5 GB of duplicate text."""

import sqlite3
import time
from pathlib import Path

from apsearch.db.sqlite import fold_greek

DB_PATH = Path("data/areios_pagos.db")


def main() -> None:
    print(f"Opening {DB_PATH} (current size: {DB_PATH.stat().st_size / (1024**3):.2f} GB)...")
    conn = sqlite3.connect(str(DB_PATH), timeout=60.0)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -128000")  # 128MB RAM cache

    # 1. Decision FTS
    print("Rebuilding decision_fts with content=''...")
    t0 = time.time()
    conn.execute("DROP TABLE IF EXISTS decision_fts")
    conn.execute(
        """
        CREATE VIRTUAL TABLE decision_fts USING fts5(
            subject_folded,
            summary_folded,
            body_folded,
            content="",
            tokenize = "unicode61"
        )
        """
    )
    cur = conn.cursor()
    cur.execute("SELECT rowid, subject, summary, body FROM decision")
    dec_rows = cur.fetchall()
    payload_dec = [
        (r[0], fold_greek(r[1]), fold_greek(r[2]), fold_greek(r[3]))
        for r in dec_rows
    ]
    conn.executemany(
        "INSERT INTO decision_fts(rowid, subject_folded, summary_folded, body_folded) VALUES (?, ?, ?, ?)",
        payload_dec,
    )
    conn.commit()
    print(f"  Indexed {len(payload_dec):,} decisions in {time.time()-t0:.1f}s")

    # 2. Chunk FTS
    print("Rebuilding chunk_fts with content=''...")
    t0 = time.time()
    conn.execute("DROP TABLE IF EXISTS chunk_fts")
    conn.execute(
        """
        CREATE VIRTUAL TABLE chunk_fts USING fts5(
            content_folded,
            content="",
            tokenize = "unicode61"
        )
        """
    )

    batch_size = 50000
    offset = 0
    cur.execute("SELECT count(*) FROM chunk")
    total_chunks = cur.fetchone()[0]

    while offset < total_chunks:
        cur.execute("SELECT id, content FROM chunk LIMIT ? OFFSET ?", (batch_size, offset))
        chunk_rows = cur.fetchall()
        if not chunk_rows:
            break
        payload_chunks = [(r[0], fold_greek(r[1])) for r in chunk_rows]
        conn.executemany(
            "INSERT INTO chunk_fts(rowid, content_folded) VALUES (?, ?)",
            payload_chunks,
        )
        conn.commit()
        offset += len(chunk_rows)
        print(f"  Indexed {offset:,} / {total_chunks:,} chunks ({offset/total_chunks*100:.1f}%)...")

    print(f"  Total chunks indexed in {time.time()-t0:.1f}s")

    # 3. VACUUM to reclaim space
    print("Running VACUUM to reclaim ~3.5 GB of deleted duplicate text...")
    t0 = time.time()
    conn.execute("VACUUM")
    conn.commit()
    conn.close()

    new_size = DB_PATH.stat().st_size
    print(f"VACUUM complete in {time.time()-t0:.1f}s!")
    print(f"New DB size: {new_size / (1024**3):.2f} GB ({new_size / (1024**2):.0f} MB)")


if __name__ == "__main__":
    main()
