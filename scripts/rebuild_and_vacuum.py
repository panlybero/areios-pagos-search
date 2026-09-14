import sqlite3
import sys
import time
from pathlib import Path

from apsearch.db.sqlite import fold_greek

DB_PATH = Path("data/areios_pagos.db")
sys.stdout.reconfigure(line_buffering=True)


def main():
    print("Opening database...")
    conn = sqlite3.connect(str(DB_PATH), timeout=60.0)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    print("Recreating decision_fts with content=''...")
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
    conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM decision")
    total = cur.fetchone()[0]
    print(f"Indexing {total:,} decisions in batches...")

    batch_size = 3000
    offset = 0
    t0 = time.time()
    while offset < total:
        cur.execute(
            "SELECT rowid, subject, summary, body FROM decision LIMIT ? OFFSET ?",
            (batch_size, offset),
        )
        rows = cur.fetchall()
        if not rows:
            break
        payload = [
            (r[0], fold_greek(r[1]), fold_greek(r[2]), fold_greek(r[3]))
            for r in rows
        ]
        conn.executemany(
            "INSERT INTO decision_fts(rowid, subject_folded, summary_folded, body_folded) VALUES (?, ?, ?, ?)",
            payload,
        )
        conn.commit()
        offset += len(rows)
        print(f"  {offset:,} / {total:,} decisions indexed ({offset/total*100:.1f}%)...")

    print(f"decision_fts complete in {time.time()-t0:.1f}s!")

    print("Starting VACUUM...")
    t0 = time.time()
    conn.execute("VACUUM")
    conn.commit()
    conn.close()

    size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"VACUUM COMPLETE in {time.time()-t0:.1f}s! New DB size: {size_mb:.0f} MB ({size_mb/1024:.2f} GB)")


if __name__ == "__main__":
    main()
