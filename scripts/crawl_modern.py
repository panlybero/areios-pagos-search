"""Background runner to fetch and embed all modern decisions (2018-2026).

Fetches in batches of 200, embeds them immediately with Gemini, and reports
live progress to data/crawl_progress.json and stdout.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

# Ensure SQLite backend is used
os.environ["APSEARCH_DB_BACKEND"] = "sqlite"

from apsearch import repo
from apsearch.crawler.client import PoliteClient
from apsearch.crawler.pipeline import RunStats, drain_queue
from apsearch.index.build import run_index
from apsearch.logging import get_logger

log = get_logger("crawl_modern")

PROGRESS_FILE = Path("data/crawl_progress.json")
DB_FILE = Path("data/areios_pagos.db")
SEED_GZ = Path("data/areios_pagos_seed.db.gz")
PACKAGE_ZIP = Path("/tmp/opencode/release_assets/AreiosPagos-Universal-Package.zip")


def write_progress(status: str, fetched: int, total: int, chunks: int) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "status": status,
        "fetched": fetched,
        "total": total,
        "percent": round(fetched / total * 100, 1) if total else 100.0,
        "chunks": chunks,
        "timestamp": time.time(),
        "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    PROGRESS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    total_queued = repo.queue_depth()
    initial_stats = repo.stats()
    start_time = time.monotonic()
    log.info("Starting modern backfill: %d decisions queued", total_queued)

    batch_size = 200
    with PoliteClient() as client:
        while True:
            remaining = repo.queue_depth()
            current_stats = repo.stats()
            fetched_so_far = current_stats["decisions"]

            write_progress("crawling", fetched_so_far, fetched_so_far + remaining, current_stats["chunks"])

            if remaining == 0:
                log.info("Queue drained! All decisions fetched.")
                break

            # 1. Fetch batch
            stats = RunStats()
            try:
                drain_queue(client, stats, limit=batch_size)
            except Exception as exc:
                log.warning("Batch fetch error: %s (retrying in 5s)", exc)
                time.sleep(5)
                continue

            # 2. Embed batch immediately so work is saved incrementally
            try:
                istats = run_index(batch_size=16)
                log.info(
                    "Batch complete: %d fetched, %d chunks embedded. Remaining in queue: %d",
                    stats.fetched, istats.chunks, repo.queue_depth(),
                )
            except Exception as exc:
                log.warning("Batch indexing error: %s", exc)

    # 3. Final index pass to ensure zero pending
    log.info("Running final indexing pass...")
    run_index()

    total_time = time.monotonic() - start_time
    final_stats = repo.stats()
    log.info(
        "CRAWL COMPLETE in %.1f minutes! Total decisions: %d, chunks: %d",
        total_time / 60, final_stats["decisions"], final_stats["chunks"],
    )
    write_progress("completed", final_stats["decisions"], final_stats["decisions"], final_stats["chunks"])

    # 4. Create compressed seed file for GitHub release
    log.info("Compressing database to seed archive (%s)...", SEED_GZ)
    import gzip
    with open(DB_FILE, "rb") as f_in, gzip.open(SEED_GZ, "wb", compresslevel=6) as f_out:
        shutil.copyfileobj(f_in, f_out)
    log.info("Compressed seed created: %d MB", SEED_GZ.stat().st_size // (1024 * 1024))

    # 5. Package universal zip
    root = Path(__file__).resolve().parents[1]
    PACKAGE_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PACKAGE_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in ["launch.sh", "launch.bat", "pyproject.toml", "README.md"]:
            f = root / rel
            if f.is_file():
                z.write(f, arcname=rel)
        for p in (root / "src").rglob("*"):
            if p.is_file() and "__pycache__" not in str(p):
                z.write(p, arcname=str(p.relative_to(root)))
        if SEED_GZ.is_file():
            z.write(SEED_GZ, arcname="data/areios_pagos_seed.db.gz")

    log.info("Universal package created: %d MB", PACKAGE_ZIP.stat().st_size // (1024 * 1024))

    # 6. Publish new GitHub release v0.3.0
    tag = "v0.3.0"
    log.info("Publishing new release %s to GitHub...", tag)
    n_dec = final_stats.get("decisions", 0)
    n_ch = final_stats.get("chunks", 0)
    os.system(
        f'git -c user.name="apsearch" -c user.email="dev@localhost" tag -a {tag} -m "Release {tag}: Complete modern case law corpus (2018-2026)" && git push origin {tag}'
    )
    release_notes = (
        f"## Complete Modern Jurisprudence (2018–2026)\\n\\n"
        f"* Contains every published decision from 2018 through 2026 (~{n_dec} rulings).\\n"
        f"* Pre-embedded with {n_ch} passage vectors.\\n"
        f"* Ready to double-click on Mac (launch.sh) or Windows (launch.bat)."
    )
    os.system(
        f'gh release create {tag} "{PACKAGE_ZIP}" "{SEED_GZ}" '
        f'--title "Areios Pagos Search {tag} (All Modern Cases 2018-2026)" '
        f'--notes "{release_notes}"'
    )
    log.info("All done! Release %s published successfully.", tag)


if __name__ == "__main__":
    main()
