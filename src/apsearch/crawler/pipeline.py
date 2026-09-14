"""Crawl orchestration: backfill, daily incremental poll, thematic index."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from apsearch import repo
from apsearch.config import settings
from apsearch.crawler.client import CrawlBudgetExceeded, PoliteClient
from apsearch.crawler.discover import (
    FIRST_YEAR,
    current_year,
    discover_theme,
    discover_year,
    fetch_theme_index,
)
from apsearch.crawler.fetch import fetch_decision, merge_ref
from apsearch.crawler.parse import DecisionRef
from apsearch.logging import get_logger

log = get_logger(__name__)


@dataclass
class RunStats:
    discovered: int = 0
    fetched: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    errors: int = 0
    requests: int = 0
    cache_hits: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_discovered": self.discovered,
            "n_fetched": self.fetched,
            "n_changed": self.new + self.changed,
            "n_errors": self.errors,
            "notes": "; ".join(self.notes) or None,
        }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover(
    client: PoliteClient,
    years: list[int],
    stats: RunStats,
    force_refresh: bool = False,
) -> dict[str, DecisionRef]:
    """Enumerate decisions for the given years and queue the unseen ones."""
    all_refs: dict[str, DecisionRef] = {}
    for year in years:
        try:
            found, audit = discover_year(client, year, force_refresh=force_refresh)
        except CrawlBudgetExceeded:
            stats.notes.append(f"request budget hit during discovery of {year}")
            raise
        except Exception as exc:  # noqa: BLE001 - one bad year must not kill the run
            log.exception("discovery failed for %s", year)
            stats.errors += 1
            repo.record_partition(year, 6, 1, 0, False, status="error", error=str(exc))
            continue

        for part, truncated in audit:
            repo.record_partition(
                part.year, part.category_id, part.chamber_id,
                n_found=len(found), truncated=truncated,
                status="truncated" if truncated else "done",
            )
        all_refs.update(found)
        stats.discovered += len(found)
        repo.enqueue(found.values())
        log.info("year %s: %d decisions discovered", year, len(found))
    return all_refs


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def drain_queue(
    client: PoliteClient,
    stats: RunStats,
    limit: int | None = None,
    refs: dict[str, DecisionRef] | None = None,
    batch: int = 100,
) -> None:
    """Fetch queued decisions until the queue is empty or `limit` is reached."""
    refs = refs or {}
    processed = 0
    workers = max(1, settings.crawl_concurrency)

    while True:
        take = batch if limit is None else min(batch, limit - processed)
        if take <= 0:
            return
        rows = repo.take_queue(take)
        if not rows:
            return

        def _fetch_one(row):
            cd = row["cd"]
            try:
                dec = fetch_decision(client, cd, row["number"], row["year"])
                dec = merge_ref(dec, refs.get(cd))
                if not dec.is_usable:
                    return cd, None, ValueError(f"unusable body ({len(dec.body)} chars)")
                return cd, dec, None
            except CrawlBudgetExceeded:
                raise
            except Exception as err:  # noqa: BLE001
                return cd, None, err

        if workers <= 1:
            results = [_fetch_one(r) for r in rows]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_fetch_one, rows))

        for cd, dec, exc in results:
            if exc is not None:
                log.warning("fetch failed for %s: %s", cd, exc)
                if "500" in str(exc) or "404" in str(exc):
                    # Unrecoverable server error on the court's site: drop so it doesn't stall the queue
                    repo.dequeue(cd)
                else:
                    repo.mark_queue_error(cd, str(exc))
                stats.errors += 1
                processed += 1
                continue

            outcome = repo.upsert_decision(dec)
            repo.dequeue(cd)
            stats.fetched += 1
            setattr(stats, outcome, getattr(stats, outcome) + 1)
            processed += 1

        log.info(
            "fetched %d/%s (queue depth %d)",
            processed,
            limit if limit is not None else "all",
            repo.queue_depth(),
        )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_backfill(
    year_from: int = FIRST_YEAR,
    year_to: int | None = None,
    fetch_limit: int | None = None,
    skip_done: bool = True,
) -> RunStats:
    """Full historical crawl. Resumable: completed years are skipped."""
    year_to = year_to or current_year()
    stats = RunStats()
    run_id = repo.start_run("backfill")
    years = [y for y in range(year_from, year_to + 1)]
    if skip_done:
        skipped = [y for y in years if repo.year_is_done(y)]
        years = [y for y in years if y not in skipped]
        if skipped:
            log.info("skipping %d already-completed years", len(skipped))

    with PoliteClient() as client:
        try:
            refs = discover(client, years, stats)
            drain_queue(client, stats, limit=fetch_limit, refs=refs)
        except CrawlBudgetExceeded as exc:
            log.warning("stopping early: %s", exc)
        finally:
            stats.requests = client.n_requests
            stats.cache_hits = client.n_cache_hits
    repo.finish_run(run_id, **stats.as_dict())
    return stats


def run_incremental(lookback_years: int = 1, fetch_limit: int | None = None) -> RunStats:
    """Daily poll.

    New decisions land in the current year (and briefly in the previous one, as
    late publications trickle in), so only those years are re-enumerated.
    Listings are fetched with ``force_refresh`` because they are exactly the
    thing that changes; decision pages still come from the archive if present.
    """
    stats = RunStats()
    run_id = repo.start_run("incremental")
    this_year = current_year()
    years = list(range(this_year - lookback_years, this_year + 1))

    with PoliteClient() as client:
        try:
            refs = discover(client, years, stats, force_refresh=True)
            new_cds = set(refs) - repo.known_cds(list(refs))
            log.info("incremental: %d listed, %d new", len(refs), len(new_cds))
            stats.notes.append(f"{len(new_cds)} new decisions")
            drain_queue(client, stats, limit=fetch_limit, refs=refs)
        except CrawlBudgetExceeded as exc:
            log.warning("stopping early: %s", exc)
        finally:
            stats.requests = client.n_requests
            stats.cache_hits = client.n_cache_hits
    repo.finish_run(run_id, **stats.as_dict())
    return stats


def run_themes(max_age_days: int = 30, limit: int | None = None) -> RunStats:
    """Refresh the controlled vocabulary and its decision links.

    The thematic index is the site's own editorial classification -- far better
    than anything we could infer -- so it is worth the extra requests. It
    changes slowly, hence the 30-day default staleness window.
    """
    stats = RunStats()
    run_id = repo.start_run("themes")
    with PoliteClient() as client:
        try:
            themes = fetch_theme_index(client, force_refresh=True)
            repo.upsert_themes(themes)
            log.info("theme vocabulary: %d headings", len(themes))

            pending = repo.themes_needing_crawl(max_age_days)
            if limit:
                pending = pending[:limit]
            for i, t in enumerate(pending, 1):
                try:
                    refs = discover_theme(client, t["code"], force_refresh=True)
                except CrawlBudgetExceeded:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.warning("theme %s failed: %s", t["code"], exc)
                    stats.errors += 1
                    continue
                repo.enqueue(refs)  # a theme page can surface unseen decisions
                repo.link_theme(t["code"], [r.cd for r in refs])
                stats.discovered += len(refs)
                if i % 25 == 0:
                    log.info("themes %d/%d", i, len(pending))
        except CrawlBudgetExceeded as exc:
            log.warning("stopping early: %s", exc)
        finally:
            stats.requests = client.n_requests
            stats.cache_hits = client.n_cache_hits
    repo.finish_run(run_id, **stats.as_dict())
    return stats
