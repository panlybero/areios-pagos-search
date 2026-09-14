"""Command-line interface."""

from __future__ import annotations

import json as jsonlib

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from apsearch.config import settings
from apsearch.logging import setup_logging

app = typer.Typer(add_completion=False, help="Areios Pagos case-law search")
db_app = typer.Typer(help="Database schema management")
crawl_app = typer.Typer(help="Fetch decisions from areiospagos.gr")
index_app = typer.Typer(help="Chunk and embed decisions")
app.add_typer(db_app, name="db")
app.add_typer(crawl_app, name="crawl")
app.add_typer(index_app, name="index")

console = Console()


@app.callback()
def _root() -> None:
    setup_logging()


# ---------------------------------------------------------------------- db
@db_app.command("migrate")
def db_migrate(
    vector_index: bool = typer.Option(False, help="Also build the HNSW index now"),
) -> None:
    """Create or update the schema."""
    from apsearch.db import migrate

    migrate(with_vector_index=vector_index)
    console.print("[green]schema up to date[/green]")


@db_app.command("stats")
def db_stats() -> None:
    """Show corpus and index statistics."""
    from apsearch import repo

    s = repo.stats()
    t = Table(show_header=False, box=None)
    for k, v in s.items():
        t.add_row(f"[cyan]{k}[/cyan]", f"{v:,}" if isinstance(v, int) else str(v))
    console.print(Panel(t, title="apsearch", expand=False))


@db_app.command("cache-stats")
def db_cache_stats() -> None:
    """Show query embedding cache statistics."""
    from apsearch.db import pool
    from apsearch.search.cache import cache_stats

    with pool().connection() as conn, conn.cursor() as cur:
        s = cache_stats(cur)
    t = Table(show_header=False, box=None)
    for k, v in s.items():
        t.add_row(f"[cyan]{k}[/cyan]", f"{v:,}" if isinstance(v, int) else str(v))
    console.print(Panel(t, title="query cache", expand=False))


@db_app.command("prune-cache")
def db_prune_cache(
    max_entries: int = typer.Option(None, help="Max entries to retain (LRU)"),
    ttl_days: int = typer.Option(None, help="Purge queries older than N days"),
) -> None:
    """Evict expired and LRU queries from the cache to reclaim disk space."""
    from apsearch.db import pool
    from apsearch.search.cache import prune_query_cache

    with pool().connection() as conn, conn.cursor() as cur:
        deleted = prune_query_cache(cur, max_entries=max_entries, ttl_days=ttl_days)
    console.print(f"[green]pruned[/green] {deleted:,} cached queries")


# ------------------------------------------------------------------- crawl
@crawl_app.command("backfill")
def crawl_backfill(
    year_from: int = typer.Option(1994, help="First year to crawl"),
    year_to: int = typer.Option(None, help="Last year (default: current)"),
    fetch_limit: int = typer.Option(None, help="Stop after N decision fetches"),
    all_years: bool = typer.Option(False, "--all", help="Re-crawl completed years"),
) -> None:
    """Historical crawl. Resumable -- safe to interrupt and re-run."""
    from apsearch.crawler.pipeline import run_backfill

    s = run_backfill(year_from, year_to, fetch_limit, skip_done=not all_years)
    _print_run(s)


@crawl_app.command("poll")
def crawl_poll(
    lookback_years: int = typer.Option(1, help="How many prior years to re-check"),
    fetch_limit: int = typer.Option(None, help="Stop after N decision fetches"),
) -> None:
    """Daily incremental poll for newly published decisions."""
    from apsearch.crawler.pipeline import run_incremental

    s = run_incremental(lookback_years, fetch_limit)
    _print_run(s)


@crawl_app.command("fetch")
def crawl_fetch(
    limit: int = typer.Option(None, help="Stop after N decisions"),
) -> None:
    """Fetch decisions already discovered but not yet downloaded."""
    from apsearch.crawler.client import PoliteClient
    from apsearch.crawler.pipeline import RunStats, drain_queue

    s = RunStats()
    with PoliteClient() as client:
        try:
            drain_queue(client, s, limit=limit)
        finally:
            s.requests, s.cache_hits = client.n_requests, client.n_cache_hits
    _print_run(s)


@crawl_app.command("themes")
def crawl_themes(
    max_age_days: int = typer.Option(30, help="Re-crawl headings older than this"),
    limit: int = typer.Option(None, help="Max headings this run"),
) -> None:
    """Refresh the site's thematic index (controlled vocabulary + links)."""
    from apsearch.crawler.pipeline import run_themes

    s = run_themes(max_age_days, limit)
    _print_run(s)


def _print_run(s) -> None:
    t = Table(show_header=False, box=None)
    t.add_row("discovered", f"{s.discovered:,}")
    t.add_row("fetched", f"{s.fetched:,}")
    t.add_row("new", f"[green]{s.new:,}[/green]")
    t.add_row("changed", f"{s.changed:,}")
    t.add_row("unchanged", f"{s.unchanged:,}")
    t.add_row("errors", f"[red]{s.errors:,}[/red]" if s.errors else "0")
    t.add_row("http requests", f"{s.requests:,}")
    t.add_row("cache hits", f"{s.cache_hits:,}")
    console.print(Panel(t, title="crawl run", expand=False))


# ------------------------------------------------------------------- index
@index_app.command("build")
def index_build(
    limit: int = typer.Option(None, help="Index at most N decisions"),
    batch_size: int = typer.Option(16, help="Decisions per embedding batch"),
    skip_vector_index: bool = typer.Option(False, help="Don't build HNSW afterwards"),
) -> None:
    """Chunk + embed decisions that are new or changed."""
    from apsearch.index.build import run_index

    s = run_index(limit, batch_size, build_index_after=not skip_vector_index)
    console.print(
        f"[green]indexed[/green] {s.decisions:,} decisions / {s.chunks:,} chunks "
        f"in {s.seconds:.0f}s ({s.rate:.1f} chunks/s)"
    )


@index_app.command("reset")
def index_reset(
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
) -> None:
    """Drop all chunks + embeddings so the corpus can be re-embedded."""
    from apsearch.index.build import reset_index

    if not yes and not typer.confirm("Delete all chunks and embeddings?"):
        raise typer.Abort()
    reset_index()
    console.print("[yellow]index reset[/yellow]")


@index_app.command("vector-index")
def index_vector() -> None:
    """Build the HNSW index (slow; do it once after a bulk load)."""
    from apsearch.db import create_vector_index

    create_vector_index()
    console.print("[green]HNSW index built[/green]")


# ------------------------------------------------------------------ search
@app.command("search")
def cli_search(
    query: str = typer.Argument(..., help="Query text (Greek)"),
    mode: str = typer.Option("hybrid", help="hybrid | keyword | semantic"),
    limit: int = typer.Option(10, "-n", help="Number of results"),
    year_from: int = typer.Option(None),
    year_to: int = typer.Option(None),
    category: str = typer.Option(None, help="e.g. ΠΟΛΙΤΙΚΕΣ / ΠΟΙΝΙΚΕΣ"),
    chamber: str = typer.Option(None, help="e.g. Α1, Β2, ΟΛΟΜΕΛΕΙΑ"),
    theme: list[str] = typer.Option(None, help="Filter by subject heading"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Search the corpus."""
    from apsearch.search.hybrid import Filters, search

    filters = Filters(
        year_from=year_from, year_to=year_to, category=category,
        chamber=chamber, themes=list(theme or []),
    )
    results = search(query, mode=mode, limit=limit, filters=filters)

    if json_out:
        typer.echo(jsonlib.dumps([r.to_dict() for r in results],
                                 ensure_ascii=False, indent=2))
        return

    if not results:
        console.print("[yellow]no results[/yellow]")
        return

    for i, r in enumerate(results, 1):
        head = (
            f"[bold cyan]{i}. {r.citation}[/bold cyan]  "
            f"[dim]{r.category or ''} {r.chamber or ''}[/dim]  "
            f"[dim]score={r.score:.4f} via {','.join(sorted(r.ranks))}[/dim]"
        )
        console.print(head)
        if r.subject:
            console.print(f"   [magenta]{r.subject}[/magenta]")
        for p in r.passages[:2]:
            txt = (p.highlight or p.content or "").replace("\n", " ")[:340]
            txt = txt.replace("**", "[bold yellow]", 1)
            console.print(f"   [dim]{p.part}:[/dim] {txt}")
        console.print(f"   [blue underline]{r.url}[/blue underline]\n")


@app.command("get")
def cli_get(
    citation: str = typer.Argument(None, help="e.g. 144/2015"),
    cd: str = typer.Option(None, help="Opaque site id"),
    body: bool = typer.Option(False, help="Include full text"),
) -> None:
    """Fetch a decision by citation or id."""
    from apsearch.search.hybrid import get_decision

    number = year = None
    if citation and "/" in citation:
        n, _, y = citation.partition("/")
        number, year = int(n), int(y)
    rows = get_decision(cd=cd, number=number, year=year, include_body=body)
    for r in rows:
        r.pop("first_seen", None), r.pop("last_fetched", None)
    typer.echo(jsonlib.dumps(rows, ensure_ascii=False, indent=2, default=str))


@app.command("themes")
def cli_themes(
    prefix: str = typer.Argument(None, help="Filter by substring"),
    limit: int = typer.Option(40, "-n"),
) -> None:
    """List subject headings from the site's controlled vocabulary."""
    from apsearch.search.hybrid import list_themes

    t = Table("code", "heading", "decisions")
    for row in list_themes(prefix, limit):
        t.add_row(str(row["code"]), row["label"], f"{row['n_decisions']:,}")
    console.print(t)


# ------------------------------------------------------------------ serving
@app.command("launch")
def cli_launch(
    host: str = typer.Option("127.0.0.1", help="Host to bind Web UI"),
    port: int = typer.Option(8765, help="Port for Web UI"),
    mcp_port: int = typer.Option(8766, help="Port for MCP server"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't auto-open browser"),
    no_sync: bool = typer.Option(False, "--no-sync", help="Don't auto-check for new rulings on launch"),
) -> None:
    """Launch the full desktop application (Web UI + MCP server + auto-sync)."""
    import threading
    import time
    import webbrowser

    import uvicorn

    from apsearch.db import migrate
    from apsearch.logging import get_logger

    log = get_logger("apsearch.app")

    # 1. Ensure DB schema exists
    migrate()

    # 2. Start MCP server in background thread on mcp_port
    def _run_mcp():
        from apsearch.mcp.server import run as run_mcp

        try:
            run_mcp(transport="http", host="0.0.0.0", port=mcp_port)
        except Exception as exc:
            log.warning("MCP server background start failed: %s", exc)

    mcp_thread = threading.Thread(target=_run_mcp, daemon=True)
    mcp_thread.start()

    # 3. Optional auto-sync on launch in background thread
    if not no_sync and settings.gemini_api_key:

        def _bg_startup_sync():
            time.sleep(3)
            try:
                from apsearch.crawler.pipeline import run_incremental
                from apsearch.index.build import run_index

                log.info("Checking for new rulings published since last launch...")
                run_incremental(lookback_years=1)
                run_index()
            except Exception as exc:
                log.debug("Startup sync check: %s", exc)

        sync_thread = threading.Thread(target=_bg_startup_sync, daemon=True)
        sync_thread.start()

    # 4. Open browser
    if not no_browser:

        def _open_tab():
            time.sleep(1.2)
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=_open_tab, daemon=True).start()

    console.print(f"[bold green]Areios Pagos Search running at:[/bold green] http://{host}:{port}")
    console.print(f"[bold cyan]MCP server active at:[/bold cyan] http://localhost:{mcp_port}/mcp")

    # Pass the ASGI app object directly rather than the "module:attr" string
    # form. uvicorn's string form does its own importlib.import_module() at
    # runtime, which PyInstaller's static analysis cannot see (there is no
    # real `import` statement to trace) -- inside the frozen --onefile build
    # that module is therefore never bundled, and this fails with
    # "Could not import module 'apsearch.api.main'" the moment uvicorn tries
    # to load it. Importing it here for real fixes both problems: PyInstaller
    # bundles it correctly, and uvicorn skips the redundant re-import.
    from apsearch.api.main import app as asgi_app

    uvicorn.run(
        asgi_app,
        host=host,
        port=port,
        log_config=None,
    )


@app.command("serve")
def cli_serve(
    host: str = typer.Option(None), port: int = typer.Option(None)
) -> None:
    """Run the REST API."""
    import uvicorn

    from apsearch.api.main import app as asgi_app

    uvicorn.run(
        asgi_app,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
    )


@app.command("mcp")
def cli_mcp(
    transport: str = typer.Option("stdio", help="stdio | http | sse"),
    host: str = typer.Option(None), port: int = typer.Option(None),
) -> None:
    """Run the MCP server (for Claude, agents, etc.)."""
    from apsearch.mcp.server import run

    run(transport=transport, host=host or settings.host, port=port or settings.port)


if __name__ == "__main__":
    app()
