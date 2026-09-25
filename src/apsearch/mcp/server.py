"""MCP server exposing the case-law index to agents.

Transports
----------
``stdio``  for local clients (Claude Desktop, editors).
``http``   streamable HTTP, so the same process can run on Cloud Run.

Tool design notes
-----------------
Agents are bad at guessing controlled vocabularies, so ``list_themes`` exposes
the court's own subject headings and ``search_decisions`` accepts them as a
filter. Results are returned without full decision text -- a single decision
runs to ~37k characters and would blow the context window -- and the agent is
expected to call ``get_decision`` for the one or two it actually needs.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from apsearch.logging import get_logger

log = get_logger(__name__)

INSTRUCTIONS = """\
Search engine over the published case law of the Άρειος Πάγος (Supreme Court of
Greece), covering 1994 to the present.

Queries should normally be written in Greek: the corpus is Greek and the
lexical index is Greek-aware. Semantic search will tolerate other languages but
recall will be lower.

Suggested workflow:
  1. `search_decisions` with a natural-language description of the legal issue.
  2. Narrow with `year_from` / `year_to` / `chamber` / `themes` if needed.
  3. `get_decision` for the full text of the few decisions that matter.

Modes: `hybrid` (default; lexical + semantic) is almost always right.
Use `keyword` for citations and exact terms of art, `semantic` for
paraphrase or when you do not know the Greek terminology.

Citations look like `144/2015` (decision number / year).
"""


def _server_class():
    """Support both MCP SDK generations.

    SDK 2.x renamed FastMCP to MCPServer and changed how host/port are passed.
    Pinning would be simpler but would also force every consumer onto one SDK
    version, so we adapt instead.
    """
    try:
        from mcp.server.mcpserver import MCPServer  # SDK >= 2.0

        return MCPServer, 2
    except ImportError:
        from mcp.server.fastmcp import FastMCP  # SDK 1.x

        return FastMCP, 1


def build_server():
    cls, _ = _server_class()
    mcp = cls("apsearch", instructions=INSTRUCTIONS)

    @mcp.tool()
    def search_decisions(
        query: Annotated[str, Field(description="Legal issue or terms, preferably Greek")],
        mode: Annotated[
            Literal["hybrid", "keyword", "semantic"],
            Field(description="hybrid=both, keyword=exact terms, semantic=meaning"),
        ] = "hybrid",
        limit: Annotated[int, Field(ge=1, le=50, description="Max results")] = 10,
        year_from: Annotated[int | None, Field(description="Earliest year")] = None,
        year_to: Annotated[int | None, Field(description="Latest year")] = None,
        category: Annotated[
            str | None, Field(description="ΠΟΛΙΤΙΚΕΣ (civil) or ΠΟΙΝΙΚΕΣ (criminal)")
        ] = None,
        chamber: Annotated[
            str | None, Field(description="e.g. Α1, Β2, Γ, ΣΤ, ΟΛΟΜΕΛΕΙΑ")
        ] = None,
        themes: Annotated[
            list[str] | None,
            Field(description="Subject headings from list_themes; matched as substrings"),
        ] = None,
    ) -> dict:
        """Search Areios Pagos decisions. Returns metadata + matching passages,
        not full texts -- follow up with get_decision."""
        from apsearch.search.hybrid import Filters, search

        try:
            results = search(
                query,
                mode=mode,
                limit=limit,
                filters=Filters(
                    year_from=year_from, year_to=year_to, category=category,
                    chamber=chamber, themes=list(themes or []),
                ),
            )
            return {
                "query": query,
                "mode": mode,
                "count": len(results),
                "results": [r.to_dict() for r in results],
            }
        except Exception as exc:
            log.exception("search_decisions failed for %r: %s", query, exc)
            return {
                "error": f"search failed: {exc}",
                "query": query,
                "mode": mode,
                "count": 0,
                "results": [],
            }

    @mcp.tool()
    def get_decision(
        citation: Annotated[
            str | None, Field(description="Citation like '144/2015'")
        ] = None,
        cd: Annotated[str | None, Field(description="Opaque id from search results")] = None,
        include_body: Annotated[
            bool, Field(description="Include full text (can be very long)")
        ] = True,
        max_chars: Annotated[
            int, Field(ge=500, le=200000, description="Truncate body to this length")
        ] = 60000,
    ) -> dict:
        """Fetch one decision by citation (e.g. '144/2015') or by opaque id."""
        from apsearch.search.hybrid import get_decision as _get

        number = year = None
        if citation and "/" in citation:
            n, _, y = citation.partition("/")
            try:
                number, year = int(n.strip()), int(y.strip())
            except ValueError:
                return {"error": f"could not parse citation {citation!r}; expected 'N/YYYY'"}
        if not cd and number is None:
            return {"error": "provide either `citation` ('144/2015') or `cd`"}

        rows = _get(cd=cd, number=number, year=year, include_body=include_body)
        if not rows:
            return {"error": "not found", "citation": citation, "cd": cd}
        for r in rows:
            r["first_seen"] = str(r.get("first_seen"))
            r["last_fetched"] = str(r.get("last_fetched"))
            body = r.get("body")
            if body and len(body) > max_chars:
                r["body"] = body[:max_chars]
                r["body_truncated"] = True
                r["body_total_chars"] = len(body)
        return {"count": len(rows), "decisions": rows}

    @mcp.tool()
    def list_themes(
        contains: Annotated[
            str | None, Field(description="Filter headings containing this substring")
        ] = None,
        limit: Annotated[int, Field(ge=1, le=300)] = 50,
    ) -> dict:
        """List the court's own subject headings (λήμματα), usable as `themes`
        filters in search_decisions."""
        from apsearch.search.hybrid import list_themes as _list

        rows = _list(contains, limit)
        return {"count": len(rows), "themes": rows}

    @mcp.tool()
    def corpus_stats() -> dict:
        """Size and coverage of the local index (year range, counts, freshness)."""
        from apsearch import repo

        return repo.stats()

    return mcp


def _relaxed_transport_security():
    """Return transport-security settings that accept any Host header.

    The SDK auto-enables DNS-rebinding protection for localhost-bound servers
    and restricts the Host header to localhost, which rejects the forwarded
    Host header when the server sits behind a reverse proxy such as
    ``tailscale serve`` (HTTP 421 "Invalid Host header"). The HTTP/SSE
    transports are only ever reached via loopback or a trusted private proxy,
    so that browser-focused defence is not applicable; disable it.
    """
    try:
        from mcp.server.transport_security import TransportSecuritySettings
    except ImportError:  # pragma: no cover - SDK without the middleware
        return None
    return TransportSecuritySettings(enable_dns_rebinding_protection=False)


def run(transport: str = "stdio", host: str = "0.0.0.0", port: int = 8080) -> None:
    mcp = build_server()
    _, generation = _server_class()

    if transport == "stdio":
        mcp.run(transport="stdio")
        return
    if transport in ("http", "streamable-http"):
        security = _relaxed_transport_security()
        if generation >= 2:
            mcp.run(transport="streamable-http", host=host, port=port, transport_security=security)
        else:
            mcp.settings.host = host
            mcp.settings.port = port
            mcp.run(transport="streamable-http")
        return
    if transport == "sse":
        security = _relaxed_transport_security()
        if generation >= 2:
            mcp.run(transport="sse", host=host, port=port, transport_security=security)
        else:
            mcp.settings.host = host
            mcp.settings.port = port
            mcp.run(transport="sse")
        return
    raise ValueError(f"unknown transport {transport!r} (expected stdio | http | sse)")


if __name__ == "__main__":
    run()
