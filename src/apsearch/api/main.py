"""REST API.

Deliberately thin -- it wraps the same functions the CLI and MCP server use.
Health endpoints are split the way Cloud Run wants them: ``/healthz`` is a
liveness probe that must never touch the database, ``/readyz`` is a readiness
probe that must.
"""

from __future__ import annotations

from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from apsearch.config import settings
from apsearch.logging import setup_logging

setup_logging()

app = FastAPI(
    title="Areios Pagos case-law search",
    version="0.1.0",
    description="Hybrid keyword + semantic search over Greek Supreme Court decisions.",
)


async def require_key(x_api_key: str | None = Header(default=None)) -> None:
    """No-op unless APSEARCH_API_KEY is configured."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


class Passage(BaseModel):
    part: str
    ordinal: int
    text: str


class SearchHit(BaseModel):
    cd: str
    citation: str
    number: int | None = None
    year: int | None = None
    category: str | None = None
    chamber: str | None = None
    subject: str | None = None
    summary: str | None = None
    themes: list[str] = []
    url: str
    score: float
    matched_by: list[str] = []
    passages: list[Passage] = []


class SearchResponse(BaseModel):
    query: str
    mode: str
    count: int
    results: list[SearchHit]


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict:
    """Liveness: process is up. Must not depend on the database."""
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
async def readyz() -> dict:
    """Readiness: the database is reachable and the schema exists."""
    from apsearch.db import query

    try:
        query("SELECT 1")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc
    return {"status": "ready"}


@app.get("/search", response_model=SearchResponse, dependencies=[Depends(require_key)])
async def search_endpoint(
    q: str = Query(..., description="Query text, preferably Greek"),
    mode: Literal["hybrid", "keyword", "semantic"] = "hybrid",
    limit: int = Query(20, ge=1, le=100),
    year_from: int | None = None,
    year_to: int | None = None,
    category: str | None = None,
    chamber: str | None = None,
    theme: list[str] | None = Query(default=None),
) -> SearchResponse:
    from apsearch.search.hybrid import Filters, search

    results = search(
        q, mode=mode, limit=limit,
        filters=Filters(
            year_from=year_from, year_to=year_to, category=category,
            chamber=chamber, themes=list(theme or []),
        ),
    )
    return SearchResponse(
        query=q, mode=mode, count=len(results),
        results=[SearchHit(**r.to_dict()) for r in results],
    )


@app.get("/decisions/{cd}", dependencies=[Depends(require_key)])
async def get_by_cd(cd: str, include_body: bool = True) -> dict:
    from apsearch.search.hybrid import get_decision

    rows = get_decision(cd=cd, include_body=include_body)
    if not rows:
        raise HTTPException(status_code=404, detail="decision not found")
    return rows[0]


@app.get("/citation/{number}/{year}", dependencies=[Depends(require_key)])
async def get_by_citation(number: int, year: int, include_body: bool = True) -> dict:
    from apsearch.search.hybrid import get_decision

    rows = get_decision(number=number, year=year, include_body=include_body)
    if not rows:
        raise HTTPException(status_code=404, detail="decision not found")
    return {"count": len(rows), "decisions": rows}


@app.get("/themes", dependencies=[Depends(require_key)])
async def themes_endpoint(
    contains: str | None = None, limit: int = Query(100, ge=1, le=1000)
) -> dict:
    from apsearch.search.hybrid import list_themes

    rows = list_themes(contains, limit)
    return {"count": len(rows), "themes": rows}


@app.get("/stats", dependencies=[Depends(require_key)])
async def stats_endpoint() -> dict:
    from apsearch import repo

    return repo.stats()
