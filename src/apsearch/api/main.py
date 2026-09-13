"""REST API & Local Web UI server."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Literal

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from apsearch.config import save_user_config, settings
from apsearch.logging import setup_logging

setup_logging()

app = FastAPI(
    title="Areios Pagos case-law search",
    version="0.1.0",
    description="Hybrid keyword + semantic search over Greek Supreme Court decisions.",
)


def _index_html() -> str:
    # Look for templates/index.html next to this file
    tpl = Path(__file__).resolve().parent / "templates" / "index.html"
    if tpl.is_file():
        return tpl.read_text(encoding="utf-8")
    try:
        return resources.files("apsearch.api").joinpath("templates/index.html").read_text(encoding="utf-8")
    except Exception:
        return "<h1>Areios Pagos Search</h1><p>Template not found.</p>"


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


class SetKeyRequest(BaseModel):
    key: str


class BackfillRequest(BaseModel):
    year_from: int
    year_to: int


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui() -> HTMLResponse:
    return HTMLResponse(content=_index_html())


@app.get("/api/status")
async def app_status() -> dict:
    from apsearch import repo

    s = repo.stats()
    return {
        "has_api_key": bool(settings.gemini_api_key),
        "db_backend": settings.db_backend,
        "model": settings.gemini_embed_model if settings.embed_backend == "gemini" else settings.embed_model,
        "stats": s,
    }


@app.post("/api/config/key")
async def configure_api_key(req: SetKeyRequest) -> dict:
    key = req.key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="API key cannot be empty")

    # Probe Gemini API to verify key validity
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:countTokens?key={key}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json={"contents": [{"parts": [{"text": "test"}]}]})
            if resp.status_code != 200:
                err_msg = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
                return {"success": False, "error": f"Invalid key: {err_msg}"}
    except Exception as exc:
        return {"success": False, "error": f"Connection error: {exc}"}

    # Save to local user config file and update runtime settings
    save_user_config({"gemini_api_key": key})
    settings.gemini_api_key = key
    from apsearch.index.embed import reset_backend

    reset_backend()
    return {"success": True, "message": "API key verified and saved successfully"}


@app.post("/api/sync/poll")
async def trigger_incremental_poll(bg: BackgroundTasks) -> dict:
    def _run():
        from apsearch.crawler.pipeline import run_incremental
        from apsearch.index.build import run_index

        run_incremental(lookback_years=1)
        run_index()

    bg.add_task(_run)
    return {"status": "started", "message": "Ο έλεγχος για νέες αποφάσεις ξεκίνησε στο παρασκήνιο"}


@app.post("/api/sync/backfill")
async def trigger_backfill(req: BackfillRequest, bg: BackgroundTasks) -> dict:
    if req.year_from > req.year_to:
        raise HTTPException(status_code=400, detail="year_from must be <= year_to")

    def _run():
        from apsearch.crawler.pipeline import run_backfill
        from apsearch.index.build import run_index

        run_backfill(year_from=req.year_from, year_to=req.year_to)
        run_index()

    bg.add_task(_run)
    return {
        "status": "started",
        "message": f"Η ανάκτηση για τα έτη {req.year_from}-{req.year_to} ξεκίνησε στο παρασκήνιο",
    }


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
async def readyz() -> dict:
    if settings.db_backend == "sqlite":
        from apsearch.db.sqlite import connect

        try:
            with connect() as conn:
                conn.execute("SELECT 1")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc
        return {"status": "ready"}

    from apsearch.db import query

    try:
        query("SELECT 1")
    except Exception as exc:
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
        q,
        mode=mode,
        limit=limit,
        filters=Filters(
            year_from=year_from,
            year_to=year_to,
            category=category,
            chamber=chamber,
            themes=list(theme or []),
        ),
    )
    return SearchResponse(
        query=q,
        mode=mode,
        count=len(results),
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
