"""Fetching and normalising a single decision."""

from __future__ import annotations

import hashlib

from apsearch.crawler.client import PoliteClient
from apsearch.crawler.parse import Decision, DecisionRef, parse_ai_json, parse_decision
from apsearch.logging import get_logger

log = get_logger(__name__)

DISPLAY_PATH = "/nomologia/apofaseis_DISPLAY.asp"
AI_JSON_PATH = "/nomologia/apofasi_ai_v2.asp"

#: Below this, the HTML body extraction is assumed to have failed and we fall
#: back to the site's own clean-text JSON endpoint.
MIN_BODY_CHARS = 400


def decision_url(cd: str, number: int | None = None, year: int | None = None) -> str:
    url = f"https://www.areiospagos.gr{DISPLAY_PATH}?cd={cd}"
    if number and year:
        url += f"&apof={number}_{year}"
    return url


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_decision(
    client: PoliteClient,
    cd: str,
    number: int | None = None,
    year: int | None = None,
    force_refresh: bool = False,
) -> Decision:
    """Fetch one decision.

    Primary source is the HTML display page, because it is the only place that
    carries the editorial "Θέμα" (subject headings) and "Περίληψη" (headnote).
    The JSON endpoint is used only as a body fallback, so the common case costs
    exactly one request.
    """
    params = {"cd": cd}
    if number and year:
        params["apof"] = f"{number}_{year}"

    markup = client.get(DISPLAY_PATH, params=params, force_refresh=force_refresh)
    dec = parse_decision(markup, cd)

    if len(dec.body) < MIN_BODY_CHARS:
        log.warning("short body for %s (%d chars); trying JSON endpoint", cd, len(dec.body))
        payload = client.get_json(AI_JSON_PATH, params={"cd": cd}, force_refresh=force_refresh)
        if text := parse_ai_json(payload):
            dec.body = text

    if number and not dec.number:
        dec.number = number
    if year and not dec.year:
        dec.year = year
    return dec


def merge_ref(dec: Decision, ref: DecisionRef | None) -> Decision:
    """Fill gaps in a parsed decision from its listing row."""
    if ref is None:
        return dec
    dec.number = dec.number or ref.number
    dec.year = dec.year or ref.year
    dec.category = dec.category or ref.category
    dec.chamber = dec.chamber or ref.chamber
    if not dec.summary and ref.snippet:
        # Listing snippets are truncated at ~500 chars with an ellipsis; only
        # useful when the display page had no headnote at all.
        dec.summary = ref.snippet
    return dec
