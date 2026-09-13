"""Discovery: enumerating every decision the site will show us.

The search endpoint silently truncates at 3000 rows ("Η αναζήτηση επέστρεψε
περισσότερα από 3000 αποτελέσματα"), and offers no pagination. So discovery is
an *adaptive partition search*: start coarse, and split a partition only when
the response comes back truncated. This keeps request volume near the minimum
needed for completeness.

Split ladder
------------
1. (year, all categories, all chambers)          -- 1 request per year
2. (year, one category, all chambers)            -- 5 requests
3. (year, one category, one chamber)             -- up to 15 requests
4. (year, category, chamber, number <= / >= N)   -- binary split on number

Level 4 exists because the form exposes only a single comparison operator per
query, so a closed range is expressed as two complementary open ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from apsearch.crawler.client import PoliteClient
from apsearch.crawler.parse import DecisionRef, parse_listing
from apsearch.logging import get_logger

log = get_logger(__name__)

SEARCH_PATH = "/nomologia/apofaseis_result.asp"
INDEX_PATH = "/nomologia/apofaseis.asp"

#: Site sentinels for "no filter".
CATEGORY_ALL = 6
CHAMBER_ALL = 1

#: X_TMHMA values -> label
CATEGORIES: dict[int, str] = {
    1: "ΠΟΛΙΤΙΚΕΣ",
    2: "ΠΟΙΝΙΚΕΣ",
    3: "Νόμου 3068/2002",
    4: "Νόμου 4239/2014",
    5: "Πράξεις Νόμου 4842/2021",
}

#: X_SUB_TMHMA values -> label (1 == "all")
CHAMBERS: dict[int, str] = {
    2: "Α", 13: "Α1", 12: "Α2", 14: "Α3",
    3: "Β", 4: "Β1", 5: "Β2",
    6: "Γ", 7: "Δ", 8: "Ε", 9: "ΣΤ", 10: "Ζ",
    11: "ΟΛΟΜΕΛΕΙΑ",
    15: "Α Ποιν. Διακ.", 16: "Β Ποιν. Διακ.",
}

OP_EQ, OP_GTE, OP_LTE = 1, 2, 3

#: Earliest year with published decisions (verified empirically: 1994).
FIRST_YEAR = 1994


@dataclass(slots=True)
class Partition:
    year: int
    category_id: int = CATEGORY_ALL
    chamber_id: int = CHAMBER_ALL
    number: int | None = None
    number_op: int = OP_EQ

    def describe(self) -> str:
        cat = CATEGORIES.get(self.category_id, "ΟΛΕΣ")
        cham = CHAMBERS.get(self.chamber_id, "ΟΛΕΣ")
        s = f"{self.year}/{cat}/{cham}"
        if self.number is not None:
            op = {OP_EQ: "=", OP_GTE: ">=", OP_LTE: "<="}[self.number_op]
            s += f"/num{op}{self.number}"
        return s

    def form(self) -> dict[str, str]:
        return {
            "X_TMHMA": str(self.category_id),
            "X_SUB_TMHMA": str(self.chamber_id),
            "X_TELESTIS_number": str(self.number_op),
            "x_number": "" if self.number is None else str(self.number),
            "X_TELESTIS_ETOS": str(OP_EQ),
            "x_ETOS": str(self.year),
        }


def current_year() -> int:
    return date.today().year


def search_partition(
    client: PoliteClient, part: Partition, force_refresh: bool = False
) -> tuple[list[DecisionRef], bool]:
    """Run one search query. Returns (refs, truncated)."""
    markup = client.post(
        SEARCH_PATH,
        data=part.form(),
        params={"S": "1"},
        force_refresh=force_refresh,
    )
    refs, truncated = parse_listing(markup)
    log.info("search %s -> %d refs%s", part.describe(), len(refs),
             " (TRUNCATED)" if truncated else "")
    return refs, truncated


def discover_year(
    client: PoliteClient,
    year: int,
    force_refresh: bool = False,
    max_depth: int = 4,
) -> tuple[dict[str, DecisionRef], list[tuple[Partition, bool]]]:
    """Enumerate every decision for a year, splitting partitions as needed.

    Returns a ``{cd: DecisionRef}`` map (deduplicated across partitions) and the
    list of ``(partition, truncated)`` results for bookkeeping.
    """
    found: dict[str, DecisionRef] = {}
    audit: list[tuple[Partition, bool]] = []

    def absorb(refs: list[DecisionRef]) -> None:
        for r in refs:
            # Prefer the richer record (one carrying a summary snippet).
            prev = found.get(r.cd)
            if prev is None or (not prev.snippet and r.snippet):
                found[r.cd] = r

    # --- level 1: whole year -------------------------------------------------
    root = Partition(year=year)
    refs, truncated = search_partition(client, root, force_refresh)
    absorb(refs)
    audit.append((root, truncated))
    if not truncated or max_depth < 2:
        return found, audit

    # --- level 2: by category ------------------------------------------------
    for cat_id in CATEGORIES:
        part = Partition(year=year, category_id=cat_id)
        refs, truncated = search_partition(client, part, force_refresh)
        absorb(refs)
        audit.append((part, truncated))
        if not truncated or max_depth < 3:
            continue

        # --- level 3: by chamber --------------------------------------------
        for cham_id in CHAMBERS:
            sub = Partition(year=year, category_id=cat_id, chamber_id=cham_id)
            refs, truncated = search_partition(client, sub, force_refresh)
            absorb(refs)
            audit.append((sub, truncated))
            if not truncated or max_depth < 4:
                continue

            # --- level 4: binary split on decision number --------------------
            absorb_number_split(client, sub, absorb, audit, force_refresh)

    return found, audit


def absorb_number_split(
    client: PoliteClient,
    base: Partition,
    absorb,
    audit: list[tuple[Partition, bool]],
    force_refresh: bool,
    lo: int = 1,
    hi: int = 20000,
    depth: int = 0,
) -> None:
    """Recursively halve the decision-number range until nothing truncates."""
    if depth > 6:
        log.warning("number split exhausted for %s; some rows may be missing",
                    base.describe())
        return
    mid = (lo + hi) // 2
    for op, bound in ((OP_LTE, mid), (OP_GTE, mid + 1)):
        part = Partition(
            year=base.year,
            category_id=base.category_id,
            chamber_id=base.chamber_id,
            number=bound,
            number_op=op,
        )
        refs, truncated = search_partition(client, part, force_refresh)
        absorb(refs)
        audit.append((part, truncated))
        if truncated:
            if op == OP_LTE:
                absorb_number_split(client, base, absorb, audit, force_refresh,
                                    lo, mid, depth + 1)
            else:
                absorb_number_split(client, base, absorb, audit, force_refresh,
                                    mid + 1, hi, depth + 1)


def discover_theme(
    client: PoliteClient, code: int, force_refresh: bool = False
) -> list[DecisionRef]:
    """All decisions filed under one subject heading of the thematic index."""
    markup = client.get(
        SEARCH_PATH, params={"s": "2", "code": str(code)}, force_refresh=force_refresh
    )
    refs, _ = parse_listing(markup)
    return refs


def fetch_theme_index(client: PoliteClient, force_refresh: bool = False):
    """The site's controlled vocabulary of legal subjects."""
    from apsearch.crawler.parse import parse_theme_index

    markup = client.get(INDEX_PATH, force_refresh=force_refresh)
    return parse_theme_index(markup)
