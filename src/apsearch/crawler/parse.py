"""HTML parsers for areiospagos.gr.

The site is hand-written 1990s-era ASP output: unbalanced tags, `<td>` blocks
emitted outside `<tr>`, attributes without quotes, and `&nbsp` without the
trailing semicolon.  A DOM parser "fixes" this markup in ways that vary between
pages, so the listing/decision parsers work on the raw markup with anchored
regexes, which is both more predictable and easier to assert on in tests.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_BREAK_RE = re.compile(r"(?is)<br\s*/?>|</p\s*>|</tr\s*>|</div\s*>")
_SCRIPT_RE = re.compile(r"(?is)<(script|style)\b.*?</\1\s*>")
_WS_RE = re.compile(r"[ \t\u00a0\u2000-\u200b]+")
_NL_RE = re.compile(r"\n\s*\n\s*\n+")


def html_to_text(fragment: str) -> str:
    """Convert a markup fragment to clean plain text."""
    if not fragment:
        return ""
    s = _SCRIPT_RE.sub(" ", fragment)
    s = _BREAK_RE.sub("\n", s)
    s = _TAG_RE.sub(" ", s)
    # The site emits bare `&nbsp` entities; unescape them before the rest.
    s = s.replace("&nbsp", " ")
    s = html.unescape(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WS_RE.sub(" ", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    s = _NL_RE.sub("\n\n", s)
    return s.strip()


#: Greek final sigma and diacritics are normalised away for *matching* keys
#: only -- never for stored text.
def fold_greek(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", s).lower().replace("ς", "σ")


# ---------------------------------------------------------------------------
# Search-result listings  (apofaseis_result.asp)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DecisionRef:
    cd: str
    number: int | None
    year: int | None
    category: str | None = None
    chamber: str | None = None
    snippet: str | None = None

    @property
    def citation(self) -> str:
        return f"{self.number}/{self.year}"

    @property
    def url(self) -> str:
        return (
            "https://www.areiospagos.gr/nomologia/apofaseis_DISPLAY.asp"
            f"?cd={self.cd}&apof={self.number}_{self.year}"
        )


_LISTING_RE = re.compile(
    r"""<a[^>]*href="apofaseis_DISPLAY\.asp\?cd=(?P<cd>[A-Za-z0-9]+)
        &(?:amp;)?apof=(?P<num>\d+)_(?P<year>\d+)
        (?:&(?:amp;)?info=(?P<info>[^"]*))?"[^>]*>.*?</a>
        \s*</td>\s*
        <td[^>]*>(?P<snippet>.*?)</td>""",
    re.VERBOSE | re.DOTALL | re.IGNORECASE,
)

#: Fallback when the trailing snippet cell is missing/malformed.
_LISTING_MIN_RE = re.compile(
    r"apofaseis_DISPLAY\.asp\?cd=(?P<cd>[A-Za-z0-9]+)&(?:amp;)?apof=(?P<num>\d+)_(?P<year>\d+)"
    r"(?:&(?:amp;)?info=(?P<info>[^\"&]*))?",
    re.IGNORECASE,
)

TRUNCATION_MARKER = "περισσότερα από"


def _split_info(info: str | None) -> tuple[str | None, str | None]:
    """`info` looks like "ΠΟΛΙΤΙΚΕΣ -  Γ" -> ("ΠΟΛΙΤΙΚΕΣ", "Γ")."""
    if not info:
        return None, None
    text = html.unescape(info).replace("+", " ")
    if "-" in text:
        cat, _, cham = text.partition("-")
        cat, cham = cat.strip(), cham.strip()
        return (cat or None), (cham or None)
    return text.strip() or None, None


def parse_listing(markup: str) -> tuple[list[DecisionRef], bool]:
    """Parse a search-result page.

    Returns ``(refs, truncated)`` where ``truncated`` is True when the site hit
    its hard 3000-row response cap and silently dropped the remainder.
    """
    refs: list[DecisionRef] = []
    seen: set[str] = set()

    for m in _LISTING_RE.finditer(markup):
        cd = m.group("cd")
        if cd in seen:
            continue
        seen.add(cd)
        cat, cham = _split_info(m.group("info"))
        snippet = html_to_text(m.group("snippet") or "")
        refs.append(
            DecisionRef(
                cd=cd,
                number=_int(m.group("num")),
                year=_int(m.group("year")),
                category=cat,
                chamber=cham,
                snippet=snippet or None,
            )
        )

    # Catch anchors the strict pattern missed (malformed trailing cell).
    for m in _LISTING_MIN_RE.finditer(markup):
        cd = m.group("cd")
        if cd in seen:
            continue
        seen.add(cd)
        cat, cham = _split_info(m.group("info"))
        refs.append(
            DecisionRef(
                cd=cd,
                number=_int(m.group("num")),
                year=_int(m.group("year")),
                category=cat,
                chamber=cham,
            )
        )

    truncated = TRUNCATION_MARKER in markup
    return refs, truncated


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Thematic index  (apofaseis.asp -> s=2&code=N)
# ---------------------------------------------------------------------------

_THEME_RE = re.compile(
    r"apofaseis_result\.asp\?s=2&(?:amp;)?code=(?P<code>\d+)\"[^>]*>(?P<label>[^<]+)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Theme:
    code: int
    label: str
    slug: str


def parse_theme_index(markup: str) -> list[Theme]:
    """Extract the controlled vocabulary of legal subjects from apofaseis.asp."""
    out: dict[int, Theme] = {}
    for m in _THEME_RE.finditer(markup):
        code = int(m.group("code"))
        slug = html.unescape(m.group("label")).strip()
        label = slug.replace("_", " ").strip()
        if label and code not in out:
            out[code] = Theme(code=code, label=label, slug=slug)
    return list(out.values())


# ---------------------------------------------------------------------------
# Decision page  (apofaseis_DISPLAY.asp)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Decision:
    cd: str
    number: int | None = None
    year: int | None = None
    category: str | None = None
    chamber: str | None = None
    subject: str | None = None
    summary: str | None = None
    body: str = ""
    subjects: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return bool(self.body and len(self.body) > 200)


_HEADER_RE = re.compile(
    r"Απόφαση\s*<b>\s*(?P<num>\d+)\s*/\s*(?P<year>\d+)\s*</b>"
    r"(?:[^()<]|<[^>]+>)*?\(\s*(?P<info>[^)]*?)\s*\)",
    re.IGNORECASE,
)

_SUBJECT_RE = re.compile(
    r"<b>\s*Θέμα\s*:?\s*</b>\s*(?:<br\s*/?>)?(?P<val>.*?)</i>",
    re.IGNORECASE | re.DOTALL,
)

_SUMMARY_RE = re.compile(
    r"<b>\s*Περίληψη\s*:?\s*</b>\s*(?:<br\s*/?>)?(?P<val>.*?)"
    r"(?=<p\s+align=['\"]?justify['\"]?\s+style=['\"]line-height:\s*160%)",
    re.IGNORECASE | re.DOTALL,
)

_BODY_RE = re.compile(
    r"<p\s+align=['\"]?justify['\"]?\s+style=['\"]line-height:\s*160%['\"]\s*>"
    r"(?P<val>.*?)"
    r"(?=<p\s+align=['\"]?center['\"]?|</font>|</body>)",
    re.IGNORECASE | re.DOTALL,
)

#: Everything before this marker is chrome (nav, AI-summary modals, scripts).
_CONTENT_ANCHOR = re.compile(r"<font\s+face=\"Arial\"\s+size=\"3\">", re.IGNORECASE)


def parse_decision(markup: str, cd: str) -> Decision:
    """Parse a decision display page into structured fields."""
    body_region = markup
    anchor = _CONTENT_ANCHOR.search(markup)
    if anchor:
        body_region = markup[anchor.end() :]

    dec = Decision(cd=cd)

    if m := _HEADER_RE.search(body_region):
        dec.number = _int(m.group("num"))
        dec.year = _int(m.group("year"))
        info = html.unescape(m.group("info") or "")
        # Header reads "(Γ, ΠΟΛΙΤΙΚΕΣ)" -- chamber first, category second.
        parts = [p.strip() for p in info.split(",") if p.strip()]
        if len(parts) >= 2:
            dec.chamber, dec.category = parts[0], parts[1]
        elif parts:
            dec.category = parts[0]

    if m := _SUBJECT_RE.search(body_region):
        subject = html_to_text(m.group("val"))
        dec.subject = subject or None
        dec.subjects = [s.strip(" .") for s in subject.split(",") if s.strip(" .")]

    if m := _SUMMARY_RE.search(body_region):
        dec.summary = html_to_text(m.group("val")) or None

    if m := _BODY_RE.search(body_region):
        dec.body = html_to_text(m.group("val"))
    else:
        # Fall back to the whole content region minus the known header blocks.
        text = html_to_text(body_region)
        dec.body = text

    return dec


# ---------------------------------------------------------------------------
# Clean-text JSON endpoint  (apofasi_ai_v2.asp)
# ---------------------------------------------------------------------------


def parse_ai_json(payload: dict) -> str:
    """Extract plain text from the site's own AI-oriented JSON endpoint.

    Used as a fallback when the HTML body extraction looks wrong; it returns
    pre-cleaned text but carries no subject/summary metadata.
    """
    if payload.get("status") != "success":
        return ""
    return (payload.get("document_text") or "").strip()
