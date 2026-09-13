"""Chunking Greek court decisions for passage-level retrieval.

Areios Pagos decisions have a stable rhetorical skeleton, and the boundaries
between those parts are the most meaningful places to split:

    Αριθμός N/YYYY            header, bench composition
    ΣΚΕΦΘΗΚΕ ΣΥΜΦΩΝΑ ΜΕ ΤΟ ΝΟΜΟ   the legal reasoning (the part people search)
    ΓΙΑ ΤΟΥΣ ΛΟΓΟΥΣ ΑΥΤΟΥΣ        the operative part / disposition

Within the reasoning, numbered sections (Ι., ΙΙ., Α., 1.) each develop one
legal issue. We therefore split on structural markers first and only fall back
to sentence packing inside an oversized section -- so a chunk rarely straddles
two unrelated legal points.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from apsearch.config import settings

#: Major structural headings, in document order.
SECTION_MARKERS = [
    r"ΣΚΕΦΘΗΚΕ\s+ΣΥΜΦΩΝΑ\s+ΜΕ\s+ΤΟ\s+ΝΟΜΟ",
    r"ΓΙΑ\s+ΤΟΥΣ\s+ΛΟΓΟΥΣ\s+ΑΥΤΟΥΣ",
    r"ΚΡΙΘΗΚΕ[,\s]",
    r"ΔΗΜΟΣΙΕΥΘΗΚΕ[,\s]",
]
_SECTION_RE = re.compile("|".join(f"({m})" for m in SECTION_MARKERS))

#: Numbered subdivisions at the start of a line: "Ι.", "ΙΙΙ.", "Α.", "1.", "2.-"
_SUBDIVISION_RE = re.compile(
    r"^\s*(?:[ΙVΧ]{1,5}|[Α-Ω]|\d{1,2})\s*[\.\)\-]\s+",
    re.MULTILINE,
)

#: Greek sentence terminators. Note "." is also used in abbreviations
#: (π.χ., άρθρ., Α.Π.), so require a following capital / whitespace+capital.
_SENTENCE_RE = re.compile(r"(?<=[\.\;\·\!\?])\s+(?=[Α-ΩΆΈΉΊΌΎΏA-Z])")


@dataclass(slots=True)
class Chunk:
    ordinal: int
    content: str
    part: str
    char_start: int
    char_end: int


def split_sections(text: str) -> list[tuple[int, int]]:
    """Offsets of the document's top-level structural sections."""
    bounds = [0]
    for m in _SECTION_RE.finditer(text):
        if m.start() > bounds[-1]:
            bounds.append(m.start())
    bounds.append(len(text))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]


def _split_oversized(text: str, offset: int, target: int, overlap: int) -> list[tuple[int, int]]:
    """Split a too-large block, preferring subdivision then sentence boundaries."""
    # Prefer numbered subdivisions.
    marks = [m.start() for m in _SUBDIVISION_RE.finditer(text)]
    pieces: list[tuple[int, int]] = []
    if marks:
        bounds = sorted({0, *marks, len(text)})
        cur_start = bounds[0]
        for i in range(1, len(bounds)):
            if bounds[i] - cur_start >= target or i == len(bounds) - 1:
                pieces.append((cur_start, bounds[i]))
                cur_start = bounds[i]
        pieces = [p for p in pieces if p[1] > p[0]]

    # Any piece still too large gets packed by sentence with overlap.
    out: list[tuple[int, int]] = []
    for start, end in pieces or [(0, len(text))]:
        block = text[start:end]
        if len(block) <= target * 1.5:
            out.append((offset + start, offset + end))
            continue
        out.extend(_pack_by_sentence(block, offset + start, target, overlap))
    return out


def _pack_by_sentence(
    block: str, offset: int, target: int, overlap: int
) -> list[tuple[int, int]]:
    """Pack a block into ~`target`-sized spans ending on sentence boundaries.

    Two invariants matter here, and violating either is expensive:

    * **Forward progress.** ``cursor`` must always advance. Naively setting
      ``cursor = stop - overlap`` fails when the chosen boundary lands within
      ``overlap`` of the cursor: the cursor then creeps forward one character
      per iteration, emitting hundreds of near-duplicate spans. (Observed on a
      real 79k-character decision: 757 spans, some a single character long,
      covering the text 2.1x.) When the overlap would not move us forward we
      simply continue from ``stop``.
    * **No degenerate boundaries.** A candidate boundary is only accepted if it
      is at least ``min_chunk`` past the cursor, so a cluster of short
      sentences cannot produce a stream of tiny chunks.
    """
    n = len(block)
    if n == 0:
        return []
    min_chunk = max(target // 3, 1)
    breaks = [m.start() for m in _SENTENCE_RE.finditer(block)]

    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < n:
        limit = min(cursor + target, n)
        if limit >= n:
            stop = n
        else:
            candidates = [b for b in breaks if cursor + min_chunk < b <= limit]
            stop = candidates[-1] if candidates else limit
        spans.append((offset + cursor, offset + stop))
        if stop >= n:
            break
        # Guaranteed progress: fall back to `stop` if the overlap would stall.
        nxt = stop - overlap
        cursor = nxt if nxt > cursor else stop
    return spans


def chunk_decision(
    body: str,
    summary: str | None = None,
    subject: str | None = None,
    target: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Produce retrieval units for one decision.

    The headnote (Περίληψη) is emitted as its own chunk: it is an editorial
    abstract of exactly what the decision holds, so it is usually the single
    highest-signal passage in the document.
    """
    target = target or settings.chunk_target_chars
    overlap = overlap or settings.chunk_overlap_chars
    chunks: list[Chunk] = []
    ordinal = 0

    if summary and summary.strip():
        head = summary.strip()
        if subject:
            head = f"{subject.strip()}\n\n{head}"
        chunks.append(Chunk(ordinal, head, "summary", 0, 0))
        ordinal += 1

    body = (body or "").strip()
    if not body:
        return chunks

    for sec_start, sec_end in split_sections(body):
        block = body[sec_start:sec_end]
        if not block.strip():
            continue
        if len(block) <= target * 1.5:
            spans = [(sec_start, sec_end)]
        else:
            spans = _split_oversized(block, sec_start, target, overlap)
        for a, b in spans:
            content = body[a:b].strip()
            if len(content) < 50:  # drop headings-only fragments
                continue
            chunks.append(Chunk(ordinal, content, "body", a, b))
            ordinal += 1

    return chunks
