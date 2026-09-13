"""Chunking tests.

The regression test here guards a bug that cost 7x in embedding spend before it
was caught, so it is worth keeping explicit.
"""

from __future__ import annotations

from pathlib import Path

from apsearch.crawler.parse import parse_decision
from apsearch.index.chunk import chunk_decision, split_sections

FIXTURES = Path(__file__).parent / "fixtures"


def real_decision():
    return parse_decision(
        (FIXTURES / "decision_144_2015.html").read_text(encoding="utf-8"), "CD1"
    )


class TestSectioning:
    def test_splits_on_structural_headings(self):
        body = (
            "Αριθμός 1/2020 κείμενο εδώ. " * 3
            + "ΣΚΕΦΘΗΚΕ ΣΥΜΦΩΝΑ ΜΕ ΤΟ ΝΟΜΟ σκεπτικό. " * 3
            + "ΓΙΑ ΤΟΥΣ ΛΟΓΟΥΣ ΑΥΤΟΥΣ διατακτικό."
        )
        sections = split_sections(body)
        assert len(sections) >= 3
        # Sections must tile the document exactly: no gaps, no overlaps.
        assert sections[0][0] == 0
        assert sections[-1][1] == len(body)
        for (_, end), (start, _) in zip(sections, sections[1:], strict=False):
            assert end == start


class TestChunking:
    def test_headnote_becomes_its_own_chunk(self):
        d = real_decision()
        chunks = chunk_decision(d.body, d.summary, d.subject)
        assert chunks[0].part == "summary"
        # Subject headings are prepended so the abstract carries its topic.
        assert "Αδικοπραξία" in chunks[0].content

    def test_chunks_are_near_target_size(self):
        d = real_decision()
        chunks = [c for c in chunk_decision(d.body) if c.part == "body"]
        sizes = [len(c.content) for c in chunks]
        assert max(sizes) < 3500
        assert sum(sizes) / len(sizes) > 400

    def test_no_runaway_duplication(self):
        """Regression: a stalled cursor emitted 757 overlapping spans.

        With `chunk_target=1400` and `overlap=200` the expected coverage is
        ~1.2x the source length. Anything near 2x means the packing loop is
        failing to make forward progress and re-emitting the same text.
        """
        d = real_decision()
        chunks = [c for c in chunk_decision(d.body) if c.part == "body"]
        coverage = sum(len(c.content) for c in chunks) / len(d.body)
        assert coverage < 1.45, f"chunk coverage {coverage:.2f}x indicates duplication"
        # And no degenerate slivers.
        assert min(len(c.content) for c in chunks) >= 50

    def test_pathological_input_terminates(self):
        """Text with one early sentence break then a long run used to stall."""
        body = "Α. " + "x" * 60_000
        chunks = chunk_decision(body)
        assert 0 < len(chunks) < 100
        coverage = sum(len(c.content) for c in chunks) / len(body)
        assert coverage < 1.5

    def test_ordinals_are_dense_and_ordered(self):
        d = real_decision()
        chunks = chunk_decision(d.body, d.summary, d.subject)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_empty_body(self):
        assert chunk_decision("") == []
        only_summary = chunk_decision("", summary="περίληψη εδώ")
        assert len(only_summary) == 1
        assert only_summary[0].part == "summary"
