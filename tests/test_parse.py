"""Parser tests. All run against saved fixtures -- never the live site."""

from __future__ import annotations

from pathlib import Path

import pytest

from apsearch.crawler.parse import (
    fold_greek,
    html_to_text,
    parse_decision,
    parse_listing,
    parse_theme_index,
)

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestListing:
    def test_parses_refs_and_metadata(self):
        refs, truncated = parse_listing(fixture("listing_1_2024.html"))
        assert not truncated
        assert len(refs) == 6
        first = refs[0]
        assert first.cd == "3WAN68WIXBZE1RD7Z427YKZ081VMSG"
        assert (first.number, first.year) == (1, 2024)
        assert first.citation == "1/2024"
        assert first.category == "ΠΟΛΙΤΙΚΕΣ"
        assert first.chamber == "Γ"

    def test_detects_the_3000_row_cap(self):
        """The site truncates silently; missing this would lose data."""
        refs, truncated = parse_listing(fixture("listing_year2024.html"))
        assert truncated is True
        assert len(refs) == 3000
        assert len({r.cd for r in refs}) == 3000

    def test_theme_listing_carries_summaries(self):
        refs, truncated = parse_listing(fixture("theme_497.html"))
        assert not truncated
        assert refs[0].snippet
        assert "Διεκδικητική αγωγή" in refs[0].snippet

    def test_empty_markup_is_not_an_error(self):
        refs, truncated = parse_listing("<html><body>nothing</body></html>")
        assert refs == []
        assert truncated is False


class TestThemeIndex:
    def test_extracts_controlled_vocabulary(self):
        themes = parse_theme_index(fixture("theme_index.html"))
        assert len(themes) > 700
        by_code = {t.code: t for t in themes}
        assert by_code[497].label == "Αδικοπραξία"
        # Underscores in the site's slugs must become spaces in the label.
        assert all("_" not in t.label for t in themes)


class TestDecision:
    def test_parses_full_record(self):
        d = parse_decision(fixture("decision_144_2015.html"), "CD1")
        assert (d.number, d.year) == (144, 2015)
        assert d.chamber == "Γ"
        assert d.category == "ΠΟΛΙΤΙΚΕΣ"
        assert "Αδικοπραξία" in d.subjects
        assert d.summary and "Διεκδικητική αγωγή ακινήτου" in d.summary
        assert d.is_usable
        assert d.body.startswith("Αριθμός 144/2015")
        # Body must stop before the page's navigation chrome.
        assert "Επιστροφή" not in d.body
        assert "ΔΗΜΟΣΙΕΥΘΗΚΕ" in d.body

    def test_handles_decision_without_headnote(self):
        """Older/plain decisions have no Θέμα or Περίληψη; body must survive."""
        d = parse_decision(fixture("decision_1_2024.html"), "CD2")
        assert (d.number, d.year) == (1, 2024)
        assert d.subjects == []
        assert d.summary is None
        assert d.is_usable and len(d.body) > 30000

    def test_strips_ai_modal_chrome(self):
        """The page embeds an LLM prompt in a <script>; it must not leak in."""
        d = parse_decision(fixture("decision_1_2024.html"), "CD2")
        assert "prompt_for_AI" not in d.body
        assert "ChatGPT" not in d.body


class TestTextNormalisation:
    def test_handles_unterminated_nbsp_entity(self):
        # The site emits `&nbsp` with no semicolon.
        assert html_to_text("a&nbsp&nbspb") == "a b"

    def test_br_becomes_newline(self):
        assert html_to_text("a<br>b") == "a\nb"

    @pytest.mark.parametrize(
        "a,b",
        [
            ("Αδικοπραξία", "αδικοπραξια"),
            ("ΑΝΑΊΡΕΣΗ", "αναιρεση"),
            ("λόγος", "λογοσ"),  # final sigma normalised
        ],
    )
    def test_fold_greek(self, a, b):
        assert fold_greek(a) == b
