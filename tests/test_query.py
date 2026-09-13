"""Greek query-construction tests.

These encode the two Postgres defects the query builder exists to work around,
so that a future "simplification" cannot silently undo them.
"""

from __future__ import annotations

import pytest

from apsearch.search.query import (
    STOPWORDS,
    build_prefix_tsquery,
    content_tokens,
    fold,
    has_operators,
    is_acronym,
    trim_unstable_suffix,
)


class TestAcronyms:
    @pytest.mark.parametrize("tok", ["ΑΠ", "ΑΚ", "ΠΚ", "ΚΠΔ", "ΝΔ", "ΚΠολΔ", "ΕΣΔΑ"])
    def test_legal_abbreviations_recognised(self, tok):
        assert is_acronym(tok)

    @pytest.mark.parametrize("tok", ["από", "αν", "Δημοσίου", "αδικοπραξία", "α"])
    def test_ordinary_words_are_not_acronyms(self, tok):
        assert not is_acronym(tok)

    def test_acronym_survives_stopword_filtering(self):
        """ΑΠ (Άρειος Πάγος) and από both stem to 'απ'.

        Filtering on stems would delete the court's own name from queries.
        """
        assert "ΑΠ" in content_tokens("αναίρεση ΑΠ")
        assert "από" not in content_tokens("αναίρεση από το δικαστήριο")

    def test_an_is_ambiguous_and_resolved_by_case(self):
        # ΑΝ = Αναγκαστικός Νόμος; αν = "if"
        assert "ΑΝ" in content_tokens("ΑΝ 173/1967")
        assert "αν" not in content_tokens("αν συντρέχει λόγος")


class TestStopwords:
    def test_common_function_words_dropped(self):
        assert content_tokens("η παραγραφή των αξιώσεων κατά του Δημοσίου") == [
            "παραγραφή", "αξιώσεων", "Δημοσίου",
        ]

    def test_numbers_always_kept(self):
        assert content_tokens("άρθρο 559 ΚΠολΔ") == ["άρθρο", "559", "ΚΠολΔ"]

    def test_stopword_list_is_accent_folded(self):
        for w in STOPWORDS:
            assert fold(w) == w, f"{w!r} is not in folded form"


class TestPrefixExpansion:
    def test_inflection_variants_converge(self):
        """The whole point: nominative and genitive must produce one prefix."""
        assert trim_unstable_suffix("αδικοπραξι") == "αδικοπραξ"
        assert trim_unstable_suffix("αδικοπραξ") == "αδικοπραξ"

    def test_short_stems_are_not_truncated(self):
        # Truncating these would match far too much.
        assert trim_unstable_suffix("δικα") == "δικα"
        assert trim_unstable_suffix("αρθρ") == "αρθρ"

    def test_short_lexemes_stay_exact(self):
        q = build_prefix_tsquery(["ακ", "αδικοπραξ"])
        assert "ακ &" in q or "ακ " in q
        assert "ακ:*" not in q
        assert "αδικοπραξ:*" in q

    def test_conjunctive_and_disjunctive(self):
        assert build_prefix_tsquery(["αναιρεσ", "αποφασ"], True) == "αναιρεσ:* & αποφασ:*"
        assert build_prefix_tsquery(["αναιρεσ", "αποφασ"], False) == "αναιρεσ:* | αποφασ:*"

    def test_duplicates_collapsed(self):
        assert build_prefix_tsquery(["αναιρεσ", "αναιρεσ"]) == "αναιρεσ:*"

    def test_empty(self):
        assert build_prefix_tsquery([]) == ""


class TestOperatorDetection:
    @pytest.mark.parametrize("q", ['"λόγος αναιρέσεως"', "παραγραφή -ποινική", "α OR β"])
    def test_explicit_syntax_detected(self, q):
        assert has_operators(q)

    @pytest.mark.parametrize("q", ["αδικοπραξία και αποζημίωση", "άρθρο 559 ΚΠολΔ"])
    def test_plain_queries_are_not_operators(self, q):
        assert not has_operators(q)
