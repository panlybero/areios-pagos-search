"""Fusion maths and crawler-politeness tests (no network, no database)."""

from __future__ import annotations

import time

import pytest

from apsearch.crawler.client import RateLimiter, encode_form
from apsearch.crawler.discover import CHAMBERS, Partition
from apsearch.search.hybrid import rrf


class TestRRF:
    def test_agreement_between_retrievers_wins(self):
        fused = rrf({
            "chunk_lexical": ["a", "b", "c"],
            "semantic": ["b", "a", "d"],
        }, weights={"chunk_lexical": 1.0, "semantic": 1.0}, k=60)
        order = sorted(fused, key=lambda cd: fused[cd][0], reverse=True)
        # 'b' is 2nd+1st, 'a' is 1st+2nd -> tie broken consistently; both beat
        # the items that only one retriever found.
        assert set(order[:2]) == {"a", "b"}
        assert set(order[2:]) == {"c", "d"}

    def test_only_best_rank_per_source_counts(self):
        fused = rrf({"semantic": ["a", "a", "a"]}, k=60)
        assert fused["a"][1] == {"semantic": 1}
        assert fused["a"][0] == pytest.approx(1 / 61)

    def test_weights_are_applied(self):
        fused = rrf(
            {"chunk_lexical": ["a"], "semantic": ["b"]},
            weights={"chunk_lexical": 2.0, "semantic": 1.0},
            k=60,
        )
        assert fused["a"][0] > fused["b"][0]

    def test_records_provenance(self):
        fused = rrf({"chunk_lexical": ["a"], "semantic": ["a"]}, k=60)
        assert set(fused["a"][1]) == {"chunk_lexical", "semantic"}

    def test_empty(self):
        assert rrf({}) == {}


class TestPoliteness:
    def test_rate_limiter_enforces_minimum_interval(self):
        limiter = RateLimiter(0.25)
        start = time.monotonic()
        for _ in range(3):
            limiter.wait()
        # First call is free; the next two must each wait >= 0.9 * interval.
        assert time.monotonic() - start >= 0.4

    def test_backoff_penalty_delays_everyone(self):
        limiter = RateLimiter(0.0)
        limiter.penalise(0.3)
        start = time.monotonic()
        limiter.wait()
        assert time.monotonic() - start >= 0.25


class TestFormEncoding:
    def test_greek_encoded_as_windows_1253(self):
        """The site is a legacy ASP app; UTF-8 form bodies come back as mojibake."""
        body = encode_form({"x": "Αναίρεση"})
        assert body == b"x=%C1%ED%E1%DF%F1%E5%F3%E7"

    def test_ascii_unaffected(self):
        assert encode_form({"X_TMHMA": "6"}) == b"X_TMHMA=6"


class TestPartitions:
    def test_form_payload_matches_site_fields(self):
        form = Partition(year=2024, category_id=2, chamber_id=9).form()
        assert form["x_ETOS"] == "2024"
        assert form["X_TMHMA"] == "2"
        assert form["X_SUB_TMHMA"] == "9"
        assert form["x_number"] == ""

    def test_number_bound_is_rendered(self):
        form = Partition(year=2024, number=500, number_op=3).form()
        assert form["x_number"] == "500"
        assert form["X_TELESTIS_number"] == "3"

    def test_chamber_table_matches_site(self):
        assert CHAMBERS[11] == "ΟΛΟΜΕΛΕΙΑ"
        assert CHAMBERS[13] == "Α1"
