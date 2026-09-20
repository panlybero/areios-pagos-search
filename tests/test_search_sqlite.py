"""Full retrieval-pipeline tests against a real (in-memory) SQLite database.

Uses the deterministic hashing embedding backend so nothing hits the network.
"""

from __future__ import annotations

import pytest

import apsearch.db.sqlite as sqlite_mod
from apsearch.config import settings
from apsearch.crawler.parse import Decision, Theme
from apsearch.db.sqlite import fold_greek


@pytest.fixture
def env_db(tmp_path, monkeypatch):
    """An isolated SQLite DB with the hashing embedding backend (dim 4)."""
    monkeypatch.setattr(settings, "embed_backend", "hashing-test")
    monkeypatch.setattr(settings, "embed_dim", 4)
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "test.db"))
    # Don't try to seed a real corpus during tests.
    monkeypatch.setattr(sqlite_mod, "ensure_seed_db", lambda path: None)

    from apsearch.index.embed import reset_backend

    reset_backend()

    from apsearch.db import migrate

    migrate()
    yield
    reset_backend()


def _decision(cd: str, number: int, year: int, body: str, subject: str = "Θέμα") -> Decision:
    return Decision(
        cd=cd,
        number=number,
        year=year,
        category="ΠΟΛΙΤΙΚΕΣ",
        chamber="Α1",
        subject=subject,
        summary=f"Περίληψη {subject}",
        body=body,
    )


BODY_A = (
    "ΣΚΕΦΘΗΚΕ ΣΥΜΦΩΝΑ ΜΕ ΤΟ ΝΟΜΟ. Η παραγραφή των αξιώσεων κατά του Δημοσίου "
    "διέπεται από το άρθρο 90 του νόμου 2362/1995. Η αξίωση αποζημιώσεως από "
    "αδικοπραξία υπόκειται σε πενταετή παραγραφή. Το δικαστήριο έκρινε ότι η "
    "αγωγή ασκήθηκε εμπροθέσμως. ΓΙΑ ΤΟΥΣ ΛΟΓΟΥΣ ΑΥΤΟΥΣ απορρίπτει την αίτηση."
)

BODY_B = (
    "ΣΚΕΦΘΗΚΕ ΣΥΜΦΩΝΑ ΜΕ ΤΟ ΝΟΜΟ. Η σύμβαση εργασίας αορίστου χρόνου καταγγέλθηκε "
    "καταχρηστικώς. Ο μισθωτός δικαιούται αποζημίωση απόλυσης. ΓΙΑ ΤΟΥΣ ΛΟΓΟΥΣ "
    "ΑΥΤΟΥΣ δέχεται εν μέρει την αγωγή του εργαζομένου."
)


class TestFullPipeline:
    def test_keyword_semantic_and_hybrid_search(self, env_db):
        from apsearch import repo
        from apsearch.index.build import run_index
        from apsearch.search.hybrid import search

        repo.upsert_decision(_decision("cd_a", 100, 2024, BODY_A, "Παραγραφή"))
        repo.upsert_decision(_decision("cd_b", 101, 2024, BODY_B, "Εργατικό"))
        run_index()

        kw = search("παραγραφή", mode="keyword", limit=5)
        assert kw and any(r.cd == "cd_a" for r in kw)

        sem = search("καταγγελία σύμβασης εργασίας", mode="semantic", limit=5)
        assert isinstance(sem, list)

        hyb = search("παραγραφή αδικοπραξίας", mode="hybrid", limit=5)
        assert any(r.cd == "cd_a" for r in hyb)

    def test_get_decision_by_citation(self, env_db):
        from apsearch import repo
        from apsearch.index.build import run_index
        from apsearch.search.hybrid import get_decision

        repo.upsert_decision(_decision("cd_a", 100, 2024, BODY_A, "Παραγραφή"))
        run_index()

        rows = get_decision(number=100, year=2024)
        assert len(rows) == 1
        assert rows[0]["cd"] == "cd_a"
        assert rows[0]["body"].startswith("ΣΚΕΦΘΗΚΕ")

    def test_list_themes_counts_links(self, env_db):
        from apsearch import repo
        from apsearch.search.hybrid import list_themes

        repo.upsert_themes([Theme(code=1, label="Παραγραφή", slug="Παραγραφή")])
        repo.upsert_decision(_decision("cd_a", 100, 2024, BODY_A, "Παραγραφή"))
        repo.link_theme(1, ["cd_a"])

        rows = list_themes("Παραγραφή")
        assert any(r["label"] == "Παραγραφή" and r["n_decisions"] == 1 for r in rows)


class TestContentlessFtsUpsert:
    """Regression: contentless decision_fts must not retain stale tokens after an update."""

    def test_reindexed_decision_drops_old_terms(self, env_db):
        from apsearch import repo
        from apsearch.db.sqlite import connect

        repo.upsert_decision(_decision("cd_a", 100, 2024, BODY_A, "Παραγραφή"))

        with connect() as conn:
            hit = conn.execute(
                "SELECT rowid FROM decision_fts WHERE decision_fts MATCH ?",
                (fold_greek("παραγραφή"),),
            ).fetchone()
            assert hit is not None

        # Same cd, different body: "παραγραφή" disappears, "καταγγέλθηκε" appears.
        repo.upsert_decision(_decision("cd_a", 100, 2024, BODY_B, "Εργατικό"))

        with connect() as conn:
            old = conn.execute(
                "SELECT rowid FROM decision_fts WHERE decision_fts MATCH ?",
                (fold_greek("παραγραφή"),),
            ).fetchone()
            new = conn.execute(
                "SELECT rowid FROM decision_fts WHERE decision_fts MATCH ?",
                (fold_greek("καταγγέλθηκε"),),
            ).fetchone()
        assert old is None
        assert new is not None


class TestMCPTools:
    @pytest.mark.asyncio
    async def test_all_tools_registered(self, env_db):
        from apsearch.mcp.server import build_server

        names = {t.name for t in await build_server().list_tools()}
        assert names == {
            "search_decisions", "get_decision", "list_themes", "corpus_stats",
        }

    @pytest.mark.asyncio
    async def test_get_decision_rejects_bad_citation(self, env_db):
        from apsearch.mcp.server import build_server

        server = build_server()
        result = await server.call_tool("get_decision", {"citation": "nonsense"})
        payload = result[1] if isinstance(result, tuple) else result
        assert "error" in str(payload)
