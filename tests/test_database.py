"""Tests for the small subset of TriageDatabase methods used by the
batch-redo flow in app.py."""

from __future__ import annotations

import pytest

from triage.database import TriageDatabase


@pytest.fixture()
def db(tmp_path):
    return TriageDatabase(db_path=str(tmp_path / "triage_test.db"))


def _insert(db: TriageDatabase, label: str, batch_id: str | None) -> int:
    return db.save_analysis(
        incident_text=f"row for {label}/{batch_id}",
        final_label=label,
        max_prob=0.9,
        analysis_mode="batch" if batch_id else "single",
        batch_id=batch_id,
    )


def test_delete_batch_removes_only_matching_rows(db):
    a1 = _insert(db, "malware", "batch-a")
    a2 = _insert(db, "phishing", "batch-a")
    b1 = _insert(db, "benign", "batch-b")

    n = db.delete_batch("batch-a")

    assert n == 2
    assert db.get_analysis_by_id(a1) is None
    assert db.get_analysis_by_id(a2) is None
    assert db.get_analysis_by_id(b1) is not None


def test_delete_batch_cascades_to_dependents(db):
    aid = _insert(db, "malware", "batch-x")

    db.add_bookmark(
        incident_text="row for malware/batch-x",
        final_label="malware",
        analysis_id=aid,
    )
    db.add_note(note_text="dig deeper", analysis_id=aid)
    tag_id = db.create_tag("priority")
    db.add_tag_to_analysis(aid, tag_id)

    assert any(b.get("analysis_id") == aid for b in db.get_bookmarks())
    assert db.get_notes_for_analysis(aid)
    assert db.get_tags_for_analysis(aid)

    db.delete_batch("batch-x")

    assert db.get_analysis_by_id(aid) is None
    assert not any(b.get("analysis_id") == aid for b in db.get_bookmarks())
    assert db.get_notes_for_analysis(aid) == []
    assert db.get_tags_for_analysis(aid) == []
    # Tag definition itself is preserved; only the association is dropped.
    assert any(t["name"] == "priority" for t in db.get_all_tags())


def test_delete_batch_idempotent_on_unknown_id(db):
    _insert(db, "malware", "batch-real")

    assert db.delete_batch("batch-does-not-exist") == 0
    assert db.delete_batch("") == 0
    # Calling twice on the same batch_id: first wipes, second is a no-op.
    assert db.delete_batch("batch-real") == 1
    assert db.delete_batch("batch-real") == 0


def test_count_history_and_count_demo_events(db):
    assert db.count_history() == 0
    assert db.count_demo_events() == 0

    _insert(db, "malware", "demo")
    _insert(db, "phishing", "demo")
    _insert(db, "benign", None)
    _insert(db, "web_attack", "user-batch-xyz")

    assert db.count_history() == 4
    assert db.count_demo_events() == 2

    db.delete_batch("demo")
    assert db.count_history() == 2
    assert db.count_demo_events() == 0
