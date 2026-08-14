import copy
import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app import agent, chat, session_quiz
from app.storage import (
    DEFAULT_USER_ID,
    MigrationConflictError,
    SQLiteStorage,
    reset_current_user,
    set_current_user,
)


def _state(book_id: str, word: str, grade=None):
    return {
        "book_id": book_id,
        "source": f"book_{book_id}.csv",
        "groups": {
            "test": {
                "type": "test",
                "label": "test",
                "words": [{"word": word, "decomposition": ""}],
            }
        },
        "ungrouped": [],
        "user": {"cards": {word: {"last_grade": grade}}},
    }


def test_legacy_json_migration_is_copy_only_and_restart_safe(tmp_path):
    legacy_dir = tmp_path / "legacy"
    materials_dir = legacy_dir / "materials"
    materials_dir.mkdir(parents=True)
    state_path = legacy_dir / "legacy-book.json"
    state_path.write_text(json.dumps(_state("legacy-book", "alpha", 2)), encoding="utf-8")
    material_path = materials_dir / "legacy-book.json"
    material_path.write_text(json.dumps({
        "version": 1,
        "book_id": "legacy-book",
        "documents": [{"document_id": "doc-1", "source_name": "notes.txt"}],
        "chunks": [{
            "document_id": "doc-1",
            "source_name": "notes.txt",
            "chunk_index": 0,
            "text": "durable notes",
        }],
    }), encoding="utf-8")
    before_state = state_path.read_bytes()
    before_material = material_path.read_bytes()
    db_path = tmp_path / "learner.sqlite3"

    first = SQLiteStorage(db_path, legacy_dir)
    first.initialize()

    assert first.migration_summary.books_imported == 1
    assert first.migration_summary.materials_imported == 1
    assert first.load_book_state("legacy-book", DEFAULT_USER_ID)["user"]["cards"]["alpha"]["last_grade"] == 2
    assert first.load_material_store("legacy-book", DEFAULT_USER_ID)["chunks"][0]["text"] == "durable notes"
    assert state_path.read_bytes() == before_state
    assert material_path.read_bytes() == before_material

    restarted = SQLiteStorage(db_path, legacy_dir)
    restarted.initialize()
    assert restarted.migration_summary.books_imported == 0
    assert restarted.load_book_state("legacy-book", DEFAULT_USER_ID) == first.load_book_state(
        "legacy-book", DEFAULT_USER_ID
    )


def test_changed_legacy_json_fails_closed_after_migration(tmp_path):
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    state_path = legacy_dir / "book-1.json"
    state_path.write_text(json.dumps(_state("book-1", "alpha", 2)), encoding="utf-8")
    db_path = tmp_path / "learner.sqlite3"
    SQLiteStorage(db_path, legacy_dir).initialize()

    changed = _state("book-1", "alpha", 5)
    state_path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(MigrationConflictError, match="changed after SQLite import"):
        SQLiteStorage(db_path, legacy_dir).initialize()


def test_user_book_and_material_isolation(tmp_path):
    storage = SQLiteStorage(tmp_path / "isolated.sqlite3")
    storage.save_book_state(_state("shared-book", "alpha", 1), "alice")
    storage.save_book_state(_state("shared-book", "beta", 5), "bob")
    storage.save_material_store(
        "shared-book",
        {"version": 1, "book_id": "shared-book", "documents": [], "chunks": [{"text": "alice only"}]},
        "alice",
    )

    assert "alpha" in storage.load_book_state("shared-book", "alice")["user"]["cards"]
    assert "beta" in storage.load_book_state("shared-book", "bob")["user"]["cards"]
    assert storage.load_book_state("shared-book", "charlie") is None
    assert storage.load_material_store("shared-book", "alice")["chunks"][0]["text"] == "alice only"
    assert storage.load_material_store("shared-book", "bob") is None
    assert [item["bookId"] for item in storage.list_books("alice")] == ["shared-book"]


def test_updates_persist_across_repository_restart(tmp_path):
    db_path = tmp_path / "restart.sqlite3"
    first = SQLiteStorage(db_path)
    original = _state("book-1", "alpha", 2)
    first.save_book_state(original, "alice")
    updated = copy.deepcopy(first.load_book_state("book-1", "alice"))
    updated["user"]["cards"]["alpha"].update({"last_grade": 4, "reps": 2})
    first.save_book_state(updated, "alice")

    restarted = SQLiteStorage(db_path)
    card = restarted.load_book_state("book-1", "alice")["user"]["cards"]["alpha"]
    assert card == {"last_grade": 4, "reps": 2}


def test_api_identity_scopes_existing_review_flow(isolated_sqlite_storage):
    isolated_sqlite_storage.save_book_state(_state("shared-book", "alpha"), "alice")
    isolated_sqlite_storage.save_book_state(_state("shared-book", "beta", 5), "bob")

    with TestClient(main.app) as client:
        alice_books = client.get("/api/books", headers={"X-User-ID": "alice"})
        bob_books = client.get("/api/books", headers={"X-User-ID": "bob"})
        missing = client.get("/api/books/shared-book/overview", headers={"X-User-ID": "charlie"})
        reviewed = client.post(
            "/api/review/shared-book",
            headers={"X-User-ID": "alice"},
            json={"word": "alpha", "grade": 4},
        )
        uploaded = client.post(
            "/api/materials/shared-book",
            headers={"X-User-ID": "alice"},
            files={"file": ("alice.txt", b"Alice-specific learning note.", "text/plain")},
        )
        bob_materials = client.get(
            "/api/materials/shared-book", headers={"X-User-ID": "bob"}
        )
        invalid_identity = client.get("/api/books", headers={"X-User-ID": "bad user"})

    assert alice_books.status_code == bob_books.status_code == 200
    assert alice_books.headers["X-LangBuddy-User"] == "alice"
    assert alice_books.json()["items"][0]["bookId"] == "shared-book"
    assert bob_books.json()["items"][0]["bookId"] == "shared-book"
    assert missing.status_code == 404
    assert reviewed.status_code == 200
    assert uploaded.status_code == 200
    assert bob_materials.status_code == 200
    assert bob_materials.json()["items"] == []
    assert invalid_identity.status_code == 400
    assert isolated_sqlite_storage.load_book_state("shared-book", "alice")["user"]["cards"]["alpha"]["last_grade"] == 4
    assert isolated_sqlite_storage.load_book_state("shared-book", "bob")["user"]["cards"]["beta"]["last_grade"] == 5


def test_sqlite_memory_chat_and_quiz_updates_are_user_scoped(
    isolated_sqlite_storage, monkeypatch
):
    isolated_sqlite_storage.save_book_state(_state("shared-book", "alpha"), "alice")
    isolated_sqlite_storage.save_book_state(_state("shared-book", "alpha", 5), "bob")

    async def deterministic_agent(*args, **kwargs):
        return "saved preference acknowledged"

    monkeypatch.setattr(agent, "run_agent", deterministic_agent)
    alice_token = set_current_user("alice")
    try:
        asyncio.run(chat.global_chat(
            "shared-book",
            chat.GlobalChatIn(message="I prefer short explanations."),
        ))
        quiz_update = session_quiz._apply_quiz_results_to_cards(
            "shared-book", {"alpha": {"total": 2, "correct": 1}}
        )
    finally:
        reset_current_user(alice_token)

    alice = isolated_sqlite_storage.load_book_state("shared-book", "alice")
    bob = isolated_sqlite_storage.load_book_state("shared-book", "bob")
    assert quiz_update["updated_count"] == 1
    assert alice["memory"]["preferences"]["response_style"]["value"] == "concise"
    assert alice["cache"]["global_chat"][-1]["content"] == "saved preference acknowledged"
    assert alice["user"]["cards"]["alpha"]["last_quiz_score"] == 0.5
    assert bob["user"]["cards"]["alpha"]["last_grade"] == 5
    assert "memory" not in bob
