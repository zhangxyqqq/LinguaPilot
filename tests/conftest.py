import pytest

from app.storage import DEFAULT_USER_ID, configure_storage, reset_current_user, set_current_user


@pytest.fixture(autouse=True)
def isolated_sqlite_storage(tmp_path, monkeypatch):
    """Keep every deterministic test isolated from runtime learner data."""
    storage = configure_storage(tmp_path / "langbuddy-test.sqlite3")
    monkeypatch.setenv("LANGBUDDY_EMBEDDINGS", "disabled")
    token = set_current_user(DEFAULT_USER_ID)
    try:
        yield storage
    finally:
        reset_current_user(token)
