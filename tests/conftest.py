import pytest

from app.storage import DEFAULT_USER_ID, configure_storage, reset_current_user, set_current_user


@pytest.fixture(autouse=True)
def isolated_sqlite_storage(tmp_path, monkeypatch):
    """Keep every deterministic test isolated from runtime learner data."""
    storage = configure_storage(tmp_path / "langbuddy-test.sqlite3")
    monkeypatch.setenv("LANGBUDDY_EMBEDDINGS", "disabled")
    monkeypatch.setenv("LANGBUDDY_RETRIEVAL_MODE", "legacy")
    monkeypatch.setenv("LANGBUDDY_RERANK", "disabled")
    # Fail closed if a deterministic test accidentally reaches a paid provider.
    import httpx
    def forbid_network(*args, **kwargs):
        raise AssertionError("external HTTP is forbidden in deterministic tests")
    async def forbid_async_network(*args, **kwargs):
        raise AssertionError("external HTTP is forbidden in deterministic tests")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbid_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbid_async_network)
    token = set_current_user(DEFAULT_USER_ID)
    try:
        yield storage
    finally:
        reset_current_user(token)
