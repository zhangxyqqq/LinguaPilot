import asyncio
import copy
import json
from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import agent
from app.agent import (
    _read_book_state,
    get_due_words,
    get_weak_words,
    run_agent,
    select_due_words,
    select_weak_words,
)


def test_select_weak_words_uses_only_persisted_signals_and_limit():
    state = {
        "user": {
            "cards": {
                "alpha": {"last_grade": 2},
                "beta": {"last_grade": 5, "last_quiz_score": 0.4, "quiz_wrong_count": 2},
                "gamma": {"last_grade": 5},
                "delta": {"last_quiz_grade": 3},
                "untouched": {"last_grade": None},
            }
        }
    }

    result = select_weak_words(state, limit=2)

    assert result["total_matches"] == 3
    assert len(result["items"]) == 2
    assert result["items"][0] == {
        "word": "beta",
        "signals": {
            "last_grade": 5,
            "last_quiz_score": 0.4,
            "quiz_wrong_count": 2,
        },
        "reason": ["low_quiz_score"],
    }
    assert "untouched" not in {item["word"] for item in result["items"]}


def test_select_due_words_parses_z_offsets_and_naive_datetimes():
    now = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
    state = {
        "user": {
            "cards": {
                "zulu": {"reps": 1, "last_grade": 4, "due_at": "2026-08-13T10:00:00Z"},
                "offset": {"reps": 2, "last_grade": 5, "due_at": "2026-08-14T13:00:00+02:00"},
                "naive": {"reps": 0, "last_grade": 1, "due_at": "2026-08-14T11:30:00"},
                "future": {"reps": 1, "last_grade": 4, "due_at": "2026-08-15T10:00:00Z"},
                "untouched": {"reps": 0, "last_grade": None, "due_at": "2026-08-01T10:00:00Z"},
                "invalid": {"reps": 1, "last_grade": 2, "due_at": "not-a-date"},
            }
        }
    }

    result = select_due_words(state, now=now)

    assert [item["word"] for item in result["items"]] == ["zulu", "offset", "naive"]
    assert result["invalid_due_at_count"] == 1
    assert all(item["reason"] == ["review_due"] for item in result["items"])


def test_selectors_are_read_only_and_handle_empty_state():
    state = {"user": {"cards": {"alpha": {"last_grade": 2, "due_at": "2020-01-01T00:00:00Z"}}}}
    before = copy.deepcopy(state)

    select_weak_words(state)
    select_due_words(state, now=datetime(2026, 1, 1, tzinfo=timezone.utc))

    assert state == before
    assert select_weak_words({}) == {"total_matches": 0, "items": []}
    assert select_due_words({}, now=datetime(2026, 1, 1, tzinfo=timezone.utc))["items"] == []


def test_read_book_state_missing_and_valid(tmp_path):
    with pytest.raises(FileNotFoundError):
        _read_book_state("missing", state_dir=tmp_path)

    expected = {"book_id": "book-1", "user": {"cards": {}}}
    (tmp_path / "book-1.json").write_text(json.dumps(expected), encoding="utf-8")
    assert _read_book_state("book-1", state_dir=tmp_path) == expected


def test_tool_schemas_do_not_expose_book_id_or_runtime_context():
    for learner_tool in (get_weak_words, get_due_words):
        properties = learner_tool.tool_call_schema.model_json_schema()["properties"]
        assert set(properties) == {"limit"}
        assert "book_id" not in properties


class _RoutingModel:
    async def ainvoke(self, messages):
        last = messages[-1]
        if isinstance(last, ToolMessage):
            return AIMessage(content=f"personalized from {last.name}: {last.content}")

        user_text = next(
            message.content
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        ).lower()
        if "weak" in user_text:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_weak_words",
                    "args": {"limit": 5},
                    "id": "weak-call",
                    "type": "tool_call",
                }],
            )
        if "review today" in user_text:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_due_words",
                    "args": {"limit": 5},
                    "id": "due-call",
                    "type": "tool_call",
                }],
            )
        return AIMessage(content="general language answer")


def test_agent_routes_general_weak_and_due_questions(monkeypatch, tmp_path):
    state = {
        "book_id": "book-1",
        "user": {
            "cards": {
                "alpha": {"reps": 0, "last_grade": 2, "due_at": "2020-01-01T00:00:00Z"},
            }
        },
    }
    (tmp_path / "book-1.json").write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(agent, "STATE_DIR", tmp_path)
    monkeypatch.setattr(agent, "_model_with_tools", lambda: _RoutingModel())

    general = asyncio.run(run_agent(
        "book-1",
        "alpha",
        [{"role": "user", "content": "What is the difference between affect and effect?"}],
    ))
    weak = asyncio.run(run_agent(
        "book-1",
        "alpha",
        [{"role": "user", "content": "What are my weak words?"}],
    ))
    due = asyncio.run(run_agent(
        "book-1",
        "alpha",
        [{"role": "user", "content": "What should I review today?"}],
    ))

    assert general == "general language answer"
    assert "get_weak_words" in weak and "alpha" in weak
    assert "get_due_words" in due and "alpha" in due


def test_plain_chat_uses_legacy_fallback_on_agent_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from app import chat, main

    state_path = tmp_path / "book-1.json"
    state_path.write_text(json.dumps({
        "book_id": "book-1",
        "groups": {},
        "ungrouped": [],
        "user": {"cards": {}},
    }), encoding="utf-8")

    async def broken_agent(*args, **kwargs):
        raise RuntimeError("agent unavailable")

    async def legacy_llm(prompt):
        return "legacy answer"

    monkeypatch.setattr(agent, "run_agent", broken_agent)
    monkeypatch.setattr(main, "_state_path", lambda book_id: state_path)
    monkeypatch.setattr(main, "call_llm", legacy_llm)

    result = asyncio.run(chat.vocab_chat(
        "book-1",
        "alpha",
        chat.ChatIn(message="hello"),
    ))

    assert result["messages"][-1]["content"] == "legacy answer"
    assert "Agent failure; using legacy plain-chat fallback" in capsys.readouterr().out
