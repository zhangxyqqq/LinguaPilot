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
    get_learner_memory,
    get_weak_words,
    run_agent,
    select_due_words,
    select_weak_words,
    search_learning_materials,
)
from app.memory import (
    apply_explicit_memory,
    compact_preferences,
    extract_explicit_memory,
    learner_memory_view,
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

    assert get_learner_memory.tool_call_schema.model_json_schema()["properties"] == {}
    material_properties = search_learning_materials.tool_call_schema.model_json_schema()["properties"]
    assert set(material_properties) == {"query", "limit"}
    assert "book_id" not in material_properties


class _RoutingModel:
    async def ainvoke(self, messages):
        last = messages[-1]
        if isinstance(last, ToolMessage):
            tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
            names = ",".join(message.name for message in tool_messages)
            content = " ".join(str(message.content) for message in tool_messages)
            return AIMessage(content=f"personalized from {names}: {content}")

        user_text = next(
            message.content
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        ).lower()
        if "combine learner and notes" in user_text:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_weak_words",
                        "args": {"limit": 5},
                        "id": "combined-weak-call",
                        "type": "tool_call",
                    },
                    {
                        "name": "search_learning_materials",
                        "args": {"query": "passive voice", "limit": 3},
                        "id": "combined-material-call",
                        "type": "tool_call",
                    },
                ],
            )
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
        if "remember about me" in user_text:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_learner_memory",
                    "args": {},
                    "id": "memory-call",
                    "type": "tool_call",
                }],
            )
        if "uploaded notes" in user_text:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_learning_materials",
                    "args": {"query": "passive voice", "limit": 3},
                    "id": "material-call",
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
        None,
        [{"role": "user", "content": "What is the difference between affect and effect?"}],
    ))
    weak = asyncio.run(run_agent(
        "book-1",
        None,
        [{"role": "user", "content": "What are my weak words?"}],
    ))
    due = asyncio.run(run_agent(
        "book-1",
        None,
        [{"role": "user", "content": "What should I review today?"}],
    ))
    memory = asyncio.run(run_agent(
        "book-1",
        "alpha",
        [{"role": "user", "content": "What do you remember about me?"}],
        memory={
            "preferences": {
                "response_style": {"value": "concise", "source": "explicit", "updated_at": "now"},
            },
            "goals": [],
            "recurring_confusions": [],
        },
    ))

    assert general == "general language answer"
    assert "get_weak_words" in weak and "alpha" in weak
    assert "get_due_words" in due and "alpha" in due
    assert "get_learner_memory" in memory and "concise" in memory


def test_explicit_memory_preferences_override_and_false_positives():
    state = {"user": {"cards": {"alpha": {"last_grade": 2}}}}
    fixed_now = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)

    assert apply_explicit_memory(state, "I prefer short explanations with examples.", fixed_now)["changed"]
    assert compact_preferences(state["memory"]) == {
        "response_style": "concise",
        "include_examples": True,
    }

    apply_explicit_memory(state, "I prefer detailed explanations.", fixed_now)
    assert compact_preferences(state["memory"])["response_style"] == "detailed"
    before = copy.deepcopy(state)
    assert apply_explicit_memory(state, "Short explanations are sometimes useful.", fixed_now) == {
        "changed": False,
        "action": "none",
    }
    assert state == before


def test_agent_routes_material_search_and_combines_it_with_learner_tools(monkeypatch, tmp_path):
    from app import materials

    state = {
        "book_id": "book-1",
        "user": {"cards": {"passive": {"last_grade": 2}}},
    }
    (tmp_path / "book-1.json").write_text(json.dumps(state), encoding="utf-8")
    material_dir = tmp_path / "materials"
    materials.add_material(
        "book-1",
        "grammar-notes.txt",
        b"The passive voice uses be plus a past participle.",
        materials_dir=material_dir,
    )
    monkeypatch.setattr(agent, "STATE_DIR", tmp_path)
    monkeypatch.setattr(materials, "MATERIALS_DIR", material_dir)
    monkeypatch.setattr(agent, "_model_with_tools", lambda: _RoutingModel())

    material_answer = asyncio.run(run_agent(
        "book-1",
        None,
        [{"role": "user", "content": "Use my uploaded notes to explain passive voice."}],
    ))
    combined_answer = asyncio.run(run_agent(
        "book-1",
        None,
        [{"role": "user", "content": "Combine learner and notes evidence."}],
    ))

    assert "search_learning_materials" in material_answer
    assert "grammar-notes.txt" in material_answer
    assert "get_weak_words" in combined_answer
    assert "search_learning_materials" in combined_answer


def test_explicit_memory_goal_confusion_upsert_and_compact_view():
    state = {}
    fixed_now = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
    apply_explicit_memory(state, "My goal is to pass B2 Danish.", fixed_now)
    apply_explicit_memory(state, "I always confuse affect and effect.", fixed_now)
    apply_explicit_memory(state, "I always confuse effect and affect.", fixed_now)

    view = learner_memory_view(state["memory"])
    assert view["goals"] == ["pass B2 Danish"]
    assert len(view["recurring_confusions"]) == 1
    assert set(view["recurring_confusions"][0]) == {"affect", "effect"}
    assert "cards" not in view


def test_memory_forget_and_reset_preserve_other_book_state():
    state = {
        "user": {"cards": {"alpha": {"last_grade": 2}}, "schedule": {"new_cursor": 3}},
        "cache": {"vocab": {"alpha": {"chat": [{"role": "user", "content": "hi"}]}}},
    }
    fixed_now = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
    apply_explicit_memory(state, "I prefer short answers.", fixed_now)
    apply_explicit_memory(state, "My goal is to pass B2 Danish.", fixed_now)

    assert apply_explicit_memory(state, "Forget my preferences.", fixed_now)["changed"]
    assert state["memory"]["preferences"] == {}
    preserved = {"user": copy.deepcopy(state["user"]), "cache": copy.deepcopy(state["cache"])}

    assert apply_explicit_memory(state, "Reset my memory.", fixed_now)["changed"]
    assert "memory" not in state
    assert state["user"] == preserved["user"]
    assert state["cache"] == preserved["cache"]


def test_memory_missing_empty_and_extraction_is_conservative():
    assert learner_memory_view(None) == {
        "preferences": {},
        "goals": [],
        "recurring_confusions": [],
    }
    assert extract_explicit_memory("Can you explain affect and effect?") == {"action": "none"}
    assert extract_explicit_memory("My goal is unclear right now.") == {"action": "none"}


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


def test_plain_chat_persists_and_resets_memory_without_touching_cards(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from app import chat, main

    state_path = tmp_path / "book-1.json"
    original_card = {"last_grade": 2, "reps": 1, "due_at": "2026-08-01T00:00:00Z"}
    state_path.write_text(json.dumps({
        "book_id": "book-1",
        "groups": {},
        "ungrouped": [],
        "user": {"cards": {"alpha": original_card}},
    }), encoding="utf-8")
    seen_memory = []

    async def memory_aware_agent(*args, **kwargs):
        seen_memory.append(copy.deepcopy(kwargs.get("memory")))
        return "ok"

    monkeypatch.setattr(agent, "run_agent", memory_aware_agent)
    monkeypatch.setattr(main, "_state_path", lambda book_id: state_path)

    asyncio.run(chat.vocab_chat(
        "book-1",
        "alpha",
        chat.ChatIn(message="I prefer short explanations with examples."),
    ))
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert compact_preferences(persisted["memory"]) == {
        "response_style": "concise",
        "include_examples": True,
    }
    assert compact_preferences(seen_memory[-1]) == compact_preferences(persisted["memory"])

    asyncio.run(chat.vocab_chat(
        "book-1",
        "alpha",
        chat.ChatIn(message="Reset my memory."),
    ))
    reset_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "memory" not in reset_state
    assert reset_state["user"]["cards"]["alpha"] == original_card


def test_global_chat_without_word_persists_history_and_preserves_word_chat(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from app import chat, main

    state_path = tmp_path / "book-1.json"
    old_word_chat = [
        {"role": "user", "content": "old word question"},
        {"role": "assistant", "content": "old word answer"},
    ]
    state_path.write_text(json.dumps({
        "book_id": "book-1",
        "groups": {},
        "ungrouped": [],
        "user": {"cards": {}},
        "cache": {"vocab": {"alpha": {"chat": old_word_chat}}},
    }), encoding="utf-8")
    calls = []

    async def global_agent(*args, **kwargs):
        calls.append(copy.deepcopy(kwargs))
        return f"global answer {len(calls)}"

    monkeypatch.setattr(agent, "run_agent", global_agent)
    monkeypatch.setattr(main, "_state_path", lambda book_id: state_path)

    first = asyncio.run(chat.global_chat(
        "book-1",
        chat.GlobalChatIn(message="What should I review today?"),
    ))
    second = asyncio.run(chat.global_chat(
        "book-1",
        chat.GlobalChatIn(message="Explain this focus.", active_word="alpha"),
    ))
    history = asyncio.run(chat.global_chat_history("book-1"))

    assert calls[0]["active_word"] is None
    assert calls[1]["active_word"] == "alpha"
    assert len(calls[1]["conversation"]) == 3
    assert first["messages"][-1]["content"] == "global answer 1"
    assert second["messages"] == history["messages"]
    assert len(history["messages"]) == 4
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["cache"]["vocab"]["alpha"]["chat"] == old_word_chat


def test_global_chat_reset_is_scoped_and_global_fallback_works(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from app import chat, main

    state_path = tmp_path / "book-1.json"
    state_path.write_text(json.dumps({
        "book_id": "book-1",
        "groups": {},
        "ungrouped": [],
        "user": {"cards": {"alpha": {"last_grade": 2}}},
        "cache": {
            "global_chat": [{"role": "user", "content": "old global message"}],
            "vocab": {"alpha": {"chat": [{"role": "user", "content": "keep me"}]}},
        },
    }), encoding="utf-8")

    async def broken_agent(*args, **kwargs):
        raise RuntimeError("global agent unavailable")

    async def legacy_llm(prompt):
        assert "Current word context: (none)" in prompt
        return "global legacy answer"

    monkeypatch.setattr(agent, "run_agent", broken_agent)
    monkeypatch.setattr(main, "_state_path", lambda book_id: state_path)
    monkeypatch.setattr(main, "call_llm", legacy_llm)

    fallback = asyncio.run(chat.global_chat(
        "book-1",
        chat.GlobalChatIn(message="What is the difference between affect and effect?"),
    ))
    assert fallback["messages"][-1]["content"] == "global legacy answer"
    assert "Agent failure; using legacy plain-chat fallback" in capsys.readouterr().out

    reset = asyncio.run(chat.global_chat(
        "book-1",
        chat.GlobalChatIn(force_new=True),
    ))
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert reset["messages"] == []
    assert persisted["cache"]["global_chat"] == []
    assert persisted["cache"]["vocab"]["alpha"]["chat"][0]["content"] == "keep me"
    assert persisted["user"]["cards"]["alpha"]["last_grade"] == 2
