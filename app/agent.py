import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, ToolRuntime, tools_condition


STATE_DIR = Path(__file__).resolve().parent.parent / "state"
MAX_TOOL_RESULTS = 20

SYSTEM_PROMPT = """You are a personalized language-learning assistant.
Use learner tools when the user asks about their progress, weak vocabulary, due reviews, or what they personally should focus on.
Answer general language questions directly when learner-specific evidence is unnecessary.
Treat tool results as the only learner-specific evidence: never claim to have read data you did not read.
Do not claim that you completed a review, score update, or quiz. Reply concisely in the user's language.
The current word is context only and does not limit what the user may ask about."""


@dataclass(frozen=True)
class AgentContext:
    book_id: str


def _bounded_limit(limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = 10
    return max(1, min(MAX_TOOL_RESULTS, value))


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_book_state(book_id: str, state_dir: Optional[Path] = None) -> Dict[str, Any]:
    base = state_dir or STATE_DIR
    path = base / f"{book_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"learner state not found for book_id={book_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid learner state for book_id={book_id}")
    return data


def select_weak_words(state: Mapping[str, Any], limit: int = 10) -> Dict[str, Any]:
    """Select weak words from persisted card evidence without modifying state."""
    cards = (state.get("user") or {}).get("cards") or {}
    if not isinstance(cards, Mapping):
        cards = {}

    matches: List[Dict[str, Any]] = []
    for word, raw_card in cards.items():
        if not isinstance(raw_card, Mapping):
            continue

        signals: Dict[str, Any] = {}
        reasons: List[str] = []

        last_grade = _as_number(raw_card.get("last_grade"))
        if last_grade is not None:
            signals["last_grade"] = int(last_grade) if last_grade.is_integer() else last_grade
            if last_grade < 4:
                reasons.append("low_last_grade")

        last_quiz_score = _as_number(raw_card.get("last_quiz_score"))
        if last_quiz_score is not None:
            signals["last_quiz_score"] = last_quiz_score
            if last_quiz_score < 0.7:
                reasons.append("low_quiz_score")

        last_quiz_grade = _as_number(raw_card.get("last_quiz_grade"))
        if last_quiz_grade is not None:
            signals["last_quiz_grade"] = (
                int(last_quiz_grade) if last_quiz_grade.is_integer() else last_quiz_grade
            )
            if last_quiz_grade < 4 and "low_quiz_score" not in reasons:
                reasons.append("low_quiz_grade")

        wrong_count = _as_number(raw_card.get("quiz_wrong_count"))
        if wrong_count is not None and wrong_count > 0:
            signals["quiz_wrong_count"] = int(wrong_count)

        if not reasons:
            continue

        matches.append({
            "word": str(word),
            "signals": signals,
            "reason": reasons,
            "_rank": (
                last_quiz_score if last_quiz_score is not None else 1.0,
                last_grade if last_grade is not None else 6.0,
                -(wrong_count or 0),
                str(word).lower(),
            ),
        })

    matches.sort(key=lambda item: item["_rank"])
    total_matches = len(matches)
    items = matches[:_bounded_limit(limit)]
    for item in items:
        item.pop("_rank", None)
    return {"total_matches": total_matches, "items": items}


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def select_due_words(
    state: Mapping[str, Any],
    limit: int = 10,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Select already-studied cards that are due, without distributing new cards."""
    cards = (state.get("user") or {}).get("cards") or {}
    if not isinstance(cards, Mapping):
        cards = {}

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)

    due: List[Dict[str, Any]] = []
    invalid_due_at_count = 0
    for word, raw_card in cards.items():
        if not isinstance(raw_card, Mapping):
            continue

        # review_today creates blank cards eagerly. A non-null grade or reps > 0
        # distinguishes an actual learning record from an untouched card.
        reps = int(_as_number(raw_card.get("reps")) or 0)
        last_grade = _as_number(raw_card.get("last_grade"))
        if reps <= 0 and last_grade is None:
            continue

        due_at = _parse_datetime(raw_card.get("due_at"))
        if due_at is None:
            if raw_card.get("due_at") not in (None, ""):
                invalid_due_at_count += 1
            continue
        if due_at > current:
            continue

        signals: Dict[str, Any] = {
            "due_at": due_at.isoformat(),
            "reps": reps,
            "days_overdue": max(0, (current - due_at).days),
        }
        if last_grade is not None:
            signals["last_grade"] = int(last_grade) if last_grade.is_integer() else last_grade
        due.append({
            "word": str(word),
            "signals": signals,
            "reason": ["review_due"],
            "_due_at": due_at,
        })

    due.sort(key=lambda item: (item["_due_at"], item["word"].lower()))
    total_matches = len(due)
    items = due[:_bounded_limit(limit)]
    for item in items:
        item.pop("_due_at", None)
    return {
        "as_of": current.isoformat(),
        "total_matches": total_matches,
        "invalid_due_at_count": invalid_due_at_count,
        "items": items,
    }


@tool
def get_weak_words(
    runtime: ToolRuntime[AgentContext, MessagesState],
    limit: int = 10,
) -> Dict[str, Any]:
    """Get this learner's weak vocabulary using persisted grades and quiz evidence."""
    state = _read_book_state(runtime.context.book_id)
    return select_weak_words(state, limit=limit)


@tool
def get_due_words(
    runtime: ToolRuntime[AgentContext, MessagesState],
    limit: int = 10,
) -> Dict[str, Any]:
    """Get this learner's already-studied vocabulary whose review time has arrived."""
    state = _read_book_state(runtime.context.book_id)
    return select_due_words(state, limit=limit)


TOOLS = [get_weak_words, get_due_words]


@lru_cache(maxsize=1)
def _model_with_tools():
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
    )
    return model.bind_tools(TOOLS)


async def _agent_node(state: MessagesState) -> Dict[str, Any]:
    response = await _model_with_tools().ainvoke(state["messages"])
    return {"messages": [response]}


@lru_cache(maxsize=1)
def _agent_graph():
    builder = StateGraph(MessagesState, context_schema=AgentContext)
    builder.add_node("agent", _agent_node)
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=False))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


def _message_content_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content or "").strip()


async def run_agent(
    book_id: str,
    active_word: str,
    conversation: List[Dict[str, str]],
) -> str:
    messages = [SystemMessage(content=f"{SYSTEM_PROMPT}\nCurrent word context: {active_word or '(none)' }.")]
    for item in conversation:
        role = item.get("role")
        content = str(item.get("content") or "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "user":
            messages.append(HumanMessage(content=content))

    result = await _agent_graph().ainvoke(
        {"messages": messages},
        context=AgentContext(book_id=book_id),
        config={"recursion_limit": 6},
    )
    final_message = result["messages"][-1]
    if not isinstance(final_message, AIMessage):
        raise RuntimeError("Agent did not return an AI message")
    text = _message_content_text(final_message)
    if not text:
        raise RuntimeError("Agent returned an empty response")
    return text
