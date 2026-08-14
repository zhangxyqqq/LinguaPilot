"""Deterministic and optional live-model evaluations for the LangGraph Agent.

The deterministic suite executes the real graph and real tools against a
temporary SQLite database. Only routing decisions are simulated, so grading is
repeatable and does not require an API key. The live suite uses the configured
model against the same synthetic fixtures and remains explicitly optional.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .agent import run_agent_detailed
from .chat import _plain_agent_response
from .storage import configure_storage


EVAL_USER = "eval-alice"
EVAL_BOOK = "eval-book"
EVAL_MEMORY = {
    "version": 1,
    "updated_at": "2026-08-14T12:00:00+00:00",
    "preferences": {
        "response_style": {
            "value": "concise",
            "source": "explicit",
            "updated_at": "2026-08-14T12:00:00+00:00",
        }
    },
    "goals": [],
    "recurring_confusions": [],
}


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    passed: bool
    expected_tools: tuple[str, ...] = ()
    observed_tools: tuple[str, ...] = ()
    checks: Mapping[str, bool] = field(default_factory=dict)
    answer_excerpt: str = ""


@dataclass(frozen=True)
class EvaluationReport:
    suite: str
    status: str
    cases: tuple[EvaluationCase, ...] = ()
    skipped_reason: Optional[str] = None

    @property
    def passed(self) -> int:
        return sum(case.passed for case in self.cases)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def success(self) -> bool:
        return self.status == "passed" and self.passed == self.total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite,
            "status": self.status,
            "passed": self.passed,
            "total": self.total,
            "skipped_reason": self.skipped_reason,
            "cases": [asdict(case) for case in self.cases],
        }


class DeterministicRoutingModel:
    """A fixed router that exercises the production graph without an LLM judge."""

    async def ainvoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            evidence = [message for message in messages if isinstance(message, ToolMessage)]
            names = ", ".join(message.name for message in evidence)
            normalized_contents = []
            for message in evidence:
                try:
                    payload = json.loads(str(message.content))
                except json.JSONDecodeError:
                    normalized_contents.append(str(message.content))
                    continue
                if message.name == "get_due_words" and isinstance(payload, dict):
                    payload.pop("as_of", None)
                    for item in payload.get("items") or []:
                        if isinstance(item, dict) and isinstance(item.get("signals"), dict):
                            item["signals"].pop("days_overdue", None)
                normalized_contents.append(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True)
                )
            contents = " ".join(normalized_contents)
            return AIMessage(content=f"Evidence from {names}: {contents}")

        user_text = next(
            str(message.content)
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        ).casefold()
        if "weak" in user_text or "focus on" in user_text:
            return self._tool_call("get_weak_words", {"limit": 5}, "weak-eval")
        if "review today" in user_text:
            return self._tool_call("get_due_words", {"limit": 5}, "due-eval")
        if "preference" in user_text or "remember about me" in user_text:
            return self._tool_call("get_learner_memory", {}, "memory-eval")
        if "uploaded material" in user_text or "uploaded notes" in user_text:
            return self._tool_call(
                "search_learning_materials",
                {"query": "passive voice", "limit": 3},
                "rag-eval",
            )
        return AIMessage(content="Affect is usually a verb; effect is usually a noun.")

    @staticmethod
    def _tool_call(name: str, args: Mapping[str, Any], call_id: str) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{
                "name": name,
                "args": dict(args),
                "id": call_id,
                "type": "tool_call",
            }],
        )


def _learner_state(book_id: str, word: str, grade: int) -> Dict[str, Any]:
    return {
        "book_id": book_id,
        "groups": {},
        "ungrouped": [word],
        "user": {
            "cards": {
                word: {
                    "reps": 1,
                    "last_grade": grade,
                    "due_at": "2020-01-01T00:00:00+00:00",
                }
            }
        },
    }


def _seed_evaluation_storage(db_path: Path) -> None:
    storage = configure_storage(db_path)
    storage.save_book_state(_learner_state(EVAL_BOOK, "weak-alpha", 2), EVAL_USER)
    storage.save_book_state(_learner_state(EVAL_BOOK, "bob-only", 1), "eval-bob")
    storage.save_book_state(_learner_state("eval-other-book", "other-only", 2), EVAL_USER)
    storage.save_material_store(
        EVAL_BOOK,
        {
            "version": 2,
            "book_id": EVAL_BOOK,
            "documents": [{
                "document_id": "grammar-notes",
                "source_name": "grammar-notes.txt",
                "uploaded_at": "2026-08-14T12:00:00+00:00",
                "char_count": 82,
                "chunk_count": 1,
            }],
            "chunks": [{
                "document_id": "grammar-notes",
                "source_name": "grammar-notes.txt",
                "chunk_index": 0,
                "text": "Passive voice uses a form of be plus a past participle.",
            }],
        },
        EVAL_USER,
    )


def _case(
    case_id: str,
    expected_tools: Sequence[str],
    observed_tools: Sequence[str],
    answer: str,
    checks: Mapping[str, bool],
) -> EvaluationCase:
    all_checks = {
        "tool_selection": tuple(observed_tools) == tuple(expected_tools),
        **checks,
    }
    return EvaluationCase(
        case_id=case_id,
        passed=all(all_checks.values()),
        expected_tools=tuple(expected_tools),
        observed_tools=tuple(observed_tools),
        checks=all_checks,
        answer_excerpt=answer[:300],
    )


async def _run_agent_case(
    case_id: str,
    prompt: str,
    expected_tools: Sequence[str],
    content_checks: Mapping[str, str],
    *,
    model: Optional[Any],
    user_id: str = EVAL_USER,
    book_id: str = EVAL_BOOK,
    memory: Optional[Mapping[str, Any]] = None,
) -> EvaluationCase:
    result = await run_agent_detailed(
        book_id=book_id,
        active_word=None,
        conversation=[{"role": "user", "content": prompt}],
        memory=memory,
        user_id=user_id,
        model=model,
    )
    answer_lower = result.answer.casefold()
    checks = {
        name: expected.casefold() in answer_lower
        for name, expected in content_checks.items()
    }
    return _case(case_id, expected_tools, result.tool_calls, result.answer, checks)


async def _fallback_case() -> EvaluationCase:
    async def broken_agent(**_kwargs):
        raise RuntimeError("synthetic Agent failure")

    async def legacy_llm(_prompt: str) -> str:
        return "legacy fallback answer"

    answer = await _plain_agent_response(
        book_id=EVAL_BOOK,
        active_word=None,
        conversation=[{"role": "user", "content": "Hello"}],
        memory=None,
        legacy_llm=legacy_llm,
        agent_runner=broken_agent,
    )
    return _case(
        "graceful_fallback",
        (),
        (),
        answer,
        {"legacy_answer_returned": answer == "legacy fallback answer"},
    )


async def _run_suite(model: Optional[Any], suite: str) -> EvaluationReport:
    cases: List[EvaluationCase] = []
    cases.append(await _run_agent_case(
        "general_tool_avoidance",
        "What is the difference between affect and effect?",
        (),
        {"mentions_affect": "affect", "mentions_effect": "effect"},
        model=model,
    ))
    cases.append(await _run_agent_case(
        "weak_tool_routing",
        "What are my weak words?",
        ("get_weak_words",),
        {"learner_evidence": "weak-alpha"},
        model=model,
    ))
    cases.append(await _run_agent_case(
        "due_tool_routing",
        "What should I review today?",
        ("get_due_words",),
        {"learner_evidence": "weak-alpha"},
        model=model,
    ))
    cases.append(await _run_agent_case(
        "learner_memory_usage",
        "What do you remember about my learning preferences?",
        ("get_learner_memory",),
        {"saved_preference": "concise"},
        model=model,
        memory=EVAL_MEMORY,
    ))
    cases.append(await _run_agent_case(
        "rag_routing_and_source",
        "Use my uploaded material to explain passive voice.",
        ("search_learning_materials",),
        {"retrieved_content": "passive voice", "source_attribution": "grammar-notes.txt"},
        model=model,
    ))
    cases.append(await _run_agent_case(
        "user_isolation",
        "What are my weak words?",
        ("get_weak_words",),
        {"own_evidence": "bob-only", "foreign_evidence_absent": ""},
        model=model,
        user_id="eval-bob",
    ))
    # An empty expected substring would always pass, so enforce the negative
    # isolation assertion explicitly after the common grader runs.
    user_case = cases[-1]
    user_checks = dict(user_case.checks)
    user_checks["foreign_evidence_absent"] = "weak-alpha" not in user_case.answer_excerpt.casefold()
    cases[-1] = EvaluationCase(
        case_id=user_case.case_id,
        passed=all(user_checks.values()),
        expected_tools=user_case.expected_tools,
        observed_tools=user_case.observed_tools,
        checks=user_checks,
        answer_excerpt=user_case.answer_excerpt,
    )
    cases.append(await _run_agent_case(
        "book_isolation",
        "What are my weak words?",
        ("get_weak_words",),
        {"own_evidence": "other-only"},
        model=model,
        book_id="eval-other-book",
    ))
    book_case = cases[-1]
    book_checks = dict(book_case.checks)
    book_checks["foreign_evidence_absent"] = "weak-alpha" not in book_case.answer_excerpt.casefold()
    cases[-1] = EvaluationCase(
        case_id=book_case.case_id,
        passed=all(book_checks.values()),
        expected_tools=book_case.expected_tools,
        observed_tools=book_case.observed_tools,
        checks=book_checks,
        answer_excerpt=book_case.answer_excerpt,
    )
    cases.append(await _fallback_case())
    status = "passed" if all(case.passed for case in cases) else "failed"
    return EvaluationReport(suite=suite, status=status, cases=tuple(cases))


async def run_deterministic_evaluations() -> EvaluationReport:
    with tempfile.TemporaryDirectory(prefix="langbuddy-eval-") as directory:
        _seed_evaluation_storage(Path(directory) / "evaluation.sqlite3")
        previous_embeddings = os.environ.get("LANGBUDDY_EMBEDDINGS")
        os.environ["LANGBUDDY_EMBEDDINGS"] = "disabled"
        try:
            return await _run_suite(DeterministicRoutingModel(), "deterministic")
        finally:
            if previous_embeddings is None:
                os.environ.pop("LANGBUDDY_EMBEDDINGS", None)
            else:
                os.environ["LANGBUDDY_EMBEDDINGS"] = previous_embeddings


async def run_live_evaluations() -> EvaluationReport:
    if not os.getenv("OPENAI_API_KEY"):
        return EvaluationReport(
            suite="live-model",
            status="skipped",
            skipped_reason="OPENAI_API_KEY is not configured",
        )
    with tempfile.TemporaryDirectory(prefix="langbuddy-live-eval-") as directory:
        _seed_evaluation_storage(Path(directory) / "evaluation.sqlite3")
        return await _run_suite(None, "live-model")


def render_summary(report: EvaluationReport) -> str:
    lines = [
        f"# Agent evaluation: {report.suite}",
        "",
        f"Status: **{report.status.upper()}** ({report.passed}/{report.total} passed)",
    ]
    if report.skipped_reason:
        lines.extend(["", f"Reason: {report.skipped_reason}"])
    if report.cases:
        lines.extend(["", "| Case | Result | Expected tools | Observed tools |", "|---|---|---|---|"])
        for case in report.cases:
            expected = ", ".join(case.expected_tools) or "none"
            observed = ", ".join(case.observed_tools) or "none"
            result = "PASS" if case.passed else "FAIL"
            lines.append(f"| {case.case_id} | {result} | {expected} | {observed} |")
    return "\n".join(lines) + "\n"


def write_reports(
    report: EvaluationReport,
    json_output: Optional[Path],
    summary_output: Optional[Path],
) -> None:
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if summary_output:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(render_summary(report), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the LangBuddy Agent")
    parser.add_argument("--mode", choices=("deterministic", "live"), default="deterministic")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    args = _parser().parse_args(list(argv) if argv is not None else None)
    report = asyncio.run(
        run_live_evaluations() if args.mode == "live" else run_deterministic_evaluations()
    )
    write_reports(report, args.json_output, args.summary_output)
    print(render_summary(report), end="")
    return 0 if report.status in {"passed", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
