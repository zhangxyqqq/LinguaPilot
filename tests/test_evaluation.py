import asyncio
import json

from app.evaluation import (
    render_summary,
    run_deterministic_evaluations,
    run_live_evaluations,
    write_reports,
)


def test_deterministic_agent_evaluation_covers_required_behaviors():
    report = asyncio.run(run_deterministic_evaluations())

    assert report.success
    assert report.total == 8
    assert {case.case_id for case in report.cases} == {
        "general_tool_avoidance",
        "weak_tool_routing",
        "due_tool_routing",
        "learner_memory_usage",
        "rag_routing_and_source",
        "user_isolation",
        "book_isolation",
        "graceful_fallback",
    }


def test_evaluation_writes_machine_and_human_readable_reports(tmp_path):
    report = asyncio.run(run_deterministic_evaluations())
    json_path = tmp_path / "result.json"
    summary_path = tmp_path / "summary.md"

    write_reports(report, json_path, summary_path)

    machine_result = json.loads(json_path.read_text(encoding="utf-8"))
    assert machine_result["status"] == "passed"
    assert machine_result["passed"] == machine_result["total"] == 8
    assert "Agent evaluation: deterministic" in summary_path.read_text(encoding="utf-8")
    assert "8/8 passed" in render_summary(report)


def test_live_evaluation_is_explicitly_optional_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    report = asyncio.run(run_live_evaluations())

    assert report.status == "skipped"
    assert report.total == 0
    assert "OPENAI_API_KEY" in (report.skipped_reason or "")
