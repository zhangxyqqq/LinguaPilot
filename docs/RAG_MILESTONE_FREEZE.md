# RAG milestone freeze

Status: **source scope frozen for the job-search milestone**, 2026-09-13.

Acceptance evidence:

- All 33 original regressions preserved; implementation-freeze `pytest -q`: **72 passed in 1.17s**.
- Full benchmark rankings/metrics match across separate processes with
  `PYTHONHASHSEED=1` and `777`; compileall and `git diff --check` pass.
- Independent deterministic agent evaluation: 8/8 passed.
- Seven retrieval/chunking configurations compared across all 24 queries.
- Original scorer and pre-upgrade outputs preserved with SHA-256 fingerprints.
- BM25, exact vector mapping, RRF, optional bounded reranking, source attribution,
  index lifecycle, fallback and user/book isolation covered by tests.
- Concurrent cache deletion and competing material upload regressions pass.
- No paid model calls; external HTTP transports blocked in the test suite.
- Default remains legacy / baseline chunks / reranker disabled by the documented
  promotion rule. Negative results retained; no semantic-quality gain claimed.
- Learning state, single-agent topology, FastAPI routes, SQLite persistence,
  lightweight frontend, requirements and Dockerfile preserved.
- README, design notes, evidence summary and interview map reflect measured data.

Validation boundary: local Docker daemon was unavailable, so Docker build/run
could not be revalidated. Configuration and dependencies were not changed.
No claim of container execution, live embedding quality or production performance
is included in this freeze. No further feature scope is proposed.

## Pre-commit hygiene review

All 21 modified/untracked files were reviewed and should be committed:

| Classification | Files |
|---|---|
| Source code | `app/agent.py`, `app/materials.py`, `app/storage.py`, `app/retrieval.py`, `app/retrieval_baseline.py`, `app/retrieval_evaluation.py` |
| Deterministic tests | `tests/conftest.py`, `tests/test_retrieval_pipeline.py` |
| Benchmark fixtures/protocol | `benchmarks/retrieval/corpus.json`, `benchmarks/retrieval/PROTOCOL.md` |
| Configuration example | `.env.example` (empty API key, portable relative database path) |
| Documentation | `README.md`, `docs/RETRIEVAL_DESIGN.md`, `docs/JOB_SEARCH_RAG_SUMMARY.md`, `docs/INTERVIEW_RAG_MAP.md`, `docs/RAG_MILESTONE_FREEZE.md` |
| Reproducible evidence | `evaluation_results/retrieval_pre_upgrade.json`, `evaluation_results/retrieval_v1.json`, `evaluation_results/retrieval_v1.md`, `evaluation_results/agent_rag_milestone.json`, `evaluation_results/agent_rag_milestone.md` |

Retain the generated evidence deliberately: the pre-upgrade results anchor the
frozen scorer; comparison JSON contains all per-query results and configuration
settings; Markdown supports human review; agent reports verify the current tool
output contract. They contain synthetic fixtures, not learner data. Python/OS/
architecture metadata provides timing context without private filesystem paths,
usernames or hostnames. Historical reports and timing observations are unchanged.

Exclude the already-ignored `.env`, `state/`, imported `data/book_????????.csv`,
`.venv/`, `.idea/`, `.pytest_cache/`, and Python bytecode caches. These are private
configuration, learner/runtime data or local artifacts; leave them on disk.
No ignored files are already tracked. `.gitignore` needs no changes.

Hygiene changes are documentation-only: README now sends verification reports
to temporary directories instead of overwriting retained evidence, and spells
out the three alternative retrieval-mode values separately.

Validation: 72 deterministic tests passed; agent evaluation passed 8/8 and
reproduced both retained reports exactly. All seven retrieval configurations
reproduced all 24 per-query rankings/scores, aggregate quality metrics, settings,
and the no-promotion decision. Fresh latency observations are not golden values.
README and job-search tables match the retained JSON/Markdown exactly. All
pre-upgrade outputs and fingerprints verified; temporary rerun reports removed.
No application source, test, fixture, evidence file or architecture changed in
this hygiene pass. No staging or commit performed.
