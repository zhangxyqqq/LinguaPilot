# Job-search RAG evidence summary

LinguaPilot is an AI-assisted language-learning application, not a generic RAG
platform. The demonstrated engineering story is measured retrieval choices with
negative results and a conservative promotion gate.

## Demonstrated capabilities and actual technologies

- PDF/TXT/Markdown/CSV extraction; bounded overlapping or paragraph chunks.
- Python BM25, normalized exact cosine vector index, cached embeddings in SQLite,
  RRF candidate fusion, and bounded deterministic lexical reranking.
- OpenAI embedding adapter remains optional; no live model calls were made for
  this milestone. No FAISS, Qdrant, cross-encoder or external vector service.
- Python, FastAPI, SQLite, LangGraph, LangChain/OpenAI adapters, pypdf, pytest;
  existing vanilla JavaScript frontend and Dockerfile remain in place.
- Runtime user/book isolation, four learner tools, persistent memory, deterministic
  SM-2 and quiz updates are preserved. Model output never writes learner state.

## Benchmark and exact measured comparison

12 repository-owned synthetic lessons; 24 authored queries with grade-2 direct
and grade-1 supporting source judgments. Returned chunks must contain an exact
judgment anchor. Duplicate sources get no repeated credit. All queries are used
in every configuration. Top-k=3, modern candidate depth=12/branch, fused pool=12,
RRF constant=60, BM25 k1=1.5/b=0.75. Frozen baseline scores all chunks. Dense
experiments use 256-dimensional deterministic signed token hashes; they are not
learned semantic retrieval. Seven timed repetitions per query follow one warmup.

| Configuration | Recall@3 | Hit@3 | MRR@3 | nDCG@3 | median ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 0.875000 | 0.916667 | 0.916667 | 0.902186 | 0.244 | 0.268 |
| legacy_weighted | 0.895833 | 0.916667 | 0.916667 | 0.907924 | 0.590 | 0.637 |
| bm25 | 0.875000 | 0.916667 | 0.916667 | 0.902186 | 0.268 | 0.301 |
| dense | 0.770833 | 0.833333 | 0.833333 | 0.811613 | 0.402 | 0.436 |
| hybrid | 0.895833 | 0.916667 | 0.916667 | 0.907924 | 0.666 | 0.716 |
| hybrid_rerank | 0.895833 | 0.916667 | 0.916667 | 0.907924 | 0.812 | 0.947 |
| bm25_paragraph | 0.875000 | 0.916667 | 0.916667 | 0.900684 | 0.298 | 0.341 |

Full floating-point metrics, per-query rankings and timing environment are in
`evaluation_results/retrieval_v1.json`. Reranking added local median latency
without changing any aggregate quality metric. Paragraph chunks increased the
index from 12 to 24 chunks and reduced nDCG from 0.902186 to 0.900684.

## Default, negative results and trade-offs

No candidate passed the predeclared promotion gate. Default remains `legacy`,
900/120 baseline chunks, reranking disabled. Legacy still optionally uses its old
0.45/0.55 OpenAI weighted fusion when embeddings are enabled; live OpenAI quality
was not measured here. BM25 tied lexical baseline on all aggregate quality metrics.
RRF and frozen weighted fusion tied when using fake vectors. Their nDCG gain over
lexical was only 0.005738, below the 0.02 gate and ineligible for semantic promotion.
Fake dense-only retrieval underperformed lexical and failed synonym probes.

The exact vector scan has transparent source mapping and no second persisted
index to synchronize; it is O(chunks × dimensions). SQLite compare-and-swap
prevents embedding cache writes from resurrecting deleted material. Replacement
requires original bytes again and retains the material ID, not old chunk semantics.
The local coverage/bigram reranker is cheap and reproducible but untrained.

## Limitations to state accurately

This is a small authored development benchmark with no held-out set, independent
annotation, production traffic, significance claim, or live embedding measurement.
Hash fixtures establish deterministic plumbing, not semantic understanding.
Short lessons do not establish optimal chunk sizes, large-PDF behavior or scale.
PDF extraction has no OCR or page-coordinate attribution. User headers implement
local identity/isolation, not authentication. No new infrastructure was added.
The Dockerfile is preserved, but Docker runtime validation was unavailable because
the daemon was not running.

## CV-safe factual statements

- Built and evaluated BM25, local vector search, RRF fusion and bounded reranking
  inside a LangGraph language-learning assistant with FastAPI and SQLite.
- Authored a reproducible 24-query retrieval benchmark with source-grounded graded
  judgments, Recall@3, MRR@3 and nDCG@3, separate from agent-routing evaluation.
- Used a documented promotion gate and retained the simpler default when BM25
  tied the baseline and reranking failed to improve retrieval quality.
- Added deterministic regressions for scoped retrieval, embedding invalidation,
  source mapping and concurrent deletion; 72 tests pass without paid model calls.

Do not claim “improved semantic accuracy,” “production-scale vector database,”
“cross-encoder reranking,” or measured user-learning gains from this work.

## Interview drill-down areas

BM25 normalization versus the original query-subspace scorer; cosine/vector
mapping; candidate recall and RRF; bounded reranking; graded gain/discount and
duplicate handling; benchmark bias; promotion thresholds; scoped storage; cache
hash/model invalidation; concurrent mutation control; graceful degradation.
See `docs/INTERVIEW_RAG_MAP.md` for implementation pointers.
