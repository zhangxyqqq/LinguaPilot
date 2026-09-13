"""Local, scoped retrieval primitives. No provider calls or application-state writes."""
from __future__ import annotations

import hashlib
import math
import logging
from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, Sequence

from .retrieval_baseline import _tokens as tokens


def chunk_key(chunk):
    return (str(chunk['source_name']), str(chunk['document_id']), int(chunk['chunk_index']))


def text_digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalized_vector(vector):
    values = tuple(float(v) for v in vector)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('invalid embedding values')
    norm = math.sqrt(sum(v * v for v in values))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('invalid embedding norm')
    return tuple(v / norm for v in values)


class BM25Index:
    """Okapi BM25 using corpus-wide length normalization and positive RSJ IDF."""
    def __init__(self, chunks, k1=1.5, b=0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError('invalid BM25 parameters')
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.terms = [Counter(tokens(c['text'])) for c in chunks]
        self.lengths = [sum(t.values()) for t in self.terms]
        self.average = sum(self.lengths) / max(1, len(chunks)) or 1
        self.df = Counter(t for doc in self.terms for t in doc)

    def search(self, query, depth):
        scores = {}
        for i, terms in enumerate(self.terms):
            score = 0.0
            for term in sorted(set(tokens(query))):
                tf = terms[term]
                if tf:
                    idf = math.log1p((len(self.chunks) - self.df[term] + .5) / (self.df[term] + .5))
                    score += idf * tf * (self.k1 + 1) / (
                        tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.average))
            if score > 0:
                scores[i] = score
        return ranked(scores, self.chunks, depth)


class ExactVectorIndex:
    """Derived normalized vectors mapped to chunk positions in ONE scoped snapshot.

    Cached raw vectors remain in SQLite. Reconstructing this view cannot leave an
    orphan sidecar or accidentally reuse an index for a different user/book.
    """
    def __init__(self, chunks, model_id):
        self.chunks = list(chunks)
        self.entries = []
        dimension = None
        for i, chunk in enumerate(chunks):
            if chunk.get('embedding_model') != model_id:
                continue
            if chunk.get('embedding_text_sha256') not in (None, text_digest(chunk['text'])):
                raise ValueError('stale embedding text digest')
            if not isinstance(chunk.get('embedding'), list):
                continue
            vector = normalized_vector(chunk['embedding'])
            if dimension is not None and len(vector) != dimension:
                raise ValueError('inconsistent embedding dimensions')
            dimension = len(vector)
            self.entries.append((i, vector))
        self.dimension = dimension

    def search(self, query_vector, depth):
        query = normalized_vector(query_vector)
        if self.dimension is not None and len(query) != self.dimension:
            raise ValueError('query embedding dimension mismatch')
        scores = {}
        for i, vector in self.entries:
            score = max(-1., min(1., sum(a * b for a, b in zip(query, vector))))
            if score >= .2:
                scores[i] = score
        return ranked(scores, self.chunks, depth)


def ranked(scores, chunks, depth):
    return sorted(scores.items(), key=lambda pair: (-pair[1], chunk_key(chunks[pair[0]])))[:depth]


def reciprocal_rank_fusion(branches, constant=60):
    if constant <= 0:
        raise ValueError('RRF constant must be positive')
    scores = Counter()
    for branch in branches:
        seen = set()
        for rank, (index, _) in enumerate(branch, 1):
            if index not in seen:
                scores[index] += 1 / (constant + rank)
                seen.add(index)
    return dict(scores)


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> Sequence[float]: ...


class CoverageReranker:
    """Untrained lexical coverage + adjacent query-bigram matching; no semantic claim."""
    def score(self, query, texts):
        q = tokens(query)
        terms = set(q)
        pairs = set(zip(q, q[1:]))
        scores = []
        for text in texts:
            words = tokens(text)
            coverage = len(terms & set(words)) / max(1, len(terms))
            phrase = len(pairs & set(zip(words, words[1:]))) / max(1, len(pairs))
            scores.append(coverage + .25 * phrase)
        return scores


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = 'legacy'
    candidate_depth: int = 12
    pool_size: int = 12
    rerank: bool = False

    def __post_init__(self):
        if self.mode not in {'legacy', 'bm25', 'dense', 'hybrid'}:
            raise ValueError('unknown retrieval mode')
        if not 8 <= self.candidate_depth <= 100 or not 8 <= self.pool_size <= 100:
            raise ValueError('candidate bounds must be between 8 and 100')
        if self.mode == 'legacy' and self.rerank:
            raise ValueError('legacy baseline does not support reranking')


# Updated only through the documented lexical promotion gate.
DEFAULT_CONFIG = RetrievalConfig(mode='legacy')


def retrieve(store, query, limit, config, query_embedding=None, embedding_model=None,
             reranker=None, trace=None):
    """Return bounded evidence; opt-in trace receives diagnostics, never the LLM."""
    start = perf_counter()
    chunks = list(store.get('chunks') or [])
    limit = max(1, min(8, int(limit or 5)))
    stages = {}
    errors = []
    lexical, dense = [], []
    t = perf_counter()
    if config.mode in {'bm25', 'hybrid'}:
        lexical = BM25Index(chunks).search(query, config.candidate_depth)
    stages['lexical_ms'] = (perf_counter() - t) * 1000
    t = perf_counter()
    if config.mode in {'dense', 'hybrid'} and query_embedding is not None and embedding_model:
        try:
            index = ExactVectorIndex(chunks, embedding_model)
            if chunks and not index.entries:
                raise ValueError('vector index is empty')
            dense = index.search(query_embedding, config.candidate_depth)
        except (ValueError, TypeError, OverflowError) as exc:
            errors.append('dense:' + type(exc).__name__)
    if config.mode == 'hybrid' and (query_embedding is None or not embedding_model):
        errors.append('dense_unavailable:bm25_fallback')
    if config.mode == 'dense' and (query_embedding is None or not embedding_model or errors):
        lexical = BM25Index(chunks).search(query, config.candidate_depth)
        errors.append('dense_unavailable:bm25_fallback')
    stages['dense_ms'] = (perf_counter() - t) * 1000
    t = perf_counter()
    fused = reciprocal_rank_fusion([lexical, dense]) if config.mode == 'hybrid' else dict(dense or lexical)
    pool = ranked(fused, chunks, config.pool_size)
    stages['fusion_ms'] = (perf_counter() - t) * 1000
    fused_ranks = {i: rank for rank, (i, _) in enumerate(pool, 1)}
    rerank_scores = {}
    used = False
    t = perf_counter()
    if config.rerank and pool:
        try:
            scores = list((reranker or CoverageReranker()).score(query, [chunks[i]['text'] for i, _ in pool]))
            if len(scores) != len(pool) or not all(math.isfinite(float(s)) for s in scores):
                raise ValueError('invalid reranker scores')
            rerank_scores = {i: float(score) for (i, _), score in zip(pool, scores)}
            pool.sort(key=lambda pair: (-rerank_scores[pair[0]], fused_ranks[pair[0]]))
            used = True
        except Exception as exc:
            rerank_scores = {}
            errors.append('reranker:' + type(exc).__name__)
    stages['rerank_ms'] = (perf_counter() - t) * 1000
    lex = {i: (r, s) for r, (i, s) in enumerate(lexical, 1)}
    sem = {i: (r, s) for r, (i, s) in enumerate(dense, 1)}
    diagnostics, items = [], []
    for rank, (i, score) in enumerate(pool, 1):
        c = chunks[i]
        item = {k: c[k] for k in ('document_id', 'source_name', 'chunk_index', 'text')}
        item.update(score=round(rerank_scores.get(i, score), 6),
                    lexical_score=round(lex.get(i, (None, 0))[1], 6),
                    semantic_score=round(sem.get(i, (None, 0))[1], 6),
                    match='hybrid' if i in lex and i in sem else 'semantic' if i in sem else 'lexical')
        if rank <= limit:
            items.append(item)
        diagnostics.append({k: item[k] for k in ('document_id', 'source_name', 'chunk_index')} | {
            'lexical_rank': lex.get(i, (None,))[0], 'lexical_score': item['lexical_score'],
            'dense_rank': sem.get(i, (None,))[0], 'dense_score': item['semantic_score'],
            'fused_rank': fused_ranks[i], 'fused_score': score,
            'reranker_score': rerank_scores.get(i), 'final_rank': rank if rank <= limit else None})
    for error in errors:
        logging.getLogger(__name__).warning('Retrieval degraded: %s', error)
    stages['total_ms'] = (perf_counter() - start) * 1000
    if trace is not None:
        trace.update(mode=config.mode, query=query, candidate_counts={'lexical':len(lexical), 'dense':len(dense), 'pool':len(pool)},
                     reranker_used=used, errors=errors, stages=stages, candidates=diagnostics)
    return {'total_documents': len(store.get('documents') or []),
            'retrieval': {'strategy': config.mode, 'reranker_used': used, 'fallbacks': errors}, 'items': items}
