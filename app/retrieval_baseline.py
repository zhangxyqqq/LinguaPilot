"""Frozen retrieval functions from 381fbc56e92e31b7f78ecf1043a29a7c179d5a4a.
Do not edit: historical query-subspace TF-IDF / weighted-fusion baseline.
"""
import math
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence
MAX_SEARCH_RESULTS = 8
LEXICAL_WEIGHT = 0.45
SEMANTIC_WEIGHT = 0.55
MIN_SEMANTIC_SCORE = 0.2

def chunk_text(text: str, target_size: int = 900, overlap: int = 120) -> List[str]:
    """Split extracted text into bounded overlapping chunks without external models."""
    normalized = re.sub(r"[ \t]+", " ", text).strip()
    if not normalized:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + target_size)
        if end < len(normalized):
            boundary = max(normalized.rfind("\n", start + target_size // 2, end),
                           normalized.rfind(" ", start + target_size // 2, end))
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks

def _tokens(text: str) -> List[str]:
    return [token for token in re.findall(r"[^\W_]+(?:['’-][^\W_]+)?", text.casefold()) if len(token) > 1]

def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))

def search_material_store(
    store: Mapping[str, Any],
    query: str,
    limit: int = 5,
    query_embedding: Optional[Sequence[float]] = None,
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Rank chunks with explainable weighted lexical/semantic score fusion."""
    chunks = [item for item in (store.get("chunks") or []) if isinstance(item, Mapping)]
    query_terms = Counter(_tokens(query))
    bounded_limit = max(1, min(MAX_SEARCH_RESULTS, int(limit or 5)))
    semantic_enabled = bool(query_embedding and embedding_model)
    strategy = "hybrid" if semantic_enabled else "lexical"
    metadata = {
        "strategy": strategy,
        "weights": {
            "lexical": LEXICAL_WEIGHT if semantic_enabled else 1.0,
            "semantic": SEMANTIC_WEIGHT if semantic_enabled else 0.0,
        },
    }
    if not chunks or (not query_terms and not semantic_enabled):
        return {
            "total_documents": len(store.get("documents") or []),
            "retrieval": metadata,
            "items": [],
        }

    chunk_terms = [Counter(_tokens(str(chunk.get("text") or ""))) for chunk in chunks]
    document_frequency = Counter()
    for terms in chunk_terms:
        document_frequency.update(terms.keys())
    chunk_count = len(chunks)
    idf = {
        term: math.log((1 + chunk_count) / (1 + document_frequency.get(term, 0))) + 1
        for term in query_terms
    }
    query_norm = math.sqrt(sum((count * idf[term]) ** 2 for term, count in query_terms.items())) or 1.0

    matches: List[Dict[str, Any]] = []
    for chunk, terms in zip(chunks, chunk_terms):
        dot = sum(query_terms[term] * idf[term] * terms.get(term, 0) * idf[term] for term in query_terms)
        lexical_score = 0.0
        if dot > 0:
            chunk_norm = math.sqrt(
                sum((terms.get(term, 0) * idf[term]) ** 2 for term in query_terms)
            ) or 1.0
            lexical_score = dot / (query_norm * chunk_norm)
        semantic_score = 0.0
        if semantic_enabled and chunk.get("embedding_model") == embedding_model:
            vector = chunk.get("embedding")
            if isinstance(vector, list):
                semantic_score = _cosine_similarity(query_embedding or [], vector)
        if lexical_score <= 0 and semantic_score < MIN_SEMANTIC_SCORE:
            continue
        if semantic_enabled:
            score = LEXICAL_WEIGHT * lexical_score + SEMANTIC_WEIGHT * semantic_score
            match_kind = (
                "hybrid" if lexical_score > 0 and semantic_score >= MIN_SEMANTIC_SCORE
                else "semantic" if semantic_score >= MIN_SEMANTIC_SCORE
                else "lexical"
            )
        else:
            score = lexical_score
            match_kind = "lexical"
        matches.append({
            "document_id": str(chunk.get("document_id") or ""),
            "source_name": str(chunk.get("source_name") or ""),
            "chunk_index": int(chunk.get("chunk_index") or 0),
            "text": str(chunk.get("text") or ""),
            "score": round(score, 4),
            "lexical_score": round(lexical_score, 4),
            "semantic_score": round(semantic_score, 4),
            "match": match_kind,
        })
    matches.sort(key=lambda item: (-item["score"], item["source_name"], item["chunk_index"]))
    return {
        "total_documents": len(store.get("documents") or []),
        "retrieval": metadata,
        "items": matches[:bounded_limit],
    }
