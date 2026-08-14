import io
import json
import math
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from fastapi import APIRouter, File, HTTPException, UploadFile

from .storage import get_storage


ROOT_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT_DIR / "state"
MATERIALS_DIR = STATE_DIR / "materials"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_EXTRACTED_CHARS = 1_000_000
MAX_SEARCH_RESULTS = 8
LEXICAL_WEIGHT = 0.45
SEMANTIC_WEIGHT = 0.55
MIN_SEMANTIC_SCORE = 0.2
ALLOWED_SUFFIXES = {".txt", ".md", ".csv", ".pdf"}
_USE_DEFAULT_EMBEDDER = object()

router = APIRouter()


class EmbeddingProvider(Protocol):
    model_id: str

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


class OpenAIEmbeddingProvider:
    """Small OpenAI adapter; vectors are persisted with each material chunk."""

    def __init__(self, model: str, dimensions: int):
        from openai import OpenAI

        self.model = model
        self.dimensions = dimensions
        self.model_id = f"openai:{model}:{dimensions}"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), 64):
            response = self.client.embeddings.create(
                model=self.model,
                dimensions=self.dimensions,
                input=list(texts[start:start + 64]),
            )
            vectors.extend([list(item.embedding) for item in response.data])
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


@lru_cache(maxsize=1)
def get_embedding_provider() -> Optional[EmbeddingProvider]:
    if os.getenv("LANGBUDDY_EMBEDDINGS", "enabled").strip().casefold() in {
        "0", "false", "off", "disabled",
    }:
        return None
    if not os.getenv("OPENAI_API_KEY"):
        return None
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    try:
        dimensions = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "256"))
    except ValueError:
        dimensions = 256
    return OpenAIEmbeddingProvider(model=model, dimensions=max(64, min(1536, dimensions)))


def _validate_book_id(book_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", book_id):
        raise ValueError("invalid book_id")
    return book_id


def _store_path(book_id: str, materials_dir: Optional[Path] = None) -> Path:
    safe_book_id = _validate_book_id(book_id)
    return (materials_dir or MATERIALS_DIR) / f"{safe_book_id}.json"


def _empty_store(book_id: str) -> Dict[str, Any]:
    return {"version": 2, "book_id": book_id, "documents": [], "chunks": []}


def _normalize_store(data: Mapping[str, Any], book_id: str) -> Dict[str, Any]:
    if data.get("book_id") != book_id:
        raise ValueError(f"invalid material store for book_id={book_id}")
    documents = data.get("documents") if isinstance(data.get("documents"), list) else []
    chunks = data.get("chunks") if isinstance(data.get("chunks"), list) else []
    return {
        "version": 2,
        "book_id": book_id,
        "documents": [dict(item) for item in documents if isinstance(item, Mapping)],
        "chunks": [dict(item) for item in chunks if isinstance(item, Mapping)],
    }


def load_material_store(
    book_id: str,
    materials_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    if materials_dir is None:
        data = get_storage().load_material_store(book_id, user_id)
        if data is None:
            return _empty_store(book_id)
        if not isinstance(data, Mapping):
            raise ValueError(f"invalid material store for book_id={book_id}")
        return _normalize_store(data, book_id)
    path = _store_path(book_id, materials_dir)
    if not path.exists():
        return _empty_store(book_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"invalid material store for book_id={book_id}")
    return _normalize_store(data, book_id)


def _write_material_store(
    book_id: str,
    store: Mapping[str, Any],
    materials_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> None:
    if materials_dir is None:
        get_storage().save_material_store(book_id, store, user_id)
        return
    path = _store_path(book_id, materials_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _extract_text(filename: str, raw: bytes) -> str:
    suffix = Path(filename).suffix.casefold()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("supported material types: .txt, .md, .csv, .pdf")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("PDF support requires pypdf") from exc
        try:
            reader = PdfReader(io.BytesIO(raw))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            raise ValueError("could not extract text from PDF") from exc
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
    text = text.replace("\x00", "").strip()
    if not text:
        raise ValueError("uploaded material contains no extractable text")
    return text[:MAX_EXTRACTED_CHARS]


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


def _resolve_embedder(value: Any) -> Optional[EmbeddingProvider]:
    return get_embedding_provider() if value is _USE_DEFAULT_EMBEDDER else value


def _cache_chunk_embeddings(
    chunks: List[Dict[str, Any]], provider: Optional[EmbeddingProvider]
) -> bool:
    if provider is None:
        return False
    pending = [
        chunk for chunk in chunks
        if chunk.get("embedding_model") != provider.model_id
        or not isinstance(chunk.get("embedding"), list)
    ]
    if not pending:
        return False
    vectors = provider.embed_documents([str(chunk.get("text") or "") for chunk in pending])
    if len(vectors) != len(pending):
        raise ValueError("embedding provider returned an unexpected vector count")
    for chunk, vector in zip(pending, vectors):
        chunk["embedding_model"] = provider.model_id
        chunk["embedding"] = [float(value) for value in vector]
    return True


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def add_material(
    book_id: str,
    filename: str,
    raw: bytes,
    materials_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
    embedding_provider: Any = _USE_DEFAULT_EMBEDDER,
) -> Dict[str, Any]:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("uploaded material exceeds 5 MB")
    safe_name = Path(filename or "material.txt").name
    text = _extract_text(safe_name, raw)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("uploaded material contains no indexable text")

    store = load_material_store(book_id, materials_dir)
    document_id = uuid.uuid4().hex[:12]
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    document = {
        "document_id": document_id,
        "source_name": safe_name,
        "uploaded_at": timestamp,
        "char_count": len(text),
        "chunk_count": len(chunks),
    }
    store["documents"].append(document)
    new_chunks = [{
        "document_id": document_id,
        "source_name": safe_name,
        "chunk_index": index,
        "text": chunk,
    } for index, chunk in enumerate(chunks)]
    provider = _resolve_embedder(embedding_provider)
    if provider is not None:
        try:
            _cache_chunk_embeddings(new_chunks, provider)
            document["embedding_model"] = provider.model_id
        except Exception as exc:
            print(f"Material embedding failed; lexical retrieval remains available: {type(exc).__name__}")
    store["chunks"].extend(new_chunks)
    _write_material_store(book_id, store, materials_dir)
    return document


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


def search_learning_materials_for_book(
    book_id: str,
    query: str,
    limit: int = 5,
    materials_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
    embedding_provider: Any = _USE_DEFAULT_EMBEDDER,
) -> Dict[str, Any]:
    store = load_material_store(book_id, materials_dir, user_id)
    provider = _resolve_embedder(embedding_provider)
    query_embedding: Optional[List[float]] = None
    if provider is not None:
        try:
            changed = _cache_chunk_embeddings(store["chunks"], provider)
            if changed:
                _write_material_store(book_id, store, materials_dir, user_id)
            query_embedding = provider.embed_query(query)
        except Exception as exc:
            print(f"Semantic retrieval failed; using lexical retrieval: {type(exc).__name__}")
            provider = None
            query_embedding = None
    return search_material_store(
        store,
        query=query,
        limit=limit,
        query_embedding=query_embedding,
        embedding_model=provider.model_id if provider else None,
    )


def delete_material(
    book_id: str,
    document_id: str,
    materials_dir: Optional[Path] = None,
) -> bool:
    store = load_material_store(book_id, materials_dir)
    before = len(store["documents"])
    store["documents"] = [item for item in store["documents"] if item.get("document_id") != document_id]
    if len(store["documents"]) == before:
        return False
    store["chunks"] = [item for item in store["chunks"] if item.get("document_id") != document_id]
    _write_material_store(book_id, store, materials_dir)
    return True


def reset_materials(book_id: str, materials_dir: Optional[Path] = None) -> bool:
    if materials_dir is None:
        return get_storage().delete_material_store(book_id)
    path = _store_path(book_id, materials_dir)
    if not path.exists():
        return False
    path.unlink()
    return True


def _require_book(book_id: str) -> None:
    try:
        safe_book_id = _validate_book_id(book_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not get_storage().book_exists(safe_book_id):
        raise HTTPException(404, "book not found")


@router.post("/api/materials/{book_id}")
async def upload_material(book_id: str, file: UploadFile = File(...)):
    _require_book(book_id)
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        document = add_material(book_id, file.filename or "material.txt", raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"book_id": book_id, "document": document}


@router.get("/api/materials/{book_id}")
def list_materials(book_id: str):
    _require_book(book_id)
    store = load_material_store(book_id)
    return {"book_id": book_id, "items": store["documents"]}


@router.delete("/api/materials/{book_id}/{document_id}")
def remove_material(book_id: str, document_id: str):
    _require_book(book_id)
    if not delete_material(book_id, document_id):
        raise HTTPException(404, "material not found")
    return {"book_id": book_id, "deleted": document_id}


@router.delete("/api/materials/{book_id}")
def clear_materials(book_id: str):
    _require_book(book_id)
    return {"book_id": book_id, "reset": reset_materials(book_id)}
