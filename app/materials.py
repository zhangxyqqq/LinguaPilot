import io
import json
import math
import os
import re
import uuid
import logging
from time import perf_counter
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from fastapi import APIRouter, File, HTTPException, UploadFile

from .storage import get_storage, validate_user_id
from . import retrieval
from .retrieval_baseline import search_material_store as legacy_search

logger = logging.getLogger(__name__)


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


def _store_path(book_id: str, materials_dir: Optional[Path] = None, user_id: Optional[str] = None) -> Path:
    safe_book_id = _validate_book_id(book_id)
    directory = materials_dir or MATERIALS_DIR
    if user_id is not None:
        directory = directory / validate_user_id(user_id)
    return directory / f"{safe_book_id}.json"


def _empty_store(book_id: str) -> Dict[str, Any]:
    return {"version": 2, "book_id": book_id, "documents": [], "chunks": []}


def _normalize_store(data: Mapping[str, Any], book_id: str) -> Dict[str, Any]:
    if data.get("book_id") != book_id:
        raise ValueError(f"invalid material store for book_id={book_id}")
    documents = data.get("documents") if isinstance(data.get("documents"), list) else []
    chunks = data.get("chunks") if isinstance(data.get("chunks"), list) else []
    return {
        "_expected_json": json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
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
    path = _store_path(book_id, materials_dir, user_id)
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
        payload = {k: v for k, v in store.items() if k != "_expected_json"}
        get_storage().save_material_store(book_id, payload, user_id,
                                          expected_json=store.get("_expected_json"))
        return
    path = _store_path(book_id, materials_dir, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({k: v for k, v in store.items() if k != "_expected_json"}, ensure_ascii=False, indent=2), encoding="utf-8")
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


def chunk_text(text: str, target_size: int = 900, overlap: int = 120, strategy: str = "fixed") -> List[str]:
    """Split extracted text into bounded overlapping chunks without external models."""
    if target_size < 2 or overlap < 0 or overlap >= target_size:
        raise ValueError("require target_size >= 2 and 0 <= overlap < target_size")
    if strategy not in {"fixed", "paragraph"}:
        raise ValueError("unknown chunking strategy")
    if strategy == "paragraph":
        return [chunk for paragraph in re.split(r"\n\s*\n", text)
                for chunk in chunk_text(paragraph, target_size, overlap)]
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
    pending = []
    for chunk in chunks:
        valid = (chunk.get("embedding_model") == provider.model_id
                 and chunk.get("embedding_text_sha256") == retrieval.text_digest(chunk["text"]))
        try:
            retrieval.normalized_vector(chunk.get("embedding") or [])
        except (ValueError, TypeError, OverflowError):
            valid = False
        if not valid:
            pending.append(chunk)
    dimensions = {len(c["embedding"]) for c in chunks if c not in pending}
    if len(dimensions) > 1:
        pending = list(chunks)
    if not pending:
        return False
    vectors = provider.embed_documents([chunk["text"] for chunk in pending])
    if len(vectors) != len(pending):
        raise ValueError("embedding provider returned an unexpected vector count")
    # Validate the entire response before changing any chunk (atomic cache fill).
    parsed = [[float(v) for v in vector] for vector in vectors]
    dimensions = {len(vector) for vector in parsed}
    dimensions.update(len(c["embedding"]) for c in chunks if c not in pending)
    if len(dimensions) != 1:
        raise ValueError("inconsistent embedding dimensions")
    for vector in parsed:
        retrieval.normalized_vector(vector)
    for chunk, vector in zip(pending, parsed):
        chunk["embedding_model"] = provider.model_id
        chunk["embedding"] = vector
        chunk["embedding_text_sha256"] = retrieval.text_digest(chunk["text"])
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
    *, user_id: Optional[str] = None, chunking: str = "fixed",
    replace_document_id: Optional[str] = None,
) -> Dict[str, Any]:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("uploaded material exceeds 5 MB")
    safe_name = Path(filename or "material.txt").name
    text = _extract_text(safe_name, raw)
    chunks = chunk_text(text, strategy=chunking)
    if not chunks:
        raise ValueError("uploaded material contains no indexable text")

    store = load_material_store(book_id, materials_dir, user_id)
    if replace_document_id and not any(d["document_id"] == replace_document_id for d in store["documents"]):
        raise ValueError("material to replace not found")
    document_id = replace_document_id or uuid.uuid4().hex[:12]
    store["documents"] = [d for d in store["documents"] if d["document_id"] != document_id]
    store["chunks"] = [c for c in store["chunks"] if c["document_id"] != document_id]
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    document = {
        "document_id": document_id,
        "source_name": safe_name,
        "uploaded_at": timestamp,
        "char_count": len(text),
        "chunk_count": len(chunks),
        "chunking": chunking,
        "content_sha256": retrieval.text_digest(text),
    }
    store["documents"].append(document)
    new_chunks = [{
        "document_id": document_id,
        "source_name": safe_name,
        "chunk_index": index,
        "text": chunk,
    } for index, chunk in enumerate(chunks)]
    try:
        provider = _resolve_embedder(embedding_provider)
        if provider is not None:
            _cache_chunk_embeddings(new_chunks, provider)
            document["embedding_model"] = provider.model_id
    except Exception as exc:
        document["embedding_status"] = "unavailable"
        logger.warning("Material embedding failed; lexical retrieval remains available: %s", type(exc).__name__)
    store["chunks"].extend(new_chunks)
    _write_material_store(book_id, store, materials_dir, user_id)
    return document


def search_material_store(store, query, limit=5, query_embedding=None, embedding_model=None,
                          *, config=None, reranker=None, trace=None):
    config = config or retrieval.DEFAULT_CONFIG
    if config.mode == "legacy":
        started = perf_counter()
        result = legacy_search(store, query, limit, query_embedding, embedding_model)
        elapsed = (perf_counter() - started) * 1000
        if trace is not None:
            # Legacy has a single combined scoring stage; do not invent stage timings.
            trace.update(mode="legacy", query=query, reranker_used=False, errors=[],
                         candidate_counts={"scored": len(store.get("chunks") or [])},
                         stages={"legacy_scoring_ms": elapsed, "total_ms": elapsed},
                         candidates=[{k:v for k,v in item.items() if k != "text"} |
                                     {"fused_rank":rank, "final_rank":rank, "reranker_score":None}
                                     for rank,item in enumerate(result["items"], 1)])
        return result
    return retrieval.retrieve(store, query, limit, config, query_embedding,
                              embedding_model, reranker, trace)


def search_learning_materials_for_book(
    book_id: str,
    query: str,
    limit: int = 5,
    materials_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
    embedding_provider: Any = _USE_DEFAULT_EMBEDDER,
    *, config=None, reranker=None, trace=None,
) -> Dict[str, Any]:
    store = load_material_store(book_id, materials_dir, user_id)
    config = config or retrieval.RetrievalConfig(
        mode=os.getenv("LANGBUDDY_RETRIEVAL_MODE", retrieval.DEFAULT_CONFIG.mode),
        rerank=os.getenv("LANGBUDDY_RERANK", "disabled") == "enabled")
    provider = None
    query_embedding: Optional[List[float]] = None
    fallback = None
    try:
        if config.mode != "bm25" and store["chunks"] and query.strip():
            provider = _resolve_embedder(embedding_provider)
    except Exception as exc:
        fallback = "provider:" + type(exc).__name__
    if provider is not None and store["chunks"] and query.strip():
        try:
            changed = _cache_chunk_embeddings(store["chunks"], provider)
            if changed:
                _write_material_store(book_id, store, materials_dir, user_id)
            query_embedding = provider.embed_query(query)
            retrieval.ExactVectorIndex(store["chunks"], provider.model_id).search(query_embedding, 1)
        except Exception as exc:
            logger.warning("Semantic retrieval failed; using lexical retrieval: %s", type(exc).__name__)
            fallback = "embedding:" + type(exc).__name__
            # A concurrent deletion/replacement may have won the cache CAS.
            store = load_material_store(book_id, materials_dir, user_id)
            provider = None
            query_embedding = None
    result = search_material_store(
        store,
        query=query,
        limit=limit,
        query_embedding=query_embedding,
        embedding_model=provider.model_id if provider else None,
        config=config, reranker=reranker, trace=trace,
    )
    if fallback:
        logger.warning("Retrieval degraded: %s", fallback)
        if trace is not None:
            trace.setdefault("errors", []).append(fallback)
        result["retrieval"]["fallbacks"] = result["retrieval"].get("fallbacks", []) + [fallback]
    return result


def delete_material(
    book_id: str,
    document_id: str,
    materials_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> bool:
    store = load_material_store(book_id, materials_dir, user_id)
    before = len(store["documents"])
    store["documents"] = [item for item in store["documents"] if item.get("document_id") != document_id]
    if len(store["documents"]) == before:
        return False
    store["chunks"] = [item for item in store["chunks"] if item.get("document_id") != document_id]
    _write_material_store(book_id, store, materials_dir, user_id)
    return True


def reset_materials(book_id: str, materials_dir: Optional[Path] = None, user_id: Optional[str] = None) -> bool:
    if materials_dir is None:
        return get_storage().delete_material_store(book_id, user_id)
    path = _store_path(book_id, materials_dir, user_id)
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
