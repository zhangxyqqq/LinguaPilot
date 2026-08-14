import io
import json
import math
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from .storage import get_storage


ROOT_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT_DIR / "state"
MATERIALS_DIR = STATE_DIR / "materials"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_EXTRACTED_CHARS = 1_000_000
MAX_SEARCH_RESULTS = 8
ALLOWED_SUFFIXES = {".txt", ".md", ".csv", ".pdf"}

router = APIRouter()


def _validate_book_id(book_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", book_id):
        raise ValueError("invalid book_id")
    return book_id


def _store_path(book_id: str, materials_dir: Optional[Path] = None) -> Path:
    safe_book_id = _validate_book_id(book_id)
    return (materials_dir or MATERIALS_DIR) / f"{safe_book_id}.json"


def _empty_store(book_id: str) -> Dict[str, Any]:
    return {"version": 1, "book_id": book_id, "documents": [], "chunks": []}


def load_material_store(
    book_id: str,
    materials_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    if materials_dir is None:
        data = get_storage().load_material_store(book_id, user_id)
        if data is None:
            return _empty_store(book_id)
        if not isinstance(data, Mapping) or data.get("book_id") != book_id:
            raise ValueError(f"invalid material store for book_id={book_id}")
        documents = data.get("documents") if isinstance(data.get("documents"), list) else []
        chunks = data.get("chunks") if isinstance(data.get("chunks"), list) else []
        return {
            "version": 1,
            "book_id": book_id,
            "documents": [dict(item) for item in documents if isinstance(item, Mapping)],
            "chunks": [dict(item) for item in chunks if isinstance(item, Mapping)],
        }
    path = _store_path(book_id, materials_dir)
    if not path.exists():
        return _empty_store(book_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or data.get("book_id") != book_id:
        raise ValueError(f"invalid material store for book_id={book_id}")
    documents = data.get("documents") if isinstance(data.get("documents"), list) else []
    chunks = data.get("chunks") if isinstance(data.get("chunks"), list) else []
    return {
        "version": 1,
        "book_id": book_id,
        "documents": [dict(item) for item in documents if isinstance(item, Mapping)],
        "chunks": [dict(item) for item in chunks if isinstance(item, Mapping)],
    }


def _write_material_store(
    book_id: str,
    store: Mapping[str, Any],
    materials_dir: Optional[Path] = None,
) -> None:
    if materials_dir is None:
        get_storage().save_material_store(book_id, store)
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


def add_material(
    book_id: str,
    filename: str,
    raw: bytes,
    materials_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
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
    store["chunks"].extend({
        "document_id": document_id,
        "source_name": safe_name,
        "chunk_index": index,
        "text": chunk,
    } for index, chunk in enumerate(chunks))
    _write_material_store(book_id, store, materials_dir)
    return document


def search_material_store(
    store: Mapping[str, Any],
    query: str,
    limit: int = 5,
) -> Dict[str, Any]:
    """Rank book-scoped chunks with a small deterministic TF-IDF index."""
    chunks = [item for item in (store.get("chunks") or []) if isinstance(item, Mapping)]
    query_terms = Counter(_tokens(query))
    bounded_limit = max(1, min(MAX_SEARCH_RESULTS, int(limit or 5)))
    if not chunks or not query_terms:
        return {"total_documents": len(store.get("documents") or []), "items": []}

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
        if dot <= 0:
            continue
        chunk_norm = math.sqrt(sum((terms.get(term, 0) * idf[term]) ** 2 for term in query_terms)) or 1.0
        score = dot / (query_norm * chunk_norm)
        matches.append({
            "document_id": str(chunk.get("document_id") or ""),
            "source_name": str(chunk.get("source_name") or ""),
            "chunk_index": int(chunk.get("chunk_index") or 0),
            "text": str(chunk.get("text") or ""),
            "score": round(score, 4),
        })
    matches.sort(key=lambda item: (-item["score"], item["source_name"], item["chunk_index"]))
    return {"total_documents": len(store.get("documents") or []), "items": matches[:bounded_limit]}


def search_learning_materials_for_book(
    book_id: str,
    query: str,
    limit: int = 5,
    materials_dir: Optional[Path] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    return search_material_store(
        load_material_store(book_id, materials_dir, user_id),
        query=query,
        limit=limit,
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
