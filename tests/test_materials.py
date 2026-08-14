from io import BytesIO
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app import materials
from app.materials import (
    add_material,
    delete_material,
    load_material_store,
    reset_materials,
    search_learning_materials_for_book,
)
from app.storage import get_storage, reset_current_user, set_current_user


class _SemanticEmbedder:
    model_id = "deterministic:concepts:v1"

    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    @staticmethod
    def _vector(text):
        value = text.casefold()
        if any(term in value for term in ("car", "automobile", "vehicle")):
            return [1.0, 0.0, 0.0]
        if any(term in value for term in ("teacher", "instructor", "educator")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return self._vector(text)


class _FailingEmbedder(_SemanticEmbedder):
    model_id = "deterministic:failing:v1"

    def embed_documents(self, texts):
        raise RuntimeError("embedding service unavailable")


def _text_pdf_bytes(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_ref = writer._add_object(font)
    resources = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
    })
    page[NameObject("/Resources")] = resources
    safe_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = DecodedStreamObject()
    content.set_data(f"BT /F1 12 Tf 72 720 Td ({safe_text}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(content)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_material_upload_search_and_source_attribution(tmp_path):
    document = add_material(
        "book-1",
        "lecture-notes.md",
        b"The passive voice uses a form of be plus the past participle.\n\nActive voice names the doer.",
        materials_dir=tmp_path,
        now=datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc),
    )

    result = search_learning_materials_for_book(
        "book-1",
        "How does passive voice use the past participle?",
        materials_dir=tmp_path,
    )

    assert document["document_id"]
    assert result["total_documents"] == 1
    assert result["items"][0]["source_name"] == "lecture-notes.md"
    assert result["items"][0]["document_id"] == document["document_id"]
    assert "past participle" in result["items"][0]["text"]


def test_material_search_no_match_and_book_scope(tmp_path):
    add_material("book-a", "a.txt", b"Danish modal verbs include kan and skal.", materials_dir=tmp_path)
    add_material("book-b", "b.txt", b"Spanish nouns have grammatical gender.", materials_dir=tmp_path)

    book_a = search_learning_materials_for_book("book-a", "modal verbs", materials_dir=tmp_path)
    no_match = search_learning_materials_for_book("book-a", "Spanish gender", materials_dir=tmp_path)

    assert [item["source_name"] for item in book_a["items"]] == ["a.txt"]
    assert no_match["items"] == []
    assert all(item["source_name"] != "b.txt" for item in book_a["items"])


def test_semantic_retrieval_finds_synonym_without_lexical_overlap_and_caches_vectors(tmp_path):
    embedder = _SemanticEmbedder()
    add_material(
        "book-1",
        "transport.txt",
        b"An automobile needs regular maintenance and careful servicing.",
        materials_dir=tmp_path,
        embedding_provider=embedder,
    )

    lexical_only = search_learning_materials_for_book(
        "book-1", "car", materials_dir=tmp_path, embedding_provider=None
    )
    semantic = search_learning_materials_for_book(
        "book-1", "car", materials_dir=tmp_path, embedding_provider=embedder
    )
    repeated = search_learning_materials_for_book(
        "book-1", "vehicle", materials_dir=tmp_path, embedding_provider=embedder
    )

    assert lexical_only["items"] == []
    assert semantic["retrieval"] == {
        "strategy": "hybrid",
        "weights": {"lexical": 0.45, "semantic": 0.55},
    }
    assert semantic["items"][0]["source_name"] == "transport.txt"
    assert semantic["items"][0]["lexical_score"] == 0
    assert semantic["items"][0]["semantic_score"] == 1
    assert semantic["items"][0]["match"] == "semantic"
    assert repeated["items"][0]["source_name"] == "transport.txt"
    assert embedder.document_calls == 1
    assert embedder.query_calls == 2

    persisted = load_material_store("book-1", tmp_path)
    assert persisted["version"] == 2
    assert persisted["chunks"][0]["embedding_model"] == embedder.model_id
    assert persisted["chunks"][0]["embedding"] == [1.0, 0.0, 0.0]


def test_existing_chunks_are_backfilled_once_and_embedding_failure_falls_back(tmp_path):
    add_material(
        "book-1",
        "grammar.txt",
        b"The passive voice uses a past participle.",
        materials_dir=tmp_path,
        embedding_provider=None,
    )
    embedder = _SemanticEmbedder()
    first = search_learning_materials_for_book(
        "book-1", "passive voice", materials_dir=tmp_path, embedding_provider=embedder
    )
    second = search_learning_materials_for_book(
        "book-1", "passive", materials_dir=tmp_path, embedding_provider=embedder
    )

    assert first["items"][0]["source_name"] == "grammar.txt"
    assert second["items"][0]["source_name"] == "grammar.txt"
    assert embedder.document_calls == 1

    fallback = search_learning_materials_for_book(
        "book-1", "past participle", materials_dir=tmp_path, embedding_provider=_FailingEmbedder()
    )
    assert fallback["retrieval"]["strategy"] == "lexical"
    assert fallback["items"][0]["source_name"] == "grammar.txt"


def test_hybrid_retrieval_remains_user_scoped_in_sqlite():
    storage = get_storage()
    for owner in ("alice", "bob"):
        storage.save_book_state({"book_id": "shared-book", "user": {"cards": {}}}, owner)
    embedder = _SemanticEmbedder()

    alice_token = set_current_user("alice")
    try:
        add_material(
            "shared-book",
            "alice.txt",
            b"An automobile needs regular maintenance.",
            embedding_provider=embedder,
        )
    finally:
        reset_current_user(alice_token)
    bob_token = set_current_user("bob")
    try:
        add_material(
            "shared-book",
            "bob.txt",
            b"An educator prepares a lesson plan.",
            embedding_provider=embedder,
        )
    finally:
        reset_current_user(bob_token)

    alice = search_learning_materials_for_book(
        "shared-book", "car", user_id="alice", embedding_provider=embedder
    )
    bob = search_learning_materials_for_book(
        "shared-book", "car", user_id="bob", embedding_provider=embedder
    )

    assert [item["source_name"] for item in alice["items"]] == ["alice.txt"]
    assert bob["items"] == []


def test_material_delete_reset_and_empty_store(tmp_path):
    assert load_material_store("book-1", tmp_path)["documents"] == []
    first = add_material("book-1", "first.txt", b"alpha vocabulary note", materials_dir=tmp_path)
    second = add_material("book-1", "second.txt", b"beta vocabulary note", materials_dir=tmp_path)

    assert delete_material("book-1", first["document_id"], tmp_path)
    store = load_material_store("book-1", tmp_path)
    assert [item["document_id"] for item in store["documents"]] == [second["document_id"]]
    assert all(item["document_id"] == second["document_id"] for item in store["chunks"])
    assert not delete_material("book-1", "missing", tmp_path)

    assert reset_materials("book-1", tmp_path)
    assert load_material_store("book-1", tmp_path)["documents"] == []
    assert not reset_materials("book-1", tmp_path)


def test_material_filename_is_sanitized_and_invalid_book_id_rejected(tmp_path):
    document = add_material("book-1", "../../notes.txt", b"safe content", materials_dir=tmp_path)
    assert document["source_name"] == "notes.txt"

    try:
        load_material_store("../other", tmp_path)
    except ValueError as exc:
        assert "invalid book_id" in str(exc)
    else:
        raise AssertionError("invalid book_id should be rejected")


def test_pdf_api_upload_extract_retrieve_source_and_book_scope():
    for book_id in ("book-a", "book-b"):
        get_storage().save_book_state({"book_id": book_id, "user": {"cards": {}}})

    app = FastAPI()
    app.include_router(materials.router)
    client = TestClient(app)
    pdf = _text_pdf_bytes("The subjunctive expresses wishes and hypothetical situations.")

    uploaded = client.post(
        "/api/materials/book-a",
        files={"file": ("lesson.pdf", pdf, "application/pdf")},
    )
    assert uploaded.status_code == 200
    document = uploaded.json()["document"]
    assert document["source_name"] == "lesson.pdf"
    assert document["chunk_count"] == 1

    retrieved = search_learning_materials_for_book(
        "book-a", "When is the subjunctive used for hypothetical wishes?"
    )
    assert retrieved["items"][0]["source_name"] == "lesson.pdf"
    assert "hypothetical situations" in retrieved["items"][0]["text"]
    assert search_learning_materials_for_book(
        "book-b", "subjunctive hypothetical wishes"
    )["items"] == []
