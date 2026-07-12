from backend.context.citation_builder import (
    build_deduplicated_citations,
    citation_key,
)


def test_citation_key_prefers_citation_path():
    doc = {"citationPath": "manual/section-3", "id": "doc-1", "title": "T"}
    assert citation_key(doc) == "citationPath::manual/section-3"


def test_citation_key_falls_back_to_id_then_fields():
    assert citation_key({"id": "doc-9"}) == "id::doc-9"
    key = citation_key({"title": "Brakes", "machine": "M1", "manualType": "service"})
    assert key == "fallback::Brakes::M1::service"


def test_build_deduplicated_citations_assigns_sequential_ids():
    docs = [
        {"citationPath": "a", "title": "First"},
        {"citationPath": "b", "title": "Second"},
        {"citationPath": "a", "title": "Duplicate of first"},
    ]
    citations, id_by_key = build_deduplicated_citations(docs)

    assert [c["citationId"] for c in citations] == [1, 2]
    assert id_by_key["citationPath::a"] == 1
    assert id_by_key["citationPath::b"] == 2


def test_build_deduplicated_citations_empty_input():
    citations, id_by_key = build_deduplicated_citations([])
    assert citations == []
    assert id_by_key == {}
