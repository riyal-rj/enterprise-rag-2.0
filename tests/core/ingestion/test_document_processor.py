from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from app.core.ingestion.document_processor import DoclingDocumentProcessor


@dataclass
class _FakeProvenance:
    page_no: int


@dataclass
class _FakeDocItem:
    prov: list[_FakeProvenance] = field(default_factory=list)


@dataclass
class _FakeChunkMeta:
    doc_items: list[_FakeDocItem]


@dataclass
class _FakeChunk:
    text: str
    meta: _FakeChunkMeta


class _FakeChunker:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks

    def chunk(self, doc: object) -> list[_FakeChunk]:
        return self._chunks


class _FakeConversionResult:
    def __init__(self, document: object) -> None:
        self.document = document


class _FakeConverter:
    def __init__(self, document: object = None) -> None:
        self._document = document
        self.calls: list[object] = []

    def convert(self, source: object) -> _FakeConversionResult:
        self.calls.append(source)
        return _FakeConversionResult(self._document)


def _processor(chunks: list[_FakeChunk]) -> tuple[DoclingDocumentProcessor, _FakeConverter]:
    converter = _FakeConverter()
    processor = DoclingDocumentProcessor(
        converter=cast(DocumentConverter, converter),
        chunker=cast(HybridChunker, _FakeChunker(chunks)),
    )
    return processor, converter


def test_process_document_maps_chunk_text_and_source() -> None:
    chunks = [
        _FakeChunk(
            text="hello world",
            meta=_FakeChunkMeta(doc_items=[_FakeDocItem(prov=[_FakeProvenance(page_no=3)])]),
        )
    ]
    processor, converter = _processor(chunks)

    result = processor.process_document("/data/policies/handbook.pdf")

    assert len(result) == 1
    assert result[0].text == "hello world"
    assert result[0].source == "handbook.pdf"
    assert result[0].page_number == 3
    assert len(converter.calls) == 1


def test_process_document_uses_file_name_not_full_path_as_source() -> None:
    chunks = [
        _FakeChunk(
            text="text",
            meta=_FakeChunkMeta(doc_items=[_FakeDocItem(prov=[_FakeProvenance(page_no=1)])]),
        )
    ]
    processor, _ = _processor(chunks)

    result = processor.process_document("/some/nested/dir/report.pdf")

    assert result[0].source == "report.pdf"


def test_process_document_chunk_without_provenance_has_no_page_number() -> None:
    chunks = [
        _FakeChunk(text="heading-only chunk", meta=_FakeChunkMeta(doc_items=[_FakeDocItem(prov=[])]))
    ]
    processor, _ = _processor(chunks)

    result = processor.process_document("doc.pdf")

    assert result[0].page_number is None


def test_process_document_returns_one_chunk_per_docling_chunk_in_order() -> None:
    chunks = [
        _FakeChunk(text="first", meta=_FakeChunkMeta(doc_items=[_FakeDocItem(prov=[])])),
        _FakeChunk(text="second", meta=_FakeChunkMeta(doc_items=[_FakeDocItem(prov=[])])),
    ]
    processor, _ = _processor(chunks)

    result = processor.process_document("doc.pdf")

    assert [c.text for c in result] == ["first", "second"]


def test_process_document_returns_empty_list_for_no_chunks() -> None:
    processor, _ = _processor([])

    assert processor.process_document("empty.pdf") == []
