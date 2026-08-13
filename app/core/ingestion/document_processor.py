"""Adapter around the Docling document-conversion/chunking pipeline.

Same shape as ``chat_client.py``/``embedding_client.py``: wraps an external
SDK (Docling) behind an in-house Protocol, and the converter is built via a
``build_*`` factory from injected settings rather than constructed inline —
callers depend on ``DocumentProcessor``, never on ``docling`` directly, and
the converter/chunker can be swapped for fakes in tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, cast

from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.doc_chunk import DocChunk
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DocumentChunk(BaseModel):
    """One chunk of a converted source document, ready for embedding."""

    text: str
    source: str
    page_number: int | None = None


class DocumentProcessor(Protocol):
    """Contract for a document conversion/chunking pipeline."""

    def process_document(self, file_path: str) -> list[DocumentChunk]: ...


def build_docling_converter(accelerator_device: str, num_threads: int) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=num_threads, device=AcceleratorDevice(accelerator_device)
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


class DoclingDocumentProcessor:
    """:class:`DocumentProcessor` backed by Docling's converter + hybrid chunker."""

    def __init__(self, converter: DocumentConverter, chunker: HybridChunker) -> None:
        self._converter = converter
        self._chunker = chunker

    def process_document(self, file_path: str) -> list[DocumentChunk]:
        """Convert ``file_path`` and split it into :class:`DocumentChunk`\\ s.

        ``page_number`` is read from the first provenance entry of a
        chunk's first doc item; a chunk's provenance list can legitimately
        be empty (e.g. a synthesized heading-only chunk), so it's left
        ``None`` rather than raised on. Conversion failures propagate as
        raised by Docling (``raises_on_error`` defaults to ``True``)
        instead of being swallowed here — the caller decides how to
        surface a bad input file.
        """
        path = Path(file_path)
        result = self._converter.convert(path)
        source_name = path.name

        chunks: list[DocumentChunk] = []
        for base_chunk in self._chunker.chunk(result.document):
            # HybridChunker.chunk() is typed to yield the generic BaseChunk,
            # but is documented to always yield DocChunk (whose DocMeta
            # carries doc_items) - a framework guarantee, not something this
            # call site needs to re-verify at runtime.
            chunk = cast(DocChunk, base_chunk)
            item = chunk.meta.doc_items[0]
            page_number = item.prov[0].page_no if item.prov else None
            chunks.append(
                DocumentChunk(text=chunk.text, source=source_name, page_number=page_number)
            )

        logger.info(
            "ingestion.document_processed",
            extra={"source": source_name, "chunk_count": len(chunks)},
        )
        return chunks
