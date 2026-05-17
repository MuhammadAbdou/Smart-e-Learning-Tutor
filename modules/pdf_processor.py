"""
modules/pdf_processor.py — Smart e-Learning Tutor
===================================================
Handles PDF upload, text extraction, and chunking for the RAG pipeline.
Designed to slot cleanly into the existing module architecture.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Optional heavy deps — imported lazily so the app still boots if missing
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF  (pip install pymupdf)
    _PYMUPDF_OK = True
except ImportError:
    _PYMUPDF_OK = False

try:
    from pypdf import PdfReader  # fallback  (pip install pypdf)
    _PYPDF_OK = True
except ImportError:
    _PYPDF_OK = False


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A single text chunk ready for embedding."""
    chunk_id: int
    text: str
    page: int          # 1-based page number (0 = unknown)
    source: str          # original filename
    char_start: int = 0
    char_end: int = 0


@dataclass
class ProcessedDocument:
    """Result of processing one uploaded PDF."""
    filename: str
    total_pages: int
    full_text: str
    chunks: List[DocumentChunk] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.full_text.strip())


# ── Core processor ──────────────────────────────────────────────────────────

class PDFProcessor:
    """
    Extract text from a PDF byte-stream and split it into overlapping chunks
    suitable for FAISS embedding.

    Parameters
    ----------
    chunk_size   : target character length of each chunk (default 800)
    overlap      : overlap between consecutive chunks in chars (default 150)
    min_chunk_len: discard chunks shorter than this (default 80)
    """

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 150,
        min_chunk_len: int = 80,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_len = min_chunk_len

        if not _PYMUPDF_OK and not _PYPDF_OK:
            raise ImportError(
                "Install PyMuPDF or pypdf:\n"
                "  pip install pymupdf  OR  pip install pypdf"
            )

    # ── Public API ──────────────────────────────────────────────────────────

    def process(self, pdf_bytes: bytes, filename: str = "document.pdf") -> ProcessedDocument:
        """
        Main entry point.  Accepts raw PDF bytes, returns a ProcessedDocument.
        """
        try:
            pages_text = (
                self._extract_pymupdf(pdf_bytes)
                if _PYMUPDF_OK
                else self._extract_pypdf(pdf_bytes)
            )
        except Exception as exc:
            return ProcessedDocument(
                filename=filename,
                total_pages=0,
                full_text="",
                error=f"Extraction failed: {exc}",
            )

        full_text = "\n\n".join(t for t in pages_text if t.strip())
        total_pages = len(pages_text)
        chunks = self._build_chunks(pages_text, filename)

        return ProcessedDocument(
            filename=filename,
            total_pages=total_pages,
            full_text=full_text,
            chunks=chunks,
        )

    # ── Extraction back-ends ─────────────────────────────────────────────────

    @staticmethod
    def _extract_pymupdf(pdf_bytes: bytes) -> List[str]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return [page.get_text("text") for page in doc]

    @staticmethod
    def _extract_pypdf(pdf_bytes: bytes) -> List[str]:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return [
            (page.extract_text() or "")
            for page in reader.pages
        ]

    # ── Chunking ─────────────────────────────────────────────────────────────

    def _build_chunks(
        self,
        pages_text: List[str],
        source: str,
    ) -> List[DocumentChunk]:
        """
        Split per-page text into overlapping character windows, then clean up.
        Returns a flat list of DocumentChunk objects with unique chunk_ids.
        """
        chunks: List[DocumentChunk] = []
        chunk_id = 0

        for page_num, raw_text in enumerate(pages_text, start=1):
            cleaned = self._clean_text(raw_text)
            if not cleaned:
                continue

            start = 0
            while start < len(cleaned):
                end = min(start + self.chunk_size, len(cleaned))
                text = cleaned[start:end].strip()

                if len(text) >= self.min_chunk_len:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            text=text,
                            page=page_num,
                            source=source,
                            char_start=start,
                            char_end=end,
                        )
                    )
                    chunk_id += 1

                if end >= len(cleaned):
                    break
                start = end - self.overlap   # sliding overlap

        return chunks

    # ── Text cleaning ────────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove PDF artefacts and normalise whitespace."""
        text = re.sub(r"\x00", "", text)                      # null bytes
        text = re.sub(r"[ \t]+", " ", text)                   # compress spaces
        text = re.sub(r"\n{3,}", "\n\n", text)                # excess newlines
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # de-hyphenate
        return text.strip()
