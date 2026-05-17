"""
modules/rag_engine.py — Smart e-Learning Tutor
================================================
Retrieval-Augmented Generation (RAG) engine.

Pipeline
--------
1. Ingest lecture files (PDF / DOCX / TXT) → extract text → chunk.
2. Embed chunks with OpenAI-compatible embeddings (via OpenRouter).
3. Store vectors in an in-memory FAISS index.
4. At query time: embed the question → retrieve top-k chunks → send to LLM.

File ingestion (v4)
--------------------
index_file(bytes, filename)  — unified entry point for any supported format
  PDF  → modules/pdf_processor.PDFProcessor   (PyMuPDF / pypdf)
  DOCX → built-in DocxIngester               (python-docx)
  TXT / MD → built-in TxtIngester            (UTF-8 decode)

Web Search (v3 — fixed)
------------------------
Web search uses perplexity/sonar via OpenRouter (live internet built-in).
Triggered when:
  - enable_web_search=True        (always use web)
  - web_search_fallback=True AND FAISS scores are weak  (smart fallback)
  - No documents indexed yet      (pure web mode)

gpt-4o-mini  → doc-only answers  (fast, cheap)
perplexity/sonar → web answers   (live search, no tool parameter needed)

Dependencies:
    pip install faiss-cpu numpy pymupdf python-docx
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import requests

from modules.pdf_processor import DocumentChunk, PDFProcessor, ProcessedDocument

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import faiss
    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False

try:
    from docx import Document as DocxDocument
    _DOCX_OK = True
except ImportError:
    _DOCX_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  INTERNAL FILE INGESTERS
#  These convert raw bytes → List[DocumentChunk] so RAGEngine stays format-agnostic
# ══════════════════════════════════════════════════════════════════════════════

class _DocxIngester:
    """
    Extract text from a DOCX byte stream and split into DocumentChunk objects.
    Mirrors PDFProcessor's interface so RAGEngine can treat both identically.
    """

    MAX_CHARS = 40_000
    CHUNK_SIZE = 800
    OVERLAP = 150
    MIN_LEN = 80

    def ingest(self, docx_bytes: bytes, filename: str) -> ProcessedDocument:
        if not _DOCX_OK:
            return ProcessedDocument(
                filename=filename,
                total_pages=0,
                full_text="",
                error="python-docx not installed. Run: pip install python-docx",
            )
        try:
            doc = DocxDocument(io.BytesIO(docx_bytes))
            # Collect non-empty paragraph texts; headings become natural separators
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = self._clean("\n\n".join(paragraphs))[: self.MAX_CHARS]
            chunks = self._chunk(full_text, filename)
            return ProcessedDocument(
                filename=filename,
                total_pages=0,        # DOCX has no reliable page count
                full_text=full_text,
                chunks=chunks,
            )
        except Exception as exc:
            return ProcessedDocument(
                filename=filename, total_pages=0, full_text="",
                error=f"DOCX extraction failed: {exc}",
            )

    def _chunk(self, text: str, source: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_id = 0
        start = 0
        while start < len(text):
            end = min(start + self.CHUNK_SIZE, len(text))
            body = text[start:end].strip()
            if len(body) >= self.MIN_LEN:
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=body,
                    page=0,          
                    source=source,
                    char_start=start,
                    char_end=end,
                ))
                chunk_id += 1
            if end >= len(text):
                break
            start = end - self.OVERLAP
        return chunks

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\x00", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        return text.strip()


class _TxtIngester:
    """
    Ingest plain-text (.txt / .md) bytes into DocumentChunk objects.
    """

    MAX_CHARS = 40_000
    CHUNK_SIZE = 800
    OVERLAP = 150
    MIN_LEN = 80

    def ingest(self, txt_bytes: bytes, filename: str) -> ProcessedDocument:
        try:
            text = txt_bytes.decode("utf-8", errors="replace")
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()[: self.MAX_CHARS]
            chunks = self._chunk(text, filename)
            return ProcessedDocument(
                filename=filename,
                total_pages=0,
                full_text=text,
                chunks=chunks,
            )
        except Exception as exc:
            return ProcessedDocument(
                filename=filename, total_pages=0, full_text="",
                error=f"Text extraction failed: {exc}",
            )

    def _chunk(self, text: str, source: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_id = 0
        start = 0
        while start < len(text):
            end = min(start + self.CHUNK_SIZE, len(text))
            body = text[start:end].strip()
            if len(body) >= self.MIN_LEN:
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=body,
                    page=0,
                    source=source,
                    char_start=start,
                    char_end=end,
                ))
                chunk_id += 1
            if end >= len(text):
                break
            start = end - self.OVERLAP
        return chunks


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WebSearchResult:
    """A single web source cited in an answer."""
    title:   str
    url:     str
    snippet: str


@dataclass
class RetrievedContext:
    """Chunks returned by a FAISS similarity search."""
    chunks:      List[DocumentChunk]
    query:       str
    scores:      List[float] = field(default_factory=list)
    web_results: List[WebSearchResult] = field(default_factory=list)

    @property
    def context_text(self) -> str:
        parts = []
        for i, chunk in enumerate(self.chunks, start=1):
            label = f"Page {chunk.page}" if chunk.page else "DOCX/TXT"
            parts.append(
                f"[Source: {chunk.source}, {label}, Chunk {i}]\n"
                f"{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    @property
    def has_weak_chunks(self) -> bool:
        """
        True when FAISS returned no chunks or all L2 distances are large.
        Threshold 1.2 works well for text-embedding-3-small normalised vectors.
        """
        if not self.chunks:
            return True
        return all(s > 1.2 for s in self.scores)


@dataclass
class RAGAnswer:
    """Structured response from RAGEngine.ask()."""
    answer:      str
    context:     RetrievedContext
    used_web:    bool = False
    web_sources: List[WebSearchResult] = field(default_factory=list)
    error:       str = ""

    @property
    def ok(self) -> bool:
        return not self.error


# ══════════════════════════════════════════════════════════════════════════════
#  RAG ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# Supported extensions → friendly label
SUPPORTED_FORMATS = {
    "pdf":  "PDF",
    "docx": "DOCX",
    "txt":  "TXT",
    "md":   "Markdown",
}


class RAGEngine:
    """
    In-memory FAISS RAG engine with multi-format file ingestion
    and optional live web search via Perplexity Sonar.

    Parameters
    ----------
    api_key             : OpenRouter API key
    model               : chat model for doc-only answers (gpt-4o-mini)
    web_model           : model for web search (perplexity/sonar)
    embedding_model     : OpenAI-compatible embedding model
    top_k               : number of FAISS chunks to retrieve
    max_tokens          : max tokens in the LLM answer
    enable_web_search   : always use live web search
    web_search_fallback : auto web-search when FAISS results are weak
    """

    _DIM_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    # ── System prompts ────────────────────────────────────────────────────────

    _SYSTEM_DOCS = """You are a knowledgeable AI tutor answering questions about uploaded study material.

INSTRUCTIONS:
- Answer ONLY based on the provided context excerpts.
- If the answer is not in the context, say: "I couldn't find that in the uploaded document."
- Cite the source and page number when possible, e.g. (Page 3).
- Be clear, structured, and educational.
- If the student asks for an explanation, go step by step.
- Keep answers under 350 words unless the question requires more detail."""

    _SYSTEM_WEB = """You are a knowledgeable AI tutor with access to live web search.

INSTRUCTIONS:
- Search the web for accurate, up-to-date information to answer the question.
- Never guess or hallucinate — rely only on real search results.
- If document context is also provided, prefer it for subject-specific content
  and use web search for recent events or missing details.
- Clearly indicate when information comes from a recent web source.
- Be clear, structured, and educational.
- Keep answers under 400 words unless the question requires more detail."""

    _SYSTEM_WEB_ONLY = """You are a knowledgeable AI tutor with access to live web search.

INSTRUCTIONS:
- Search the web for accurate, up-to-date answers.
- Never make up information — rely on search results.
- Cite sources when possible.
- Be clear, structured, and educational.
- Keep answers under 400 words."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        web_model: str = "perplexity/sonar",
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 4,
        max_tokens: int = 768,
        enable_web_search: bool = False,
        web_search_fallback: bool = True,
    ) -> None:
        if not _FAISS_OK:
            raise ImportError("FAISS is required.\n  pip install faiss-cpu")

        self.api_key = api_key
        self.model = model
        self.web_model = web_model
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.enable_web_search = enable_web_search
        self.web_search_fallback = web_search_fallback

        self._base_url = "https://openrouter.ai/api/v1"
        self._embed_url = f"{self._base_url}/embeddings"
        self._chat_url = f"{self._base_url}/chat/completions"

        # FAISS state
        self._index:     Optional[faiss.IndexFlatL2] = None
        self._chunks:    List[DocumentChunk] = []
        self._dim:       int = self._DIM_MAP.get(embedding_model, 1536)
        self._doc_names: List[str] = []

        # Ingesters
        self._pdf_proc = PDFProcessor()
        self._docx_ingester = _DocxIngester()
        self._txt_ingester = _TxtIngester()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    @property
    def indexed_documents(self) -> List[str]:
        return list(self._doc_names)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    # ── File ingestion (NEW unified entry point) ──────────────────────────────

    def index_file(self, file_bytes: bytes, filename: str) -> Tuple[int, str]:
        """
        Unified entry point for indexing any supported lecture file.

        Accepts raw bytes + filename, auto-detects format, extracts text,
        chunks it, embeds the chunks, and adds them to the FAISS index.

        Returns
        -------
        (n_chunks_indexed, error_message)
        error_message is "" on success, non-empty on failure.

        Supported formats: PDF, DOCX, TXT, MD
        """
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if ext not in SUPPORTED_FORMATS:
            return 0, (
                f"Unsupported format '.{ext}'. "
                f"Please upload: {', '.join(SUPPORTED_FORMATS.keys()).upper()}"
            )

        # ── Route to correct ingester ─────────────────────────────────────────
        if ext == "pdf":
            doc = self._pdf_proc.process(file_bytes, filename=filename)
        elif ext == "docx":
            doc = self._docx_ingester.ingest(file_bytes, filename)
        else:  # txt / md
            doc = self._txt_ingester.ingest(file_bytes, filename)

        if not doc.ok:
            return 0, doc.error

        n_indexed = self.index_document(doc)
        return n_indexed, ""

    def index_document(self, doc: ProcessedDocument) -> int:
        """
        Embed all chunks in a ProcessedDocument and add to FAISS.
        Returns number of chunks indexed. (kept for backward compatibility)
        """
        if not doc.ok:
            return 0

        texts = [c.text for c in doc.chunks]
        vectors = self._embed_batch(texts)
        if not vectors:
            return 0

        matrix = np.array(vectors, dtype="float32")
        if self._index is None:
            self._index = faiss.IndexFlatL2(self._dim)

        self._index.add(matrix)
        self._chunks.extend(doc.chunks)

        if doc.filename not in self._doc_names:
            self._doc_names.append(doc.filename)

        return len(vectors)

    def ask(self, question: str) -> RAGAnswer:
        """
        Main entry point.

        Decision logic:
        ┌──────────────────────────────────────────────────────────────┐
        │ No docs indexed     →  Perplexity Sonar (pure web)           │
        │ Docs indexed        →  FAISS retrieve                        │
        │   weak OR web=ON    →  Perplexity Sonar + doc context        │
        │   good AND web=OFF  →  gpt-4o-mini + doc context only        │
        └──────────────────────────────────────────────────────────────┘
        """
        # ── Pure web (no docs) ────────────────────────────────────────────────
        if not self.is_ready:
            try:
                answer, web_sources = self._call_web_model(
                    system=self._SYSTEM_WEB_ONLY,
                    user_msg=question,
                )
                return RAGAnswer(
                    answer=answer,
                    context=RetrievedContext(chunks=[], query=question),
                    used_web=True,
                    web_sources=web_sources,
                )
            except Exception as exc:
                return RAGAnswer(
                    answer=f"⚠️ Web search failed: {exc}",
                    context=RetrievedContext(chunks=[], query=question),
                    error=str(exc),
                )

        # ── FAISS retrieval ───────────────────────────────────────────────────
        try:
            context = self._retrieve(question)
        except Exception as exc:
            return RAGAnswer(
                answer=f"⚠️ Retrieval error: {exc}",
                context=RetrievedContext(chunks=[], query=question),
                error=str(exc),
            )

        # ── Route: web supplement or docs only ────────────────────────────────
        use_web = (
            self.enable_web_search
            or (self.web_search_fallback and context.has_weak_chunks)
        )

        try:
            if use_web:
                if context.chunks and not context.has_weak_chunks:
                    user_msg = (
                        f"CONTEXT FROM UPLOADED STUDY MATERIAL:\n\n"
                        f"{context.context_text}\n\n"
                        f"---\n\n"
                        f"Also search the web for anything not covered above.\n\n"
                        f"STUDENT QUESTION: {question}"
                    )
                else:
                    user_msg = (
                        f"The uploaded documents don't contain enough information "
                        f"about this topic.\n\n"
                        f"STUDENT QUESTION: {question}"
                    )
                answer, web_sources = self._call_web_model(
                    system=self._SYSTEM_WEB,
                    user_msg=user_msg,
                )
                return RAGAnswer(
                    answer=answer,
                    context=context,
                    used_web=True,
                    web_sources=web_sources,
                )
            else:
                answer = self._call_doc_model(question, context)
                return RAGAnswer(answer=answer, context=context)

        except Exception as exc:
            return RAGAnswer(
                answer=f"⚠️ Generation error: {exc}",
                context=context,
                error=str(exc),
            )

    def clear(self) -> None:
        """Reset the FAISS index — called on session reset."""
        self._index = None
        self._chunks = []
        self._doc_names = []

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        retry: int = 2,
    ) -> List[List[float]]:
        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            for attempt in range(retry + 1):
                try:
                    resp = requests.post(
                        self._embed_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type":  "application/json",
                        },
                        json={"model": self.embedding_model, "input": batch},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    all_vectors.extend(
                        item["embedding"] for item in resp.json()["data"]
                    )
                    break
                except Exception:
                    if attempt == retry:
                        raise
                    time.sleep(1.5 ** attempt)
        return all_vectors

    def _embed_one(self, text: str) -> np.ndarray:
        vec = self._embed_batch([text])[0]
        return np.array([vec], dtype="float32")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve(self, question: str) -> RetrievedContext:
        q_vec = self._embed_one(question)
        k = min(self.top_k, len(self._chunks))

        distances, indices = self._index.search(q_vec, k)
        distances = distances[0].tolist()
        indices = indices[0].tolist()

        chunks = [self._chunks[idx] for idx in indices if 0 <= idx < len(self._chunks)]
        scores = [distances[i] for i in range(len(chunks))]

        return RetrievedContext(chunks=chunks, query=question, scores=scores)

    # ── Generation — docs only (gpt-4o-mini) ─────────────────────────────────

    def _call_doc_model(self, question: str, context: RetrievedContext) -> str:
        user_msg = (
            f"CONTEXT FROM STUDY MATERIAL:\n\n"
            f"{context.context_text}\n\n"
            f"---\n\n"
            f"STUDENT QUESTION: {question}"
        )
        resp = requests.post(
            self._chat_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": self._SYSTEM_DOCS},
                    {"role": "user",   "content": user_msg},
                ],
            },
            timeout=45,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    # ── Generation — web (perplexity/sonar) ──────────────────────────────────

    def _call_web_model(
        self,
        system: str,
        user_msg: str,
    ) -> Tuple[str, List[WebSearchResult]]:
        """
        Perplexity Sonar has live internet search built-in.
        No special tool parameter — just send the message.
        """
        resp = requests.post(
            self._chat_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model": self.web_model,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_msg},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"].strip()

        web_sources: List[WebSearchResult] = []
        for c in data.get("citations", []):
            if isinstance(c, dict):
                web_sources.append(WebSearchResult(
                    title=c.get("title",   ""),
                    url=c.get("url",     ""),
                    snippet=c.get("snippet", ""),
                ))
            elif isinstance(c, str):
                web_sources.append(WebSearchResult(title="", url=c, snippet=""))

        return answer, web_sources
