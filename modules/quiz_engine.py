"""
modules/quiz_engine.py
----------------------
LLM-powered quiz generation & answer grading.

v2 additions:
  - LectureExtractor  : extracts text from uploaded PDF / DOCX / TXT bytes
  - More question types: MCQ, True/False, Short Answer, Fill-in-the-Blank, Ordering
  - Higher default question counts
  - Richer generation prompt for more variety and depth

Classes:
  LectureExtractor — extract raw text from PDF / DOCX / TXT bytes
  QuizGenerator    — generate questions from extracted text
  QuizScorer       — validate answers and compute session score
  QuizSession      — dataclass holding a generated quiz
  SessionResult    — dataclass holding scored results
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import uuid
import requests
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Optional extraction deps ─────────────────────────────────────────────────
try:
    import fitz           # PyMuPDF
    _PYMUPDF_OK = True
except ImportError:
    _PYMUPDF_OK = False

try:
    from pypdf import PdfReader
    _PYPDF_OK = True
except ImportError:
    _PYPDF_OK = False

try:
    from docx import Document as DocxDocument
    _DOCX_OK = True
except ImportError:
    _DOCX_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedLecture:
    """Result of extracting text from an uploaded lecture file."""
    filename: str
    text: str
    page_count: int = 0
    word_count: int = 0
    file_type: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text.strip())

    @property
    def preview(self) -> str:
        """First 300 chars for UI display."""
        return self.text[:300].strip() + ("…" if len(self.text) > 300 else "")


class LectureExtractor:
    """
    Extracts plain text from uploaded lecture files.

    Supported formats:
      - PDF  (.pdf)  — via PyMuPDF (primary) or pypdf (fallback)
      - DOCX (.docx) — via python-docx
      - TXT  (.txt)  — direct decode

    Usage:
        extractor = LectureExtractor()
        result = extractor.extract(file_bytes, filename="lecture.pdf")
        if result.ok:
            text = result.text
    """

    MAX_CHARS = 40_000   # cap sent to LLM to avoid token overflow

    def extract(self, file_bytes: bytes, filename: str) -> ExtractedLecture:
        """Dispatch to the correct extractor based on file extension."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if ext == "pdf":
            return self._extract_pdf(file_bytes, filename)
        elif ext in ("docx",):
            return self._extract_docx(file_bytes, filename)
        elif ext in ("txt", "md"):
            return self._extract_txt(file_bytes, filename)
        else:
            return ExtractedLecture(
                filename=filename,
                text="",
                file_type=ext,
                error=f"Unsupported file type '.{ext}'. Please upload PDF, DOCX, or TXT.",
            )

    # ── PDF ──────────────────────────────────────────────────────────────────

    def _extract_pdf(self, file_bytes: bytes, filename: str) -> ExtractedLecture:
        if _PYMUPDF_OK:
            return self._pdf_pymupdf(file_bytes, filename)
        elif _PYPDF_OK:
            return self._pdf_pypdf(file_bytes, filename)
        else:
            return ExtractedLecture(
                filename=filename, text="", file_type="pdf",
                error="No PDF library found. Install: pip install pymupdf",
            )

    def _pdf_pymupdf(self, file_bytes: bytes, filename: str) -> ExtractedLecture:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pages = [page.get_text("text") for page in doc]
            text = self._clean("\n\n".join(pages))
            return ExtractedLecture(
                filename=filename,
                text=text[:self.MAX_CHARS],
                page_count=len(pages),
                word_count=len(text.split()),
                file_type="pdf",
            )
        except Exception as exc:
            return ExtractedLecture(filename=filename, text="", file_type="pdf",
                                    error=f"PDF extraction failed: {exc}")

    def _pdf_pypdf(self, file_bytes: bytes, filename: str) -> ExtractedLecture:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [p.extract_text() or "" for p in reader.pages]
            text = self._clean("\n\n".join(pages))
            return ExtractedLecture(
                filename=filename,
                text=text[:self.MAX_CHARS],
                page_count=len(pages),
                word_count=len(text.split()),
                file_type="pdf",
            )
        except Exception as exc:
            return ExtractedLecture(filename=filename, text="", file_type="pdf", error=f"PDF extraction failed: {exc}")

    # ── DOCX ─────────────────────────────────────────────────────────────────

    def _extract_docx(self, file_bytes: bytes, filename: str) -> ExtractedLecture:
        if not _DOCX_OK:
            return ExtractedLecture(
                filename=filename, text="", file_type="docx",
                error="python-docx not installed. Run: pip install python-docx",
            )
        try:
            doc  = DocxDocument(io.BytesIO(file_bytes))
            text = self._clean("\n".join(p.text for p in doc.paragraphs if p.text.strip()))
            return ExtractedLecture(
                filename=filename,
                text=text[:self.MAX_CHARS],
                page_count=0,          # DOCX has no reliable page count
                word_count=len(text.split()),
                file_type="docx",
            )
        except Exception as exc:
            return ExtractedLecture(filename=filename, text="", file_type="docx", error=f"DOCX extraction failed: {exc}")

    # ── TXT / MD ─────────────────────────────────────────────────────────────

    def _extract_txt(self, file_bytes: bytes, filename: str) -> ExtractedLecture:
        try:
            text = file_bytes.decode("utf-8", errors="replace")
            text = self._clean(text)
            return ExtractedLecture(
                filename=filename,
                text=text[:self.MAX_CHARS],
                word_count=len(text.split()),
                file_type="txt",
            )
        except Exception as exc:
            return ExtractedLecture(filename=filename, text="", file_type="txt", error=f"Text extraction failed: {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\x00", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  TYPES
# ══════════════════════════════════════════════════════════════════════════════

class QuestionType(str, Enum):
    MCQ = "mcq"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    FILL_BLANK = "fill_blank"
    ORDERING = "ordering"


def _normalize_qtype(raw: str) -> str:
    """
    Normalise LLM-returned q_type strings to QuestionType enum values.
    Handles: 'true/false', 'True/False', 'short answer', 'fill in the blank', etc.
    """
    s = raw.strip().lower()
    s = re.sub(r"[\s/\-]+", "_", s)
    # aliases
    aliases = {
        "fill_in_the_blank": "fill_blank",
        "fill_in_blank":     "fill_blank",
        "fillintheblank":    "fill_blank",
        "ordering":          "ordering",
        "order":             "ordering",
        "sequence":          "ordering",
        "true_false":        "true_false",
        "truefalse":         "true_false",
        "mcq":               "mcq",
        "multiple_choice":   "mcq",
        "short_answer":      "short_answer",
        "short":             "short_answer",
    }
    return aliases.get(s, s)


@dataclass
class Question:
    id:             str
    q_type:         QuestionType
    question:       str
    options:        Optional[List[str]]
    correct_answer: str
    explanation:    str
    topic:          str
    difficulty:     str
    # ordering questions also carry the scrambled sequence
    ordering_items: List[str] = field(default_factory=list)


@dataclass
class QuizSession:
    quiz_id:   str
    topic:     str
    generated: str
    questions: List[Question]
    source:    str = "manual"   # "manual" | "pdf" | "docx" | "txt"


@dataclass
class AnswerResult:
    question_id:    str
    q_type:         str
    is_correct:     bool
    score:          float
    student_answer: str
    correct_answer: str
    feedback:       str


@dataclass
class SessionResult:
    quiz_id:         str
    topic:           str
    total_questions: int
    correct:         int
    score_pct:       float
    results:         List[AnswerResult]
    completed_at:    str


# ══════════════════════════════════════════════════════════════════════════════
#  QUIZ GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

_GEN_SYSTEM = """You are an expert educational question writer for a university-level AI tutor.
Generate diverse, high-quality quiz questions from the provided lecture material.

Return ONLY a valid JSON object (no markdown, no preamble):
{
  "questions": [
    {
      "id": "q1",
      "q_type": "mcq",
      "question": "...",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "correct_answer": "A",
      "explanation": "...",
      "difficulty": "easy|medium|hard"
    },
    {
      "id": "q2",
      "q_type": "true_false",
      "question": "...",
      "options": null,
      "correct_answer": "True",
      "explanation": "...",
      "difficulty": "easy"
    },
    {
      "id": "q3",
      "q_type": "short_answer",
      "question": "...",
      "options": null,
      "correct_answer": "Ideal 1-2 sentence answer.",
      "explanation": "...",
      "difficulty": "medium"
    },
    {
      "id": "q4",
      "q_type": "fill_blank",
      "question": "The process of converting a decimal number to binary is called ______.",
      "options": null,
      "correct_answer": "binary conversion",
      "explanation": "...",
      "difficulty": "easy"
    },
    {
      "id": "q5",
      "q_type": "ordering",
      "question": "Put the following steps of the TCP handshake in the correct order:",
      "options": ["ACK", "SYN", "SYN-ACK"],
      "correct_answer": "SYN, SYN-ACK, ACK",
      "explanation": "...",
      "difficulty": "hard"
    }
  ]
}

STRICT RULES:
- q_type MUST be exactly one of: mcq, true_false, short_answer, fill_blank, ordering
- mcq        : options = ["A. ...", "B. ...", "C. ...", "D. ..."]; correct_answer = single capital letter
- true_false : options = null; correct_answer = "True" or "False"
- short_answer: options = null; correct_answer = ideal 1–2 sentence model answer
- fill_blank : options = null; question contains "______"; correct_answer = the missing word/phrase
- ordering   : options = list of items to order (scrambled); correct_answer = comma-separated correct sequence
- Vary difficulty: mix of easy, medium, hard
- Base questions STRICTLY on the provided lecture text — no outside knowledge
- Make questions test genuine understanding, not just memorisation
- Return ONLY valid JSON — no extra text"""


class QuizGenerator:
    """
    Generates quiz questions from lecture text or an ExtractedLecture.

    Default counts (higher than v1):
      MCQ=5, True/False=3, Short Answer=2, Fill-blank=2, Ordering=1
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.model = model

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_from_lecture(
        self,
        lecture: ExtractedLecture,
        n_mcq:        int = 5,
        n_tf:         int = 3,
        n_sa:         int = 2,
        n_fill:       int = 2,
        n_ordering:   int = 1,
    ) -> QuizSession:
        """
        Generate a quiz from an ExtractedLecture object (from file upload).
        Topic is inferred from the filename.
        """
        topic = lecture.filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
        session = self.generate(
            lecture_text=lecture.text,
            topic=topic,
            n_mcq=n_mcq,
            n_tf=n_tf,
            n_sa=n_sa,
            n_fill=n_fill,
            n_ordering=n_ordering,
        )
        session.source = lecture.file_type
        return session

    def generate(
        self,
        lecture_text: str,
        topic:        str = "General",
        n_mcq:        int = 5,
        n_tf:         int = 3,
        n_sa:         int = 2,
        n_fill:       int = 2,
        n_ordering:   int = 1,
    ) -> QuizSession:
        """
        Generate a quiz from raw lecture text.
        Backwards-compatible with the v1 signature (n_fill / n_ordering default to 0
        if callers don't pass them).
        """
        counts = {
            "mcq":          n_mcq,
            "true_false":   n_tf,
            "short_answer": n_sa,
            "fill_blank":   n_fill,
            "ordering":     n_ordering,
        }
        counts_desc = ", ".join(
            f"{n} {t}" for t, n in counts.items() if n > 0
        )

        prompt = (
            f'Lecture text:\n"""\n{lecture_text}\n"""\n\n'
            f"Generate exactly: {counts_desc}.\n"
            f'Omit any type with count 0. Topic label: "{topic}"\n'
            f"q_type must be one of: mcq, true_false, short_answer, fill_blank, ordering"
        )

        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 3500,
                    "messages": [
                        {"role": "system", "content": _GEN_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                },
                timeout=90,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("```").strip()
            data = json.loads(raw)

        except json.JSONDecodeError as exc:
            logger.error("QuizGenerator: invalid JSON from LLM: %s", exc)
            raise ValueError("Quiz generation returned invalid JSON. Please try again.") from exc
        except Exception as exc:
            logger.error("QuizGenerator: API call failed: %s", exc, exc_info=True)
            raise

        questions = self._parse_questions(data, topic)

        if not questions:
            raise ValueError("No valid questions were generated. Please try again.")

        return QuizSession(
            quiz_id=f"quiz_{uuid.uuid4().hex[:8]}",
            topic=topic,
            generated=datetime.now(timezone.utc).isoformat(),
            questions=questions,
            source="manual",
        )

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_questions(self, data: dict, topic: str) -> List[Question]:
        questions = []
        for q in data.get("questions", []):
            raw_type = q.get("q_type", "")
            try:
                q_type = QuestionType(_normalize_qtype(raw_type))
            except ValueError:
                logger.warning("Skipping unknown q_type %r", raw_type)
                continue

            # ordering: store scrambled items in ordering_items
            ordering_items: List[str] = []
            if q_type == QuestionType.ORDERING and q.get("options"):
                ordering_items = list(q["options"])

            questions.append(Question(
                id=q.get("id", f"q{len(questions)+1}"),
                q_type=q_type,
                question=q.get("question", ""),
                options=q.get("options") if q_type == QuestionType.MCQ else None,
                correct_answer=q.get("correct_answer", ""),
                explanation=q.get("explanation", ""),
                topic=topic,
                difficulty=q.get("difficulty", "medium"),
                ordering_items=ordering_items,
            ))
        return questions


# ══════════════════════════════════════════════════════════════════════════════
#  QUIZ SCORER
# ══════════════════════════════════════════════════════════════════════════════

_GRADE_SYSTEM = """Grade a short-answer or fill-in-the-blank quiz response.
Return ONLY JSON with no markdown or preamble:
{"is_correct": true/false, "feedback": "one sentence explanation"}"""


class QuizScorer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.model = model

    def score(self, quiz: QuizSession, answers: Dict[str, str]) -> SessionResult:
        results: List[AnswerResult] = []

        for q in quiz.questions:
            student_ans = answers.get(q.id, "").strip()

            if not student_ans:
                results.append(AnswerResult(
                    question_id=q.id, q_type=q.q_type, is_correct=False,
                    score=0.0, student_answer="(no answer)",
                    correct_answer=q.correct_answer, feedback="No answer provided.",
                ))

            elif q.q_type == QuestionType.MCQ:
                correct = student_ans.strip().upper() == q.correct_answer.strip().upper()
                results.append(AnswerResult(
                    question_id=q.id, q_type=q.q_type, is_correct=correct,
                    score=1.0 if correct else 0.0,
                    student_answer=student_ans, correct_answer=q.correct_answer,
                    feedback=(q.explanation if correct
                              else f"Correct: {q.correct_answer}. {q.explanation}"),
                ))

            elif q.q_type == QuestionType.TRUE_FALSE:
                correct = student_ans.strip().lower() == q.correct_answer.strip().lower()
                results.append(AnswerResult(
                    question_id=q.id, q_type=q.q_type, is_correct=correct,
                    score=1.0 if correct else 0.0,
                    student_answer=student_ans, correct_answer=q.correct_answer,
                    feedback=(q.explanation if correct
                              else f"Correct: {q.correct_answer}. {q.explanation}"),
                ))

            elif q.q_type == QuestionType.ORDERING:
                # Compare normalised comma-separated sequences
                def _norm(s: str) -> List[str]:
                    return [x.strip().lower() for x in s.split(",")]
                correct = _norm(student_ans) == _norm(q.correct_answer)
                results.append(AnswerResult(
                    question_id=q.id, q_type=q.q_type, is_correct=correct,
                    score=1.0 if correct else 0.0,
                    student_answer=student_ans, correct_answer=q.correct_answer,
                    feedback=(q.explanation if correct
                              else f"Correct order: {q.correct_answer}. {q.explanation}"),
                ))

            else:
                # short_answer and fill_blank — LLM semantic grading
                grade = self.grade_short(q.question, q.correct_answer, student_ans)
                results.append(AnswerResult(
                    question_id=q.id, q_type=q.q_type,
                    is_correct=grade["is_correct"],
                    score=1.0 if grade["is_correct"] else 0.0,
                    student_answer=student_ans, correct_answer=q.correct_answer,
                    feedback=grade.get("feedback", q.explanation),
                ))

        correct_count = sum(1 for r in results if r.is_correct)
        score_pct = (correct_count / len(quiz.questions)) if quiz.questions else 0.0

        return SessionResult(
            quiz_id=quiz.quiz_id, topic=quiz.topic,
            total_questions=len(quiz.questions), correct=correct_count,
            score_pct=round(score_pct, 3), results=results,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    def grade_short(self, question: str, ideal: str, student: str) -> dict:
        """LLM-grade a short-answer or fill-blank response."""
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":      self.model,
                    "max_tokens": 150,
                    "messages": [
                        {"role": "system", "content": _GRADE_SYSTEM},
                        {
                            "role": "user",
                            "content": (
                                f"Question: {question}\n"
                                f"Ideal answer: {ideal}\n"
                                f"Student answer: {student}"
                            ),
                        },
                    ],
                },
                timeout=30,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("```").strip()
            return json.loads(raw)

        except json.JSONDecodeError as exc:
            logger.warning("grade_short: JSON parse failed: %s", exc)
            return {"is_correct": False, "feedback": "Could not grade automatically."}
        except Exception as exc:
            logger.warning("grade_short: API call failed: %s", exc, exc_info=True)
            return {"is_correct": False, "feedback": "Could not grade automatically."}
