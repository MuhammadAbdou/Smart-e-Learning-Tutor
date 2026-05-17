"""
modules/image_processor.py — Smart e-Learning Tutor
=====================================================
Handles image upload, base64 encoding, and vision-prompt construction
so the AI tutor can understand diagrams, equations, lecture slides, etc.

Designed to drop into the existing module architecture without changes.
Relies on the same OpenRouter endpoint already used by app.py.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Optional

import requests

# PIL is optional — used only for local resizing / format normalisation
try:
    from PIL import Image as PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class ImageAnalysisResult:
    """Result returned by ImageProcessor.analyse()."""
    description:  str           # full model response
    ocr_text:     str           # extracted text (may be empty)
    has_math:     bool          # heuristic: contains equations / formulas
    has_diagram:  bool          # heuristic: looks like a diagram / chart
    error:        str = ""

    @property
    def ok(self) -> bool:
        return not self.error


# ── Core processor ───────────────────────────────────────────────────────────

class ImageProcessor:
    """
    Accepts raw image bytes, sends them to a vision-capable model via
    OpenRouter, and returns a structured ImageAnalysisResult.

    Parameters
    ----------
    api_key  : OpenRouter API key (same key used by app.py)
    model    : vision-capable model string (default gpt-4o-mini which supports vision)
    max_side : resize longest side to this many pixels before sending (default 1024)
    """

    # Prompt templates
    _SYSTEM = (
        "You are an expert AI tutor. "
        "When given an image, you:\n"
        "1. Describe what you see clearly and concisely.\n"
        "2. If there is any text, transcribe it verbatim under the heading **Text found:**.\n"
        "3. If there are equations or formulas, explain them step by step.\n"
        "4. If there is a diagram, chart, or flowchart, explain its structure and meaning.\n"
        "5. Suggest how a student could use this image to study.\n"
        "Keep your response under 400 words unless the content demands more."
    )

    _QUESTION_PROMPT = (
        "A student uploaded the following image and asks:\n\n"
        "{question}\n\n"
        "Answer based on the image content. "
        "If the image contains a question or problem, solve it step by step."
    )

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        max_side: int = 1024,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_side = max_side
        self._base_url = "https://openrouter.ai/api/v1/chat/completions"

    # ── Public API ───────────────────────────────────────────────────────────

    def analyse(
        self,
        image_bytes: bytes,
        media_type: str = "image/png",
        question: Optional[str] = None,
    ) -> ImageAnalysisResult:
        """
        Send *image_bytes* to the vision model and return a structured result.

        Parameters
        ----------
        image_bytes : raw bytes of the uploaded image
        media_type  : MIME type, e.g. "image/jpeg", "image/png"
        question    : optional student question about the image
        """
        try:
            processed_bytes, media_type = self._preprocess(image_bytes, media_type)
            b64 = base64.b64encode(processed_bytes).decode("utf-8")

            user_text = (
                self._QUESTION_PROMPT.format(question=question)
                if question
                else "Please analyse this image for a student."
            )

            messages = [
                {"role": "system", "content": self._SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                },
            ]

            resp = requests.post(
                self._base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 1024,
                    "messages": messages,
                },
                timeout=45,
            )
            resp.raise_for_status()
            description = resp.json()["choices"][0]["message"]["content"].strip()

            return ImageAnalysisResult(
                description=description,
                ocr_text=self._extract_ocr_section(description),
                has_math=self._detect_math(description),
                has_diagram=self._detect_diagram(description),
            )

        except Exception as exc:
            return ImageAnalysisResult(
                description="",
                ocr_text="",
                has_math=False,
                has_diagram=False,
                error=str(exc),
            )

    # ── Pre-processing ───────────────────────────────────────────────────────

    def _preprocess(
        self,
        image_bytes: bytes,
        media_type: str,
    ) -> tuple[bytes, str]:
        """
        Optionally resize large images with PIL. Falls back to raw bytes.
        Always converts RGBA → RGB so JPEG encoding does not fail.
        """
        if not _PIL_OK:
            return image_bytes, media_type

        try:
            img = PILImage.open(io.BytesIO(image_bytes))

            # Convert palette / RGBA → RGB
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
                media_type = "image/jpeg"

            # Resize if too large
            w, h = img.size
            longest = max(w, h)
            if longest > self.max_side:
                scale = self.max_side / longest
                img = img.resize(
                    (int(w * scale), int(h * scale)),
                    PILImage.LANCZOS,
                )

            buf = io.BytesIO()
            fmt = "JPEG" if "jpeg" in media_type or "jpg" in media_type else "PNG"
            img.save(buf, format=fmt, quality=85)
            return buf.getvalue(), media_type

        except Exception:
            return image_bytes, media_type   # fallback: send raw

    # ── Heuristic helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_ocr_section(text: str) -> str:
        """Pull out text under a **Text found:** heading if present."""
        marker = "**Text found:**"
        if marker in text:
            after = text.split(marker, 1)[1]
            # Take lines until the next heading or end
            lines = []
            for line in after.splitlines():
                if line.startswith("**") and lines:
                    break
                lines.append(line)
            return "\n".join(lines).strip()
        return ""

    @staticmethod
    def _detect_math(text: str) -> bool:
        """Heuristic: does the response mention equations / formulas?"""
        math_keywords = {
            "equation", "formula", "integral", "derivative", "matrix",
            "theorem", "proof", "summation", "∑", "∫", "∂", "sqrt", "log(",
        }
        lower = text.lower()
        return any(kw in lower for kw in math_keywords)

    @staticmethod
    def _detect_diagram(text: str) -> bool:
        """Heuristic: does the response describe a diagram / chart?"""
        diagram_keywords = {
            "diagram", "chart", "flowchart", "graph", "arrow", "node",
            "axis", "bar", "pie", "table", "layout", "structure",
        }
        lower = text.lower()
        return any(kw in lower for kw in diagram_keywords)
