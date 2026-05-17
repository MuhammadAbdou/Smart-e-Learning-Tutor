"""
modules/intent_detection.py
---------------------------
LLM-based intent classifier.

Classifies student input into:
  explain | quiz | hint | progress | summary

Returns: { "intent": str, "confidence": float, "topic": str | None }
"""
import json
import logging
import os
import requests
from typing import Optional

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VALID_INTENTS = {"explain", "quiz", "hint", "progress", "summary"}
_FALLBACK = {"intent": "explain", "confidence": 0.5, "topic": None}

SYSTEM_PROMPT = """You are an intent classifier for an AI tutoring chatbot.

Return ONLY a JSON object:
{
  "intent": "explain | quiz | hint | progress | summary",
  "confidence": 0.0-1.0,
  "topic": string or null
}

Rules:
- Return only JSON
- No markdown
- No explanation
"""


class IntentDetector:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is missing")
        self.model = model

    def detect(self, message: str) -> dict:
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 100,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": message},
                    ],
                },
                timeout=30,
            )
            response.raise_for_status()
            result = json.loads(response.json()["choices"][0]["message"]["content"])

        except Exception as exc:
            logger.warning("IntentDetector API call failed: %s",
                           exc, exc_info=True)
            return dict(_FALLBACK)

        # Validate & normalise
        raw_conf = result.get("confidence", 0.5)
        try:
            confidence = round(float(raw_conf), 2)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        raw_topic = result.get("topic")
        topic: Optional[str] = str(raw_topic) if isinstance(raw_topic, str) else None

        intent = result.get("intent")
        if intent not in VALID_INTENTS:
            logger.warning("Unknown intent %r;defaulting to 'explain'",
                           intent)
            intent = "explain"
            confidence = 0.5
        return {"intent": intent, "confidence": confidence, "topic": topic}
