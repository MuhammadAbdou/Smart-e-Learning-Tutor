"""
modules/adaptive_engine.py
--------------------------
Rules-based adaptive learning & personalised feedback.

Classes:
  PerformanceAnalyzer  — classify mastery & trend per topic
  DifficultyAdapter    — map (mastery, trend) → difficulty
  FeedbackGenerator    — LLM-generated personalised feedback text
  AdaptiveEngine       — orchestrator: adapt(profile) → AdaptiveResponse
"""

import json
import logging
import os
import requests
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Enums ────────────────────────────────────────────────────────────────


class MasteryLevel(str, Enum):
    STRUGGLING = "struggling"
    DEVELOPING = "developing"
    PROFICIENT = "proficient"
    MASTERED = "mastered"


class Difficulty(str, Enum):
    BEGINNER = "beginner"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVANCED = "advanced"


class Trend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    NEW = "new"


T = {
    "mastered": 0.85,
    "proficient": 0.70,
    "developing": 0.50,
    "weak": 0.60,
    "reinforce": 0.65,
    "trend_up": 0.08,
    "trend_down": -0.08,
}


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class TopicAnalysis:
    topic: str
    mastery: MasteryLevel
    trend: Trend
    avg_score: float
    attempts: int
    latest_score: float
    is_weak: bool
    needs_reinforcement: bool
    recommended_difficulty: Difficulty

@dataclass
class AdaptiveResponse:
    student_name: str
    priority_topic: Optional[str]
    difficulty: Difficulty
    action: str
    action_reason: str
    encouragement: str
    weak_topics: List[str]
    next_steps: List[str]
    feedback: str


# ── PerformanceAnalyzer ───────────────────────────────────────────────────

class PerformanceAnalyzer:

    def mastery(self, avg: float) -> MasteryLevel:
        if avg >= T["mastered"]:   return MasteryLevel.MASTERED
        if avg >= T["proficient"]: return MasteryLevel.PROFICIENT
        if avg >= T["developing"]: return MasteryLevel.DEVELOPING
        return MasteryLevel.STRUGGLING

    def trend(self, latest: float, avg: float, n: int) -> Trend:
        if n < 2:
            return Trend.NEW
        d = latest - avg
        if d >= T["trend_up"]:
            return Trend.IMPROVING
        if d <= T["trend_down"]:
            return Trend.DECLINING
        return Trend.STABLE

    def analyse_topic(self, topic: str, stats: dict) -> TopicAnalysis:
        avg, latest, n = stats["avg_score"], stats["latest_score"], stats["attempts"]
        m = self.mastery(avg)
        tr = self.trend(latest, avg, n)
        return TopicAnalysis(
            topic=topic, mastery=m, trend=tr,
            avg_score=round(avg, 3), attempts=n,
            latest_score=round(latest, 3),
            is_weak=avg < T["weak"],
            needs_reinforcement=avg < T["reinforce"],
            recommended_difficulty=DifficultyAdapter.recommend(m, tr),
        )

    def analyse_all(self, profile: dict) -> Dict[str, TopicAnalysis]:
        return {
            t: self.analyse_topic(t, s)
            for t, s in profile.get("topic_stats", {}).items()
        }


# ── DifficultyAdapter ─────────────────────────────────────────────────────

class DifficultyAdapter:
    _TABLE: Dict[Tuple[MasteryLevel, Trend], Difficulty] = {
        (MasteryLevel.MASTERED,   Trend.IMPROVING):  Difficulty.ADVANCED,
        (MasteryLevel.MASTERED,   Trend.STABLE):     Difficulty.ADVANCED,
        (MasteryLevel.MASTERED,   Trend.DECLINING):  Difficulty.ADVANCED,
        (MasteryLevel.MASTERED,   Trend.NEW):        Difficulty.ADVANCED,
        (MasteryLevel.PROFICIENT, Trend.IMPROVING):  Difficulty.HARD,
        (MasteryLevel.PROFICIENT, Trend.STABLE):     Difficulty.MEDIUM,
        (MasteryLevel.PROFICIENT, Trend.NEW):        Difficulty.MEDIUM,
        (MasteryLevel.PROFICIENT, Trend.DECLINING):  Difficulty.EASY,
        (MasteryLevel.DEVELOPING, Trend.IMPROVING):  Difficulty.MEDIUM,
        (MasteryLevel.DEVELOPING, Trend.STABLE):     Difficulty.EASY,
        (MasteryLevel.DEVELOPING, Trend.NEW):        Difficulty.EASY,
        (MasteryLevel.DEVELOPING, Trend.DECLINING):  Difficulty.BEGINNER,
        (MasteryLevel.STRUGGLING, Trend.IMPROVING):  Difficulty.BEGINNER,
        (MasteryLevel.STRUGGLING, Trend.STABLE):     Difficulty.BEGINNER,
        (MasteryLevel.STRUGGLING, Trend.DECLINING):  Difficulty.BEGINNER,
        (MasteryLevel.STRUGGLING, Trend.NEW):        Difficulty.BEGINNER,
    }

    @staticmethod
    def recommend(m: MasteryLevel, tr: Trend) -> Difficulty:
        return DifficultyAdapter._TABLE.get((m, tr), Difficulty.EASY)

    @staticmethod
    def action(m: MasteryLevel, tr: Trend) -> Tuple[str, str]:
        if m == MasteryLevel.MASTERED:
            return "advance", "Student has mastered this topic."
        if m == MasteryLevel.PROFICIENT and tr == Trend.IMPROVING:
            return "quiz", "Strong upward trend — push harder."
        if m == MasteryLevel.PROFICIENT:
            return "quiz", "Solid grasp — consolidate with practice."
        if m == MasteryLevel.DEVELOPING and tr == Trend.IMPROVING:
            return "quiz", "Improving — keep momentum."
        if m == MasteryLevel.DEVELOPING:
            return "review", "Partial understanding — review first."
        if tr == Trend.IMPROVING:
            return "explain", "Struggling but improving — clearer examples needed."
        return "explain", "Struggling — simplify and step through basics."


# ── FeedbackGenerator ─────────────────────────────────────────────────────

_FB_SYSTEM = """You are a warm,
encouraging AI tutor writing brief personalised feedback.
Write 3–4 sentences: open with the student's name + encouragement,
highlight a strength,
identify the top weak area, end with one specific actionable tip.
Warm, motivating prose. No bullets. No scores or percentages."""


class FeedbackGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.model = model

    def generate(self, profile: dict, topic_analyses: Dict[str, TopicAnalysis]) -> str:
        overall = profile.get("overall_avg_score", 0)
        level = ("excellent" if overall >= T["mastered"]
                 else "good"
                 if overall >= T["proficient"]
                 else "developing" if overall >= T["developing"]
                 else "needs support")
        strong = [t for t, a in topic_analyses.items()
                  if a.mastery in (MasteryLevel.PROFICIENT,
                                   MasteryLevel.MASTERED)]
        weak = profile.get("weak_topics", [])
        ctx = {
            "student_name":  profile.get("name", "Student"),
            "overall_level": level,
            "strong_topics": strong or ["none yet"],
            "weak_topics":   weak or ["none"],
        }
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":      self.model,
                    "max_tokens": 250,
                    "messages": [
                        {"role": "system", "content": _FB_SYSTEM},
                        {"role": "user",   "content": json.dumps(ctx)},
                    ],
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("FeedbackGenerator failed: %s", exc, exc_info=True)
            return f"Keep up the great work, {profile.get('name', 'Student')}! Review your weak topics and take another quiz to track your improvement."


# ── AdaptiveEngine ────────────────────────────────────────────────────────

class AdaptiveEngine:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
    ):
        self.analyzer = PerformanceAnalyzer()
        self.feedback = FeedbackGenerator(api_key=api_key, model=model)

    def adapt(self, profile: dict) -> AdaptiveResponse:
        analyses = self.analyzer.analyse_all(profile)
        overall = profile.get("overall_avg_score", 0.0)
        weak = profile.get("weak_topics", [])

        struggling = [t for t, a in analyses.items()
                      if a.mastery == MasteryLevel.STRUGGLING]
        priority = (max(struggling, key=lambda t:
                    analyses[t].attempts) if struggling
                    else (weak[0] if weak else None))

        if priority and priority in analyses:
            ta = analyses[priority]
            diff = ta.recommended_difficulty
            action, reason = DifficultyAdapter.action(ta.mastery, ta.trend)
        else:
            m = self.analyzer.mastery(overall)
            diff = DifficultyAdapter.recommend(m, Trend.STABLE)
            action, reason = DifficultyAdapter.action(m, Trend.STABLE)

        next_steps = self._next_steps(analyses, weak)
        encourage = self._encourage(overall)
        feedback = self.feedback.generate(profile, analyses)

        return AdaptiveResponse(
            student_name=profile.get("name", "Student"),
            priority_topic=priority,
            difficulty=diff,
            action=action,
            action_reason=reason,
            encouragement=encourage,
            weak_topics=weak,
            next_steps=next_steps,
            feedback=feedback,
        )

    def _next_steps(self, analyses: Dict[str, TopicAnalysis], weak: List[str]) -> List[str]:
        steps: List[str] = []
        for t in weak[:2]:
            a = analyses.get(t)
            if a and a.needs_reinforcement:
                steps.append(f"Re-read the fundamentals of '{t}' before your next quiz.")
            else:
                steps.append(f"Take a beginner quiz on '{t}' to rebuild confidence.")
        developing = [t for t, a in analyses.items()
                      if a.mastery == MasteryLevel.DEVELOPING and t not in weak]
        if developing:
            steps.append(f"Practice medium-difficulty questions on '{developing[0]}'.")
        strong = [t for t, a in analyses.items()
                  if a.mastery in (MasteryLevel.PROFICIENT, MasteryLevel.MASTERED)]
        if strong:
            steps.append(f"Try an advanced challenge on '{strong[0]}' — you're ready.")
        return steps or ["Keep up consistent practice to reinforce your progress."]

    @staticmethod
    def _encourage(avg: float) -> str:
        if avg >= T["mastered"]:
            return "Outstanding — you are excelling!"
        if avg >= T["proficient"]:
            return "Great work — building strong skills."
        if avg >= T["developing"]:
            return "Good effort — every session moves you forward."
        return "Stay persistent — growth takes time and you are making progress."
