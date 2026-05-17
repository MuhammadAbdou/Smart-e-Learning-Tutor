"""
smart_tutor.modules
-------------------
All AI logic modules for the Smart e-Learning Tutor.
"""

from .intent_detection import IntentDetector

from .student_profile import StudentProfile, init_db, list_all_students

from .quiz_engine import (
    QuizGenerator,
    QuizScorer,
    QuizSession,
    Question,
    SessionResult,
    AnswerResult,
    QuestionType,
)

from .adaptive_engine import (
    AdaptiveEngine,
    AdaptiveResponse,
    TopicAnalysis,
    MasteryLevel,
    Difficulty,
    Trend,
)

__all__ = [
    # intent
    "IntentDetector",

    # profile
    "StudentProfile",
    "init_db",
    "list_all_students",

    # quiz
    "QuizGenerator",
    "QuizScorer",
    "QuizSession",
    "Question",
    "SessionResult",
    "AnswerResult",
    "QuestionType",

    # adaptive
    "AdaptiveEngine",
    "AdaptiveResponse",
    "TopicAnalysis",
    "MasteryLevel",
    "Difficulty",
    "Trend",
]
