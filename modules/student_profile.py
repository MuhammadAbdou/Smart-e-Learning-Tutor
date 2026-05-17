"""
modules/student_profile.py
--------------------------
SQLite-backed student profile & progress tracking.  (Phase 3)

Tables:
  students       — id, name, created_at, last_active
  quiz_scores    — per-attempt scores per topic
  topics_covered — topics the student has encountered
"""
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "tutor.db"
WEAK_THRESHOLD = 0.60


# ── helpers ────────────────────────────────────────────────────────────

def _now_iso() -> str:

    return datetime.now(timezone.utc).isoformat()


def _conn(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(
        path,
        check_same_thread=False,
    )
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


# ── DB init ────────────────────────────────────────────────────────────

def init_db(path: Path = DB_PATH) -> None:
    with _conn(path) as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS students (
                student_id  TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                last_active TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS quiz_scores (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT NOT NULL,
                topic       TEXT NOT NULL,
                score       REAL NOT NULL,
                recorded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS topics_covered (
                student_id TEXT NOT NULL,
                topic      TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                PRIMARY KEY (student_id, topic)
            );
        """)


# ── module-level helpers ───────────────────────────────────────────────

def list_all_students(path: Path = DB_PATH) -> List[dict]:
    init_db(path)
    with _conn(path) as c:
        rows = c.execute("SELECT * FROM students ORDER BY name").fetchall()
    return [dict(r) for r in rows]


# ── main class ─────────────────────────────────────────────────────────

class StudentProfile:
    """
    Manage one student's profile.

    Usage:
        p = StudentProfile.get_or_create("alice_01", "Alice Johnson")
        p.record_score("recursion", 0.75)
        summary = p.get_progress()
    """

    def __init__(self, student_id: str, name: str, db_path: Path = DB_PATH):
        self.student_id = student_id
        self.name = name
        self.db_path = db_path

    # ── constructors ──────────────────────────────────────────────────

    @classmethod
    def get_or_create(
        cls,
        student_id: str,
        name: str,
        db_path: Path = DB_PATH,
    ) -> "StudentProfile":
        init_db(db_path)
        now = _now_iso()
        with _conn(db_path) as c:
            row = c.execute(
                "SELECT name FROM students WHERE student_id=?", (student_id,)
            ).fetchone()
            if row is None:
                c.execute(
                    "INSERT INTO students VALUES (?,?,?,?)",
                    (student_id, name, now, now),
                )
            else:

                c.execute(
                    "UPDATE students SET name=?,last_active=? WHERE student_id=?",
                    (name, now, student_id),
                )
        return cls(student_id, name, db_path)

    # ── writes ────────────────────────────────────────────────────────

    def mark_topic_covered(self, topic: str) -> None:
        topic = topic.strip().lower()
        with _conn(self.db_path) as c:
            c.execute(
                "INSERT OR IGNORE INTO topics_covered VALUES (?,?,?)",
                (self.student_id, topic, _now_iso()),
            )

    def record_score(self, topic: str, score: float) -> None:
        """Record a quiz result. score = 0.0–1.0."""
        score = max(0.0, min(1.0, float(score)))
        topic = topic.strip().lower()
        with _conn(self.db_path) as c:
            c.execute(
                "INSERT INTO quiz_scores (student_id,topic,score,recorded_at) "
                "VALUES (?,?,?,?)",
                (self.student_id, topic, score, _now_iso()),
            )
        self.mark_topic_covered(topic)

    # ── reads ─────────────────────────────────────────────────────────

    def get_progress(self, weak_threshold: float = WEAK_THRESHOLD) -> dict:
        """
        Return a complete structured progress dict.
        Compatible with AdaptiveEngine.adapt().

        Args:
            weak_threshold:
            avg score below which a topic is considered weak (default 0.60).
        """
        with _conn(self.db_path) as c:

            student_row = c.execute(
                "SELECT * FROM students WHERE student_id=?", (self.student_id,)
            ).fetchone()

            # Per-topic aggregates (avg, count, best)
            scores = c.execute(
                "SELECT topic, AVG(score) avg, COUNT(*) cnt, MAX(score) best "
                "FROM quiz_scores WHERE student_id=? GROUP BY topic",
                (self.student_id,),
            ).fetchall()

            agg = c.execute(
                "SELECT COUNT(*) cnt, COALESCE(AVG(score), 0.0) avg "
                "FROM quiz_scores WHERE student_id=?",
                (self.student_id,),
            ).fetchone()
            total_q = agg["cnt"]
            overall = agg["avg"]

            # Latest score per topic (correlated subquery — one DB call)
            latest_rows = c.execute(
                "SELECT topic, score FROM quiz_scores "
                "WHERE student_id=? AND recorded_at=("
                "  SELECT MAX(recorded_at) FROM quiz_scores q2 "
                "  WHERE q2.student_id=quiz_scores.student_id "
                "    AND q2.topic=quiz_scores.topic"
                ")",
                (self.student_id,),
            ).fetchall()

            covered = [r["topic"] for r in c.execute(
                "SELECT topic FROM topics_covered WHERE student_id=? ORDER BY first_seen",
                (self.student_id,),
            ).fetchall()]

        student_name: str = student_row["name"] if student_row else self.name
        if student_row is None:
            logger.warning(
                "get_progress: no students row found for student_id=%r; "
                "using in-memory name %r",
                self.student_id, self.name,
            )

        latest_map = {r["topic"]: r["score"] for r in latest_rows}
        topic_stats: dict = {}
        weak: List[str] = []

        for r in scores:
            avg = round(r["avg"], 3)
            topic = r["topic"]

            latest: Optional[float] = latest_map.get(topic)
            topic_stats[topic] = {
                "avg_score": avg,
                "attempts": r["cnt"],
                "best_score": round(r["best"], 3),
                "latest_score": round(latest, 3)
                if latest is not None else avg,
                "is_weak": avg < weak_threshold,
            }
            if avg < weak_threshold:
                weak.append(topic)

        return {
            "student_id": self.student_id,
            "name": student_name,
            "topics_covered": covered,
            "weak_topics": weak,
            "total_questions_answered": total_q,
            "overall_avg_score": round(overall, 3),
            "topic_stats": topic_stats,
        }
