"""
app.py — Smart e-Learning Tutor
================================
Main entry point. Run with:  streamlit run app.py

v2 additions (architecture-compatible):
  • Tab 4 — 📄 Documents  : PDF upload + RAG-powered Q&A (FAISS)
  • Tab 5 — 🖼️ Image Q&A  : Image upload + OCR / vision understanding
  All new code lives in modules/rag_engine.py, modules/pdf_processor.py,
  and modules/image_processor.py.  No existing logic was modified.
"""

import uuid
import requests
from typing import Optional

import streamlit as st
import os

from modules.intent_detection import IntentDetector
from modules.student_profile import StudentProfile, init_db
from modules.quiz_engine import QuizGenerator, QuizScorer, QuizSession, LectureExtractor
from modules.adaptive_engine import AdaptiveEngine
from modules.pdf_processor import PDFProcessor
from modules.image_processor import ImageProcessor
from modules.rag_engine import RAGEngine

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart e-Learning Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Shared singletons ──────────────────────────────────────────────────
@st.cache_resource
def get_modules():
    init_db()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment variables")

    MODEL = "openai/gpt-4o-mini"

    return {
        # ── existing ──────────────────────────────────────────────────
        "intent":    IntentDetector(api_key=api_key, model=MODEL),
        "quiz_gen":  QuizGenerator(api_key=api_key,  model=MODEL),
        "scorer":    QuizScorer(api_key=api_key,     model=MODEL),
        "adaptive":  AdaptiveEngine(api_key=api_key, model=MODEL),
        "extractor": LectureExtractor(),
        "api_key":   api_key,
        "model":     MODEL,
        # ── new multimodal ────────────────────────────────────────────
        "pdf_proc":  PDFProcessor(),
        "img_proc":  ImageProcessor(api_key=api_key, model=MODEL),
        "rag":       RAGEngine(
            api_key=api_key,
            model=MODEL,
            enable_web_search=False,   # toggled live from the UI
            web_search_fallback=True,    # auto web search when docs lack the answer
        ),
    }


M = get_modules()

# ── CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.user-msg {
    background:#534AB7;color:#EEEDFE;
    padding:10px 14px;border-radius:16px 16px 4px 16px;
    margin:6px 0 6px 20%;font-size:14px;line-height:1.5;
}
.bot-msg {
    background:#F1EFE8;color:#2C2C2A;
    padding:10px 14px;border-radius:16px 16px 16px 4px;
    margin:6px 20% 6px 0;font-size:14px;line-height:1.5;
}
.intent-pill {
    display:inline-block;padding:2px 9px;border-radius:99px;
    font-size:11px;font-weight:500;margin-bottom:3px;
}
.prog-bar-bg  { background:#E8E8E8;border-radius:4px;
                height:10px;overflow:hidden;margin:3px 0; }
.prog-bar-fill{ height:100%;border-radius:4px; }
.topic-card{
    border:1px solid #E8E8E8;border-radius:10px;
    padding:12px 16px;margin-bottom:8px;background:#FAFAFA;
    display:flex;justify-content:space-between;align-items:center;
}
.fb-box {
    background:#EEEDFE;border-left:3px solid #534AB7;
    padding:12px 16px;border-radius:0 8px 8px 0;
    font-size:14px;line-height:1.6;color:#26215C;margin-top:8px;
}
.diff-badge {
    padding:2px 10px;border-radius:99px;
    font-size:12px;font-weight:500;
}
.rag-msg {
    background:#E8F5E9;color:#1B5E20;
    padding:10px 14px;border-radius:16px 16px 16px 4px;
    margin:6px 20% 6px 0;font-size:14px;line-height:1.5;
}
.rag-source {
    background:#F9FBE7;border-left:3px solid #8BC34A;
    padding:8px 12px;border-radius:0 6px 6px 0;
    font-size:12px;color:#33691E;margin-top:4px;
}
.img-result {
    background:#FFF3E0;border-left:3px solid #FF9800;
    padding:12px 16px;border-radius:0 8px 8px 0;
    font-size:14px;line-height:1.6;color:#4E342E;margin-top:8px;
}
.doc-badge {
    display:inline-block;padding:3px 10px;border-radius:99px;
    font-size:12px;font-weight:500;background:#E3F2FD;color:#0D47A1;
    margin:2px 3px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────
def _default_session_state():
    return {
        "student_id": str(uuid.uuid4())[:8],
        "student_name": "Unknown Student",
        "profile": None,
        "chat_history": [],
        "quiz_session": None,
        "quiz_topic": "",
        "quiz_answers": {},
        "quiz_submitted": {},
        "quiz_final": None,
        "rag_history": [],          # [{role, content, sources?}]
        "rag_doc_info": [],         # [{filename, pages, chunks}]
        "img_history": [],          # [{role, content, has_math?, has_diagram?}]
        "chat_web_search": False,   # live web search toggle for Chat tab
    }


_defaults = _default_session_state()
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def profile() -> StudentProfile:
    if "_profile_obj" not in st.session_state:
        st.session_state["_profile_obj"] = StudentProfile.get_or_create(
            st.session_state.student_id, st.session_state.student_name
        )
    return st.session_state["_profile_obj"]


def refresh_profile():
    st.session_state.pop("_profile_obj", None)
    st.session_state.profile = profile().get_progress()


if st.session_state.profile is None:
    refresh_profile()

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR  (unchanged)
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 Student Profile")
    name = st.text_input("Your name", value=st.session_state.student_name)
    if st.button("Load / Create Profile",
                 use_container_width=True, type="primary"):
        st.session_state.student_name = name
        st.session_state.pop("_profile_obj", None)
        refresh_profile()
        st.success(f"Welcome, {name}!")

    p = st.session_state.profile or {}
    if p:
        st.divider()
        overall = p.get("overall_avg_score", 0)
        level = ("🏆 Excellent" if overall >= 0.85
                 else "✅ Proficient"
                 if overall >= 0.70
                 else "📈 Developing"
                 if overall >= 0.50
                 else "⚠️ Needs support")

        c1, c2 = st.columns(2)
        c1.metric("Overall",  f"{overall:.0%}")
        c2.metric("Attempts", p.get("total_questions_answered", 0))
        st.caption(f"Level: {level}")

        weak = p.get("weak_topics", [])
        if weak:
            st.markdown("**⚠️ Weak topics**")
            for w in weak:
                st.caption(f"• {w}")

        strong = [t for t, s in p.get("topic_stats", {}).items()
                  if s["avg_score"] >= 0.70]
        if strong:
            st.markdown("**✅ Strong topics**")
            for s in strong[:3]:
                st.caption(f"• {s}")

        st.divider()
        if st.button("🔄 Refresh", use_container_width=True):
            refresh_profile()
            st.rerun()
        if st.button("🗑️ Reset session", use_container_width=True):
            fresh = _default_session_state()
            for k in ("chat_history", "quiz_session", "quiz_answers",
                      "quiz_submitted", "quiz_final", "quiz_topic",
                      "rag_history", "rag_doc_info", "img_history"):
                st.session_state[k] = fresh[k]
            M["rag"].clear()
            st.rerun()

    # ── NEW: RAG index status in sidebar ──────────────────────────────
    if st.session_state.rag_doc_info:
        st.divider()
        st.markdown("**📚 Indexed documents**")
        for d in st.session_state.rag_doc_info:
            st.caption(
                f"📄 {d['filename']} "
                f"({d['pages']}p · {d['chunks']} chunks)"
            )

# ══════════════════════════════════════════════════════════════════════
#  TABS  — two new tabs appended; existing three are untouched
# ══════════════════════════════════════════════════════════════════════
tab_chat, tab_quiz, tab_progress, tab_docs, tab_image = st.tabs(
    ["💬 Chat", "📝 Quiz", "📊 Progress", "📄 Documents", "🖼️ Image Q&A"]
)

# ─────────────────────────────────────────────────────────────────────
# CHAT TAB  (completely unchanged)
# ─────────────────────────────────────────────────────────────────────
INTENT_STYLE = {
    "explain":  ("🔵", "#E6F1FB", "#0C447C"),
    "quiz":     ("🟣", "#EEEDFE", "#3C3489"),
    "hint":     ("🟡", "#FAEEDA", "#633806"),
    "progress": ("🟢", "#EAF3DE", "#27500A"),
    "summary":  ("🟠", "#FAECE7", "#712B13"),
}

CHAT_SYSTEM = """You are a warm, expert AI tutor.
Adapt your style to the student's context.
- explain  → clear explanation + a relatable example
- quiz     → ask ONE question, wait for answer
- hint     → Socratic nudge only, NOT the full answer
- progress → summarise performance encouragingly
- summary  → concise topic recap
Keep replies focused, under 200 words unless asked for more."""

CHAT_WEB_SYSTEM = """You are a warm, expert AI tutor with access to live web search.
You can look up current events, recent results, latest news, and any information beyond your training data.
Adapt your style to the student's context.
- For factual/recent questions: search the web and give an accurate, up-to-date answer.
- explain  → clear explanation + a relatable example
- quiz     → ask ONE question, wait for answer
- hint     → Socratic nudge only, NOT the full answer
- progress → summarise performance encouragingly
- summary  → concise topic recap
Always prioritise accuracy. If you searched the web, briefly mention the source.
Keep replies focused, under 250 words unless asked for more."""


def _chat_api_call(
    api_key: str,
    model: str,
    system: str,
    history: list,
    use_web: bool,
) -> tuple[str, bool]:
    """
    Make one chat completion call.
    When use_web=True, switches to perplexity/sonar for live search.
    Returns (reply_text, web_was_used).
    """
    WEB_MODEL = "perplexity/sonar"
    chosen_model = WEB_MODEL if use_web else model
    chosen_system = CHAT_WEB_SYSTEM if use_web else system

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        json={
            "model":      chosen_model,
            "max_tokens": 600 if use_web else 512,
            "messages": [
                {"role": "system", "content": chosen_system},
                *history,
            ],
        },
        timeout=45,
    )
    response.raise_for_status()
    reply = response.json()["choices"][0]["message"]["content"].strip()
    return reply, use_web


with tab_chat:
    st.markdown("### 💬 Chat with your AI Tutor")
    st.caption("Ask anything — explain a concept, request a quiz, ask for a hint, or check your progress.")

    # ── Web search toggle ─────────────────────────────────────────────
    ws_col1, ws_col2 = st.columns([3, 1])
    chat_web_on = ws_col1.toggle(
        "🌐 Enable live web search",
        value=st.session_state.chat_web_search,
        key="chat_web_toggle",
        help=(
            "ON  — uses Perplexity Sonar to search the internet in real time.\n"
            "OFF — uses GPT-4o-mini with its training knowledge only."
        ),
    )
    st.session_state.chat_web_search = chat_web_on
    if chat_web_on:
        ws_col2.success("🌐 Web ON")
    else:
        ws_col2.info("🧠 AI only")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            intent = msg.get("intent", "explain")
            icon, bg, col = INTENT_STYLE.get(intent, ("💬", "#F1EFE8",
                                                      "#2C2C2A"))
            web_badge = " &nbsp;🌐 <small>web search</small>" if msg.get("used_web") else ""
            st.markdown(
                f'<div><span class="intent-pill"style="background:{bg};color:{col}">'
                f'{icon} {intent}</span>{web_badge}</div>'
                f'<div class="bot-msg">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("**Quick prompts:**")
    qc = st.columns(4)
    quick = ["Explain recursion",
             "Quiz me on sorting",
             "Give me a hint",
             "Show my progress"]
    for i, label in enumerate(quick):
        if qc[i].button(label, key=f"qp_{i}", use_container_width=True):
            st.session_state["_prefill"] = label

    prefill = st.session_state.pop("_prefill", "")
    user_input = st.chat_input("Type your message…") or prefill or ""

    if user_input:
        st.session_state.chat_history.append({"role": "user",
                                              "content": user_input})

        with st.spinner("Thinking…"):
            intent_result = M["intent"].detect(user_input)
            intent = intent_result.get("intent", "explain")
            topic = intent_result.get("topic")

            p = st.session_state.profile or {}
            ctx = (
                f"Student: {p.get('name', 'Student')} | "
                f"Avg: {p.get('overall_avg_score', 0):.0%} | "
                f"Weak: {', '.join(p.get('weak_topics', []) or ['none'])} | "
                f"Intent: {intent}"
            )

            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history[-12:]
            ]

            try:
                reply, web_used = _chat_api_call(
                    api_key=M["api_key"],
                    model=M["model"],
                    system=CHAT_SYSTEM + f"\n\nStudent context: {ctx}",
                    history=history,
                    use_web=st.session_state.chat_web_search,
                )
            except Exception as e:
                reply = f"⚠️ Error: {e}"
                web_used = False

        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply,
             "intent": intent, "used_web": web_used}
        )

        if topic:
            profile().mark_topic_covered(topic)
            refresh_profile()

        st.rerun()

# ─────────────────────────────────────────────────────────────────────
# QUIZ TAB  (completely unchanged)
# ─────────────────────────────────────────────────────────────────────
with tab_quiz:
    st.markdown("### 📝 Quiz Generator")

    with st.expander("⚙️ Quiz settings",
                     expanded=not bool(st.session_state.quiz_session)):

        # ── Input mode toggle ─────────────────────────────────────────
        input_mode = st.radio(
            "Content source",
            ["📁 Upload lecture file", "✏️ Paste text / topic"],
            horizontal=True,
            help="Upload a PDF or DOCX lecture file, or paste text manually.",
        )
        st.divider()

        extracted_lecture = None

        if input_mode == "📁 Upload lecture file":
            uploaded_lecture = st.file_uploader(
                "Upload lecture file",
                type=["pdf", "docx", "txt"],
                help="PDF, DOCX, or TXT — max ~40,000 characters will be used.",
                key="quiz_lecture_upload",
            )
            topic_in = st.text_input(
                "Topic label (optional — auto-detected from filename)",
                placeholder="e.g. Operating Systems",
                key="quiz_topic_upload",
            )
            if uploaded_lecture:
                with st.spinner(f"Extracting text from {uploaded_lecture.name}…"):
                    extracted_lecture = M["extractor"].extract(
                        uploaded_lecture.read(), uploaded_lecture.name
                    )
                if extracted_lecture.ok:
                    auto_topic = (topic_in.strip()
                                  or uploaded_lecture.name.rsplit(".", 1)[0]
                                      .replace("_", " ").replace("-", " ").title())
                    topic_in = auto_topic
                    st.success(
                        f"✅ Extracted **{extracted_lecture.word_count:,} words** "
                        f"from {uploaded_lecture.name}"
                    )
                    with st.expander("📄 Preview extracted text"):
                        st.caption(extracted_lecture.preview)
                else:
                    st.error(f"❌ {extracted_lecture.error}")
                    extracted_lecture = None
        else:
            topic_in = st.text_input(
                "Topic",
                placeholder="e.g. Binary Search Trees",
                key="quiz_topic_manual",
            )
            lecture_in = st.text_area(
                "Paste lecture text (optional — leave blank for a general topic quiz)",
                height=140,
                placeholder="Paste any lecture notes or leave empty…",
                key="quiz_text_manual",
            )

        st.divider()

        # ── Question count controls (5 types) ─────────────────────────
        st.markdown("**Question counts**")
        col1, col2, col3 = st.columns(3)
        col4, col5 = st.columns(2)
        n_mcq = col1.number_input("MCQ", 0, 10, 5)
        n_tf = col2.number_input("True / False", 0, 10, 3)
        n_sa = col3.number_input("Short answer", 0, 10, 2)
        n_fill = col4.number_input("Fill in the blank", 0, 10, 2)
        n_ordering = col5.number_input("Ordering", 0, 5, 1)

        total_q = n_mcq + n_tf + n_sa + n_fill + n_ordering
        st.caption(f"Total questions to generate: **{total_q}**")

        can_generate = bool(topic_in) and (
            extracted_lecture is not None
            or input_mode == "✏️ Paste text / topic"
        )

        if st.button("🚀 Generate Quiz", type="primary",
                     use_container_width=True,
                     disabled=not can_generate or total_q == 0):
            with st.spinner(f"Generating {total_q} questions…"):
                try:
                    if extracted_lecture is not None:
                        # Override topic with user input if provided
                        extracted_lecture.filename = (
                            topic_in + "." + extracted_lecture.file_type
                            if topic_in else extracted_lecture.filename
                        )
                        quiz = M["quiz_gen"].generate(
                            lecture_text=extracted_lecture.text,
                            topic=topic_in,
                            n_mcq=n_mcq,
                            n_tf=n_tf,
                            n_sa=n_sa,
                            n_fill=n_fill,
                            n_ordering=n_ordering,
                        )
                    else:
                        text = lecture_in.strip() if input_mode == "✏️ Paste text / topic" else ""
                        text = text or f"General knowledge quiz about {topic_in}."
                        quiz = M["quiz_gen"].generate(
                            lecture_text=text,
                            topic=topic_in,
                            n_mcq=n_mcq,
                            n_tf=n_tf,
                            n_sa=n_sa,
                            n_fill=n_fill,
                            n_ordering=n_ordering,
                        )
                    st.session_state.quiz_session = quiz
                    st.session_state.quiz_topic = topic_in
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = {}
                    st.session_state.quiz_final = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    quiz: Optional[QuizSession] = st.session_state.quiz_session
    if quiz:
        submitted = st.session_state.quiz_submitted
        st.markdown(f"**Topic:** {st.session_state.quiz_topic} &nbsp;|&nbsp; **Questions:** {len(quiz.questions)}")
        st.divider()

        for q in quiz.questions:
            qid = q.id
            qtype = q.q_type
            diff_icon = {"easy": "🟢",
                         "medium": "🟡", "hard": "🔴"}.get(q.difficulty, "⚪")
            type_label = {
                "mcq": "MCQ",
                "true_false": "True / False",
                "short_answer": "Short answer",
                "fill_blank": "Fill in the blank",
                "ordering": "Ordering",
            }.get(qtype, qtype)

            st.markdown(f"**{q.question}** &nbsp; `{type_label}` {diff_icon}")

            if qid in submitted:
                res = submitted[qid]
                if res["correct"]:
                    st.success(f"✅ Correct! {q.explanation}")
                else:
                    st.error(f"❌ {res.get('feedback', 'Incorrect.')} &nbsp; **Correct:** {q.correct_answer}")
            else:
                if qtype == "mcq":
                    choice = st.radio("", q.options or [], key=f"q_{qid}",
                                      index=None, label_visibility="collapsed")
                    if st.button("Submit", key=f"sub_{qid}") and choice:
                        correct = choice[0].upper() == q.correct_answer.upper()
                        submitted[qid] = {"correct": correct,
                                          "feedback": q.explanation}
                        st.session_state.quiz_submitted = submitted
                        st.rerun()

                elif qtype == "true_false":
                    choice = st.radio("", ["True", "False"], key=f"q_{qid}",
                                      index=None, label_visibility="collapsed")
                    if st.button("Submit", key=f"sub_{qid}") and choice:
                        correct = choice == q.correct_answer
                        submitted[qid] = {"correct": correct,
                                          "feedback": q.explanation}
                        st.session_state.quiz_submitted = submitted
                        st.rerun()

                elif qtype == "short_answer":
                    ans = st.text_input("Your answer:", key=f"q_{qid}")
                    if st.button("Submit", key=f"sub_{qid}") and ans.strip():
                        with st.spinner("Grading…"):
                            grade = M["scorer"].grade_short(q.question,
                                                            q.correct_answer,
                                                            ans)
                        submitted[qid] = {
                            "correct":  grade["is_correct"],
                            "feedback": grade.get("feedback", q.explanation),
                        }
                        st.session_state.quiz_submitted = submitted
                        st.rerun()

                elif qtype == "fill_blank":
                    ans = st.text_input("Fill in the blank:", key=f"q_{qid}",
                                        placeholder="Type the missing word/phrase…")
                    if st.button("Submit", key=f"sub_{qid}") and ans.strip():
                        with st.spinner("Grading…"):
                            grade = M["scorer"].grade_short(q.question,
                                                            q.correct_answer,
                                                            ans)
                        submitted[qid] = {
                            "correct":  grade["is_correct"],
                            "feedback": grade.get("feedback", q.explanation),
                        }
                        st.session_state.quiz_submitted = submitted
                        st.rerun()

                elif qtype == "ordering":
                    st.caption("Drag or type the correct order as a comma-separated list.")
                    items_display = " · ".join(q.ordering_items) if q.ordering_items else q.correct_answer
                    st.info(f"Items to order: {items_display}")
                    ans = st.text_input(
                        "Your order (comma-separated):", key=f"q_{qid}",
                        placeholder="e.g. Step A, Step B, Step C",
                    )
                    if st.button("Submit", key=f"sub_{qid}") and ans.strip():
                        def _norm(s):
                            return [x.strip().lower() for x in s.split(",")]
                        correct = _norm(ans) == _norm(q.correct_answer)
                        submitted[qid] = {
                            "correct":  correct,
                            "feedback": q.explanation,
                        }
                        st.session_state.quiz_submitted = submitted
                        st.rerun()

            st.divider()

        if len(submitted) == len(quiz.questions) and st.session_state.quiz_final is None:
            correct_count = sum(1 for r in submitted.values() if r["correct"])
            score_pct = correct_count / len(quiz.questions)
            st.session_state.quiz_final = score_pct
            profile().record_score(st.session_state.quiz_topic, score_pct)
            refresh_profile()

        if st.session_state.quiz_final is not None:
            score = st.session_state.quiz_final
            correct_count = sum(1 for r in submitted.values() if r["correct"])
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Score", f"{score:.0%}")
            mc2.metric("Correct", f"{correct_count}/{len(quiz.questions)}")
            mc3.metric("Topic", st.session_state.quiz_topic)

            if score >= 0.85:
                st.success("🏆 Excellent! Topic mastered.")
            elif score >= 0.70:
                st.info("✅ Good work — keep practising.")
            elif score >= 0.50:
                st.warning("📈 Developing — revisit the material.")
            else:
                st.error("⚠️ Struggling — review the basics.")

            if st.button("🔄 New Quiz", use_container_width=True):
                st.session_state.quiz_session = None
                st.session_state.quiz_submitted = {}
                st.session_state.quiz_final = None
                st.rerun()

# ─────────────────────────────────────────────────────────────────────
# PROGRESS TAB  (completely unchanged)
# ─────────────────────────────────────────────────────────────────────
with tab_progress:
    st.markdown("### 📊 Progress Dashboard")

    if st.button("🔄 Refresh", key="ref_prog"):
        refresh_profile()

    p = st.session_state.profile or {}
    stats = p.get("topic_stats", {})

    if not stats:
        st.info("Complete at least one quiz to see your progress here.")
    else:
        overall = p.get("overall_avg_score", 0)
        total_q = p.get("total_questions_answered", 0)
        weak = p.get("weak_topics", [])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall avg", f"{overall:.0%}")
        m2.metric("Total attempts", total_q)
        m3.metric("Topics covered", len(stats))
        m4.metric("Weak topics", len(weak))
        st.divider()

        st.markdown("#### Per-topic breakdown")
        for topic, s in sorted(stats.items(), key=lambda x: x[1]["avg_score"]):
            avg = s["avg_score"]
            pct = int(avg * 100)
            bar_color = ("#639922" if avg >= 0.85 else
                         "#378ADD" if avg >= 0.70 else
                         "#EF9F27" if avg >= 0.50 else "#E24B4A")
            mastery = ("Mastered 🏆" if avg >= 0.85 else
                       "Proficient ✅" if avg >= 0.70 else
                       "Developing 📈" if avg >= 0.50 else "Struggling ⚠️")
            cn, cb, ci = st.columns([2, 4, 2])
            cn.markdown(f"**{topic}**")
            cb.markdown(
                f'<div class="prog-bar-bg">'
                f'<div class="prog-bar-fill" style="width:{pct}%;background:{bar_color}"></div>'
                f'</div>', unsafe_allow_html=True,
            )
            ci.caption(f"{pct}% · {s['attempts']}× · {mastery}")

        st.divider()

        st.markdown("#### 🎯 Adaptive recommendations")
        if st.button("Run adaptive engine", type="primary"):
            with st.spinner("Analysing performance and generating feedback…"):
                adapt_resp = M["adaptive"].adapt(p)

            st.markdown(f'<div class="fb-box">{adapt_resp.feedback}</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f"**Priority topic:** {adapt_resp.priority_topic or '—'}&nbsp;|&nbsp; "
                f"**Recommended difficulty:** `{adapt_resp.difficulty.value}` &nbsp;|&nbsp; "
                f"**Action:** `{adapt_resp.action}`"
            )
            st.markdown("**Next steps:**")
            for step in adapt_resp.next_steps:
                st.markdown(f"• {step}")

        st.divider()
        st.markdown("#### Difficulty map")

        def _diff_style(avg: float) -> tuple:
            if avg >= 0.85:
                return "Advanced", "#26215C", "#EEEDFE"
            if avg >= 0.70:
                return "Hard", "#0C447C", "#E6F1FB"
            if avg >= 0.50:
                return "Medium", "#633806", "#FAEEDA"
            return "Beginner", "#791F1F", "#FCEBEB"

        for topic, s in sorted(stats.items(), key=lambda x: x[1]["avg_score"]):
            diff, tc, bg = _diff_style(s["avg_score"])
            action_lbl = ("Advance to next topic" if s["avg_score"] >= 0.85
                          else
                          "Quiz at higher difficulty" if s["avg_score"] >= 0.70
                          else
                          "Review & re-quiz" if s["avg_score"] >= 0.50
                          else
                          "Re-read fundamentals")
            st.markdown(
                f'<div class="topic-card">'
                f'<span style="font-size:14px"><b>{topic}</b></span>'
                f'<span class="diff-badge" style="background:{bg};color:{tc}">{diff}</span>'
                f'<span style="font-size:13px;color:#5F5E5A">{action_lbl}</span>'
                f'</div>',
                unsafe_allow_html=True,)


# ══════════════════════════════════════════════════════════════════════
#  TAB 4 — 📄 DOCUMENTS
#  PDF upload → FAISS indexing → RAG chat
# ══════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("### 📄 Document Q&A — RAG Mode")
    st.caption(
        "Upload lecture notes, textbooks, or any study PDF. "
        "The AI will read and index them so you can ask questions grounded in the text."
    )

    # ── Upload section ────────────────────────────────────────────────
    with st.expander(
        "📤 Upload PDF(s)",
        expanded=not bool(st.session_state.rag_doc_info),
    ):
        uploaded_files = st.file_uploader(
            "Choose one or more lecture files",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key="pdf_uploader",
            help="PDF, DOCX, or TXT files are chunked and embedded into a FAISS vector index.",
        )

        col_a, col_b = st.columns([3, 1])
        do_index = col_a.button(
            "📥 Index documents",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        )
        do_clear = col_b.button(
            "🗑️ Clear index",
            use_container_width=True,
            disabled=not M["rag"].is_ready,
        )

        if do_clear:
            M["rag"].clear()
            st.session_state.rag_doc_info = []
            st.session_state.rag_history = []
            st.success("Index cleared.")
            st.rerun()

        if do_index and uploaded_files:
            progress_bar = st.progress(0, text="Reading PDFs…")
            new_docs = []

            for i, uf in enumerate(uploaded_files):
                progress_bar.progress(
                    i / len(uploaded_files),
                    text=f"Processing {uf.name}…",
                )
                file_bytes = uf.read()
                ext = uf.name.rsplit(".", 1)[-1].lower() if "." in uf.name else ""

                # Use the unified index_file() — handles PDF, DOCX, TXT automatically
                with st.spinner(f"Extracting & embedding {uf.name}…"):
                    n_indexed, err = M["rag"].index_file(file_bytes, uf.name)

                if err:
                    st.error(f"❌ {uf.name}: {err}")
                    continue

                # Retrieve page count for PDF; DOCX/TXT show word count instead
                page_info = "—"
                if ext == "pdf":
                    try:
                        import fitz
                        doc_tmp = fitz.open(stream=file_bytes, filetype="pdf")
                        page_info = str(doc_tmp.page_count)
                    except Exception:
                        page_info = "?"

                new_docs.append({
                    "filename": uf.name,
                    "pages":    page_info,
                    "chunks":   n_indexed,
                    "type":     ext.upper(),
                })

            progress_bar.progress(1.0, text="Done!")

            # Merge into session state (avoid duplicates by filename)
            existing_names = {d["filename"] for d in st.session_state.rag_doc_info}
            for d in new_docs:
                if d["filename"] not in existing_names:
                    st.session_state.rag_doc_info.append(d)

            if new_docs:
                types_str = ", ".join(sorted({d["type"] for d in new_docs}))
                st.success(
                    f"✅ Indexed {len(new_docs)} file(s) [{types_str}] — "
                    f"{M['rag'].chunk_count} total chunks in the vector store."
                )
                st.rerun()

    # ── Web search toggle ─────────────────────────────────────────────
    st.divider()
    col_ws1, col_ws2 = st.columns([3, 1])
    web_search_on = col_ws1.toggle(
        "🌐 Enable live web search",
        value=False,
        help=(
            "When ON — the AI always supplements answers with live web results, "
            "bypassing the training cutoff for recent topics.\n\n"
            "When OFF — web search is used automatically only when your uploaded "
            "documents don't contain a good answer (smart fallback)."
        ),
    )
    M["rag"].enable_web_search = web_search_on
    if web_search_on:
        col_ws2.success("🌐 Web ON")
    else:
        col_ws2.info("📄 Docs only")
    st.divider()

    # ── Indexed document badges ───────────────────────────────────────
    if st.session_state.rag_doc_info:
        def _doc_icon(t: str) -> str:
            return {"PDF": "📕", "DOCX": "📘", "TXT": "📃", "MD": "📝"}.get(t.upper(), "📄")

        badges = "".join(
            f'<span class="doc-badge">{_doc_icon(d.get("type",""))}' 
            f' {d["filename"]} ({d["pages"]}p · {d["chunks"]} chunks)</span>'
            for d in st.session_state.rag_doc_info
        )
        st.markdown(badges, unsafe_allow_html=True)
        st.markdown("")

    # ── RAG chat history ──────────────────────────────────────────────
    for msg in st.session_state.rag_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-msg">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            # Show web search badge if answer used live search
            if msg.get("used_web"):
                st.caption("🌐 Answer includes live web search results")
            st.markdown(
                f'<div class="rag-msg">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            # Show FAISS source chunks
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📎 {len(sources)} document chunk(s) used"):
                    for src in sources:
                        st.markdown(
                            f'<div class="rag-source">'
                            f'<b>{src["source"]}</b> · Page {src["page"]}<br>'
                            f'<small>{src["preview"]}</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            # Show web sources
            web_sources = msg.get("web_sources", [])
            if web_sources:
                with st.expander(f"🌐 {len(web_sources)} web source(s) used"):
                    for ws in web_sources:
                        title = ws.get("title", "")   or ws.get("url", "")
                        url = ws.get("url", "")
                        snippet = ws.get("snippet", "")
                        st.markdown(
                            f'<div class="rag-source">'
                            f'<b>{title}</b><br>'
                            f'<a href="{url}" target="_blank">{url}</a><br>'
                            f'<small>{snippet}</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # ── RAG input ─────────────────────────────────────────────────────
    if not M["rag"].is_ready:
        st.info("⬆️ Upload and index at least one PDF to start asking questions.")
    else:
        st.markdown(
            f"**📚 {M['rag'].chunk_count} chunks indexed** across "
            f"{len(M['rag'].indexed_documents)} document(s)."
        )

        rag_input = st.chat_input(
            "Ask a question about your documents…",
            key="rag_chat_input",
        )

        # Quick document prompts
        rq = st.columns(3)
        doc_quick = [
            "Summarise the key concepts",
            "What are the main topics?",
            "Explain the most important idea",
        ]
        for i, label in enumerate(doc_quick):
            if rq[i].button(label, key=f"rqp_{i}", use_container_width=True):
                st.session_state["_rag_prefill"] = label

        rag_prefill = st.session_state.pop("_rag_prefill", "")
        question = rag_input or rag_prefill or ""

        if question:
            st.session_state.rag_history.append(
                {"role": "user", "content": question}
            )

            with st.spinner("Searching documents and generating answer…"):
                result = M["rag"].ask(question)

            source_snippets = [
                {
                    "source": c.source,
                    "page": c.page,
                    "preview": c.text[:220].replace("\n", " ") + "…",
                }
                for c in result.context.chunks
            ] if result.ok else []

            st.session_state.rag_history.append({
                "role": "assistant",
                "content": result.answer,
                "sources": source_snippets,
                "used_web": result.used_web,
                "web_sources": [
                    {
                        "title": ws.title,
                        "url": ws.url,
                        "snippet": ws.snippet,
                    }
                    for ws in result.web_sources
                ],
            })
            st.rerun()


# ══════════════════════════════════════════════════════════════════════
#  TAB 5 — 🖼️ IMAGE Q&A
#  Image upload + OCR + vision understanding
# ══════════════════════════════════════════════════════════════════════
with tab_image:
    st.markdown("### 🖼️ Image Q&A — Multimodal Learning")
    st.caption(
        "Upload a diagram, equation, lecture slide, or handwritten notes. "
        "The AI tutor will read, analyse, and explain it — then answer your questions."
    )

    # ── Upload section ────────────────────────────────────────────────
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        key="img_uploader",
        help="Supports PNG, JPEG, WebP, GIF. Max recommended size: 5 MB.",
    )

    if uploaded_image:
        # Preview
        st.image(uploaded_image, caption=uploaded_image.name, use_column_width=True)
        img_bytes = uploaded_image.read()
        media_type = uploaded_image.type or "image/png"

        # Store bytes in session so they survive reruns
        if st.session_state.get("_last_img_name") != uploaded_image.name:
            st.session_state["_last_img_name"] = uploaded_image.name
            st.session_state["_last_img_bytes"] = img_bytes
            st.session_state["_last_img_mime"] = media_type
            # Auto-analyse on new upload
            st.session_state["_img_auto_analyse"] = True

        # ── Auto analysis on first upload ─────────────────────────────
        if st.session_state.pop("_img_auto_analyse", False):
            with st.spinner("🔍 Analysing image…"):
                result = M["img_proc"].analyse(
                    st.session_state["_last_img_bytes"],
                    media_type=st.session_state["_last_img_mime"],
                )

            if result.ok:
                # Build a rich summary label
                tags = []
                if result.has_math:
                    tags.append("🔢 Math/Equations")
                if result.has_diagram:
                    tags.append("📊 Diagram/Chart")
                if result.ocr_text:
                    tags.append("📝 Text detected")

                tag_str = " · ".join(tags) if tags else "📷 General image"
                st.session_state.img_history.append({
                    "role": "assistant",
                    "content": result.description,
                    "tags": tag_str,
                    "ocr": result.ocr_text,
                })
            else:
                st.session_state.img_history.append({
                    "role": "assistant",
                    "content": f"⚠️ Could not analyse image: {result.error}",
                    "tags": "",
                    "ocr": "",
                })
            st.rerun()

    # ── Image chat history ────────────────────────────────────────────
    for msg in st.session_state.img_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-msg">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            tags = msg.get("tags", "")
            if tags:
                st.caption(tags)
            st.markdown(
                f'<div class="img-result">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            ocr = msg.get("ocr", "")
            if ocr:
                with st.expander("📝 Extracted text (OCR)"):
                    st.code(ocr, language=None)

    # ── Follow-up question input ──────────────────────────────────────
    has_image_loaded = bool(st.session_state.get("_last_img_bytes"))

    if not has_image_loaded:
        st.info("⬆️ Upload an image above to get started.")
    else:
        # Quick image prompts
        iq = st.columns(4)
        img_quick = [
            "Explain this step by step",
            "What formula is shown?",
            "Quiz me on this diagram",
            "Summarise this slide",
        ]
        for i, label in enumerate(img_quick):
            if iq[i].button(label, key=f"iqp_{i}", use_container_width=True):
                st.session_state["_img_prefill"] = label

        img_prefill = st.session_state.pop("_img_prefill", "")
        img_question = (
            st.chat_input("Ask about the image…", key="img_chat_input")
            or img_prefill
            or ""
        )

        if img_question:
            st.session_state.img_history.append(
                {"role": "user", "content": img_question}
            )

            with st.spinner("Analysing image and answering…"):
                result = M["img_proc"].analyse(
                    st.session_state["_last_img_bytes"],
                    media_type=st.session_state["_last_img_mime"],
                    question=img_question,
                )

            if result.ok:
                tags = []
                if result.has_math:
                    tags.append("🔢 Math/Equations")
                if result.has_diagram:
                    tags.append("📊 Diagram/Chart")
                tag_str = " · ".join(tags) if tags else ""

                st.session_state.img_history.append({
                    "role": "assistant",
                    "content": result.description,
                    "tags": tag_str,
                    "ocr": result.ocr_text,
                })
            else:
                st.session_state.img_history.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {result.error}",
                    "tags": "",
                    "ocr": "",
                })
            st.rerun()

        # Clear image conversation button
        if st.session_state.img_history:
            if st.button("🗑️ Clear image conversation", key="clear_img"):
                st.session_state.img_history = []
                st.session_state["_last_img_name"] = None
                st.session_state["_last_img_bytes"] = None
                st.session_state["_last_img_mime"] = None
                st.rerun()
