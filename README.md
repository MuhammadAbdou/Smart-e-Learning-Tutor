# 🎓 Smart e-Learning Tutor
### AI-Powered Adaptive Learning System

An intelligent AI-driven tutoring system that delivers **personalized learning, adaptive quizzes, multimodal understanding, and performance tracking** using Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and computer vision.

Built with **Python · Streamlit · OpenRouter/OpenAI API · FAISS · SQLite**

---
## 🎥 Demo

https://github.com/MuhammadAbdou/Smart-e-Learning-Tutor/raw/main/assets/demo.mp4

---

## 🚀 Features

### 💬 AI Chat Tutor
An intelligent conversational assistant that understands student intent and responds dynamically.

- Concept explanations with relatable examples
- Quiz generation on demand
- Hint-based learning (Socratic method — nudges, not answers)
- Progress analysis and encouragement
- Topic summarization
- **🌐 Live web search mode** — powered by Perplexity Sonar, answers questions beyond the model's training cutoff (recent events, latest results, current news)

Powered by an **LLM-based intent detection system** that classifies every message before responding.

---

### 📝 AI Quiz Generator
Automatically generates educational quizzes from any text, topic, or uploaded lecture file.

**Input modes:**
- 📁 Upload a lecture file (PDF, DOCX, TXT) — content is auto-extracted
- ✏️ Paste lecture text or enter a topic manually

**Question types (5 varieties):**
- Multiple Choice Questions (MCQ)
- True / False
- Short Answer
- Fill in the Blank
- Ordering / Sequencing

**Features:**
- Configurable question counts per type (up to 10 per type)
- Difficulty labels: Easy · Medium · Hard
- Fully LLM-generated with instant feedback and explanations

---

### 🧠 Smart Quiz Scoring System
Advanced evaluation engine for automatic grading.

- Exact matching for MCQ, True/False, and Ordering
- LLM-based semantic grading for Short Answer and Fill-in-the-Blank
- Instant per-question feedback with explanations
- Session score summary with mastery label

---

### 📊 Adaptive Learning Engine
A personalized AI-driven learning path system.

- Tracks student performance per topic across sessions
- Detects mastery levels: **Struggling → Developing → Proficient → Mastered**
- Identifies learning trends: **Improving · Stable · Declining**
- Dynamically recommends difficulty adjustments
- Generates personalized next-step learning paths and motivational feedback

---

### 📄 Document Q&A — RAG Mode
Upload any lecture file and chat with it using Retrieval-Augmented Generation.

**Supported formats:** PDF · DOCX · TXT · Markdown

- Extracts and chunks text automatically per format
- Embeds chunks with **OpenAI-compatible embeddings**
- Stores vectors in an in-memory **FAISS** index
- Retrieves the most relevant passages per question
- Answers grounded in the document with **cited source and page**
- Supports multiple files indexed simultaneously
- **🌐 Smart web fallback** — if the document lacks the answer, automatically searches the web via Perplexity Sonar
- **🌐 Force web mode** — toggle to always supplement answers with live internet results

---

### 🖼️ Image Q&A — Multimodal Learning
Upload diagrams, equations, lecture slides, or handwritten notes.

- Sends images to a **vision-capable LLM** (GPT-4o-mini)
- Automatically describes the image on upload
- Extracts any text via **OCR**
- Detects math/equations and diagrams with structured explanations
- Supports follow-up questions on the same image
- Quick prompts: *Explain step by step · What formula is shown? · Quiz me on this*

---

### 📚 Student Progress Tracking
Persistent analytics system backed by **SQLite**.

- Quiz attempts and scores per topic
- Weak and strong topic identification
- Visual per-topic progress bars with mastery levels
- Difficulty map with recommended next actions
- Persists across sessions via student profile

---

## 🏗️ Project Architecture

```
smart_tutor/
├── app.py                      # Main Streamlit entry point
├── requirements.txt
├── README.md
│
├── modules/
│   ├── intent_detection.py     # LLM intent classifier
│   ├── student_profile.py      # SQLite profile & progress tracking
│   ├── quiz_engine.py          # Quiz generation, file extraction & scoring
│   ├── adaptive_engine.py      # Adaptive learning recommendations
│   ├── pdf_processor.py        # PDF extraction & chunking
│   ├── image_processor.py      # Vision API & OCR
│   └── rag_engine.py           # FAISS RAG pipeline + DOCX/TXT ingestion
│
└── data/
    └── tutor.db                # SQLite student database
```

---

## 🧠 AI & ML Stack

| Component | Technology |
|---|---|
| Chat & tutoring | OpenRouter → GPT-4o-mini |
| Live web search (Chat) | Perplexity Sonar via OpenRouter |
| Intent detection | LLM classifier |
| Quiz generation | LLM prompt engineering |
| Short-answer & fill-blank grading | LLM semantic evaluation |
| Document embeddings | OpenAI `text-embedding-3-small` |
| Vector search | FAISS (IndexFlatL2) |
| RAG web fallback | Perplexity Sonar via OpenRouter |
| Image understanding | GPT-4o-mini Vision |
| OCR | Vision LLM (structured extraction) |
| Student data | SQLite |

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/MuhammadAbdou/Smart-e-Learning-Tutor.git
cd smart-e-learning-tutor

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
# Windows
set OPENAI_API_KEY=your_api_key_here

# Linux / macOS
export OPENAI_API_KEY=your_api_key_here

# 5. Run the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

> ⚠️ **After updating any module**, stop the app and restart with `streamlit run app.py` to clear the module cache.

---

## 📋 Requirements

```
Python          3.10 minimum · 3.11 recommended
streamlit       ≥ 1.35
requests        ≥ 2.31
faiss-cpu       ≥ 1.8
numpy           ≥ 1.26
pymupdf         ≥ 1.24
pypdf           ≥ 4.2
Pillow          ≥ 10.3
python-docx     ≥ 1.1
```

---

## 🖥️ Application Tabs

| Tab | Description |
|---|---|
| 💬 Chat | Conversational AI tutor with intent detection + optional live web search |
| 📝 Quiz | Generate quizzes from uploaded files or text — 5 question types |
| 📊 Progress | Dashboard with topic breakdown and adaptive recommendations |
| 📄 Documents | Upload PDF/DOCX/TXT and ask questions grounded in your study material |
| 🖼️ Image Q&A | Upload images, diagrams, or slides for visual explanation and OCR |

---

## 📈 Roadmap

- [-] AI chat tutor with intent detection
- [-] Live web search in Chat (Perplexity Sonar)
- [-] LLM quiz generation — MCQ, True/False, Short Answer
- [-] Fill-in-the-Blank and Ordering question types
- [-] Lecture file upload for quiz generation (PDF, DOCX, TXT)
- [-] Semantic short-answer and fill-blank grading
- [-] Adaptive learning engine
- [-] SQLite student progress tracking
- [-] PDF / DOCX / TXT upload + RAG document Q&A
- [-] Smart web fallback in RAG (Perplexity Sonar)
- [-] Image upload + OCR + vision Q&A
- [ ] Voice tutor interface
- [ ] Advanced analytics dashboard
- [ ] Multi-student classroom mode
- [ ] Mobile-optimised UI
- [ ] Cloud deployment (Streamlit Cloud / Railway)

---

## 👨‍💻 Author

**Mohammed Abdou**
Built with ❤️ using Python, Streamlit, and OpenAI-compatible LLMs.

---

## ⭐ Support

If you find this project useful, give it a ⭐ on GitHub — it helps a lot!
