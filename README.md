# DocuMind 🤖
> Knovos Internship Project | Tanishk | May 2026

A multi-feature AI-powered document intelligence web app built with **Flask + OpenAI GPT-4o + PostgreSQL**. Features RAG-powered Q&A with multi-turn context, OCR, PII extraction, sentiment analysis, and document classification — with full database persistence and authentication.

**Status:** 🚀 All 5 features live and production-ready

---

## Features

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 1 | 💬 **AI Chat** | ✅ Working | Multi-turn RAG over PDF, TXT, DOCX — markdown rendering, source citations |
| 2 | 🔍 **OCR** | ✅ Working | Extract text from images (PNG, JPG, JPEG) via GPT-4o Vision |
| 3 | 🔐 **PII Extractor** | ✅ Working | Custom-prompt PII detection with table output + JSON export |
| 4 | 📊 **Sentiment Analysis** | ✅ Working | Sentiment, confidence, tone, reasoning — legal/eDiscovery focused |
| 5 | 📁 **Document Classifier** | ✅ Working | Classify as Relevant / Not Relevant / Uncertain with reasoning |

---

## Project Structure

```
DocuMind/
├── ai_suite/
│   ├── app.py                      # Flask app — all routes and endpoints
│   ├── schema.sql                  # PostgreSQL schema (run once to initialise)
│   ├── core/
│   │   ├── database.py             # PostgreSQL layer (connection pool, CRUD)
│   │   ├── llm.py                  # OpenAI GPT-4o wrapper (singleton client)
│   │   ├── rag.py                  # RAG pipeline — chunking, embeddings, retrieval
│   │   ├── pii.py                  # PII extraction
│   │   ├── ocr.py                  # OCR via GPT-4o Vision
│   │   └── classifier.py           # Document classifier (cosine similarity + LLM)
│   ├── static/
│   │   ├── css/app.css             # UI styles
│   │   └── js/app.js               # Client-side logic
│   ├── templates/
│   │   ├── index.html              # Main UI (5-panel layout)
│   │   └── login.html              # Password login page
│   ├── requirements.txt
│   ├── .env                        # API keys and secrets (never commit)
│   └── .gitignore
├── .gitignore
└── README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Backend | Flask |
| Frontend | HTML / CSS / JavaScript |
| LLM | OpenAI API (GPT-4o) |
| Embeddings | OpenAI text-embedding-3-small |
| Database | PostgreSQL + psycopg2 |
| Chunking | Section/heading/clause-aware + fallback splitting |
| Document Parsing | pdfplumber, PyPDF2, pypdf, python-docx |
| Env Management | python-dotenv |

---

## Prerequisites

- Python 3.10+
- PostgreSQL 16+ (running locally or remotely)
- OpenAI API key

---

## Setup

### 1. Clone and navigate

```bash
git clone https://github.com/Tanishk190/DocuMind.git
cd DocuMind/ai_suite
```

### 2. Create environment

```bash
# conda (recommended)
conda create -n documind python=3.10
conda activate documind

# or venv
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up PostgreSQL

```sql
CREATE DATABASE documind;
CREATE USER documind_user WITH PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE documind TO documind_user;
GRANT ALL ON SCHEMA public TO documind_user;
```

Then run the schema:

```bash
psql -U documind_user -d documind -f schema.sql
```

### 5. Configure environment

Create `.env` in `ai_suite/`:

```env
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://documind_user:your-password@localhost:5432/documind
APP_PASSWORD=your-app-password        # enables login gate
SECRET_KEY=your-secret-key            # Flask session key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small   # optional
```

### 6. Run

```bash
python app.py
```

Visit: **http://127.0.0.1:5000**

---

## How It Works

### 💬 AI Chat

**User Flow:**
1. Upload PDF, TXT, or DOCX via the Knowledge Base panel
2. Documents are chunked (section/heading/clause-aware), embedded, and stored in PostgreSQL
3. Ask a question — system retrieves top-3 semantically similar chunks
4. GPT-4o answers with multi-turn context (remembers last 6 messages)
5. Sources shown as citation pills under each answer

**Key Features:**
- **Multi-turn context:** Follow-up questions like "What about section 3?" work correctly
- **Markdown rendering:** Bold, lists, tables, code blocks rendered in the UI
- **Source citations:** Every answer shows the source filename and chunk number
- **Persistent history:** Chat history survives server restarts (stored in DB)
- **Selective search:** Check/uncheck documents to restrict which files are queried
- **Document deletion:** Remove indexed documents without logging out

### 🔍 OCR

- Upload PNG/JPG/JPEG images
- Text extracted via **GPT-4o Vision** (not a local model)
- Copy to clipboard or download as TXT

### 🔐 PII Extractor

- Paste text, customize the system prompt to define what to extract
- Returns structured JSON rendered as a table
- Export as JSON file or copy to clipboard

### 📊 Sentiment Analysis

- Input: direct text or uploaded document (not both)
- Truncates to 3000 chars with a warning if exceeded
- Legal-focused prompt flags hostile/adversarial tone
- Returns sentiment, confidence (0–100), tone, and reasoning

### 📁 Document Classifier

- Define relevance criteria in plain English
- Upload documents — each is scored by cosine similarity
- LLM called for mid-range scores to refine the classification
- Returns Relevant / Not Relevant / Uncertain with reasoning per file

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key |
| `DATABASE_URL` | ✅ | — | PostgreSQL connection string |
| `APP_PASSWORD` | ❌ | (no auth) | Enables password login gate |
| `SECRET_KEY` | ❌ | auto-generated | Flask session secret |
| `OPENAI_EMBEDDING_MODEL` | ❌ | `text-embedding-3-small` | Embedding model |

### Chunking defaults (`core/rag.py`)

```python
chunk_size = 600    # characters per chunk
overlap    = 110    # ~18% overlap between chunks
k          = 3      # top-k chunks retrieved per query
```

---

## Troubleshooting

**Flask won't start**
- Check `OPENAI_API_KEY` and `DATABASE_URL` are set in `.env`
- Verify PostgreSQL is running: `pg_isready -h localhost`
- Run `python -c "from app import app; print('OK')"` to catch import errors

**"No readable text found"**
- Scanned PDFs produce no text — run OCR first, then paste the result
- TXT and DOCX files are most reliable

**Chat answers from wrong documents**
- Use the Knowledge Base dropdown to uncheck irrelevant documents

**Sentiment / classifier errors**
- Documents over 3000 chars are truncated — a warning is shown in the response

**Embeddings slow on first index**
- OpenAI embeddings API is called once per chunk; subsequent indexing of the same text reuses cached embeddings from the DB

---

## Data & Privacy

All documents and text are sent to the **OpenAI API** for processing. Do not upload files you are not authorised to share with a third-party AI service. The app displays a consent banner on first use. No data is stored outside your own PostgreSQL database.

---

## Recent Changes (May 31, 2026)

- ✅ PostgreSQL persistence — all results, chat history, and embeddings stored in DB
- ✅ Password authentication with login/logout
- ✅ Multi-turn chat context (GPT-4o remembers conversation history)
- ✅ Markdown rendering in AI responses
- ✅ Source citations under every answer
- ✅ Document deletion from Knowledge Base
- ✅ Chat history restored on page refresh
- ✅ Consent banner for OpenAI data disclosure
- ✅ Scanned PDF warnings surfaced to UI
- ✅ Sentiment truncation warning in API response
- ✅ OpenAI client singleton (was re-created on every call)
- ✅ Removed unused transformers/torch from requirements

---

## Next Steps (TODO)

- [ ] User accounts with roles (admin, staff, read-only)
- [ ] Client/matter-level document organisation
- [ ] Audit log viewer in the UI
- [ ] Deploy to production (Railway / Render)
- [ ] Chat history export (JSON/CSV)

---

## Author

**Tanishk** — B.Tech AI & Data Science
Built as part of internship at [Knovos](https://www.knovos.com) — AI-powered eDiscovery & legal tech

**GitHub:** https://github.com/Tanishk190/DocuMind
