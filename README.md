# AI Suite 🤖
> Knovos Internship Project | Tanishk | May 2026

A multi-feature AI-powered document intelligence web app built with **Flask + OpenAI GPT-4o**. Features RAG-powered Q&A, OCR, PII extraction, sentiment analysis, and document classification.

**Status:** 🚀 AI Chat feature actively developed and working with semantic search RAG

---

## Features

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 1 | 💬 **AI Chat** | ✅ Working | Semantic search RAG over PDF, TXT, DOCX with GPT-4o |
| 2 | 🔍 **OCR** | 📋 Planned | Extract text from images (PNG, JPG, JPEG) |
| 3 | 🔐 **PII Extractor** | 📋 Planned | Detect sensitive info: names, emails, SSNs, etc. |
| 4 | 📊 **Sentiment Analysis** | 📋 Planned | Sentiment, confidence, tone, reasoning |
| 5 | 📁 **Document Classifier** | 📋 Planned | Classify as Relevant / Not Relevant / Uncertain |

---

## Project Structure

```
AI-Bot/
├── ai_suite/
│   ├── app.py                      # Flask main app
│   ├── core/
│   │   ├── llm.py                  # OpenAI GPT-4o wrapper
│   │   └── rag.py                  # Semantic RAG pipeline (embeddings + retrieval)
│   ├── static/
│   │   ├── css/app.css             # Styling
│   │   └── js/app.js               # Client-side logic
│   ├── templates/
│   │   └── index.html              # UI layout
│   ├── requirements.txt            # Python dependencies
│   ├── .env                        # API keys (not committed)
│   └── .gitignore
├── .gitignore                      # Root-level gitignore
└── README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Backend | Flask |
| Frontend | HTML/CSS/JavaScript |
| LLM | OpenAI API (GPT-4o) |
| Retrieval | Semantic search (sentence-transformers all-MiniLM-L6-v2) |
| Chunking | Multi-strategy (paragraph → line → sentence → character) |
| Document Parsing | PyPDF2, python-docx |
| Env Management | python-dotenv |

---

## Setup

### 1. Clone and navigate

```bash
git clone https://github.com/Tanishk190/AI-Bot.git
cd AI-Bot/ai_suite
```

### 2. Create conda environment (recommended)

```bash
conda create -n ai-suite python=3.10
conda activate ai-suite
```

Or use venv:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

First-time run will download the sentence-transformers model (~80MB) for semantic search.

### 4. Set up API key

Create `.env` in `ai_suite/`:

```
OPENAI_API_KEY=sk-your-key-here
```

This file is in `.gitignore` — **never commit it**.

### 5. Run Flask

```bash
python app.py   
```

Visit: **http://127.0.0.1:5000**

---

## How It Works

### 💬 AI Chat (Currently Active)

**User Flow:**
1. Upload PDF, TXT, or DOCX files
2. System extracts text and chunks using smart multi-strategy splitting
3. Chunks are embedded using sentence-transformers
4. User asks a question
5. System finds semantically similar chunks (not just keyword matching)
6. GPT-4o answers based only on retrieved context

**Key Features:**
- **Semantic Search:** Understands meaning, not just keywords ("author" ≈ "writer")
- **Smart Chunking:** Paragraph → Line → Sentence → Character fallbacks
- **Retrieval Quality:** Top-3 chunks with highest semantic similarity
- **Context Window:** Only provides relevant context to GPT-4o

**Example:**
```
Q: Who wrote this?
✅ Found: "The author, John Smith, wrote..."
✅ Found: "Written by Jane Doe in 2020..."
✅ Answers accurately with semantic search
```

---

## Current Development

**Recent Changes (May 19, 2026):**
- ✅ Migrated from Hugging Face to OpenAI GPT-4o
- ✅ Implemented semantic search RAG (sentence-transformers)
- ✅ Multi-strategy document chunking
- ✅ Flask web interface with real-time indexing
- ✅ Clean error handling and fallbacks
- ✅ Updated UI ("Searching Documents..." instead of old messages)

**Git Branch:** `AI_chat`

---

## Configuration

### Environment Variables

```env
OPENAI_API_KEY=sk-...  # Your OpenAI API key
```

### Chunking Settings (core/rag.py)

```python
chunk_size = 512       # Characters per chunk
overlap = 100          # Overlap between chunks for context
```

### Retrieval Settings (core/rag.py)

```python
k = 3                  # Number of chunks to retrieve
```

---

## Troubleshooting

**"not found in documents"**
- Ensure documents are indexed first (click "Index documents")
- Semantic search may need more training data for very niche topics
- Try simpler, more specific questions

**Flask won't start**
- Check `OPENAI_API_KEY` is set in `.env`
- Verify all dependencies installed: `pip install -r requirements.txt`
- On Windows, try: `python -m flask run`

**Slow first load**
- Sentence-transformers downloads model on first use (~80MB)
- This only happens once - subsequent loads are cached

---

## Requirements

```
Flask
openai
python-docx
pypdf
Pillow
python-dotenv
PyPDF2
sentence-transformers
numpy
```

---

## Next Steps (TODO)

- [ ] Build OCR feature (EasyOCR + image upload)
- [ ] Build PII Extractor (pattern detection + GPT-4o)
- [ ] Build Sentiment Analysis (GPT-4o analysis + UI cards)
- [ ] Build Document Classifier (cosine similarity + relevance scoring)
- [ ] Add database persistence (chat history + indexed documents)
- [ ] Deploy to production (Heroku/Railway)

---

## Author

**Tanishk** — B.Tech AI & Data Science  
Built as part of internship at [Knovos](https://www.knovos.com) — AI-powered eDiscovery & legal tech

**GitHub:** https://github.com/Tanishk190/AI-Bot