# DocuMind🤖
> Knovos Internship Project | Tanishk | May 2026

A multi-feature AI-powered document intelligence web app built with **Flask + OpenAI GPT-4o**. Features RAG-powered Q&A, OCR, PII extraction, sentiment analysis, and document classification.

**Status:** 🚀 AI Chat feature actively developed and working with semantic search RAG

---

## Features

| # | Feature | Status | Description |
|---|---------|--------|-------------|
| 1 | 💬 **AI Chat** | ✅ Working | Semantic search RAG over PDF, TXT, DOCX with GPT-4o |
| 2 | 🔍 **OCR** | 📋 Planned | Extract text from images (PNG, JPG, JPEG) |
| 3 | 🔐 **PII Extractor** | ✅ Working | Custom-prompt PII detection with JSON output |
| 4 | 📊 **Sentiment Analysis** | 📋 Planned | Sentiment, confidence, tone, reasoning |
| 5 | 📁 **Document Classifier** | 📋 Planned | Classify as Relevant / Not Relevant / Uncertain |

---

## Project Structure

```
AI-Bot/
├── ai_suite/
│   ├── app.py                      # Flask main app with all endpoints
│   ├── core/
│   │   ├── llm.py                  # OpenAI GPT-4o wrapper
│   │   ├── rag.py                  # Semantic RAG pipeline (embeddings + retrieval)
│   │   └── pii.py                  # PII extraction with custom prompts
│   ├── static/
│   │   ├── css/app.css             # Styling with PII editor section
│   │   └── js/app.js               # Client-side logic (chat + PII handlers)
│   ├── templates/
│   │   └── index.html              # UI layout (5 feature tabs)
│   ├── requirements.txt            # Python dependencies
│   ├── .env                        # API keys (not committed)
│   └── .gitignore
├── .gitignore                      # Root-level gitignore (venv, conda, IDE)
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

### 💬 AI Chat

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

### 🔐 PII Extractor

**User Flow:**
1. Paste text to analyze into the input textarea
2. Customize the system prompt to specify which PII fields to extract
3. Click "Extract PII" button
4. Get JSON output with all detected PII fields
5. Copy JSON or download as file

**Key Features:**
- **Customizable Prompts:** Edit system prompt to define exactly what to extract
- **JSON Output:** Structured, parseable results
- **Easy Sharing:** Copy or download extracted PII
- **Safe Processing:** Uses OpenAI API, no data persistence

**Example System Prompt:**
```
Extract PII from the text below. Return valid JSON with these fields:
You are a PII (Personally Identifiable Information) extraction specialist. Extract all personal and sensitive information from the given text. Return ONLY a valid JSON object with All the PII feilds mentioned . Each Person Should be a top level Key name Person 1 , Person 2 , etc. If a field is not found, set it to null. Be thorough and accurate.
```

---

## Current Development

**Recent Changes (May 20, 2026):**
- ✅ Built PII Extractor with editable system/user prompts
- ✅ Added JSON parsing with markdown code fence handling
- ✅ Removed token tracking feature (simplified API)
- ✅ Fixed Flask import errors and unpacking bugs
- ✅ Tested PII extraction end-to-end

**Previous Milestones:**
- ✅ Migrated from Hugging Face to OpenAI GPT-4o
- ✅ Implemented semantic search RAG (sentence-transformers)
- ✅ Multi-strategy document chunking
- ✅ Flask web interface with real-time indexing

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
- [ ] Build Sentiment Analysis (GPT-4o analysis + UI cards)
- [ ] Build Document Classifier (cosine similarity + relevance scoring)
- [ ] Add database persistence (chat history + indexed documents)
- [ ] Add chat history export (JSON/CSV)
- [ ] Deploy to production (Heroku/Railway)

---

## Author

**Tanishk** — B.Tech AI & Data Science  
Built as part of internship at [Knovos](https://www.knovos.com) — AI-powered eDiscovery & legal tech

**GitHub:** https://github.com/Tanishk190/AI-Bot
