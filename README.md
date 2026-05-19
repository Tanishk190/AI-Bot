# AI Suite 🤖
> Knovos Internship Project | Tanishk | May 2026

A multi-feature AI-powered document intelligence web app built in Python using Streamlit. Five distinct tools accessible via sidebar navigation, all powered by OpenAI GPT-4o with RAG where applicable.

---

## Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | 💬 **AI Chat** | RAG-powered Q&A over uploaded PDF, TXT, and DOCX files |
| 2 | 🔍 **OCR** | Extract text from images (PNG, JPG, JPEG), downloadable as ZIP |
| 3 | 🔐 **PII Extractor** | Detect names, emails, phone numbers, SSNs and export as JSON |
| 4 | 📊 **Sentiment Analysis** | Returns sentiment, confidence %, tone, and GPT-4o reasoning |
| 5 | 📁 **Document Classifier** | Classifies docs as Relevant / Not Relevant / Uncertain against user-defined criteria |

---

## Project Structure

```
ai_suite/
├── app.py                         # Main entry point + sidebar navigation
├── pages/
│   ├── 1_AI_Chat.py               # Feature 1: RAG-powered chat
│   ├── 2_OCR.py                   # Feature 2: Image to text extraction
│   ├── 3_PII_Extractor.py         # Feature 3: PII extraction to JSON
│   ├── 4_Sentiment_Analysis.py    # Feature 4: Sentiment analysis
│   └── 5_Document_Classifier.py   # Feature 5: Relevance classification
├── core/
│   ├── llm.py                     # OpenAI GPT-4o wrapper
│   ├── rag.py                     # Embedding, ChromaDB storage, retrieval
│   └── ocr.py                     # EasyOCR text extraction
├── utils/
│   └── helpers.py                 # parse_llm_json(), create_zip(), save_json()
├── requirements.txt
├── .env                           # API keys — never commit this
└── .gitignore
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Frontend/UI | Streamlit (multi-page app) |
| LLM | OpenAI API (GPT-4o) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB (in-memory) |
| RAG Orchestration | LangChain + LangChain-Community |
| OCR | EasyOCR |
| Document Parsing | `pypdf`, `python-docx` |
| Similarity | `scikit-learn` cosine similarity |
| Env Management | `python-dotenv` |
| Export | `zipfile`, `json` (stdlib) |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Tanishk190/AI-Bot.git
cd AI-Bot/ai_suite
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** EasyOCR downloads ~200MB of model weights on first run.

### 4. Add your API key

Create a `.env` file in `ai_suite/`:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

This file is in `.gitignore` — never commit it.

### 5. Run the app

```bash
streamlit run app.py
```

---

## How Each Feature Works

### 💬 AI Chat
Upload PDF/TXT/DOCX files → documents are chunked (size 500, overlap 50) and embedded into ChromaDB → ask a question → top-5 relevant chunks are retrieved → GPT-4o answers using only retrieved context. Chat history and vectorstore persist across reruns via `st.session_state`.

### 🔍 OCR
Upload PNG/JPG/JPEG images → EasyOCR extracts text from each → results displayed per image → all text files bundled into a downloadable ZIP.

### 🔐 PII Extractor
Paste text → GPT-4o identifies PII entities: `name`, `email`, `phone`, `dob`, `address`, `ssn`, `organization`, `other` → structured JSON displayed via `st.json()` → downloadable as `.json`.

### 📊 Sentiment Analysis
Input text → GPT-4o returns `sentiment` (Positive/Negative/Neutral), `confidence` (0–100), `tone` (single word), and `reasoning` (1–2 sentences) → displayed as metric cards with color-coded progress bars.

### 📁 Document Classifier
Define relevance criteria (e.g. "financial fraud or contract disputes") → upload documents → each doc is embedded → cosine similarity computed against criteria embedding → GPT-4o makes final call: **Relevant** / **Not Relevant** / **Uncertain** with reasoning → results shown in a color-coded table.

---

## Core Modules

### `core/llm.py`
Thin OpenAI wrapper. All features call `get_completion(system_prompt, user_prompt)`. Client is lazily initialized once per session using the key from `.env`.

### `core/rag.py`
Full RAG pipeline:
- `load_documents(files)` — reads PDF/TXT/DOCX, splits into 500-token chunks
- `embed_and_store(chunks)` — embeds and stores in in-memory ChromaDB
- `retrieve(query, vectorstore, k=5)` — returns top-k relevant chunks
- `embed_text(text)` — single string embedding (used by Document Classifier)

### `core/ocr.py`
EasyOCR wrapper with lazy model loading to avoid re-downloading on every Streamlit rerun:
- `extract_text(image_file)` → plain string
- `extract_text_with_confidence(image_file)` → bounding boxes + confidence scores (debug mode)

### `utils/helpers.py`
Shared utilities:
- `parse_llm_json(response)` — strips markdown fences, safely parses JSON from GPT-4o responses
- `create_zip(texts)` — bundles `{filename: text}` dict into a `BytesIO` ZIP
- `save_json(data)` — serializes a dict to `BytesIO` for download

---

## Requirements

```
openai
streamlit
langchain
langchain-community
chromadb
sentence-transformers
easyocr
Pillow
python-dotenv
scikit-learn
pypdf
python-docx
```

---

## Key Design Decisions

- **API key via `.env` only** — never hardcoded anywhere in the codebase
- **`parse_llm_json()` on all LLM responses** — GPT-4o sometimes wraps JSON in markdown fences; this strips them before parsing
- **ChromaDB in-memory** — no disk persistence needed for demo/internship scope
- **Lazy model loading** — EasyOCR and sentence-transformers initialized once and reused to avoid repeated downloads
- **`st.session_state`** — persists vectorstore and chat history across Streamlit reruns

---

## Author

**Tanishk** — B.Tech AI & Data Science, ADIT (CVM University)
Built as part of internship at [Knovos](https://www.knovos.com) — AI-powered eDiscovery & legal technology.