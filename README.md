# DocuMind 
> Knovos Internship Project | Tanishk | May 2026

---

## Project Overview

A multi-feature AI-powered document intelligence web application built in Python using Streamlit. The app has 5 distinct features accessible via a sidebar navigation, all powered by OpenAI API (GPT-4o) with RAG (Retrieval-Augmented Generation) where applicable.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Frontend/UI | Streamlit (multi-page app) |
| LLM | OpenAI API (GPT-4o) |
| RAG - Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| RAG - Vector Store | ChromaDB |
| RAG - Orchestration | LangChain + LangChain-Community |
| OCR | EasyOCR or pytesseract + Pillow |
| Env Management | python-dotenv |
| Export | zipfile, json (stdlib) |

---

## Project Structure

```
ai_suite/
├── app.py                        # Main entry point + sidebar navigation
├── pages/
│   ├── 1_AI_Chat.py              # Feature 1: RAG-powered chat
│   ├── 2_OCR.py                  # Feature 2: Image to text extraction
│   ├── 3_PII_Extractor.py        # Feature 3: PII extraction to JSON
│   ├── 4_Sentiment_Analysis.py   # Feature 4: Sentiment analysis
│   └── 5_Document_Classifier.py  # Feature 5: Relevance classification
├── core/
│   ├── llm.py                    # OpenAI API wrapper
│   ├── rag.py                    # RAG pipeline (embed, store, retrieve)
│   └── ocr.py                    # OCR logic (EasyOCR/pytesseract)
├── utils/
│   └── helpers.py                # JSON export, zip creation, file utils
├── requirements.txt
├── .env                          # API keys (never commit this)
└── .gitignore
```

---

## Environment Variables (.env)

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

---

## Features — Detailed Specification

---

### Feature 1: AI Chat (`pages/1_AI_Chat.py`)

**Purpose:** RAG-powered conversational AI over uploaded documents.

**User Flow:**
1. User uploads one or more documents (PDF, TXT, DOCX)
2. Documents are chunked, embedded, and stored in ChromaDB
3. User types a question in the chat input
4. System retrieves top-k relevant chunks from ChromaDB
5. Retrieved chunks + user question are sent to GPT-4o
6. Response is shown in chat UI with history maintained via `st.session_state`

**Key Implementation Details:**
- Use `st.session_state.messages` for chat history
- Use `st.session_state.vectorstore` to persist ChromaDB across reruns
- Chunking: chunk size 500, overlap 50 (LangChain `RecursiveCharacterTextSplitter`)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Top-k retrieval: k=3 or k=5
- System prompt: instruct GPT-4o to answer only from retrieved context

**UI Elements:**
- File uploader (multi-file, accept pdf/txt/docx)
- Chat message history (user bubbles + AI bubbles)
- Text input at bottom
- "Clear chat" button

---

### Feature 2: OCR (`pages/2_OCR.py`)

**Purpose:** Extract text from uploaded images using OCR.

**User Flow:**
1. User uploads one or more images (PNG, JPG, JPEG)
2. OCR runs on each image and extracts text
3. Extracted text is displayed in a text area
4. User can copy text to clipboard OR download all text as a ZIP file

**Key Implementation Details:**
- Use EasyOCR (`easyocr.Reader(['en'])`) — preferred over pytesseract (no system install needed)
- If multiple images uploaded, extract each separately and combine or zip
- ZIP creation: use Python `zipfile` stdlib, one `.txt` file per image inside the ZIP
- Copy to clipboard: use `st.code()` with copy button or `pyperclip` (note: may not work in all browsers via Streamlit)

**UI Elements:**
- Multi-image file uploader
- Preview of uploaded images (`st.image()`)
- Extracted text display (`st.text_area()`)
- Two buttons: "Copy Text" and "Download as ZIP"
- `st.download_button()` for ZIP

---

### Feature 3: PII Extractor (`pages/3_PII_Extractor.py`)

**Purpose:** Extract Personally Identifiable Information from a text paragraph and export as JSON.

**PII Fields to Extract:**
- `name` — full name
- `email` — email address
- `phone` — phone number
- `dob` — date of birth
- `address` — physical address
- `ssn` — social security number (if present)
- `organization` — company/org name
- `other` — any other sensitive identifiers

**User Flow:**
1. User pastes a paragraph into a text area
2. Clicks "Extract PII"
3. GPT-4o is called with a structured prompt instructing it to return only valid JSON
4. JSON is parsed and displayed in a formatted code block
5. User can download the JSON file

**Key Implementation Details:**
- System prompt: "You are a PII extraction assistant. Extract all personally identifiable information from the given text. Return ONLY a valid JSON object with the fields: name, email, phone, dob, address, ssn, organization, other. If a field is not found, set it to null."
- Parse response: `json.loads(response)` — strip markdown fences if present
- Use `st.json()` to display output
- `st.download_button()` with `mime="application/json"`

**UI Elements:**
- Text area for input paragraph
- "Extract PII" button
- JSON output display (`st.json()`)
- Download JSON button

---

### Feature 4: Sentiment Analysis (`pages/4_Sentiment_Analysis.py`)

**Purpose:** Analyze the sentiment of input text using GPT-4o.

**Output Fields:**
- `sentiment` — Positive / Negative / Neutral
- `confidence` — percentage (0–100)
- `tone` — e.g. Enthusiastic, Frustrated, Calm, Sarcastic
- `reasoning` — 1–2 sentence explanation from the model

**User Flow:**
1. User pastes or types text into a text area
2. Clicks "Analyze"
3. GPT-4o returns structured JSON with the above fields
4. Results are displayed as metric cards + progress bars

**Key Implementation Details:**
- System prompt: "You are a sentiment analysis assistant. Analyze the sentiment of the given text. Return ONLY a valid JSON object with fields: sentiment (Positive/Negative/Neutral), confidence (integer 0-100), tone (single word descriptor), reasoning (1-2 sentences)."
- Use `st.metric()` for sentiment label and confidence
- Use `st.progress()` for visual confidence bar
- Color code: Positive = green, Negative = red, Neutral = gray

**UI Elements:**
- Text area for input
- "Analyze" button
- Metric cards: Sentiment, Confidence, Tone
- Progress bars for Positive / Neutral / Negative scores
- Reasoning text block

---

### Feature 5: Document Classifier (`pages/5_Document_Classifier.py`)

**Purpose:** Classify uploaded documents as Relevant or Not Relevant based on user-defined criteria using RAG + LLM.

**User Flow:**
1. User types their relevance criteria (e.g. "financial fraud or contract disputes")
2. User uploads multiple documents (PDF, TXT, DOCX)
3. Each document's content is extracted and embedded
4. Criteria is also embedded
5. Cosine similarity computed between criteria embedding and each document
6. GPT-4o makes final classification decision with reasoning for each doc
7. Results shown in a table with Relevant / Not Relevant / Uncertain tags

**Key Implementation Details:**
- Embed the criteria string using same `sentence-transformers` model
- Embed each document (full text or first 1000 tokens)
- Compute cosine similarity (use `sklearn.metrics.pairwise.cosine_similarity`)
- Send top-similarity chunks + criteria to GPT-4o for final classification
- System prompt: "Given the relevance criteria and the document content, classify the document as Relevant, Not Relevant, or Uncertain. Return ONLY a valid JSON with: classification (Relevant/Not Relevant/Uncertain), confidence (0-100), reasoning (1 sentence)."
- Display results in `st.dataframe()` or custom card list

**UI Elements:**
- Text input for relevance criteria
- Multi-file uploader
- "Classify Documents" button
- Results table with filename, classification badge, confidence, reasoning
- Color coded tags: green = Relevant, red = Not Relevant, gray = Uncertain

---

## Core Modules — Specification

---

### `core/llm.py`

```python
# Responsibilities:
# - Initialize OpenAI client
# - Single function: get_completion(system_prompt, user_prompt) -> str
# - Handle errors gracefully
# - Load API key from .env
```

---

### `core/rag.py`

```python
# Responsibilities:
# - load_documents(files) -> list of text chunks
# - embed_and_store(chunks) -> ChromaDB vectorstore
# - retrieve(query, vectorstore, k=5) -> list of relevant chunks
# - Uses: sentence-transformers, ChromaDB, LangChain splitter
```

---

### `core/ocr.py`

```python
# Responsibilities:
# - extract_text(image_file) -> str
# - Uses EasyOCR
# - Returns extracted text as string
```

---

### `utils/helpers.py`

```python
# Responsibilities:
# - create_zip(texts: dict) -> BytesIO  (filename -> text content)
# - save_json(data: dict) -> BytesIO
# - parse_llm_json(response: str) -> dict  (strips markdown fences, parses JSON)
```

---

## Build Order

1. `core/llm.py` — OpenAI wrapper (foundation)
2. `core/rag.py` — RAG pipeline
3. `core/ocr.py` — OCR logic
4. `utils/helpers.py` — shared utilities
5. `pages/1_AI_Chat.py` — Feature 1 (uses llm + rag)
6. `pages/3_PII_Extractor.py` — Feature 3 (uses llm only, simplest)
7. `pages/4_Sentiment_Analysis.py` — Feature 4 (uses llm only)
8. `pages/2_OCR.py` — Feature 2 (uses ocr + helpers)
9. `pages/5_Document_Classifier.py` — Feature 5 (uses llm + rag)
10. `app.py` — main entry + navigation polish

---

## Requirements.txt

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

## Important Rules / Constraints

- Never hardcode the OpenAI API key — always load from `.env`
- Add `.env` to `.gitignore`
- All LLM responses that expect JSON must go through `parse_llm_json()` in helpers
- Use `st.session_state` to persist vectorstore and chat history across Streamlit reruns
- EasyOCR first run downloads models (~200MB) — warn the user in UI
- ChromaDB stores data in-memory for this project (no persistence needed for demo)
- All file uploads handled via `st.file_uploader()` and read as `BytesIO`

---

## UI Design Reference

- Sidebar navigation with 5 items (icons + labels)
- Clean, minimal design — white cards, subtle borders
- Badge on topbar shows current feature mode (RAG Enabled / LLM Powered etc.)
- Color coding: Relevant = green, Not Relevant = red/orange, Neutral = gray
- Positive sentiment = green, Negative = red, Neutral = gray

---

*Last updated: May 2026*
