# AI Suite

Flask app for a local AI document intelligence suite with five tools:

- AI Chat
- OCR
- PII Extractor
- Sentiment Analysis
- Document Classifier

## Run

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Hugging Face

AI Chat uses the Hugging Face Inference API.

Create an `.env` file:

```text
HF_API_TOKEN=your_hugging_face_token
HF_CHAT_MODEL=katanemo/Arch-Router-1.5B:hf-inference
HF_FALLBACK_MODELS=
```

Some Meta Llama models are gated on Hugging Face, so your account may need to accept the model access terms before the model responds. The app can try fallback models when the primary provider is unavailable.

OCR remains separate and does not use the LLM.

The current working feature is AI Chat: upload PDF, TXT, or DOCX files, index them, and ask questions against retrieved document context.
