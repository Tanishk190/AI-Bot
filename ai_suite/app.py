from flask import Flask, render_template


app = Flask(__name__)

TOOLS = [
    {
        "id": "chat",
        "title": "AI Chat",
        "badge": "RAG Enabled",
        "icon": "MSG",
    },
    {
        "id": "ocr",
        "title": "OCR",
        "badge": "Image to Text",
        "icon": "OCR",
    },
    {
        "id": "pii",
        "title": "PII Extractor",
        "badge": "LLM Powered",
        "icon": "ID",
    },
    {
        "id": "sentiment",
        "title": "Sentiment",
        "page_title": "Sentiment Analysis",
        "badge": "LLM Powered",
        "icon": "SEN",
    },
    {
        "id": "classify",
        "title": "Classifier",
        "page_title": "Document Classifier",
        "badge": "RAG + LLM",
        "icon": "TAG",
    },
]


@app.route("/")
def index():
    return render_template("index.html", tools=TOOLS)


if __name__ == "__main__":
    app.run(debug=True)
