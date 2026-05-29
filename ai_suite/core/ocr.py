"""OCR using GPT-4o Vision."""
from __future__ import annotations
from typing import Iterable
import base64

try:
    from core.llm import initialize_client
except ModuleNotFoundError:
    from ai_suite.core.llm import initialize_client


def extract_ocr_text(images: Iterable) -> list[dict]:
    """Extract text from uploaded image files using GPT-4o Vision."""
    client = initialize_client()
    results = []

    for file_storage in images:
        if not file_storage or not file_storage.filename:
            continue

        try:
            file_storage.stream.seek(0)
        except Exception:
            pass

        try:
            image_data = base64.b64encode(file_storage.stream.read()).decode("utf-8")
            filename = file_storage.filename.lower()

            if filename.endswith(".png"):
                mime = "image/png"
            elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
                mime = "image/jpeg"
            elif filename.endswith(".webp"):
                mime = "image/webp"
            else:
                mime = "image/jpeg"  # default

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_data}"}
                        },
                        {
                            "type": "text",
                            "text": "Extract all text from this image exactly as it appears. Preserve formatting, layout, and line breaks where possible."
                        }
                    ]
                }],
                max_tokens=2000
            )

            text = response.choices[0].message.content.strip()
            results.append({"filename": file_storage.filename, "text": text})

        except Exception as e:
            results.append({"filename": file_storage.filename, "text": "", "error": str(e)})

    return results