"""OCR using LightOnOCR-2-1B via Hugging Face Transformers."""
from __future__ import annotations

from typing import Iterable

_ocr_model = None
_ocr_processor = None
_ocr_device = None
_ocr_dtype = None

MODEL_ID = "lightonai/LightOnOCR-2-1B"


def _load_ocr_model():
    """Lazy-load OCR model and processor."""
    global _ocr_model, _ocr_processor, _ocr_device, _ocr_dtype

    if _ocr_model is not None and _ocr_processor is not None:
        return _ocr_model, _ocr_processor, _ocr_device, _ocr_dtype

    import torch
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

    if torch.backends.mps.is_available():
        _ocr_device = "mps"
        _ocr_dtype = torch.float32
    elif torch.cuda.is_available():
        _ocr_device = "cuda"
        _ocr_dtype = torch.bfloat16
    else:
        _ocr_device = "cpu"
        _ocr_dtype = torch.float32

    _ocr_processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)
    _ocr_model = LightOnOcrForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=_ocr_dtype
    ).to(_ocr_device)
    _ocr_model.eval()

    return _ocr_model, _ocr_processor, _ocr_device, _ocr_dtype


def extract_ocr_text(images: Iterable) -> list[dict]:
    """Extract text from uploaded image files using LightOnOCR-2-1B."""
    import torch
    from PIL import Image

    model, processor, device, dtype = _load_ocr_model()
    results = []

    for file_storage in images:
        if not file_storage or not file_storage.filename:
            continue

        try:
            file_storage.stream.seek(0)
        except Exception:
            pass

        image = Image.open(file_storage.stream).convert("RGB")

        conversation = [
            {"role": "user", "content": [{"type": "image", "image": image}]}
        ]

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024)

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]
        text = processor.decode(generated_ids, skip_special_tokens=True)

        results.append({"filename": file_storage.filename, "text": text})

    return results