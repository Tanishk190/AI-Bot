"""PII Extraction using OpenAI."""
import json

try:
    from core.llm import generate_completion, parse_llm_json
except ModuleNotFoundError:
    from ai_suite.core.llm import generate_completion, parse_llm_json


def extract_pii(text: str, system_prompt: str) -> dict:
    
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")
    
    if not system_prompt or not system_prompt.strip():
        raise ValueError("System prompt cannot be empty")
    
    user_prompt = f"Text to analyze:\n{text}\n\nProvide ONLY the JSON output, no other text."
    response = generate_completion(user_prompt, system_prompt=system_prompt)
    return parse_llm_json(response)


def format_pii_for_display(pii_data: dict) -> str:
    """Format PII data as formatted JSON string for display."""
    return json.dumps(pii_data, indent=2, ensure_ascii=False)
