"""PII Extraction using OpenAI."""
import json
from typing import Optional


def extract_pii(text: str, system_prompt: str) -> dict:
    """
    Extract PII from text using custom system prompt.
    
    Args:
        text: The input text to extract PII from
        system_prompt: Custom system prompt for extraction
        
    Returns:
        Dictionary of extracted PII data
    """
    try:
        from core.llm import generate_completion
    except ModuleNotFoundError:
        from ai_suite.core.llm import generate_completion
    
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")
    
    if not system_prompt or not system_prompt.strip():
        raise ValueError("System prompt cannot be empty")
    
    # Build the user prompt
    user_prompt = f"Extract PII from this text:\n\n{text}"
    
    # Create a modified prompt that includes the system instruction
    full_prompt = f"{system_prompt}\n\nText to analyze:\n{text}\n\nProvide ONLY the JSON output, no other text."
    
    # Call LLM
    response = generate_completion(full_prompt)
    
    # Parse JSON response
    pii_data = _parse_pii_response(response)
    
    return pii_data


def _parse_pii_response(response: str) -> dict:
    """
    Parse JSON from LLM response, handling markdown fences.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Parsed JSON dict
    """
    # Remove markdown code fences if present
    if response.strip().startswith("```"):
        # Find the start of actual JSON
        lines = response.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.strip().startswith("```json"):
                in_json = True
                continue
            elif line.strip().startswith("```"):
                in_json = False
                continue
            if in_json or (json_lines and not line.strip().startswith("```")):
                json_lines.append(line)
        response = "\n".join(json_lines)
    
    # Try to parse JSON
    try:
        data = json.loads(response.strip())
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse PII extraction response as JSON: {str(e)}") from e


def format_pii_for_display(pii_data: dict) -> str:
    """Format PII data as formatted JSON string for display."""
    return json.dumps(pii_data, indent=2, ensure_ascii=False)
