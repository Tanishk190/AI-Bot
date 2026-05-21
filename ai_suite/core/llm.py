"""LLM wrapper for OpenAI GPT-4o models."""
import json
import os
from openai import OpenAI, APIError, APIConnectionError


def initialize_client():
    api_key = os.getenv("OPENAI_API_KEY")  # read here
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY...")
    return OpenAI(api_key=api_key)
DEFAULT_MODEL = "gpt-4o"


def generate_completion(prompt: str, model: str = DEFAULT_MODEL, system_prompt: str = None) -> str:
    """
    Generate completion using OpenAI GPT-4o.
    
    Args:
        prompt: The prompt text
        model: Model name (default: gpt-4o)
    
    Returns:
        Generated text response
    """
    try:
        client = initialize_client()
        system_message = system_prompt or "You are a helpful AI assistant for document analysis."
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except APIConnectionError as exc:
        raise RuntimeError(f"Could not connect to OpenAI API: {str(exc)}") from exc
    except APIError as exc:
        raise RuntimeError(f"OpenAI API error: {str(exc)}") from exc


def build_rag_prompt(question: str, context_blocks: list[str]) -> str:
    """Build RAG prompt from question and context."""
    context = "\n\n".join(context_blocks).strip()
    return f"""You are an AI assistant for document question answering.
Use only the provided context from the uploaded documents.
If the user asks what the document is about, summarize the main topic and key points from the context.
And give the answer in point form.
If the answer is not supported by the context, say: I could not find it in the uploaded documents.

Context:
{context}

Question:
{question}

    Answer:"""


def parse_llm_json(response: str) -> dict:
    """
    Parse JSON from LLM response, handling markdown fences.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Parsed JSON dict
    """
    if response.strip().startswith("```"):
        lines = response.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.strip().startswith("```json"):
                in_json = True
                continue
            if line.strip().startswith("```"):
                in_json = False
                continue
            if in_json or (json_lines and not line.strip().startswith("```")):
                json_lines.append(line)
        response = "\n".join(json_lines)

    try:
        return json.loads(response.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM response as JSON: {str(exc)}") from exc

