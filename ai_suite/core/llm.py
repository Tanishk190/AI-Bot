"""LLM wrapper for OpenAI GPT-4o models."""
import json
import os
from openai import OpenAI, APIError, APIConnectionError


_client = None


def initialize_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY...")
        _client = OpenAI(api_key=api_key)
    return _client


DEFAULT_MODEL = "gpt-4o"


def generate_completion(prompt: str, model: str = DEFAULT_MODEL, system_prompt: str = None) -> str:
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


def generate_chat_completion(messages: list[dict], model: str = DEFAULT_MODEL,
                             system_prompt: str = None) -> str:
    """Generate completion with full message history for multi-turn conversation."""
    try:
        client = initialize_client()
        system_message = system_prompt or "You are a helpful AI assistant for document analysis."
        full_messages = [{"role": "system", "content": system_message}] + messages
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=0.2,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except APIConnectionError as exc:
        raise RuntimeError(f"Could not connect to OpenAI API: {str(exc)}") from exc
    except APIError as exc:
        raise RuntimeError(f"OpenAI API error: {str(exc)}") from exc


def stream_chat_completion(messages: list[dict], model: str = DEFAULT_MODEL,
                           system_prompt: str = None):
    """Yield answer text deltas for a multi-turn conversation as they arrive.

    Mirrors generate_chat_completion but uses OpenAI streaming so the caller can
    forward tokens to the client in real time. Raises RuntimeError on API failure
    (callers iterating this generator should wrap it in try/except).
    """
    try:
        client = initialize_client()
        system_message = system_prompt or "You are a helpful AI assistant for document analysis."
        full_messages = [{"role": "system", "content": system_message}] + messages
        stream = client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=0.2,
            max_tokens=2000,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except APIConnectionError as exc:
        raise RuntimeError(f"Could not connect to OpenAI API: {str(exc)}") from exc
    except APIError as exc:
        raise RuntimeError(f"OpenAI API error: {str(exc)}") from exc


def build_rag_prompt(question: str, context_blocks: list[str]) -> tuple[str, str]:
    """Build system and user prompts for RAG."""
    context = "\n\n".join(context_blocks).strip()
    system_prompt = (
    "You are an AI assistant for document question answering. "
    "Use only the provided context from the uploaded documents. "
    "Structure your response as:\n"
    "Answer: <your answer>\n\n"
    "Sources: <filename> | Page <page number if given> | Chunk #<chunk number>\n\n"
    "If the answer requires calculating a difference between two dates or numbers found in the context, compute it and state the result. "
    "If the answer is not in the context, say: I could not find it in the uploaded documents."
)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    return system_prompt, user_prompt


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
