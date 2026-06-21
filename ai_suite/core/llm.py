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
    """Build system and user prompts for RAG.

    Context blocks are numbered [1], [2], ... and the model is asked to cite
    using ONLY that bracket index, never the underlying page/chunk numbers.
    The bracket index is the one piece of information the model cannot
    transpose or misremember (it just has to repeat a small integer it was
    just given), and the actual page/chunk/source metadata is substituted in
    afterward in Python from the real Chunk objects — see
    `resolve_citation_brackets` in app.py. This removes the failure mode where
    the model retypes a page or chunk number from the wrong block.
    """
    numbered_blocks = [f"[{i}] {block}" for i, block in enumerate(context_blocks, start=1)]
    context = "\n\n".join(numbered_blocks).strip()
    system_prompt = (
        "You are an AI assistant for document question answering for a Chartered Accountancy / audit firm. "
        "Use only the provided context blocks from the uploaded documents. "
        "Each context block is prefixed with a bracketed index, like [1] or [2], followed by its source file, page, and chunk number. "
        "Cite your sources inline, immediately after the relevant statement, using ONLY that bracketed index, e.g. [1] or [2][3] for a statement supported by two blocks. "
        "CRITICAL: do NOT write out the filename, page number, or chunk number yourself anywhere in your answer — "
        "write only the bracketed index number, exactly as it appears at the start of the context block you used. "
        "The real source citation will be attached automatically after your answer based on that index. "
        "If a question spans multiple clauses or sections, cover every relevant one you find in the context and cite each with its index. "
        "If the question has multiple distinct parts, explicitly address each part — if one part is not covered by the context, "
        "say so for that part specifically rather than silently omitting it. "
        "If the question refers to a specific clause, section, or item that is not present in the provided "
        "context, say it is not in the provided pages rather than answering from a different section. "
        "If the answer requires calculating a difference between two dates or numbers found in the context, compute it and state the result. "
        "When a question asks about a value or condition ACROSS a table — for example a total, or 'how much X overall', "
        "or 'which rows have Y', or any question that is not pinned to one specific named row — you MUST scan EVERY row of the "
        "relevant table before answering, not just the first matching row. The context may present table data both as a grid and "
        "as one self-describing line per row (each line pairs every column label with its value, e.g. 'Section: 194I | Amount not "
        "deposited: 24,000'); use these row lines to check each row individually. Report every row whose value is non-zero / "
        "non-Nil / relevant, even if other rows are Nil. Never conclude a value is Nil or zero overall when any single row contains "
        "a non-Nil value for that column. "
        "When you present extracted data as a markdown table, every data row must have a value in EVERY column. "
        "If a value for a cell is not present in the context, write 'Not specified in document' in that cell rather than leaving it blank — "
        "never emit a table row that has fewer cells than the header or an empty cell. "
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
    text = response.strip()
    if text.startswith("```"):
        # Strip the opening fence line (``` or ```json) and the closing fence.
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # drop closing fence
        text = "\n".join(lines)

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM response as JSON: {str(exc)}") from exc