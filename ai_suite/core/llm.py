"""LLM wrapper for OpenAI GPT-4o models."""
import os
from openai import OpenAI, APIError, APIConnectionError


API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o"


def initialize_client():
    """Initialize OpenAI client."""
    if not API_KEY:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Add it to your .env file or set the environment variable."
        )
    return OpenAI(api_key=API_KEY)


def generate_completion(prompt: str, model: str = DEFAULT_MODEL) -> str:
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
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for document analysis."},
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
If the answer is not supported by the context, say: I could not find it in the uploaded documents.

Context:
{context}

Question:
{question}

Answer:"""

