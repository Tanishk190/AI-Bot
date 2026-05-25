"""Document classifier using semantic similarity and LLM reasoning."""
import numpy as np

try:
    from core.rag import load_documents, _get_embedding, chunk_text
    from core.llm import generate_completion, parse_llm_json
except ModuleNotFoundError:
    from ai_suite.core.rag import load_documents, _get_embedding, chunk_text
    from ai_suite.core.llm import generate_completion, parse_llm_json


def classify_documents(documents, criteria: str) -> list[dict]:
    """
    Classify uploaded documents against user-defined criteria.

    Args:
        documents: List of werkzeug FileStorage objects
        criteria: User-defined relevance criteria (string)

    Returns:
        List of dicts with 'filename', 'similarity_score', 'classification', 'reasoning'
    """
    if not criteria or not criteria.strip():
        raise ValueError("Classification criteria is required.")

    if not documents or not any(d for d in documents if d and d.filename):
        raise ValueError("At least one document is required.")

    # Reset file streams and filter valid documents
    valid_docs = []
    for doc in documents:
        if doc and doc.filename:
            try:
                doc.stream.seek(0)
            except Exception:
                pass
            valid_docs.append(doc)
    
    if not valid_docs:
        raise ValueError("No valid documents provided.")

    # Load and chunk all documents
    all_chunks = load_documents(valid_docs)
    if not all_chunks:
        raise ValueError("No readable text found in uploaded documents.")

    # Get embedding for criteria
    criteria_embedding = _get_embedding(criteria)
    if criteria_embedding is None:
        raise RuntimeError("Could not embed classification criteria.")

    # Group chunks by source document
    docs_by_source = {}
    for chunk in all_chunks:
        if chunk.source not in docs_by_source:
            docs_by_source[chunk.source] = []
        docs_by_source[chunk.source].append(chunk)

    results = []

    for filename, chunks in docs_by_source.items():
        # Calculate similarity scores for all chunks
        similarities = []
        for chunk in chunks:
            chunk_embedding = _get_embedding(chunk.text)
            if chunk_embedding is None:
                continue

            similarity = np.dot(criteria_embedding, chunk_embedding) / (
                np.linalg.norm(criteria_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append(similarity)

        if not similarities:
            results.append({
                "filename": filename,
                "similarity_score": 0.0,
                "classification": "UNCERTAIN",
                "reasoning": "Could not process document."
            })
            continue

        # Average similarity across document
        avg_similarity = float(np.mean(similarities))
        max_similarity = float(np.max(similarities))

        # Get top 3 chunks for LLM reasoning
        ranked_chunks = sorted(
            zip(chunks, similarities),
            key=lambda x: x[1],
            reverse=True
        )
        top_chunks = ranked_chunks[:3]
        doc_summary = "\n---\n".join([f"{chunk.text}" for chunk, _ in top_chunks])

        # Call LLM for final classification
        system_prompt = (
            "You are a document relevance classifier. Analyze the given document and criteria, "
            "then classify as RELEVANT, NOT_RELEVANT, or UNCERTAIN. "
            "Return ONLY a valid JSON object with 'classification' and 'reasoning' fields."
        )
        user_prompt = (
            f"Criteria: {criteria}\n\n"
            f"Document content:\n{doc_summary}\n\n"
            f"Semantic similarity score: {max_similarity:.2f}\n\n"
            "Classify this document and provide reasoning."
        )

        try:
            llm_response = generate_completion(user_prompt, system_prompt=system_prompt)
            llm_result = parse_llm_json(llm_response)
            classification = llm_result.get("classification", "UNCERTAIN").upper()
            reasoning = llm_result.get("reasoning", "No reasoning provided.")
        except Exception as e:
            # Fallback to similarity-based classification if LLM fails
            if max_similarity > 0.7:
                classification = "RELEVANT"
                reasoning = f"High semantic similarity ({max_similarity:.2f}) with criteria."
            elif max_similarity > 0.3:
                classification = "UNCERTAIN"
                reasoning = f"Medium semantic similarity ({max_similarity:.2f}) with criteria. Manual review recommended."
            else:
                classification = "NOT_RELEVANT"
                reasoning = f"Low semantic similarity ({max_similarity:.2f}) with criteria."

        results.append({
            "filename": filename,
            "similarity_score": round(max_similarity, 3),
            "classification": classification,
            "reasoning": reasoning
        })

    return results
