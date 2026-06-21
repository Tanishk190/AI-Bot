"""Tests for the retrieval ranking helpers — BM25 keyword scoring and score
normalization. These are the pieces that fix incomplete multi-clause retrieval,
so they're worth locking in."""
from core.rag import _tokenize, _keyword_scores, _normalize


def test_tokenize_lowercases_and_splits_alnum():
    assert _tokenize("269SS cash, cheque!") == ["269ss", "cash", "cheque"]


def test_normalize_scales_to_unit_range():
    assert _normalize([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]


def test_normalize_handles_all_equal():
    # No spread -> all zeros (avoids divide-by-zero)
    assert _normalize([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_keyword_scores_rank_relevant_chunk_higher():
    rows = [
        {"text": "Particulars of each loan or deposit under section 269SS taken by cheque"},
        {"text": "Particulars of depreciation rate on plant and machinery"},
    ]
    scores = _keyword_scores("loan or deposit 269SS cheque cash", rows)
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_keyword_scores_rare_term_outweighs_common_term():
    # "cash" appears in every chunk (low IDF); "269SS" appears in only one
    # (high IDF), so the chunk with the rare term should rank top.
    rows = [
        {"text": "loan or deposit under section 269SS taken otherwise than by cash"},
        {"text": "payment made in cash to the supplier"},
        {"text": "expenses paid in cash during the year"},
        {"text": "petty cash reconciliation statement"},
    ]
    scores = _keyword_scores("269SS cash", rows)
    assert scores[0] == max(scores)


def test_keyword_scores_empty_query_returns_zeros():
    rows = [{"text": "anything"}, {"text": "something else"}]
    assert _keyword_scores("", rows) == [0.0, 0.0]
