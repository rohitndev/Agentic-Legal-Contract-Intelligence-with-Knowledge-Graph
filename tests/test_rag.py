"""Tests for CUAD RAG retrieval and GraphRAG community summaries."""

from src.rag import CUADCorpus, GraphRAG, Retriever


def test_corpus_loads():
    corpus = CUADCorpus()
    assert len(corpus) >= 15


def test_retriever_returns_relevant_precedent():
    retriever = Retriever()
    results = retriever.similar_precedents(
        "The party's aggregate liability shall not be limited.", top_k=3
    )
    assert results
    assert results[0]["similarity"] > 0


def test_graphrag_community_summaries():
    clauses = [
        {"heading": "Indemnity", "risk": {"top_category": "unlimited_indemnification",
                                          "top_name": "Unlimited Indemnification", "risk_score": 9.0}},
        {"heading": "Liability", "risk": {"top_category": "uncapped_liability",
                                          "top_name": "Uncapped Liability", "risk_score": 8.0}},
    ]
    summaries = GraphRAG().community_summaries(clauses)
    assert len(summaries) == 2
    assert summaries[0]["avg_risk_score"] >= summaries[1]["avg_risk_score"]
