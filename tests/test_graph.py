"""Tests for the Legal Knowledge Graph builder and multi-hop queries."""

from src.graph import GraphQueries, KnowledgeGraph


def _sample_clauses():
    return [
        {"clause_id": "C001", "heading": "Indemnification",
         "entities": {"obligations": ["shall indemnify the other party"]},
         "risk": {"top_category": "unlimited_indemnification",
                  "top_name": "Unlimited Indemnification", "rag_level": "red", "risk_score": 9.0}},
        {"clause_id": "C002", "heading": "Governing Law",
         "entities": {"obligations": []},
         "risk": {"top_category": "governing_law", "top_name": "Governing Law",
                  "rag_level": "green", "risk_score": 2.5}},
    ]


def test_graph_build_creates_nodes():
    kg = KnowledgeGraph()
    stats = kg.build("contract::test", ["Acme Inc.", "Beta LLC"], _sample_clauses())
    assert stats["nodes"] > 0
    assert stats["relationships"] > 0
    kg.close()


def test_high_risk_query():
    kg = KnowledgeGraph()
    kg.build("contract::test", ["Acme Inc."], _sample_clauses())
    high = GraphQueries(kg).high_risk_clauses(threshold=7.0)
    assert any("Indemnification" in h["heading"] for h in high)
    kg.close()


def test_risk_distribution():
    kg = KnowledgeGraph()
    kg.build("contract::test", [], _sample_clauses())
    dist = GraphQueries(kg).risk_category_distribution()
    assert sum(dist.values()) >= 2
    kg.close()
