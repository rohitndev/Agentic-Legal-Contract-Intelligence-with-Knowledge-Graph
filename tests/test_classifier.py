"""Tests for the 15-category risk classifier."""

from src.nlp import RiskClassifier
from src.nlp.categories import MACRO_F1, RISK_CATEGORIES


def test_fifteen_categories_defined():
    assert len(RISK_CATEGORIES) == 15
    assert MACRO_F1 > 0.85  # fine-tuned Legal-BERT target


def test_uncapped_liability_is_red():
    text = ("Notwithstanding anything to the contrary, the Supplier's aggregate "
            "liability shall not be limited and shall be unlimited for all damages.")
    risk = RiskClassifier().classify(text)
    assert risk.top_category == "uncapped_liability"
    assert risk.rag_level == "red"
    assert risk.risk_score >= 7


def test_indemnification_detected():
    text = "Licensee shall indemnify, defend, and hold harmless Licensor without limitation."
    risk = RiskClassifier().classify(text)
    assert risk.top_category == "unlimited_indemnification"


def test_benign_clause_is_green():
    text = "Section headings are inserted for reference and do not affect interpretation."
    risk = RiskClassifier().classify(text)
    assert risk.rag_level == "green"
