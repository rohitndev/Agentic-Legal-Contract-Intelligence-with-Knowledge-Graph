"""Tests for the legal NER pipeline."""

from src.nlp import NERPipeline


def test_ner_extracts_parties_and_money(sample_contract_text):
    ents = NERPipeline().extract(sample_contract_text)
    assert any("Northwind" in p for p in ents.parties)
    assert any("Globex" in p for p in ents.parties)
    assert ents.monetary_values  # "USD 5,000" / "$..." should be picked up
    assert ents.obligations
    assert ents.termination_conditions


def test_ner_ip_references():
    text = "All intellectual property and any patent or trademark shall be assigned."
    ents = NERPipeline().extract(text)
    assert "intellectual property" in ents.ip_references
