"""Tests for PDF parsing, structure extraction, and section detection."""

from src.parsing import SectionDetector, StructureExtractor


def test_section_detector_finds_clauses(sample_contract_text):
    clauses = SectionDetector().detect(sample_contract_text)
    assert len(clauses) >= 10
    headings = [c.heading.upper() for c in clauses]
    assert any("INDEMNIFICATION" in h for h in headings)
    assert any("GOVERNING LAW" in h for h in headings)


def test_clause_ids_are_unique(sample_contract_text):
    clauses = SectionDetector().detect(sample_contract_text)
    ids = [c.clause_id for c in clauses]
    assert len(ids) == len(set(ids))


def test_structure_extractor(sample_contract_text):
    structure = StructureExtractor().extract(sample_contract_text)
    assert structure.effective_date
    assert "Confidential Information" in structure.defined_terms or structure.defined_terms
    assert "between" in structure.party_block.lower()
