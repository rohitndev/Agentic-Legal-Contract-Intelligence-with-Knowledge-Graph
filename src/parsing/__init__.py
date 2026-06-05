"""Contract ingestion: PDF text extraction, structure preservation, section detection."""

from .pdf_parser import ParsedContract, PDFParser
from .section_detector import Clause, SectionDetector
from .structure_extractor import StructureExtractor

__all__ = [
    "PDFParser",
    "ParsedContract",
    "StructureExtractor",
    "SectionDetector",
    "Clause",
]
