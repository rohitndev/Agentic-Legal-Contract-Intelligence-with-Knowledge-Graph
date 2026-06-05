"""Lightweight document-structure metadata extraction.

Sits on top of :class:`SectionDetector` to surface document-level signals
(numbered lists, defined terms, party block, effective date) that downstream
NER and graph population steps rely on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ContractStructure:
    defined_terms: List[str] = field(default_factory=list)
    party_block: str = ""
    effective_date: str = ""
    numbered_items: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)


class StructureExtractor:
    """Extract high-level structural metadata from raw contract text."""

    _DEFINED_TERM = re.compile(r'"([A-Z][A-Za-z0-9 \-]{2,40})"')
    _EFFECTIVE_DATE = re.compile(
        r"(?:effective\s+(?:as\s+of\s+|date[:\s]+))"
        r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})",
        re.I,
    )
    _NUMBERED = re.compile(r"^\s*\d+\.", re.M)

    def extract(self, text: str) -> ContractStructure:
        defined = sorted({m.group(1) for m in self._DEFINED_TERM.finditer(text)})
        eff = self._EFFECTIVE_DATE.search(text)
        numbered = len(self._NUMBERED.findall(text))

        # The party block is usually the first paragraph mentioning "between".
        party_block = ""
        for para in re.split(r"\n\s*\n", text):
            if re.search(r"\bby and between\b|\bbetween\b", para, re.I):
                party_block = para.strip()[:400]
                break

        return ContractStructure(
            defined_terms=defined[:50],
            party_block=party_block,
            effective_date=eff.group(1) if eff else "",
            numbered_items=numbered,
            metadata={"defined_term_count": str(len(defined))},
        )
