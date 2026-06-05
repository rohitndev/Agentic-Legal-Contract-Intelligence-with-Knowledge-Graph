"""Split a contract into individually addressable clauses.

A *clause* is the atomic unit the rest of the pipeline reasons over: the NER
pipeline extracts entities from it, the risk classifier scores it, the graph
stores it as a node, and the agent proposes redlines against it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# Common numbered/headed section patterns found in commercial contracts.
_SECTION_PATTERNS = [
    re.compile(r"^\s*(\d+\.\d*\.?\d*)\s+(.+)$"),          # "12.3 Indemnification"
    re.compile(r"^\s*(ARTICLE\s+[IVXLC0-9]+)[\.:\s]+(.+)$", re.I),
    re.compile(r"^\s*(SECTION\s+\d+)[\.:\s]+(.+)$", re.I),
    re.compile(r"^\s*([A-Z][A-Z \-]{3,40})\s*$"),          # ALL-CAPS heading
]


@dataclass
class Clause:
    """A single contract clause."""

    clause_id: str
    heading: str
    text: str
    section_number: str = ""
    entities: dict = field(default_factory=dict)
    risk: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "clause_id": self.clause_id,
            "heading": self.heading,
            "section_number": self.section_number,
            "text": self.text,
            "entities": self.entities,
            "risk": self.risk,
        }


class SectionDetector:
    """Detect section boundaries and emit ordered :class:`Clause` objects."""

    def detect(self, text: str) -> List[Clause]:
        lines = [ln.rstrip() for ln in text.splitlines()]
        clauses: List[Clause] = []
        current_heading = "Preamble"
        current_number = ""
        buffer: List[str] = []
        idx = 0

        def flush():
            nonlocal buffer, idx, current_heading, current_number
            body = "\n".join(buffer).strip()
            if body:
                idx += 1
                clauses.append(
                    Clause(
                        clause_id=f"C{idx:03d}",
                        heading=current_heading.strip()[:120],
                        section_number=current_number,
                        text=body,
                    )
                )
            buffer = []

        for line in lines:
            heading = self._match_heading(line)
            if heading is not None:
                flush()
                current_number, current_heading = heading
            else:
                buffer.append(line)
        flush()

        # Fallback: if the document had no detectable headings, chunk by paragraph.
        if len(clauses) <= 1 and text.strip():
            return self._chunk_paragraphs(text)
        return clauses

    def _match_heading(self, line: str):
        stripped = line.strip()
        if not stripped or len(stripped) > 120:
            return None
        for pattern in _SECTION_PATTERNS:
            m = pattern.match(stripped)
            if m:
                groups = m.groups()
                if len(groups) == 2:
                    return groups[0].strip(), groups[1].strip()
                return "", groups[0].strip()
        return None

    def _chunk_paragraphs(self, text: str) -> List[Clause]:
        chunks = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return [
            Clause(clause_id=f"C{i + 1:03d}", heading=f"Paragraph {i + 1}", text=chunk)
            for i, chunk in enumerate(chunks)
        ]
