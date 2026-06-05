"""CUAD contract corpus loader.

CUAD (Contract Understanding Atticus Dataset) ships 510 annotated contracts.
This module loads ``data/cuad_corpus.json`` — a curated set of annotated clause
exemplars (one or more per risk category) used as RAG precedents. When the full
CUAD dataset is downloaded into ``data/cuad/``, it is loaded transparently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..config import DATA_DIR


@dataclass
class CorpusClause:
    clause_id: str
    category: str
    category_name: str
    text: str
    source_contract: str = "CUAD"


class CUADCorpus:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (DATA_DIR / "cuad_corpus.json")
        self.clauses: List[CorpusClause] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.clauses = []
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.clauses = [
            CorpusClause(
                clause_id=item["clause_id"],
                category=item["category"],
                category_name=item.get("category_name", item["category"]),
                text=item["text"],
                source_contract=item.get("source_contract", "CUAD"),
            )
            for item in raw
        ]

    def __len__(self) -> int:
        return len(self.clauses)

    def texts(self) -> List[str]:
        return [c.text for c in self.clauses]
