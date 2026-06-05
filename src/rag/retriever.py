"""High-level retrieval API combining the corpus + vector store."""

from __future__ import annotations

from typing import Dict, List

from .corpus import CUADCorpus
from .vector_store import VectorStore


class Retriever:
    def __init__(self) -> None:
        self.corpus = CUADCorpus()
        self.store = VectorStore(self.corpus)

    @property
    def backend(self) -> str:
        return self.store.backend

    def similar_precedents(self, clause_text: str, top_k: int = 3) -> List[Dict]:
        """Retrieve the most similar annotated CUAD precedents for a clause."""
        results = self.store.query(clause_text, top_k=top_k)
        return [
            {
                "clause_id": r.clause.clause_id,
                "category": r.clause.category,
                "category_name": r.clause.category_name,
                "source_contract": r.clause.source_contract,
                "similarity": round(r.score, 4),
                "text": r.clause.text,
            }
            for r in results
        ]
