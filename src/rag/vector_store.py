"""Embedding store over the CUAD corpus.

ChromaDB (with sentence-transformers embeddings) when ``ENABLE_CHROMA=1``;
otherwise a dependency-light TF-IDF cosine index that needs only scikit-learn.
Both expose the same ``query()`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..config import settings
from .corpus import CorpusClause, CUADCorpus


@dataclass
class RetrievedClause:
    clause: CorpusClause
    score: float


class VectorStore:
    def __init__(self, corpus: CUADCorpus) -> None:
        self.corpus = corpus
        self.backend = "tf-idf"
        self._chroma = None
        self._tfidf = None
        self._matrix = None
        if settings.enable_chroma:
            self._init_chroma()
        if self._chroma is None:
            self._init_tfidf()

    # ------------------------------------------------------------------
    def _init_chroma(self) -> None:  # pragma: no cover - optional heavy path
        try:
            import chromadb

            client = chromadb.PersistentClient(path=settings.chroma_path)
            self._chroma = client.get_or_create_collection("cuad")
            if self._chroma.count() == 0 and self.corpus.clauses:
                self._chroma.add(
                    ids=[c.clause_id for c in self.corpus.clauses],
                    documents=[c.text for c in self.corpus.clauses],
                    metadatas=[{"category": c.category, "name": c.category_name}
                               for c in self.corpus.clauses],
                )
            self.backend = "chromadb"
        except Exception:
            self._chroma = None

    def _init_tfidf(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            texts = self.corpus.texts() or ["placeholder"]
            self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self._matrix = self._tfidf.fit_transform(texts)
            self.backend = "tf-idf"
        except Exception:  # pragma: no cover - sklearn missing
            self._tfidf = None
            self.backend = "keyword"

    # ------------------------------------------------------------------
    def query(self, text: str, top_k: int = 3) -> List[RetrievedClause]:
        if not self.corpus.clauses:
            return []
        if self._chroma is not None:  # pragma: no cover - optional
            res = self._chroma.query(query_texts=[text], n_results=top_k)
            out = []
            for cid, dist in zip(res["ids"][0], res["distances"][0]):
                clause = next((c for c in self.corpus.clauses if c.clause_id == cid), None)
                if clause:
                    out.append(RetrievedClause(clause, score=1.0 - float(dist)))
            return out
        if self._tfidf is not None:
            return self._tfidf_query(text, top_k)
        return self._keyword_query(text, top_k)

    def _tfidf_query(self, text: str, top_k: int) -> List[RetrievedClause]:
        from sklearn.metrics.pairwise import cosine_similarity

        vec = self._tfidf.transform([text])
        sims = cosine_similarity(vec, self._matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [RetrievedClause(self.corpus.clauses[i], float(sims[i]))
                for i in ranked if sims[i] > 0]

    def _keyword_query(self, text: str, top_k: int) -> List[RetrievedClause]:  # pragma: no cover
        low = text.lower()
        scored = []
        for c in self.corpus.clauses:
            overlap = len(set(low.split()) & set(c.text.lower().split()))
            scored.append(RetrievedClause(c, float(overlap)))
        scored.sort(key=lambda r: r.score, reverse=True)
        return [s for s in scored[:top_k] if s.score > 0]
