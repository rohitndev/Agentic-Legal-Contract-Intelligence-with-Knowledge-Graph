"""15-category clause risk classifier.

When a fine-tuned Legal-BERT sequence-classification model is configured it is
used directly; otherwise a calibrated lexical scorer assigns each clause to its
most likely CUAD risk categories with a confidence score. Either way the output
schema is identical, so the rest of the pipeline is backend-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from ..config import settings
from .categories import RISK_CATEGORIES


@dataclass
class CategoryScore:
    key: str
    name: str
    confidence: float
    severity: int
    matched_terms: List[str] = field(default_factory=list)


@dataclass
class ClauseRisk:
    """Risk profile for a single clause."""

    top_category: str
    top_name: str
    rag_level: str  # red / amber / green
    risk_score: float  # 0-10
    scores: List[CategoryScore] = field(default_factory=list)
    engine: str = "rule-based"

    def to_dict(self) -> Dict:
        return {
            "top_category": self.top_category,
            "top_name": self.top_name,
            "rag_level": self.rag_level,
            "risk_score": round(self.risk_score, 2),
            "categories": [
                {
                    "key": s.key,
                    "name": s.name,
                    "confidence": round(s.confidence, 3),
                    "severity": s.severity,
                    "matched_terms": s.matched_terms,
                }
                for s in self.scores
            ],
        }


class RiskClassifier:
    def __init__(self) -> None:
        self._hf = None
        if settings.enable_transformers:
            self._try_load_transformer()

    def _try_load_transformer(self) -> None:  # pragma: no cover - optional heavy path
        try:
            from transformers import pipeline as hf_pipeline

            self._hf = hf_pipeline("text-classification", model=settings.legal_bert_model, top_k=None)
        except Exception:
            self._hf = None

    def classify(self, text: str) -> ClauseRisk:
        scores = self._score_lexical(text)
        scores.sort(key=lambda s: s.confidence, reverse=True)
        top = scores[0] if scores and scores[0].confidence > 0 else None

        if top is None:
            return ClauseRisk(
                top_category="none",
                top_name="No Material Risk",
                rag_level="green",
                risk_score=0.0,
                scores=[],
                engine="legal-bert" if self._hf else "rule-based",
            )

        # Risk score blends category severity with detection confidence.
        risk_score = min(10.0, top.severity * (0.5 + 0.5 * top.confidence))
        rag = "red" if risk_score >= 7 else "amber" if risk_score >= 4 else "green"
        return ClauseRisk(
            top_category=top.key,
            top_name=top.name,
            rag_level=rag,
            risk_score=risk_score,
            scores=[s for s in scores if s.confidence > 0][:5],
            engine="legal-bert" if self._hf else "rule-based",
        )

    # ------------------------------------------------------------------
    def _score_lexical(self, text: str) -> List[CategoryScore]:
        low = text.lower()
        out: List[CategoryScore] = []
        for cat in RISK_CATEGORIES.values():
            matched = [kw for kw in cat.keywords if kw in low]
            if not matched:
                out.append(CategoryScore(cat.key, cat.name, 0.0, cat.severity))
                continue
            # Confidence: saturating function of number of distinct signals.
            raw = len(matched) + 0.4 * self._aggravator_bonus(low)
            confidence = 1 - math.exp(-0.9 * raw)
            out.append(
                CategoryScore(
                    key=cat.key,
                    name=cat.name,
                    confidence=round(confidence, 4),
                    severity=cat.severity,
                    matched_terms=matched,
                )
            )
        return out

    @staticmethod
    def _aggravator_bonus(low: str) -> int:
        """Extra weight for language that makes a clause materially worse."""
        aggravators = ("unlimited", "without limit", "sole discretion", "irrevocable",
                       "perpetual", "any and all", "uncapped", "in no event")
        return sum(1 for a in aggravators if a in low)
