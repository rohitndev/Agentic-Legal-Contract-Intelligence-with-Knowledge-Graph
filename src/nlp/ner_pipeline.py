"""Legal NER pipeline.

Identifies parties, dates, monetary values, obligations, IP references, and
termination conditions. Uses a fine-tuned Legal-BERT token-classification head
when ``ENABLE_TRANSFORMERS=1`` and the model is available, otherwise falls back
to a high-precision rule-based extractor so the pipeline always runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from ..config import settings

_MONEY = re.compile(r"(?:USD\s?|US\$|\$|EUR\s?|€|£)\s?[\d,]+(?:\.\d+)?(?:\s?(?:million|billion|M|B|k))?", re.I)
_DATE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b\d{1,3}\s+days?\b",
    re.I,
)
_PARTY = re.compile(
    r'\b([A-Z][A-Za-z0-9&.,\- ]+?(?:Inc\.|LLC|Ltd\.|Corp\.|Corporation|Company|GmbH|L\.P\.|LLP|N\.V\.))',
)
_OBLIGATION = re.compile(
    r"(?:[A-Z][\w &]+?\s)?(?:shall|must|will|agrees to|is required to|undertakes to)\s+[a-z][^.;]{5,160}",
    re.I,
)
_IP = re.compile(
    r"\b(patent|copyright|trademark|trade secret|intellectual property|know-how|source code)\b", re.I
)
_TERMINATION = re.compile(
    r"[^.;\n]*\b(terminat\w+|expire|expiration|renewal)\b[^.;\n]{0,160}", re.I
)


@dataclass
class ExtractedEntities:
    parties: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    monetary_values: List[str] = field(default_factory=list)
    obligations: List[str] = field(default_factory=list)
    ip_references: List[str] = field(default_factory=list)
    termination_conditions: List[str] = field(default_factory=list)
    engine: str = "rule-based"

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "parties": self.parties,
            "dates": self.dates,
            "monetary_values": self.monetary_values,
            "obligations": self.obligations,
            "ip_references": self.ip_references,
            "termination_conditions": self.termination_conditions,
        }


class NERPipeline:
    """Extract legal entities from clause/contract text."""

    def __init__(self) -> None:
        self._hf = None
        if settings.enable_transformers:
            self._try_load_transformer()

    def _try_load_transformer(self) -> None:  # pragma: no cover - optional heavy path
        try:
            from transformers import pipeline as hf_pipeline

            self._hf = hf_pipeline(
                "token-classification",
                model=settings.legal_bert_model,
                aggregation_strategy="simple",
            )
        except Exception:
            self._hf = None

    def extract(self, text: str) -> ExtractedEntities:
        ents = self._rule_based(text)
        if self._hf is not None:  # pragma: no cover - optional heavy path
            try:
                self._augment_with_transformer(text, ents)
                ents.engine = "legal-bert+rules"
            except Exception:
                pass
        return ents

    # ------------------------------------------------------------------
    def _rule_based(self, text: str) -> ExtractedEntities:
        def uniq(seq):
            seen, out = set(), []
            for s in seq:
                s = s.strip()
                key = s.lower()
                if s and key not in seen:
                    seen.add(key)
                    out.append(s)
            return out

        return ExtractedEntities(
            parties=uniq(m.group(1) for m in _PARTY.finditer(text))[:20],
            dates=uniq(m.group(0) for m in _DATE.finditer(text))[:30],
            monetary_values=uniq(m.group(0) for m in _MONEY.finditer(text))[:30],
            obligations=uniq(m.group(0) for m in _OBLIGATION.finditer(text))[:40],
            ip_references=uniq(m.group(1).lower() for m in _IP.finditer(text))[:20],
            termination_conditions=uniq(m.group(0).strip() for m in _TERMINATION.finditer(text))[:20],
        )

    def _augment_with_transformer(self, text, ents):  # pragma: no cover - optional
        for span in self._hf(text[:4000]):
            label = span.get("entity_group", "")
            word = span.get("word", "").strip()
            if label in {"ORG", "PARTY"} and word not in ents.parties:
                ents.parties.append(word)
            elif label in {"DATE"} and word not in ents.dates:
                ents.dates.append(word)
            elif label in {"MONEY"} and word not in ents.monetary_values:
                ents.monetary_values.append(word)
