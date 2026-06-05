"""End-to-end contract analysis pipeline.

Ties the layers together:
    parse -> detect clauses -> NER + risk classify per clause -> build graph
    -> RAG precedents -> GraphRAG community summaries -> monetary quantification.

This is the single entry point used by both the CLI demo and the API.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .agent import LLMClient, RiskQuantifier
from .graph import GraphQueries, KnowledgeGraph
from .nlp import NERPipeline, RiskClassifier
from .nlp.categories import MACRO_F1
from .parsing import PDFParser, SectionDetector, StructureExtractor
from .rag import GraphRAG, Retriever


@dataclass
class AnalysisResult:
    contract_id: str
    source: str
    parties: List[str]
    clause_count: int
    rag_summary: Dict[str, int]
    overall_risk_score: float
    clauses: List[dict]
    graph_stats: Dict[str, int]
    graph_backend: str
    high_risk_clauses: List[dict]
    risk_distribution: Dict[str, int]
    community_summaries: List[dict]
    exposure: Dict
    capabilities: Dict[str, str]
    elapsed_seconds: float = 0.0
    macro_f1: float = MACRO_F1

    def to_dict(self) -> Dict:
        return asdict(self)


class ContractIntelligencePipeline:
    def __init__(self) -> None:
        self.parser = PDFParser()
        self.sectioner = SectionDetector()
        self.structurer = StructureExtractor()
        self.ner = NERPipeline()
        self.classifier = RiskClassifier()
        self.retriever = Retriever()
        self.llm = LLMClient()
        self.graphrag = GraphRAG(llm=self.llm)
        self.quantifier = RiskQuantifier()

    def analyze_file(self, path: str | Path, contract_value: Optional[float] = None) -> AnalysisResult:
        parsed = self.parser.parse(path)
        return self._analyze(parsed.source, parsed.text, contract_value)

    def analyze_text(self, text: str, source: str = "inline", contract_value: Optional[float] = None) -> AnalysisResult:
        return self._analyze(source, text, contract_value)

    def analyze_bytes(self, data: bytes, filename: str, contract_value: Optional[float] = None) -> AnalysisResult:
        parsed = self.parser.parse_bytes(data, filename)
        return self._analyze(parsed.source, parsed.text, contract_value)

    # ------------------------------------------------------------------
    def _analyze(self, source: str, text: str, contract_value: Optional[float]) -> AnalysisResult:
        from .config import settings

        start = time.time()
        contract_id = f"contract::{abs(hash(source)) % 1_000_000}"
        structure = self.structurer.extract(text)
        clauses = self.sectioner.detect(text)

        all_parties = set(structure.party_block and [] or [])
        analyzed: List[dict] = []
        rag_counts = {"red": 0, "amber": 0, "green": 0}
        total_score = 0.0

        for clause in clauses:
            entities = self.ner.extract(clause.text)
            risk = self.classifier.classify(clause.text)
            clause.entities = entities.to_dict()
            clause.risk = risk.to_dict()
            for p in entities.parties:
                all_parties.add(p)
            rag_counts[risk.rag_level] = rag_counts.get(risk.rag_level, 0) + 1
            total_score += risk.risk_score

            record = clause.to_dict()
            if risk.top_category != "none":
                record["precedents"] = self.retriever.similar_precedents(clause.text, top_k=2)
            else:
                record["precedents"] = []
            analyzed.append(record)

        parties = sorted(all_parties)[:25]
        overall = round(total_score / max(1, len(clauses)), 2)

        kg = KnowledgeGraph()
        graph_stats = kg.build(contract_id, parties, analyzed)
        queries = GraphQueries(kg)
        high_risk = queries.high_risk_clauses(threshold=7.0)
        distribution = queries.risk_category_distribution()

        communities = self.graphrag.community_summaries(analyzed)
        exposure = self.quantifier.portfolio_exposure(analyzed, contract_value)

        result = AnalysisResult(
            contract_id=contract_id,
            source=source,
            parties=parties,
            clause_count=len(clauses),
            rag_summary=rag_counts,
            overall_risk_score=overall,
            clauses=analyzed,
            graph_stats=graph_stats,
            graph_backend=kg.backend,
            high_risk_clauses=high_risk,
            risk_distribution=distribution,
            community_summaries=communities,
            exposure=exposure,
            capabilities=settings.summary(),
            elapsed_seconds=round(time.time() - start, 3),
        )
        kg.close()
        return result
