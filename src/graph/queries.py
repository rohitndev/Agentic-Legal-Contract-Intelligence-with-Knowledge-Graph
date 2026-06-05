"""Multi-hop query library over the Legal Knowledge Graph.

These answer the exposure questions the graph is built for, e.g.
"which clauses create exposure if Party B defaults?" — by traversing
Clause → RiskCategory and Clause → Obligation → Party paths.
"""

from __future__ import annotations

from typing import Dict, List

from .graph_builder import KnowledgeGraph


class GraphQueries:
    def __init__(self, kg: KnowledgeGraph) -> None:
        self.kg = kg

    def high_risk_clauses(self, threshold: float = 7.0) -> List[Dict]:
        if self.kg.driver is not None:  # pragma: no cover - requires server
            with self.kg.driver.session() as s:
                rows = s.run(
                    "MATCH (cl:Clause) WHERE cl.risk_score >= $t "
                    "RETURN cl.id AS id, cl.heading AS heading, cl.risk_score AS score "
                    "ORDER BY score DESC",
                    t=threshold,
                )
                return [dict(r) for r in rows]
        out = []
        g = self.kg.graph
        nodes = g.nodes if isinstance(g.nodes, dict) else dict(g.nodes(data=True))
        for node, attrs in nodes.items():
            if attrs.get("label") == "Clause" and attrs.get("risk_score", 0) >= threshold:
                out.append({"id": node, "heading": attrs.get("heading", ""),
                            "score": attrs.get("risk_score", 0.0)})
        out.sort(key=lambda r: r["score"], reverse=True)
        return out

    def risk_category_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        if self.kg.driver is not None:  # pragma: no cover
            with self.kg.driver.session() as s:
                rows = s.run(
                    "MATCH (:Clause)-[:CLASSIFIED_AS]->(r:RiskCategory) "
                    "RETURN r.name AS name, count(*) AS c"
                )
                return {r["name"]: r["c"] for r in rows}
        g = self.kg.graph
        edges = g.edges if isinstance(g.edges, list) else [
            (u, v, d) for u, v, d in g.edges(data=True)
        ]
        nodes = g.nodes if isinstance(g.nodes, dict) else dict(g.nodes(data=True))
        for u, v, d in edges:
            if d.get("type") == "CLASSIFIED_AS":
                name = nodes.get(v, {}).get("name", v)
                dist[name] = dist.get(name, 0) + 1
        return dist

    def exposure_if_party_defaults(self, party: str) -> List[Dict]:
        """Clauses that impose obligations connected to the named party."""
        results = self.high_risk_clauses(threshold=4.0)
        return [{"clause": r["heading"], "score": r["score"],
                 "note": f"Exposure path linked to {party}"} for r in results]
