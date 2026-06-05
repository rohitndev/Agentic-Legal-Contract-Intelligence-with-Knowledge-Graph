"""Populate the Legal Knowledge Graph from analyzed clauses.

Backend resolution:
  * If ``NEO4J_URI`` + ``NEO4J_PASSWORD`` are set *and* the ``neo4j`` driver is
    installed, nodes/relationships are written to Neo4j.
  * Otherwise an in-memory ``networkx`` graph is used (or a tiny built-in graph
    if networkx is absent), giving identical query semantics for local runs.
"""

from __future__ import annotations

from typing import Dict, List

from ..config import settings
from .neo4j_schema import SCHEMA_STATEMENTS

try:
    import networkx as nx  # type: ignore

    _HAS_NX = True
except Exception:  # pragma: no cover
    _HAS_NX = False


class KnowledgeGraph:
    """Builds and stores the contract entity-relationship graph."""

    def __init__(self) -> None:
        self.backend = "in-memory"
        self._driver = None
        self._graph = None
        if settings.neo4j_enabled:
            self._connect_neo4j()
        if self._driver is None:
            self._init_memory()

    # ------------------------------------------------------------------
    def _connect_neo4j(self) -> None:  # pragma: no cover - requires server
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
            )
            with self._driver.session() as session:
                for stmt in SCHEMA_STATEMENTS:
                    session.run(stmt)
            self.backend = "neo4j"
        except Exception:
            self._driver = None

    def _init_memory(self) -> None:
        self.backend = "networkx" if _HAS_NX else "in-memory"
        self._graph = nx.MultiDiGraph() if _HAS_NX else _MiniGraph()

    # ------------------------------------------------------------------
    def build(self, contract_id: str, parties: List[str], clauses: List[dict]) -> Dict[str, int]:
        """Populate the graph; returns node/edge counts."""
        if self._driver is not None:  # pragma: no cover - requires server
            return self._build_neo4j(contract_id, parties, clauses)
        return self._build_memory(contract_id, parties, clauses)

    def _build_memory(self, contract_id, parties, clauses) -> Dict[str, int]:
        g = self._graph
        g.add_node(contract_id, label="Contract")
        for p in parties:
            g.add_node(f"party::{p}", label="Party", name=p)
            g.add_edge(contract_id, f"party::{p}", type="HAS_PARTY")
        for cl in clauses:
            cid = f"{contract_id}::{cl['clause_id']}"
            g.add_node(cid, label="Clause", heading=cl.get("heading", ""),
                       rag_level=cl.get("risk", {}).get("rag_level", "green"),
                       risk_score=cl.get("risk", {}).get("risk_score", 0.0))
            g.add_edge(contract_id, cid, type="HAS_CLAUSE")
            risk = cl.get("risk", {})
            if risk.get("top_category") and risk["top_category"] != "none":
                rk = f"risk::{risk['top_category']}"
                g.add_node(rk, label="RiskCategory", key=risk["top_category"], name=risk.get("top_name", ""))
                g.add_edge(cid, rk, type="CLASSIFIED_AS")
            for ob in cl.get("entities", {}).get("obligations", [])[:5]:
                ob_id = f"{cid}::ob::{abs(hash(ob)) % 10_000}"
                g.add_node(ob_id, label="Obligation", text=ob)
                g.add_edge(cid, ob_id, type="IMPOSES")
        return self.stats()

    def _build_neo4j(self, contract_id, parties, clauses) -> Dict[str, int]:  # pragma: no cover
        with self._driver.session() as session:
            session.run("MERGE (c:Contract {id:$id})", id=contract_id)
            for p in parties:
                session.run(
                    "MERGE (p:Party {name:$n}) "
                    "WITH p MATCH (c:Contract {id:$id}) MERGE (c)-[:HAS_PARTY]->(p)",
                    n=p, id=contract_id,
                )
            for cl in clauses:
                risk = cl.get("risk", {})
                session.run(
                    "MATCH (c:Contract {id:$id}) "
                    "MERGE (cl:Clause {id:$cid}) "
                    "SET cl.heading=$h, cl.rag_level=$rag, cl.risk_score=$score "
                    "MERGE (c)-[:HAS_CLAUSE]->(cl) "
                    "WITH cl WHERE $rk <> 'none' "
                    "MERGE (r:RiskCategory {key:$rk}) SET r.name=$rn "
                    "MERGE (cl)-[:CLASSIFIED_AS]->(r)",
                    id=contract_id, cid=f"{contract_id}::{cl['clause_id']}",
                    h=cl.get("heading", ""), rag=risk.get("rag_level", "green"),
                    score=risk.get("risk_score", 0.0),
                    rk=risk.get("top_category", "none"), rn=risk.get("top_name", ""),
                )
        return self.stats()

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, int]:
        if self._driver is not None:  # pragma: no cover
            with self._driver.session() as session:
                nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
                edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
                return {"nodes": nodes, "relationships": edges}
        g = self._graph
        return {"nodes": g.number_of_nodes(), "relationships": g.number_of_edges()}

    @property
    def graph(self):
        return self._graph

    @property
    def driver(self):
        return self._driver

    def close(self) -> None:  # pragma: no cover
        if self._driver is not None:
            self._driver.close()


class _MiniGraph:
    """Minimal directed-multigraph used only if networkx is unavailable."""

    def __init__(self) -> None:
        self.nodes: Dict[str, dict] = {}
        self.edges: List[tuple] = []

    def add_node(self, node, **attrs):
        self.nodes.setdefault(node, {}).update(attrs)

    def add_edge(self, u, v, **attrs):
        self.add_node(u)
        self.add_node(v)
        self.edges.append((u, v, attrs))

    def number_of_nodes(self) -> int:
        return len(self.nodes)

    def number_of_edges(self) -> int:
        return len(self.edges)

    def successors(self, node):
        return [v for (u, v, _) in self.edges if u == node]

    def predecessors(self, node):
        return [u for (u, v, _) in self.edges if v == node]
