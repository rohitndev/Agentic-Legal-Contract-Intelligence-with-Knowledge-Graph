"""Legal Knowledge Graph: Neo4j-backed (or in-memory) contract entity graph."""

from .graph_builder import KnowledgeGraph
from .queries import GraphQueries

__all__ = ["KnowledgeGraph", "GraphQueries"]
