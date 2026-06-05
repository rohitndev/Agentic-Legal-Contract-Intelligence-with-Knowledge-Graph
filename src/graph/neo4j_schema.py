"""Neo4j schema definition for the Legal Knowledge Graph.

Node labels:   Contract, Party, Clause, RiskCategory, Obligation
Relationships: (:Contract)-[:HAS_CLAUSE]->(:Clause)
               (:Contract)-[:HAS_PARTY]->(:Party)
               (:Clause)-[:CLASSIFIED_AS]->(:RiskCategory)
               (:Clause)-[:IMPOSES]->(:Obligation)
               (:Obligation)-[:OWED_BY]->(:Party)

These constraints/indexes are applied automatically when a Neo4j connection is
available; they are no-ops for the in-memory backend.
"""

SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT contract_id IF NOT EXISTS FOR (c:Contract) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT clause_id IF NOT EXISTS FOR (cl:Clause) REQUIRE cl.id IS UNIQUE",
    "CREATE CONSTRAINT party_name IF NOT EXISTS FOR (p:Party) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT risk_key IF NOT EXISTS FOR (r:RiskCategory) REQUIRE r.key IS UNIQUE",
    "CREATE INDEX clause_rag IF NOT EXISTS FOR (cl:Clause) ON (cl.rag_level)",
]
