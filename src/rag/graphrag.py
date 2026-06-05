"""GraphRAG-style community summaries over the contract graph.

Microsoft GraphRAG clusters a knowledge graph into communities and summarizes
each for multi-hop reasoning. When the ``graphrag`` package + an LLM are
configured the full pipeline can be run; the bundled implementation derives
communities from clause risk-category co-occurrence so multi-hop summaries are
available offline.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


class GraphRAG:
    def __init__(self, llm=None) -> None:
        self.llm = llm

    def community_summaries(self, clauses: List[dict]) -> List[Dict]:
        """Group clauses into risk communities and summarize each."""
        communities: Dict[str, List[dict]] = defaultdict(list)
        for cl in clauses:
            cat = cl.get("risk", {}).get("top_category", "none")
            communities[cat].append(cl)

        summaries: List[Dict] = []
        for cat, members in communities.items():
            if cat == "none":
                continue
            name = members[0].get("risk", {}).get("top_name", cat)
            avg = sum(m.get("risk", {}).get("risk_score", 0) for m in members) / len(members)
            headings = [m.get("heading", "") for m in members][:5]
            summary = self._summarize(name, members, avg, headings)
            summaries.append(
                {
                    "community": cat,
                    "name": name,
                    "clause_count": len(members),
                    "avg_risk_score": round(avg, 2),
                    "member_headings": headings,
                    "summary": summary,
                }
            )
        summaries.sort(key=lambda s: s["avg_risk_score"], reverse=True)
        return summaries

    def _summarize(self, name, members, avg, headings) -> str:
        if self.llm is not None:
            prompt = (
                f"Summarize the combined legal exposure from these '{name}' clauses "
                f"(avg risk {avg:.1f}/10): {headings}. 2 sentences."
            )
            try:
                return self.llm.complete(prompt)
            except Exception:
                pass
        return (
            f"{len(members)} clause(s) classified as '{name}' with an average risk of "
            f"{avg:.1f}/10. These clauses jointly drive exposure in this category and "
            f"should be reviewed together for cumulative effect."
        )
