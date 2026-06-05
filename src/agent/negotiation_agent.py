"""Multi-turn negotiation agent.

Implemented as a small state machine over the steps:
    classify_intent -> retrieve_context -> generate_response -> track_redline

When the ``langgraph`` package is installed the same nodes are wired into a
``StateGraph`` for production; otherwise the equivalent sequential executor runs
the nodes directly. Conversation context (turns + redline history) is preserved
across calls so the agent supports genuine multi-turn negotiation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..rag.retriever import Retriever
from .llm import LLMClient
from .redline_tracker import RedlineTracker


@dataclass
class AgentState:
    session_id: str
    turns: List[Dict] = field(default_factory=list)
    last_intent: str = ""


class NegotiationAgent:
    """Conversational agent that proposes redlines and tracks negotiation."""

    def __init__(self, retriever: Optional[Retriever] = None, llm: Optional[LLMClient] = None) -> None:
        self.retriever = retriever or Retriever()
        self.llm = llm or LLMClient()
        self.sessions: Dict[str, AgentState] = {}
        self.redlines: Dict[str, RedlineTracker] = {}
        self._graph = self._build_langgraph()

    # ------------------------------------------------------------------
    def _build_langgraph(self):  # pragma: no cover - optional heavy path
        try:
            from langgraph.graph import END, StateGraph

            sg = StateGraph(dict)
            sg.add_node("classify", lambda s: {**s, "intent": self._classify(s["message"])})
            sg.add_node("retrieve", lambda s: {**s, "context": self._retrieve(s["message"])})
            sg.add_node("respond", lambda s: {**s, "reply": self._respond(s)})
            sg.set_entry_point("classify")
            sg.add_edge("classify", "retrieve")
            sg.add_edge("retrieve", "respond")
            sg.add_edge("respond", END)
            return sg.compile()
        except Exception:
            return None

    # ------------------------------------------------------------------
    def chat(self, session_id: str, message: str, clause: Optional[dict] = None) -> Dict:
        state = self.sessions.setdefault(session_id, AgentState(session_id))
        tracker = self.redlines.setdefault(session_id, RedlineTracker())

        if self._graph is not None:  # pragma: no cover - optional
            try:
                result = self._graph.invoke({"message": message, "clause": clause})
                intent, context, reply = result["intent"], result["context"], result["reply"]
            except Exception:
                intent, context, reply = self._run_sequential(message, clause)
        else:
            intent, context, reply = self._run_sequential(message, clause)

        redline = None
        if intent == "redline" and clause is not None:
            rewrite = self._propose_rewrite(clause, context)
            rl = tracker.add(
                clause_id=clause.get("clause_id", "?"),
                original=clause.get("text", "")[:500],
                proposed=rewrite,
                rationale=reply,
            )
            redline = rl.to_dict()

        state.turns.append({"user": message, "agent": reply, "intent": intent})
        state.last_intent = intent
        return {
            "session_id": session_id,
            "intent": intent,
            "reply": reply,
            "precedents": context,
            "redline": redline,
            "turn": len(state.turns),
            "redline_history": tracker.history(),
        }

    def _run_sequential(self, message, clause):
        intent = self._classify(message)
        context = self._retrieve(message if clause is None else clause.get("text", message))
        reply = self._respond({"message": message, "intent": intent, "context": context, "clause": clause})
        return intent, context, reply

    # ---- nodes --------------------------------------------------------
    @staticmethod
    def _classify(message: str) -> str:
        low = message.lower()
        if any(k in low for k in ("rewrite", "redline", "propose", "revise", "change")):
            return "redline"
        if any(k in low for k in ("risk", "exposure", "dangerous", "liable")):
            return "risk_query"
        if any(k in low for k in ("similar", "precedent", "market", "standard")):
            return "precedent"
        return "general"

    def _retrieve(self, text: str) -> List[Dict]:
        return self.retriever.similar_precedents(text, top_k=3)

    def _respond(self, state: Dict) -> str:
        intent = state["intent"]
        context = state.get("context", [])
        clause = state.get("clause")
        ctx_note = ""
        if context:
            ctx_note = f" Comparable CUAD precedent: '{context[0]['category_name']}' " \
                       f"(similarity {context[0]['similarity']})."
        if intent == "redline" and clause:
            return (self.llm.complete(
                f"Explain why this clause needs a redline: {clause.get('text','')[:600]}",
                system="You are a senior contracts attorney.") + ctx_note)
        if intent == "risk_query":
            return self.llm.complete(
                f"Give a risk narrative for: {state['message']}",
                system="You are a senior contracts attorney.") + ctx_note
        return self.llm.complete(state["message"], system="You are a helpful contracts attorney.") + ctx_note

    def _propose_rewrite(self, clause: dict, context: List[Dict]) -> str:
        return self.llm.complete(
            f"Rewrite this clause to be balanced and mutual, adding reasonable caps: "
            f"{clause.get('text','')[:600]}",
            system="You are a contracts attorney. Output only the revised clause.",
        )
