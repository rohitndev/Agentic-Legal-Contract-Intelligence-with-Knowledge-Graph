"""Track redline history across a negotiation session.

A redline records a proposed change to a clause: the original text, the
suggested rewrite, the rationale, and its status (proposed/accepted/rejected).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Redline:
    clause_id: str
    original: str
    proposed: str
    rationale: str
    status: str = "proposed"
    revision: int = 1
    ts: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict:
        return {
            "clause_id": self.clause_id,
            "original": self.original,
            "proposed": self.proposed,
            "rationale": self.rationale,
            "status": self.status,
            "revision": self.revision,
        }


class RedlineTracker:
    def __init__(self) -> None:
        self._history: List[Redline] = []

    def add(self, clause_id: str, original: str, proposed: str, rationale: str) -> Redline:
        revision = sum(1 for r in self._history if r.clause_id == clause_id) + 1
        rl = Redline(clause_id, original, proposed, rationale, revision=revision)
        self._history.append(rl)
        return rl

    def set_status(self, clause_id: str, status: str) -> None:
        for rl in self._history:
            if rl.clause_id == clause_id:
                rl.status = status

    def history(self) -> List[Dict]:
        return [r.to_dict() for r in self._history]

    def __len__(self) -> int:
        return len(self._history)
