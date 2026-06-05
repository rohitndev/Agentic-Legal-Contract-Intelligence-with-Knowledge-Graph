"""Map identified risk clauses to monetary exposure estimates.

Uses ``data/risk_benchmarks.json`` (per-category industry exposure benchmarks)
to translate a clause's risk category + severity into an expected-exposure
figure, e.g. "unlimited indemnification: $2.5M expected exposure".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from ..config import DATA_DIR


class RiskQuantifier:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (DATA_DIR / "risk_benchmarks.json")
        self.benchmarks: Dict[str, dict] = {}
        if self.path.exists():
            self.benchmarks = json.loads(self.path.read_text(encoding="utf-8"))

    def quantify(self, category: str, risk_score: float, contract_value: float | None = None) -> Dict:
        bench = self.benchmarks.get(category)
        if not bench:
            return {
                "category": category,
                "expected_exposure_usd": 0.0,
                "basis": "no benchmark available",
            }
        base = float(bench["base_exposure_usd"])
        # Scale the benchmark exposure by the normalized clause risk score.
        factor = max(0.2, risk_score / 10.0)
        if contract_value and bench.get("pct_of_contract_value"):
            base = max(base, contract_value * float(bench["pct_of_contract_value"]))
        exposure = round(base * factor, 2)
        return {
            "category": category,
            "expected_exposure_usd": exposure,
            "probability": bench.get("claim_probability", 0.0),
            "basis": bench.get("basis", "CUAD industry benchmark"),
        }

    def portfolio_exposure(self, clauses: List[dict], contract_value: float | None = None) -> Dict:
        """Aggregate expected exposure across all risk-bearing clauses."""
        total = 0.0
        breakdown: List[Dict] = []
        for cl in clauses:
            risk = cl.get("risk", {})
            cat = risk.get("top_category", "none")
            if cat == "none":
                continue
            q = self.quantify(cat, risk.get("risk_score", 0.0), contract_value)
            if q["expected_exposure_usd"] > 0:
                q["clause"] = cl.get("heading", cl.get("clause_id", ""))
                breakdown.append(q)
                total += q["expected_exposure_usd"]
        breakdown.sort(key=lambda x: x["expected_exposure_usd"], reverse=True)
        return {
            "total_expected_exposure_usd": round(total, 2),
            "clause_count": len(breakdown),
            "breakdown": breakdown,
        }
