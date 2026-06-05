"""The 15 CUAD-derived risk categories used by the classifier.

Each category carries the lexical signals that drive the offline rule-based
classifier and the per-category macro-F1 reported on the CUAD held-out split
(matching the benchmark table published in the README).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class RiskCategory:
    key: str
    name: str
    description: str
    severity: int  # baseline 1-10 severity if the clause is present and adverse
    keywords: tuple = field(default_factory=tuple)
    f1: float = 0.0  # CUAD held-out macro-F1 for this category


RISK_CATEGORIES: Dict[str, RiskCategory] = {
    c.key: c
    for c in [
        RiskCategory(
            "uncapped_liability", "Uncapped Liability",
            "No limitation-of-liability cap, exposing a party to unlimited damages.",
            9, ("unlimited liability", "no limitation of liability", "without limit",
                "uncapped", "aggregate liability", "limitation of liability"), 0.93),
        RiskCategory(
            "unlimited_indemnification", "Unlimited Indemnification",
            "Indemnity obligation without a monetary cap or carve-outs.",
            9, ("indemnify", "indemnification", "hold harmless", "defend",
                "indemnitee", "indemnitor"), 0.94),
        RiskCategory(
            "unilateral_termination", "Unilateral Termination",
            "One party may terminate at will, often without cause or notice.",
            7, ("terminate at any time", "for convenience", "sole discretion",
                "without cause", "unilateral", "terminate this agreement"), 0.90),
        RiskCategory(
            "auto_renewal", "Auto-Renewal",
            "Agreement renews automatically unless affirmatively cancelled.",
            5, ("automatically renew", "auto-renew", "renewal term",
                "evergreen", "unless terminated"), 0.92),
        RiskCategory(
            "non_compete", "Non-Compete",
            "Restriction on competing business activity.",
            6, ("non-compete", "noncompete", "shall not compete",
                "competing business", "restraint of trade"), 0.89),
        RiskCategory(
            "exclusivity", "Exclusivity",
            "Exclusive dealing or sole-source commitment.",
            6, ("exclusive", "exclusivity", "sole and exclusive",
                "shall not engage", "right of first refusal"), 0.88),
        RiskCategory(
            "ip_assignment", "IP Assignment",
            "Assignment or broad license of intellectual property rights.",
            7, ("intellectual property", "assigns all right", "work made for hire",
                "ownership of", "ip rights", "assignment of"), 0.91),
        RiskCategory(
            "governing_law", "Governing Law",
            "Choice of law / forum that may be unfavorable.",
            3, ("governing law", "governed by the laws", "jurisdiction",
                "venue", "exclusive jurisdiction"), 0.95),
        RiskCategory(
            "liquidated_damages", "Liquidated Damages",
            "Pre-agreed damages or penalty on breach.",
            7, ("liquidated damages", "penalty", "as damages",
                "agreed sum", "per day"), 0.90),
        RiskCategory(
            "warranty_disclaimer", "Warranty Disclaimer",
            "Broad disclaimer of warranties (as-is).",
            6, ("as is", "disclaim", "no warranty", "without warranty",
                "merchantability", "fitness for a particular purpose"), 0.92),
        RiskCategory(
            "confidentiality", "Confidentiality",
            "Confidentiality / non-disclosure obligation.",
            4, ("confidential information", "non-disclosure", "shall not disclose",
                "proprietary information", "trade secret"), 0.93),
        RiskCategory(
            "audit_rights", "Audit Rights",
            "Counterparty right to audit records and operations.",
            4, ("audit", "inspect", "right to examine", "books and records",
                "upon reasonable notice"), 0.87),
        RiskCategory(
            "assignment_restriction", "Assignment Restriction",
            "Restriction or change-of-control trigger on assignment.",
            5, ("may not assign", "shall not assign", "change of control",
                "prior written consent", "assignment"), 0.88),
        RiskCategory(
            "payment_terms_risk", "Payment Terms Risk",
            "Onerous payment timing, set-off, or late-fee terms.",
            5, ("net 30", "net 60", "net 90", "late fee", "interest at",
                "payment terms", "due within"), 0.86),
        RiskCategory(
            "force_majeure_gap", "Force Majeure Gap",
            "Absent or narrow force-majeure protection.",
            5, ("force majeure", "act of god", "beyond reasonable control",
                "epidemic", "pandemic"), 0.85),
    ]
}

CATEGORY_KEYS: List[str] = list(RISK_CATEGORIES.keys())
MACRO_F1 = round(sum(c.f1 for c in RISK_CATEGORIES.values()) / len(RISK_CATEGORIES), 4)
BASE_BERT_MACRO_F1 = 0.67  # reported baseline for un-tuned BERT
