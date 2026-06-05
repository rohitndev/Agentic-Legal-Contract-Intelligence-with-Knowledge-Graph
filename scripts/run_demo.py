"""Run the full contract intelligence pipeline on a contract and print a report.

Usage:
    python -m scripts.run_demo                         # uses bundled sample MSA
    python -m scripts.run_demo path/to/contract.pdf
    python -m scripts.run_demo path/to/contract.txt --value 5000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output so RAG heatmap icons render on any console (e.g. Windows cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# Allow running as `python scripts/run_demo.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, settings  # noqa: E402
from src.pipeline import ContractIntelligencePipeline  # noqa: E402

RAG_ICON = {"red": "🔴", "amber": "🟡", "green": "🟢"}


def _bar(title: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def main() -> None:
    ap = argparse.ArgumentParser(description="Legal Contract Intelligence demo")
    ap.add_argument("contract", nargs="?", default=str(DATA_DIR / "sample_contracts" / "sample_msa.txt"))
    ap.add_argument("--value", type=float, default=5_000_000, help="Total contract value (USD)")
    args = ap.parse_args()

    _bar("AGENTIC LEGAL CONTRACT INTELLIGENCE WITH KNOWLEDGE GRAPH")
    print("  Active capabilities:")
    for k, v in settings.summary().items():
        print(f"    - {k:<16}: {v}")

    pipeline = ContractIntelligencePipeline()
    print(f"\n  Analyzing: {args.contract}")
    result = pipeline.analyze_file(args.contract, contract_value=args.value)

    _bar("CONTRACT OVERVIEW")
    print(f"  Contract ID        : {result.contract_id}")
    print(f"  Parties            : {', '.join(result.parties) or 'n/a'}")
    print(f"  Clauses analyzed   : {result.clause_count}")
    print(f"  Overall risk score : {result.overall_risk_score}/10")
    print(f"  RAG heatmap        : "
          f"{RAG_ICON['red']} {result.rag_summary['red']} red  "
          f"{RAG_ICON['amber']} {result.rag_summary['amber']} amber  "
          f"{RAG_ICON['green']} {result.rag_summary['green']} green")
    print(f"  Knowledge graph    : {result.graph_backend} "
          f"({result.graph_stats['nodes']} nodes, {result.graph_stats['relationships']} edges)")
    print(f"  Classifier macro-F1: {result.macro_f1}")
    print(f"  Elapsed            : {result.elapsed_seconds}s")

    _bar("CLAUSE RISK HEATMAP")
    for cl in result.clauses:
        risk = cl["risk"]
        icon = RAG_ICON.get(risk["rag_level"], "⚪")
        heading = (cl["heading"][:42]).ljust(42)
        print(f"  {icon} {cl['clause_id']}  {heading}  "
              f"{risk['top_name']:<26} {risk['risk_score']:>4}/10")

    _bar("GRAPHRAG COMMUNITY SUMMARIES (multi-hop risk)")
    for c in result.community_summaries[:6]:
        print(f"  • {c['name']} (x{c['clause_count']}, avg {c['avg_risk_score']}/10)")
        print(f"      {c['summary']}")

    _bar("MONETARY EXPOSURE QUANTIFICATION")
    exp = result.exposure
    print(f"  Total expected exposure: ${exp['total_expected_exposure_usd']:,.0f}")
    for b in exp["breakdown"][:8]:
        print(f"    - {b['clause'][:40]:<40} ${b['expected_exposure_usd']:>12,.0f}  "
              f"({b['category']})")

    _bar("DONE")
    print("  Start the API with:  uvicorn api.main:app --reload")
    print("  Open docs at:        http://localhost:8000/docs\n")


if __name__ == "__main__":
    main()
