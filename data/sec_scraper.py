"""SEC EDGAR material-contract scraper.

Fetches 8-K / 10-K material contract exhibits from the free EDGAR full-text
search API for building an extended contract corpus. Network access is required
only when this script is invoked; the rest of the platform runs without it.

Usage:
    python -m data.sec_scraper --query "master services agreement" --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EDGAR_FTS = "https://efts.sec.gov/LATEST/search-index?q={query}&forms=8-K"
HEADERS = {"User-Agent": "legal-contract-intelligence research contact@example.com"}


def scrape(query: str, limit: int = 5, out: Path | None = None) -> list:
    try:
        import requests
    except Exception:
        print("requests not installed; run: pip install requests", file=sys.stderr)
        return []

    url = "https://efts.sec.gov/LATEST/search-index"
    params = {"q": query, "forms": "8-K", "hits": str(limit)}
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])[:limit]
    except Exception as exc:  # network/endpoint changes are tolerated
        print(f"EDGAR request failed ({exc}); returning empty result.", file=sys.stderr)
        return []

    records = [
        {
            "accession": h.get("_id", ""),
            "display_names": h.get("_source", {}).get("display_names", []),
            "form": h.get("_source", {}).get("file_type", ""),
        }
        for h in hits
    ]
    if out:
        out.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Wrote {len(records)} records to {out}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEC EDGAR material-contract scraper")
    parser.add_argument("--query", default="master services agreement")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    scrape(args.query, args.limit, args.out)
