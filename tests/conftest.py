"""Shared test fixtures and path setup."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR  # noqa: E402


@pytest.fixture(scope="session")
def sample_contract_text() -> str:
    return (DATA_DIR / "sample_contracts" / "sample_msa.txt").read_text(encoding="utf-8")
