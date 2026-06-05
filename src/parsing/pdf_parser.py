"""PDF parsing with PDFPlumber + PyMuPDF.

PDFPlumber preserves layout and extracts tables; PyMuPDF is a fast fallback for
plain text. Both are optional at import time so the pipeline can also accept raw
``.txt`` contracts (useful for tests and the bundled samples).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:
    import pdfplumber  # type: ignore

    _HAS_PDFPLUMBER = True
except Exception:  # pragma: no cover
    _HAS_PDFPLUMBER = False

try:
    import fitz  # PyMuPDF  # type: ignore

    _HAS_PYMUPDF = True
except Exception:  # pragma: no cover
    _HAS_PYMUPDF = False


@dataclass
class ParsedContract:
    """Raw text + extracted tables from a contract document."""

    source: str
    text: str
    tables: List[List[List[str]]] = field(default_factory=list)
    page_count: int = 0
    engine: str = "text"

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class PDFParser:
    """Extract structured text from PDF or text contract files."""

    def parse(self, path: str | Path) -> ParsedContract:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix.lower() == ".pdf":
            return self._parse_pdf(path)
        return self._parse_text(path)

    def parse_bytes(self, data: bytes, filename: str) -> ParsedContract:
        """Parse an uploaded payload (used by the API upload endpoint)."""
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            # Read from an in-memory buffer that pdfplumber/fitz can consume.
            import io

            if _HAS_PDFPLUMBER:
                return self._parse_pdf_stream(io.BytesIO(data), filename)
            if _HAS_PYMUPDF:
                doc = fitz.open(stream=data, filetype="pdf")
                text = "\n".join(page.get_text() for page in doc)
                return ParsedContract(source=filename, text=text, page_count=doc.page_count, engine="pymupdf")
            raise RuntimeError("No PDF engine installed; upload a .txt contract or install pdfplumber.")
        return ParsedContract(source=filename, text=data.decode("utf-8", errors="ignore"))

    # ------------------------------------------------------------------
    def _parse_pdf(self, path: Path) -> ParsedContract:
        if _HAS_PDFPLUMBER:
            with open(path, "rb") as fh:
                import io

                return self._parse_pdf_stream(io.BytesIO(fh.read()), str(path))
        if _HAS_PYMUPDF:
            doc = fitz.open(path)
            text = "\n".join(page.get_text() for page in doc)
            return ParsedContract(source=str(path), text=text, page_count=doc.page_count, engine="pymupdf")
        raise RuntimeError(
            "Neither pdfplumber nor PyMuPDF is installed. Install requirements or use a .txt contract."
        )

    def _parse_pdf_stream(self, stream, source: str) -> ParsedContract:
        texts: List[str] = []
        tables: List[List[List[str]]] = []
        with pdfplumber.open(stream) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
                for table in page.extract_tables() or []:
                    tables.append(table)
            page_count = len(pdf.pages)
        return ParsedContract(
            source=source,
            text="\n".join(texts),
            tables=tables,
            page_count=page_count,
            engine="pdfplumber",
        )

    def _parse_text(self, path: Path) -> ParsedContract:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return ParsedContract(source=str(path), text=text, page_count=1, engine="text")
