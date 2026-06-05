"""API routes for contract intelligence."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src import __version__
from src.agent import NegotiationAgent
from src.config import settings

from .schemas import (
    AnalyzeTextRequest,
    ChatRequest,
    HealthResponse,
    UploadResponse,
)
from .storage import storage
from .tasks import get_pipeline, submit_analysis

router = APIRouter()
_agent: NegotiationAgent | None = None


def _get_agent() -> NegotiationAgent:
    global _agent
    if _agent is None:
        _agent = NegotiationAgent()
    return _agent


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, capabilities=settings.summary())


@router.post("/contracts/upload", response_model=UploadResponse, tags=["contracts"])
async def upload_contract(
    file: UploadFile = File(...),
    contract_value: float | None = Form(None),
) -> UploadResponse:
    """Upload a contract PDF/TXT, persist it, and run the analysis pipeline."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    report_id = uuid.uuid4().hex[:12]
    location = storage.put_contract(file.filename or f"{report_id}.pdf", data)
    mode, payload = submit_analysis(report_id, data, file.filename or "contract.pdf", contract_value)
    return UploadResponse(
        report_id=report_id,
        status="processing" if mode == "async" else "completed",
        location=location,
        async_task_id=payload if mode == "async" else None,
    )


@router.post("/contracts/analyze", tags=["contracts"])
def analyze_text(req: AnalyzeTextRequest) -> dict:
    """Analyze raw contract text synchronously and return the full report."""
    result = get_pipeline().analyze_text(req.text, req.source, req.contract_value)
    report = result.to_dict()
    report_id = uuid.uuid4().hex[:12]
    report["report_id"] = report_id
    storage.put_report(report_id, report)
    return report


@router.get("/reports/{report_id}", tags=["contracts"])
def get_report(report_id: str) -> dict:
    report = storage.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.get("/reports/{report_id}/exposure", tags=["risk"])
def get_exposure(report_id: str) -> dict:
    report = storage.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report.get("exposure", {})


@router.post("/agent/chat", tags=["agent"])
def chat(req: ChatRequest) -> dict:
    """Multi-turn negotiation chat. Optionally targets a clause from a report."""
    clause = None
    if req.report_id and req.clause_id:
        report = storage.get_report(req.report_id)
        if report:
            clause = next(
                (c for c in report.get("clauses", []) if c["clause_id"] == req.clause_id), None
            )
    return _get_agent().chat(req.session_id, req.message, clause)
