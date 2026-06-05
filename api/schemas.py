"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AnalyzeTextRequest(BaseModel):
    text: str = Field(..., description="Raw contract text to analyze")
    source: str = Field("inline", description="Label for the contract source")
    contract_value: Optional[float] = Field(None, description="Total contract value in USD (optional)")


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Negotiation session identifier")
    message: str = Field(..., description="User message to the negotiation agent")
    clause_id: Optional[str] = Field(None, description="Clause to negotiate, from a prior report")
    report_id: Optional[str] = Field(None, description="Report containing the clause")


class HealthResponse(BaseModel):
    status: str
    version: str
    capabilities: Dict[str, str]


class UploadResponse(BaseModel):
    report_id: str
    status: str
    location: str
    async_task_id: Optional[str] = None


class AnalysisSummary(BaseModel):
    contract_id: str
    parties: List[str]
    clause_count: int
    overall_risk_score: float
    rag_summary: Dict[str, int]
    total_expected_exposure_usd: float
    macro_f1: float
