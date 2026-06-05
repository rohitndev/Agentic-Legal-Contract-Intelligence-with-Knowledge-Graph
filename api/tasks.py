"""Async contract processing.

When ``CELERY_BROKER_URL`` is set a Celery app is created and ``analyze_contract``
becomes a real task; otherwise the same function runs synchronously in-process so
the upload endpoint always works without a broker.
"""

from __future__ import annotations

from typing import Optional

from src.config import settings
from src.pipeline import ContractIntelligencePipeline

from .storage import storage

_pipeline: Optional[ContractIntelligencePipeline] = None


def get_pipeline() -> ContractIntelligencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ContractIntelligencePipeline()
    return _pipeline


def _run_analysis(report_id: str, data: bytes, filename: str, contract_value: Optional[float]) -> dict:
    result = get_pipeline().analyze_bytes(data, filename, contract_value)
    report = result.to_dict()
    report["report_id"] = report_id
    storage.put_report(report_id, report)
    return report


# ----------------------------------------------------------------------
celery_app = None
if settings.celery_enabled:  # pragma: no cover - requires broker
    try:
        from celery import Celery

        celery_app = Celery(
            "legal_intelligence",
            broker=settings.celery_broker_url,
            backend=settings.celery_result_backend or settings.celery_broker_url,
        )

        @celery_app.task(name="analyze_contract")
        def analyze_contract_task(report_id, data_b64, filename, contract_value):
            import base64

            return _run_analysis(report_id, base64.b64decode(data_b64), filename, contract_value)

    except Exception:
        celery_app = None


def submit_analysis(report_id: str, data: bytes, filename: str, contract_value: Optional[float]):
    """Return (mode, task_id_or_report). Celery if available, else synchronous."""
    if celery_app is not None:  # pragma: no cover - requires broker
        import base64

        async_result = analyze_contract_task.delay(
            report_id, base64.b64encode(data).decode(), filename, contract_value
        )
        return "async", async_result.id
    report = _run_analysis(report_id, data, filename, contract_value)
    return "sync", report
