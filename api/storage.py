"""Contract / report storage abstraction.

Persists uploaded contract PDFs and JSON analysis reports to AWS S3 when
``AWS_BUCKET`` is configured (and ``boto3`` is installed), otherwise to the
local ``storage/`` directory. Same interface either way.
"""

from __future__ import annotations

import json
from typing import Optional

from src.config import STORAGE_DIR, settings


class Storage:
    def __init__(self) -> None:
        self.backend = "local"
        self._s3 = None
        if settings.s3_enabled:
            self._init_s3()
        (STORAGE_DIR / "contracts").mkdir(parents=True, exist_ok=True)
        (STORAGE_DIR / "reports").mkdir(parents=True, exist_ok=True)

    def _init_s3(self) -> None:  # pragma: no cover - requires AWS
        try:
            import boto3

            self._s3 = boto3.client("s3", region_name=settings.aws_region)
            self.backend = "s3"
        except Exception:
            self._s3 = None
            self.backend = "local"

    # ------------------------------------------------------------------
    def put_contract(self, filename: str, data: bytes) -> str:
        key = f"contracts/{filename}"
        if self._s3 is not None:  # pragma: no cover
            self._s3.put_object(Bucket=settings.aws_bucket, Key=key, Body=data)
            return f"s3://{settings.aws_bucket}/{key}"
        path = STORAGE_DIR / key
        path.write_bytes(data)
        return str(path)

    def put_report(self, report_id: str, report: dict) -> str:
        key = f"reports/{report_id}.json"
        payload = json.dumps(report, indent=2).encode("utf-8")
        if self._s3 is not None:  # pragma: no cover
            self._s3.put_object(Bucket=settings.aws_bucket, Key=key, Body=payload)
            return f"s3://{settings.aws_bucket}/{key}"
        path = STORAGE_DIR / key
        path.write_bytes(payload)
        return str(path)

    def get_report(self, report_id: str) -> Optional[dict]:
        key = f"reports/{report_id}.json"
        if self._s3 is not None:  # pragma: no cover
            try:
                obj = self._s3.get_object(Bucket=settings.aws_bucket, Key=key)
                return json.loads(obj["Body"].read())
            except Exception:
                return None
        path = STORAGE_DIR / key
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None


storage = Storage()
