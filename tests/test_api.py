"""End-to-end API tests using FastAPI's TestClient."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "knowledge_graph" in body["capabilities"]


def test_analyze_text_endpoint(sample_contract_text):
    resp = client.post("/contracts/analyze",
                        json={"text": sample_contract_text, "source": "test", "contract_value": 1_000_000})
    assert resp.status_code == 200
    report = resp.json()
    assert report["clause_count"] >= 10
    assert report["exposure"]["total_expected_exposure_usd"] > 0
    assert "report_id" in report


def test_report_roundtrip(sample_contract_text):
    resp = client.post("/contracts/analyze", json={"text": sample_contract_text})
    report_id = resp.json()["report_id"]
    fetched = client.get(f"/reports/{report_id}")
    assert fetched.status_code == 200
    assert fetched.json()["contract_id"] == resp.json()["contract_id"]


def test_agent_chat():
    resp = client.post("/agent/chat",
                        json={"session_id": "s1", "message": "What are the risks of unlimited liability?"})
    assert resp.status_code == 200
    assert resp.json()["reply"]


def test_upload_contract():
    files = {"file": ("contract.txt", b"8. INDEMNIFICATION\nProvider shall indemnify without limitation.",
                      "text/plain")}
    resp = client.post("/contracts/upload", files=files)
    assert resp.status_code == 200
    assert resp.json()["status"] in {"completed", "processing"}
