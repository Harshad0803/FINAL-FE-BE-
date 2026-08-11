import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import app


def test_stage3_payload_contains_structured_sections():
    client = TestClient(app)
    response = client.post(
        "/validation/stage3/run",
        data={
            "intake_json": json.dumps({
                "algorithm": "Logistic Regression",
                "methodology": "Logistic Regression",
                "calibration_method": "Platt scaling",
                "default_definition": "Default indicator",
                "independence_confirmed": True,
                "assumptions": ["linearity", "independence"],
                "limitations": ["limited sample"],
            })
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert "featureRelevance" in body
    assert "thresholdChecks" in body
    assert "ragRules" in body
    assert "summary" in body
    assert "regulatoryAlignment" in body
    assert "raw_findings" in body
    assert "replicated_importances" in body
    assert "pending_llm_ids" in body
    assert "llm_ran" in body
    assert "timestamp" in body

    assert isinstance(body["thresholdChecks"], list)
    assert isinstance(body["ragRules"], list)
    assert isinstance(body["summary"], dict)
    assert isinstance(body["regulatoryAlignment"], dict)

    regulatory = body["regulatoryAlignment"]
    assert "verdict" in regulatory
    assert "counts" in regulatory
    assert "remediation_summary" in regulatory
    assert "regulatory_references" in regulatory
