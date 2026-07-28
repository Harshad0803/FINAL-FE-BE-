import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


def test_group_stage2_filters_threshold_checks_by_framework():
    report = {
        "findings_by_stage": {
            "Stage 2: Data Validation": [
                {
                    "check_id": "2.1",
                    "title": "RBI finding",
                    "severity": "HIGH",
                    "status": "FAIL",
                    "source": "RBI",
                    "principle": "Chapter IV Section 26(1)",
                    "observed": "1",
                    "threshold": "0",
                    "detail": "RBI detail",
                },
                {
                    "check_id": "2.2",
                    "title": "IFRS finding",
                    "severity": "MEDIUM",
                    "status": "WARN",
                    "source": "IFRS 9",
                    "principle": "B5.5.28",
                    "observed": "2",
                    "threshold": "1",
                    "detail": "IFRS detail",
                },
                {
                    "check_id": "2.3",
                    "title": "SS1/23 finding",
                    "severity": "LOW",
                    "status": "PASS",
                    "source": "SS1/23",
                    "principle": "P3.2",
                    "observed": "3",
                    "threshold": "2",
                    "detail": "SS detail",
                },
            ]
        }
    }

    grouped = main._group_stage2(report, selected_frameworks=["RBI"])

    assert [item["check_id"] for item in grouped["thresholdChecks"]] == ["2.1"]
    assert grouped["summary"]["total"] == 1
    assert grouped["summary"]["fail"] == 1
    assert grouped["regulatoryAlignment"]["regulatory_references"] == ["RBI Model Risk Management"]


def test_stage2_llm_check_follows_selected_frameworks(monkeypatch):
    class DummyAgent:
        def __init__(self):
            self.calls = []

        def check_documents_with_llm(self, docs, stage=None, only_rule_ids=None, frameworks=None):
            self.calls.append({
                "docs": docs,
                "stage": stage,
                "only_rule_ids": only_rule_ids,
                "frameworks": frameworks,
            })
            return []

    dummy_agent = DummyAgent()
    monkeypatch.setattr(main, "_load_source_agent2", lambda: dummy_agent)

    client = TestClient(main.app)
    response = client.post(
        "/validation/stage2/llm-check",
        data={"intake_json": json.dumps({"frameworks": ["RBI"]})},
    )

    assert response.status_code == 200
    assert dummy_agent.calls[0]["frameworks"] == ["RBI"]
