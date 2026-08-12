import asyncio
import json

from main import validation_stage7_run


def test_regulatory_stage7_summary_counts_pending_and_na_from_individual_checks():
    intake = {
        "frameworks": ["IFRS 9", "IFRS 7", "SS1/23", "SS11/13"],
    }

    result = asyncio.run(validation_stage7_run(intake_json=json.dumps(intake)))
    checks = result["checks"]
    summary = result["summary"]

    assert len(checks) == 10
    assert [c["check_id"] for c in checks] == [
        "7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "7.9", "7.10",
    ]

    status_counts = {
        "PASS": sum(1 for c in checks if c.get("status") == "PASS"),
        "WARN": sum(1 for c in checks if c.get("status") == "WARN"),
        "FAIL": sum(1 for c in checks if c.get("status") == "FAIL"),
        "PENDING": sum(1 for c in checks if c.get("status") == "PENDING"),
        "N/A": sum(1 for c in checks if c.get("status") not in ("PASS", "WARN", "FAIL", "PENDING")),
    }

    assert status_counts["PENDING"] == 6
    assert status_counts["N/A"] == 0

    assert summary["total"] == len(checks)
    assert summary["pass"] == status_counts["PASS"]
    assert summary["warn"] == status_counts["WARN"]
    assert summary["fail"] == status_counts["FAIL"]
    assert summary["na"] == status_counts["N/A"]
    assert summary["pending"] == status_counts["PENDING"]
