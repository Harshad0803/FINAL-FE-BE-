import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Ensure the package path includes the parent (24-06) so `main` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import app


def test_validation_stage3_smoke_payload():
    client = TestClient(app)
    errors = []

    resp = client.get('/validation/intake')
    if resp.status_code != 200:
        errors.append(f"GET /validation/intake returned {resp.status_code}")

    resp = client.post('/validation/agent2', data={'intake_json': json.dumps({})})
    if resp.status_code != 200:
        errors.append(f"POST /validation/agent2 returned {resp.status_code}")
    else:
        payload = resp.json()
        if 'report' not in payload:
            errors.append('POST /validation/agent2 missing report')

    resp = client.post('/validation/stage3/run', data={'intake_json': json.dumps({})})
    if resp.status_code != 200:
        errors.append(f"POST /validation/stage3/run returned {resp.status_code} - {resp.text}")
    else:
        payload = resp.json()
        required_keys = [
            'featureRelevance', 'thresholdChecks', 'ragRules', 'summary',
            'regulatoryAlignment', 'raw_findings', 'replicated_importances',
            'pending_llm_ids', 'llm_ran', 'timestamp'
        ]
        missing = [k for k in required_keys if k not in payload]
        if missing:
            errors.append(f"/validation/stage3/run missing keys: {missing}")

    resp = client.post('/validation/stage3/llm-check', data={'intake_json': json.dumps({})})
    if resp.status_code == 200:
        try:
            resp.json()
        except Exception:
            errors.append('POST /validation/stage3/llm-check returned non-json')

    assert not errors, '\n'.join(errors)
