"""
persistence.py - Lightweight CSV-based run logging for the Aegis Credit backend.

Appends one row per completed pipeline stage to a CSV log so runs survive a
server restart and can be looked up later. No SQLite / Excel involved. Full
step output is saved alongside as a JSON artifact so the UI can resume state
without recomputation.

To persist a new step anywhere in the app:
1. Pick log_file: DEV_PIPELINE_LOG or VALIDATION_PIPELINE_LOG (or define a
   new constant here if it's a new workspace).
2. Right before your endpoint's return statement, call:

       persistence.log_event(
           log_file,
           stage="your_stage_name",
           payload={...short summary fields...},
           full_payload=result,  # the actual dict you're about to return
       )

   wrapped in try/except so persistence failures never break the endpoint.

No registration list, no central config file to update - new stages
automatically appear in /history/dev, /history/validation, /history/latest,
/history/artifact, and the Activity Log page.
"""

import csv
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
APP_DATA_DIR = BACKEND_DIR / "app_data"
ARTIFACTS_DIR = APP_DATA_DIR / "artifacts"

APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEV_PIPELINE_LOG = "dev_pipeline_log.csv"
VALIDATION_PIPELINE_LOG = "validation_pipeline_log.csv"

_FIELDNAMES = ["run_id", "timestamp", "stage", "summary", "artifact_path"]


def save_artifact(log_file: str, stage: str, run_id: str, full_payload: dict) -> str:
    log_stem = Path(log_file).stem
    artifact_dir = ARTIFACTS_DIR / log_stem / stage
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{run_id}.json"
    artifact_path.write_text(json.dumps(full_payload, default=str, indent=2), encoding="utf-8")
    return str(artifact_path.relative_to(BACKEND_DIR))


def log_event(log_file: str, stage: str, payload: dict, full_payload: Optional[dict] = None) -> str:
    run_id = str(uuid.uuid4())
    timestamp = pd.Timestamp.now().isoformat()

    artifact_path = ""
    if full_payload is not None:
        artifact_path = save_artifact(log_file, stage, run_id, full_payload)

    row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "stage": stage,
        "summary": json.dumps(payload, default=str),
        "artifact_path": artifact_path,
    }

    log_path = APP_DATA_DIR / log_file
    file_exists = log_path.exists()

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return run_id


def read_log(log_file: str) -> list:
    log_path = APP_DATA_DIR / log_file
    if not log_path.exists():
        return []

    rows = []
    with open(log_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary_raw = row.get("summary")
            try:
                row["summary"] = json.loads(summary_raw) if summary_raw else None
            except (json.JSONDecodeError, TypeError):
                pass
            rows.append(row)

    return rows


def get_latest(log_file: str, stage: str) -> Optional[Dict[str, Any]]:
    rows = [r for r in read_log(log_file) if r.get("stage") == stage]
    if not rows:
        return None

    latest = max(rows, key=lambda r: r.get("timestamp") or "")

    full_payload = None
    artifact_path = latest.get("artifact_path")
    if artifact_path:
        candidate = BACKEND_DIR / artifact_path
        if candidate.exists():
            try:
                full_payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                full_payload = None

    return {
        "run_id": latest.get("run_id"),
        "timestamp": latest.get("timestamp"),
        "summary": latest.get("summary"),
        "full_payload": full_payload,
    }


def get_artifact(log_file: str, run_id: str) -> Optional[Dict[str, Any]]:
    rows = read_log(log_file)
    match = next((r for r in rows if r.get("run_id") == run_id), None)
    if match is None:
        return None

    artifact_path = match.get("artifact_path")
    if not artifact_path:
        return None

    candidate = BACKEND_DIR / artifact_path
    if not candidate.exists():
        return None

    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
