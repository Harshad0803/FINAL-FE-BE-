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
import re
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
MODEL_INVENTORY_FILENAME = "model_inventory.json"

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


def get_model_inventory_path() -> Path:
    return APP_DATA_DIR / MODEL_INVENTORY_FILENAME


def _is_development_record(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    status = str(item.get("status") or "").lower()
    if status.startswith("in development") or status.startswith("development") or status in {"draft", "in dev"}:
        return True
    if item.get("development_date") is not None:
        return True
    return False


def _is_validation_record(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    status = str(item.get("status") or "").lower()
    validation_status = str(item.get("validation_status") or "").lower()
    if "under validation" in status or "validated" in status or "in review" in status:
        return True
    if validation_status or item.get("validation_stage") or item.get("last_validation"):
        return True
    return False


def _is_legacy_empty_validation_record(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    model_id = str(item.get("model_id") or "").strip()
    if not model_id:
        return False
    model_name = str(item.get("model_name") or "").strip()
    if model_name:
        return False
    if item.get("validation_status") or item.get("overall_status") or item.get("validation_stage") or item.get("last_validation"):
        return False
    if item.get("status") or item.get("documentation_path") or item.get("related_model_id"):
        return False
    return True


def _normalize_inventory_store(store: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(store, dict):
        store = {"development": [], "validation": [], "models": [], "data_sources": [], "history": []}

    development_records = store.get("development") if isinstance(store.get("development"), list) else []
    validation_records = store.get("validation") if isinstance(store.get("validation"), list) else []
    legacy_models = store.get("models") if isinstance(store.get("models"), list) else []
    if not development_records and legacy_models:
        development_records = [item for item in legacy_models if _is_development_record(item)]
    if not validation_records and legacy_models:
        validation_records = [item for item in legacy_models if _is_validation_record(item)]
    validation_records = [item for item in validation_records if not _is_legacy_empty_validation_record(item)]

    merged_models = []
    seen_ids = set()
    for collection in (development_records, validation_records):
        for item in collection:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("model_id") or "").strip()
            key = model_id or str(item.get("model_name") or "")
            if key and key not in seen_ids:
                merged_models.append(item)
                seen_ids.add(key)
    for item in legacy_models:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("model_id") or "").strip()
        key = model_id or str(item.get("model_name") or "")
        if key and key not in seen_ids:
            merged_models.append(item)
            seen_ids.add(key)

    store["development"] = development_records
    store["validation"] = validation_records
    store["models"] = merged_models
    store.setdefault("data_sources", [])
    store.setdefault("history", [])
    return store


def load_model_inventory() -> Dict[str, Any]:
    path = get_model_inventory_path()
    if not path.exists():
        return _normalize_inventory_store({"development": [], "validation": [], "models": [], "data_sources": [], "history": []})

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _normalize_inventory_store({"development": [], "validation": [], "models": [], "data_sources": [], "history": []})

    if not isinstance(data, dict):
        return _normalize_inventory_store({"development": [], "validation": [], "models": [], "data_sources": [], "history": []})

    store = {
        "development": data.get("development") if isinstance(data.get("development"), list) else [],
        "validation": data.get("validation") if isinstance(data.get("validation"), list) else [],
        "models": data.get("models") if isinstance(data.get("models"), list) else [],
        "data_sources": data.get("data_sources") if isinstance(data.get("data_sources"), list) else [],
        "history": data.get("history") if isinstance(data.get("history"), list) else [],
    }
    return _normalize_inventory_store(store)


def save_model_inventory(store: Dict[str, Any]) -> None:
    store = _normalize_inventory_store(store)
    path = get_model_inventory_path()
    path.write_text(json.dumps(store, indent=2, default=str), encoding="utf-8")


def _slugify_model_name(name: str) -> str:
    if not name:
        return "model"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    slug = slug.strip("._-") or "model"
    return slug


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("\r", " ").replace("\n", " ")


def _write_model_report_pdf(model_name: str, metadata: Dict[str, Any]) -> str:
    report_dir = APP_DATA_DIR / "model_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _slugify_model_name(model_name)
    report_path = report_dir / f"{safe_name}_model_report.pdf"

    title = f"Model Report: {model_name}"
    summary_lines = [
        f"Model Name: {metadata.get('model_name') or model_name}",
        f"Algorithm: {metadata.get('algorithm') or 'Unknown'}",
        f"Owner: {metadata.get('model_owner') or 'Unknown'}",
        f"Business Unit: {metadata.get('business_unit') or 'Unknown'}",
        f"Model Version: {metadata.get('model_version') or 'Unknown'}",
        f"Development Date: {metadata.get('development_date') or 'Unknown'}",
        f"Status: {metadata.get('status') or 'In Development'}",
        f"Purpose: {metadata.get('business_purpose') or 'Development model'}",
    ]
    content_lines = [title] + summary_lines

    pdf_objects = []
    pdf_objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    pdf_objects.append("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    pdf_objects.append("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>")

    content_stream = "BT\n/F1 18 Tf\n50 750 Td\n(" + _escape_pdf_text(title) + ") Tj\nET\n"
    y = 720
    for line in summary_lines:
        y -= 22
        if y < 60:
            break
        content_stream += "BT\n/F1 11 Tf\n50 " + str(y) + " Td\n(" + _escape_pdf_text(line) + ") Tj\nET\n"
    pdf_objects.append(f"<< /Length {len(content_stream.encode('latin-1'))} >>\nstream\n{content_stream}endstream")
    pdf_objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf_bytes = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj_text in enumerate(pdf_objects, start=1):
        offsets.append(len(pdf_bytes))
        pdf_bytes.extend(f"{index} 0 obj\n{obj_text}\nendobj\n".encode("latin-1"))

    xref_offset = len(pdf_bytes)
    pdf_bytes.extend(f"xref\n0 {len(pdf_objects) + 1}\n".encode("latin-1"))
    pdf_bytes.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf_bytes.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf_bytes.extend(f"trailer\n<< /Size {len(pdf_objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1"))

    report_path.write_bytes(pdf_bytes)
    return str(report_path.relative_to(BACKEND_DIR))


def _ensure_development_documentation_path(model_name: str, model_entry: Optional[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    model_entry = model_entry or {}
    metadata = metadata or model_entry
    current = str(model_entry.get("documentation_path") or "").strip()
    if current and current.endswith(".pdf") and (BACKEND_DIR / current).exists():
        return current

    report_loc = _write_model_report_pdf(model_name, {
        "model_name": model_name,
        "algorithm": model_entry.get("algorithm") or metadata.get("algorithm") or "Unknown",
        "model_owner": model_entry.get("model_owner") or metadata.get("model_owner") or "Unknown",
        "business_unit": model_entry.get("business_unit") or metadata.get("business_unit") or "Unknown",
        "model_version": model_entry.get("model_version") or metadata.get("model_version") or "Unknown",
        "development_date": model_entry.get("development_date") or metadata.get("development_date") or pd.Timestamp.now().strftime("%Y-%m-%d"),
        "status": model_entry.get("status") or metadata.get("status") or "In Development",
        "business_purpose": model_entry.get("business_purpose") or metadata.get("business_purpose") or "Development model",
    })
    return report_loc


def ensure_model_inventory_seed() -> Dict[str, Any]:
    store = load_model_inventory()
    for model in store.get("development", []):
        if not isinstance(model, dict):
            continue
        model_name = str(model.get("model_name") or "").strip()
        status = str(model.get("status") or "").lower()
        if not model_name:
            continue
        if status.startswith("in development") or status.startswith("development") or status in {"draft", "in dev"}:
            generated = _ensure_development_documentation_path(model_name, model)
            if generated:
                model["documentation_path"] = generated
    save_model_inventory(store)
    return store


def next_model_id(store: Dict[str, Any]) -> str:
    used = {item.get("model_id", "") for item in store.get("models", []) if isinstance(item, dict)}
    index = 1
    while True:
        candidate = f"MOD-{index:03d}"
        if candidate not in used:
            return candidate
        index += 1


def append_history_event(store: Dict[str, Any], model_id: str, event: str, description: str, user: Optional[str] = None, validation_run_id: Optional[str] = None) -> Dict[str, Any]:
    entry = {
        "model_id": model_id,
        "event": event,
        "description": description,
        "timestamp": pd.Timestamp.now().isoformat(),
        "user": user or "system",
    }
    if validation_run_id:
        entry["validation_run_id"] = validation_run_id
    store.setdefault("history", []).append(entry)
    return entry


def next_validation_run_id(store: Dict[str, Any]) -> str:
    existing_ids = {
        item.get("validation_run_id")
        for item in store.get("history", [])
        if isinstance(item, dict) and item.get("validation_run_id")
    }
    index = 1
    while True:
        candidate = f"VAL-{index:03d}"
        if candidate not in existing_ids:
            return candidate
        index += 1


def next_validation_model_id(store: Dict[str, Any]) -> str:
    used = {
        str(item.get("model_id", "")).strip()
        for item in store.get("validation", [])
        if isinstance(item, dict) and str(item.get("model_id", "")).strip()
    }
    index = 1
    while True:
        candidate = f"VAL-{index:03d}"
        if candidate not in used:
            return candidate
        index += 1


_PLACEHOLDER_VALUES = {
    "",
    "unknown",
    "tbd",
    "n/a",
    "na",
    "none",
    "null",
    "not provided",
    "not available",
    "unspecified",
    "system",
    "user",
    "custom",
}


def _normalize_metadata_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        values = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(values) if values else None
    text = str(value).strip()
    return text or None


def _is_placeholder_metadata(value: Any) -> bool:
    normalized = _normalize_metadata_value(value)
    if normalized is None:
        return True
    return normalized.lower() in _PLACEHOLDER_VALUES


def _prefer_real_value(*candidates: Any) -> Optional[str]:
    for candidate in candidates:
        value = _normalize_metadata_value(candidate)
        if value and not _is_placeholder_metadata(value):
            return value
    for candidate in candidates:
        value = _normalize_metadata_value(candidate)
        if value:
            return value
    return None


def _find_existing_model_entry(store: Dict[str, Any], business_model_name: Optional[str], model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(store, dict):
        return None

    collections = [
        store.get("development", []),
        store.get("validation", []),
        store.get("models", []),
    ]
    candidate_model_id = (str(model_id or "")).strip()
    if candidate_model_id:
        for collection in collections:
            match = next(
                (
                    item for item in collection
                    if isinstance(item, dict) and str(item.get("model_id") or "").strip() == candidate_model_id
                ),
                None,
            )
            if match is not None:
                return match

    candidate_name = (str(business_model_name or "")).strip()
    if candidate_name:
        normalized_name = candidate_name.casefold()
        for collection in collections:
            match = next(
                (
                    item for item in collection
                    if isinstance(item, dict)
                    and str(item.get("model_name") or "").strip().casefold() == normalized_name
                ),
                None,
            )
            if match is not None:
                return match
    return None


def _find_validation_inventory_entry(store: Dict[str, Any], business_model_name: Optional[str], model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    validation_inventory = store.get("validation", [])
    candidate_model_id = (str(model_id or "")).strip()
    if candidate_model_id:
        match = next(
            (
                item for item in validation_inventory
                if isinstance(item, dict) and str(item.get("model_id") or "").strip() == candidate_model_id
            ),
            None,
        )
        if match is not None:
            return match

    candidate_name = (str(business_model_name or "")).strip()
    if candidate_name:
        normalized_name = candidate_name.casefold()
        return next(
            (
                item for item in validation_inventory
                if isinstance(item, dict)
                and str(item.get("model_name") or "").strip().casefold() == normalized_name
            ),
            None,
        )
    return None


def _find_development_inventory_entry(store: Dict[str, Any], business_model_name: Optional[str], model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    development_inventory = store.get("development", [])
    candidate_model_id = (str(model_id or "")).strip()
    if candidate_model_id:
        match = next(
            (
                item for item in development_inventory
                if isinstance(item, dict) and str(item.get("model_id") or "").strip() == candidate_model_id
            ),
            None,
        )
        if match is not None:
            return match

    candidate_name = (str(business_model_name or "")).strip()
    if candidate_name:
        normalized_name = candidate_name.casefold()
        return next(
            (
                item for item in development_inventory
                if isinstance(item, dict)
                and str(item.get("model_name") or "").strip().casefold() == normalized_name
            ),
            None,
        )
    return None


def _derive_regulatory_framework(intake_payload: Dict[str, Any]) -> Optional[str]:
    frameworks = intake_payload.get("frameworks")
    if isinstance(frameworks, list):
        for framework in frameworks:
            value = _normalize_metadata_value(framework)
            if value and not _is_placeholder_metadata(value):
                return value
    value = _prefer_real_value(
        intake_payload.get("regulatory_framework"),
        intake_payload.get("framework"),
        intake_payload.get("framework_name"),
    )
    return value


def _derive_documentation_path(intake_payload: Dict[str, Any], existing: Optional[Dict[str, Any]] = None) -> Optional[str]:
    for key in (
        "mdd_document_path",
        "documentation_path",
        "document_path",
        "mdd_path",
        "mdd_file_path",
        "documentation_url",
        "document_url",
    ):
        value = intake_payload.get(key)
        normalized = _normalize_metadata_value(value)
        if normalized and not _is_placeholder_metadata(normalized):
            return normalized

    existing_value = existing.get("documentation_path") if isinstance(existing, dict) else None
    normalized_existing = _normalize_metadata_value(existing_value)
    if normalized_existing and not _is_placeholder_metadata(normalized_existing):
        return normalized_existing
    return None


def _derive_overall_validation_status(validation_payload: Dict[str, Any]) -> str:
    data_status = validation_payload.get("data_validation_status") or "Not Started"
    conceptual_status = validation_payload.get("conceptual_soundness_status") or "Not Started"
    backtesting_status = validation_payload.get("backtesting_status") or "Not Started"
    findings_count = int(validation_payload.get("findings_count") or 0)

    completed_with_findings = any(status in {"Completed With Findings"} for status in [data_status, conceptual_status, backtesting_status])
    completed = all(status in {"Completed", "Completed With Findings"} for status in [data_status, conceptual_status, backtesting_status])

    if completed and findings_count > 0:
        return "Validated With Findings"
    if completed:
        return "Validated"
    if any(status in {"In Progress", "Completed", "Completed With Findings"} for status in [data_status, conceptual_status, backtesting_status]):
        return "Under Validation"
    if completed_with_findings:
        return "Validated With Findings"
    return "Under Validation"


def update_model_validation_status(store: Dict[str, Any], model_id: str, stage: str, status: str, findings_count: Optional[int] = None, last_run: Optional[str] = None, description: Optional[str] = None, user: Optional[str] = None) -> Dict[str, Any]:
    validation_entry = next((item for item in store.setdefault("validation", []) if item.get("model_id") == model_id), None)
    if validation_entry is None:
        validation_entry = {
            "model_id": model_id,
            "model_name": "Unknown Model",
            "data_validation_status": "Not Started",
            "conceptual_soundness_status": "Not Started",
            "backtesting_status": "Not Started",
            "overall_status": "Under Validation",
            "findings_count": 0,
            "last_validation": None,
        }
        store.setdefault("validation", []).append(validation_entry)

    if stage == "data_validation":
        validation_entry["data_validation_status"] = status
    elif stage == "conceptual_soundness":
        validation_entry["conceptual_soundness_status"] = status
    elif stage == "backtesting":
        validation_entry["backtesting_status"] = status

    if findings_count is not None:
        validation_entry["findings_count"] = findings_count
    if last_run:
        validation_entry["last_validation"] = last_run

    validation_entry["overall_status"] = _derive_overall_validation_status(validation_entry)
    validation_entry["status"] = validation_entry["overall_status"]
    validation_entry["validation_status"] = validation_entry["overall_status"]
    validation_entry["validation_stage"] = stage
    validation_entry["validation_date"] = last_run or pd.Timestamp.now().isoformat()

    if description:
        append_history_event(store, model_id, stage.replace("_", " ").title(), description, user=user)
    return validation_entry


def create_data_source(store: Dict[str, Any], model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data_sources = store.setdefault("data_sources", [])
    normalized_name = (payload.get("file_name") or "unknown").strip()
    normalized_ref = (payload.get("storage_reference") or normalized_name or "").strip()
    normalized_type = (payload.get("source_type") or "file").strip() or "file"
    matching = next(
        (
            item
            for item in data_sources
            if isinstance(item, dict)
            and item.get("model_id") == model_id
            and (item.get("storage_reference") or item.get("file_name") or "").strip() == normalized_ref
            and (item.get("source_type") or "file").strip() == normalized_type
        ),
        None,
    )
    if matching is not None:
        return matching

    existing_ids = {item.get("data_source_id", "") for item in data_sources if isinstance(item, dict)}
    index = 1
    while True:
        candidate = f"DS-{index:03d}"
        if candidate not in existing_ids:
            break
        index += 1

    entry = {
        "data_source_id": candidate,
        "model_id": model_id,
        "file_name": normalized_name,
        "source_type": normalized_type,
        "purpose": payload.get("purpose") or "Validation dataset",
        "uploaded_by": payload.get("uploaded_by") or "system",
        "uploaded_at": payload.get("uploaded_at") or pd.Timestamp.now().isoformat(),
        "record_count": payload.get("record_count"),
        "column_count": payload.get("column_count"),
        "target_variable": payload.get("target_variable"),
        "storage_reference": normalized_ref,
    }
    data_sources.append(entry)
    append_history_event(store, model_id, "Data Source Linked", f"Linked data source {entry['file_name']} to model {model_id}", user=entry["uploaded_by"])
    return entry


def register_development_model(
    store: Dict[str, Any],
    business_model_name: Optional[str],
    estimator_name: Optional[str],
    intake_payload: Optional[Dict[str, Any]] = None,
    dataset_payload: Optional[Dict[str, Any]] = None,
    user: Optional[str] = None,
) -> Dict[str, Any]:
    intake_payload = intake_payload or {}
    dataset_payload = dataset_payload or {}
    business_model_name = (business_model_name or "").strip() or (estimator_name or "").strip()
    estimator_name = (estimator_name or "").strip() or business_model_name

    development_store = store.setdefault("development", [])
    existing = next(
        (
            item for item in development_store
            if isinstance(item, dict)
            and str(item.get("model_id") or "").strip() == str(intake_payload.get("model_id") or "").strip()
        ),
        None,
    )
    if existing is None:
        existing = next(
            (
                item for item in development_store
                if isinstance(item, dict)
                and str(item.get("model_name") or "").strip().casefold() == business_model_name.casefold()
            ),
            None,
        )

    now = pd.Timestamp.now().isoformat()
    model_owner = _prefer_real_value(
        intake_payload.get("model_owner"),
        intake_payload.get("owner"),
        existing.get("model_owner") if isinstance(existing, dict) else None,
        user,
        "Unknown",
    )
    business_unit = _prefer_real_value(
        intake_payload.get("business_unit"),
        intake_payload.get("owning_team"),
        existing.get("business_unit") if isinstance(existing, dict) else None,
        user,
        "Unknown",
    )
    model_version = _prefer_real_value(
        intake_payload.get("model_version"),
        existing.get("model_version") if isinstance(existing, dict) else None,
        "1.0",
    )
    regulatory_framework = _derive_regulatory_framework(intake_payload) or _prefer_real_value(
        existing.get("regulatory_framework") if isinstance(existing, dict) else None,
        "Unknown",
    )
    model_type = _prefer_real_value(
        intake_payload.get("model_type"),
        existing.get("model_type") if isinstance(existing, dict) else None,
        "Custom",
    ) or "Custom"
    business_purpose = _prefer_real_value(
        intake_payload.get("model_purpose"),
        intake_payload.get("business_purpose"),
        existing.get("business_purpose") if isinstance(existing, dict) else None,
        "Development model",
    ) or "Development model"
    development_date = _prefer_real_value(
        intake_payload.get("development_date"),
        existing.get("development_date") if isinstance(existing, dict) else None,
        now.split("T", 1)[0],
    )
    status = _prefer_real_value(
        existing.get("status") if isinstance(existing, dict) else None,
        intake_payload.get("status"),
        "In Development",
    ) or "In Development"
    documentation_path = _derive_documentation_path(intake_payload, existing)
    if not documentation_path or not str(documentation_path).endswith(".pdf"):
        documentation_path = _ensure_development_documentation_path(
            business_model_name,
            existing,
            {
                "algorithm": estimator_name,
                "model_owner": model_owner,
                "business_unit": business_unit,
                "model_version": model_version,
                "development_date": development_date,
                "status": status,
                "business_purpose": business_purpose,
            },
        )

    if existing is None:
        model_id = next_model_id(store)
        entry = {
            "model_id": model_id,
            "model_name": business_model_name,
            "model_type": model_type,
            "business_purpose": business_purpose,
            "model_owner": model_owner,
            "business_unit": business_unit,
            "model_version": model_version,
            "regulatory_framework": regulatory_framework,
            "algorithm": estimator_name,
            "documentation_path": documentation_path,
            "development_date": development_date,
            "status": status,
            "created_at": now,
            "updated_at": now,
        }
        development_store.append(entry)
        append_history_event(store, model_id, "Model Registered", f"Model {business_model_name} was added to the Development Inventory.", user=user or model_owner or "system")
        existing = entry
    else:
        model_id = existing.get("model_id")
        existing["model_type"] = _prefer_real_value(existing.get("model_type"), model_type, "Custom") or "Custom"
        existing["business_purpose"] = business_purpose
        existing["model_owner"] = model_owner
        existing["business_unit"] = business_unit
        existing["model_version"] = model_version
        existing["regulatory_framework"] = regulatory_framework
        existing["algorithm"] = estimator_name or existing.get("algorithm")
        existing["status"] = status
        existing["development_date"] = development_date
        if documentation_path:
            existing["documentation_path"] = documentation_path
        existing["updated_at"] = now
        append_history_event(store, model_id, "Model Updated", f"Model {business_model_name} metadata was updated in the Development Inventory.", user=user or model_owner or "system")

    combined = list(store.get("development", [])) + list(store.get("validation", []))
    seen = set()
    deduped = []
    for item in combined:
        if not isinstance(item, dict):
            continue
        key = str(item.get("model_id") or item.get("model_name") or "")
        if key and key not in seen:
            deduped.append(item)
            seen.add(key)
    store["models"] = deduped

    if dataset_payload:
        create_data_source(store, model_id, {
            "file_name": dataset_payload.get("file_name") or "development_dataset",
            "storage_reference": dataset_payload.get("storage_reference") or dataset_payload.get("file_name") or "",
            "source_type": dataset_payload.get("source_type") or "file",
            "purpose": dataset_payload.get("purpose") or "Development dataset",
            "record_count": dataset_payload.get("record_count"),
            "column_count": dataset_payload.get("column_count"),
            "target_variable": dataset_payload.get("target_variable"),
            "uploaded_by": dataset_payload.get("uploaded_by") or user or model_owner or "user",
            "uploaded_at": dataset_payload.get("uploaded_at") or now,
        })

    return {
        "model_id": model_id,
        "model_name": business_model_name,
        "algorithm": estimator_name,
    }


def register_validation_run(
    store: Dict[str, Any],
    business_model_name: Optional[str],
    estimator_name: Optional[str],
    intake_payload: Optional[Dict[str, Any]] = None,
    dataset_payload: Optional[Dict[str, Any]] = None,
    target_col: Optional[str] = None,
    user: Optional[str] = None,
) -> Dict[str, Any]:
    intake_payload = intake_payload or {}
    dataset_payload = dataset_payload or {}
    business_model_name = (business_model_name or "").strip() or (estimator_name or "").strip()
    estimator_name = (estimator_name or "").strip() or business_model_name

    validation_store = store.setdefault("validation", [])
    incoming_model_id = _normalize_metadata_value(intake_payload.get("model_id") or dataset_payload.get("model_id"))
    related_development = _find_development_inventory_entry(store, business_model_name, incoming_model_id)
    existing = _find_validation_inventory_entry(store, business_model_name, incoming_model_id)

    now = pd.Timestamp.now().isoformat()
    validation_run_id = next_validation_run_id(store)
    model_owner = _prefer_real_value(
        intake_payload.get("model_owner"),
        intake_payload.get("owner"),
        existing.get("model_owner") if isinstance(existing, dict) else None,
        user,
        "Unknown",
    )
    business_unit = _prefer_real_value(
        intake_payload.get("owning_team"),
        intake_payload.get("business_unit"),
        existing.get("business_unit") if isinstance(existing, dict) else None,
        user,
        "Unknown",
    )
    model_version = _prefer_real_value(
        intake_payload.get("model_version"),
        existing.get("model_version") if isinstance(existing, dict) else None,
        "1.0",
    )
    regulatory_framework = _derive_regulatory_framework(intake_payload) or _prefer_real_value(
        existing.get("regulatory_framework") if isinstance(existing, dict) else None,
        "Unknown",
    )
    documentation_path = _derive_documentation_path(intake_payload, existing)

    if existing is None:
        model_id = next_validation_model_id(store)
        entry = {
            "model_id": model_id,
            "related_model_id": related_development.get("model_id") if isinstance(related_development, dict) and related_development.get("model_id") else incoming_model_id if incoming_model_id and incoming_model_id.startswith("MOD-") else None,
            "model_name": business_model_name,
            "model_type": _prefer_real_value(intake_payload.get("model_type"), "Custom") or "Custom",
            "business_purpose": _prefer_real_value(
                intake_payload.get("model_purpose"),
                intake_payload.get("business_purpose"),
                "Validation model",
            ) or "Validation model",
            "model_owner": model_owner,
            "business_unit": business_unit,
            "model_version": model_version,
            "regulatory_framework": regulatory_framework,
            "algorithm": estimator_name,
            "documentation_path": documentation_path,
            "status": "Under Validation",
            "validation_status": "Under Validation",
            "validation_stage": "data_validation",
            "validation_date": now,
            "created_at": now,
            "updated_at": now,
        }
        validation_store.append(entry)
        append_history_event(store, model_id, "Validation Inventory Registered", f"Validation engagement {business_model_name} was added to the Validation Inventory.", user=user or model_owner or "system")
        existing = entry
    else:
        model_id = existing.get("model_id")
        existing["model_name"] = business_model_name
        existing["related_model_id"] = related_development.get("model_id") if isinstance(related_development, dict) and related_development.get("model_id") else existing.get("related_model_id") or (incoming_model_id if incoming_model_id and incoming_model_id.startswith("MOD-") else None)
        existing["model_type"] = _prefer_real_value(existing.get("model_type"), intake_payload.get("model_type"), "Custom") or "Custom"
        existing["model_owner"] = model_owner
        existing["business_unit"] = business_unit
        existing["model_version"] = model_version
        existing["regulatory_framework"] = regulatory_framework
        if documentation_path:
            existing["documentation_path"] = documentation_path
        if estimator_name:
            existing["algorithm"] = estimator_name
        existing["validation_status"] = "Under Validation"
        existing["validation_stage"] = "data_validation"
        existing["validation_date"] = now
        existing["updated_at"] = now
        if existing.get("status") is None:
            existing["status"] = "Under Validation"
        model_id = existing.get("model_id")

    create_data_source(store, model_id, {
        "file_name": dataset_payload.get("file_name") or "uploaded_dataset",
        "storage_reference": dataset_payload.get("storage_reference") or dataset_payload.get("file_name") or "",
        "source_type": dataset_payload.get("source_type") or "file",
        "purpose": dataset_payload.get("purpose") or "Validation dataset",
        "record_count": dataset_payload.get("record_count"),
        "column_count": dataset_payload.get("column_count"),
        "target_variable": target_col or dataset_payload.get("target_variable"),
        "uploaded_by": dataset_payload.get("uploaded_by") or user or intake_payload.get("model_owner") or "user",
        "uploaded_at": dataset_payload.get("uploaded_at") or now,
    })

    update_model_validation_status(
        store,
        model_id,
        stage="data_validation",
        status="In Progress",
        last_run=now,
        description="Validation run started",
        user=user or intake_payload.get("model_owner") or "user",
    )
    validation_entry = next((item for item in store.setdefault("validation", []) if item.get("model_id") == model_id), None)
    if validation_entry is not None:
        validation_entry["status"] = validation_entry.get("overall_status") or "Under Validation"
        validation_entry["validation_status"] = validation_entry.get("overall_status") or "Under Validation"
        validation_entry["validation_stage"] = "data_validation"
        validation_entry["validation_date"] = now

    combined = list(store.get("development", [])) + list(store.get("validation", []))
    seen = set(); deduped = []
    for item in combined:
        if not isinstance(item, dict):
            continue
        key = str(item.get("model_id") or item.get("model_name") or "")
        if key and key not in seen:
            deduped.append(item)
            seen.add(key)
    store["models"] = deduped

    append_history_event(store, model_id, "Validation Run Started", f"Validation run {validation_run_id} started for {business_model_name} using {estimator_name}.", user=user or intake_payload.get("model_owner") or "system", validation_run_id=validation_run_id)
    append_history_event(store, model_id, "Validation Run Completed", f"Validation run {validation_run_id} completed for {business_model_name}.", user=user or intake_payload.get("model_owner") or "system", validation_run_id=validation_run_id)
    return {
        "model_id": model_id,
        "model_name": business_model_name,
        "algorithm": estimator_name,
    }
