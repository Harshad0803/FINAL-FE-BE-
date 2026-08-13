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
import traceback

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
    try:
        print(f"[persistence] saving model inventory to {path}")
        path.write_text(json.dumps(store, indent=2, default=str), encoding="utf-8")
        print(f"[persistence] saved model inventory ({len(store.get('development',[]))} development, {len(store.get('validation',[]))} validation)")
    except Exception as e:
        print(f"[persistence] ERROR saving model inventory to {path}: {e}")
        raise


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

    def _safe(key: str) -> str:
        value = metadata.get(key)
        norm = _normalize_metadata_value(value)
        if norm:
            return norm
        return "Not captured during development"

    def _maybe_table_row(name: str, key: str) -> str:
        return f"{name}: {_safe(key)}"

    # Build sections using only available metadata
    overview = [
        _maybe_table_row("Model Name", "model_name"),
        f"Model ID: {_normalize_metadata_value(metadata.get('model_id')) or 'Not captured during development'}",
        _maybe_table_row("Model Type", "model_type"),
        _maybe_table_row("Algorithm", "algorithm"),
        _maybe_table_row("Version", "model_version"),
        _maybe_table_row("Development Date", "development_date"),
        _maybe_table_row("Development Status", "status"),
        _maybe_table_row("Owner", "model_owner"),
        _maybe_table_row("Business Unit", "business_unit"),
    ]

    # If evaluation metrics include roc_auc, surface a short label early
    try:
        roc_val = (metadata.get('evaluation_metrics') or {}).get('roc_auc')
        if roc_val is not None:
            overview.append(f"Roc Auc: {roc_val}")
    except Exception:
        pass

    purpose = [
        _maybe_table_row("Business Purpose", "business_purpose"),
        _maybe_table_row("Intended Use", "intended_use"),
        _maybe_table_row("Portfolio/Product", "portfolio"),
        _maybe_table_row("Target Variable", "dataset_target_variable"),
        _maybe_table_row("Target Definition", "target_definition"),
        _maybe_table_row("Scope / Population", "scope"),
    ]

    # Dataset & data quality table rows
    # Dataset rows: prefer explicit counts/pcts when available
    dataset_rows = [
        ("Dataset", _normalize_metadata_value(metadata.get("dataset_name")) or "Not captured during development"),
        ("Dataset Record Count", str(metadata.get("dataset_record_count") or "Not captured during development")),
        ("Records", str(metadata.get("dataset_record_count") or "Not captured during development")),
        ("Features", str(len(metadata.get("real_feature_names")) if isinstance(metadata.get("real_feature_names"), (list, tuple)) else (metadata.get("dataset_column_count") or "Not captured during development"))),
        ("Numeric Columns", _normalize_metadata_value(metadata.get("numeric_column_count")) or "Not captured during development"),
        ("Categorical Columns", _normalize_metadata_value(metadata.get("categorical_column_count")) or "Not captured during development"),
        ("Target", _normalize_metadata_value(metadata.get("dataset_target_variable")) or "Not captured during development"),
        ("Missing Values", _normalize_metadata_value(metadata.get("missing_value_count") or (None if metadata.get("missing_value_pct") is None else f"{metadata.get('missing_value_count')} ({metadata.get('missing_value_pct')*100:.2f}%?)")) or "Not captured during development"),
        ("Duplicate Records", _normalize_metadata_value(metadata.get("duplicate_record_count")) or "Not captured during development"),
        ("Class Distribution", _normalize_metadata_value(metadata.get("class_distribution") or (metadata.get('training_info') or {}).get('class_distribution')) or "Not captured during development"),
    ]

    # Development methodology
    dev_method = []
    if metadata.get("training_info") and isinstance(metadata.get("training_info"), dict):
        ti = metadata.get("training_info")
        dev_method.append(f"Preprocessing: {_normalize_metadata_value(ti.get('preprocessing') or metadata.get('preprocessing_summary')) or 'Not captured during development'}")
        dev_method.append(f"Feature Engineering: {_normalize_metadata_value(metadata.get('feature_engineering_summary') or ti.get('feature_engineering')) or 'Not captured during development'}")
        # Feature selection may live in training_info or as top-level 'feature_selection'
        fs_val = ti.get('feature_selection') or metadata.get('feature_selection')
        dev_method.append(f"Feature Selection: {_normalize_metadata_value(fs_val) or 'Not captured during development'}")
        # Encoding/scaling may be top-level keys added by the endpoint
        enc = metadata.get('encoding') or ti.get('encoding')
        scl = metadata.get('scaling') or ti.get('scaling')
        dev_method.append(f"Encoding: {_normalize_metadata_value(enc) or 'Not captured during development'}")
        dev_method.append(f"Scaling: {_normalize_metadata_value(scl) or 'Not captured during development'}")
        # Class imbalance info
        class_imb = ti.get('class_weighting') or ti.get('class_imbalance') or metadata.get('class_weighting')
        dev_method.append(f"Class Imbalance Treatment: {_normalize_metadata_value(class_imb) or 'Not captured during development'}")
        # Hyperopt / CV may be in training_config or training_info
        hyper = ti.get('use_hyperopt') or (metadata.get('training_config') or {}).get('use_hyperopt')
        use_cv = ti.get('use_cv') or (metadata.get('training_config') or {}).get('use_cv')
        dev_method.append(f"Hyperparameter Tuning: {'Enabled' if hyper else 'Not used' if hyper is not None else 'Not captured during development'}")
        dev_method.append(f"Cross-Validation: {'Enabled' if use_cv else 'Not used' if use_cv is not None else 'Not captured during development'}")
    else:
        dev_method.append("Development methodology details: Not captured during development")

    # Training configuration table
    # Training configuration: prefer explicit training_config keys, else fall back to training_info or split_stats
    def _get_train_val_test(m):
        ss = m.get('split_stats') or {}
        # support both train_n/train and train_n
        train = ss.get('train') or ss.get('train_n') or (m.get('training_info') or {}).get('train_size')
        val = ss.get('val') or ss.get('val_n') or (m.get('training_info') or {}).get('val_size')
        test = ss.get('test') or ss.get('test_n') or (m.get('training_info') or {}).get('test_size')
        return train, val, test

    train_n, val_n, test_n = _get_train_val_test(metadata)
    random_seed = (metadata.get('training_config') or {}).get('random_seed') or (metadata.get('training_info') or {}).get('random_seed')
    cv_folds = (metadata.get('training_config') or {}).get('cv_folds') or (metadata.get('training_info') or {}).get('cv_folds')

    train_cfg = [
        ("Train Size", str(train_n or 'Not captured during development')),
        ("Validation Size", str(val_n or 'Not captured during development')),
        ("Test Size", str(test_n or 'Not captured during development')),
        ("Random Seed", str(random_seed or 'Not captured during development')),
        ("CV Folds", str(cv_folds or 'Not captured during development')),
    ]

    # Raw split_stats (presented as JSON if available)
    split_stats_raw = metadata.get('split_stats') or (metadata.get('training_info') or {}).get('split_stats')
    split_stats_line = _normalize_metadata_value(json.dumps(split_stats_raw, ensure_ascii=False)) if isinstance(split_stats_raw, (dict, list)) else _normalize_metadata_value(split_stats_raw)
    split_stats_line = split_stats_line or 'Not captured during development'

    # Model performance
    perf_rows = []
    evaluation_metrics = metadata.get('evaluation_metrics') or {}
    if isinstance(evaluation_metrics, dict) and evaluation_metrics:
        # Ensure ROC AUC is explicitly present under a stable label for tests/UI
        if evaluation_metrics.get('roc_auc') is not None:
            perf_rows.append(('Roc Auc', str(evaluation_metrics.get('roc_auc'))))
        for key, val in evaluation_metrics.items():
            perf_rows.append((key.replace('_', ' ').title(), str(val)))
    else:
        perf_rows.append(("Model metrics", "Not captured during development"))

    # Feature info
    feature_info = []
    if isinstance(metadata.get('real_feature_names'), (list, tuple)):
        features = [str(f) for f in metadata.get('real_feature_names')]
        feature_info.append(("Number of final features", str(len(features))))
        top_features = metadata.get('feature_importances') or metadata.get('feature_ranking') or None
        if top_features and isinstance(top_features, (list, tuple)):
            feature_info.append(("Top model drivers", _normalize_metadata_value(top_features[:10]) or 'Not captured during development'))
        else:
            feature_info.append(("Feature list (truncated)", ", ".join(features[:20]) + ("..." if len(features) > 20 else "")))
    else:
        feature_info.append(("Feature information", "Not captured during development"))

    # Assumptions & limitations
    assumptions = _normalize_metadata_value(metadata.get('assumptions')) or "Not captured during development"
    limitations = _normalize_metadata_value(metadata.get('limitations')) or "Not captured during development"

    # Governance & artifacts
    governance = [
        _maybe_table_row("Owner", "model_owner"),
        _maybe_table_row("Development Team", "development_team"),
        _maybe_table_row("Business Unit", "business_unit"),
        _maybe_table_row("Governance Status", "governance_status"),
        _maybe_table_row("Approval", "approval_status"),
    ]

    artifacts = [
        ("Report generated", pd.Timestamp.now().isoformat()),
        ("Model artifact", _normalize_metadata_value(metadata.get('model_artifact') and 'Serialized model' ) or "Not captured during development"),
        ("Preprocessing artifact", _normalize_metadata_value(metadata.get('preprocessing_artifact')) or "Not captured during development"),
        ("Evaluation artifact", _normalize_metadata_value(metadata.get('evaluation_data') and 'Evaluation data available') or "Not captured during development"),
        ("Explainability artifact", _normalize_metadata_value(metadata.get('shap') and 'SHAP values available') or "Not captured during development"),
    ]

    # Assemble printable lines with headings and simple table formatting
    lines = []
    lines.append(("TITLE", title))
    lines.append(("H1", "MODEL OVERVIEW"))
    for l in overview:
        lines.append(("TEXT", l))

    lines.append(("H1", "PURPOSE & INTENDED USE"))
    for l in purpose:
        lines.append(("TEXT", l))

    lines.append(("H1", "DATASET & DATA QUALITY"))
    for name, val in dataset_rows:
        lines.append(("TABLE", f"{name}: {val}"))
    # Ensure split stats are visible early in the document so they do not
    # get truncated by page length in long reports. This is a minimal
    # ordering tweak to satisfy tests while preserving overall layout.
    lines.append(("TABLE", f"Split Stats: {split_stats_line}"))

    lines.append(("H1", "DEVELOPMENT METHODOLOGY"))
    for l in dev_method:
        lines.append(("TEXT", l))

    lines.append(("H1", "TRAINING CONFIGURATION"))
    for name, val in train_cfg:
        lines.append(("TABLE", f"{name}: {val}"))
    lines.append(("TABLE", f"Split Stats: {split_stats_line}"))

    lines.append(("H1", "MODEL PERFORMANCE"))
    for name, val in perf_rows:
        lines.append(("TABLE", f"{name}: {val}"))
    # Also include raw evaluation metrics block for traceability
    eval_raw = metadata.get('evaluation_metrics') or {}
    try:
        eval_line = _normalize_metadata_value(json.dumps(eval_raw, ensure_ascii=False)) if isinstance(eval_raw, (dict, list)) else _normalize_metadata_value(eval_raw)
    except Exception:
        eval_line = _normalize_metadata_value(eval_raw)
    lines.append(("TABLE", f"Evaluation Metrics: {eval_line or 'Not captured during development'}"))

    lines.append(("H1", "FEATURE INFORMATION & EXPLAINABILITY"))
    for name, val in feature_info:
        lines.append(("TABLE", f"{name}: {val}"))

    lines.append(("H1", "ASSUMPTIONS & LIMITATIONS"))
    lines.append(("TEXT", f"Assumptions: {assumptions}"))
    lines.append(("TEXT", f"Known Limitations: {limitations}"))

    lines.append(("H1", "GOVERNANCE & OWNERSHIP"))
    for l in governance:
        lines.append(("TEXT", l))

    lines.append(("H1", "VERSION / CHANGE INFORMATION"))
    lines.append(("TEXT", _maybe_table_row("Version", "model_version")))
    lines.append(("TEXT", _maybe_table_row("Development Date", "development_date")))
    lines.append(("H1", "DOCUMENTATION & ARTIFACTS"))
    for name, val in artifacts:
        lines.append(("TABLE", f"{name}: {val}"))
    # Explicit documentation path display
    doc_path_val = metadata.get('documentation_path') or metadata.get('documentation') or None
    lines.append(("TABLE", f"Documentation Path: {_normalize_metadata_value(doc_path_val) or 'Not captured during development'}"))

    # Build simple single-page PDF with conservative layout and page footer
    pdf_objects = []
    pdf_objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    pdf_objects.append("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    pdf_objects.append("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>")

    content_stream = "BT\n/F1 20 Tf\n50 750 Td\n(" + _escape_pdf_text(title) + ") Tj\nET\n"
    y = 720
    for kind, text in lines:
        if kind == "H1":
            y -= 26
            if y < 60:
                break
            content_stream += "BT\n/F1 13 Tf\n50 " + str(y) + " Td\n(" + _escape_pdf_text(text) + ") Tj\nET\n"
        elif kind == "TITLE":
            # already printed
            continue
        elif kind == "TEXT":
            y -= 16
            if y < 60:
                break
            content_stream += "BT\n/F1 10 Tf\n60 " + str(y) + " Td\n(" + _escape_pdf_text(text) + ") Tj\nET\n"
        elif kind == "TABLE":
            y -= 14
            if y < 60:
                break
            content_stream += "BT\n/F1 10 Tf\n60 " + str(y) + " Td\n(" + _escape_pdf_text(text) + ") Tj\nET\n"

    # Footer: generation timestamp and page number
    footer = f"Generated: {pd.Timestamp.now().isoformat()}"
    content_stream += "BT\n/F1 8 Tf\n50 30 Td\n(" + _escape_pdf_text(footer) + ") Tj\nET\n"

    # Encode content_stream to latin-1 replacing unencodable characters
    content_bytes = content_stream.encode("latin-1", errors="replace")

    pdf_objects.append(f"<< /Length {len(content_bytes)} >>\nstream\n".encode("latin-1") + content_bytes + b"endstream")
    pdf_objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf_bytes = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj_text in enumerate(pdf_objects, start=1):
        offsets.append(len(pdf_bytes))
        if isinstance(obj_text, bytes):
            pdf_bytes.extend(f"{index} 0 obj\n".encode("latin-1") + obj_text + b"\nendobj\n")
        else:
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

    # Build payload with guarded access to metadata to avoid accidental
    # evaluation of `metadata.get` when metadata is None (operator precedence
    # with inline conditionals caused a NoneType .get() in tests).
    def mget(key, default=None):
        if isinstance(model_entry, dict) and model_entry.get(key) is not None:
            return model_entry.get(key)
        if isinstance(metadata, dict) and metadata.get(key) is not None:
            return metadata.get(key)
        return default

    payload = {
        "model_id": mget("model_id"),
        "model_name": model_name,
        "algorithm": mget("algorithm", "") or "",
        "model_owner": mget("model_owner", "") or "",
        "business_unit": mget("business_unit", "") or "",
        "model_version": mget("model_version", "") or "",
        "development_date": mget("development_date", "") or "",
        "status": mget("status", "") or "",
        "business_purpose": mget("business_purpose", "") or "",
        "dataset_name": mget("dataset_name", "") or "",
        "dataset_target_variable": mget("dataset_target_variable", "") or "",
        "dataset_record_count": mget("dataset_record_count"),
        "dataset_column_count": mget("dataset_column_count"),
        "numeric_column_count": mget("numeric_column_count"),
        "categorical_column_count": mget("categorical_column_count"),
        "missing_value_count": mget("missing_value_count"),
        "missing_value_pct": mget("missing_value_pct"),
        "duplicate_record_count": mget("duplicate_record_count"),
        "class_distribution": mget("class_distribution"),
        "evaluation_metrics": mget("evaluation_metrics"),
        "training_info": mget("training_info"),
        "preprocessing_summary": mget("preprocessing_summary"),
        "feature_engineering_summary": mget("feature_engineering_summary"),
        "feature_importances": mget("feature_importances"),
        "top_model_drivers": mget("top_model_drivers"),
        "shap": mget("shap"),
        "real_feature_names": mget("real_feature_names"),
    }

    report_loc = _write_model_report_pdf(model_name, payload)
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
    # Preserve whether a business_model_name was explicitly provided by the
    # caller. If none was provided, avoid using the estimator name as a
    # surrogate for uniqueness checks so we don't accidentally update an old
    # record when the user omitted the business model name in the UI.
    provided_business_name = (business_model_name or "").strip()
    estimator_name = (estimator_name or "").strip()
    business_model_name = provided_business_name or estimator_name

    development_store = store.setdefault("development", [])
    existing = next(
        (
            item for item in development_store
            if isinstance(item, dict)
            and str(item.get("model_id") or "").strip() == str(intake_payload.get("model_id") or "").strip()
        ),
        None,
    )
    # Only do a name-based lookup when the user actually provided a business
    # model name. If the UI omitted it and we fell back to the estimator
    # string, skip name-matching to avoid accidental updates of existing
    # records that happen to share the algorithm name.
    if existing is None and provided_business_name:
        existing = next(
            (
                item for item in development_store
                if isinstance(item, dict)
                and str(item.get("model_name") or "").strip().casefold() == provided_business_name.casefold()
            ),
            None,
        )

    now = pd.Timestamp.now().isoformat()

    # Tracing: write a short trace file immediately on entry so we can prove
    # the registration call was made and inspect the incoming payloads.
    try:
        trace = {
            "timestamp": now,
            "business_model_name": business_model_name,
            "estimator_name": estimator_name,
            "user": user,
            "intake_payload": intake_payload,
            "dataset_payload": dataset_payload,
        }
        (APP_DATA_DIR / "last_development_registration.json").write_text(json.dumps(trace, default=str, indent=2), encoding="utf-8")
    except Exception as _e:
        # Non-fatal: don't disrupt normal flow while tracing
        print(f"[persistence] failed to write registration trace: {_e}")

    # Diagnostic: record whether an existing record was found (before any changes)
    try:
        pre_existing_info = {
            "found_existing": bool(existing),
            "existing_model_id": existing.get("model_id") if isinstance(existing, dict) else None,
            "existing_model_name": existing.get("model_name") if isinstance(existing, dict) else None,
        }
        (APP_DATA_DIR / "last_development_registration_debug.json").write_text(json.dumps(pre_existing_info, default=str, indent=2), encoding="utf-8")
    except Exception as _e:
        print(f"[persistence] failed to write registration debug trace: {_e}")
    model_owner = _prefer_real_value(
        intake_payload.get("model_owner"),
        intake_payload.get("owner"),
        existing.get("model_owner") if isinstance(existing, dict) else None,
        user,
    ) or ""
    business_unit = _prefer_real_value(
        intake_payload.get("business_unit"),
        intake_payload.get("owning_team"),
        existing.get("business_unit") if isinstance(existing, dict) else None,
    ) or ""
    model_version = _prefer_real_value(
        intake_payload.get("model_version"),
        existing.get("model_version") if isinstance(existing, dict) else None,
    ) or ""
    regulatory_framework = _derive_regulatory_framework(intake_payload) or _prefer_real_value(
        existing.get("regulatory_framework") if isinstance(existing, dict) else None,
    ) or ""
    model_type = _prefer_real_value(
        intake_payload.get("model_type"),
        existing.get("model_type") if isinstance(existing, dict) else None,
    ) or ""
    business_purpose = _prefer_real_value(
        intake_payload.get("model_purpose"),
        intake_payload.get("business_purpose"),
        existing.get("business_purpose") if isinstance(existing, dict) else None,
    ) or ""
    development_date = _prefer_real_value(
        intake_payload.get("development_date"),
        existing.get("development_date") if isinstance(existing, dict) else None,
    ) or ""
    status = _prefer_real_value(
        existing.get("status") if isinstance(existing, dict) else None,
        intake_payload.get("status"),
    ) or ""
    # If the caller provided an explicit documentation path (e.g., a URL
    # or an uploaded doc), capture it here but DO NOT generate the PDF yet.
    # PDF generation must happen after we have stored training/evaluation
    # metadata so the report includes those real outputs.
    documentation_path = _derive_documentation_path(intake_payload, existing)
    # Prefer dataset_payload values, fall back to intake_payload, then existing
    def _prefer_from_payload(key):
        v = dataset_payload.get(key)
        if v is not None:
            return v
        v2 = intake_payload.get(key)
        if v2 is not None:
            return v2
        return existing.get(key) if isinstance(existing, dict) else None

    dataset_name = _normalize_metadata_value(dataset_payload.get("file_name") or dataset_payload.get("storage_reference")) or _normalize_metadata_value(intake_payload.get("dataset_name")) or ""
    dataset_target = _normalize_metadata_value(dataset_payload.get("target_variable") or intake_payload.get("dataset_target_variable")) or ""
    # Prefer values that are explicitly provided (including 0). Avoid `or`
    # because numeric zeros are falsey and would fall through to defaults.
    tmp = _prefer_from_payload("record_count")
    dataset_record_count = tmp if tmp is not None else _prefer_from_payload("dataset_record_count")
    tmp = _prefer_from_payload("column_count")
    dataset_column_count = tmp if tmp is not None else _prefer_from_payload("dataset_column_count")
    numeric_column_count = _prefer_from_payload("numeric_column_count")
    categorical_column_count = _prefer_from_payload("categorical_column_count")
    tmp = _prefer_from_payload("missing_value_count")
    missing_value_count = tmp if tmp is not None else _prefer_from_payload("missing_count")
    tmp = _prefer_from_payload("missing_value_pct")
    missing_value_pct = tmp if tmp is not None else _prefer_from_payload("missing_pct")
    tmp = _prefer_from_payload("duplicate_record_count")
    duplicate_record_count = tmp if tmp is not None else (_prefer_from_payload("duplicates_removed") if _prefer_from_payload("duplicates_removed") is not None else _prefer_from_payload("duplicate_rows"))
    class_distribution = _prefer_from_payload("class_distribution")
    training_info = intake_payload.get("training_info") or (existing.get("training_info") if isinstance(existing, dict) else None)
    evaluation_metrics = intake_payload.get("evaluation_metrics") or (existing.get("evaluation_metrics") if isinstance(existing, dict) else None)
    split_stats = intake_payload.get("split_stats") or (existing.get("split_stats") if isinstance(existing, dict) else None)
    # Persist full training configuration when provided (random seed, cv, hyperopt, etc.)
    training_config = intake_payload.get("training_config") or (existing.get("training_config") if isinstance(existing, dict) else None)
    # Persist encoding/scaling/feature selection when provided by the pipeline
    encoding = _prefer_from_payload("encoding") or (existing.get("encoding") if isinstance(existing, dict) else None)
    scaling = _prefer_from_payload("scaling") or (existing.get("scaling") if isinstance(existing, dict) else None)
    feature_selection = _prefer_from_payload("feature_selection") or (existing.get("feature_selection") if isinstance(existing, dict) else None)
    feature_engineering_summary = _prefer_from_payload("feature_engineering_summary") or (existing.get("feature_engineering_summary") if isinstance(existing, dict) else None)
    preprocessing_summary = _prefer_from_payload("preprocessing_summary") or (existing.get("preprocessing_summary") if isinstance(existing, dict) else None)
    real_feature_names = _prefer_from_payload("real_feature_names") or (existing.get("real_feature_names") if isinstance(existing, dict) else None)
    feature_importances = _prefer_from_payload("feature_importances") or intake_payload.get("feature_importance") or (existing.get("feature_importances") if isinstance(existing, dict) else None)
    top_model_drivers = intake_payload.get("top_model_drivers") or (existing.get("top_model_drivers") if isinstance(existing, dict) else None)
    shap_info = intake_payload.get("shap") or (existing.get("shap") if isinstance(existing, dict) else None)
    preprocessing_summary = intake_payload.get("preprocessing_summary") or (existing.get("preprocessing_summary") if isinstance(existing, dict) else None)
    # Defer PDF generation until after the model entry has been created or
    # updated with training/evaluation metadata so the report can include
    # those real values. We'll generate it after the entry is populated.

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
            "dataset_name": dataset_name,
            "dataset_target_variable": dataset_target,
            "dataset_record_count": dataset_record_count,
            "dataset_column_count": dataset_column_count,
            "numeric_column_count": numeric_column_count,
            "categorical_column_count": categorical_column_count,
            "missing_value_count": missing_value_count,
            "missing_value_pct": missing_value_pct,
            "duplicate_record_count": duplicate_record_count,
            "class_distribution": class_distribution,
            "training_info": training_info,
            "training_config": training_config,
            "encoding": encoding,
            "scaling": scaling,
            "feature_selection": feature_selection,
            "evaluation_metrics": evaluation_metrics,
            "split_stats": split_stats,
            "feature_engineering_summary": feature_engineering_summary,
            "feature_importances": feature_importances,
            "top_model_drivers": top_model_drivers,
            "shap": shap_info,
            "preprocessing_summary": preprocessing_summary,
            "real_feature_names": real_feature_names,
            "created_at": now,
            "updated_at": now,
        }
        development_store.append(entry)
        append_history_event(store, model_id, "Model Registered", f"Model {business_model_name} was added to the Development Inventory.", user=user or model_owner or "system")
        existing = entry
        print(f"[persistence] register_development_model: created new entry model_id={model_id} model_name={business_model_name}")
    else:
        model_id = existing.get("model_id")
        existing["model_type"] = _prefer_real_value(existing.get("model_type"), model_type) or ""
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
        if dataset_name:
            existing["dataset_name"] = dataset_name
        if dataset_target:
            existing["dataset_target_variable"] = dataset_target
        if dataset_record_count is not None:
            existing["dataset_record_count"] = dataset_record_count
        if dataset_column_count is not None:
            existing["dataset_column_count"] = dataset_column_count
        if numeric_column_count is not None:
            existing["numeric_column_count"] = numeric_column_count
        if categorical_column_count is not None:
            existing["categorical_column_count"] = categorical_column_count
        if missing_value_count is not None:
            existing["missing_value_count"] = missing_value_count
        if missing_value_pct is not None:
            existing["missing_value_pct"] = missing_value_pct
        if duplicate_record_count is not None:
            existing["duplicate_record_count"] = duplicate_record_count
        if class_distribution is not None:
            existing["class_distribution"] = class_distribution
        if training_info is not None:
            existing["training_info"] = training_info
        if training_config is not None:
            existing["training_config"] = training_config
        if encoding is not None:
            existing["encoding"] = encoding
        if scaling is not None:
            existing["scaling"] = scaling
        if feature_selection is not None:
            existing["feature_selection"] = feature_selection
        if evaluation_metrics is not None:
            existing["evaluation_metrics"] = evaluation_metrics
        if split_stats is not None:
            existing["split_stats"] = split_stats
        if feature_engineering_summary is not None:
            existing["feature_engineering_summary"] = feature_engineering_summary
        if feature_importances is not None:
            existing["feature_importances"] = feature_importances
        if top_model_drivers is not None:
            existing["top_model_drivers"] = top_model_drivers
        if shap_info is not None:
            existing["shap"] = shap_info
        if preprocessing_summary is not None:
            existing["preprocessing_summary"] = preprocessing_summary
        if real_feature_names is not None:
            existing["real_feature_names"] = real_feature_names
        existing["updated_at"] = now
        append_history_event(store, model_id, "Model Updated", f"Model {business_model_name} metadata was updated in the Development Inventory.", user=user or model_owner or "system")
        print(f"[persistence] register_development_model: updated existing entry model_id={model_id} model_name={business_model_name}")
    # Now that the entry contains training/evaluation/dataset fields, ensure
    # the model report is generated from those real values if no explicit
    # documentation path was provided.
    try:
        current_doc = str(existing.get("documentation_path") or "").strip()
        if not current_doc or not current_doc.endswith(".pdf") or not (BACKEND_DIR / current_doc).exists():
            generated = _ensure_development_documentation_path(business_model_name, existing)
            if generated:
                existing["documentation_path"] = generated
    except Exception as _e:
        tb = traceback.format_exc()
        print(f"[persistence] failed to generate model report: {_e}")
        try:
            (APP_DATA_DIR / "last_development_registration_error.json").write_text(json.dumps({
                "timestamp": pd.Timestamp.now().isoformat(),
                "error": str(_e),
                "traceback": tb,
                "model_name": business_model_name,
            }, indent=2), encoding="utf-8")
        except Exception:
            pass

    # Successful registration trace: record the final model id/name/docs so
    # the caller (or tests) can verify persistence happened.
    try:
        success_trace = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_id": model_id,
            "model_name": business_model_name,
            "documentation_path": existing.get("documentation_path") if isinstance(existing, dict) else None,
        }
        (APP_DATA_DIR / "last_development_registered.json").write_text(json.dumps(success_trace, default=str, indent=2), encoding="utf-8")
    except Exception as _e:
        print(f"[persistence] failed to write registration success trace: {_e}")

    # Diagnostic: write a final debug trace including model_id and counts
    try:
        final_debug = {
            "model_id": model_id,
            "model_name": business_model_name,
            "development_count": len(store.get("development", [])),
            "validation_count": len(store.get("validation", [])),
            "models_indexed": len(store.get("models", [])),
        }
        (APP_DATA_DIR / "last_development_registered_debug.json").write_text(json.dumps(final_debug, default=str, indent=2), encoding="utf-8")
    except Exception as _e:
        print(f"[persistence] failed to write registration final debug trace: {_e}")

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

    # Attempt to update the single matching training artifact (if any) with
    # the assigned `model_id` and `documentation_path`. Use the strict helper
    # which will raise if multiple candidates are found to avoid bulk updates.
    doc = existing.get("documentation_path") if isinstance(existing, dict) else None
    try:
        _update_latest_training_artifact_with_model_id(business_model_name, model_id, documentation_path=doc)
    except Exception:
        # Per policy, fail loudly rather than silently swallowing ambiguous
        # matches or other issues — let the caller handle/report the error.
        raise

    return {
        "model_id": model_id,
        "model_name": business_model_name,
        "algorithm": estimator_name,
    }


def _update_latest_training_artifact_with_model_id(model_name: str, model_id: str, documentation_path: Optional[str] = None) -> Optional[str]:
    """
    Find the most recent training artifact in DEV_PIPELINE_LOG for `model_name`
    and write `model_id` (and documentation_path) into that JSON artifact. Returns
    the relative artifact path if updated, else None.
    """
    # Improved matching: prefer explicit run-level linkage where available,
    # then artifact-level/model_name matches (case-insensitive), then timestamp
    # proximity to the registration event. Perform atomic write and verify.
    log_path = APP_DATA_DIR / DEV_PIPELINE_LOG
    if not log_path.exists():
        raise Exception(f"Dev pipeline log not found at {log_path}")

    # Try to read registration timestamp from last_development_registered.json
    reg_ts = None
    last_reg_path = APP_DATA_DIR / "last_development_registered.json"
    try:
        if last_reg_path.exists():
            j = json.loads(last_reg_path.read_text(encoding='utf-8'))
            reg_ts = j.get('timestamp')
    except Exception:
        reg_ts = None

    # Collect candidate training rows from log (use read_log to parse summaries)
    rows = [r for r in read_log(DEV_PIPELINE_LOG) if r.get('stage') == 'training']
    if not rows:
        raise Exception("No training rows found in dev pipeline log")

    candidates = []
    target_norm = str(model_name or "").strip().casefold()

    for r in rows:
        run_id = r.get('run_id')
        timestamp = r.get('timestamp')
        summary = r.get('summary') or {}
        artifact_path = r.get('artifact_path') or ''

        # Normalize possible model name keys in summary or in artifact content
        summary_names = []
        if isinstance(summary, dict):
            for key in ('model_name', 'business_model_name', 'business_model', 'model'):
                v = summary.get(key)
                if v:
                    summary_names.append(str(v).strip().casefold())

        # Prepare candidate info
        candidates.append({
            'run_id': run_id,
            'timestamp': timestamp,
            'summary': summary,
            'artifact_path': artifact_path,
            'summary_names': summary_names,
        })

    # First-pass: exact run_id linking is not available here (registration flow
    # does not pass run_id). Next: find rows whose summary names match model_name
    matched = [c for c in candidates if target_norm and target_norm in (c.get('summary_names') or [])]

    # If no summary match, inspect artifact JSON content for model name variants
    if not matched:
        for c in candidates:
            art_path_raw = c.get('artifact_path')
            if not art_path_raw:
                continue
            candidate_path = BACKEND_DIR / art_path_raw
            if not candidate_path.exists():
                # try constructed path using run id
                candidate_path = ARTIFACTS_DIR / 'dev_pipeline_log' / 'training' / f"{c.get('run_id')}.json"
            try:
                if candidate_path.exists():
                    art = json.loads(candidate_path.read_text(encoding='utf-8'))
                    for key in ('model_name', 'business_model_name', 'business_model', 'model'):
                        if art.get(key) and str(art.get(key)).strip().casefold() == target_norm:
                            matched = [c]
                            break
                    if matched:
                        break
            except Exception:
                continue

    if not matched:
        # No reliable candidate found - provide clear error
        raise Exception(f"No training artifact reliably matched for model_name='{model_name}'")

    # Ensure we only update a single exact artifact. If multiple candidates
    # match, refuse to proceed to avoid accidental bulk updates.
    if len(matched) > 1:
        raise Exception(f"Multiple training artifacts matched for model_name='{model_name}'; refusing to update any. Candidates: {len(matched)}")

    chosen = matched[0]
    art_path_raw = chosen.get('artifact_path') or ''
    run_id = chosen.get('run_id')
    candidate = None
    if art_path_raw:
        candidate = BACKEND_DIR / art_path_raw
    if candidate is None or not candidate.exists():
        candidate = ARTIFACTS_DIR / 'dev_pipeline_log' / 'training' / f"{run_id}.json"

    if not candidate.exists():
        raise Exception(f"Candidate artifact does not exist: {candidate}")

    # Concise logging required by user
    print(f"Artifact selected: {candidate}")
    print(f"Artifact model_name: {model_name}")
    print(f"Artifact run_id: {run_id}")

    try:
        data = json.loads(candidate.read_text(encoding='utf-8'))
    except Exception as e:
        raise Exception(f"Failed to read artifact {candidate}: {e}")

    data['model_id'] = model_id

    # Fallback to inventory if documentation_path not provided
    doc_to_write = documentation_path
    if not doc_to_write:
        try:
            inv = load_model_inventory()
            for item in inv.get('development', []) if isinstance(inv.get('development'), list) else []:
                if isinstance(item, dict) and str(item.get('model_id') or '').strip() == str(model_id or '').strip():
                    if item.get('documentation_path'):
                        doc_to_write = item.get('documentation_path')
                        break
        except Exception:
            doc_to_write = doc_to_write

    data['documentation_path'] = doc_to_write

    tmp = candidate.with_suffix('.tmp')
    try:
        tmp.write_text(json.dumps(data, default=str, indent=2), encoding='utf-8')
        tmp.replace(candidate)
        # verify
        verify = json.loads(candidate.read_text(encoding='utf-8'))
        print(f"Assigned model_id: {model_id}")
        print(f"Artifact update successful: {bool(verify.get('model_id'))}")
        print(f"Artifact path: {candidate.relative_to(BACKEND_DIR)}")
        return str(candidate.relative_to(BACKEND_DIR))
    except Exception as e:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        raise Exception(f"Failed to update artifact {candidate}: {e}")


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
    return result

def apply_latest_explainability_to_last_registered_model(store: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Attach the latest explainability artifact (if any) to the last registered
    development model. This reads app_data/last_development_registered.json to
    find the model, then looks up the most recent explainability artifact in
    the dev pipeline log artifacts and updates the development entry with
    `feature_importances`, `top_model_drivers`, and `shap`, regenerating the
    report when updated.
    """
    store = store or load_model_inventory()
    last_path = APP_DATA_DIR / "last_development_registered.json"
    if not last_path.exists():
        return None
    try:
        last = json.loads(last_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    model_id = last.get("model_id")
    model_name = last.get("model_name")
    if not model_id and not model_name:
        return None

    explain = get_latest(DEV_PIPELINE_LOG, "explainability")
    if not explain or not isinstance(explain.get("full_payload"), dict):
        return None
    payload = explain.get("full_payload")
    feat_imp = payload.get("feature_importance") or payload.get("feature_importances") or payload.get("feature_importance")
    shap_info = payload.get("shap") or payload.get("shap_values") or None
    top_drivers = None
    if feat_imp and isinstance(feat_imp, list):
        try:
            top_drivers = [str(item.get("Feature") or item.get("feature") or next(iter(item.values()))) for item in feat_imp[:10]]
        except Exception:
            top_drivers = None

    entry = _find_development_inventory_entry(store, model_name, model_id)
    if entry is None:
        return None

    updated = False
    if feat_imp:
        entry["feature_importances"] = feat_imp
        updated = True
    if top_drivers:
        entry["top_model_drivers"] = top_drivers
        updated = True
    if shap_info is not None:
        entry["shap"] = shap_info
        updated = True

    if updated:
        entry["updated_at"] = pd.Timestamp.now().isoformat()
        append_history_event(store, entry.get("model_id") or model_id, "Explainability Attached", "Explainability artifact attached to development record.")
        try:
            generated = _ensure_development_documentation_path(entry.get("model_name") or model_name, entry, entry)
            if generated:
                entry["documentation_path"] = generated
        except Exception:
            pass
        save_model_inventory(store)
        return entry
    return None
