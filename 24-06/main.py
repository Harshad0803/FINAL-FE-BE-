"""
main.py - FastAPI backend for the Credit Risk ML POC

Thin REST wrapper that exposes the latest pipeline logic from the
Credit-Risk-Poc-main source-of-truth project.
"""

import base64
import io
import json
import importlib.util
import os
import tempfile
import random
import re
import sys
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Union

BACKEND_DIR = Path(__file__).resolve().parent
SOURCE_OF_TRUTH_DIR = Path(__file__).resolve().parent.parent / "Credit-Risk-Poc-main"
for path in [BACKEND_DIR, SOURCE_OF_TRUTH_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0 if path == BACKEND_DIR else 1, path_str)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

_local_agent2_spec = importlib.util.spec_from_file_location("local_agent2", Path(__file__).resolve().parent / "agent2.py")
if _local_agent2_spec is None or _local_agent2_spec.loader is None:
    raise ImportError("Could not load local Agent2 from 24-06/agent2.py")
_local_agent2_module = importlib.util.module_from_spec(_local_agent2_spec)
_local_agent2_spec.loader.exec_module(_local_agent2_module)
Agent2 = _local_agent2_module.Agent2
from build_rules import RULES_PATH

from utils import (
    generate_synthetic_credit_dataset, detect_column_types,
    detect_target_candidates, detect_task_type, df_to_csv_download, model_to_download,
)
from preprocessing_new import (
    build_preprocessing_report, prepare_data, rebuild_preprocessor_for, finalize_xy,
    get_feature_names_from_fitted_preprocessor,
    classify_missing_treatment, select_imputation_strategy, SemanticImputer,
    REVIEW_MISSING_THRESHOLD, MISSING_VALUE_LIMITATION_NOTE,
    estimate_drop_impact,
)
from feature_engineering import (
    analyze_for_feature_engineering, apply_feature_engineering,
    compute_univariate_gini,
)
try:
    from feature_engineering import resolve_ead_configuration
except ImportError:
    _legacy_fe_path = Path(__file__).resolve().parent / "feature_engineering.py"
    _legacy_fe_spec = importlib.util.spec_from_file_location("legacy_feature_engineering", _legacy_fe_path)
    if _legacy_fe_spec is None or _legacy_fe_spec.loader is None:
        raise
    _legacy_fe = importlib.util.module_from_spec(_legacy_fe_spec)
    _legacy_fe_spec.loader.exec_module(_legacy_fe)
    resolve_ead_configuration = _legacy_fe.resolve_ead_configuration
from model_selector import (
    recommend_model, compute_dataset_characteristics, characteristics_from_summary,
    get_model_instance, get_hyperparameter_grid, CLASSIFICATION_MODELS,
)
from train_new import split_data, compute_split_stats, train_model
import evaluate_new as eval_engine
import fred_client
import data_integration as di
import persistence

compute_binary_metrics = eval_engine.compute_binary_metrics
compute_regression_metrics = eval_engine.compute_regression_metrics
compute_heteroscedasticity_check = eval_engine.compute_heteroscedasticity_check
compute_roc_curve = eval_engine.compute_roc_curve
compute_pr_curve = eval_engine.compute_pr_curve
compute_threshold_analysis = eval_engine.compute_threshold_analysis
compute_score_distribution = eval_engine.compute_score_distribution
compute_gain_chart = eval_engine.compute_gain_chart
compute_lift_chart = eval_engine.compute_lift_chart
compute_temporal_analysis_bundle = eval_engine.compute_temporal_analysis_bundle
plot_roc_curve = eval_engine.plot_roc_curve
plot_pr_curve = eval_engine.plot_pr_curve
plot_confusion_matrix = eval_engine.plot_confusion_matrix
plot_threshold_analysis = eval_engine.plot_threshold_analysis
plot_score_distribution = eval_engine.plot_score_distribution
plot_lift_chart = eval_engine.plot_lift_chart
from explainability import (
    extract_feature_importance, compute_shap_values, generate_prediction_reasoning,
    generate_model_summary,
)
from val_replication_core import (
    extract_metrics_from_mdd, parse_mdd_file, run_replication, evaluate_replication_checks,
    run_bias_check, detect_protected_columns, _gini, _ks,
)
from validation_stress_core import run_stress_suite, run_manual_shock

_validation_agent2_path = SOURCE_OF_TRUTH_DIR / "validation_agent2.py"
_validation_agent2_spec = importlib.util.spec_from_file_location("source_validation_agent2", _validation_agent2_path)
if _validation_agent2_spec is None or _validation_agent2_spec.loader is None:
    raise ImportError(f"Could not load ValidationAgent2 from {_validation_agent2_path}")
_validation_agent2_module = importlib.util.module_from_spec(_validation_agent2_spec)
_validation_agent2_spec.loader.exec_module(_validation_agent2_module)
ValidationAgent2 = _validation_agent2_module.ValidationAgent2
# ValidationAgent2.check_regulatory_compliance() (SOURCE_OF_TRUTH_DIR's
# validation_agent2.py) inlines these keyword lists as literal _mdd_contains()
# args rather than module constants — mirrored here verbatim (lines ~1442,
# ~1462 of that file) so Stage 8's findings compilation checks the MDD with
# the exact same keyword sets Stage 7's 7.1/7.2 checks use.
CALIBRATION_KEYWORDS = ["calibrat", "platt", "isotonic", "through the cycle", "ttc"]
STAGING_KEYWORDS = ["stage 1", "stage 2", "stage 3", "staging", "sicr", "lifetime ecl", "12 month"]

# Source-of-truth Agent2: the RAG rule-matching agent (rag_store/val_mdd_rules.json,
# check_for_validation / check_mdd_keywords / check_documents_with_llm). This is a
# different class from the local `Agent2` above (24-06/agent2.py, rag_store/rules.json)
# which only handles Stage 1/2/4/5 compliance flags and has no RAG rule methods.
_source_agent2_path = SOURCE_OF_TRUTH_DIR / "agent2.py"
_source_agent2_spec = importlib.util.spec_from_file_location("source_agent2", _source_agent2_path)
if _source_agent2_spec is None or _source_agent2_spec.loader is None:
    raise ImportError(f"Could not load source Agent2 from {_source_agent2_path}")
_source_agent2_module = importlib.util.module_from_spec(_source_agent2_spec)
_source_agent2_spec.loader.exec_module(_source_agent2_module)
SourceAgent2 = _source_agent2_module.Agent2
_rule_matches_frameworks = _source_agent2_module._rule_matches_frameworks


app = FastAPI()

_ALLOWED_ORIGINS = [
    "http://192.168.1.19:3000",
    "http://192.168.1.19:5173",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8081",
    "http://localhost:3001",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8081",
    "http://127.0.0.1:3001",
    "https://final-ok9cvxfh0-harshads-projects-d63c4e68.vercel.app",
]
_ALLOWED_ORIGIN_REGEX = re.compile(r"https://.*-harshads-projects-d63c4e68\.vercel\.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_origin_regex=_ALLOWED_ORIGIN_REGEX.pattern,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Turn unhandled exceptions into a normal JSON response with CORS headers,
    instead of letting them fall through to Starlette's default 500 handling.

    Handlers registered for the bare `Exception` class run inside Starlette's
    ServerErrorMiddleware (see Starlette's own docs on this) — which sits
    OUTSIDE CORSMiddleware in the stack no matter what handler is registered
    here, so CORSMiddleware never gets a chance to add
    Access-Control-Allow-Origin to this response. Confirmed by testing: the
    plain-JSONResponse version of this handler (no manual headers) still came
    back with no CORS header on a genuine crash. The browser then reports
    "Failed to fetch"/a CORS error for what's actually a server-side 500,
    indistinguishable from a real network failure. Fix is to set the header
    ourselves, directly on the response this handler returns.
    """
    response = JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )
    origin = request.headers.get("origin")
    if origin and (origin in _ALLOWED_ORIGINS or _ALLOWED_ORIGIN_REGEX.match(origin)):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Vary"] = "Origin"
    return response


def run_validation_agent2(val_df: Optional[pd.DataFrame], intake_json: dict, mdd_text: str = "") -> dict:
    agent = ValidationAgent2()
    # If no dataframe was provided (or it's empty), pass None so the
    # ValidationAgent2 checks that expect `None` will short-circuit safely.
    df_arg = val_df if (val_df is not None and getattr(val_df, "empty", False) is False) else None
    return agent.run_all_checks(df_arg, intake_json or {}, mdd_text or "")


def _detect_temporal_date_columns(df: pd.DataFrame) -> List[str]:
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if date_cols:
        return date_cols
    for c in df.select_dtypes(include="object").columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.8:
                date_cols.append(c)
        except Exception:
            continue
    return date_cols


def _build_temporal_analysis(df: pd.DataFrame, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    date_columns = _detect_temporal_date_columns(df)
    if not date_columns:
        return {
            "date_columns": [],
            "default_date_column": None,
            "default_frequency": "Quarterly",
            "frequency_options": ["Monthly", "Quarterly", "Half-Yearly", "Yearly"],
            "summary": None,
            "plot_data": [],
        }

    default_date_column = date_columns[0]
    frequency_options = ["Monthly", "Quarterly", "Half-Yearly", "Yearly"]
    freq_map = {
        "Monthly": "ME",
        "Quarterly": "QE",
        "Half-Yearly": "6ME",
        "Yearly": "YE",
    }
    plot_data_by_freq: Dict[str, Any] = {}
    summaries: Dict[str, Any] = {}
    for freq_name in frequency_options:
        freq_key = freq_map[freq_name]
        dates = df[default_date_column]
        summary = compute_temporal_stability_summary(dates, y_true, y_proba, freq=freq_key)
        plot_data = []
        for row in summary.get("by_period", []):
            plot_data.append({
                "period": row["period"],
                "actual_rate": row["actual_rate"],
                "predicted_rate": row["predicted_rate"],
                "gap": row["gap"],
                "flagged": row["flagged"],
            })
        plot_data_by_freq[freq_name] = plot_data
        summaries[freq_name] = summary
    return {
        "date_columns": date_columns,
        "default_date_column": default_date_column,
        "default_frequency": "Quarterly",
        "frequency_options": frequency_options,
        "summary": summaries.get("Quarterly"),
        "plot_data": plot_data_by_freq.get("Quarterly", []),
        "plot_data_by_freq": plot_data_by_freq,
        "summaries_by_freq": summaries,
    }


def _build_validation_intake_snapshot(mode: str = "clean") -> Dict[str, Any]:
    demo_mode = "flawed" if mode.lower() == "flawed" else "clean"
    if demo_mode == "flawed":
        model_data = {
            "model_name": "PD_Model_Internal_v1",
            "model_owner": "Unknown",
            "owning_team": "Risk",
            "lead_validator": "Reeyaz Miglani",
            "model_type": "PD (Probability of Default)",
            "model_version": "v1.0",
            "model_tier": "Tier 1 — High Risk",
            "model_purpose": "Credit risk model.",
            "target_col": "default",
            "algorithm": "XGBoost",
            "default_definition": "Internal default indicator",
            "calibration_method": "Not specified",
            "macro_variables_mentioned": False,
            "model_inventory_registered": False,
            "independence_confirmed": False,
        }
        reported_metrics = {
            "roc_auc": 0.78,
            "gini": 0.56,
            "ks": 0.48,
            "cv_mean_auc": 0.73,
            "accuracy": 0.79,
            "precision": 0.61,
            "recall": 0.72,
            "f1": 0.65,
        }
        mdd_file = SOURCE_OF_TRUTH_DIR / "demo_data" / "flawed_mdd.txt"
        hyperparams_file = SOURCE_OF_TRUTH_DIR / "demo_data" / "flawed_params.json"
        demo_label = "Demo B"
    else:
        model_data = {
            "model_name": "PD_XGBoost_RetailCredit_v2",
            "model_owner": "Sarah Chen",
            "owning_team": "Retail Credit Risk",
            "lead_validator": "Reeyaz Miglani",
            "model_type": "PD (Probability of Default)",
            "model_version": "v2.1.0",
            "model_tier": "Tier 2 — Medium Risk",
            "model_purpose": (
                "This model estimates the probability of default for unsecured retail credit customers. "
                "Used for IFRS 9 ECL staging, credit decisioning, and risk appetite reporting. "
                "Scope: personal loan portfolio only. Out of scope: mortgages, business lending."
            ),
            "algorithm": "XGBoost",
            "default_definition": "90 days past due (IFRS 9 / CRR Art.178)",
            "calibration_method": "Platt scaling (TTC)",
            "macro_variables_mentioned": True,
            "model_inventory_registered": True,
            "independence_confirmed": True,
        }
        reported_metrics = {
            "roc_auc": 0.82,
            "gini": 0.64,
            "ks": 0.58,
            "cv_mean_auc": 0.80,
            "accuracy": 0.85,
            "precision": 0.72,
            "recall": 0.71,
            "f1": 0.72,
        }
        mdd_file = SOURCE_OF_TRUTH_DIR / "demo_data" / "clean_mdd.txt"
        hyperparams_file = SOURCE_OF_TRUTH_DIR / "demo_data" / "clean_params.json"
        demo_label = "Demo A"

    # Derive the Risk Tier banner from the same model_tier used above (and
    # edited via the Model Metadata dropdown) so the two never disagree.
    # Cadence wording mirrors Stage 8's monitoring-frequency-by-tier logic
    # (see _build_stage8_findings's "driven by model tier" block) rather than
    # inventing new phrasing.
    tier_value = model_data["model_tier"].split("—")[0].strip()
    if "1" in tier_value or "High" in model_data["model_tier"]:
        tier_description = "High risk — monthly independent validation required."
    elif "2" in tier_value or "Medium" in model_data["model_tier"]:
        tier_description = "Material — quarterly independent validation required."
    else:
        tier_description = "Standard — semi-annual independent validation required."

    try:
        mdd_text = mdd_file.read_text(encoding="utf-8")
    except Exception:
        mdd_text = ""
    try:
        hyperparams = json.loads(hyperparams_file.read_text(encoding="utf-8"))
    except Exception:
        hyperparams = {}

    return {
        "demo_mode": demo_mode,
        "demo_label": demo_label,
        "val_intake_data": {**model_data, "mdd_text": mdd_text},
        "val_mdd_text": mdd_text,
        "val_mdd_reported_metrics": reported_metrics,
        "val_hyperparams": hyperparams,
        "chk_inventory": True,
        "chk_tier": True,
        "chk_artifacts": True,
        "chk_prev_findings": True,
        "chk_reg_scope": True,
        "chk_independence": True,
        "chk_plan_approved": True,
        "chk_attestation": True,
        "display": {
            "title": "Stage 1 — Intake & Governance",
            "description": "Capture model metadata, upload all required artifacts, and complete the governance attestation checklist before proceeding to automated validation stages.",
            "modelMetadata": {
                "title": "Model metadata",
                "description": "Key registration details supplied by the development team.",
                "registeredLabel": "Registered",
                "items": [
                    ["Model ID", "CR-PD-XGB-027"],
                    ["Model name", "Retail PD — XGBoost Champion"],
                    ["Owner", "A. Khurana · Risk Validation"],
                    ["Developer", "Credit Risk Modelling, EMEA"],
                    ["Version", "v1.7.6"],
                    ["Risk tier", "Tier 2 — Material"],
                    ["Last validated", "12 Apr 2026"],
                    ["Next review", "12 Jul 2026"],
                ],
            },
            "targetDefinition": {
                "title": "Target definition",
                "expression": "default_12m ∈ {0, 1}",
                "detail": "positive class = 90+ DPD within 12m",
                "baseRateLabel": "Base rate",
                "baseRate": "4.7%",
                "sampleSizeLabel": "Sample size",
                "sampleSize": "219,486",
            },
            "riskTier": {
                "title": "Risk tier",
                "value": tier_value,
                "description": tier_description,
            },
            "artifacts": [
                {"fileName": "retail_pd_validation.csv", "status": "Uploaded", "timestamp": "Uploaded 21 Jun 2026 · 09:13", "required": True},
                {"fileName": "retail_pd_mdd.pdf", "status": "Uploaded", "timestamp": "Uploaded 21 Jun 2026 · 09:15", "required": True},
                {"fileName": "training_pipeline.zip", "status": "Uploaded", "timestamp": "Uploaded 21 Jun 2026 · 09:17", "required": True},
                {"fileName": "data_profile.xlsx", "status": "Optional", "timestamp": "Pending review", "required": False},
                {"fileName": "assumptions_limitations.pdf", "status": "Optional", "timestamp": "Pending review", "required": False},
                {"fileName": "performance_report.xlsx", "status": "Optional", "timestamp": "Pending review", "required": False},
            ],
            "artifactSummary": "3 required · 3 optional",
            "artifactTitle": "Artifact inventory",
            "artifactDescription": "Uploaded evidence to support subsequent validation stages.",
            "governance": {
                "title": "Governance attestation",
                "description": "Confirm the model and validation plan are ready to proceed.",
                "status": "Pending review",
                "checklist": [
                    "Model is registered in the model inventory",
                    "Risk tier assignment has been documented",
                    "Submitted artifacts cover dataset, MDD, and training code",
                    "Previous validation findings (if any) have been reviewed",
                    "Regulatory scope (IFRS 9 / SS1/23 / SS11/13) is identified",
                    "Independent validation team has no conflict of interest",
                    "Validation plan has been approved by the Head of Model Risk",
                ],
            },
            "nextStep": {
                "description": "Once intake is confirmed, proceed to Stage 2 data validation and automated checks.",
                "label": "Proceed to Stage 2",
                "path": "/validation/data-quality",
            },
        },
    }



@app.get("/validation/intake")
async def validation_intake(mode: str = "clean") -> Dict[str, Any]:
    return _build_validation_intake_snapshot(mode)


# Intake draft persistence — file-based, keyed by model_name since there's no
# real user/session/auth system in this app yet. Deliberately simple: no
# uniqueness enforcement (two submissions named identically will overwrite
# each other's draft) — an accepted known limitation, not solved here.
INTAKE_DRAFTS_DIR = BACKEND_DIR / "intake_drafts"


def _intake_draft_slug(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", model_name.strip()).strip("_")
    if not slug:
        raise HTTPException(status_code=400, detail="model_name is required to save/load a draft.")
    return slug


class IntakeDraftRequest(BaseModel):
    model_name: str
    data: Dict[str, Any]


@app.post("/validation/intake/draft")
async def save_intake_draft(payload: IntakeDraftRequest) -> Dict[str, Any]:
    slug = _intake_draft_slug(payload.model_name)
    INTAKE_DRAFTS_DIR.mkdir(exist_ok=True)
    saved_at = pd.Timestamp.now().isoformat()
    record = {"model_name": payload.model_name, "data": payload.data, "saved_at": saved_at}
    (INTAKE_DRAFTS_DIR / f"{slug}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    result = {"saved": True, "model_name": payload.model_name, "saved_at": saved_at}

    try:
        persistence.log_event(
            persistence.VALIDATION_PIPELINE_LOG,
            stage="intake",
            payload={"model_name": payload.model_name},
            full_payload={**result, "data": payload.data},
        )
    except Exception as e:
        print(f"[persistence] failed to log intake: {e}")

    return result


@app.get("/validation/intake/draft")
async def get_intake_draft(model_name: str) -> Dict[str, Any]:
    slug = _intake_draft_slug(model_name)
    path = INTAKE_DRAFTS_DIR / f"{slug}.json"
    if not path.exists():
        return {"found": False}
    record = json.loads(path.read_text(encoding="utf-8"))
    return {"found": True, **record}


@app.get("/history/dev")
async def history_dev() -> Dict[str, Any]:
    return {"runs": persistence.read_log(persistence.DEV_PIPELINE_LOG)}


@app.get("/history/validation")
async def history_validation() -> Dict[str, Any]:
    return {"runs": persistence.read_log(persistence.VALIDATION_PIPELINE_LOG)}


@app.get("/history/artifact")
async def history_artifact(log_file: str, run_id: str) -> Dict[str, Any]:
    return {"artifact": persistence.get_artifact(log_file, run_id)}


@app.get("/history/latest")
async def history_latest(log_file: str, stage: str) -> Dict[str, Any]:
    return {"latest": persistence.get_latest(log_file, stage)}


@app.post("/data/feature-decision-log")
async def feature_decision_log(
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    target_col: str = Form(...),
    synthetic_samples: Optional[int] = Form(None),
) -> Dict[str, Any]:
    """Return the Feature Decision Log CSV (exact format as the original Streamlit app).
    This reconstructs the training-split plan and fe_summary then returns a CSV payload.
    """
    df = await _read_dataframe(file=file, csv_text=csv_text, synthetic_samples=synthetic_samples)
    col_types = detect_column_types(df)
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found")
    task_type = detect_task_type(df[target_col])
    X, y, _ = finalize_xy(df, col_types, target_col)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.15, val_size=0.15,
        task_type=task_type, random_state=42,
    )
    plan = analyze_for_feature_engineering(X_train, y_train, col_types, task_type)
    X_engineered, fe_summary = apply_feature_engineering(X_train, plan)

    # Recreate the exact log rows as in the Streamlit app
    _orig_cols = list(X.columns)
    _fe_iv = plan.get("iv_scores", {})
    _fe_vif = plan.get("multicollinearity", {}).get("vif", {})
    _transformed_src = set(
        plan.get("log_transform_cols", [])
        + plan.get("binning_cols", [])[:5]
        + plan.get("freq_encoding_cols", [])
        + plan.get("woe_cols", [])
    )
    _col_reason: dict = {}
    for _fstep in plan.get("applied_steps", []):
        for _fc in _fstep.get("columns", []):
            _col_reason[_fc] = f"{_fstep['step']}: {_fstep.get('reason','')}"
    _removed_set = set(fe_summary.get("removed", []))
    _log_rows = []
    for _fc in _orig_cols:
        if _fc in _removed_set:
            _dec, _rsn = "Removed", _col_reason.get(_fc, "Removed during feature engineering")
        elif _fc in _transformed_src:
            _dec, _rsn = "Transformed", _col_reason.get(_fc, "Transformation applied")
        else:
            _dec, _rsn = "Retained", "No transformation required"
        _log_rows.append({
            "Feature": _fc, "Decision": _dec, "Reason": _rsn,
            "IV Score": _fe_iv.get(_fc, ""), "VIF Score": _fe_vif.get(_fc, ""),
        })
    for _fc in fe_summary.get("added", []):
        _rsn = "New feature created during engineering"
        for _sfx in ("_log", "_bin", "_freq", "_woe"):
            if _fc.endswith(_sfx):
                _src = _fc[: -len(_sfx)]
                if _src in _col_reason:
                    _rsn = _col_reason[_src]
                    break
        if "_x_" in _fc and _rsn == "New feature created during engineering":
            for _part in _fc.split("_x_"):
                if _part in _col_reason:
                    _rsn = _col_reason[_part]
                    break
        _log_rows.append({
            "Feature": _fc, "Decision": "Added", "Reason": _rsn,
            "IV Score": _fe_iv.get(_fc, ""), "VIF Score": _fe_vif.get(_fc, ""),
        })

    # Create CSV bytes
    csv_bytes = pd.DataFrame(_log_rows).to_csv(index=False).encode("utf-8")
    return {"file_name": "feature_decision_log.csv", "content_base64": base64.b64encode(csv_bytes).decode("utf-8")}


class ComplianceRequest(BaseModel):
    stage: str
    payload: Dict[str, Any]


class TierRequest(BaseModel):
    training_config: Dict[str, Any]
    metrics: Dict[str, Any]
    fe_summary: Optional[Dict[str, Any]] = None


_agent2: Optional[Agent2] = None
_source_agent2: Optional["SourceAgent2"] = None
# The local 24-06/rag_store/val_mdd_rules.json copy is stale (pre-dates the
# "conceptual_soundness"/"data_validation" stage-name rename in the rules'
# `stage` field, still has the old "conceptual"/"data" short forms), which
# silently makes check_mdd_keywords()/check_for_validation() match zero rules.
# Load from the source-of-truth copy instead, same as ValidationAgent2 above.
VAL_MDD_RULES_PATH = SOURCE_OF_TRUTH_DIR / "rag_store" / "val_mdd_rules.json"


def _load_agent2() -> Optional[Agent2]:
    global _agent2
    if _agent2 is not None:
        return _agent2
    try:
        _agent2 = Agent2(str(RULES_PATH))
    except Exception:
        _agent2 = None
    return _agent2


class _SyncFileLike:
    """Wraps a FastAPI UploadFile's underlying SpooledTemporaryFile with the
    sync .name/.seek()/.read() interface parse_mdd_file() expects (ported from
    Streamlit's UploadedFile). SpooledTemporaryFile.name is a read-only property
    in this Python version, so it can't just be assigned onto the file object
    directly (that raises AttributeError, and call sites that swallow it end up
    silently parsing "" instead of the document — every keyword-search RAG rule
    then reports 0 keyword hits against an MDD that was never actually read).
    """

    def __init__(self, upload_file: UploadFile):
        self._file = upload_file.file
        self.name = upload_file.filename or ""

    def seek(self, *args, **kwargs):
        return self._file.seek(*args, **kwargs)

    def read(self, *args, **kwargs):
        return self._file.read(*args, **kwargs)


def _sync_file_like(upload_file: UploadFile) -> _SyncFileLike:
    upload_file.file.seek(0)
    return _SyncFileLike(upload_file)


def _load_source_agent2() -> Optional["SourceAgent2"]:
    """Cached loader for the RAG rule-matching Agent2 (val_mdd_rules.json)."""
    global _source_agent2
    if _source_agent2 is not None:
        return _source_agent2
    try:
        _source_agent2 = SourceAgent2(str(VAL_MDD_RULES_PATH))
    except Exception:
        _source_agent2 = None
    return _source_agent2


async def _read_dataframe(
    file: Optional[UploadFile] = None,
    csv_text: Optional[str] = None,
    synthetic_samples: Optional[int] = None,
) -> pd.DataFrame:
    if synthetic_samples and synthetic_samples > 0:
        return generate_synthetic_credit_dataset(n_samples=synthetic_samples)

    # csv_text represents the CURRENT state of the pipeline (e.g. after FRED
    # macro features, feature engineering, etc. have been attached and carried
    # forward from an earlier step) and must win whenever present. `file` is
    # only used as a fallback for the very first upload, before any csv_text
    # exists yet — if both happen to be sent together (e.g. the frontend still
    # holds onto the originally uploaded file in state), silently preferring
    # `file` would discard every transformation applied since the initial
    # upload with no error or signal that it happened.
    if csv_text:
        return pd.read_csv(io.StringIO(csv_text), keep_default_na=True)

    if file is not None:
        name = file.filename.lower()
        if name.endswith(".csv"):
            file.file.seek(0)
            return pd.read_csv(file.file, keep_default_na=True)
        if name.endswith(('.xls', '.xlsx')):
            file.file.seek(0)
            return pd.read_excel(file.file, engine="openpyxl")
        raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV or XLSX.")

    raise HTTPException(status_code=400, detail="Provide a file upload, CSV text, or synthetic_samples.")


def _serialize_dataframe(df: pd.DataFrame, max_rows: int = 5) -> Dict[str, Any]:
    return {
        "shape": list(df.shape),
        "columns": df.columns.astype(str).tolist(),
        "preview": df.head(max_rows).replace({pd.NA: None}).to_dict(orient="records"),
    }


def _json_safe_scalar(v: Any) -> Any:
    """Convert a single numpy/pandas scalar to a plain JSON-safe Python value —
    numpy int/float types aren't natively JSON-serializable, and non-finite
    floats (inf/-inf/nan) become the non-standard `Infinity`/`NaN` tokens that
    Python's json encoder tolerates but JavaScript's JSON.parse rejects."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _serialize_stage5_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    out = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            out[key] = _serialize_stage5_payload(value)
        elif isinstance(value, (list, tuple)):
            out[key] = [_serialize_stage5_payload(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, (np.integer, np.floating, np.bool_)):
            out[key] = _json_safe_scalar(value)
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            out[key] = None
        elif isinstance(value, (np.ndarray,)):
            out[key] = value.tolist()
        else:
            out[key] = _json_safe_scalar(value)
    return out


def _to_base64(obj: Any) -> str:
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _from_base64(encoded: str) -> Any:
    try:
        raw = base64.b64decode(encoded.encode("utf-8"))
        return joblib.load(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load artifact: {exc}")


async def _load_model_artifact(model_artifact: UploadFile) -> Any:
    """
    Read a base64-encoded, pickled model pipeline sent as a multipart FILE
    part (not a plain Form field). A serialized sklearn Pipeline (encoder +
    estimator) easily exceeds the non-file form-field size cap that
    Starlette/python-multipart enforce to keep memory-only fields small —
    file parts stream to a spooled temp file instead and aren't subject to
    that cap, which is why this must be File(...) rather than Form(...).
    """
    try:
        raw_b64 = (await model_artifact.read()).decode("utf-8")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read model artifact upload: {exc}")
    return _from_base64(raw_b64)


def _infer_target_column(df: pd.DataFrame, requested_target: Optional[str] = None) -> Optional[str]:
    if requested_target is not None:
        if requested_target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{requested_target}' not found")
        return requested_target

    target_candidates = detect_target_candidates(df)
    priority_names = {"default", "target", "label", "y", "loan_status"}
    for col in df.columns.astype(str):
        col_lower = col.lower()
        if col_lower in priority_names:
            return col

    if target_candidates:
        preferred = None
        for candidate in target_candidates:
            candidate_lower = candidate.lower()
            if any(keyword in candidate_lower for keyword in ("default", "target", "label", "y", "outcome", "flag", "risk")):
                preferred = candidate
                break
        if preferred is not None:
            return preferred
        return target_candidates[0]

    for col in df.columns.astype(str):
        col_lower = col.lower()
        if col_lower in {"loan_status", "default", "target", "label", "y"}:
            return col

    for col in reversed(df.columns.tolist()):
        if "id" in col.lower():
            continue
        col_vals = df[col].dropna().unique()
        if set(col_vals).issubset({0, 1, 0.0, 1.0}) and len(col_vals) == 2:
            return col

    return None


def _build_data_profile(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    col_types = detect_column_types(df)
    target_candidates = detect_target_candidates(df)
    resolved_target_col = _infer_target_column(df, target_col)
    if resolved_target_col is not None and not target_candidates:
        # Ensure the chosen target is also available as a candidate so the
        # frontend can reliably select it for downstream pages like feature
        # engineering when the heuristic finds it by name rather than by binary
        # target scoring.
        target_candidates = [resolved_target_col]
    task_type = None
    if resolved_target_col is not None:
        task_type = detect_task_type(df[resolved_target_col])

    leakage_risk_cols: List[str] = []
    if resolved_target_col is not None and task_type == "binary":
        target_numeric = pd.to_numeric(df[resolved_target_col], errors="coerce")
        if target_numeric.notna().sum() >= 2:
            for col in col_types.get("numeric", []):
                if col == resolved_target_col:
                    continue
                try:
                    corr = df[col].corr(target_numeric)
                    if pd.notna(corr) and abs(corr) > 0.95:
                        leakage_risk_cols.append(col)
                except Exception:
                    pass

    date_integrity: Dict[str, Any] = {}
    for dt_col in col_types.get("datetime", []):
        try:
            parsed = pd.to_datetime(df[dt_col], errors="coerce")
            valid = parsed.dropna()
            if valid.empty:
                continue
            today = pd.Timestamp.today().normalize()
            future_count = int((valid > today).sum())
            ancient_count = int((valid.dt.year < 1900).sum())
            date_integrity[dt_col] = {
                "min_date": str(valid.min().date()),
                "max_date": str(valid.max().date()),
                "future_count": future_count,
                "ancient_count": ancient_count,
            }
        except Exception:
            continue

    missing_cells = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1]) if df.shape[0] and df.shape[1] else 0
    missing_percentage = round((missing_cells / total_cells * 100) if total_cells else 0.0, 4)
    missing_by_column: Dict[str, Dict[str, Union[int, float]]] = {}
    for col in df.columns.astype(str):
        count = int(df[col].isna().sum())
        pct = round((count / df.shape[0] * 100) if df.shape[0] else 0.0, 4)
        missing_by_column[col] = {"count": count, "percentage": pct}

    duplicate_rows = int(df.duplicated().sum())
    duplicate_rate = round((duplicate_rows / len(df) * 100) if len(df) else 0.0, 4)
    numeric_feature_count = len(col_types.get("numeric", []))
    categorical_feature_count = len(col_types.get("categorical", []))

    outlier_analysis: Dict[str, Any] = {}
    for col in list(col_types.get("numeric", []))[:8]:
        try:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((series < lower) | (series > upper)).sum()
            outlier_frac = round(float(outliers / len(series)), 4) if len(series) else 0.0
            outlier_analysis[col] = {
                "outlier_count": int(outliers),
                "outlier_fraction": outlier_frac,
                "has_outliers": bool(outlier_frac > 0.02),
            }
        except Exception:
            continue

    class_distribution: Optional[Dict[str, int]] = None
    if resolved_target_col is not None and resolved_target_col in df.columns:
        class_distribution = {
            str(k): int(v)
            for k, v in df[resolved_target_col].value_counts(dropna=False).to_dict().items()
        }

    # For large datasets, sample data for expensive computations
    df_sample = df.sample(n=min(5000, len(df)), random_state=42) if len(df) > 5000 else df
    
    numeric_columns = list(col_types.get("numeric", []))
    numeric_columns = numeric_columns[:10]
    correlation_matrix: Dict[str, Any] = {"columns": numeric_columns, "values": []}
    if len(numeric_columns) >= 1:
        try:
            corr = df_sample[numeric_columns].corr(method="pearson")
            correlation_matrix["values"] = corr.fillna(0).round(4).values.tolist()
        except Exception:
            correlation_matrix["values"] = []

    summary_stats = []
    try:
        description = df_sample.describe(include="all").T.round(3)
        summary_stats = description.replace({np.nan: ""}).reset_index().rename(columns={"index": "Column"}).to_dict(orient="records")
    except Exception:
        summary_stats = []

    column_type_table = []
    for col in df.columns.astype(str):
        series = df[col]
        column_type_table.append({
            "Column": col,
            "Detected Type": next((t for t, cols in col_types.items() if col in cols), "unknown"),
            "Pandas Dtype": str(series.dtype),
            "Unique Values": int(series.nunique(dropna=True)),
            "Non-Null Count": int(series.notna().sum()),
        })

    distribution_histograms = []
    for col in list(col_types.get("numeric", []))[:4]:
        values = pd.to_numeric(df_sample[col], errors="coerce").dropna()
        if values.empty:
            distribution_histograms.append({"column": col, "bins": [], "counts": []})
            continue
        counts, bin_edges = np.histogram(values, bins=20)
        distribution_histograms.append({
            "column": col,
            "bins": [float(x) for x in bin_edges.tolist()],
            "counts": [int(x) for x in counts.tolist()],
        })

    data_dictionary = []
    for col in df.columns.astype(str):
        series = df[col]
        miss_n = int(series.isna().sum())
        miss_pct = round((miss_n / len(df) * 100) if len(df) else 0.0, 2)
        unique_values = int(series.nunique(dropna=True))
        is_num = pd.api.types.is_numeric_dtype(series)
        try:
            dd_min = round(float(series.min()), 4) if is_num else ""
            dd_max = round(float(series.max()), 4) if is_num else ""
            dd_mean = round(float(series.mean()), 4) if is_num else ""
        except Exception:
            dd_min = ""
            dd_max = ""
            dd_mean = ""
        sample_values = ", ".join(str(v) for v in series.dropna().unique()[:3])
        data_dictionary.append({
            "Column": col,
            "Detected Type": next((t for t, cols in col_types.items() if col in cols), "unknown"),
            "Missing Count": miss_n,
            "Missing %": miss_pct,
            "Unique Values": unique_values,
            "Min": dd_min,
            "Max": dd_max,
            "Mean": dd_mean,
            "Sample Values": sample_values,
        })

    target_summary: Dict[str, Any] = {}
    if resolved_target_col is not None and resolved_target_col in df.columns:
        target_values = df[resolved_target_col].dropna()
        task_label = "Unknown"
        if task_type == "binary":
            task_label = "Binary classification"
        elif task_type == "multiclass":
            task_label = "Multiclass classification"
        elif task_type == "regression":
            task_label = "Regression"

        imbalance_ratio: Optional[float] = None
        is_imbalanced = False
        if len(target_values) > 0:
            counts = target_values.value_counts(dropna=False)
            if len(counts) >= 2 and counts.min() > 0:
                imbalance_ratio = round(float(counts.max() / counts.min()), 2)
                is_imbalanced = imbalance_ratio > 3

        target_summary = {
            "selected_target": resolved_target_col,
            "task_type": task_type,
            "task_label": task_label,
            "target_candidates": target_candidates,
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": is_imbalanced,
            "suggestion": (
                "Imbalanced target distribution detected; consider class weighting or resampling."
                if is_imbalanced else "Target distribution appears balanced for the current profile."
            ),
        }

    profile = {
        "shape": list(df.shape),
        "columns": df.columns.astype(str).tolist(),
        "col_types": col_types,
        "target_col": resolved_target_col,
        "target_candidates": target_candidates,
        "task_type": task_type,
        "dataset_name": dataset_name,
        "leakage_risk_cols": leakage_risk_cols,
        "date_integrity": date_integrity,
        "missing_cells": missing_cells,
        "missing_percentage": missing_percentage,
        "missing_by_column": missing_by_column,
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": duplicate_rate,
        "numeric_feature_count": numeric_feature_count,
        "categorical_feature_count": categorical_feature_count,
        "class_distribution": class_distribution,
        "outlier_analysis": outlier_analysis,
        "correlation_matrix": correlation_matrix,
        "summary_stats": summary_stats,
        "column_type_table": column_type_table,
        "distribution_histograms": distribution_histograms,
        "data_dictionary": data_dictionary,
        "data_preview": _serialize_dataframe(df, max_rows=10)["preview"],
        "target_summary": target_summary,
    }

    if resolved_target_col is not None:
        agent = _load_agent2()
        if agent is not None:
            try:
                agent.check_data(df, col_types, leakage_risk_cols=leakage_risk_cols)
                agent.check_rules_from_agent1("data", {
                    "n_rows": len(df),
                    "n_cols": len(df.columns),
                    "missing_pct": round(float(df.isna().mean().mean()), 4),
                    "target_col": resolved_target_col,
                })
                profile["agent2_flags_data"] = agent.get_stage_summary("data")["flags"]
                profile["agent2_report"] = agent.get_full_report()
            except Exception as exc:
                profile["agent2_error"] = f"Agent2 data check failed: {exc}"
                profile["agent2_flags_data"] = []
                profile["agent2_report"] = {
                    "metadata": {"generated_at": None, "rules_source": "rag_store/rules.json", "stages_checked": [], "total_rules": 0, "rules_passed": 0, "rules_flagged": 0, "compliance_score": 0},
                    "summary": {"high_severity": 0, "medium_severity": 0, "low_severity": 0, "overall_status": "WARN"},
                    "flags_by_stage": {"data": []},
                    "flags_by_source": {},
                    "all_flags": [],
                    "model_tier": None,
                }

    return profile


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@app.post("/data/upload")
async def upload_data(
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    synthetic_samples: Optional[int] = Form(None),
    demo_mode: Optional[str] = Form(None),
) -> Dict[str, Any]:
    if demo_mode and file is None and csv_text is None and not synthetic_samples:
        demo_mode = demo_mode.lower()
        demo_file = SOURCE_OF_TRUTH_DIR / "demo_data" / ("flawed_portfolio.csv" if demo_mode == "flawed" else "clean_portfolio.csv")
        if not demo_file.exists():
            raise HTTPException(status_code=400, detail=f"Demo dataset not found for mode '{demo_mode}'.")
        df = pd.read_csv(demo_file, keep_default_na=True)
        dataset_name = "Demo B dataset" if demo_mode == "flawed" else "Demo A dataset"
        profile = _build_data_profile(df, dataset_name=dataset_name)
        profile["csv_text"] = df.to_csv(index=False)
        profile["source_type"] = "demo"
        profile["demo_mode"] = demo_mode
        try:
            persistence.log_event(
                persistence.DEV_PIPELINE_LOG,
                stage="data_upload",
                payload={
                    "filename": dataset_name,
                    "row_count": int(df.shape[0]),
                    "column_count": int(df.shape[1]),
                },
                full_payload=profile,
            )
        except Exception as e:
            print(f"[persistence] failed to log data_upload: {e}")
        return profile

    df = await _read_dataframe(file=file, csv_text=csv_text, synthetic_samples=synthetic_samples)
    dataset_name = file.filename if file is not None else "Synthetic Credit Dataset"
    profile = _build_data_profile(df, dataset_name=dataset_name)
    if synthetic_samples and synthetic_samples > 0 and file is None:
        profile["csv_text"] = df.to_csv(index=False)
        profile["source_type"] = "synthetic"
        profile["synthetic_samples"] = int(synthetic_samples)
    else:
        profile["source_type"] = "file"
    try:
        persistence.log_event(
            persistence.DEV_PIPELINE_LOG,
            stage="data_upload",
            payload={
                "filename": dataset_name,
                "row_count": int(df.shape[0]),
                "column_count": int(df.shape[1]),
            },
            full_payload=profile,
        )
    except Exception as e:
        print(f"[persistence] failed to log data_upload: {e}")
    return profile


def _fetch_remote_csv(api_url: str, http_method: str = "GET") -> str:
    method = http_method.upper().strip()
    if method != "GET":
        raise HTTPException(status_code=400, detail="Only GET is supported for API data sources.")
    try:
        parsed = urllib.parse.urlparse(api_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("API URL must use http or https")
        request = urllib.request.Request(api_url, method="GET", headers={"User-Agent": "AegisCreditDataUploader/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")
            content_type = response.headers.get("Content-Type", "")
            if "csv" not in content_type.lower() and "text" not in content_type.lower():
                raise HTTPException(status_code=400, detail="Only CSV API responses are currently supported.")
            return text
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=exc.code, detail=f"API request failed: {exc.reason}")
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=400, detail=f"API request failed: {exc.reason}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _normalize_db_type(db_type: str) -> str:
    if db_type is None:
        return ""
    return db_type.strip().lower()


def _build_database_url(
    db_type: str,
    host: Optional[str],
    port: Optional[str],
    database: Optional[str],
    username: Optional[str],
    password: Optional[str],
) -> str:
    db_type = _normalize_db_type(db_type)
    database = (database or "").strip()
    if not database:
        raise HTTPException(status_code=400, detail="Database name/path is required.")

    if db_type == "sqlite":
        if database == ":memory:":
            return "sqlite:///:memory:"
        path = Path(database)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        return f"sqlite:///{path.as_posix()}"

    username = (username or "").strip()
    password = (password or "").strip()
    host = (host or "localhost").strip() or "localhost"
    port = (port or "").strip()

    if db_type == "postgresql":
        driver = "psycopg2"
        default_port = "5432"
    elif db_type == "mysql":
        driver = "pymysql"
        default_port = "3306"
    elif db_type in ("sqlserver", "mssql"):
        driver = "pyodbc"
        default_port = "1433"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported database type '{db_type}'.")

    port = port or default_port
    if db_type in ("sqlserver", "mssql"):
        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password are required for SQL Server.")
        driver_query = "?driver=ODBC+Driver+17+for+SQL+Server"
        return f"mssql+{driver}://{username}:{password}@{host}:{port}/{database}{driver_query}"

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required for database connections.")

    return f"{db_type}+{driver}://{username}:{password}@{host}:{port}/{database}"


def _create_database_engine(
    db_type: str,
    host: Optional[str],
    port: Optional[str],
    database: Optional[str],
    username: Optional[str],
    password: Optional[str],
):
    url = _build_database_url(db_type, host, port, database, username, password)
    try:
        engine = create_engine(url, future=True, connect_args={"connect_timeout": 10} if _normalize_db_type(db_type) != "sqlite" else {})
        return engine
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=400, detail=f"Database URL error: {exc}")


def _inspect_database_tables(engine) -> List[str]:
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    views = inspector.get_view_names()
    return sorted(set(tables + views))


@app.post("/data/database/test")
async def test_database_connection(
    db_type: str = Form(...),
    host: Optional[str] = Form(None),
    port: Optional[str] = Form(None),
    database: Optional[str] = Form(None),
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
) -> Dict[str, Any]:
    engine = _create_database_engine(db_type, host, port, database, username, password)
    try:
        with engine.connect() as connection:
            # SQLAlchemy 2.0 requires an executable SQL expression; use text()
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=400, detail=f"Database connection failed: {exc}")
    finally:
        engine.dispose()
    return {"status": "ok"}


@app.post("/data/database/tables")
async def list_database_tables(
    db_type: str = Form(...),
    host: Optional[str] = Form(None),
    port: Optional[str] = Form(None),
    database: Optional[str] = Form(None),
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
) -> Dict[str, Any]:
    engine = _create_database_engine(db_type, host, port, database, username, password)
    try:
        tables = _inspect_database_tables(engine)
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=400, detail=f"Failed to list tables: {exc}")
    finally:
        engine.dispose()
    return {"tables": tables}


@app.post("/data/database/fetch")
async def fetch_database_table(
    db_type: str = Form(...),
    host: Optional[str] = Form(None),
    port: Optional[str] = Form(None),
    database: Optional[str] = Form(None),
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    selected_table: str = Form(...),
) -> Dict[str, Any]:
    engine = _create_database_engine(db_type, host, port, database, username, password)
    try:
        if "." in selected_table:
            schema, table_name = selected_table.split(".", 1)
        else:
            schema = None
            table_name = selected_table

        try:
            df = pd.read_sql_table(table_name, con=engine, schema=schema)
        except Exception:
            query = f"SELECT * FROM {selected_table}"
            df = pd.read_sql_query(query, con=engine)
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch table '{selected_table}': {exc}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch table '{selected_table}': {exc}")
    finally:
        engine.dispose()

    profile = _build_data_profile(df, dataset_name=selected_table)
    profile["csv_text"] = df.to_csv(index=False)
    profile["source_type"] = "database"
    profile["database_type"] = db_type
    profile["database_host"] = host
    profile["database_name"] = database
    profile["database_table"] = selected_table
    return profile


def _save_upload_to_temp(file: UploadFile, suffix: str = ".db") -> str:
    """SQLite needs a real file path, not an in-memory stream — write the
    upload to a temp file. Caller is responsible for deleting it (see
    try/finally at each call site below)."""
    file.file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        return tmp.name


def _unlink_quietly(path: str) -> None:
    """Best-effort temp-file cleanup. On Windows a sqlite3 connection can
    still hold the file open for a moment after its Python object goes out
    of scope, so os.unlink() can raise PermissionError here even though the
    actual request already completed successfully — that shouldn't turn a
    good response into a 500."""
    try:
        os.unlink(path)
    except OSError:
        pass


@app.post("/data/integration/sqlite/inspect")
async def integration_sqlite_inspect(db_file: UploadFile = File(...)) -> Dict[str, Any]:
    """Part B — upload a SQLite database and discover every table plus its
    schema, so the frontend can populate the Loan / Collateral table
    dropdowns. Nothing here assumes specific table names."""
    tmp_path = _save_upload_to_temp(db_file)
    try:
        tables = di.list_tables(tmp_path)
        if not tables:
            raise HTTPException(status_code=400, detail="No tables found in the uploaded database.")
        return {"tables": [di.inspect_table(tmp_path, t) for t in tables]}
    except di.DataIntegrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        _unlink_quietly(tmp_path)


@app.post("/data/integration/relationships")
async def integration_relationships(
    db_file: Optional[UploadFile] = File(None),
    customer_file: Optional[UploadFile] = File(None),
    customer_csv_text: Optional[str] = Form(None),
    selected_tables: str = Form(...),  # comma-separated table names from the db
) -> Dict[str, Any]:
    """Part C — automatic relationship discovery across the customer dataset
    and whichever SQLite tables the user selected. Returns ranked join
    candidates with confidence + human-readable reasons; the frontend
    pre-fills the suggested (highest-confidence) join per table pair but
    always lets the user override it before running the integration."""
    table_names = [t.strip() for t in selected_tables.split(",") if t.strip()]
    if not table_names:
        raise HTTPException(status_code=400, detail="No tables selected.")

    tables: Dict[str, pd.DataFrame] = {}
    tmp_path = None
    try:
        if db_file is not None:
            tmp_path = _save_upload_to_temp(db_file)
            available = set(di.list_tables(tmp_path))
            for t in table_names:
                if t not in available:
                    raise HTTPException(status_code=400, detail=f"Table '{t}' not found in the uploaded database.")
                tables[t] = di.SQLiteDataSource(t, tmp_path, t).load()

        if customer_file is not None or customer_csv_text:
            customer_df = await _read_dataframe(file=customer_file, csv_text=customer_csv_text)
            tables["customer"] = customer_df

        if len(tables) < 2:
            raise HTTPException(status_code=400, detail="At least two sources are required to discover relationships.")

        primary_keys, candidates = di.discover_relationships(tables)
        return {
            "primary_keys": [pk.as_dict() for pk in primary_keys],
            "candidates": [c.as_dict() for c in candidates],
        }
    except di.DataIntegrationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if tmp_path:
            _unlink_quietly(tmp_path)


@app.post("/data/integration/run")
async def integration_run(
    customer_file: Optional[UploadFile] = File(None),
    customer_csv_text: Optional[str] = Form(None),
    db_file: Optional[UploadFile] = File(None),
    loan_table: Optional[str] = Form(None),
    collateral_table: Optional[str] = Form(None),
    join_specs_json: Optional[str] = Form(None),
    fetch_macro: bool = Form(False),
    macro_date_col: Optional[str] = Form(None),
    fred_api_key: Optional[str] = Form(None),
    join_how: str = Form("left"),
) -> Dict[str, Any]:
    """Parts D/E/F/G/I — run the full integration: merge the customer CSV
    with whichever SQLite tables were selected using the user-confirmed join
    keys (join_specs_json — a JSON list of {right_table, left_key,
    right_key} the frontend built from discover_relationships' suggestions,
    possibly overridden), then optionally attach FRED macro series by date.

    Returns the SAME shape /data/upload already returns (via
    _build_data_profile + csv_text) so nothing downstream — preprocessing,
    feature engineering, model training — needs to change. See Part I.
    """
    if customer_file is None and not customer_csv_text:
        raise HTTPException(status_code=400, detail="Customer data (file or csv_text) is required as the base of the integration.")

    customer_df = await _read_dataframe(file=customer_file, csv_text=customer_csv_text)
    source_metadata: List[di.SourceMetadata] = [di.CSVDataSource("customer", customer_df).metadata(customer_df)]

    tmp_path = None
    try:
        tables: Dict[str, pd.DataFrame] = {}
        selected_sqlite_tables = [t for t in (loan_table, collateral_table) if t]
        if selected_sqlite_tables:
            if db_file is None:
                raise HTTPException(status_code=400, detail="A SQLite database file is required when a loan/collateral table is selected.")
            tmp_path = _save_upload_to_temp(db_file)
            for t in selected_sqlite_tables:
                source = di.SQLiteDataSource(t, tmp_path, t)
                tdf = source.load()
                tables[t] = tdf
                source_metadata.append(source.metadata(tdf))

        try:
            join_specs_raw = json.loads(join_specs_json) if join_specs_json else []
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="join_specs_json must be valid JSON.")

        join_specs = [
            di.TableJoinSpec(
                right_table=spec["right_table"], left_key=spec["left_key"], right_key=spec["right_key"],
                how=di.JoinStrategy(spec.get("how", join_how)),
            )
            for spec in join_specs_raw
        ]

        integrator = di.DatasetIntegrator()
        try:
            integrated = integrator.integrate("customer", customer_df, tables, join_specs, source_metadata)
        except di.DataIntegrationError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if fetch_macro:
            api_key = (fred_api_key or "").strip()
            if not api_key:
                integrator.report.warnings.append("No FRED API key was provided — macro data was not fetched.")
            else:
                integrated = integrator.attach_macro(integrated, macro_date_col, fred_client, api_key)

        dataset_name = "Integrated dataset"
        profile = _build_data_profile(integrated, dataset_name=dataset_name)
        profile["csv_text"] = integrated.to_csv(index=False)
        profile["source_type"] = "integrated"
        profile["integration_report"] = integrator.report.as_dict()
        return profile
    finally:
        if tmp_path:
            _unlink_quietly(tmp_path)



async def data_profile(
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    target_col: Optional[str] = Form(None),
    synthetic_samples: Optional[int] = Form(None),
) -> Dict[str, Any]:
    df = await _read_dataframe(file=file, csv_text=csv_text, synthetic_samples=synthetic_samples)
    dataset_name = file.filename if file is not None else None
    profile = _build_data_profile(df, target_col=target_col, dataset_name=dataset_name)
    if target_col is not None:
        profile["preprocessing_report"] = build_preprocessing_report(df, profile["col_types"], target_col)
    return profile


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalar types to native Python so dict/list
    structures built from pandas/numpy computations serialize cleanly."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_safe(obj.to_dict(orient="records"))
    return obj


def _expand_plotly_bdata(obj):
    """
    Recursively expand Plotly's compact typed-array JSON encoding back into
    plain number lists.

    Since plotly.py ~5.24+, fig.to_json() encodes numeric trace arrays as
    {"dtype": "f8", "bdata": "<base64>"} instead of a plain JSON array, to
    shrink payload size. This requires a fairly recent plotly.js on the
    frontend to decode — older/most bundled versions just silently fail to
    plot that trace, so a chart's tab looks like nothing happened when
    clicked even though real figure JSON is present. Expanding it back to
    plain arrays here keeps this working regardless of the frontend's
    plotly.js version.
    """
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj and isinstance(obj["bdata"], str):
            try:
                raw = base64.b64decode(obj["bdata"])
                flat = np.frombuffer(raw, dtype=obj["dtype"])
                # Multi-dimensional arrays (e.g. heatmap z) are flattened into
                # a single bdata blob with a companion "shape" field like
                # "2, 2" — without reshaping here we'd hand the frontend a
                # flat list where it expects nested rows, which breaks any
                # code that does z.map(row => row.map(...)).
                shape = obj.get("shape")
                if shape:
                    dims = [int(d) for d in str(shape).split(",") if d.strip()]
                    if len(dims) > 1:
                        flat = flat.reshape(dims)
                return flat.tolist()
            except Exception:
                return obj
        return {k: _expand_plotly_bdata(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_plotly_bdata(v) for v in obj]
    return obj


def _figure_json(fig) -> Optional[Dict[str, Any]]:
    """
    Convert a Plotly go.Figure into a plain JSON-safe dict for the
    PlotlyChart React component. Goes through fig.to_json() (Plotly's own
    encoder) rather than fig.to_dict(), since traces built directly from
    numpy arrays (as the eval_engine plot_* functions do) aren't otherwise
    serializable by FastAPI's default JSON encoder — then expands any
    compact typed-array encoding back to plain arrays (see
    _expand_plotly_bdata) so the frontend can actually render it.
    """
    if fig is None:
        return None
    try:
        return _expand_plotly_bdata(json.loads(fig.to_json()))
    except Exception:
        return None


def _build_evaluation_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    task_type: str,
    threshold: Optional[float] = None,
    dates: Optional[pd.Series] = None,
    date_columns: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Shared by /models/train and /models/evaluate: computes the metric suite
    (with KS statistic + Brier score for binary tasks) and builds the full
    evaluation_data payload the Evaluation page's charts read from — Plotly
    figure JSON per tab (ROC/PR/confusion/threshold/score-distribution/lift),
    the heteroscedasticity residual check, and the temporal-stability bundle
    (Monthly/Quarterly/Yearly actual-vs-predicted) when an origination date
    is available. Returns (metrics, evaluation_data).

    `threshold=None` (the default) auto-selects the F1-maximizing threshold
    via compute_binary_metrics()/select_best_threshold() instead of a fixed
    0.5 cut-off — every downstream chart (confusion matrix, threshold curve,
    etc.) is built against that same resolved threshold so nothing drifts
    out of sync. Pass an explicit float to pin a specific operating point
    instead (e.g. a reviewer-chosen threshold on the /models/evaluate page).
    """
    if task_type == "binary":
        metrics = compute_binary_metrics(y_true, y_pred, y_proba, threshold=threshold)
        # Resolve to the threshold actually used (auto-selected or explicit)
        # so every chart/payload field below is consistent with `metrics`.
        threshold = metrics.get("threshold_used", threshold if threshold is not None else 0.5)
        hetero_input = y_proba if y_proba is not None else y_pred
        roc_curve_pts = compute_roc_curve(y_true, y_proba) if y_proba is not None else []
        pr_curve_pts = compute_pr_curve(y_true, y_proba) if y_proba is not None else []
        threshold_pts = compute_threshold_analysis(y_true, y_proba) if y_proba is not None else []
        score_dist_pts = compute_score_distribution(y_true, y_proba) if y_proba is not None else []
        gain_chart_pts = compute_gain_chart(y_true, y_proba) if y_proba is not None else []
        lift_chart_pts = compute_lift_chart(y_true, y_proba) if y_proba is not None else []
    else:
        metrics = compute_regression_metrics(y_true, y_pred)
        hetero_input = y_pred
        roc_curve_pts = pr_curve_pts = threshold_pts = score_dist_pts = gain_chart_pts = lift_chart_pts = []

    hetero_check = compute_heteroscedasticity_check(y_true, hetero_input, task_type=task_type)

    roc_fig = pr_fig = threshold_fig = score_dist_fig = lift_fig = confusion_fig = None
    if task_type == "binary" and y_proba is not None:
        try:
            roc_fig = _figure_json(plot_roc_curve(y_true, y_proba))
            pr_fig = _figure_json(plot_pr_curve(y_true, y_proba))
            threshold_fig = _figure_json(plot_threshold_analysis(y_true, y_proba))
            score_dist_fig = _figure_json(plot_score_distribution(np.asarray(y_true).astype(int), y_proba))
            lift_fig = _figure_json(plot_lift_chart(y_true, y_proba))
        except Exception:
            pass
    if task_type == "binary" and metrics.get("confusion_matrix"):
        try:
            confusion_fig = _figure_json(
                plot_confusion_matrix(metrics["confusion_matrix"], labels=["Non-Default (0)", "Default (1)"])
            )
        except Exception:
            confusion_fig = None

    temporal_analysis = None
    if task_type == "binary" and y_proba is not None and dates is not None:
        try:
            temporal_analysis = compute_temporal_analysis_bundle(
                dates, y_true, y_proba, date_columns=date_columns or [],
            )
        except Exception:
            temporal_analysis = None

    evaluation_data = {
        "metrics": metrics,
        "heteroscedasticity_check": hetero_check,
        "threshold": threshold,
        "threshold_selection": metrics.get("threshold_selection") if task_type == "binary" else None,
        "task_type": task_type,
        "roc_curve": roc_curve_pts,
        "pr_curve": pr_curve_pts,
        "threshold_analysis": threshold_pts,
        "score_distribution": score_dist_pts,
        "gain_chart": gain_chart_pts,
        "lift_chart": lift_chart_pts,
        "roc_curve_figure": roc_fig,
        "pr_curve_figure": pr_fig,
        "confusion_matrix_figure": confusion_fig,
        "threshold_analysis_figure": threshold_fig,
        "score_distribution_figure": score_dist_fig,
        "lift_chart_figure": lift_fig,
        "temporal_analysis": temporal_analysis,
    }
    return metrics, evaluation_data


@app.post("/data/preprocess")
async def preprocess_data(
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    target_col: str = Form(...),
    synthetic_samples: Optional[int] = Form(None),
    test_size: float = Form(0.15),
    val_size: float = Form(0.15),
    random_seed: int = Form(42),
    # ── Confirmed reviewer choices (all optional; sensible "accept the
    #    platform's proposal" defaults apply when omitted) ──
    treatment_overrides: Optional[str] = Form(None),  # JSON {col: treatment}
    drop_cols: Optional[str] = Form(None),             # JSON [col, ...]
    transform_choices: Optional[str] = Form(None),     # JSON {col: "none"|"log1p"|"yeo_johnson"}
    strategy_override: Optional[str] = Form(None),     # "mice" | "knn" | "median"
) -> Dict[str, Any]:
    df = await _read_dataframe(file=file, csv_text=csv_text, synthetic_samples=synthetic_samples)
    col_types = detect_column_types(df)
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found")

    test_size = float(test_size)
    val_size = float(val_size)
    if not 0 < test_size < 1:
        raise HTTPException(status_code=400, detail="test_size must be between 0 and 1")
    if not 0 < val_size < 1:
        raise HTTPException(status_code=400, detail="val_size must be between 0 and 1")
    if test_size + val_size >= 1:
        raise HTTPException(status_code=400, detail="test_size + val_size must be less than 1")

    def _parse_json_form(raw: Optional[str], default):
        if not raw:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in request: {exc}")

    _treatment_overrides: Dict[str, str] = _parse_json_form(treatment_overrides, {})
    _drop_cols: List[str] = _parse_json_form(drop_cols, [])
    _transform_choices: Dict[str, str] = _parse_json_form(transform_choices, {})

    task_type = detect_task_type(df[target_col])
    X, y, clean_info = finalize_xy(df, col_types, target_col)

    # Split FIRST — every statistic learned below (missing-value treatment
    # proposal, imputation strategy, skew/transform recommendations, the
    # fitted preprocessor) comes from the TRAINING split only. This matches
    # /models/train and the Streamlit reference app's leakage-safe design;
    # the previous version of this endpoint built its report from the full
    # pre-split dataset, which this fixes.
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=test_size, val_size=val_size,
        task_type=task_type, random_state=random_seed,
    )

    # ── Phase 1: PROPOSE — classify_missing_treatment() on TRAIN only.
    #    Nothing is applied yet; this is purely diagnostic. ──
    missing_treatment_proposal = classify_missing_treatment(X_train, col_types)

    # ── Phase 2: resolve the CONFIRMED treatment per column ──
    # A column resolved to "review_flag" (>40% missing) that the reviewer
    # chose to KEEP (not in drop_cols) can't stay "review_flag" — that
    # treatment means "leave untouched", so it needs a real, recalibrated
    # treatment instead. See classify_missing_treatment's force_include_cols
    # docstring.
    treatment_map: Dict[str, Dict[str, Any]] = {}
    recalibrated_cols: List[Dict[str, str]] = []
    for col, info in missing_treatment_proposal.items():
        if col in _drop_cols:
            continue
        resolved_treatment = _treatment_overrides.get(col, info["treatment"])

        if resolved_treatment == "review_flag":
            recalibrated = classify_missing_treatment(
                X_train[[col]], col_types, force_include_cols=[col]
            )
            if col in recalibrated:
                treatment_map[col] = {
                    **recalibrated[col],
                    "reason": (
                        f"Recalibrated — kept despite "
                        f"{info['evidence'].get('missing_pct', 0):.1%} missing (over the "
                        f"{int(REVIEW_MISSING_THRESHOLD * 100)}% review threshold). "
                        f"{recalibrated[col]['reason']}"
                    ),
                }
                recalibrated_cols.append({"column": col, "treatment": treatment_map[col]["treatment"]})
                continue
            treatment_map[col] = {
                "treatment": "statistical", "reason": info["reason"], "evidence": info["evidence"],
            }
            continue

        if col in _treatment_overrides:
            treatment_map[col] = {
                "treatment": resolved_treatment,
                "reason": f"Manually overridden by reviewer (platform proposed: {info['treatment']}).",
                "evidence": info["evidence"],
            }
        else:
            treatment_map[col] = info

    # ── Phase 3: joint imputation strategy for the "statistical" block ──
    statistical_cols = [c for c, v in treatment_map.items() if v["treatment"] == "statistical"]
    imputation_strategy = select_imputation_strategy(X_train, statistical_cols)
    if strategy_override in ("mice", "knn", "median") and statistical_cols:
        imputation_strategy = {
            "method": strategy_override,
            "reason": (
                f"Manually overridden by reviewer "
                f"(platform proposed: {imputation_strategy['method']})."
            ),
            "diagnostics": imputation_strategy.get("diagnostics", {}),
        }

    # ── Phase 4: fit SemanticImputer on TRAIN only, apply unchanged to val/test ──
    col_types_for_fit = {k: [c for c in v if c not in _drop_cols] for k, v in col_types.items()}
    imputer = SemanticImputer(
        col_types=col_types_for_fit, treatment_map=treatment_map, strategy_choice=imputation_strategy,
    )
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    _drop_cols_present = [c for c in _drop_cols if c in X_train.columns]
    if _drop_cols_present:
        X_train = X_train.drop(columns=_drop_cols_present)
        X_val = X_val.drop(columns=[c for c in _drop_cols_present if c in X_val.columns], errors="ignore")
        X_test = X_test.drop(columns=[c for c in _drop_cols_present if c in X_test.columns], errors="ignore")

    # ── Phase 5: preprocessing report (scaler/encoder strategy + skew/transform
    #    recommendations) on the now-imputed TRAIN split. transform_choices is
    #    what actually drives the fitted ColumnTransformer below — nothing
    #    here auto-applies a log/Yeo-Johnson transform on its own. ──
    prep_report = build_preprocessing_report(
        X_train.assign(**{target_col: y_train}), col_types, target_col,
        transform_choices=_transform_choices,
    )

    preprocessor = rebuild_preprocessor_for(X_train, col_types, target_col, prep_report)
    preprocessor.fit(X_train)
    processed_matrix = preprocessor.transform(X_train)
    processed_feature_names = get_feature_names_from_fitted_preprocessor(preprocessor)
    processed_df = pd.DataFrame(processed_matrix, columns=processed_feature_names)
    processed_df[target_col] = y_train.reset_index(drop=True)

    def _summary_row(column: str, feature_type: str, scaler: str, imputer_label: str, encoding: str, outlier_strategy: str, transform: str):
        return {
            "feature": column,
            "type": feature_type,
            "scaler": scaler,
            "imputer": imputer_label,
            "encoding": encoding,
            "outlier_strategy": outlier_strategy,
            "transform": transform,
        }

    strategy_summary = []
    for col, info in prep_report.get("numeric", {}).items():
        scaler = "Robust" if info.get("scaler") == "robust" else "Standard"
        imputer_label = info.get("imputer", "mean").capitalize()
        encoding = "-"
        outlier_strategy = "Robust scaling" if info.get("has_outliers") else "Standard scaling"
        confirmed_t = _transform_choices.get(col, "none")
        rec_t = (info.get("transform_recommendation") or {}).get("transform", "none")
        transform_label = (
            f"{confirmed_t.replace('_', '-').title()} (confirmed)" if confirmed_t in ("log1p", "yeo_johnson")
            else (f"Suggested: {rec_t.replace('_', '-').title()}" if rec_t in ("log1p", "yeo_johnson") else "-")
        )
        strategy_summary.append(_summary_row(col, "Numeric", scaler, imputer_label, encoding, outlier_strategy, transform_label))

    for col, info in prep_report.get("categorical", {}).items():
        scaler = "-"
        imputer_label = "Mode"
        encoding = "OneHot" if info.get("encoding") == "onehot" else "Ordinal"
        outlier_strategy = "-"
        strategy_summary.append(_summary_row(col, "Categorical", scaler, imputer_label, encoding, outlier_strategy, "-"))

    for col in prep_report.get("boolean", {}):
        strategy_summary.append(_summary_row(col, "Boolean", "-", "-", "-", "-", "-"))

    for col in prep_report.get("datetime", {}):
        strategy_summary.append(_summary_row(col, "Datetime", "-", "-", "-", "-", "-"))

    feature_names = list(X_train.columns)
    split_stats = compute_split_stats(X_train, X_val, X_test, y_train, y_val, y_test)

    class_distribution_chart = []
    for split_name in ["train", "val", "test"]:
        class_dist = split_stats.get(f"{split_name}_class_dist", {}) or {}
        for cls, prop in class_dist.items():
            class_distribution_chart.append({
                "split": split_name.capitalize(),
                "class": str(cls),
                "proportion": float(prop),
            })

    summary_metrics = {
        "features_basic": X_train.shape[1],
        "numeric_columns": len(prep_report.get("numeric", {})),
        "categorical_columns": len(prep_report.get("categorical", {})),
        "boolean_columns": len(prep_report.get("boolean", {})),
        "datetime_columns": len(prep_report.get("datetime", {})),
        "duplicates_removed": clean_info.get("duplicates_removed", 0),
        "ecl_only_cols_dropped": clean_info.get("ecl_only_cols_dropped", []),
    }

    result = {
        "col_types": col_types,
        "target_col": target_col,
        "feature_names": feature_names,
        "processed_feature_names": processed_feature_names,
        "x_shape": list(X_train.shape),
        "y_shape": list(y_train.shape),
        "feature_count": X_train.shape[1],
        "numeric_feature_count": len(prep_report.get("numeric", {})),
        "categorical_feature_count": len(prep_report.get("categorical", {})),
        "boolean_feature_count": len(prep_report.get("boolean", {})),
        "datetime_feature_count": len(prep_report.get("datetime", {})),
        "duplicates_removed": clean_info.get("duplicates_removed", 0),
        "ecl_only_cols_dropped": clean_info.get("ecl_only_cols_dropped", []),
        "split_config": {"test_size": test_size, "val_size": val_size, "random_seed": int(random_seed)},
        "split_stats": split_stats,
        "class_distribution_chart": class_distribution_chart,
        "summary_metrics": summary_metrics,
        "x_preview": _serialize_dataframe(X_train, max_rows=5)["preview"],
        "processed_dataset_preview": _serialize_dataframe(processed_df, max_rows=5)["preview"],
        "target_preview": y_train.head(5).replace({pd.NA: None}).tolist(),
        "y_preview": y_train.head(5).replace({pd.NA: None}).tolist(),
        "preprocessing_report": prep_report,
        "preprocessing_strategy_summary": strategy_summary,
        "processed_dataset_csv": processed_df.to_csv(index=False),
        # ── New: the interactive missing-value + transform workflow ──
        "missing_treatment_proposal": _json_safe(missing_treatment_proposal),
        "applied_treatment_map": _json_safe(treatment_map),
        "imputation_strategy": _json_safe(imputation_strategy),
        "recalibrated_columns": recalibrated_cols,
        "dropped_columns": _drop_cols_present,
        "transform_recommendations": _json_safe(prep_report.get("transform_recommendations", {})),
        "applied_transform_choices": _transform_choices,
        "missing_value_limitation_note": MISSING_VALUE_LIMITATION_NOTE,
        "review_missing_threshold": REVIEW_MISSING_THRESHOLD,
        "original_dataset_csv": df.to_csv(index=False),
    }

    try:
        applied_steps = [t.get("treatment") for t in treatment_map.values()]
        persistence.log_event(
            persistence.DEV_PIPELINE_LOG,
            stage="preprocessing",
            payload={
                "target_col": target_col,
                "steps_applied": sorted(set(applied_steps)),
                "dropped_columns": _drop_cols_present,
                "resulting_x_shape": list(X_train.shape),
            },
            full_payload=result,
        )
    except Exception as e:
        print(f"[persistence] failed to log preprocessing: {e}")

    return result


@app.post("/data/drop-impact")
async def drop_impact(
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    target_col: str = Form(...),
    synthetic_samples: Optional[int] = Form(None),
    test_size: float = Form(0.15),
    val_size: float = Form(0.15),
    random_seed: int = Form(42),
    columns: str = Form(...),  # JSON list of column names to analyze
) -> Dict[str, Any]:
    """
    On-demand "impact of dropping this feature" analysis for one or more
    sparse (review_flag) columns — called lazily by the frontend when a
    reviewer expands a column's impact panel, not on every /data/preprocess
    call, since it re-derives the correlation matrix per requested column.

    Re-derives the SAME train split /data/preprocess uses (same file, target,
    split config) so the estimate is computed on training data only — no
    leakage from validation/test. Returns a lightweight, standalone estimate
    (predictive importance via a quick IV, redundancy via correlation with
    other numeric features) — NOT the authoritative IV/WOE from
    /data/feature-engineering, which runs later on cleaned/engineered data.
    See preprocessing_new.estimate_drop_impact() for the full explanation.
    """
    df = await _read_dataframe(file=file, csv_text=csv_text, synthetic_samples=synthetic_samples)
    col_types = detect_column_types(df)
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found")

    test_size = float(test_size)
    val_size = float(val_size)
    if not 0 < test_size < 1:
        raise HTTPException(status_code=400, detail="test_size must be between 0 and 1")
    if not 0 < val_size < 1:
        raise HTTPException(status_code=400, detail="val_size must be between 0 and 1")
    if test_size + val_size >= 1:
        raise HTTPException(status_code=400, detail="test_size + val_size must be less than 1")

    try:
        requested_cols: List[str] = json.loads(columns) if columns else []
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in 'columns': {exc}")
    if not isinstance(requested_cols, list) or not requested_cols:
        raise HTTPException(status_code=400, detail="'columns' must be a non-empty JSON list of column names")

    task_type = detect_task_type(df[target_col])
    X, y, _clean_info = finalize_xy(df, col_types, target_col)
    X_train, _X_val, _X_test, y_train, _y_val, _y_test = split_data(
        X, y, test_size=test_size, val_size=val_size,
        task_type=task_type, random_state=random_seed,
    )

    results: Dict[str, Any] = {}
    for col in requested_cols:
        if col not in X_train.columns:
            results[col] = {"error": f"Column '{col}' not found in the training features."}
            continue
        results[col] = estimate_drop_impact(col, X_train, y_train, col_types)

    return {"drop_impact": _json_safe(results)}


@app.post("/data/feature-engineering")
async def feature_engineering(
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    target_col: str = Form(...),
    synthetic_samples: Optional[int] = Form(None),
    ead_mode: Optional[str] = Form(None),
    ead_col: Optional[str] = Form(None),
    ead_loan_col: Optional[str] = Form(None),
    ead_interest_col: Optional[str] = Form(None),
    ead_years_col: Optional[str] = Form(None),
    ead_term_col: Optional[str] = Form(None),
    ead_years_months: Optional[str] = Form(None),
    ead_term_months: Optional[str] = Form(None),
    model_family: str = Form("linear"),
    transform_choices: Optional[str] = Form(None),
    confirmed_remove_cols: Optional[str] = Form(None),
    woe_pending_drop: Optional[str] = Form(None),
) -> Dict[str, Any]:
    try:
        transform_choices_dict = json.loads(transform_choices) if transform_choices else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"transform_choices must be valid JSON: {exc}")
    try:
        confirmed_remove_cols_list = json.loads(confirmed_remove_cols) if confirmed_remove_cols else None
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"confirmed_remove_cols must be valid JSON: {exc}")
    try:
        woe_pending_drop_list = json.loads(woe_pending_drop) if woe_pending_drop else []
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"woe_pending_drop must be valid JSON: {exc}")

    df = await _read_dataframe(file=file, csv_text=csv_text, synthetic_samples=synthetic_samples)
    col_types = detect_column_types(df)
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found")
    task_type = detect_task_type(df[target_col])
    X, y, _ = finalize_xy(df, col_types, target_col)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.15, val_size=0.15,
        task_type=task_type, random_state=42,
    )
    plan = analyze_for_feature_engineering(
        X_train, y_train, col_types, task_type, model_family, transform_choices_dict,
    )
    if confirmed_remove_cols_list is not None:
        plan["confirmed_remove_cols"] = confirmed_remove_cols_list
    plan["woe_pending_drop"] = woe_pending_drop_list
    X_engineered, fe_summary = apply_feature_engineering(X_train, plan)
    numeric_cols = [c for c in col_types.get("numeric", []) if c in X_train.columns]
    gini_scores = compute_univariate_gini(X_train, y_train, numeric_cols) if task_type == "binary" else {}
    ead_configuration = resolve_ead_configuration(
        df,
        mode=ead_mode or "auto",
        ob_col=ead_col,
        la_col=ead_loan_col,
        ir_col=ead_interest_col,
        ye_col=ead_years_col,
        tm_col=ead_term_col,
        ye_months=str(ead_years_months).lower() in {"true", "1", "yes"},
        tm_months=str(ead_term_months).lower() in {"true", "1", "yes"},
    )
    # resolve_ead_configuration() includes a raw pandas Series under "series"
    # for callers that do further pandas math with it (e.g. the Streamlit
    # app's ECL workflow). This API only ever returns "summary" (mean/median/
    # min/max) to the frontend — features.tsx's ead_configuration type doesn't
    # even declare a "series" field — so drop it here rather than at the
    # source, to avoid changing resolve_ead_configuration's contract for other
    # callers. Left in, it makes FastAPI/Pydantic response serialization
    # crash with "Unable to serialize unknown type: <class 'pandas.Series'>".
    ead_configuration = {k: v for k, v in ead_configuration.items() if k != "series"}
    engineered_feature_names = list(X_engineered.columns.astype(str))
    dropped_features = [col for col in fe_summary.get("removed", []) if col in X_train.columns]
    selected_features = engineered_feature_names
    encoding_summary = {
        "log_transform_columns": plan.get("log_transform_cols", []),
        "interaction_pairs": plan.get("interaction_pairs", []),
        "binning_columns": plan.get("binning_cols", [])[:5],
        "frequency_encoding_columns": plan.get("freq_encoding_cols", []),
        "woe_columns": plan.get("woe_cols", []),
        "low_variance_columns": plan.get("low_variance_cols", []),
        "low_iv_columns": plan.get("low_iv_cols", []),
    }
    feature_engineering_report = {
        "applied_steps": plan.get("applied_steps", []),
        "summary": fe_summary,
        "multicollinearity": plan.get("multicollinearity", {}),
        "gini_scores": gini_scores,
        "ead_configuration": ead_configuration,
    }
    feature_importance_summary = {
        "mi_scores": plan.get("mi_scores", {}),
        "gini_scores": gini_scores,
        "iv_scores": plan.get("iv_scores", {}),
    }

    result = {
        "col_types": col_types,
        "target_col": target_col,
        "task_type": task_type,
        "feature_engineering_plan": plan,
        "feature_engineering_summary": fe_summary,
        "engineered_feature_names": engineered_feature_names,
        "selected_features": selected_features,
        "dropped_features": dropped_features,
        "encoding_summary": encoding_summary,
        "feature_engineering_report": feature_engineering_report,
        "feature_importance_summary": feature_importance_summary,
        "x_engineered_shape": list(X_engineered.shape),
        "x_engineered_preview": _serialize_dataframe(X_engineered, max_rows=5)["preview"],
        "final_engineered_dataset_preview": _serialize_dataframe(X_engineered, max_rows=5)["preview"],
        "x_engineered_csv": X_engineered.to_csv(index=False),
        "gini_scores": gini_scores,
        "ead_configuration": ead_configuration,
        "available_numeric_columns": [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
        "interaction_features": plan.get("interaction_features", []),
        "interaction_scores": plan.get("interaction_scores", {}),
        "feature_scores": plan.get("feature_scores", {}),
        "transform_recommendations": plan.get("transform_recommendations", {}),
    }

    try:
        persistence.log_event(
            persistence.DEV_PIPELINE_LOG,
            stage="feature_engineering",
            payload={
                "target_col": target_col,
                "features_added": plan.get("applied_steps", []),
                "features_removed": dropped_features,
                "final_feature_count": len(engineered_feature_names),
            },
            full_payload=result,
        )
    except Exception as e:
        print(f"[persistence] failed to log feature_engineering: {e}")

    return result


@app.post("/data/macro/date-columns")
async def macro_date_columns(
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    synthetic_samples: Optional[int] = Form(None),
) -> Dict[str, Any]:
    """
    List the columns in the dataset usable for point-in-time FRED macro
    alignment (real dates only, origination-pattern matches first), plus the
    best-effort auto-detected default. Mirrors the old Streamlit app's date
    dropdown (see fred_client.list_date_columns_for_macro/detect_macro_date_col) —
    default/charge-off-dated columns are never auto-selected, even as a fallback.
    """
    df = await _read_dataframe(file=file, csv_text=csv_text, synthetic_samples=synthetic_samples)
    candidates = fred_client.list_date_columns_for_macro(df)
    return {
        "candidates": [{"column": c, "is_preferred": pref} for c, pref in candidates],
        "default_date_col": fred_client.detect_macro_date_col(df),
    }


@app.post("/data/macro/fetch")
async def macro_fetch(
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    synthetic_samples: Optional[int] = Form(None),
    date_col: Optional[str] = Form(None),
    fred_api_key: str = Form(...),
) -> Dict[str, Any]:
    """
    Fetch FRED macro features (GDP, unemployment, Fed funds rate) aligned to
    the calendar month of `date_col` and attach them to the dataset — same
    semantics as the old app's "Fetch FRED macro features" button. Returns the
    augmented dataset as CSV text so the frontend can carry it forward as
    `csv_text` into /data/preprocess and /data/feature-engineering, plus the
    new macro column names for display.

    The FRED API key is supplied by the caller on every request (entered in
    the UI) — it is never read from an environment variable or stored
    anywhere on the server between requests.

    `date_col` is optional: when omitted, the loan-origination date column is
    auto-detected via fred_client.detect_macro_date_col (same heuristic used
    to populate /data/macro/date-columns's default_date_col), so the caller
    doesn't need to resolve it up front.
    """
    api_key = (fred_api_key or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="A FRED API key is required.")

    df = await _read_dataframe(file=file, csv_text=csv_text, synthetic_samples=synthetic_samples)

    resolved_date_col = date_col or fred_client.detect_macro_date_col(df)
    if not resolved_date_col:
        raise HTTPException(
            status_code=400,
            detail="No origination/observation date column could be auto-detected in this dataset.",
        )
    if resolved_date_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Date column '{resolved_date_col}' not found")

    try:
        client = fred_client.FREDClient(api_key=api_key, cache_dir=".fred_cache")
        df_with_macro, macro_cols = fred_client.attach_macro_features(
            df, fred_client=client, date_col=resolved_date_col,
        )
    except fred_client.FREDError as exc:
        raise HTTPException(status_code=502, detail=f"FRED fetch failed: {exc}")

    if not macro_cols:
        raise HTTPException(status_code=400, detail="No macro columns were attached — check the date column.")

    return {
        "macro_columns": macro_cols,
        "date_col_used": resolved_date_col,
        "csv_with_macro": df_with_macro.to_csv(index=False),
        "preview": _serialize_dataframe(df_with_macro, max_rows=5)["preview"],
        "shape": list(df_with_macro.shape),
    }


@app.get("/models/list")
async def list_classification_models() -> Dict[str, Any]:
    """Returns the available classification model keys (e.g. for the Replication
    stage's model selector), independent of any dataset-based recommendation."""
    return {"models": list(CLASSIFICATION_MODELS.keys())}


@app.post("/models/recommend")
async def recommend_models_endpoint(
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    target_col: Optional[str] = Form(None),
    n_samples: Optional[int] = Form(None),
    n_features: Optional[int] = Form(None),
    class_imbalance_ratio: Optional[float] = Form(None),
    task_type: Optional[str] = Form(None),
    synthetic_samples: Optional[int] = Form(None),
    explainability_priority: float = Form(0.5),
) -> Dict[str, Any]:
    fe_summary = None
    if file is not None or csv_text or synthetic_samples:
        df = await _read_dataframe(file=file, csv_text=csv_text, synthetic_samples=synthetic_samples)
        if target_col is None or target_col not in df.columns:
            raise HTTPException(status_code=400, detail="target_col is required for dataset-based recommendations")
        col_types = detect_column_types(df)
        task_type = detect_task_type(df[target_col])

        # Prepare data (drop ids, dedupe) but DO NOT fit preprocessors here
        X, y, preproc, prep_report, feature_names = prepare_data(df, col_types, target_col)

        # Split the data exactly like the original Streamlit flow so recommendations
        # are based on the TRAIN set after FE (parity requirement).
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=0.15, val_size=0.15, task_type=task_type, random_state=42
        )

        # Learn FE plan on TRAIN and apply to TRAIN to get the actual feature set
        try:
            plan = analyze_for_feature_engineering(X_train, y_train, col_types, task_type)
            X_train_engineered, fe_summary = apply_feature_engineering(X_train, plan)
        except Exception:
            # Fallback: no FE applied
            X_train_engineered = X_train.copy()
            fe_summary = None

        # Dataset characteristics feed the weighted recommendation engine —
        # measured on the actual (post-FE) training data, not just its shape.
        characteristics = compute_dataset_characteristics(
            X_train_engineered, y_train, col_types, task_type,
            explainability_priority=explainability_priority,
        )
        n_samples = characteristics["_raw"]["n_samples"]
        n_features = characteristics["_raw"]["n_features"]
        class_imbalance_ratio = characteristics["_raw"]["imbalance_ratio"]
    else:
        if n_samples is None or n_features is None or task_type is None:
            raise HTTPException(status_code=400, detail="Either dataset or numeric summary values are required")
        if class_imbalance_ratio is None:
            class_imbalance_ratio = 1.0
        # No dataset in hand — fall back to a summary-only characteristic set
        # (missingness/categorical share/multicollinearity/non-linearity default
        # to neutral since we can't measure them without the data).
        characteristics = characteristics_from_summary(
            n_samples=n_samples, n_features=n_features, class_imbalance_ratio=class_imbalance_ratio,
            explainability_priority=explainability_priority,
        )

    result = recommend_model(characteristics, task_type=task_type)

    return {
        "recommended_model": result["recommended_model"],
        "all_models": result["all_models"],
        "characteristics": {k: v for k, v in characteristics.items() if k != "_raw"} | {"raw": characteristics.get("_raw", {})},
        "training": {
            "train_n": int(n_samples),
            "train_features": int(n_features),
            "imbalance_ratio": float(class_imbalance_ratio),
        },
        "task_type": task_type,
        "feature_engineering_summary": fe_summary,
    }

        
@app.post("/models/train")
async def train_model_endpoint(
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    target_col: str = Form(...),
    model_name: str = Form(...),
    test_size: float = Form(0.15),
    val_size: float = Form(0.15),
    random_seed: int = Form(42),
    use_cv: bool = Form(False),
    cv_folds: int = Form(5),
    use_hyperopt: bool = Form(False),
    scale_pos_weight: float = Form(1.0),
    manual_params: Optional[str] = Form(None),
    use_feature_engineering: bool = Form(False),
    synthetic_samples: Optional[int] = Form(None),
    use_oot: bool = Form(False),
    date_col: Optional[str] = Form(None),
    threshold: Optional[float] = Form(None),
) -> Dict[str, Any]:
    df = await _read_dataframe(file=file, csv_text=csv_text, synthetic_samples=synthetic_samples)
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found")
    col_types = detect_column_types(df)
    task_type = detect_task_type(df[target_col])
    X, y, _ = finalize_xy(df, col_types, target_col)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=test_size, val_size=val_size,
        task_type=task_type, random_state=random_seed,
    )

    # ── Origination/observation date for Out-of-Time (OOT) validation ──
    # Use the reviewer-selected date column if provided and valid, otherwise
    # fall back to the first auto-detected datetime column (same detection
    # `detect_column_types` already exposes via the data-profile endpoint).
    origination_date_col = date_col if (date_col and date_col in df.columns) else None
    if origination_date_col is None:
        datetime_candidates = col_types.get("datetime", [])
        if datetime_candidates:
            origination_date_col = datetime_candidates[0]

    dates_train = None
    if origination_date_col and origination_date_col in df.columns:
        try:
            dates_train = df.loc[X_train.index, origination_date_col]
        except Exception:
            dates_train = None
    dates_test = None
    if origination_date_col and origination_date_col in df.columns:
        try:
            dates_test = df.loc[X_test.index, origination_date_col]
        except Exception:
            dates_test = None
    prep_report = build_preprocessing_report(X_train.assign(**{target_col: y_train}), col_types, target_col)
    fe_summary = None
    plan = None
    if use_feature_engineering:
        plan = analyze_for_feature_engineering(X_train, y_train, col_types, task_type)
        X_train, fe_summary = apply_feature_engineering(X_train, plan)
        X_val, _ = apply_feature_engineering(X_val, plan)
        X_test, _ = apply_feature_engineering(X_test, plan)
    if manual_params:
        try:
            manual_params_dict = json.loads(manual_params)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"manual_params must be valid JSON: {exc}")
    else:
        manual_params_dict = None
    if manual_params_dict is not None:
        model = get_model_instance(model_name, task_type)
        valid_keys = set(model.get_params().keys())
        manual_params_dict = {k: v for k, v in manual_params_dict.items() if k in valid_keys}
        model = type(model)(**manual_params_dict)
    else:
        model = get_model_instance(
            model_name,
            task_type,
            scale_pos_weight=scale_pos_weight if model_name == "XGBoost" else None,
        )
    param_grid = None
    if use_hyperopt:
        param_grid = get_hyperparameter_grid(model_name, task_type)
    pipeline, training_info, real_feature_names = train_model(
        X_train, y_train,
        col_types=col_types,
        target_col=target_col,
        prep_report=prep_report,
        model=model,
        use_cv=use_cv,
        cv_folds=cv_folds,
        use_hyperopt=use_hyperopt,
        param_grid=param_grid,
        task_type=task_type,
        model_name=model_name,
        dates_train=dates_train,
        use_oot=use_oot,
    )
    split_stats = compute_split_stats(X_train, X_val, X_test, y_train, y_val, y_test)

    # Evaluate on hold-out test set to mirror the original Streamlit training workflow.
    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception:
            y_proba = None

    metrics, evaluation_data = _build_evaluation_data(
        y_test.values, y_pred, y_proba, task_type, threshold=threshold,
        dates=dates_test, date_columns=[origination_date_col] if origination_date_col else [],
    )

    result = {
        "task_type": task_type,
        "model_name": model_name,
        "real_feature_names": real_feature_names,
        "training_config": {
            "model_name": model_name,
            "test_size": test_size,
            "val_size": val_size,
            "random_seed": random_seed,
            "use_cv": use_cv,
            "cv_folds": cv_folds,
            "use_hyperopt": use_hyperopt,
            "scale_pos_weight": scale_pos_weight,
            "use_feature_engineering": use_feature_engineering,
            "manual_params": manual_params_dict or {},
            "use_oot": use_oot,
            "date_col": origination_date_col,
        },
        "training_info": training_info,
        "split_stats": split_stats,
        "feature_engineering_summary": fe_summary,
        "low_iv_columns": plan.get("low_iv_cols", []) if plan else [],
        "low_variance_columns": plan.get("low_variance_cols", []) if plan else [],
        "dropped_high_corr_pairs": plan.get("drop_high_corr_pairs", []) if plan else [],
        "applied_steps": plan.get("applied_steps", []) if plan else [],
        "evaluation_metrics": metrics,
        "evaluation_data": evaluation_data,
        "model_artifact": _to_base64(pipeline),
    }

    try:
        persistence.log_event(
            persistence.DEV_PIPELINE_LOG,
            stage="training",
            payload={
                "model_name": model_name,
                "hyperparameters": manual_params_dict or {},
                "metrics": metrics,
            },
            full_payload=result,
        )
    except Exception as e:
        print(f"[persistence] failed to log training: {e}")

    return result


def _lighten_for_comparison(model, model_name: str, max_estimators: int = 100):
    """
    Reduce ensemble size for comparison-only runs so each candidate trains
    fast. Comparison is meant to give a directional read on which model is
    worth committing to — not a final number — so trading a bit of accuracy
    for speed here is the right call. Full-fidelity numbers come from the
    dedicated /models/train run once the user picks a model.
    """
    if hasattr(model, "n_estimators") and getattr(model, "n_estimators", None):
        try:
            if model.n_estimators > max_estimators:
                model.set_params(n_estimators=max_estimators)
        except Exception:
            pass
    return model


@app.post("/models/compare")
async def compare_models_endpoint(
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    target_col: str = Form(...),
    model_names: str = Form(...),  # JSON-encoded list of model names
    test_size: float = Form(0.15),
    val_size: float = Form(0.15),
    random_seed: int = Form(42),
    use_feature_engineering: bool = Form(False),
    synthetic_samples: Optional[int] = Form(None),
) -> Dict[str, Any]:
    """
    Lightweight, fast comparison across candidate models on the SAME split.

    Deliberately skips everything that makes /models/train slow when run
    N times: no cross-validation, no hyperparameter search, no OOT, no
    ROC/PR/threshold/gain-chart curves, and no base64 model-artifact
    serialization. Just fit -> predict -> summary metrics per model, so a
    reviewer can quickly see which model is worth committing to before
    running the full, config-rich /models/train on that one model.
    """
    try:
        names = json.loads(model_names)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"model_names must be a JSON array: {exc}")
    if not isinstance(names, list) or not names:
        raise HTTPException(status_code=400, detail="model_names must be a non-empty list")

    df = await _read_dataframe(file=file, csv_text=csv_text, synthetic_samples=synthetic_samples)
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found")

    col_types = detect_column_types(df)
    task_type = detect_task_type(df[target_col])
    X, y, _ = finalize_xy(df, col_types, target_col)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=test_size, val_size=val_size,
        task_type=task_type, random_state=random_seed,
    )

    prep_report = build_preprocessing_report(X_train.assign(**{target_col: y_train}), col_types, target_col)
    fe_summary = None
    if use_feature_engineering:
        plan = analyze_for_feature_engineering(X_train, y_train, col_types, task_type)
        X_train, fe_summary = apply_feature_engineering(X_train, plan)
        X_test, _ = apply_feature_engineering(X_test, plan)

    results = []
    for name in names:
        try:
            model = get_model_instance(name, task_type)
            model = _lighten_for_comparison(model, name)
            pipeline, training_info, _ = train_model(
                X_train, y_train,
                col_types=col_types,
                target_col=target_col,
                prep_report=prep_report,
                model=model,
                use_cv=False,
                use_hyperopt=False,
                task_type=task_type,
                model_name=name,
                use_oot=False,
            )
            y_pred = pipeline.predict(X_test)
            y_proba = None
            if hasattr(pipeline, "predict_proba"):
                try:
                    y_proba = pipeline.predict_proba(X_test)
                except Exception:
                    y_proba = None

            if task_type == "binary":
                metrics = compute_binary_metrics(y_test.values, y_pred, y_proba, threshold=None)
            else:
                metrics = compute_regression_metrics(y_test.values, y_pred)

            results.append({
                "model_name": name,
                **metrics,
                "training_time_s": training_info.get("training_time_s"),
            })
        except Exception as e:
            results.append({"model_name": name, "error": str(e)})

    return {
        "task_type": task_type,
        "comparison": results,
        "feature_engineering_summary": fe_summary,
    }


@app.post("/models/evaluate")
async def evaluate_model(
    model_artifact: UploadFile = File(...),
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    target_col: str = Form(...),
    threshold: Optional[float] = Form(None),
    synthetic_samples: Optional[int] = Form(None),
    date_col: Optional[str] = Form(None),
) -> Dict[str, Any]:
    df = await _read_dataframe(file=file, csv_text=csv_text, synthetic_samples=synthetic_samples)
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found")
    pipeline = await _load_model_artifact(model_artifact)
    y_true = df[target_col].values
    X_eval = df.drop(columns=[target_col], errors='ignore')
    y_pred = pipeline.predict(X_eval)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_eval)
        except Exception:
            y_proba = None
    task_type = detect_task_type(df[target_col])
    # Same origination-date resolution as /models/train: reviewer-selected
    # column if valid, else the first auto-detected datetime column — used
    # to power the Temporal tab's actual-vs-predicted-over-time analysis.
    col_types = detect_column_types(df)
    origination_date_col = date_col if (date_col and date_col in df.columns) else None
    if origination_date_col is None:
        datetime_candidates = col_types.get("datetime", [])
        if datetime_candidates:
            origination_date_col = datetime_candidates[0]
    dates_eval = df[origination_date_col] if origination_date_col else None

    metrics, evaluation_data = _build_evaluation_data(
        y_true, y_pred, y_proba, task_type, threshold=threshold,
        dates=dates_eval, date_columns=[origination_date_col] if origination_date_col else [],
    )

    # Same response shape as /models/train so callers can treat both
    # endpoints' evaluation payload identically.
    result = {
        "evaluation_metrics": metrics,
        "evaluation_data": evaluation_data,
    }

    try:
        persistence.log_event(
            persistence.DEV_PIPELINE_LOG,
            stage="evaluation",
            payload={"evaluation_metrics": metrics},
            full_payload=result,
        )
    except Exception as e:
        print(f"[persistence] failed to log evaluation: {e}")

    return result


@app.post("/models/explain")
async def explain_model(
    model_artifact: UploadFile = File(...),
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    target_col: Optional[str] = Form(None),
    max_shap_samples: int = Form(100),
    sample_idx: int = Form(0),
    synthetic_samples: Optional[int] = Form(None),
    # ── Same fields /models/train already returns in training_config — needed
    #    to deterministically re-derive the EXACT engineered test split the
    #    model was fitted on. Without this, X came straight from the raw
    #    upload, whose columns don't match what the fitted preprocessor
    #    expects whenever use_feature_engineering was True at train time
    #    (bin/WOE/interaction columns are missing) — this is what caused
    #    "SHAP computation failed" for every model type, not just non-tree
    #    ones. Mirrors the old Streamlit app's use of X_test_engineered. ──
    use_feature_engineering: bool = Form(False),
    test_size: float = Form(0.15),
    val_size: float = Form(0.15),
    random_seed: int = Form(42),
    # ── Summary tab: same inputs generate_model_summary() takes in the old app ──
    metrics: Optional[str] = Form(None),
    task_type: Optional[str] = Form(None),
    # Feature importance is cheap and shown immediately in the old UI; SHAP
    # (especially KernelExplainer, nsamples=100) is not, and is only computed
    # on an explicit "Compute SHAP Values" click there. Default True keeps
    # existing callers working; the frontend passes False for the initial
    # Feature-Importance-only load.
    compute_shap: bool = Form(True),
) -> Dict[str, Any]:
    df = await _read_dataframe(file=file, csv_text=csv_text, synthetic_samples=synthetic_samples)
    if target_col is not None and target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found")
    pipeline = await _load_model_artifact(model_artifact)

    if target_col is not None:
        col_types = detect_column_types(df)
        resolved_task_type = task_type or detect_task_type(df[target_col])
        X, y, _ = finalize_xy(df, col_types, target_col)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=test_size, val_size=val_size,
            task_type=resolved_task_type, random_state=random_seed,
        )
        if use_feature_engineering:
            plan = analyze_for_feature_engineering(X_train, y_train, col_types, resolved_task_type)
            X_test, _ = apply_feature_engineering(X_test, plan)
        X_explain = X_test
    else:
        X_explain = df

    importance_df = extract_feature_importance(pipeline)
    importance = []
    if importance_df is not None:
        importance = importance_df.to_dict(orient="records")

    shap_info: Dict[str, Any] = {"shap_available": False}
    shap_result = compute_shap_values(pipeline, X_explain, max_samples=max_shap_samples) if compute_shap else None
    if shap_result is not None:
        explainer, shap_values, X_df, names = shap_result
        mean_abs = list(
            pd.DataFrame({"Feature": X_df.columns, "MeanAbsSHAP": np.abs(shap_values).mean(axis=0)})
            .sort_values("MeanAbsSHAP", ascending=False)
            .head(20)
            .to_dict(orient="records")
        )
        reasoning = None
        sample_shap: List[Dict[str, Any]] = []
        sample_features: Dict[str, Any] = {}
        if 0 <= sample_idx < len(X_df):
            try:
                model_proba = pipeline.predict_proba(X_explain) if hasattr(pipeline, "predict_proba") else np.zeros((len(X_df), 2))
            except Exception:
                model_proba = np.zeros((len(X_df), 2))
            reasoning = generate_prediction_reasoning(shap_values, X_df, model_proba, sample_idx, threshold=0.5)
            shap_row = shap_values[sample_idx]
            feat_row = X_df.iloc[sample_idx]
            # Top 12 by |SHAP| magnitude, then re-sorted ascending by signed
            # value — the explainability chart renders this list top-to-bottom
            # as-is (no further sort/slice on the frontend), so ordering here
            # is what determines the bar chart's reading order.
            sample_shap = sorted(
                sorted(
                    (
                        {
                            "Feature": feat,
                            "SHAP": _json_safe_scalar(shap_row[i]),
                            "Value": _json_safe_scalar(feat_row.iloc[i]),
                        }
                        for i, feat in enumerate(X_df.columns)
                    ),
                    key=lambda r: abs(r["SHAP"]) if r["SHAP"] is not None else 0.0,
                    reverse=True,
                )[:12],
                key=lambda r: r["SHAP"] if r["SHAP"] is not None else 0.0,
            )
            sample_features = {feat: _json_safe_scalar(feat_row.iloc[i]) for i, feat in enumerate(X_df.columns)}
        shap_info = {
            "shap_available": True,
            "shap_mean_abs": mean_abs,
            "sample_idx": sample_idx,
            "sample_reasoning": reasoning,
            "sample_shap": sample_shap,
            "sample_features": sample_features,
        }

    summary_text = None
    if metrics:
        try:
            metrics_dict = json.loads(metrics)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"metrics must be valid JSON: {exc}")
        summary_text = generate_model_summary(metrics_dict, importance_df, task_type or "binary")

    result = {
        "feature_importance": importance,
        "shap": shap_info,
        "summary": summary_text,
    }

    try:
        persistence.log_event(
            persistence.DEV_PIPELINE_LOG,
            stage="explainability",
            payload={
                "method": "shap" if shap_info.get("shap_available") else "feature_importance",
                "compute_shap": compute_shap,
            },
            full_payload=result,
        )
    except Exception as e:
        print(f"[persistence] failed to log explainability: {e}")

    return result


def _run_benchmark_comparison(
    replication_result: Dict[str, Any],
    challenger_model_name: str,
    target_col: str,
    y_true: Optional[np.ndarray],
    y_proba: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Fits the selected challenger model and compares it against the
    already-fitted champion.

    Reuses the exact same lightweight fit path as /models/compare —
    get_model_instance() + _lighten_for_comparison() + train_model() with
    no CV/hyperopt/OOT, through the same rebuild_preprocessor_for()
    pipeline the champion itself was fit with — rather than a separate,
    cruder hand-rolled imputer+scaler path limited to three hardcoded
    "industry standard" models. This makes champion-vs-challenger an
    apples-to-apples comparison, and lets any model the reviewer picked
    from the challenger comparison step serve as the benchmark.
    """
    X_train = replication_result.get("X_train")
    X_test = replication_result.get("X_test")
    y_train = replication_result.get("y_train")
    col_types = replication_result.get("col_types")
    prep_report = replication_result.get("prep_report_fe") or replication_result.get("prep_report")
    task_type = replication_result.get("task_type", "binary")

    if (
        X_train is None or X_test is None or y_train is None
        or y_true is None or y_proba is None
        or col_types is None or prep_report is None
    ):
        return {
            "model_name": challenger_model_name,
            "status": "SKIP",
            "metrics": {},
            "comparison": {},
        }

    try:
        model = get_model_instance(challenger_model_name, task_type)
        model = _lighten_for_comparison(model, challenger_model_name)
        pipeline, _training_info, _ = train_model(
            X_train, y_train,
            col_types=col_types,
            target_col=target_col,
            prep_report=prep_report,
            model=model,
            use_cv=False,
            use_hyperopt=False,
            task_type=task_type,
            model_name=challenger_model_name,
            use_oot=False,
        )
        if not hasattr(pipeline, "predict_proba"):
            return {
                "model_name": challenger_model_name,
                "status": "SKIP",
                "metrics": {},
                "comparison": {},
            }
        bm_proba_2d = pipeline.predict_proba(X_test)
        bm_pred = pipeline.predict(X_test)
        bm_metrics = compute_binary_metrics(np.asarray(y_true).astype(int), bm_pred, bm_proba_2d, threshold=None)
        # compute_binary_metrics() never returns "gini" — it's not one of the
        # keys it computes. The champion side (val_replication_core._fit_core)
        # adds it separately via _gini()/_ks() after calling the same
        # function; do the same here so the challenger isn't silently missing
        # it in the champion-vs-challenger comparison.
        bm_scores = eval_engine._to_scores(bm_proba_2d)
        bm_metrics["gini"] = _gini(np.asarray(y_true).astype(int), bm_scores)
        bm_metrics["ks"] = _ks(np.asarray(y_true).astype(int), bm_scores)
    except Exception as e:
        return {
            "model_name": challenger_model_name,
            "status": "ERROR",
            "error": str(e),
            "metrics": {},
            "comparison": {},
        }

    bm_auc = bm_metrics.get("roc_auc")
    bm_gini = bm_metrics.get("gini")
    bm_recall = bm_metrics.get("recall")

    champion_metrics = replication_result.get("metrics", {})
    champion_auc = champion_metrics.get("roc_auc")
    champion_recall = champion_metrics.get("recall")
    champion_gini = champion_metrics.get("gini")

    return {
        "model_name": challenger_model_name,
        "status": "OK",
        "metrics": {
            "roc_auc": bm_auc,
            "gini": bm_gini,
            "recall": bm_recall,
            # Exposed so a low recall is explainable rather than a black box:
            # compute_binary_metrics() auto-selects the F1-maximizing
            # threshold (see evaluate_new.select_best_threshold) rather than
            # a naive 0.5 cutoff, so "recall is low" usually means that
            # threshold landed high to protect precision on this dataset's
            # class balance — these fields show exactly where and why.
            "precision": bm_metrics.get("precision"),
            "f1": bm_metrics.get("f1"),
            "accuracy": bm_metrics.get("accuracy"),
            "ks": bm_metrics.get("ks"),
            "threshold_used": bm_metrics.get("threshold_used"),
        },
        "comparison": {
            "champion_vs_challenger": {
                "roc_auc": {
                    "champion": champion_auc,
                    "challenger": bm_auc,
                },
                "gini": {
                    "champion": champion_gini,
                    "challenger": bm_gini,
                },
                "recall": {
                    "champion": champion_recall,
                    "challenger": bm_recall,
                },
            }
        },
    }


def _build_performance_report(
    df: Optional[pd.DataFrame],
    replication_result: Dict[str, Any],
    intake_payload: Dict[str, Any],
    mdd_text: str,
) -> Dict[str, Any]:
    """Full performance report (metrics, ROC/PR curves, confusion matrix,
    calibration, score distribution, train/test AUC gap, metric checks,
    compliance findings) built from an already-fitted replication result.

    This used to live only in /validation/performance. Stage 3 (Model
    Replication, /validation/replication) and the old Stage 4 "Performance"
    tab are now one combined page, so both endpoints call this — replication
    for the combined page's charts, performance for the champion metrics +
    ROC curve its benchmarking view still needs. Factored out rather than
    duplicated so the two stay in sync.
    """
    y_true = replication_result.get("y_test")
    y_proba = replication_result.get("y_proba")
    y_true_arr = np.asarray(y_true).astype(int) if y_true is not None else None
    y_proba_arr = np.asarray(y_proba).astype(float) if y_proba is not None else None

    metrics = dict(replication_result.get("metrics") or {})
    metrics = {k: _json_safe_scalar(v) for k, v in metrics.items()}

    roc_curve = []
    if y_true_arr is not None and y_proba_arr is not None:
        roc_curve = compute_roc_curve(y_true_arr, y_proba_arr)

    pr_curve = []
    if y_true_arr is not None and y_proba_arr is not None:
        pr_curve = compute_pr_curve(y_true_arr, y_proba_arr)

    threshold_analysis = []
    if y_true_arr is not None and y_proba_arr is not None:
        threshold_analysis = compute_threshold_analysis(y_true_arr, y_proba_arr)

    score_distribution = []
    if y_true_arr is not None and y_proba_arr is not None:
        score_distribution = compute_score_distribution(y_true_arr, y_proba_arr, n_bins=40)

    # Auto-calibrate the operating threshold the same way /models/evaluate
    # does (select_best_threshold, maximizes F1) instead of a fixed 0.5
    # cut-off, so the confusion matrix and its displayed threshold always
    # line up with the threshold actually used to build it.
    threshold_selection = None
    cm = []
    if y_true_arr is not None and y_proba_arr is not None:
        threshold_selection = eval_engine.select_best_threshold(y_true_arr, y_proba_arr, metric="f1")
        resolved_threshold = threshold_selection["threshold"]
        y_pred = (y_proba_arr >= resolved_threshold).astype(int)
        cm_matrix = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).tolist()
        cm = {"labels": [0, 1], "matrix": cm_matrix}

    calibration_points = []
    if y_true_arr is not None and y_proba_arr is not None:
        cal_df = pd.DataFrame({"actual": y_true_arr, "predicted": y_proba_arr})
        cal_df["bin"] = pd.qcut(cal_df["predicted"], q=10, duplicates="drop")
        grouped = (
            cal_df.groupby("bin", observed=True)
            .agg(actual_rate=("actual", "mean"), predicted_rate=("predicted", "mean"), n=("actual", "count"))
            .reset_index()
        )
        calibration_points = [
            {
                "bin": str(row["bin"]),
                "actual_rate": float(row["actual_rate"]),
                "predicted_rate": float(row["predicted_rate"]),
                "n": int(row["n"]),
            }
            for _, row in grouped.iterrows()
        ]

    cv_mean_auc = replication_result.get("cv_mean_auc")
    test_auc = metrics.get("roc_auc")
    train_test_gap = None
    train_test_gap_status = "WARN"
    if cv_mean_auc is not None and test_auc is not None:
        train_test_gap = round(abs(float(cv_mean_auc) - float(test_auc)), 4)
        train_test_gap_status = "PASS" if train_test_gap <= 0.10 else "FAIL"
    elif test_auc is not None:
        train_test_gap = round(float(test_auc) if np.isfinite(float(test_auc)) else 0.0, 4)
        train_test_gap_status = "WARN"

    validation_report = run_validation_agent2(df, intake_payload or {}, mdd_text)
    stage5_findings = [
        finding for finding in validation_report.get("all_findings", [])
        if finding.get("stage") == "Stage 5: Performance Validation"
    ]

    performance_checks = []
    if metrics.get("roc_auc") is not None:
        performance_checks.append({
            "check": "ROC-AUC",
            "threshold": ">= 0.70",
            "status": "PASS" if float(metrics["roc_auc"]) >= 0.70 else "FAIL",
            "value": metrics["roc_auc"],
        })
    if metrics.get("gini") is not None:
        performance_checks.append({
            "check": "Gini",
            "threshold": ">= 0.40",
            "status": "PASS" if float(metrics["gini"]) >= 0.40 else "FAIL",
            "value": metrics["gini"],
        })
    if metrics.get("recall") is not None:
        performance_checks.append({
            "check": "Recall",
            "threshold": ">= 0.60",
            "status": "PASS" if float(metrics["recall"]) >= 0.60 else "FAIL",
            "value": metrics["recall"],
        })
    if metrics.get("precision") is not None:
        performance_checks.append({
            "check": "Precision",
            "threshold": ">= 0.50",
            "status": "PASS" if float(metrics["precision"]) >= 0.50 else "FAIL",
            "value": metrics["precision"],
        })
    if metrics.get("f1") is not None:
        performance_checks.append({
            "check": "F1 Score",
            "threshold": ">= 0.55",
            "status": "PASS" if float(metrics["f1"]) >= 0.55 else "FAIL",
            "value": metrics["f1"],
        })
    if metrics.get("brier_score") is not None:
        performance_checks.append({
            "check": "Brier Score",
            "threshold": "< 0.25",
            "status": "PASS" if float(metrics["brier_score"]) < 0.25 else "FAIL",
            "value": metrics["brier_score"],
        })
    if metrics.get("pr_auc") is not None:
        performance_checks.append({
            "check": "PR-AUC",
            "threshold": ">= 0.30",
            "status": "PASS" if float(metrics["pr_auc"]) >= 0.30 else "FAIL",
            "value": metrics["pr_auc"],
        })
    if metrics.get("ks") is not None:
        performance_checks.append({
            "check": "KS Statistic",
            "threshold": ">= 0.30",
            "status": "PASS" if float(metrics["ks"]) >= 0.30 else "FAIL",
            "value": metrics["ks"],
        })

    return {
        "metrics": metrics,
        "roc_curve": {"points": roc_curve, "auc": metrics.get("roc_auc")},
        "pr_curve": {"points": pr_curve, "average_precision": metrics.get("pr_auc")},
        "confusion_matrix": cm,
        "threshold_selection": threshold_selection,
        "score_distribution": {"bins": score_distribution},
        "calibration_chart": {"points": calibration_points},
        "train_test_auc_gap": {
            "cv_mean_auc": cv_mean_auc,
            "test_auc": test_auc,
            "gap": train_test_gap,
            "status": train_test_gap_status,
        },
        "metric_checks": performance_checks,
        "compliance_findings": stage5_findings,
        "threshold_analysis": threshold_analysis,
    }


@app.post("/validation/performance")
async def validation_performance(
    model_name: str = Form(...),
    target_col: str = Form(...),
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    challenger_model_name: str = Form("Logistic Regression"),
    seeds: Optional[str] = Form("42,43,44,45,46"),
    test_size: float = Form(0.15),
    val_size: float = Form(0.15),
    random_seed: int = Form(42),
    cv_folds: int = Form(5),
    mdd_file: Optional[UploadFile] = File(None),
    reported_json: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Stage 4 — Benchmarking only.

    The full performance report (ROC/PR curves, confusion matrix,
    calibration, score distribution, metric checks, compliance findings)
    moved to /validation/replication, since Stage 3 (Model Replication) and
    the old Stage 4 "Performance" tab are now a single combined page. This
    endpoint still fits the champion model (same replication core) because
    the champion-vs-challenger comparison needs a freshly-fit champion to
    compare against the selected challenger model, and the ROC overlay
    chart needs the champion's ROC points.

    `mdd_file` / `reported_json` (optional, combinable — same precedence as
    /validation/replication: MDD-extracted values win per-key over
    `reported_json`) restore what this endpoint used to drop when Stage 3/4
    were split: wherever the MDD documents a metric for the champion (e.g.
    recall), that documented value is shown here instead of the freshly
    re-fit replica's recomputed number, since the champion side of this
    comparison is meant to reflect what was actually reported for the
    champion, not a brand-new fit's incidental test-split performance.
    Metrics the MDD doesn't mention keep the replica's computed value.

    `challenger_model_name` is a real registry model name (see
    GET /models/list) — typically whichever model the reviewer picked
    after comparing several candidates via POST /models/compare, the same
    lightweight comparison used in the model development pipeline's Model
    Selection step. It's fit here via the same path (get_model_instance +
    train_model, no CV/hyperopt/OOT) so the comparison is apples-to-apples
    with the champion.
    """
    df = await _read_dataframe(file=file, csv_text=csv_text)
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found")

    try:
        seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    except Exception:
        seed_list = [random_seed]

    mdd_reported: Dict[str, Any] = {}
    if reported_json:
        try:
            parsed = json.loads(reported_json)
            if isinstance(parsed, dict):
                mdd_reported.update({k: v for k, v in parsed.items() if v is not None})
        except Exception:
            pass
    if mdd_file is not None:
        try:
            # Same sync-file-object bridge /validation/replication uses —
            # parse_mdd_file expects .name/.seek()/.read(), FastAPI's
            # UploadFile is async, so reach through to the underlying
            # SpooledTemporaryFile.
            mdd_file.file.seek(0)
            mdd_underlying = mdd_file.file
            mdd_underlying.name = mdd_file.filename or ""
            mdd_text = parse_mdd_file(mdd_underlying)
            mdd_extracted = extract_metrics_from_mdd(mdd_text)
            mdd_reported.update({k: v for k, v in mdd_extracted.items() if v is not None})
        except Exception:
            pass

    replication_result = run_replication(
        df=df,
        target_col=target_col,
        model_name=model_name,
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed,
        cv_folds=cv_folds,
        reported={},
        seeds=seed_list,
    )

    # Champion metrics shown/benchmarked here prefer the MDD-reported value
    # over the freshly re-fit replica's number, per-key, wherever the MDD
    # provided one — mutated in place so both the top-level `metrics` below
    # AND _run_benchmark_comparison's champion_vs_challenger figures
    # (which reads replication_result["metrics"] directly) stay consistent.
    if mdd_reported:
        champion_metrics_raw = dict(replication_result.get("metrics") or {})
        for key, value in mdd_reported.items():
            if key in champion_metrics_raw:
                champion_metrics_raw[key] = value
        replication_result["metrics"] = champion_metrics_raw

    y_true = replication_result.get("y_test")
    y_proba = replication_result.get("y_proba")
    y_true_arr = np.asarray(y_true).astype(int) if y_true is not None else None
    y_proba_arr = np.asarray(y_proba).astype(float) if y_proba is not None else None

    metrics = dict(replication_result.get("metrics") or {})
    metrics = {k: _json_safe_scalar(v) for k, v in metrics.items()}
    if mdd_reported:
        metrics["mdd_reported"] = mdd_reported

    roc_curve = []
    if y_true_arr is not None and y_proba_arr is not None:
        roc_curve = compute_roc_curve(y_true_arr, y_proba_arr)

    benchmark_payload = _run_benchmark_comparison(
        replication_result=replication_result,
        challenger_model_name=challenger_model_name,
        target_col=target_col,
        y_true=y_true_arr,
        y_proba=y_proba_arr,
    )

    result = {
        "stage": "benchmarking",
        "report": {
            "metrics": metrics,
            "roc_curve": {"points": roc_curve, "auc": metrics.get("roc_auc")},
            "benchmark": benchmark_payload,
        },
    }

    try:
        persistence.log_event(
            persistence.VALIDATION_PIPELINE_LOG,
            stage="benchmarking",
            payload={
                "model_name": model_name,
                "metrics": metrics,
                "champion_vs_challenger": benchmark_payload.get("status") if isinstance(benchmark_payload, dict) else None,
            },
            full_payload=result,
        )
    except Exception as e:
        print(f"[persistence] failed to log benchmarking: {e}")

    return result


@app.post("/validation/compliance")
async def validation_compliance(request: ComplianceRequest) -> Dict[str, Any]:
    agent = _load_agent2()
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent2 is unavailable. Missing or invalid rules.json?")
    stage = request.stage.lower()
    payload = request.payload
    flags = []
    if stage == "data":
        df = pd.DataFrame(payload.get("data", [])) if payload.get("data") else None
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Data payload is required for data compliance checks.")
        col_types = payload.get("col_types") or detect_column_types(df)
        leakage_risk_cols = payload.get("leakage_risk_cols")
        agent.check_data(df, col_types, leakage_risk_cols=leakage_risk_cols)
        agent.check_rules_from_agent1("data", payload.get("context", {}))
        flags = agent.get_stage_summary("data")["flags"]
    elif stage == "feature":
        plan = payload.get("feature_engineering_plan")
        if not plan:
            raise HTTPException(status_code=400, detail="feature_engineering_plan is required for feature compliance.")
        all_columns = payload.get("all_columns")
        agent.check_features(plan, all_columns=all_columns)
        agent.check_rules_from_agent1("feature", payload.get("context", {}))
        flags = agent.get_stage_summary("feature")["flags"]
    elif stage == "training":
        training_config = payload.get("training_config", {})
        training_info = payload.get("training_info", {})
        test_auc = payload.get("test_auc")
        imbalance_ratio = payload.get("imbalance_ratio")
        task_type = payload.get("task_type", "binary")
        agent.check_training(
            training_config,
            training_info=training_info,
            test_auc=test_auc,
            imbalance_ratio=imbalance_ratio,
            task_type=task_type,
        )
        agent.check_rules_from_agent1("training", payload.get("context", {}))
        flags = agent.get_stage_summary("training")["flags"]
    elif stage == "evaluation":
        metrics = payload.get("metrics")
        if not metrics:
            raise HTTPException(status_code=400, detail="metrics are required for evaluation compliance.")
        training_info = payload.get("training_info", {})
        threshold = payload.get("threshold", 0.5)
        explainability_done = payload.get("explainability_done", False)
        hetero_check = payload.get("heteroscedasticity_check")
        agent.check_evaluation(
            metrics=metrics,
            training_info=training_info,
            threshold=threshold,
            explainability_done=explainability_done,
            heteroscedasticity_result=hetero_check,
            pd_output_present=payload.get("pd_output_present", False),
            staging_logic_present=payload.get("staging_logic_present", False),
            sicr_flagged=payload.get("sicr_flagged", False),
            ecl_estimated=payload.get("ecl_estimated", False),
            concentration_analysis=payload.get("concentration_analysis", False),
            exposure_reported=payload.get("exposure_reported", False),
            past_due_breakdown=payload.get("past_due_breakdown", False),
            shap_available=payload.get("shap_available", False),
        )
        agent.check_rules_from_agent1("evaluation", payload.get("context", {}))
        flags = agent.get_stage_summary("evaluation")["flags"]
    else:
        raise HTTPException(status_code=400, detail="Unknown compliance stage. Use data, feature, training, or evaluation.")
    return {
        "stage": stage,
        "flags": flags,
        "report": agent.get_full_report(),
    }


@app.post("/validation/tier")
async def validation_tier(request: TierRequest) -> Dict[str, Any]:
    agent = _load_agent2()
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent2 is unavailable. Missing or invalid rules.json?")
    tier = agent.tier_model(request.training_config, request.metrics, fe_summary=request.fe_summary)
    return {"tier": tier, "report": agent.get_full_report()}


@app.post("/validation/replication")
async def validation_replication(
    algorithm: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    target_col: str = Form(...),
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    seeds: Optional[str] = Form("42,43,44,45,46"),
    test_size: float = Form(0.15),
    val_size: float = Form(0.15),
    random_seed: int = Form(42),
    cv_folds: int = Form(5),
    mdd_file: Optional[UploadFile] = File(None),
    reported_json: Optional[str] = Form(None),
    intake_json: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Run backend replication checks. Returns {'flags', 'report'} to match existing shapes.

    `reported` (the developer-reported metrics R4.2/R4.3/R4.4/R4.8 compare against)
    can come from two places, combinable: `reported_json` (e.g. pulled from an
    already-completed training run's evaluation_metrics) and/or an uploaded MDD
    file. When both are present, values extracted from the MDD take precedence
    per-key, since the MDD is the authoritative developer documentation.

    This now also returns the full performance report (`report.metrics`,
    `report.roc_curve`, `report.pr_curve`, `report.confusion_matrix`,
    `report.calibration_chart`, `report.score_distribution`,
    `report.train_test_auc_gap`, `report.metric_checks`,
    `report.compliance_findings`, `report.threshold_analysis`) alongside the
    existing `report.replication.{result,checks}` — Stage 3 (Model
    Replication) and the old Stage 4 "Performance" tab are now a single
    combined page, so this one call produces everything it needs.
    `intake_json` is optional and only feeds `compliance_findings`.
    """
    df = await _read_dataframe(file=file, csv_text=csv_text)
    # parse seeds
    try:
        seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    except Exception:
        seed_list = [random_seed]

    resolved_model_name = (algorithm or model_name or "").strip()
    if not resolved_model_name:
        raise HTTPException(status_code=400, detail="Model or algorithm is required.")

    intake_payload: Dict[str, Any] = {}
    if intake_json:
        try:
            intake_payload = json.loads(intake_json)
        except Exception:
            intake_payload = {}

    reported: Dict[str, Any] = {}
    mdd_text = ""
    if reported_json:
        try:
            parsed = json.loads(reported_json)
            if isinstance(parsed, dict):
                reported.update({k: v for k, v in parsed.items() if v is not None})
        except Exception:
            pass
    if mdd_file is not None:
        try:
            # parse_mdd_file was ported from Streamlit and expects a sync
            # file-like object with .name/.seek()/.read(). FastAPI's
            # UploadFile itself is async (.filename, async .read()/.seek()),
            # so we reach through to the underlying SpooledTemporaryFile the
            # same way _read_dataframe() does elsewhere, and attach the
            # filename it needs for extension sniffing (.pdf/.docx/.txt).
            mdd_file.file.seek(0)
            mdd_underlying = mdd_file.file
            mdd_underlying.name = mdd_file.filename or ""
            mdd_text = parse_mdd_file(mdd_underlying)
            mdd_reported = extract_metrics_from_mdd(mdd_text)
            reported.update({k: v for k, v in mdd_reported.items() if v is not None})
        except Exception:
            pass

    result = run_replication(
        df=df,
        target_col=target_col,
        model_name=resolved_model_name,
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed,
        cv_folds=cv_folds,
        reported=reported,
        seeds=seed_list,
    )

    checks = []
    try:
        checks = evaluate_replication_checks(result, reported, seed_list)
    except Exception:
        checks = []

    # Build flags similar to Agent2: list of failing check ids
    flags = [c["id"] for c in checks if c.get("status") in ("FAIL",)]

    safe_result = dict(result)
    for key in ["pipeline", "X_train", "X_test", "y_train", "y_test", "y_proba", "y_pred"]:
        safe_result.pop(key, None)
    safe_result = _serialize_stage5_payload(safe_result)

    performance_report = _build_performance_report(df, result, intake_payload, mdd_text)

    response_payload = {
        "stage": "replication",
        "flags": flags,
        "report": {
            "replication": {"result": safe_result, "checks": checks},
            **performance_report,
        },
    }

    try:
        persistence.log_event(
            persistence.VALIDATION_PIPELINE_LOG,
            stage="replication",
            payload={
                "model_name": resolved_model_name,
                "metrics": performance_report.get("metrics"),
                "flags": flags,
            },
            full_payload=response_payload,
        )
    except Exception as e:
        print(f"[persistence] failed to log replication: {e}")

    return response_payload


@app.post("/validation/stage7/bias-check")
async def validation_stage7_bias_check(
    algorithm: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    target_col: str = Form(...),
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    protected_col: Optional[str] = Form(None),
    test_size: float = Form(0.15),
    val_size: float = Form(0.15),
    random_seed: int = Form(42),
    cv_folds: int = Form(5),
) -> Dict[str, Any]:
    """Stage 7 Fair Lending Bias Check (Explainability & Fairness tab).

    Mirrors app.py's render_val_regulatory() Tab 2 bias-check button. The
    backend is stateless (no persisted Stage 4 session), so this reruns the
    train/split via `_fit_core` (shared with /validation/replication) rather
    than reusing a cached model — same pattern /validation/performance
    already uses for "Stage 4" data.

    If `protected_col` is omitted, only detects candidate protected columns
    (age/gender/region/etc.) so the frontend can populate its selector.
    """
    df = await _read_dataframe(file=file, csv_text=csv_text)
    resolved_model_name = (algorithm or model_name or "").strip()
    if not resolved_model_name:
        raise HTTPException(status_code=400, detail="Model or algorithm is required.")

    if not protected_col:
        return {
            "success": True,
            "error": None,
            "protected_columns": detect_protected_columns(df),
            "bias_col": None,
            "rows": [],
            "check": None,
        }

    result = run_bias_check(
        df=df,
        target_col=target_col,
        model_name=resolved_model_name,
        protected_col=protected_col,
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed,
        cv_folds=cv_folds,
    )

    try:
        persistence.log_event(
            persistence.VALIDATION_PIPELINE_LOG,
            stage="fair_lending_bias",
            payload={
                "model_name": resolved_model_name,
                "protected_col": protected_col,
                "success": result.get("success"),
                "check": result.get("check"),
            },
            full_payload=result,
        )
    except Exception as e:
        print(f"[persistence] failed to log fair_lending_bias: {e}")

    return result


@app.post("/validation/stress/run")
async def validation_stress_run(
    algorithm: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    target_col: str = Form(...),
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    seeds: Optional[str] = Form("42,43,44,45,46"),
    test_size: float = Form(0.15),
    val_size: float = Form(0.15),
    random_seed: int = Form(42),
    cv_folds: int = Form(5),
    freq: Optional[str] = Form("quarterly"),
    date_col: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Stage 6 — Stress Testing & Backtesting.

    Runs everything from validation_stress_core.run_stress_suite() in one call:
    ablation-based sensitivity (6.1), macro stress scenarios (6.2), PSI score
    stability (6.4), backtesting vs realised default rate (6.8), and
    directional testing (6.9a-c). Ported from the Streamlit app's Stage 6
    tabs — see validation_stress_core.py for what was intentionally dropped
    (the LGD-dependent directional check).

    Takes the same training params as /validation/replication and retrains
    within this request (no pipeline is cached between requests in this
    backend — same tradeoff /validation/replication already makes).
    """
    df = await _read_dataframe(file=file, csv_text=csv_text)

    try:
        seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    except Exception:
        seed_list = [random_seed]

    resolved_model_name = (algorithm or model_name or "").strip()
    if not resolved_model_name:
        raise HTTPException(status_code=400, detail="Model or algorithm is required.")

    rep_result = run_replication(
        df=df,
        target_col=target_col,
        model_name=resolved_model_name,
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed,
        cv_folds=cv_folds,
        reported={},
        seeds=seed_list,
        compute_seed_stability=False,
    )

    suite = run_stress_suite(rep_result, val_df=df, freq_key=freq or "quarterly", date_col=date_col)
    result = {"stage": "stress", "report": suite}

    try:
        persistence.log_event(
            persistence.VALIDATION_PIPELINE_LOG,
            stage="stress_testing",
            payload={
                "model_name": resolved_model_name,
                "scenarios": list(suite.keys()) if isinstance(suite, dict) else None,
            },
            full_payload=result,
        )
    except Exception as e:
        print(f"[persistence] failed to log stress_testing: {e}")

    return result


@app.post("/validation/stress/shock")
async def validation_stress_shock(
    algorithm: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    target_col: str = Form(...),
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    seeds: Optional[str] = Form("42,43,44,45,46"),
    test_size: float = Form(0.15),
    val_size: float = Form(0.15),
    random_seed: int = Form(42),
    cv_folds: int = Form(5),
    shock_feature: str = Form(...),
    shock_direction: str = Form(...),
    shock_magnitude_pct: float = Form(20),
) -> Dict[str, Any]:
    """Stage 6 — manual single-feature shock (the interactive "Apply Shock"
    control). Separate from /validation/stress/run because it's re-run on
    demand with different feature/direction/magnitude each time, and there's
    no cached pipeline to reuse — so, like /validation/stress/run, it
    retrains within the request.
    """
    df = await _read_dataframe(file=file, csv_text=csv_text)

    try:
        seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    except Exception:
        seed_list = [random_seed]

    resolved_model_name = (algorithm or model_name or "").strip()
    if not resolved_model_name:
        raise HTTPException(status_code=400, detail="Model or algorithm is required.")

    if shock_direction not in ("increase", "decrease"):
        raise HTTPException(status_code=400, detail="shock_direction must be 'increase' or 'decrease'.")

    rep_result = run_replication(
        df=df,
        target_col=target_col,
        model_name=resolved_model_name,
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed,
        cv_folds=cv_folds,
        reported={},
        seeds=seed_list,
        compute_seed_stability=False,
    )

    try:
        shock_result = run_manual_shock(rep_result, shock_feature, shock_direction, shock_magnitude_pct)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"stage": "stress_shock", "result": shock_result}


@app.post("/validation/agent2")
async def validation_agent2(
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    intake_json: Optional[str] = Form(None),
    mdd_file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Run the POC ValidationAgent2 engine and return its structured report.
    If the POC module is not available, returns a minimal fallback.
    """
    df = None
    try:
        df = await _read_dataframe(file=file, csv_text=csv_text)
    except HTTPException:
        df = None

    intake = {}
    if intake_json:
        try:
            intake = json.loads(intake_json)
        except Exception:
            intake = {}

    mdd_text = ""
    if mdd_file is not None:
        try:
            mdd_text = parse_mdd_file(mdd_file)
        except Exception:
            mdd_text = ""

    report = run_validation_agent2(df, intake, mdd_text)
    # map to existing shape: flags + report
    flags = []
    try:
        flags = [f.get("check_id") for f in report.get("all_findings", []) if f.get("status") == "FAIL"]
    except Exception:
        flags = []

    result = {"stage": "agent2", "flags": flags, "report": report}

    try:
        all_findings = report.get("all_findings", [])
        persistence.log_event(
            persistence.VALIDATION_PIPELINE_LOG,
            stage="findings",
            payload={
                "num_findings": len(all_findings),
                "num_failed": len(flags),
                "overall_result": "FAIL" if flags else "PASS",
            },
            full_payload=result,
        )
    except Exception as e:
        print(f"[persistence] failed to log findings: {e}")

    return result


def _normalize_finding(f: dict) -> dict:
    """Return a frontend-friendly normalized finding dict."""
    return {
        "id": f.get("check_id"),
        "stage": f.get("stage"),
        "title": f.get("title"),
        "status": f.get("status"),
        "severity": f.get("severity"),
        "observed": f.get("observed"),
        "detail": f.get("detail"),
        "mdd_reference": f.get("mdd_reference"),
        "check_type": f.get("check_type"),
        "source": f.get("source"),
        "principle": f.get("principle"),
    }


def _derive_feature_importance_payload(report: dict, df=None, model_file: Optional[UploadFile] = None, intake: Optional[dict] = None) -> dict:
    """Build a frontend-friendly feature importance payload from available artifacts."""
    replicated = report.get("replicated_importances", {}) or {}
    importance_df = None

    if model_file is not None:
        try:
            model_bytes = model_file.file.read()
            _buf = io.BytesIO(model_bytes)
            pipeline = joblib.load(_buf)
            df_imp = extract_feature_importance(pipeline)
            if df_imp is not None:
                importance_df = df_imp.to_dict(orient="records")
        except Exception:
            importance_df = None

    if importance_df is None and df is not None and not getattr(df, "empty", True):
        try:
            target_col = None
            if intake:
                target_col = intake.get("target_col") or intake.get("target") or intake.get("default_col")
            if target_col and target_col in df.columns:
                target_series = df[target_col]
            else:
                target_candidates = [c for c in df.columns if df[c].nunique(dropna=True) <= 2]
                target_col = target_candidates[0] if target_candidates else None
                target_series = df[target_col] if target_col else None
            if target_col and target_series is not None:
                feature_candidates = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
                if feature_candidates:
                    X = df[feature_candidates].fillna(df[feature_candidates].median())
                    y = target_series.astype(int).fillna(0)
                    if y.nunique() >= 2:
                        model = LogisticRegression(max_iter=1000, random_state=42)
                        model.fit(X, y)
                        coef_abs = np.abs(model.coef_).mean(axis=0)
                        importance_series = pd.Series(coef_abs, index=feature_candidates).sort_values(ascending=False)
                        importance_df = [
                            {"Feature": feature, "Importance": float(score)}
                            for feature, score in importance_series.items()
                        ]
        except Exception:
            importance_df = None

    if importance_df is None and replicated:
        try:
            importance_df = [
                {"Feature": feature, "Importance": float(score)}
                for feature, score in sorted(replicated.items(), key=lambda item: item[1], reverse=True)
            ]
        except Exception:
            importance_df = None

    top_drivers = importance_df[:8] if importance_df else []
    return {
        "replicated_importances": replicated,
        "importance_df": importance_df,
        "top_drivers": top_drivers,
    }


def _group_stage2(report: dict, rag_results: Optional[List[dict]] = None, selected_frameworks: Optional[List[str]] = None) -> dict:
    """Map ValidationAgent2 + Agent2 RAG results into the Stage 2 UI payload.

    Mirrors _group_stage3 exactly, but for "Stage 2: Data Validation":
    `thresholdChecks` are the quantitative ValidationAgent2 findings tagged
    for that stage; `ragRules` are the qualitative MDD rules retrieved from
    the regulatory knowledge store (rag_store/val_mdd_rules.json, stage ==
    "data_validation") and cross-checked against the MDD text via
    check_mdd_keywords / check_for_validation — same RAG pipeline Stage 3
    uses, just a different stage key. No featureRelevance section here;
    that's conceptual-soundness-specific (feature importance charts).
    """
    findings = []
    try:
        findings = report.get("findings_by_stage", {}).get("Stage 2: Data Validation", [])
    except Exception:
        findings = []

    raw_findings = [f for f in findings]
    rag_results = rag_results or []

    filtered_findings = raw_findings
    if selected_frameworks:
        filtered_findings = [f for f in raw_findings if _rule_matches_frameworks(f, selected_frameworks)]

    threshold_checks = [_normalize_threshold_check(f) for f in filtered_findings]
    rag_rules = [_normalize_rag_rule(r) for r in rag_results]

    combined = filtered_findings + rag_results
    stage2_counts = {
        "pass": sum(1 for f in combined if f.get("status") == "PASS"),
        "warn": sum(1 for f in combined if f.get("status") == "WARN"),
        "fail": sum(1 for f in combined if f.get("status") == "FAIL"),
        "pending": sum(1 for f in combined if f.get("status") == "PENDING"),
        # Anything outside PASS/WARN/FAIL/PENDING (e.g. an unrecognized or
        # missing status) is bucketed here so summary.total always equals
        # the sum of every exposed bucket instead of silently over-counting.
        "na": sum(1 for f in combined if f.get("status") not in ("PASS", "WARN", "FAIL", "PENDING")),
    }
    stage2_high_fails = [f for f in combined if f.get("status") == "FAIL" and str(f.get("severity", "")).upper() == "HIGH"]
    if len(stage2_high_fails) == 0 and stage2_counts["fail"] == 0 and stage2_counts["pending"] == 0:
        stage2_verdict = "PASS"
    elif len(stage2_high_fails) == 0 and stage2_counts["fail"] == 0:
        stage2_verdict = "CONDITIONAL"
    elif len(stage2_high_fails) <= 2:
        stage2_verdict = "CONDITIONAL"
    else:
        stage2_verdict = "FAIL"

    remediation_items = []
    if stage2_high_fails:
        remediation_items.append(f"Address {len(stage2_high_fails)} high-severity finding(s) before approval.")
    if stage2_counts["warn"]:
        remediation_items.append(f"Document and remediate {stage2_counts['warn']} warning(s) in the MDD or model controls.")
    if not remediation_items:
        remediation_items.append("No material remediation required at this stage.")

    _FRAMEWORK_DISPLAY = {"IFRS9": "IFRS 9", "IFRS 9": "IFRS 9", "SS1/23": "SS1/23", "SS11/13": "SS11/13", "RBI": "RBI Model Risk Management"}

    def _display_framework_name(value: Optional[str]) -> str:
        if not value:
            return ""
        return _FRAMEWORK_DISPLAY.get(str(value), str(value))

    regulatory_references = sorted({_display_framework_name(f.get("source")) for f in combined if f.get("source")})
    if not regulatory_references:
        if selected_frameworks:
            regulatory_references = [_display_framework_name(fw) for fw in selected_frameworks]
        else:
            regulatory_references = ["IFRS 9", "SS11/13", "SS1/23"]

    regulatory = {
        "verdict": stage2_verdict,
        "counts": stage2_counts,
        "remediation_summary": " ".join(remediation_items),
        "regulatory_references": regulatory_references,
        "high_severity_fails": [_normalize_finding(f) for f in stage2_high_fails],
    }

    pending_llm_ids = [
        r.get("check_id", r.get("rule_id"))
        for r in rag_results
        if r.get("check_source") == "keyword_search" and r.get("status") != "PASS"
    ]

    return {
        "thresholdChecks": threshold_checks,
        "ragRules": rag_rules,
        "summary": {
            "total": len(combined),
            "pass": stage2_counts["pass"],
            "warn": stage2_counts["warn"],
            "fail": stage2_counts["fail"],
            "pending": stage2_counts["pending"],
            "na": stage2_counts["na"],
        },
        "regulatoryAlignment": regulatory,
        "raw_findings": raw_findings,
        "pending_llm_ids": list(dict.fromkeys(pending_llm_ids)),
    }


def _normalize_threshold_check(f: dict) -> dict:
    """Normalize a check_conceptual_soundness() finding into the
    'Recommended Threshold Checks' card schema (mirrors app.py's left column).

    check_type is passed through so the frontend can tell a quantitative
    check (check_type "data"/"cross_reference" — e.g. VIF > 10, missing data
    < 20%) from a documentation check (check_type "doc" — e.g. methodology
    justification, bias documentation). Only the former's numeric threshold
    is an industry-statistical convention rather than regulatory text, so
    only those get the "Regulatory basis" / "Threshold" split label.
    """
    return {
        "check_id": f.get("check_id", ""),
        "title": f.get("title", ""),
        "severity": str(f.get("severity", "medium")).upper(),
        "status": f.get("status", "WARN"),
        "source": f.get("source", ""),
        "principle": f.get("principle", ""),
        "observed": f.get("observed", ""),
        "threshold": f.get("threshold", ""),
        "detail": f.get("detail", ""),
        "check_type": f.get("check_type", ""),
    }


def _normalize_rag_rule(r: dict) -> dict:
    """Normalize an Agent2 RAG rule result (check_for_validation / check_mdd_keywords /
    check_documents_with_llm) into the 'RAG Agent Rules' card schema (mirrors app.py's
    right column)."""
    return {
        "rule_id": r.get("rule_id", r.get("check_id", "?")),
        "flag": r.get("flag", r.get("title", "")),
        "suggestion": r.get("suggestion", r.get("detail", "")),
        "severity": str(r.get("severity", "medium")).upper(),
        "status": r.get("status", "WARN"),
        "source": r.get("source", ""),
        "principle": r.get("principle", ""),
        "observed_value": r.get("observed_value", r.get("observed")),
        "not_verifiable": bool(r.get("not_verifiable", False)),
        "check_source": r.get("check_source", ""),
        "reasoning": r.get("reasoning", ""),
    }


def _group_stage3(report: dict, rag_results: Optional[List[dict]] = None, df=None, model_file: Optional[UploadFile] = None, intake: Optional[dict] = None, mdd_text: str = "", selected_frameworks: Optional[List[str]] = None) -> dict:
    """Map ValidationAgent2 + Agent2 RAG results into the Stage 3 UI payload.

    Returns a dict with keys: thresholdChecks, ragRules, summary, regulatoryAlignment,
    featureRelevance, raw_findings, replicated_importances, pending_llm_ids.
    """
    findings = []
    try:
        findings = report.get("findings_by_stage", {}).get("Stage 3: Conceptual Soundness", [])
    except Exception:
        findings = []

    raw_findings = [f for f in findings]
    rag_results = rag_results or []

    feature_relevance = _derive_feature_importance_payload(report, df=df, model_file=model_file, intake=intake)
    replicated = feature_relevance.get("replicated_importances", {}) or {}

    threshold_checks = [_normalize_threshold_check(f) for f in raw_findings]
    rag_rules = [_normalize_rag_rule(r) for r in rag_results]

    # Regulatory Alignment: summarize combined threshold + RAG outcomes for Stage 3.
    combined = raw_findings + rag_results
    stage3_counts = {
        "pass": sum(1 for f in combined if f.get("status") == "PASS"),
        "warn": sum(1 for f in combined if f.get("status") == "WARN"),
        "fail": sum(1 for f in combined if f.get("status") == "FAIL"),
        "pending": sum(1 for f in combined if f.get("status") == "PENDING"),
        # See _group_stage2's identical "na" bucket for the reasoning.
        "na": sum(1 for f in combined if f.get("status") not in ("PASS", "WARN", "FAIL", "PENDING")),
    }
    stage3_high_fails = [f for f in combined if f.get("status") == "FAIL" and str(f.get("severity", "")).upper() == "HIGH"]
    if len(stage3_high_fails) == 0 and stage3_counts["fail"] == 0 and stage3_counts["pending"] == 0:
        stage3_verdict = "PASS"
    elif len(stage3_high_fails) == 0 and stage3_counts["fail"] == 0:
        stage3_verdict = "CONDITIONAL"
    elif len(stage3_high_fails) <= 2:
        stage3_verdict = "CONDITIONAL"
    else:
        stage3_verdict = "FAIL"

    remediation_items = []
    if stage3_high_fails:
        remediation_items.append(f"Address {len(stage3_high_fails)} high-severity finding(s) before approval.")
    if stage3_counts["warn"]:
        remediation_items.append(f"Document and remediate {stage3_counts['warn']} warning(s) in the MDD or model controls.")
    if not remediation_items:
        remediation_items.append("No material remediation required at this stage.")

    _FRAMEWORK_DISPLAY = {"IFRS9": "IFRS 9", "SS1/23": "SS1/23", "RBI": "RBI Model Risk Management"}
    regulatory_references = sorted({f.get("source") for f in combined if f.get("source")})
    if not regulatory_references:
        if selected_frameworks:
            regulatory_references = [_FRAMEWORK_DISPLAY.get(fw, fw) for fw in selected_frameworks]
        else:
            regulatory_references = ["SS1/23", "SS11/13"]

    regulatory = {
        "verdict": stage3_verdict,
        "counts": stage3_counts,
        "remediation_summary": " ".join(remediation_items),
        "regulatory_references": regulatory_references,
        "high_severity_fails": [_normalize_finding(f) for f in stage3_high_fails],
    }

    pending_llm_ids = [
        r.get("check_id", r.get("rule_id"))
        for r in rag_results
        if r.get("check_source") == "keyword_search" and r.get("status") != "PASS"
    ]

    return {
        "thresholdChecks": threshold_checks,
        "ragRules": rag_rules,
        "summary": {
            "total": len(combined),
            "pass": stage3_counts["pass"],
            "warn": stage3_counts["warn"],
            "fail": stage3_counts["fail"],
            "pending": stage3_counts["pending"],
            "na": stage3_counts["na"],
        },
        "regulatoryAlignment": regulatory,
        "featureRelevance": feature_relevance,
        "raw_findings": raw_findings,
        "replicated_importances": replicated,
        "pending_llm_ids": list(dict.fromkeys(pending_llm_ids)),
    }


@app.post("/validation/stage2/run")
async def validation_stage2_run(
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    intake_json: Optional[str] = Form(None),
    mdd_file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Run Stage 2 Data Validation checks and return mapped JSON.

    Mirrors /validation/stage3/run exactly, one stage over: quantitative
    ValidationAgent2 findings tagged "Stage 2: Data Validation", plus the
    RAG rule-matching pipeline (rag_store/val_mdd_rules.json, stage ==
    "data_validation") — quantitative dataset-metric rules via
    check_for_validation, and a keyword-search cross-check of the MDD text
    via check_mdd_keywords. Same SourceAgent2 instance and same call
    pattern Stage 3 already uses successfully, just a different stage key.
    """
    df = None
    try:
        df = await _read_dataframe(file=file, csv_text=csv_text)
    except HTTPException:
        df = None

    intake = {}
    if intake_json:
        try:
            intake = json.loads(intake_json)
        except Exception:
            intake = {}

    selected_frameworks = intake.get("frameworks") if isinstance(intake, dict) else None

    mdd_text = ""
    if mdd_file is not None:
        try:
            # See /validation/stage3/run — parse_mdd_file needs the sync
            # file-like wrapper or every keyword-search RAG rule silently
            # sees empty MDD content and reports 0 keyword hits.
            mdd_text = parse_mdd_file(_sync_file_like(mdd_file))
        except Exception:
            mdd_text = ""

    report = run_validation_agent2(df, intake, mdd_text)

    rag_results: List[dict] = []
    source_agent = _load_source_agent2()
    if source_agent is not None:
        try:
            rag_results.extend(source_agent.check_for_validation("data_validation", {}, frameworks=selected_frameworks))
        except Exception:
            pass

        # Conceptual/qualitative rules are judged by the LLM-based agent, but
        # that call (Deloitte Agent / Ollama fallback) can take several
        # seconds to tens of seconds — too slow to block this endpoint's
        # response. So this endpoint always asks check_documents_with_llm
        # for the fast, no-network PENDING stub (pass {} regardless of
        # mdd_text) so quantitative results render immediately; the actual
        # LLM verdicts are fetched afterward by the frontend via
        # /validation/stage2/llm-check (see mapped["llm_pending"] below).
        try:
            rag_results.extend(source_agent.check_documents_with_llm({}, stage="data_validation", frameworks=selected_frameworks))
        except Exception:
            pass

    mapped = _group_stage2(report, rag_results=rag_results, selected_frameworks=selected_frameworks)
    mapped["llm_ran"] = False
    # True when there's MDD text for the frontend to send to
    # /validation/stage2/llm-check as a fast follow-up call.
    mapped["llm_pending"] = bool(mdd_text)
    mapped["timestamp"] = pd.Timestamp.now().isoformat()

    return mapped


@app.post("/validation/stage3/run")
async def validation_stage3_run(
    file: Optional[UploadFile] = File(None),
    csv_text: Optional[str] = Form(None),
    intake_json: Optional[str] = Form(None),
    mdd_file: Optional[UploadFile] = File(None),
    model_file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Run Stage 3 Conceptual Soundness checks and return mapped JSON.

    Accepts the same upload pattern as other endpoints. Keeps calculations server-side.
    """
    df = None
    try:
        df = await _read_dataframe(file=file, csv_text=csv_text)
    except HTTPException:
        df = None

    intake = {}
    if intake_json:
        try:
            intake = json.loads(intake_json)
        except Exception:
            intake = {}

    selected_frameworks = intake.get("frameworks") if isinstance(intake, dict) else None

    mdd_text = ""
    if mdd_file is not None:
        try:
            # parse_mdd_file expects a sync file-like object with .name/.seek()/
            # .read() (ported from Streamlit's UploadedFile). FastAPI's UploadFile
            # is async, so calling parse_mdd_file(mdd_file) directly silently
            # "succeeds" but reads the repr of an unawaited coroutine instead of
            # the document text — every keyword-search RAG rule then sees empty/
            # garbage MDD content and reports 0 keyword hits. Reach through to
            # the underlying SpooledTemporaryFile, same fix already applied in
            # /validation/replication.
            mdd_text = parse_mdd_file(_sync_file_like(mdd_file))
        except Exception:
            mdd_text = ""

    # Run the full ValidationAgent2 to reuse replicated_importances if available,
    # but prefer to extract Stage 3 findings only for response.
    report = run_validation_agent2(df, intake, mdd_text)

    # RAG rule-matching pipeline (mirrors app.py:5202-5231): quantitative rules
    # checked against dataset metrics, plus a keyword-search cross-check of the
    # MDD text for qualitative rules. Without this, only the 11 threshold checks
    # (3.1-3.11) are returned and the "RAG Agent Rules" column has nothing to show.
    rag_results: List[dict] = []
    source_agent = _load_source_agent2()
    if source_agent is not None:
        corr_max = 0.0
        if df is not None:
            try:
                numeric_df = df.select_dtypes(include=["number"])
                if numeric_df.shape[1] >= 2:
                    corr_matrix = numeric_df.corr().abs().values
                    np.fill_diagonal(corr_matrix, 0.0)
                    corr_max = float(corr_matrix.max())
            except Exception:
                corr_max = 0.0

        try:
            rag_results.extend(source_agent.check_for_validation("conceptual_soundness", {
                "correlation_max": corr_max,
            }, frameworks=selected_frameworks))
        except Exception:
            pass

        # Conceptual/qualitative rules are judged by the LLM-based agent, but
        # that call can take several seconds to tens of seconds — too slow
        # to block this endpoint's response. So this endpoint always asks
        # check_documents_with_llm for the fast, no-network PENDING stub
        # (pass {} regardless of mdd_text) so quantitative results render
        # immediately; the actual LLM verdicts are fetched afterward by the
        # frontend via /validation/stage3/llm-check (see
        # mapped["llm_pending"] below).
        try:
            rag_results.extend(source_agent.check_documents_with_llm({}, stage="conceptual_soundness", frameworks=selected_frameworks))
        except Exception:
            pass

    mapped = _group_stage3(report, rag_results=rag_results, df=df, model_file=model_file, intake=intake, mdd_text=mdd_text, selected_frameworks=selected_frameworks)
    # add top-level metadata
    mapped["llm_ran"] = False
    # True when there's MDD text for the frontend to send to
    # /validation/stage3/llm-check as a fast follow-up call.
    mapped["llm_pending"] = bool(mdd_text)
    mapped["timestamp"] = pd.Timestamp.now().isoformat()

    return mapped


@app.post("/validation/stage7/run")
async def validation_stage7_run(
    intake_json: Optional[str] = Form(None),
    mdd_file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Run Stage 7 Regulatory Compliance checks (7.1-7.10).

    Mirrors app.py's render_val_regulatory() Tab 1: hardcoded keyword/threshold
    checks against the MDD text and intake_json (not RAG-based, unlike Stage 3).
    Reuses ValidationAgent2.check_regulatory_compliance() via run_all_checks(),
    same pattern as /validation/stage3/run.

    If the user selected specific regulatory frameworks on Stage 1, findings
    are filtered post-hoc by their `source`/`principle` fields before being
    normalized for the frontend.
    """
    intake = {}
    if intake_json:
        try:
            intake = json.loads(intake_json)
        except Exception:
            intake = {}

    selected_frameworks = intake.get("frameworks") if isinstance(intake, dict) else None

    mdd_text = ""
    if mdd_file is not None:
        try:
            mdd_text = parse_mdd_file(_sync_file_like(mdd_file))
        except Exception:
            mdd_text = ""

    report = run_validation_agent2(None, intake, mdd_text)
    findings = report.get("findings_by_stage", {}).get("Stage 7: Regulatory Compliance", [])

    if selected_frameworks:
        findings = [f for f in findings if _rule_matches_frameworks(f, selected_frameworks)]

    checks = [_normalize_threshold_check(f) for f in findings]

    summary = {
        "total": len(checks),
        "pass": sum(1 for c in checks if c.get("status") == "PASS"),
        "warn": sum(1 for c in checks if c.get("status") == "WARN"),
        "fail": sum(1 for c in checks if c.get("status") == "FAIL"),
        "na": sum(1 for c in checks if c.get("status") not in ("PASS", "WARN", "FAIL")),
    }

    return {"checks": checks, "summary": summary}


@app.post("/validation/parse-mdd")
async def validation_parse_mdd(
    mdd_file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Parse an uploaded Model Development Document (PDF/DOCX/TXT) and extract
    any reported metrics it contains. Backs the "Upload & Parse" control on
    the Model Intake screen — was previously called by the frontend with no
    matching route registered here, which is what produced the 404.
    """
    try:
        mdd_text = parse_mdd_file(_sync_file_like(mdd_file))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse MDD: {exc}")

    metrics = extract_metrics_from_mdd(mdd_text)
    return {"mdd_text": mdd_text, "metrics": metrics}


@app.post("/validation/stage8/findings")
async def validation_stage8_findings(
    intake_json: Optional[str] = Form(None),
    mdd_text: Optional[str] = Form(None),
    validation_profile_json: Optional[str] = Form(None),
    rep_metrics_json: Optional[str] = Form(None),
    benchmark_metrics_json: Optional[str] = Form(None),
    shock_result_json: Optional[str] = Form(None),
    bias_auc_gap: Optional[float] = Form(None),
) -> Dict[str, Any]:
    """Compile the Stage 8 Findings Tracker + overall verdict.

    Direct port of app.py's render_val_findings() all_findings compilation
    (app.py:7484-7617). The backend is stateless, so every piece that would
    have been st.session_state on the Streamlit side is passed in from the
    frontend's app-context instead — each already computed by an earlier
    stage's own endpoint/response. CALIBRATION_KEYWORDS/STAGING_KEYWORDS are
    imported from the same source-of-truth validation_agent2.py Stage 3 uses,
    so the keyword lists can't drift between stages.
    """
    ij: Dict[str, Any] = {}
    if intake_json:
        try:
            ij = json.loads(intake_json) or {}
        except Exception:
            ij = {}

    mdd_lower = (mdd_text or "").lower()

    profile: Dict[str, Any] = {}
    if validation_profile_json:
        try:
            profile = json.loads(validation_profile_json) or {}
        except Exception:
            profile = {}

    rep_metrics: Dict[str, Any] = {}
    if rep_metrics_json:
        try:
            rep_metrics = json.loads(rep_metrics_json) or {}
        except Exception:
            rep_metrics = {}

    bm_metrics: Dict[str, Any] = {}
    if benchmark_metrics_json:
        try:
            bm_metrics = json.loads(benchmark_metrics_json) or {}
        except Exception:
            bm_metrics = {}

    shock_res: Dict[str, Any] = {}
    if shock_result_json:
        try:
            shock_res = json.loads(shock_result_json) or {}
        except Exception:
            shock_res = {}

    all_findings: List[Dict[str, Any]] = []

    # Stage 1 findings
    if not ij.get("model_inventory_registered"):
        all_findings.append({"stage": "Stage 1", "check": "1.2 Model Inventory",
            "severity": "HIGH", "status": "FAIL",
            "finding": "Model not registered in model inventory",
            "recommendation": "Register model with unique ID before deployment",
            "regulation": "SS1/23 P1.2"})
    if not ij.get("independence_confirmed"):
        all_findings.append({"stage": "Stage 1", "check": "1.4 Independence",
            "severity": "HIGH", "status": "FAIL",
            "finding": "Validation independence not confirmed",
            "recommendation": "Confirm validation team has no conflict of interest",
            "regulation": "SS1/23 P4.1"})

    # Stage 2 findings — /data/profile's missing_by_column is
    # {col: {"count": int, "percentage": float 0-100}} and duplicate_rate is
    # a 0-100 percentage (see main.py's build_profile ~line 916), so divide
    # by 100 to match the 0-1 fractions the legacy Streamlit thresholds use.
    missing_by_column = profile.get("missing_by_column") or {}
    if missing_by_column:
        def _pct(entry: Any) -> float:
            return float(entry.get("percentage", 0)) if isinstance(entry, dict) else float(entry or 0)
        worst_col = max(missing_by_column, key=lambda c: _pct(missing_by_column[c]))
        missing_max = _pct(missing_by_column[worst_col]) / 100.0
        if missing_max > 0.20:
            all_findings.append({"stage": "Stage 2", "check": "2.2 Missing Data",
                "severity": "HIGH", "status": "FAIL",
                "finding": f"Column '{worst_col}' has {missing_max:.1%} missing values",
                "recommendation": "Investigate data pipeline — impute or remove high-missing columns",
                "regulation": "SS1/23 P3.2"})
    dupes = float(profile.get("duplicate_rate") or 0) / 100.0
    if dupes > 0.01:
        all_findings.append({"stage": "Stage 2", "check": "2.9 Duplicates",
            "severity": "MEDIUM", "status": "WARN",
            "finding": f"Duplicate row rate: {dupes:.2%}",
            "recommendation": "Remove duplicate records before training",
            "regulation": "SS1/23 P3.2"})

    # MDD findings — Calibration/IFRS 9 Staging are Regulatory Compliance
    # checks (now under Stage 6: Explainability and Fairness); Limitations/
    # Assumptions are Conceptual Soundness checks and Forward-Looking
    # Documentation is a Data Validation check (both now under Stage 2:
    # Data & Model Soundness). These used to be lumped into one "Stage 3/7"
    # bucket before Conceptual Soundness (old Stage 3) and Regulatory
    # Compliance (old Stage 7) were merged into different current stages —
    # they no longer belong together at all, let alone under one heading.
    for kw, check, finding, rec, reg, sev, stage in [
        (CALIBRATION_KEYWORDS, "7.1 Calibration",
         "MDD has no calibration section", "Document PD calibration methodology", "IFRS 9 §5.5", "HIGH", "Stage 6"),
        (STAGING_KEYWORDS, "7.2 IFRS 9 Staging",
         "MDD does not describe IFRS 9 staging logic", "Add Stage 1/2/3 classification and SICR triggers", "IFRS 9 §5.5.3", "HIGH", "Stage 6"),
        (["limitation", "weakness"], "3.4 Limitations",
         "MDD has no limitations section", "Add model limitations per SS1/23 P3.4", "SS1/23 P3.4", "HIGH", "Stage 2"),
        (["assumption"], "3.3 Assumptions",
         "MDD does not document model assumptions", "Document all modelling assumptions with rationale", "SS1/23 P3.3", "MEDIUM", "Stage 2"),
        (["macro", "gdp", "unemployment"], "2.4b Forward-Looking Documentation (MDD)",
         "MDD does not reference macro/forward-looking variables", "Add macro overlay per IFRS 9 B5.5.49", "IFRS 9 B5.5.49", "MEDIUM", "Stage 2"),
    ]:
        if not any(k in mdd_lower for k in kw):
            all_findings.append({"stage": stage, "check": check,
                "severity": sev, "status": "FAIL",
                "finding": finding, "recommendation": rec, "regulation": reg})

    # Stage 3 (Model Replication, was old Stage 4) — AUC replication gap
    stated_auc = ij.get("stated_auc")
    rep_auc = rep_metrics.get("roc_auc")
    if stated_auc and rep_auc:
        gap = abs(float(stated_auc) - float(rep_auc))
        if gap > 0.05:
            all_findings.append({"stage": "Stage 3", "check": "4.2 AUC Replication",
                "severity": "HIGH", "status": "FAIL",
                "finding": f"AUC gap of {gap:.4f} — stated {stated_auc:.4f} vs replicated {rep_auc:.4f}",
                "recommendation": "Developer must reconcile stated and replicated performance before resubmission",
                "regulation": "SS1/23 P4.1"})

    # Stage 4 (Performance Testing, was old Stage 5) — metric thresholds
    for metric, val, threshold, op, check, reg, sev in [
        ("ROC-AUC", rep_auc, 0.70, ">=", "5.1 ROC-AUC", "SS1/23 P4.1", "HIGH"),
        ("Recall", rep_metrics.get("recall"), 0.60, ">=", "5.2 Recall", "SS1/23 P4.4", "HIGH"),
        ("Gini", round(2 * (rep_auc or 0) - 1, 4) if rep_auc else None, 0.40, ">=", "5.3 Gini", "SS11/13 §10.3", "HIGH"),
        ("Brier Score", rep_metrics.get("brier_score"), 0.25, "<=", "5.4 Brier", "SS11/13 §10.5", "MEDIUM"),
    ]:
        if val is not None:
            failed = (op == ">=" and val < threshold) or (op == "<=" and val > threshold)
            if failed:
                all_findings.append({"stage": "Stage 4", "check": check,
                    "severity": sev, "status": "FAIL",
                    "finding": f"{metric} = {val:.4f} — below regulatory minimum {op} {threshold}",
                    "recommendation": f"Model does not meet minimum {metric} threshold — consider retraining or additional data",
                    "regulation": reg})

    # Stage 4 (Performance Testing) — benchmarking
    if bm_metrics and rep_auc:
        bm_auc = bm_metrics.get("roc_auc", 0)
        if rep_auc < bm_auc - 0.02:
            all_findings.append({"stage": "Stage 4", "check": "5.B Benchmarking",
                "severity": "MEDIUM", "status": "WARN",
                "finding": f"Submitted model AUC ({rep_auc:.4f}) underperforms benchmark ({bm_auc:.4f})",
                "recommendation": "Marginal improvement over baseline may not justify model complexity — review feature set",
                "regulation": "SS1/23 P4.2"})

    # Stage 5 (Stress & Backtesting, was old Stage 6) — sensitivity shock
    if shock_res and abs(shock_res.get("pd_change_pct", 0)) > 50:
        all_findings.append({"stage": "Stage 5", "check": "6.1 Sensitivity",
            "severity": "MEDIUM", "status": "WARN",
            "finding": f"Shocking '{shock_res.get('feature')}' by {shock_res.get('magnitude_pct')}% changes avg PD by {shock_res.get('pd_change_pct', 0):.1f}%",
            "recommendation": "Model may be over-sensitive to this feature — validate stability with wider test",
            "regulation": "SS1/23 P4.3"})

    # Stage 6 (Explainability and Fairness, was old Stage 7) — bias
    if bias_auc_gap is not None and bias_auc_gap > 0.10:
        all_findings.append({"stage": "Stage 6", "check": "8.2 Fair Lending",
            "severity": "HIGH", "status": "FAIL",
            "finding": f"AUC gap of {bias_auc_gap:.4f} across protected groups",
            "recommendation": "Investigate discriminatory bias — model may treat protected groups unfairly",
            "regulation": "SS1/23 P1.3"})

    # ── Overall verdict ──────────────────────────────────────────────
    high_count = sum(1 for f in all_findings if f["severity"] == "HIGH" and f["status"] == "FAIL")
    medium_count = sum(1 for f in all_findings if f["severity"] == "MEDIUM")
    total_count = len(all_findings)

    if high_count == 0 and medium_count <= 2:
        verdict = "APPROVED"
        verdict_desc = "Model meets all regulatory requirements and validation standards."
    elif high_count <= 2:
        verdict = "CONDITIONALLY APPROVED"
        verdict_desc = f"Model may proceed subject to resolution of {high_count} HIGH finding(s) within agreed timeframe."
    else:
        verdict = "REJECTED"
        verdict_desc = f"Model has {high_count} unresolved HIGH findings. Resubmission required after remediation."

    # ── Monitoring & revalidation recommendations, driven by model tier ──
    tier = ij.get("model_tier", "Tier 2")
    if "1" in str(tier) or "High" in str(tier):
        monitoring_frequency = "Monthly"
        revalidation_trigger = "Annual (or triggered by PSI > 0.25 or AUC drop > 0.05)"
    elif "2" in str(tier) or "Medium" in str(tier):
        monitoring_frequency = "Quarterly"
        revalidation_trigger = "18-monthly (or triggered by significant portfolio change)"
    else:
        monitoring_frequency = "Semi-annually"
        revalidation_trigger = "Every 2 years (or triggered by model change)"

    return {
        "findings": all_findings,
        "verdict": verdict,
        "verdict_desc": verdict_desc,
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": max(total_count - high_count - medium_count, 0),
        "total_count": total_count,
        "monitoring_frequency": monitoring_frequency,
        "revalidation_trigger": revalidation_trigger,
        "model_tier": tier,
        "stated_auc": stated_auc,
        "replicated_auc": rep_auc,
    }


@app.post("/validation/stage2/llm-check")
async def validation_stage2_llm(
    intake_json: Optional[str] = Form(None),
    mdd_file: Optional[UploadFile] = File(None),
    only_rule_ids: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Run the LLM-based conceptual check for Stage 2 documentation items.

    Mirrors /validation/stage3/llm-check exactly, one stage over. Called by
    the frontend as a fast follow-up to /validation/stage2/run (which
    returns PENDING stubs for these rules so the page isn't blocked on the
    LLM call — see that endpoint's mapped["llm_pending"] flag).
    """
    agent = _load_source_agent2()

    intake = {}
    if intake_json:
        try:
            intake = json.loads(intake_json)
        except Exception:
            intake = {}

    selected_frameworks = intake.get("frameworks") if isinstance(intake, dict) else None

    mdd_text = ""
    if mdd_file is not None:
        try:
            mdd_text = parse_mdd_file(_sync_file_like(mdd_file))
        except Exception:
            mdd_text = ""

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent2 RAG component unavailable")

    docs = {"MDD": mdd_text} if mdd_text else {}

    only_ids = None
    if only_rule_ids:
        only_ids = [s.strip() for s in only_rule_ids.split(",") if s.strip()]

    try:
        llm_results = agent.check_documents_with_llm(
            docs,
            stage="data_validation",
            only_rule_ids=only_ids,
            frameworks=selected_frameworks,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM deep-check failed: {e}")

    return {"llm_results": [_normalize_rag_rule(r) for r in llm_results]}


@app.post("/validation/stage3/llm-check")
async def validation_stage3_llm(
    intake_json: Optional[str] = Form(None),
    mdd_file: Optional[UploadFile] = File(None),
    only_rule_ids: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Run optional LLM deep-checks for Stage 3 documentation items.

    Mirrors app.py's "Run AI Deep-Check" button (app.py:5257-5293): sends the MDD
    text plus the RAG rule set to the LLM checker agent for rules keyword search
    could not confidently resolve. Uses the source-of-truth Agent2 (val_mdd_rules.json),
    not the local 24-06 Agent2, which has no LLM/RAG rule methods.
    """
    agent = _load_source_agent2()

    intake = {}
    if intake_json:
        try:
            intake = json.loads(intake_json)
        except Exception:
            intake = {}

    selected_frameworks = intake.get("frameworks") if isinstance(intake, dict) else None

    mdd_text = ""
    if mdd_file is not None:
        try:
            # See /validation/stage3/run — parse_mdd_file needs the sync
            # underlying file, not the async UploadFile directly.
            mdd_text = parse_mdd_file(_sync_file_like(mdd_file))
        except Exception:
            mdd_text = ""

    if agent is None:
        raise HTTPException(status_code=500, detail="Agent2 RAG component unavailable")

    docs = {"MDD": mdd_text} if mdd_text else {}

    only_ids = None
    if only_rule_ids:
        only_ids = [s.strip() for s in only_rule_ids.split(",") if s.strip()]

    try:
        llm_results = agent.check_documents_with_llm(
            docs,
            stage="conceptual_soundness",
            only_rule_ids=only_ids,
            frameworks=selected_frameworks,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM deep-check failed: {e}")

    return {"llm_results": [_normalize_rag_rule(r) for r in llm_results]}
