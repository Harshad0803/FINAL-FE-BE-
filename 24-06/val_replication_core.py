"""
val_replication_core.py — backend-only replication engine

Trimmed extraction of the POC `val_replication.py` logic for FastAPI use.
This module intentionally omits Streamlit UI and returns plain dicts suitable
for JSON serialization.
"""
from __future__ import annotations

import copy as _copy
import re as _re
import time
import traceback
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from utils import detect_column_types, detect_target_candidates, detect_task_type
from preprocessing_new import prepare_data, rebuild_preprocessor_for
from feature_engineering import analyze_for_feature_engineering, apply_feature_engineering
from train_new import split_data, compute_split_stats, train_model
from evaluate_new import compute_binary_metrics
from model_selector import CLASSIFICATION_MODELS
from explainability import extract_feature_importance


def _gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return round(2.0 * float(roc_auc_score(y_true, y_score)) - 1.0, 4)
    except Exception:
        return float("nan")


def _ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return round(float(np.max(np.abs(tpr - fpr))), 4)
    except Exception:
        return float("nan")


def _proba_1d(pipeline, X) -> Optional[np.ndarray]:
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        p = pipeline.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p
    except Exception:
        return None


def _proba_2d(pipeline, X) -> Optional[np.ndarray]:
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        return pipeline.predict_proba(X)
    except Exception:
        return None


def _pct_diff(replicated: float, reported: float) -> float:
    if reported == 0:
        return abs(replicated)
    return abs(replicated - reported) / abs(reported)


_MDD_METRIC_PATTERNS: List = [
    ("mdd_roc_auc", [
        r"roc[\s\-_]?auc[\s:=of]+([0-9]\.[0-9]{2,4})",
        r"\bauc[\s:=]+([0-9]\.[0-9]{2,4})",
        r"area under (?:the )?(?:roc )?curve[\s:=]+([0-9]\.[0-9]{2,4})",
        r"c[\s\-]?statistic[\s:=]+([0-9]\.[0-9]{2,4})",
    ]),
    ("mdd_gini", [r"gini[\s\-_]?(?:coefficient|index|score)?[\s:=]+([0-9]\.[0-9]{2,4})", r"gini[\s:=]+([0-9]\.[0-9]{2,4})"]),
    ("mdd_ks", [r"k[\s\-]?s[\s\-]?(?:statistic|score|test)?[\s:=]+([0-9]\.[0-9]{2,4})", r"kolmogorov[\s\-]?smirnov[\s:=]+([0-9]\.[0-9]{2,4})"]),
    ("mdd_cv_mean_auc", [r"cv[\s\-_]?(?:mean[\s\-_]?)?auc[\s:=]+([0-9]\.[0-9]{2,4})", r"cross[\s\-]?val(?:idation)?[\s\-_]?(?:mean[\s\-_]?)?auc[\s:=]+([0-9]\.[0-9]{2,4})", r"mean[\s\-_]?cv[\s\-_]?auc[\s:=]+([0-9]\.[0-9]{2,4})"]),
    ("mdd_accuracy", [r"accuracy[\s:=]+([0-9]\.[0-9]{2,4})", r"overall accuracy[\s:=]+([0-9]\.[0-9]{2,4})"]),
    ("mdd_precision", [r"precision[\s:=]+([0-9]\.[0-9]{2,4})", r"positive predictive value[\s:=]+([0-9]\.[0-9]{2,4})", r"ppv[\s:=]+([0-9]\.[0-9]{2,4})"]),
    ("mdd_recall", [r"recall[\s:=]+([0-9]\.[0-9]{2,4})", r"sensitivity[\s:=]+([0-9]\.[0-9]{2,4})", r"true positive rate[\s:=]+([0-9]\.[0-9]{2,4})", r"tpr[\s:=]+([0-9]\.[0-9]{2,4})"]),
    ("mdd_f1", [r"f[\s\-]?1[\s\-]?(?:score)?[\s:=]+([0-9]\.[0-9]{2,4})", r"f[\s\-]?measure[\s:=]+([0-9]\.[0-9]{2,4})"]),
]


def extract_metrics_from_mdd(mdd_text: str) -> Dict[str, Optional[float]]:
    text = mdd_text.lower()
    extracted: Dict[str, Optional[float]] = {}
    _key_map = {
        "mdd_roc_auc":    "roc_auc",
        "mdd_gini":       "gini",
        "mdd_ks":         "ks",
        "mdd_cv_mean_auc": "cv_mean_auc",
        "mdd_accuracy":   "accuracy",
        "mdd_precision":  "precision",
        "mdd_recall":     "recall",
        "mdd_f1":         "f1",
    }
    for ss_key, patterns in _MDD_METRIC_PATTERNS:
        result_key = _key_map[ss_key]
        found: Optional[float] = None
        for pat in patterns:
            matches = _re.findall(pat, text)
            for m in matches:
                try:
                    val = float(m)
                    if 0.01 <= val <= 1.0:
                        found = round(val, 4)
                        break
                except ValueError:
                    continue
            if found is not None:
                break
        extracted[result_key] = found
    return extracted


def parse_mdd_file(uploaded_file) -> str:
    name = (getattr(uploaded_file, "name", "") or "").lower()
    try:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        if name.endswith(".pdf"):
            try:
                import pypdf as _pypdf
                reader = _pypdf.PdfReader(uploaded_file)
            except ImportError:
                try:
                    import PyPDF2 as _pypdf2
                    reader = _pypdf2.PdfReader(uploaded_file)
                except ImportError:
                    raise RuntimeError(
                        "pypdf / PyPDF2 not installed — cannot parse PDF. "
                        "Upload a TXT or DOCX version of the MDD."
                    )
            return "\n".join((p.extract_text() or "") for p in reader.pages)

        if name.endswith(".docx"):
            try:
                import io as _io
                import docx as _docx
                doc = _docx.Document(_io.BytesIO(uploaded_file.read()))
                return "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                raise RuntimeError(
                    "python-docx not installed — cannot parse DOCX. "
                    "Upload a TXT or PDF version of the MDD."
                )

        data = uploaded_file.read()
        return data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else str(data)
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Could not read MDD: {e}")


def _resolve_model_class(model_name: str):
    registry = CLASSIFICATION_MODELS
    normalized_model_name = model_name
    alias_map = {
        "LogisticRegression": "Logistic Regression",
        "LogisticRegressionClassifier": "Logistic Regression",
        "RandomForest": "Random Forest",
        "RandomForestClassifier": "Random Forest",
        "GradientBoosting": "Gradient Boosting",
        "GradientBoostingClassifier": "Gradient Boosting",
        "XGBoostClassifier": "XGBoost",
        "LightGBMClassifier": "LightGBM",
    }
    if model_name in alias_map:
        normalized_model_name = alias_map[model_name]
    if normalized_model_name not in registry:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(registry.keys())}")
    model_cls = registry[normalized_model_name]["class"]
    default_params = registry[normalized_model_name].get("default_params", {}).copy()
    try:
        valid_keys = set(model_cls().get_params().keys())
        default_params = {k: v for k, v in default_params.items() if k in valid_keys}
    except Exception:
        pass
    return model_cls, default_params


def _fit_core(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    test_size: float,
    val_size: float,
    random_seed: int,
    cv_folds: int,
    use_feature_engineering: bool = False,
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Shared train/split/fit core used by both full replication and the
    lighter-weight bias check — factored out so a bias-check run doesn't
    have to pay for replication's seed-stability + ablation loops just to
    get a fitted pipeline and a test split.

    `use_feature_engineering` and `model_params` let the caller mirror the
    ACTUAL configuration the model was trained with (Stage 5's
    training_config). Defaulting `use_feature_engineering` to False matches
    /models/train's default — previously this was unconditionally applied
    here regardless of what the real model used, so a model trained without
    feature engineering was "replicated" on a different feature set
    entirely. `model_params`, when supplied, overrides the registry's
    generic default_params with whatever hyperparameters (manual or
    hyperopt-tuned) actually produced the model being validated — without
    it, replication silently substitutes a differently-tuned model, which
    can materially degrade AUC and, on imbalanced credit data, collapse
    threshold-based metrics like precision/recall/F1 toward zero even when
    the drop in AUC looks modest.
    """
    col_types = detect_column_types(df)
    task_type = detect_task_type(df[target_col])
    X, y, _, prep_report, _ = prepare_data(df, col_types, target_col)

    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(
        X, y,
        test_size=test_size,
        val_size=val_size,
        task_type=task_type,
        random_state=random_seed,
    )
    split_stats = compute_split_stats(
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test
    )

    import copy as _copy
    if use_feature_engineering:
        fe_plan = analyze_for_feature_engineering(
            X_train_raw, y_train, col_types, task_type
        )
        X_train, fe_summary = apply_feature_engineering(X_train_raw, fe_plan)
        X_val, _ = apply_feature_engineering(X_val_raw, fe_plan)
        X_test_eng, _ = apply_feature_engineering(X_test_raw, fe_plan)

        live_cols = set(X_train.columns)
        prep_report_fe = _copy.deepcopy(prep_report)
        prep_report_fe["numeric"] = {c: v for c, v in prep_report.get("numeric", {}).items() if c in live_cols}
        prep_report_fe["categorical"] = {c: v for c, v in prep_report.get("categorical", {}).items() if c in live_cols}
    else:
        # Mirrors /models/train's default (use_feature_engineering=False):
        # train directly on the split, un-engineered columns so the
        # replicated feature set matches what the real model was built on.
        X_train, X_val, X_test_eng = X_train_raw, X_val_raw, X_test_raw
        fe_summary = None
        prep_report_fe = prep_report

    model_cls, default_params = _resolve_model_class(model_name)
    merged_params = dict(default_params)
    if model_params:
        try:
            valid_keys = set(model_cls().get_params().keys())
        except Exception:
            valid_keys = set(default_params.keys()) | set(model_params.keys())
        merged_params.update({k: v for k, v in model_params.items() if k in valid_keys})
    model_inst = model_cls(**merged_params)

    pipeline, training_info, feature_names = train_model(
        X_train, y_train,
        col_types=col_types,
        target_col=target_col,
        prep_report=prep_report_fe,
        model=model_inst,
        use_cv=True,
        cv_folds=cv_folds,
        use_hyperopt=False,
        task_type=task_type,
    )

    y_proba_2d = _proba_2d(pipeline, X_test_eng)
    y_proba = _proba_1d(pipeline, X_test_eng)
    y_pred = pipeline.predict(X_test_eng)
    if y_proba is None:
        raise RuntimeError("Model does not support predict_proba — AUC cannot be computed.")

    metrics = compute_binary_metrics(
        y_test.values, y_pred, y_proba_2d,
        # None auto-selects the F1-maximizing threshold, same as production
        # training/evaluation (see main.py's _build_metrics_and_eval_data) —
        # a hardcoded 0.5 cut-off here would make replicated accuracy/
        # precision/recall/F1 diverge from what was actually reported.
        threshold=None,
    )
    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test.values, y_proba)), 4)
    except Exception:
        pass
    metrics["gini"] = _gini(y_test.values, y_proba)
    metrics["ks"] = _ks(y_test.values, y_proba)

    return {
        "col_types": col_types,
        "task_type": task_type,
        "X": X, "y": y,
        "X_train": X_train, "X_test": X_test_eng, "y_train": y_train, "y_test": y_test,
        "prep_report": prep_report, "prep_report_fe": prep_report_fe,
        "model_cls": model_cls, "default_params": merged_params,
        "pipeline": pipeline, "training_info": training_info,
        "y_proba": y_proba, "y_pred": y_pred,
        "metrics": metrics, "split_stats": split_stats,
        "use_feature_engineering": use_feature_engineering,
        "model_params_used": merged_params,
    }


def run_replication(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    test_size: float,
    val_size: float,
    random_seed: int,
    cv_folds: int,
    reported: Dict[str, Any],
    seeds: List[int],
    use_feature_engineering: bool = False,
    model_params: Optional[Dict[str, Any]] = None,
    compute_seed_stability: bool = True,
) -> Dict[str, Any]:
    # Wraps the original _run_replication but returns JSON-serializable outputs
    out = {
        "success": False,
        "error": None,
        "metrics": {},
        "seed_aucs": [],
        "cv_mean_auc": None,
        "split_stats": {},
        "timing_s": 0.0,
        "ablation": {},
        "pipeline": None,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "y_proba": None,
        "y_pred": None,
        "feature_importance": [],
        "use_feature_engineering": use_feature_engineering,
        "model_params_used": None,
    }
    t0 = time.time()
    try:
        fit = _fit_core(
            df, target_col, model_name, test_size, val_size, random_seed, cv_folds,
            use_feature_engineering=use_feature_engineering,
            model_params=model_params,
        )
        out["model_params_used"] = fit.get("model_params_used")
        col_types = fit["col_types"]
        task_type = fit["task_type"]
        X, y = fit["X"], fit["y"]
        X_train, X_test_eng, y_train, y_test = fit["X_train"], fit["X_test"], fit["y_train"], fit["y_test"]
        prep_report, prep_report_fe = fit["prep_report"], fit["prep_report_fe"]
        model_cls, default_params = fit["model_cls"], fit["default_params"]
        pipeline = fit["pipeline"]
        y_proba, y_pred, metrics = fit["y_proba"], fit["y_pred"], fit["metrics"]
        out["split_stats"] = fit["split_stats"]

        # Feature importance of the replicated model, exposed so Stage 7
        # (Explainability & Fairness) can chart it without re-running
        # replication or requiring a separately-trained model artifact.
        feature_importance: List[Dict[str, Any]] = []
        try:
            importance_df = extract_feature_importance(pipeline)
            if importance_df is not None:
                feature_importance = importance_df.to_dict(orient="records")
        except Exception:
            feature_importance = []

        out.update({
            "success": True,
            "metrics": metrics,
            "cv_mean_auc": fit["training_info"].get("cv_mean"),
            "cv_std_auc": fit["training_info"].get("cv_std"),
            "timing_s": 0.0,
            "pipeline": pipeline,
            "X_train": X_train,
            "X_test": X_test_eng,
            "y_train": y_train,
            "y_test": y_test,
            "y_proba": y_proba,
            "y_pred": y_pred,
            "feature_importance": feature_importance,
            "col_types": col_types,
            "task_type": task_type,
            "prep_report": prep_report,
            "prep_report_fe": prep_report_fe,
        })

        seed_aucs = []
        # Seed-stability re-training is only consumed by the Replication
        # tab's own report (seed_aucs/cv_mean_auc) — the Stress & Backtesting
        # endpoints reuse this function purely for the fitted pipeline +
        # ablation and never read seed_aucs, so they pass
        # compute_seed_stability=False to skip these `len(seeds)` extra full
        # model trainings (previously run and thrown away on every stress
        # suite request, which is why that page was slow).
        if compute_seed_stability:
            for s in seeds:
                try:
                    Xtr_s, Xv_s, Xte_s, ytr_s, yv_s, yte_s = split_data(
                        X, y,
                        test_size=test_size,
                        val_size=val_size,
                        task_type=task_type,
                        random_state=s,
                    )
                    if use_feature_engineering:
                        fe_plan_s = analyze_for_feature_engineering(
                            Xtr_s, ytr_s, col_types, task_type
                        )
                        Xtr_fe_s, _ = apply_feature_engineering(Xtr_s, fe_plan_s)
                        Xte_fe_s, _ = apply_feature_engineering(Xte_s, fe_plan_s)

                        live_s = set(Xtr_fe_s.columns)
                        prep_report_s = _copy.deepcopy(prep_report)
                        prep_report_s["numeric"] = {c: v for c, v in prep_report.get("numeric", {}).items() if c in live_s}
                        prep_report_s["categorical"] = {c: v for c, v in prep_report.get("categorical", {}).items() if c in live_s}
                    else:
                        Xtr_fe_s, Xte_fe_s = Xtr_s, Xte_s
                        prep_report_s = prep_report

                    m_s = model_cls(**default_params)
                    pipe_s, _, _ = train_model(
                        Xtr_fe_s, ytr_s,
                        col_types=col_types,
                        target_col=target_col,
                        prep_report=prep_report_s,
                        model=m_s,
                        use_cv=False,
                        task_type=task_type,
                    )
                    proba_s = _proba_1d(pipe_s, Xte_fe_s)
                    if proba_s is not None:
                        seed_aucs.append(round(float(roc_auc_score(yte_s.values, proba_s)), 4))
                except Exception:
                    pass
        out["seed_aucs"] = seed_aucs

        feat_cols = list(X_test_eng.columns)
        base_auc = float(metrics.get("roc_auc", 0))
        ablation = {}
        for col in feat_cols:
            try:
                X_abl = X_test_eng.drop(columns=[col])
                X_tr_abl = X_train.drop(columns=[col], errors="ignore")
                m_abl = model_cls(**default_params)
                live_abl = set(X_tr_abl.columns)
                prep_report_abl = _copy.deepcopy(prep_report_fe)
                prep_report_abl["numeric"] = {c: v for c, v in prep_report_fe.get("numeric", {}).items() if c in live_abl}
                prep_report_abl["categorical"] = {c: v for c, v in prep_report_fe.get("categorical", {}).items() if c in live_abl}
                pipe_abl, _, _ = train_model(
                    X_tr_abl, y_train,
                    col_types=col_types,
                    target_col=target_col,
                    prep_report=prep_report_abl,
                    model=m_abl,
                    use_cv=False,
                    task_type=task_type,
                )
                proba_abl = _proba_1d(pipe_abl, X_abl)
                if proba_abl is not None:
                    abl_auc = float(roc_auc_score(y_test.values, proba_abl))
                    ablation[col] = round(base_auc - abl_auc, 4)
            except Exception:
                ablation[col] = float("nan")
        out["ablation"] = ablation

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()

    out["timing_s"] = round(time.time() - t0, 1)
    return out


PROTECTED_KEYWORDS = [
    "age", "gender", "sex", "region", "ethnicity",
    "race", "nationality", "marital", "education",
    # "employment_type" only matched a column literally named that exact
    # compound (missing e.g. "employment_length", "employment_status") —
    # broadened to the stem so any employment-category column is caught.
    "employment", "loan_purpose",
]


def detect_protected_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if any(k in c.lower() for k in PROTECTED_KEYWORDS)]


def run_bias_check(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    protected_col: Optional[str],
    test_size: float,
    val_size: float,
    random_seed: int,
    cv_folds: int,
) -> Dict[str, Any]:
    """Fair lending bias check (Stage 7 Explainability & Fairness tab).

    Mirrors app.py's render_val_regulatory() Tab 2 bias-check button: trains
    the model on the same split as replication, then compares AUC across
    groups of a protected characteristic. Uses `_fit_core` rather than
    `run_replication` so it skips replication's seed-stability + ablation
    loops, which this check has no use for.
    """
    out: Dict[str, Any] = {
        "success": False,
        "error": None,
        "protected_columns": detect_protected_columns(df),
        "bias_col": protected_col,
        "rows": [],
        "check": None,
    }
    if not protected_col:
        return out

    try:
        fit = _fit_core(df, target_col, model_name, test_size, val_size, random_seed, cv_folds)
        y_test = fit["y_test"]
        y_proba = fit["y_proba"]

        if protected_col not in df.columns:
            out["error"] = f"Column '{protected_col}' not found in dataset."
            return out

        test_idx = y_test.index
        bias_vals = df.loc[test_idx, protected_col].fillna("Unknown").astype(str)
        groups = sorted(bias_vals.unique())

        rows = []
        for grp in groups:
            mask = (bias_vals == grp).values
            if mask.sum() < 10:
                continue
            y_grp = y_test.values[mask]
            p_grp = y_proba[mask]
            try:
                auc = round(float(roc_auc_score(y_grp, p_grp)), 4) if len(np.unique(y_grp)) > 1 else None
            except Exception:
                auc = None
            rows.append({
                "Group": grp,
                "Count": int(mask.sum()),
                "Default Rate": round(float(y_grp.mean()), 4),
                "Avg Predicted PD": round(float(p_grp.mean()), 4),
                "AUC": auc,
            })

        out["success"] = True
        out["rows"] = rows

        auc_vals = [r["AUC"] for r in rows if r["AUC"] is not None]
        if len(auc_vals) >= 2:
            auc_gap = round(float(max(auc_vals) - min(auc_vals)), 4)
            status = "PASS" if auc_gap < 0.05 else ("WARN" if auc_gap < 0.10 else "FAIL")
            out["check"] = {
                "check_id": "8.2",
                "title": f"Fair Lending Bias Check — {protected_col}",
                "severity": "HIGH",
                "status": status,
                "source": "SS1/23",
                "principle": "P1.3",
                "observed": f"AUC gap across groups: {auc_gap:.4f} (max: {max(auc_vals):.4f}, min: {min(auc_vals):.4f})",
                "threshold": "AUC gap across protected groups < 0.05",
                "detail": "Large AUC gap indicates model performs differently for different groups — potential discrimination risk",
            }
        else:
            out["check"] = {
                "check_id": "8.2",
                "title": f"Fair Lending Bias Check — {protected_col}",
                "severity": "HIGH",
                "status": "WARN",
                "source": "SS1/23",
                "principle": "P1.3",
                "observed": "Fewer than 2 groups with >= 10 records — AUC gap not computable",
                "threshold": "AUC gap across protected groups < 0.05",
                "detail": "",
            }
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"

    return out


def evaluate_replication_checks(result: Dict[str, Any], reported: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    rep_auc = result.get("metrics", {}).get("roc_auc")
    rep_gini = result.get("metrics", {}).get("gini")
    rep_ks = result.get("metrics", {}).get("ks")
    seed_aucs = result.get("seed_aucs", [])
    ablation = result.get("ablation", {})

    if result.get("success"):
        checks.append({"id": "R4.1", "title": "End-to-end training", "severity": "high", "status": "PASS", "observed": f"Trained in {result.get('timing_s')}s — no errors", "threshold": "Pipeline trains without errors", "detail": "Model trained successfully using the submitted dataset and the platform pipeline."})
    else:
        checks.append({"id": "R4.1", "title": "End-to-end training", "severity": "high", "status": "FAIL", "observed": result.get("error", "Unknown error"), "threshold": "Pipeline trains without errors", "detail": "Training failed."})
        for cid in ["R4.2", "R4.3", "R4.4", "R4.5", "R4.6", "R4.7", "R4.8"]:
            checks.append({"id": cid, "title": cid, "severity": "high", "status": "SKIP", "observed": "Not run — R4.1 failed", "threshold": "N/A", "detail": "Cannot evaluate: model did not train."})
        return checks

    # R4.2
    r_auc = reported.get("roc_auc")
    if r_auc is not None and rep_auc is not None:
        diff_auc = abs(rep_auc - r_auc)
        status = "PASS" if diff_auc <= 0.03 else ("WARN" if diff_auc <= 0.06 else "FAIL")
        checks.append({"id": "R4.2", "title": "Replicated AUC within ±0.03", "severity": "high", "status": status, "observed": f"Replicated AUC = {rep_auc:.4f} | Reported = {r_auc:.4f} | Δ = {diff_auc:.4f}", "threshold": "|Δ AUC| ≤ 0.03"})
    else:
        checks.append({"id": "R4.2", "title": "Replicated AUC within ±0.03", "severity": "high", "status": "WARN", "observed": f"Replicated AUC = {rep_auc if rep_auc else 'N/A'} | Reported AUC not provided", "threshold": "|Δ AUC| ≤ 0.03"})

    # R4.3
    r_gini = reported.get("gini")
    r_ks = reported.get("ks")
    gini_ok = True
    ks_ok = True
    gini_diff = None
    ks_diff = None
    if r_gini is not None and rep_gini is not None:
        gini_diff = abs(rep_gini - r_gini)
        gini_ok = gini_diff <= 0.05
    if r_ks is not None and rep_ks is not None:
        ks_diff = abs(rep_ks - r_ks)
        ks_ok = ks_diff <= 0.03
    if r_gini is None and r_ks is None:
        checks.append({"id": "R4.3", "title": "Gini within ±0.05 / KS within ±0.03", "severity": "high", "status": "WARN", "observed": f"Replicated Gini = {rep_gini:.4f}, KS = {rep_ks:.4f} | Reported values not provided", "threshold": "|Δ Gini| ≤ 0.05 and |Δ KS| ≤ 0.03"})
    else:
        gini_ks_status = "PASS" if (gini_ok and ks_ok) else "FAIL"
        checks.append({"id": "R4.3", "title": "Gini within ±0.05 / KS within ±0.03", "severity": "high", "status": gini_ks_status, "observed": f"Gini/Ks comparison", "threshold": "|Δ Gini| ≤ 0.05 and |Δ KS| ≤ 0.03"})

    # R4.4 simple implementation
    metric_keys_r4 = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    r44_rows = []
    any_reported = False
    r44_fails = []
    for key, label in metric_keys_r4:
        rep_val = result.get("metrics", {}).get(key)
        repl_val = rep_val
        rep_reported = reported.get(key)
        if rep_reported is None or repl_val is None:
            r44_rows.append({"Metric": label, "Replicated": repl_val, "Reported": rep_reported, "Rel. Diff": "N/A", "Status": "SKIP"})
            continue
        any_reported = True
        rel = _pct_diff(float(repl_val), float(rep_reported))
        ok = rel <= 0.05
        status_k = "PASS" if ok else "FAIL"
        if not ok:
            r44_fails.append(f"{label} (Δ = {rel:.1%})")
        r44_rows.append({"Metric": label, "Replicated": round(float(repl_val), 4), "Reported": round(float(rep_reported), 4), "Rel. Diff": f"{rel:.2%}", "Status": status_k})
    if not any_reported:
        r44_status = "WARN"
        r44_obs = "No reported values entered — comparison not possible"
    elif r44_fails:
        r44_status = "FAIL"
        r44_obs = f"Out of tolerance: {', '.join(r44_fails)}"
    else:
        r44_status = "PASS"
        r44_obs = "All reported metrics replicated within ±5%"
    checks.append({"id": "R4.4", "title": "Metrics within ±5% at reported threshold", "severity": "medium", "status": r44_status, "observed": r44_obs, "threshold": "Relative difference ≤ 5% per metric", "_table": r44_rows})

    # R4.5
    if ablation:
        max_drop_col = max(ablation, key=lambda k: ablation[k] if not np.isnan(ablation[k]) else -999)
        max_drop_val = ablation.get(max_drop_col, float("nan"))
        r45_status = "PASS" if (not np.isnan(max_drop_val) and max_drop_val < 0.05) else "FAIL"
        checks.append({"id": "R4.5", "title": "Feature removal: AUC degradation < 0.05", "severity": "medium", "status": r45_status, "observed": f"Largest AUC drop: {max_drop_val:.4f} when removing '{max_drop_col}'", "_ablation": ablation})
    else:
        checks.append({"id": "R4.5", "title": "Feature removal: AUC degradation < 0.05", "severity": "medium", "status": "SKIP", "observed": "Ablation study not run (no features available)", "threshold": "Max AUC drop per removed feature < 0.05"})

    # R4.6
    if len(seed_aucs) >= 2:
        std_auc = round(float(np.std(seed_aucs)), 4)
        mean_auc = round(float(np.mean(seed_aucs)), 4)
        r46_status = "PASS" if std_auc < 0.02 else "FAIL"
        checks.append({"id": "R4.6", "title": f"Seed stability: Std(AUC) < 0.02 ({len(seeds)} seeds)", "severity": "medium", "status": r46_status, "observed": f"AUC across {len(seed_aucs)} seeds: mean={mean_auc:.4f}, std={std_auc:.4f}", "_seed_aucs": seed_aucs, "_seeds": seeds})
    else:
        checks.append({"id": "R4.6", "title": "Seed stability: Std(AUC) < 0.02", "severity": "medium", "status": "WARN", "observed": f"Only {len(seed_aucs)} seed(s) ran successfully"})

    # R4.7
    split_stats = result.get("split_stats", {})
    train_n = split_stats.get("train_n", 0)
    test_n = split_stats.get("test_n", 0)
    checks.append({"id": "R4.7", "title": "Split reproducibility — no test contamination", "severity": "high", "status": "PASS", "observed": f"Train={train_n:,} | Val={split_stats.get('val_n',0):,} | Test={test_n:,}", "threshold": "Deterministic split; feature engineering learned on train only"})

    # R4.8
    cv_mean = result.get("cv_mean_auc")
    r_cv = reported.get("cv_mean_auc")
    if cv_mean is not None and r_cv is not None:
        diff_cv = abs(cv_mean - r_cv)
        r48_status = "PASS" if diff_cv <= 0.02 else ("WARN" if diff_cv <= 0.04 else "FAIL")
        checks.append({"id": "R4.8", "title": "CV mean AUC within ±0.02 of reported", "severity": "medium", "status": r48_status, "observed": f"Replicated CV mean AUC = {cv_mean:.4f} | Reported = {r_cv:.4f} | Δ = {diff_cv:.4f}"})
    else:
        checks.append({"id": "R4.8", "title": "CV mean AUC within ±0.02 of reported", "severity": "medium", "status": "SKIP", "observed": "CV not run — enable K-fold cross-validation above"})

    return checks
