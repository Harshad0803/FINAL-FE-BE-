import json
import io
import base64
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Credit-Risk-Poc-main"))

from train import split_data, train_model
from preprocessing import prepare_data
from utils import detect_column_types
from sklearn.ensemble import RandomForestClassifier
import importlib.util

spec = importlib.util.spec_from_file_location("streamlit_eval", ROOT.parent / "Credit-Risk-Poc-main" / "evaluate.py")
streamlit_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(streamlit_eval)

legacy_spec = importlib.util.spec_from_file_location("legacy_eval", ROOT / "evaluate.py")
legacy_eval = importlib.util.module_from_spec(legacy_spec)
legacy_spec.loader.exec_module(legacy_eval)

sample = pd.read_csv(ROOT / "demo_data" / "clean_portfolio.csv")
target_col = "default"
col_types = detect_column_types(sample)
X_all, y_all, _, prep_report, _ = prepare_data(sample, col_types, target_col)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_all, y_all, test_size=0.2, val_size=0.1, task_type="binary")
model = RandomForestClassifier(n_estimators=50, random_state=42)
pipeline, _, _ = train_model(X_train, y_train, col_types, target_col, prep_report, model, task_type="binary")

X_eval = X_test.copy()
y_true = y_test.values
y_pred = pipeline.predict(X_eval)
y_proba = pipeline.predict_proba(X_eval)
arr = np.asarray(y_proba)
if arr.ndim == 2 and arr.shape[1] >= 2:
    scores = arr[:, 1]
elif arr.ndim == 2 and arr.shape[1] == 1:
    scores = arr[:, 0]
else:
    scores = arr

streamlit_metrics = streamlit_eval.compute_binary_metrics(y_true, y_pred, scores, threshold=0.5)
streamlit_roc = legacy_eval.compute_roc_curve(y_true, scores) if hasattr(legacy_eval, 'compute_roc_curve') else []
streamlit_pr = legacy_eval.compute_pr_curve(y_true, scores) if hasattr(legacy_eval, 'compute_pr_curve') else []
streamlit_thresh = legacy_eval.compute_threshold_analysis(y_true, scores) if hasattr(legacy_eval, 'compute_threshold_analysis') else []
streamlit_gain = legacy_eval.compute_gain_chart(y_true, scores) if hasattr(legacy_eval, 'compute_gain_chart') else []
streamlit_score = legacy_eval.compute_score_distribution(y_true, scores) if hasattr(legacy_eval, 'compute_score_distribution') else []
streamlit_hetero = streamlit_eval.compute_heteroscedasticity_check(y_true, scores, task_type="binary") if hasattr(streamlit_eval, 'compute_heteroscedasticity_check') else {}
streamlit_temporal = legacy_eval.compute_temporal_stability_summary(pd.date_range("2023-01-01", periods=len(y_true), freq="MS"), y_true, scores, freq="Monthly") if hasattr(legacy_eval, 'compute_temporal_stability_summary') else {}

# Build payload equivalent to the frontend contract
csv_text = pd.concat([X_eval.reset_index(drop=True), pd.Series(y_true, name=target_col).reset_index(drop=True)], axis=1).to_csv(index=False)
buf = io.BytesIO()
joblib.dump(pipeline, buf)
buf.seek(0)
model_b64 = base64.b64encode(buf.read()).decode("utf-8")

resp = requests.post(
    "http://127.0.0.1:8000/models/evaluate",
    data={"model_artifact": model_b64, "csv_text": csv_text, "target_col": target_col, "threshold": 0.5},
    timeout=120,
)
resp.raise_for_status()
api_payload = resp.json()

comparison = {
    "metrics": {
        "accuracy": [streamlit_metrics["accuracy"], api_payload["evaluation_metrics"]["accuracy"]],
        "precision": [streamlit_metrics["precision"], api_payload["evaluation_metrics"]["precision"]],
        "recall": [streamlit_metrics["recall"], api_payload["evaluation_metrics"]["recall"]],
        "f1": [streamlit_metrics["f1"], api_payload["evaluation_metrics"]["f1"]],
        "roc_auc": [streamlit_metrics["roc_auc"], api_payload["evaluation_metrics"]["roc_auc"]],
        "pr_auc": [streamlit_metrics["pr_auc"], api_payload["evaluation_metrics"]["pr_auc"]],
        "ks": [streamlit_metrics.get("ks_statistic"), api_payload["evaluation_metrics"].get("ks_statistic")],
    },
    "artifact_counts": {
        "roc_points": [len(streamlit_roc), len(api_payload["evaluation_data"]["roc_curve"])],
        "pr_points": [len(streamlit_pr), len(api_payload["evaluation_data"]["pr_curve"])],
        "threshold_rows": [len(streamlit_thresh), len(api_payload["evaluation_data"]["threshold_analysis"])],
        "gain_rows": [len(streamlit_gain), len(api_payload["evaluation_data"]["gain_chart"])],
        "score_rows": [len(streamlit_score), len(api_payload["evaluation_data"]["score_distribution"])],
    },
    "react_contract": {
        "uses_evaluation_metrics": "evaluation_metrics" in api_payload,
        "uses_evaluation_data": "evaluation_data" in api_payload,
        "threshold": api_payload["evaluation_data"]["threshold"],
    },
}

print(json.dumps(comparison, indent=2))
