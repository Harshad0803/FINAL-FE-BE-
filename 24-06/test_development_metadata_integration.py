import json
import uuid
from pathlib import Path

import persistence


def test_duplicate_and_profile_fields_propagate():
    store = {"development": [], "models": [], "data_sources": [], "validation": [], "history": []}

    persistence.register_development_model(
        store=store,
        business_model_name="TEST_DUP",
        estimator_name="XGBoost",
        intake_payload={},
        dataset_payload={
            "file_name": "test.csv",
            "record_count": 500,
            "column_count": 30,
            "numeric_column_count": 22,
            "categorical_column_count": 8,
            "missing_value_count": 12,
            "missing_value_pct": 0.024,
            "duplicate_record_count": 3,
            "class_distribution": {"0": 480, "1": 20},
        },
        user="tester",
    )

    assert len(store["development"]) == 1
    entry = store["development"][0]
    assert entry["numeric_column_count"] == 22
    assert entry["categorical_column_count"] == 8
    assert entry["missing_value_count"] == 12
    assert abs(float(entry["missing_value_pct"]) - 0.024) < 1e-6
    assert entry["duplicate_record_count"] == 3
    assert entry["class_distribution"]["1"] == 20


def test_feature_engineering_persisted_when_present():
    store = {"development": [], "models": [], "data_sources": [], "validation": [], "history": []}

    fe_summary = {"applied": True, "generated_columns": ["f_new_1", "f_new_2"]}

    persistence.register_development_model(
        store=store,
        business_model_name="TEST_FE",
        estimator_name="LightGBM",
        intake_payload={"feature_engineering_summary": fe_summary},
        dataset_payload={"file_name": "fe.csv", "record_count": 100},
        user="tester",
    )

    entry = store["development"][0]
    assert entry.get("feature_engineering_summary") == fe_summary


def test_explainability_artifact_attaches_and_regenerates_report(tmp_path):
    # register a development model and write last_development_registered.json
    store = {"development": [], "models": [], "data_sources": [], "validation": [], "history": []}
    result = persistence.register_development_model(
        store=store,
        business_model_name="TEST_EXPL",
        estimator_name="LightGBM",
        intake_payload={"model_owner": "tester"},
        dataset_payload={"file_name": "expl.csv", "record_count": 50},
        user="tester",
    )

    model_id = result["model_id"]
    model_name = result["model_name"]

    # craft a fake explainability payload and log it via persistence.log_event
    payload = {
        "feature_importance": [{"Feature": "A", "Importance": 1.0}],
        "shap": {"shap_available": False},
        "summary": "Test explainability",
    }
    run_id = persistence.log_event(persistence.DEV_PIPELINE_LOG, stage="explainability", payload={"method": "feature_importance"}, full_payload=payload)

    # write last_development_registered.json pointing at our model
    last_path = persistence.APP_DATA_DIR / "last_development_registered.json"
    last_path.write_text(json.dumps({"model_id": model_id, "model_name": model_name}), encoding="utf-8")

    # apply latest explainability to this store
    updated = persistence.apply_latest_explainability_to_last_registered_model(store)
    assert updated is not None
    entry = next((m for m in store["development"] if m.get("model_id") == model_id), None)
    assert entry is not None
    assert entry.get("feature_importances") is not None
    assert entry.get("top_model_drivers") is not None
    # documentation_path should point to a generated PDF
    assert entry.get("documentation_path") and entry.get("documentation_path").endswith(".pdf")