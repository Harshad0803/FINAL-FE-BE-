import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import main as backend
import persistence
import csv
import json


def _ensure_single_training_artifact(model_name: str):
    """Remove existing training log rows/artifacts for model_name and write a
    single fresh training artifact row for the given model_name.
    """
    log_path = persistence.APP_DATA_DIR / persistence.DEV_PIPELINE_LOG
    rows = persistence.read_log(persistence.DEV_PIPELINE_LOG)
    kept = []
    for r in rows:
        summ = r.get('summary') or {}
        if isinstance(summ, dict) and summ.get('model_name') == model_name:
            # delete artifact file if present
            ap = r.get('artifact_path')
            if ap:
                p = persistence.BACKEND_DIR / ap
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            continue
        kept.append(r)

    # rewrite CSV keeping non-matching rows
    if kept:
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["run_id", "timestamp", "stage", "summary", "artifact_path"])
            writer.writeheader()
            for r in kept:
                row = {k: (json.dumps(r[k]) if k == 'summary' and isinstance(r[k], dict) else r.get(k, '')) for k in ["run_id", "timestamp", "stage", "summary", "artifact_path"]}
                writer.writerow(row)
    else:
        # remove file entirely if no rows kept
        try:
            if log_path.exists():
                log_path.unlink()
        except Exception:
            pass

    # now create a single fresh artifact for this model_name
    persistence.log_event(persistence.DEV_PIPELINE_LOG, stage="training", payload={"model_name": model_name}, full_payload={"model_name": model_name})


def test_replication_model_resolution_keeps_business_identity_and_estimator_separate():
    business_name, estimator_name = backend.resolve_replication_model_inputs(
        model_name="PD_Model_Internal_v1",
        algorithm="XGBoost",
    )

    assert business_name == "PD_Model_Internal_v1"
    assert estimator_name == "XGBoost"


def test_register_validation_run_creates_validation_record_with_val_id_and_tracks_history():
    store = {"development": [], "models": [], "data_sources": [], "validation": [], "history": []}

    first_result = persistence.register_validation_run(
        store=store,
        business_model_name="PD_Model_Internal_v1",
        estimator_name="XGBoost",
        intake_payload={
            "model_type": "PD (Probability of Default)",
            "model_owner": "Alice",
            "owning_team": "Retail Credit",
            "model_version": "2.0",
            "regulatory_framework": "SR 11-7",
        },
        dataset_payload={
            "file_name": "Demo B dataset.csv",
            "storage_reference": "Demo B dataset.csv",
            "source_type": "file",
            "purpose": "Validation dataset",
            "record_count": 3180,
            "column_count": 13,
            "target_variable": "default",
            "uploaded_by": "Alice",
        },
        target_col="default",
        user="Alice",
    )

    second_result = persistence.register_validation_run(
        store=store,
        business_model_name="PD_Model_Internal_v1",
        estimator_name="XGBoost",
        intake_payload={
            "model_type": "PD (Probability of Default)",
            "model_owner": "Alice",
            "owning_team": "Retail Credit",
            "model_version": "2.0",
            "regulatory_framework": "SR 11-7",
        },
        dataset_payload={
            "file_name": "Demo B dataset.csv",
            "storage_reference": "Demo B dataset.csv",
            "source_type": "file",
            "purpose": "Validation dataset",
            "record_count": 3180,
            "column_count": 13,
            "target_variable": "default",
            "uploaded_by": "Alice",
        },
        target_col="default",
        user="Alice",
    )

    assert first_result["model_id"] == second_result["model_id"]
    assert first_result["model_id"].startswith("VAL-")
    assert not first_result["model_id"].startswith("MOD-")
    assert len(store["development"]) == 0
    assert len(store["validation"]) == 1
    assert len(store["data_sources"]) == 1
    assert len([entry for entry in store["history"] if entry.get("event") == "Validation Run Started"]) == 2
    assert first_result["algorithm"] == "XGBoost"


def test_register_validation_run_updates_placeholder_metadata_from_real_intake():
    store = {
        "models": [{
            "model_id": "MOD-001",
            "model_name": "PD_Model_Internal_v1",
            "model_type": "PD (Probability of Default)",
            "business_purpose": "Credit risk model.",
            "model_owner": "Unknown",
            "business_unit": "Risk",
            "model_version": "v1.0",
            "regulatory_framework": "Internal",
            "algorithm": "XGBoost",
            "status": "Under Validation",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        }],
        "data_sources": [],
        "validation": [],
        "history": [],
    }

    persistence.register_validation_run(
        store=store,
        business_model_name="PD_Model_Internal_v1",
        estimator_name="XGBoost",
        intake_payload={
            "model_owner": "Sarah Chen",
            "owning_team": "Retail Credit Risk",
            "model_version": "2.1",
            "regulatory_framework": "SR 11-7",
            "model_type": "PD (Probability of Default)",
            "model_purpose": "Retail portfolio PD model",
        },
        dataset_payload={
            "file_name": "validation_dataset.csv",
            "storage_reference": "validation_dataset.csv",
            "source_type": "file",
            "purpose": "Validation dataset",
            "record_count": 1000,
            "column_count": 25,
            "target_variable": "default_flag",
            "uploaded_by": "Sarah Chen",
        },
        target_col="default_flag",
        user="Sarah Chen",
    )

    model = store["models"][0]
    assert model["model_owner"] == "Sarah Chen"
    assert model["business_unit"] == "Retail Credit Risk"
    assert model["regulatory_framework"] == "SR 11-7"
    assert model["model_version"] == "2.1"


def test_development_and_validation_register_separately_with_related_reference():
    store = {"development": [], "validation": [], "models": [], "data_sources": [], "history": []}

    # create a training artifact so registration can link to it
    _ensure_single_training_artifact("PD_Model_Internal_v1")

    development_result = persistence.register_development_model(
        store=store,
        business_model_name="PD_Model_Internal_v1",
        estimator_name="XGBoost",
        intake_payload={
            "model_type": "PD (Probability of Default)",
            "model_owner": "Alice",
            "business_unit": "Retail Credit",
            "model_version": "2.0",
            "model_purpose": "Retail portfolio PD model",
            "development_date": "2026-06-01",
            "status": "In Development",
            "documentation_path": "docs/pd_model_internal_v1_mdd.pdf",
        },
        dataset_payload={
            "file_name": "development_dataset.csv",
            "storage_reference": "development_dataset.csv",
            "source_type": "file",
            "purpose": "Development dataset",
            "record_count": 5000,
            "column_count": 15,
            "target_variable": "default",
            "uploaded_by": "Alice",
        },
        user="Alice",
    )

    validation_result = persistence.register_validation_run(
        store=store,
        business_model_name="PD_Model_Internal_v1",
        estimator_name="XGBoost",
        intake_payload={
            "model_type": "PD (Probability of Default)",
            "model_owner": "Alice",
            "owning_team": "Retail Credit",
            "model_version": "2.0",
            "regulatory_framework": "SR 11-7",
            "model_purpose": "Retail portfolio PD model",
            "mdd_document_path": "docs/pd_model_internal_v1_mdd.pdf",
        },
        dataset_payload={
            "file_name": "validation_dataset.csv",
            "storage_reference": "validation_dataset.csv",
            "source_type": "file",
            "purpose": "Validation dataset",
            "record_count": 1000,
            "column_count": 25,
            "target_variable": "default",
            "uploaded_by": "Alice",
        },
        target_col="default",
        user="Alice",
    )

    assert development_result["model_id"].startswith("MOD-")
    assert validation_result["model_id"].startswith("VAL-")
    assert validation_result["model_id"] != development_result["model_id"]
    assert len(store["development"]) == 1
    assert len(store["validation"]) == 1
    development_entry = store["development"][0]
    validation_entry = store["validation"][0]
    assert development_entry["model_id"] == development_result["model_id"]
    assert validation_entry["related_model_id"] == development_result["model_id"]
    assert development_entry["status"] == "In Development"
    assert validation_entry["status"] == "Under Validation"
    assert development_entry["documentation_path"].endswith(".pdf")
    assert validation_entry["documentation_path"] == "docs/pd_model_internal_v1_mdd.pdf"
    assert development_entry is not validation_entry
    assert len(store["data_sources"]) == 2


def test_register_development_model_generates_real_pdf_documentation_for_model_report():
    store = {"models": [], "data_sources": [], "validation": [], "history": []}

    # ensure a single training artifact exists for PD_RetailCredit_v2
    _ensure_single_training_artifact("PD_RetailCredit_v2")

    persistence.register_development_model(
        store=store,
        business_model_name="PD_RetailCredit_v2",
        estimator_name="LightGBM",
        intake_payload={
            "model_owner": "Harshad",
            "business_unit": "Risk",
            "model_purpose": "Retail portfolio PD model",
            "model_version": "v2.1.0",
            "model_type": "PD (Probability of Default)",
            "development_date": "2026-08-11",
            "status": "In Development",
        },
        dataset_payload={
            "file_name": "retail_pd_dataset.csv",
            "storage_reference": "retail_pd_dataset.csv",
            "source_type": "file",
            "purpose": "Development dataset",
            "record_count": 32779,
            "column_count": 36,
            "target_variable": "default_flag",
            "uploaded_by": "Harshad",
        },
        user="Harshad",
    )

    # ensure a single training artifact exists for PD_RetailCredit_v2
    _ensure_single_training_artifact("PD_RetailCredit_v2")

    model = store["models"][0]
    assert model["model_name"] == "PD_RetailCredit_v2"
    assert model["documentation_path"].endswith(".pdf")
    assert "demo_data/" not in model["documentation_path"]
    assert "mdd" not in model["documentation_path"].lower()
    assert (persistence.BACKEND_DIR / model["documentation_path"]).exists()


def test_register_development_model_persists_training_and_evaluation_metadata():
    store = {"models": [], "data_sources": [], "validation": [], "history": []}
    # ensure a single training artifact exists for PD_RetailCredit_v4
    _ensure_single_training_artifact("PD_RetailCredit_v4")

    persistence.register_development_model(
        store=store,
        business_model_name="PD_RetailCredit_v4",
        estimator_name="LightGBM",
        intake_payload={
            "model_owner": "Harshad",
            "business_unit": "Risk",
            "model_purpose": "Retail portfolio PD model",
            "model_version": "v4.0",
            "model_type": "PD (Probability of Default)",
            "development_date": "2026-09-01",
            "status": "In Development",
            "training_info": {
                "test_size": 0.15,
                "val_size": 0.15,
                "use_cv": True,
                "use_hyperopt": False,
            },
            "evaluation_metrics": {
                "roc_auc": 0.93,
                "accuracy": 0.88,
                "f1": 0.79,
            },
        },
        dataset_payload={
            "file_name": "retail_pd_dataset_v4.csv",
            "storage_reference": "retail_pd_dataset_v4.csv",
            "source_type": "file",
            "purpose": "Development dataset",
            "record_count": 45000,
            "column_count": 38,
            "target_variable": "default_flag",
            "uploaded_by": "Harshad",
        },
        user="Harshad",
    )

    # ensure a single training artifact exists for PD_RetailCredit_v4
    _ensure_single_training_artifact("PD_RetailCredit_v4")

    model = store["models"][0]
    assert model["training_info"]["use_cv"] is True
    assert model["evaluation_metrics"]["roc_auc"] == 0.93
    assert model["evaluation_metrics"]["accuracy"] == 0.88
    assert model["documentation_path"].endswith(".pdf")
    assert (persistence.BACKEND_DIR / model["documentation_path"]).exists()
    # Verify the generated PDF includes run-specific dataset and metric fields
    pdf_path = persistence.BACKEND_DIR / model["documentation_path"]
    content = pdf_path.read_bytes()
    assert b"Dataset Record Count" in content
    assert b"Roc Auc" in content or b"Roc_auc" in content
    assert b"Cross-Validation: Enabled" in content


def test_validation_run_leaves_development_record_unchanged_and_uses_validation_id():
    store = {
        "development": [{
            "model_id": "MOD-008",
            "model_name": "PD_RetailCredit_v3",
            "model_type": "PD (Probability of Default)",
            "business_purpose": "Retail portfolio PD model",
            "model_owner": "Harshad",
            "business_unit": "Risk",
            "model_version": "v3.0",
            "regulatory_framework": "Internal",
            "algorithm": "LightGBM",
            "documentation_path": "app_data/model_reports/PD_RetailCredit_v3_model_report.pdf",
            "development_date": "2026-08-11",
            "status": "In Development",
            "created_at": "2026-08-11T00:00:00",
            "updated_at": "2026-08-11T00:00:00",
        }],
        "models": [{
            "model_id": "MOD-008",
            "model_name": "PD_RetailCredit_v3",
            "model_type": "PD (Probability of Default)",
            "business_purpose": "Retail portfolio PD model",
            "model_owner": "Harshad",
            "business_unit": "Risk",
            "model_version": "v3.0",
            "regulatory_framework": "Internal",
            "algorithm": "LightGBM",
            "documentation_path": "app_data/model_reports/PD_RetailCredit_v3_model_report.pdf",
            "development_date": "2026-08-11",
            "status": "In Development",
            "created_at": "2026-08-11T00:00:00",
            "updated_at": "2026-08-11T00:00:00",
        }],
        "data_sources": [],
        "validation": [],
        "history": [],
    }

    result = persistence.register_validation_run(
        store=store,
        business_model_name="PD_RetailCredit_v3",
        estimator_name="LightGBM",
        intake_payload={
            "model_id": "MOD-008",
            "model_type": "PD (Probability of Default)",
            "model_owner": "Harshad",
            "owning_team": "Risk",
            "model_version": "v3.0",
            "regulatory_framework": "Internal",
            "model_purpose": "Retail portfolio PD model",
            "mdd_document_path": "demo_data/clean_mdd.txt",
        },
        dataset_payload={
            "file_name": "retail_pd_validation.csv",
            "storage_reference": "retail_pd_validation.csv",
            "source_type": "file",
            "purpose": "Validation dataset",
            "record_count": 1000,
            "column_count": 25,
            "target_variable": "default_flag",
            "uploaded_by": "Harshad",
        },
        target_col="default_flag",
        user="Harshad",
    )

    assert result["model_id"].startswith("VAL-")
    assert result["model_id"] != "MOD-008"
    assert len(store["development"]) == 1
    development = store["development"][0]
    assert development["model_id"] == "MOD-008"
    assert development["status"] == "In Development"
    assert development["documentation_path"] == "app_data/model_reports/PD_RetailCredit_v3_model_report.pdf"
    validation = store["validation"][0]
    assert validation["model_id"].startswith("VAL-")
    assert validation["related_model_id"] == "MOD-008"
    assert validation["documentation_path"] == "demo_data/clean_mdd.txt"
    assert validation["overall_status"] == "Under Validation"


def test_development_and_validation_counts_change_independently():
    store = {"development": [], "validation": [], "models": [], "data_sources": [], "history": []}

    # ensure a single training artifact exists for PD_Count_A
    _ensure_single_training_artifact("PD_Count_A")

    persistence.register_development_model(
        store=store,
        business_model_name="PD_Count_A",
        estimator_name="XGBoost",
        intake_payload={
            "model_owner": "Alice",
            "business_unit": "Risk",
            "model_version": "1.0",
            "model_purpose": "Testing independence",
            "development_date": "2026-08-11",
            "status": "In Development",
        },
        dataset_payload={
            "file_name": "dev_a.csv",
            "storage_reference": "dev_a.csv",
            "source_type": "file",
            "purpose": "Development dataset",
            "record_count": 100,
            "column_count": 10,
            "target_variable": "default",
            "uploaded_by": "Alice",
        },
        user="Alice",
    )

    

    assert len(store["development"]) == 1
    assert len(store["validation"]) == 0

    persistence.register_validation_run(
        store=store,
        business_model_name="PD_Count_A",
        estimator_name="XGBoost",
        intake_payload={
            "model_id": store["development"][0]["model_id"],
            "model_owner": "Alice",
            "owning_team": "Risk",
            "model_version": "1.0",
            "regulatory_framework": "Internal",
            "model_purpose": "Testing independence",
            "mdd_document_path": "demo_data/clean_mdd.txt",
        },
        dataset_payload={
            "file_name": "val_a.csv",
            "storage_reference": "val_a.csv",
            "source_type": "file",
            "purpose": "Validation dataset",
            "record_count": 80,
            "column_count": 10,
            "target_variable": "default",
            "uploaded_by": "Alice",
        },
        target_col="default",
        user="Alice",
    )

    assert len(store["development"]) == 1
    assert len(store["validation"]) == 1

    # ensure a single training artifact exists for PD_Count_B
    _ensure_single_training_artifact("PD_Count_B")

    persistence.register_development_model(
        store=store,
        business_model_name="PD_Count_B",
        estimator_name="LightGBM",
        intake_payload={
            "model_owner": "Alice",
            "business_unit": "Risk",
            "model_version": "1.0",
            "model_purpose": "Testing independence",
            "development_date": "2026-08-11",
            "status": "In Development",
        },
        dataset_payload={
            "file_name": "dev_b.csv",
            "storage_reference": "dev_b.csv",
            "source_type": "file",
            "purpose": "Development dataset",
            "record_count": 200,
            "column_count": 12,
            "target_variable": "default",
            "uploaded_by": "Alice",
        },
        user="Alice",
    )


    assert len(store["development"]) == 2
    assert len(store["validation"]) == 1


def test_register_validation_run_keeps_doc_reference_for_demo_mdd():
    store = {"models": [], "data_sources": [], "validation": [], "history": []}

    first = persistence.register_validation_run(
        store=store,
        business_model_name="PD_XGBoost_RetailCredit_v2",
        estimator_name="XGBoost",
        intake_payload={
            "model_type": "PD (Probability of Default)",
            "model_owner": "Sarah Chen",
            "owning_team": "Retail Credit Risk",
            "model_version": "v2.1.0",
            "regulatory_framework": "SR 11-7",
            "demo_mode": "clean",
            "demo_label": "Demo A",
            "mdd_document_path": "demo_data/clean_mdd.txt",
        },
        dataset_payload={
            "file_name": "Demo A dataset.csv",
            "storage_reference": "Demo A dataset.csv",
            "source_type": "file",
            "purpose": "Validation dataset",
            "record_count": 120,
            "column_count": 14,
            "target_variable": "default",
            "uploaded_by": "Sarah Chen",
        },
        target_col="default",
        user="Sarah Chen",
    )

    second = persistence.register_validation_run(
        store=store,
        business_model_name="PD_XGBoost_RetailCredit_v2",
        estimator_name="XGBoost",
        intake_payload={
            "model_type": "PD (Probability of Default)",
            "model_owner": "Sarah Chen",
            "owning_team": "Retail Credit Risk",
            "model_version": "v2.1.0",
            "regulatory_framework": "SR 11-7",
            "demo_mode": "clean",
            "demo_label": "Demo A",
            "mdd_document_path": "demo_data/clean_mdd.txt",
        },
        dataset_payload={
            "file_name": "Demo A dataset.csv",
            "storage_reference": "Demo A dataset.csv",
            "source_type": "file",
            "purpose": "Validation dataset",
            "record_count": 120,
            "column_count": 14,
            "target_variable": "default",
            "uploaded_by": "Sarah Chen",
        },
        target_col="default",
        user="Sarah Chen",
    )

    assert first["model_id"] == second["model_id"]
    model = store["models"][0]
    assert model["documentation_path"] == "demo_data/clean_mdd.txt"
    assert sum(1 for item in store["models"] if item.get("documentation_path") == "demo_data/clean_mdd.txt") == 1


def test_register_validation_run_keeps_demo_b_reference_separate_from_dataset():
    store = {"models": [], "data_sources": [], "validation": [], "history": []}

    persistence.register_validation_run(
        store=store,
        business_model_name="PD_Model_Internal_v1",
        estimator_name="XGBoost",
        intake_payload={
            "model_type": "PD (Probability of Default)",
            "model_owner": "Unknown",
            "owning_team": "Risk",
            "model_version": "v1.0",
            "regulatory_framework": "Internal",
            "demo_mode": "flawed",
            "demo_label": "Demo B",
            "mdd_document_path": "demo_data/flawed_mdd.txt",
        },
        dataset_payload={
            "file_name": "Demo B dataset.csv",
            "storage_reference": "Demo B dataset.csv",
            "source_type": "file",
            "purpose": "Validation dataset",
            "record_count": 3180,
            "column_count": 13,
            "target_variable": "default",
            "uploaded_by": "Unknown",
        },
        target_col="default",
        user="Unknown",
    )

    model = store["models"][0]
    assert model["documentation_path"] == "demo_data/flawed_mdd.txt"
    assert any(item.get("file_name") == "Demo B dataset.csv" for item in store["data_sources"])
    assert sum(1 for item in store["data_sources"] if item.get("file_name") == "Demo B dataset.csv") == 1


def test_validation_record_is_distinct_from_development_record_and_keeps_ref_only():
    store = {"development": [], "validation": [], "models": [], "data_sources": [], "history": []}

    # ensure a single training artifact exists for this development model
    _ensure_single_training_artifact("PD_RetailCredit_v2")

    development = persistence.register_development_model(
        store=store,
        business_model_name="PD_RetailCredit_v2",
        estimator_name="XGBoost",
        intake_payload={
            "model_owner": "Harshad",
            "business_unit": "Risk",
            "model_version": "v2.0",
            "model_purpose": "Retail portfolio PD model",
            "development_date": "2026-08-11",
            "status": "In Development",
        },
        dataset_payload={
            "file_name": "pd_v2_dataset.csv",
            "storage_reference": "pd_v2_dataset.csv",
            "source_type": "file",
            "purpose": "Development dataset",
            "record_count": 500,
            "column_count": 20,
            "target_variable": "default_flag",
            "uploaded_by": "Harshad",
        },
        user="Harshad",
    )

    

    validation = persistence.register_validation_run(
        store=store,
        business_model_name="PD_RetailCredit_v2",
        estimator_name="XGBoost",
        intake_payload={
            "model_id": development["model_id"],
            "model_owner": "Harshad",
            "owning_team": "Risk",
            "model_version": "v2.0",
            "regulatory_framework": "Internal",
            "mdd_document_path": "demo_data/flawed_mdd.txt",
        },
        dataset_payload={
            "file_name": "validation_set.csv",
            "storage_reference": "validation_set.csv",
            "source_type": "file",
            "purpose": "Validation dataset",
            "record_count": 300,
            "column_count": 18,
            "target_variable": "default_flag",
            "uploaded_by": "Harshad",
        },
        target_col="default_flag",
        user="Harshad",
    )

    assert validation["model_id"].startswith("VAL-")
    assert validation["model_id"] != development["model_id"]
    assert len(store["development"]) == 1
    assert len(store["validation"]) == 1
    assert store["development"][0]["status"] == "In Development"
    assert store["development"][0]["documentation_path"].endswith(".pdf")
    assert store["validation"][0]["related_model_id"] == development["model_id"]
    assert store["validation"][0]["documentation_path"] == "demo_data/flawed_mdd.txt"
    assert store["development"][0] is not store["validation"][0]
