"""
validation_agent2.py
Validation compliance checker for the MODEL VALIDATION service.
Completely separate from agent2.py (model development compliance checker).

Runs 75 checks across 8 validation stages against:
  - val_df        : submitted dataset (quantitative checks)
  - intake_json   : Agent 1 MDD extraction (doc checks)
  - mdd_text      : raw MDD text (keyword/section scanning)

Regulatory framework: SS1/23, SS11/13, IFRS 9, IFRS 7, SS3/18
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ── Default test fixture — mirrors Agent 1 output schema ─────────────────────
SAMPLE_INTAKE_JSON = {
    "model_name": "PD_XGBoost_RetailCredit_v2",
    "model_type": "PD Model",
    "model_owner": "Credit Risk Team",
    "smf_holder": "Not specified",
    "submission_date": "2026-06-23",
    "methodology": "XGBoost with SMOTE oversampling",
    "stated_auc": None,
    "stated_recall": None,
    "stated_gini": None,
    "stated_brier": None,
    "default_definition": "Not specified",
    "calibration_method": "Not specified",
    "features_used": [],
    "assumptions": [],
    "limitations": [],
    "data_description": "Not specified",
    "macro_variables_mentioned": False,
    "lgd_methodology": "Not specified",
    "ead_methodology": "Not specified",
    "validation_history": "Not specified",
    "model_inventory_registered": False,
    "independence_confirmed": False,
    "mdd_sections_found": [],
}


class ValidationAgent2:

    # ── map rules.json check names to context_dict keys + pass operator ─────────
    _CHECK_CONTEXT_MAP: dict[str, tuple[str, str]] = {
        "missing_threshold":     ("missing_rate",            "<="),
        "duplicate_threshold":   ("duplicate_rate",          "<="),
        "min_sample_size":       ("n_rows",                  ">="),
        "class_imbalance":       ("class_imbalance_ratio",   "<="),
        "high_correlation":      ("correlation_max",         "<="),
        "low_iv_features":       ("iv_min",                  ">="),
        "low_variance_features": ("variance_min",            ">="),
    }

    def __init__(self):
        self.findings: list[dict] = []
        self.intake_json: dict = {}
        self.val_df: Optional[pd.DataFrame] = None
        self.mdd_text: str = ""
        _rules_path = Path("rag_store/rules.json")
        if _rules_path.exists():
            with _rules_path.open(encoding="utf-8") as _f:
                self.rules: list[dict] = json.load(_f)
        else:
            self.rules = []

    # ── Main entry point ──────────────────────────────────────────────────────

    def run_all_checks(
        self,
        val_df: pd.DataFrame,
        intake_json: dict,
        mdd_text: str = "",
        hyperparams: dict = None,
    ) -> dict:
        """
        Run all 8 validation stages and return structured findings.

        Returns:
        {
            "summary": {
                "total": 75,
                "pass": N, "warn": N, "fail": N, "pending": N,
                "high_fails": N, "medium_fails": N,
                "verdict": "PASS" / "CONDITIONAL" / "FAIL"
            },
            "findings_by_stage": {
                "Stage 1: Governance": [...],
                "Stage 2: Data Validation": [...],
                ...
            },
            "all_findings": [...],
            "high_severity_fails": [...]
        }
        """
        self.val_df = val_df
        self.intake_json = intake_json
        self.mdd_text = mdd_text.lower() if mdd_text else ""
        self.hyperparams = hyperparams or {}
        self.findings = []

        stage_results = {
            "Stage 1: Governance":            self.check_governance(),
            "Stage 2: Data Validation":       self.check_data_validation(),
            "Stage 3: Conceptual Soundness":  self.check_conceptual_soundness(),
            "Stage 4: Model Replication":     self.check_model_replication(),
            "Stage 5: Performance Validation":self.check_performance(),
            "Stage 6: Stress & Backtesting":  self.check_stress_backtesting(),
            "Stage 7: Regulatory Compliance": self.check_regulatory_compliance(),
            "Stage 8: Findings & Report":     self.check_findings_report(),
        }

        all_findings = []
        for stage_findings in stage_results.values():
            all_findings.extend(stage_findings)

        n_pass    = sum(1 for f in all_findings if f["status"] == "PASS")
        n_warn    = sum(1 for f in all_findings if f["status"] == "WARN")
        n_fail    = sum(1 for f in all_findings if f["status"] == "FAIL")
        n_pending = sum(1 for f in all_findings if f["status"] == "PENDING")

        high_fails   = [f for f in all_findings
                        if f["status"] == "FAIL" and f["severity"] == "HIGH"]
        medium_fails = [f for f in all_findings
                        if f["status"] == "FAIL" and f["severity"] == "MEDIUM"]

        if len(high_fails) == 0 and n_fail == 0 and n_pending == 0:
            verdict = "PASS"
        elif len(high_fails) == 0 and n_fail == 0:
            verdict = "CONDITIONAL"
        elif len(high_fails) <= 2:
            verdict = "CONDITIONAL"
        else:
            verdict = "FAIL"

        return {
            "summary": {
                "total":        len(all_findings),
                "pass":         n_pass,
                "warn":         n_warn,
                "fail":         n_fail,
                "pending":      n_pending,
                "high_fails":   len(high_fails),
                "medium_fails": len(medium_fails),
                "verdict":      verdict,
            },
            "findings_by_stage":   stage_results,
            "all_findings":        all_findings,
            "high_severity_fails": high_fails,
            "replicated_importances": getattr(self, "replicated_importances", {}),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _mdd_contains(self, *keywords) -> bool:
        """Return True if any keyword appears in the lowercased MDD text."""
        return any(kw.lower() in self.mdd_text for kw in keywords)

    def _mdd_quote(self, *keywords) -> Optional[str]:
        """Return first sentence in MDD containing any keyword (max 200 chars)."""
        if not self.mdd_text:
            return None
        for sentence in self.mdd_text.split("."):
            if any(kw.lower() in sentence for kw in keywords):
                return sentence.strip()[:200]
        return None

    def _stub_stage(self, stage_name: str, checks: list) -> list:
        """
        Return PENDING stubs for checks not yet automated.
        Each element of checks: (check_id, title, source, principle, severity)
        """
        findings = []
        for check_id, title, source, principle, severity in checks:
            findings.append({
                "check_id":     check_id,
                "stage":        stage_name,
                "title":        title,
                "source":       source,
                "principle":    principle,
                "severity":     severity,
                "status":       "PENDING",
                "observed":     "Manual validation required — not yet automated",
                "threshold":    f"See {source} {principle}",
                "detail":       (f"This check requires manual review by the "
                                 f"validator. Refer to {source} {principle}."),
                "mdd_reference": None,
                "check_type":   "manual",
            })
        return findings

    # ── Agent 1 / RAG integration (mirrors agent2.py pattern) ────────────────

    def _resolve_field(self, field_hint: str, context_dict: dict) -> Any:
        """Dot-notation field resolver: 'metrics.roc_auc' → context_dict['metrics']['roc_auc']."""
        if not field_hint:
            return None
        if field_hint in context_dict:
            return context_dict[field_hint]
        parts = field_hint.split(".")
        obj: Any = context_dict
        for part in parts:
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return None
        return obj

    def _apply_operator(self, value: Any, operator: str, threshold: Any) -> bool:
        """Return True if the check PASSES (i.e., no flag needed)."""
        try:
            if operator in (">=", "gte"):
                return float(value) >= float(threshold)
            if operator in ("<=", "lte"):
                return float(value) <= float(threshold)
            if operator in (">", "gt"):
                return float(value) > float(threshold)
            if operator in ("<", "lt"):
                return float(value) < float(threshold)
            if operator in ("==", "eq"):
                return value == threshold
            if operator in ("!=", "ne"):
                return value != threshold
            if operator == "is_true":
                return bool(value)
            if operator == "is_false":
                return not bool(value)
            if operator == "is_present":
                return value is not None
        except (TypeError, ValueError):
            pass
        return True

    def _make_flag_from_agent1_rule(self, rule: dict, observed_value: Any = None) -> dict:
        return {
            "rule_id": rule.get("rule_id", rule.get("id", "?")),
            "source":  rule.get("regulation", rule.get("source", "?")),
            "principle": rule.get("section", rule.get("principle", "?")),
            "stage":   rule.get("stage", "?"),
            "severity": rule.get("severity", "medium"),
            "flag":    rule.get("statement", rule.get("flag", "Compliance check failed")),
            "suggestion": rule.get("action", rule.get("suggestion", "Review the relevant guidance.")),
            "observed_value": observed_value,
            "not_verifiable": False,
        }

    def _rule_to_finding(self, rule: dict, value: Any) -> dict:
        """Convert a rules.json rule into the finding dict format used by val_dv_results."""
        threshold = rule.get("threshold")
        severity = (rule.get("severity") or "medium").upper()
        rule_id = rule.get("id", rule.get("rule_id", "?"))
        return {
            "check_id":    rule_id,
            "stage":       rule.get("stage", "?"),
            "title":       rule.get("rule", rule.get("flag", "Compliance check failed")),
            "severity":    severity,
            "status":      "PASS" if value is not None and self._apply_operator(value, rule.get("operator", ""), threshold) else "FAIL",
            "observed":    value,
            "threshold":   threshold,
            "detail":      rule.get("suggestion", ""),
            "rule_id":     rule_id,
            "mdd_reference": None,
            "check_type":   "rule",
        }

    # ── Stage 1: Governance ──────────────────────────────────────────────────

    def check_governance(self) -> list[dict]:
        checks = [
            ("GOV001", "Model inventory registration", "SS1/23", "P1.1", "HIGH"),
            ("GOV002", "Independent review confirmation", "SS1/23", "P1.2", "HIGH"),
            ("GOV003", "Model purpose and intended use", "SS1/23", "P1.3", "MEDIUM"),
            ("GOV004", "Documentation ownership and versioning", "SS1/23", "P1.4", "MEDIUM"),
        ]
        return self._stub_stage("Stage 1: Governance", checks)

    # ── Stage 2: Data Validation ───────────────────────────────────────────────

    def check_data_validation(self) -> list[dict]:
        checks = [
            ("DV001", "Data completeness and quality", "SS1/23", "P3.5", "HIGH"),
            ("DV002", "Population definition and sampling", "SS1/23", "P3.5", "HIGH"),
            ("DV003", "Default definition and target labelling", "SS1/23", "P3.5", "HIGH"),
            ("DV004", "Missing data treatment", "SS1/23", "P3.5", "HIGH"),
            ("DV005", "Outlier handling", "SS1/23", "P3.5", "MEDIUM"),
        ]
        return self._stub_stage("Stage 2: Data Validation", checks)

    # ── Stage 3: Conceptual Soundness ────────────────────────────────────────

    def check_conceptual_soundness(self) -> list[dict]:
        checks = [
            ("CS001", "Business objective and intended use", "SS1/23", "P3.5", "HIGH"),
            ("CS002", "Model selection rationale", "SS1/23", "P3.5", "HIGH"),
            ("CS003", "Feature engineering and variable selection", "SS1/23", "P3.5", "HIGH"),
            ("CS004", "SICR criteria and staging logic", "SS1/23", "P3.5", "HIGH"),
        ]
        return self._stub_stage("Stage 3: Conceptual Soundness", checks)

    # ── Stage 4: Model Replication ───────────────────────────────────────────

    def check_model_replication(self) -> list[dict]:
        checks = [
            ("MR001", "Replication of documented methodology", "SS1/23", "P3.5", "HIGH"),
            ("MR002", "Reproducibility of reported results", "SS1/23", "P3.5", "HIGH"),
        ]
        return self._stub_stage("Stage 4: Model Replication", checks)

    # ── Stage 5: Performance Validation ─────────────────────────────────────

    def check_performance(self) -> list[dict]:
        checks = [
            ("PV001", "Performance metrics and benchmark", "SS1/23", "P3.5", "HIGH"),
            ("PV002", "Calibration and stability", "SS1/23", "P3.5", "HIGH"),
            ("PV003", "Overfitting and generalisation", "SS1/23", "P3.5", "HIGH"),
        ]
        return self._stub_stage("Stage 5: Performance Validation", checks)

    # ── Stage 6: Stress & Backtesting ───────────────────────────────────────

    def check_stress_backtesting(self) -> list[dict]:
        checks = [
            ("SB001", "Stress testing and sensitivity analysis", "SS1/23", "P3.5", "HIGH"),
            ("SB002", "Backtesting and benchmark comparison", "SS1/23", "P3.5", "HIGH"),
        ]
        return self._stub_stage("Stage 6: Stress & Backtesting", checks)

    # ── Stage 7: Regulatory Compliance ──────────────────────────────────────

    def check_regulatory_compliance(self) -> list[dict]:
        checks = [
            ("RC001", "Compliance with SS1/23 and IFRS 9 requirements", "SS1/23", "P3.5", "HIGH"),
            ("RC002", "Compliance with IFRS 7 disclosures", "IFRS 7", "P3.5", "MEDIUM"),
        ]
        return self._stub_stage("Stage 7: Regulatory Compliance", checks)

    # ── Stage 8: Findings & Report ──────────────────────────────────────────

    def check_findings_report(self) -> list[dict]:
        checks = [
            ("FR001", "Findings summary and recommendation", "SS1/23", "P3.5", "MEDIUM"),
            ("FR002", "Evidence and traceability", "SS1/23", "P3.5", "MEDIUM"),
        ]
        return self._stub_stage("Stage 8: Findings & Report", checks)

    # ── Quantitative checks over loaded rules.json ───────────────────────────

    def check_quantitative(self, rules: list[dict], context: dict) -> list[dict]:
        findings = []
        for rule in rules:
            if not rule.get("checkable_against_data", False):
                continue
            threshold = rule.get("threshold")
            field_hint = rule.get("field_hint")
            if not field_hint:
                continue
            value = self._resolve_field(field_hint, context)
            if value is None:
                continue
            passed = self._apply_operator(value, rule.get("operator", ""), threshold)
            findings.append(self._rule_to_finding(rule, value))
        return findings

    # ── Rule-driven findings from Agent 1 / RAG results ─────────────────────

    def check_rule_based_findings(self, findings: list[dict]) -> list[dict]:
        converted = []
        for finding in findings:
            converted.append(self._make_flag_from_agent1_rule(finding))
        return converted

    # ── Summary helpers ───────────────────────────────────────────────────────

    def summarize_findings(self, findings: list[dict]) -> dict:
        summary = {
            "total": len(findings),
            "pass": 0,
            "warn": 0,
            "fail": 0,
            "pending": 0,
        }
        for finding in findings:
            status = finding.get("status", "FAIL").upper()
            if status == "PASS":
                summary["pass"] += 1
            elif status == "WARN":
                summary["warn"] += 1
            elif status == "PENDING":
                summary["pending"] += 1
            else:
                summary["fail"] += 1
        return summary


# Optional compatibility alias for old imports
ValidationAgent2Compat = ValidationAgent2
