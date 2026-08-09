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
        return {
            "check_id": rule.get("id", "R?"),
            "title":    rule.get("flag", "Rule check failed"),
            "source":   rule.get("source", ""),
            "principle": rule.get("principle", ""),
            "severity": rule.get("severity", "medium"),
            "status":   "FAIL",
            "observed": str(round(value, 4)) if isinstance(value, float) else str(value),
            "threshold": str(threshold) if threshold is not None else "See rule",
            "detail":   rule.get("suggestion", "Review the relevant guidance."),
        }

    def check_rules_from_agent1(self, stage: str, context_dict: dict) -> list[dict]:
        """
        Evaluate RAG rules for the given stage against context_dict.
        Supports two rule formats:
          1. Rules with checkable_against_data=True + logic.field_hint  (future format)
          2. Rules with check + threshold using _CHECK_CONTEXT_MAP      (current rules.json)
        Returns a list of finding dicts in the same format as val_dv_results.
        """
        new_findings: list[dict] = []

        for rule in self.rules:
            if rule.get("stage") != stage:
                continue

            # Path 1: future-format rules with checkable_against_data + logic
            if rule.get("checkable_against_data", False):
                logic = rule.get("logic") or {}
                field_hint = logic.get("field_hint", "")
                if not field_hint:
                    continue
                value = self._resolve_field(field_hint, context_dict)
                operator = logic.get("operator", "")
                threshold = logic.get("threshold")
                if value is not None and not self._apply_operator(value, operator, threshold):
                    new_findings.append(self._rule_to_finding(rule, value))
                continue

            # Path 2: current rules.json format using check + threshold mapping
            check = rule.get("check", "")
            if check not in self._CHECK_CONTEXT_MAP:
                continue
            ctx_key, operator = self._CHECK_CONTEXT_MAP[check]
            value = context_dict.get(ctx_key)
            threshold = rule.get("threshold")
            if value is None or threshold is None:
                continue
            if not self._apply_operator(value, operator, threshold):
                new_findings.append(self._rule_to_finding(rule, value))

        return new_findings

    # ── Stage 1: Governance ───────────────────────────────────────────────────

    def check_governance(self) -> list:
        findings = []
        ij = self.intake_json

        # 1.1 Model purpose defined
        purpose_found = self._mdd_contains(
            "purpose", "intended use", "business objective", "scope"
        )
        findings.append({
            "check_id": "1.1",
            "stage": "Stage 1: Governance",
            "title": "Model Purpose and Intended Use Defined",
            "source": "SS1/23", "principle": "P1.1", "severity": "HIGH",
            "status": "PASS" if purpose_found else "FAIL",
            "observed": ("Purpose section found in MDD"
                         if purpose_found
                         else "No purpose/intended use section detected in MDD"),
            "threshold": "Purpose unambiguously defined in MDD",
            "detail": "SS1/23 P1.1 requires model purpose to be clearly documented",
            "mdd_reference": self._mdd_quote("purpose", "intended use"),
            "check_type": "doc",
        })

        # 1.2 Model inventory registered
        inventory = ij.get("model_inventory_registered", False)
        findings.append({
            "check_id": "1.2",
            "stage": "Stage 1: Governance",
            "title": "Model Registered in Model Inventory",
            "source": "SS1/23", "principle": "P1.2", "severity": "HIGH",
            "status": "PASS" if inventory else "WARN",
            "observed": ("Model inventory registration confirmed"
                         if inventory
                         else "Model inventory registration not confirmed in submitted docs"),
            "threshold": "Model inventory entry present with all mandatory fields",
            "detail": "SS1/23 P1.2 requires all models to be registered in firm model inventory",
            "mdd_reference": None,
            "check_type": "doc",
        })

        # 1.3 SMF holder assigned
        smf = ij.get("smf_holder", "Not specified")
        smf_found = smf not in ("Not specified", "", None)
        findings.append({
            "check_id": "1.3",
            "stage": "Stage 1: Governance",
            "title": "SMF Accountable Individual Assigned",
            "source": "SS1/23", "principle": "P2.2", "severity": "HIGH",
            "status": "PASS" if smf_found else "FAIL",
            "observed": f"SMF holder: {smf}",
            "threshold": "Named SMF holder documented as accountable",
            "detail": "SS1/23 P2.2 and SS11/13 §5.2 require annual SMF attestation",
            "mdd_reference": self._mdd_quote("smf", "senior management",
                                              "accountable", "attestation"),
            "check_type": "doc",
        })

        # 1.4 Independence confirmed
        independence = ij.get("independence_confirmed", False)
        findings.append({
            "check_id": "1.4",
            "stage": "Stage 1: Governance",
            "title": "Validation Independence Confirmed",
            "source": "SS1/23", "principle": "P4.1", "severity": "HIGH",
            "status": "PASS" if independence else "FAIL",
            "observed": ("Independence attested by validator"
                         if independence
                         else "Independence not confirmed"),
            "threshold": "Validation team independent of development team",
            "detail": "SS1/23 P4.1 — independence is a non-negotiable requirement",
            "mdd_reference": None,
            "check_type": "doc",
        })

        # 1.5 Regulatory methodology alignment
        reg_found = self._mdd_contains(
            "pd", "lgd", "ead", "ifrs 9", "irb", "probability of default"
        )
        findings.append({
            "check_id": "1.5",
            "stage": "Stage 1: Governance",
            "title": "Regulatory Methodology Alignment Documented",
            "source": "SS11/13", "principle": "§6", "severity": "HIGH",
            "status": "PASS" if reg_found else "WARN",
            "observed": ("PD/LGD/EAD methodology referenced in MDD"
                         if reg_found
                         else "No explicit PD/LGD/EAD methodology found in MDD"),
            "threshold": "Methodology aligned with SS11/13 and IFRS 9 definitions",
            "detail": "SS11/13 §6 requires PD calibration methodology to be documented",
            "mdd_reference": self._mdd_quote("pd", "lgd", "ead",
                                              "probability of default"),
            "check_type": "doc",
        })

        # 1.6 Assumptions and limitations documented
        assumptions = ij.get("assumptions", [])
        limitations = ij.get("limitations", [])
        assum_found = (
            len(assumptions) > 0
            or len(limitations) > 0
            or self._mdd_contains("assumption", "limitation")
        )
        findings.append({
            "check_id": "1.6",
            "stage": "Stage 1: Governance",
            "title": "Assumptions and Limitations Documented",
            "source": "SS1/23", "principle": "P3.5", "severity": "MEDIUM",
            "status": "PASS" if assum_found else "FAIL",
            "observed": (f"{len(assumptions)} assumptions, {len(limitations)} "
                         f"limitations extracted from MDD"
                         if assum_found
                         else "No assumptions or limitations section found in MDD"),
            "threshold": "All material assumptions listed with limitations acknowledged",
            "detail": "SS1/23 P3.5 requires full documentation of modelling assumptions",
            "mdd_reference": self._mdd_quote("assumption", "limitation"),
            "check_type": "doc",
        })

        # 1.7 Model scope defined
        scope_found = self._mdd_contains(
            "scope", "target population", "out of scope", "applicability"
        )
        findings.append({
            "check_id": "1.7",
            "stage": "Stage 1: Governance",
            "title": "Model Scope and Target Population Defined",
            "source": "SS1/23", "principle": "P3.1", "severity": "MEDIUM",
            "status": "PASS" if scope_found else "WARN",
            "observed": ("Scope/target population found in MDD"
                         if scope_found
                         else "No explicit scope or target population in MDD"),
            "threshold": "Scope clearly bounded; out-of-scope use cases identified",
            "detail": "SS1/23 P3.1 — model must not be applied outside intended scope",
            "mdd_reference": self._mdd_quote("scope", "target population"),
            "check_type": "doc",
        })

        # 1.8 Version history maintained
        version_found = self._mdd_contains(
            "version", "change log", "revision", "history"
        )
        findings.append({
            "check_id": "1.8",
            "stage": "Stage 1: Governance",
            "title": "Version History and Change Log Maintained",
            "source": "SS11/13", "principle": "§9.1", "severity": "MEDIUM",
            "status": "PASS" if version_found else "WARN",
            "observed": ("Version history found in MDD"
                         if version_found
                         else "No version history detected in MDD"),
            "threshold": "Version history complete; material changes triggered revalidation",
            "detail": "SS11/13 §9.1 requires documentation to be maintained and current",
            "mdd_reference": self._mdd_quote("version", "change log"),
            "check_type": "doc",
        })

        # 1.9 MDD completeness overall
        required_sections = [
            "purpose", "data", "methodology", "assumption",
            "limitation", "performance", "feature", "governance",
        ]
        found_sections   = [s for s in required_sections if self._mdd_contains(s)]
        missing_sections = [s for s in required_sections if s not in found_sections]
        mdd_complete     = len(missing_sections) <= 1
        findings.append({
            "check_id": "1.9",
            "stage": "Stage 1: Governance",
            "title": "MDD Completeness — All Required Sections Present",
            "source": "SS1/23", "principle": "P3.1", "severity": "HIGH",
            "status": ("PASS" if mdd_complete
                       else "WARN" if len(missing_sections) <= 2
                       else "FAIL"),
            "observed": (f"Found {len(found_sections)}/8 required sections. "
                         f"Missing: {missing_sections}"),
            "threshold": ("MDD covers: purpose, data, methodology, assumptions, "
                          "limitations, performance, features, governance"),
            "detail": ("SS1/23 P3.1 — MDD must be sufficient for an independent "
                       "third party to understand and replicate the model"),
            "mdd_reference": None,
            "check_type": "doc",
        })

        return findings  # 9 checks: 1.1–1.9

    # ── Stage 2: Data Validation ──────────────────────────────────────────────

    def check_data_validation(self) -> list:
        findings = []
        df = self.val_df
        ij = self.intake_json

        if df is None:
            return [{
                "check_id": "2.0",
                "stage": "Stage 2: Data Validation",
                "title": "Dataset Not Uploaded",
                "source": "SS11/13", "principle": "§10.4",
                "severity": "HIGH", "status": "FAIL",
                "observed": "No dataset uploaded",
                "threshold": "Dataset required for validation",
                "detail": "Upload submitted dataset in Stage 1 to run data checks",
                "mdd_reference": None, "check_type": "data",
            }]

        # 2.1 Row/col reconciliation
        findings.append({
            "check_id": "2.1",
            "stage": "Stage 2: Data Validation",
            "title": "Row / Column Reconciliation",
            "source": "SS11/13", "principle": "§10.4", "severity": "HIGH",
            "status": "PASS",
            "observed": f"{len(df):,} rows × {df.shape[1]} columns",
            "threshold": "Dataset loads successfully",
            "detail": ("Cross-reference row/col counts against developer's "
                       "reported profile report manually"),
            "mdd_reference": None, "check_type": "data",
        })

        # 2.2 Missing data rate
        missing_pct = df.isnull().mean()
        max_missing = missing_pct.max()
        worst_col   = missing_pct.idxmax()
        dv22 = ("FAIL" if max_missing > 0.20
                else "WARN" if max_missing > 0.10
                else "PASS")
        findings.append({
            "check_id": "2.2",
            "stage": "Stage 2: Data Validation",
            "title": "Missing Data Rate",
            "source": "SS1/23", "principle": "P3.2", "severity": "HIGH",
            "status": dv22,
            "observed": f"Max missing: {max_missing:.1%} in '{worst_col}'",
            "threshold": "< 20% per column",
            "detail": ("BLOCKER: " if dv22 == "FAIL" else "")
                      + "Columns > 20% missing are a regulatory blocker",
            "mdd_reference": None, "check_type": "data",
        })

        # 2.3 Default definition — data vs MDD cross-reference
        default_kw    = ["default", "bad", "charged_off", "write_off",
                         "target", "label", "dpd_90", "is_bad"]
        found_targets = [c for c in df.columns
                         if any(k in c.lower() for k in default_kw)]
        mdd_default   = ij.get("default_definition", "Not specified")
        mdd_has_def   = mdd_default not in ("Not specified", "", None)

        if found_targets and mdd_has_def:
            status_23 = "PASS"
            obs_23    = (f"Target column '{found_targets[0]}' found. "
                         f"MDD states: '{mdd_default}'")
        elif found_targets:
            status_23 = "WARN"
            obs_23    = (f"Target column '{found_targets[0]}' found but "
                         f"MDD does not state default definition explicitly")
        else:
            status_23 = "FAIL"
            obs_23    = "No default/target column detected in dataset"
        findings.append({
            "check_id": "2.3",
            "stage": "Stage 2: Data Validation",
            "title": "Default Definition Consistency",
            "source": "IFRS 9", "principle": "B5.5.28", "severity": "HIGH",
            "status": status_23, "observed": obs_23,
            "threshold": "Default = 90 DPD or documented alternative",
            "detail": ("Confirm default definition in writing with developer. "
                       "Alternative definitions require PRA justification."),
            "mdd_reference": mdd_default if mdd_has_def else None,
            "check_type": "cross_reference",
        })

        # 2.4 Macro variables — data vs MDD cross-reference
        macro_kw    = ["gdp", "unemployment", "hpi", "inflation", "rate",
                       "macro", "cpi", "index", "interest", "house_price"]
        found_macro = [c for c in df.columns
                       if any(k in c.lower() for k in macro_kw)]
        mdd_macro   = ij.get("macro_variables_mentioned", False)

        if found_macro:
            status_24 = "PASS"
            obs_24    = f"Macro columns found: {found_macro}"
        elif mdd_macro:
            status_24 = "WARN"
            obs_24    = "MDD mentions macro variables but none detected in dataset"
        else:
            status_24 = "FAIL"
            obs_24    = "No macro variables in dataset or MDD"
        findings.append({
            "check_id": "2.4",
            "stage": "Stage 2: Data Validation",
            "title": "Forward-Looking Macro Variables Present",
            "source": "IFRS 9", "principle": "B5.5.49", "severity": "MEDIUM",
            "status": status_24, "observed": obs_24,
            "threshold": "≥ 1 macro variable; ≥ 3 ECL scenarios defined",
            "detail": ("IFRS 9 B5.5.49 requires forward-looking macro information "
                       "in ECL models. Absence is a material gap."),
            "mdd_reference": self._mdd_quote("macro", "gdp", "scenario",
                                              "forward-looking"),
            "check_type": "cross_reference",
        })

        # 2.5 Historical coverage
        date_col = None
        df_work  = df.copy()
        for col in df_work.columns:
            if pd.api.types.is_datetime64_any_dtype(df_work[col]):
                date_col = col
                break
            if df_work[col].dtype == object:
                parsed = pd.to_datetime(df_work[col], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df_work[col] = parsed
                    date_col = col
                    break

        if date_col:
            span = (df_work[date_col].max() - df_work[date_col].min()).days / 365.25
            cov_status = ("PASS" if span >= 5 else "WARN" if span >= 3 else "FAIL")
            obs_25     = (f"{span:.1f} years "
                          f"({df_work[date_col].min().date()} → "
                          f"{df_work[date_col].max().date()})")
        else:
            cov_status = "WARN"
            obs_25     = "No date column detected — coverage cannot be verified"
        findings.append({
            "check_id": "2.5",
            "stage": "Stage 2: Data Validation",
            "title": "Historical Coverage ≥ 5 Years",
            "source": "SS11/13", "principle": "§10.1", "severity": "HIGH",
            "status": cov_status, "observed": obs_25,
            "threshold": "≥ 5 years covering at least one economic cycle",
            "detail": "SS11/13 §10.1 — insufficient history is a blocker for IRB",
            "mdd_reference": self._mdd_quote("data period", "observation",
                                              "history", "years"),
            "check_type": "data",
        })

        # 2.6 Sampling strategy — doc only
        sampling_found = self._mdd_contains(
            "sampling", "selection", "population", "representative"
        )
        findings.append({
            "check_id": "2.6",
            "stage": "Stage 2: Data Validation",
            "title": "Sampling Strategy Documented",
            "source": "SS1/23", "principle": "P3.2", "severity": "MEDIUM",
            "status": "PASS" if sampling_found else "WARN",
            "observed": ("Sampling methodology found in MDD"
                         if sampling_found
                         else "Sampling strategy not documented in MDD"),
            "threshold": "No material sampling bias; methodology documented",
            "detail": ("Check MDD for survivorship bias, selection bias, "
                       "time-period bias. Cannot be fully automated."),
            "mdd_reference": self._mdd_quote("sampling", "selection", "population"),
            "check_type": "doc",
        })

        # 2.7 Transformations documented
        transform_found = self._mdd_contains(
            "transform", "encoding", "scaling", "normaliz",
            "log transform", "binning", "woe"
        )
        findings.append({
            "check_id": "2.7",
            "stage": "Stage 2: Data Validation",
            "title": "Feature Transformations Documented",
            "source": "SS1/23", "principle": "P3.5", "severity": "MEDIUM",
            "status": "PASS" if transform_found else "WARN",
            "observed": ("Transformation methodology found in MDD"
                         if transform_found
                         else "No feature transformation documentation in MDD"),
            "threshold": "All transformations documented and reproducible",
            "detail": ("If not in MDD, formally request data dictionary "
                       "before proceeding to Stage 3"),
            "mdd_reference": self._mdd_quote("transform", "encoding", "woe", "binning"),
            "check_type": "doc",
        })

        # 2.8 Target leakage
        leakage_kw = ["recovery", "loss_given", "write", "charged",
                      "resolved", "post_default", "lgd", "ead_actual"]
        leaked     = [c for c in df.columns
                      if any(k in c.lower() for k in leakage_kw)]
        high_corr  = []
        bin_cols   = [c for c in df.columns
                      if df[c].dropna().nunique() == 2]
        if bin_cols:
            try:
                tc     = bin_cols[0]
                num_df = df.select_dtypes(include=[np.number])
                if tc in num_df.columns:
                    corrs     = num_df.drop(columns=[tc]).corrwith(num_df[tc]).abs()
                    high_corr = corrs[corrs > 0.95].index.tolist()
            except Exception:
                pass
        all_leakage = list(set(leaked + high_corr))
        findings.append({
            "check_id": "2.8",
            "stage": "Stage 2: Data Validation",
            "title": "Target Leakage Detection",
            "source": "SS1/23", "principle": "P3.5", "severity": "HIGH",
            "status": "FAIL" if all_leakage else "PASS",
            "observed": (f"Suspected leakage columns: {all_leakage}"
                         if all_leakage
                         else "No leakage detected"),
            "threshold": "No post-default or future-looking predictors",
            "detail": ("Leakage produces artificially inflated AUC. "
                       "Each suspected column must be reviewed."),
            "mdd_reference": None, "check_type": "data",
        })

        # 2.9 Duplicate rate
        n_dups   = df.duplicated().sum()
        dup_rate = n_dups / len(df)
        findings.append({
            "check_id": "2.9",
            "stage": "Stage 2: Data Validation",
            "title": "Duplicate Record Rate",
            "source": "SS1/23", "principle": "P3.2", "severity": "LOW",
            "status": ("FAIL" if dup_rate > 0.01
                       else "WARN" if dup_rate > 0.001
                       else "PASS"),
            "observed": f"{n_dups:,} duplicates ({dup_rate:.2%})",
            "threshold": "< 1% duplicate rate",
            "detail": "Duplicates inflate effective sample size and bias training",
            "mdd_reference": None, "check_type": "data",
        })

        # 2.10 Class imbalance
        bin_targets = [
            c for c in df.columns
            if df[c].dropna().nunique() == 2
            and set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})
        ]
        if bin_targets:
            tc        = bin_targets[0]
            counts    = df[tc].value_counts()
            ratio     = counts.min() / counts.max()
            min_pct   = counts.min() / counts.sum()
            imb_status = ("FAIL" if ratio < 0.1
                          else "WARN" if ratio < 0.33
                          else "PASS")
            imb_obs    = (f"Minority: {min_pct:.1%} | "
                          f"Ratio: {ratio:.2f} ('{tc}')")
        else:
            imb_status = "WARN"
            imb_obs    = "No binary 0/1 target column detected automatically"
        findings.append({
            "check_id": "2.10",
            "stage": "Stage 2: Data Validation",
            "title": "Class Imbalance",
            "source": "SS1/23", "principle": "P3.2", "severity": "MEDIUM",
            "status": imb_status, "observed": imb_obs,
            "threshold": "Minority ratio > 0.33 or mitigation documented",
            "detail": "Severe imbalance without mitigation leads to poor recall",
            "mdd_reference": None, "check_type": "data",
        })

        return findings  # 10 checks: 2.1–2.10

    # ── Stage 3: Conceptual Soundness ─────────────────────────────────────────

    def check_conceptual_soundness(self) -> list:
        print("========== AGENT2 START ==========")
        findings = []
        df = self.val_df
        ij = self.intake_json

        # 3.1 Methodology appropriate and justified
        methodology  = ij.get("methodology", "Not specified")
        known_methods = ["logistic", "xgboost", "lightgbm", "random forest",
                         "gradient boost", "neural", "scorecard", "linear"]
        method_found  = (
            methodology not in ("Not specified", "", None)
            or any(m in self.mdd_text for m in known_methods)
        )
        findings.append({
            "check_id": "3.1",
            "stage": "Stage 3: Conceptual Soundness",
            "title": "Model Methodology Appropriate and Justified",
            "source": "SS1/23", "principle": "P3.1", "severity": "HIGH",
            "status": "PASS" if method_found else "WARN",
            "observed": f"Methodology: {methodology}",
            "threshold": "Methodology justified relative to alternatives",
            "detail": ("SS1/23 P3.1 — methodology must be conceptually sound "
                       "and appropriate for intended purpose"),
            "mdd_reference": self._mdd_quote("methodology", "algorithm",
                                              "approach", "model choice"),
            "check_type": "doc",
        })

        # 3.2 Feature engineering statistically justified (IV/WoE)
        iv_found = self._mdd_contains(
            "iv", "information value", "woe", "weight of evidence",
            "feature selection"
        )
        findings.append({
            "check_id": "3.2",
            "stage": "Stage 3: Conceptual Soundness",
            "title": "Feature Engineering Statistically Justified",
            "source": "SS1/23", "principle": "P3.3", "severity": "HIGH",
            "status": "PASS" if iv_found else "WARN",
            "observed": ("IV/WoE analysis documented in MDD"
                         if iv_found
                         else "No IV/WoE or feature selection justification in MDD"),
            "threshold": ("All retained features have IV > 0.02 or "
                          "documented business justification"),
            "detail": ("SS1/23 P3.3 requires statistical justification "
                       "for variable selection"),
            "mdd_reference": self._mdd_quote("information value", "woe",
                                              "feature selection"),
            "check_type": "doc",
        })

        # 3.3 Multicollinearity (VIF proxy via pairwise correlation)
        if df is not None:
            try:
                num_cols      = df.select_dtypes(include=[np.number]).columns.tolist()
                binary_targets = [c for c in num_cols if df[c].dropna().nunique() == 2]
                feature_cols  = [c for c in num_cols if c not in binary_targets][:15]
                high_vif_cols = []
                if len(feature_cols) > 1:
                    corr_matrix = df[feature_cols].corr().abs()
                    for col in feature_cols:
                        max_r = corr_matrix[col].drop(col).max()
                        if max_r > 0.85:
                            high_vif_cols.append(col)
                vif_status = ("FAIL" if len(high_vif_cols) > 3
                              else "WARN" if high_vif_cols
                              else "PASS")
                vif_obs    = (f"High multicollinearity detected in: {high_vif_cols}"
                              if high_vif_cols
                              else "No high multicollinearity detected")
            except Exception as e:
                vif_status = "WARN"
                vif_obs    = f"VIF check could not be completed: {e}"
        else:
            vif_status = "WARN"
            vif_obs    = "No dataset uploaded"
        findings.append({
            "check_id": "3.3",
            "stage": "Stage 3: Conceptual Soundness",
            "title": "Multicollinearity (VIF) Assessed",
            "source": "SS1/23", "principle": "P3.3", "severity": "MEDIUM",
            "status": vif_status, "observed": vif_obs,
            "threshold": "No feature with VIF > 10 retained without justification",
            "detail": ("High collinearity reduces model stability and "
                       "interpretability of individual feature effects"),
            "mdd_reference": self._mdd_quote("vif", "multicollinear", "correlation"),
            "check_type": "data",
        })

        # 3.4 Highly correlated feature pairs
        if df is not None:
            try:
                num_df = df.select_dtypes(include=[np.number])
                corr   = num_df.corr().abs()
                pairs  = []
                cols   = corr.columns.tolist()
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        if corr.iloc[i, j] > 0.9:
                            pairs.append(
                                f"{cols[i]}/{cols[j]} ({corr.iloc[i,j]:.2f})"
                            )
                corr_status = ("FAIL" if len(pairs) > 2
                               else "WARN" if pairs
                               else "PASS")
                corr_obs    = (f"High correlation pairs: {pairs[:5]}"
                               if pairs
                               else "No highly correlated pairs detected")
            except Exception:
                corr_status = "WARN"
                corr_obs    = "Correlation check could not be completed"
        else:
            corr_status = "WARN"
            corr_obs    = "No dataset uploaded"
        findings.append({
            "check_id": "3.4",
            "stage": "Stage 3: Conceptual Soundness",
            "title": "Highly Correlated Feature Pairs Addressed",
            "source": "SS1/23", "principle": "P3.3", "severity": "MEDIUM",
            "status": corr_status, "observed": corr_obs,
            "threshold": "No correlated pair > 0.9 retained without justification",
            "detail": ("Highly correlated features introduce redundancy "
                       "and reduce model stability"),
            "mdd_reference": None, "check_type": "data",
        })

        # 3.5 Bias and fairness evaluation
        bias_found = self._mdd_contains(
            "bias", "fairness", "discriminat", "protected",
            "demographic", "equal"
        )
        findings.append({
            "check_id": "3.5",
            "stage": "Stage 3: Conceptual Soundness",
            "title": "Bias and Fairness Evaluation Performed",
            "source": "SS1/23", "principle": "P1.3", "severity": "HIGH",
            "status": "PASS" if bias_found else "FAIL",
            "observed": ("Bias/fairness analysis found in MDD"
                         if bias_found
                         else "No bias or fairness evaluation in MDD"),
            "threshold": ("No material discriminatory bias identified or "
                          "documented with mitigation plan"),
            "detail": ("SS1/23 P1.3 — bias in credit models has regulatory "
                       "and reputational consequences"),
            "mdd_reference": self._mdd_quote("bias", "fairness", "demographic"),
            "check_type": "doc",
        })

        # 3.6 Variable selection methodology documented
        var_sel_found = self._mdd_contains(
            "variable selection", "feature selection", "lasso",
            "stepwise", "iv", "information value"
        )
        findings.append({
            "check_id": "3.6",
            "stage": "Stage 3: Conceptual Soundness",
            "title": "Variable Selection Methodology Documented",
            "source": "SS11/13", "principle": "§9.1", "severity": "MEDIUM",
            "status": "PASS" if var_sel_found else "WARN",
            "observed": ("Variable selection method found in MDD"
                         if var_sel_found
                         else "No variable selection methodology in MDD"),
            "threshold": ("Selection method documented; consistent with "
                          "industry practice"),
            "detail": ("SS11/13 §9.1 requires all rating system inputs "
                       "to be fully documented"),
            "mdd_reference": self._mdd_quote("variable selection",
                                              "feature selection"),
            "check_type": "doc",
        })

        # 3.7 Transformation assumptions justified
        transform_found = self._mdd_contains(
            "log transform", "binning", "encoding", "scaling",
            "normaliz", "standardiz"
        )
        findings.append({
            "check_id": "3.7",
            "stage": "Stage 3: Conceptual Soundness",
            "title": "Transformation Assumptions Justified",
            "source": "SS1/23", "principle": "P3.5", "severity": "LOW",
            "status": "PASS" if transform_found else "WARN",
            "observed": ("Transformation assumptions in MDD"
                         if transform_found
                         else "No transformation justification in MDD"),
            "threshold": "Transformations improve model fit; assumptions documented",
            "detail": ("Log transforms, binning choices must be "
                       "statistically motivated"),
            "mdd_reference": self._mdd_quote("transform", "encoding", "scaling"),
            "check_type": "doc",
        })

        # 3.8 Model operating boundaries defined
        boundaries_found = self._mdd_contains(
            "operating", "boundary", "boundaries", "score range",
            "input domain", "extrapolat", "out of sample"
        )
        findings.append({
            "check_id": "3.8",
            "stage": "Stage 3: Conceptual Soundness",
            "title": "Model Operating Boundaries Defined",
            "source": "SS1/23", "principle": "P3.3", "severity": "MEDIUM",
            "status": "PASS" if boundaries_found else "WARN",
            "observed": ("Operating boundaries found in MDD"
                         if boundaries_found
                         else "No operating boundaries defined in MDD"),
            "threshold": ("Score range and input domain documented; "
                          "extrapolation risks acknowledged"),
            "detail": "Without defined boundaries the model may be misapplied",
            "mdd_reference": self._mdd_quote("operating", "boundary",
                                              "score range"),
            "check_type": "doc",
        })

        # 3.9 Hyperparameter tuning documented
        hp_found = self._mdd_contains(
            "hyperparameter", "tuning", "grid search", "random search",
            "n_estimators", "max_depth", "learning rate"
        )
        findings.append({
            "check_id": "3.9",
            "stage": "Stage 3: Conceptual Soundness",
            "title": "Hyperparameter Tuning Documented",
            "source": "SS1/23", "principle": "P3.5", "severity": "MEDIUM",
            "status": "PASS" if hp_found else "WARN",
            "observed": ("Hyperparameter tuning documented in MDD"
                         if hp_found
                         else "No hyperparameter tuning documentation in MDD"),
            "threshold": "Tuning method reproducible; final parameters logged",
            "detail": "Undocumented tuning cannot be independently verified",
            "mdd_reference": self._mdd_quote("hyperparameter", "tuning"),
            "check_type": "doc",
        })

        # 3.10 Class imbalance treatment documented
        smote_found = self._mdd_contains(
            "smote", "oversamp", "undersamp", "class weight",
            "imbalance", "minority"
        )
        findings.append({
            "check_id": "3.10",
            "stage": "Stage 3: Conceptual Soundness",
            "title": "Class Imbalance Treatment Documented",
            "source": "SS1/23", "principle": "P3.5", "severity": "LOW",
            "status": "PASS" if smote_found else "WARN",
            "observed": ("Imbalance treatment found in MDD"
                         if smote_found
                         else "No class imbalance treatment in MDD"),
            "threshold": "SMOTE parameters or class weighting documented",
            "detail": "Imbalance treatment must be documented per SS1/23 P3.5",
            "mdd_reference": self._mdd_quote("smote", "oversamp", "class weight"),
            "check_type": "doc",
        })

        return findings  # 10 checks: 3.1–3.10

    # ── Stage 4: Model Replication ────────────────────────────────────────────

    def check_model_replication(self) -> list:
        self.replicated_importances: dict = {}
        findings = []
        ij = self.intake_json
        hp = self.hyperparams
        _stage = "Stage 4: Model Replication"

        # 4.1 — Hyperparameter config submitted
        if hp:
            _param_summary = ", ".join(list(hp.keys())[:5]) + ("…" if len(hp) > 5 else "")
            findings.append({
                "check_id": "4.1",
                "stage": _stage,
                "title": "Model Replicable from Submitted Artifacts",
                "source": "SS1/23", "principle": "P4.1", "severity": "HIGH",
                "status": "PASS",
                "observed": f"Hyperparameter config submitted — {len(hp)} parameters: {_param_summary}",
                "threshold": "Hyperparameter config present and sufficient for replication",
                "detail": "SS1/23 P4.1 requires all model parameters to be documented for independent replication",
                "mdd_reference": None,
                "check_type": "doc",
            })
        else:
            findings.append({
                "check_id": "4.1",
                "stage": _stage,
                "title": "Model Replicable from Submitted Artifacts",
                "source": "SS1/23", "principle": "P4.1", "severity": "HIGH",
                "status": "FAIL",
                "observed": "No hyperparameter config submitted — model cannot be independently replicated",
                "threshold": "Hyperparameter config present and sufficient for replication",
                "detail": "SS1/23 P4.1 requires all model parameters to be documented for independent replication",
                "mdd_reference": None,
                "check_type": "doc",
            })

        # 4.2 — Model type cross-reference: config model vs MDD algorithm
        hp_model = str(hp.get("model", "")).strip() if hp else ""
        ij_algo  = str(ij.get("algorithm", "")).strip()
        _normalise = lambda s: s.lower().replace("-", "").replace("_", "").replace(" ", "")
        if hp_model and ij_algo:
            _match = _normalise(hp_model) == _normalise(ij_algo)
            findings.append({
                "check_id": "4.2",
                "stage": _stage,
                "title": "Replicated Model Type Matches MDD Declaration",
                "source": "SS11/13", "principle": "§10.3", "severity": "HIGH",
                "status": "PASS" if _match else "FAIL",
                "observed": (
                    f"Config model '{hp_model}' matches MDD algorithm '{ij_algo}'" if _match
                    else f"Config model '{hp_model}' does not match MDD algorithm '{ij_algo}'"
                ),
                "threshold": "Model type in hyperparameter config matches algorithm declared in MDD",
                "detail": "SS11/13 §10.3 requires the replication to use the same model class as documented",
                "mdd_reference": None,
                "check_type": "quant",
            })
        else:
            findings.append({
                "check_id": "4.2",
                "stage": _stage,
                "title": "Replicated Model Type Matches MDD Declaration",
                "source": "SS11/13", "principle": "§10.3", "severity": "HIGH",
                "status": "WARN",
                "observed": (
                    f"Cannot cross-reference — config model: '{hp_model or '(not provided)'}', "
                    f"MDD algorithm: '{ij_algo or '(not provided)'}'"
                ),
                "threshold": "Model type in hyperparameter config matches algorithm declared in MDD",
                "detail": "SS11/13 §10.3 requires the replication to use the same model class as documented",
                "mdd_reference": None,
                "check_type": "quant",
            })

        # 4.3 — Independent replication: train model on val_df, compare AUC to stated
        _check_43 = {
            "check_id": "4.3",
            "stage": _stage,
            "title": "Independent Replicated AUC Within Tolerance (±0.05)",
            "source": "SS11/13", "principle": "§10.3", "severity": "HIGH",
            "threshold": "Replicated AUC within ±0.05 of developer-stated AUC",
            "detail": "SS11/13 §10.3 requires independent validation to reproduce model discriminatory power",
            "mdd_reference": None,
            "check_type": "quant",
        }
        stated_auc = ij.get("stated_auc")
        if not hp:
            _check_43["status"] = "WARN"
            _check_43["observed"] = "No hyperparameter config submitted — skipping independent replication"
        elif stated_auc is None:
            _check_43["status"] = "WARN"
            _check_43["observed"] = "No stated_auc in intake JSON — cannot benchmark replicated AUC"
        else:
            try:
                import xgboost as xgb
                from sklearn.metrics import roc_auc_score
                from sklearn.model_selection import train_test_split

                _df = self.val_df.copy()
                _target = ij.get("target_col")
                if not _target or _target not in _df.columns:
                    _candidates = [c for c in _df.columns
                                   if _df[c].nunique() == 2
                                   and _df[c].dtype in (np.int64, np.int32, np.float64, np.float32)]
                    _target = _candidates[0] if _candidates else None
                if _target is None:
                    raise ValueError("Target column not found in dataset")

                _feat_cols = [c for c in _df.select_dtypes(include=[np.number]).columns
                              if c != _target]
                if not _feat_cols:
                    raise ValueError("No numeric feature columns found")

                _X = _df[_feat_cols].fillna(_df[_feat_cols].median())
                _y = _df[_target]

                # Guard: dataset too small for reliable independent replication
                if len(_df) < 200 or int(_y.sum()) < 10:
                    _check_43["status"] = "FAIL"
                    _check_43["observed"] = (
                        f"Dataset too small for replication: {len(_df)} rows, "
                        f"{int(_y.sum())} defaults — minimum 200 rows and 10 defaults required"
                    )
                    _check_43["threshold"] = "Minimum 200 rows and 10 defaults required for replication"
                    _check_43["detail"] = (
                        "Insufficient data for independent replication — "
                        "this is itself a regulatory blocker under SS11/13 §10.3"
                    )
                    findings.append(_check_43)
                    findings.extend(self._stub_stage(
                        _stage,
                        [
                            ("4.4", "Feature Importance Comparison",               "SS1/23",  "P4.1",  "HIGH"),
                            ("4.5", "Brier Score Reproduced",                      "SS11/13", "§10.5", "HIGH"),
                            ("4.6", "Variable Removal Does Not Cause Instability",  "SS1/23",  "P4.3",  "MEDIUM"),
                            ("4.7", "Random Seed Sensitivity Assessed",             "SS1/23",  "P3.3",  "MEDIUM"),
                            ("4.8", "Train/Test Split Reproducible and Leak-Free",  "SS1/23",  "P3.5",  "HIGH"),
                            ("4.9", "Cross-Validation Results Reproducible",        "SS1/23",  "P3.3",  "MEDIUM"),
                        ],
                    ))
                    return findings

                _X_tr, _X_te, _y_tr, _y_te = train_test_split(
                    _X, _y, test_size=0.2, random_state=42, stratify=_y
                )

                _valid_keys = {
                    "n_estimators", "max_depth", "learning_rate", "subsample",
                    "colsample_bytree", "min_child_weight", "scale_pos_weight",
                    "gamma", "reg_alpha", "reg_lambda",
                }
                _xgb_params = {k: v for k, v in hp.items() if k in _valid_keys}
                _xgb_params.setdefault("random_state", 42)

                _clf = xgb.XGBClassifier(**_xgb_params, use_label_encoder=False, verbosity=0)
                _clf.fit(_X_tr, _y_tr)
                _proba = _clf.predict_proba(_X_te)[:, 1]
                _rep_auc = round(float(roc_auc_score(_y_te, _proba)), 4)
                _diff = abs(_rep_auc - float(stated_auc))

                if _diff <= 0.05:
                    _check_43["status"] = "PASS"
                    _check_43["observed"] = (
                        f"Replicated AUC = {_rep_auc:.4f}; stated AUC = {stated_auc}; "
                        f"absolute difference = {_diff:.4f} ≤ 0.05 tolerance"
                    )
                else:
                    _check_43["status"] = "FAIL"
                    _check_43["observed"] = (
                        f"Replicated AUC = {_rep_auc:.4f}; stated AUC = {stated_auc}; "
                        f"absolute difference = {_diff:.4f} exceeds 0.05 tolerance"
                    )

                # ── 4.4 Feature Importance Comparison ────────────────────────
                try:
                    _rep_importances = dict(zip(
                        _feat_cols,
                        _clf.feature_importances_.tolist(),
                    ))
                    self.replicated_importances = _rep_importances

                    _rep_top5 = sorted(
                        _rep_importances.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    _rep_top5_names = [f for f, _ in _rep_top5]

                    _stated_importances = hp.get("feature_importances", {})

                    if not _stated_importances:
                        findings.append({
                            "check_id": "4.4",
                            "stage": _stage,
                            "title": "Feature Importance Comparison",
                            "source": "SS1/23", "principle": "P4.1", "severity": "HIGH",
                            "status": "WARN",
                            "observed": (
                                f"Developer did not submit feature importances. "
                                f"Replicated top features: {_rep_top5_names}"
                            ),
                            "threshold": "Developer must submit feature importance scores for comparison",
                            "detail": (
                                "Cannot verify model learned same patterns without "
                                "developer's stated feature importances"
                            ),
                            "mdd_reference": None,
                            "check_type": "cross_reference",
                        })
                    else:
                        _stated_top5 = sorted(
                            _stated_importances.items(), key=lambda x: x[1], reverse=True
                        )[:5]
                        _stated_top5_names = [f for f, _ in _stated_top5]

                        _overlap = len(set(_rep_top5_names) & set(_stated_top5_names))
                        _overlap_pct = _overlap / 5
                        _top_match = (
                            _rep_top5_names[0] == _stated_top5_names[0]
                            if _rep_top5_names and _stated_top5_names else False
                        )

                        _common = set(_rep_importances.keys()) & set(_stated_importances.keys())
                        if len(_common) >= 3:
                            _rs = [_rep_importances[f] for f in _common]
                            _ss = [_stated_importances[f] for f in _common]
                            _rs_sum = sum(_rs) or 1
                            _ss_sum = sum(_ss) or 1
                            _rn = [v / _rs_sum for v in _rs]
                            _sn = [v / _ss_sum for v in _ss]
                            _mad = sum(abs(r - s) for r, s in zip(_rn, _sn)) / len(_rn)
                        else:
                            _mad = 1.0

                        if _overlap_pct >= 0.8 and _top_match and _mad < 0.15:
                            _fi_status = "PASS"
                        elif _overlap_pct >= 0.6 and _mad < 0.25:
                            _fi_status = "WARN"
                        else:
                            _fi_status = "FAIL"

                        findings.append({
                            "check_id": "4.4",
                            "stage": _stage,
                            "title": "Feature Importance Comparison",
                            "source": "SS1/23", "principle": "P4.1", "severity": "HIGH",
                            "status": _fi_status,
                            "observed": (
                                f"Replicated top 5: {_rep_top5_names} | "
                                f"Developer stated top 5: {_stated_top5_names} | "
                                f"Overlap: {_overlap}/5 | "
                                f"Top feature match: {_top_match} | "
                                f"Mean score difference: {_mad:.3f}"
                            ),
                            "threshold": (
                                ">=4/5 top features overlap, same #1 feature, "
                                "mean importance difference < 0.15"
                            ),
                            "detail": (
                                "FAIL: Model learned fundamentally different patterns — "
                                "replication failed" if _fi_status == "FAIL"
                                else "WARN: Partial overlap — minor differences acceptable "
                                "if justified" if _fi_status == "WARN"
                                else "Feature importances consistent with developer submission"
                            ),
                            "mdd_reference": None,
                            "check_type": "cross_reference",
                        })
                except Exception as _e44:
                    findings.append({
                        "check_id": "4.4",
                        "stage": _stage,
                        "title": "Feature Importance Comparison",
                        "source": "SS1/23", "principle": "P4.1", "severity": "HIGH",
                        "status": "WARN",
                        "observed": f"Feature importance comparison could not run: {_e44}",
                        "threshold": "Feature importances must be comparable between replication and submission",
                        "detail": "Manual review required",
                        "mdd_reference": None,
                        "check_type": "cross_reference",
                    })

            except ImportError:
                _check_43["status"] = "WARN"
                _check_43["observed"] = "xgboost or scikit-learn not installed — cannot run independent replication"
            except Exception as _e:
                _check_43["status"] = "WARN"
                _check_43["observed"] = f"Replication could not complete: {_e}"
        findings.append(_check_43)

        # 4.5–4.9 — stubs (manual validation required)
        findings.extend(self._stub_stage(
            _stage,
            [
                ("4.5", "Brier Score Reproduced",                    "SS11/13", "§10.5", "HIGH"),
                ("4.6", "Variable Removal Does Not Cause Instability","SS1/23",  "P4.3",  "MEDIUM"),
                ("4.7", "Random Seed Sensitivity Assessed",           "SS1/23",  "P3.3",  "MEDIUM"),
                ("4.8", "Train/Test Split Reproducible and Leak-Free","SS1/23",  "P3.5",  "HIGH"),
                ("4.9", "Cross-Validation Results Reproducible",      "SS1/23",  "P3.3",  "MEDIUM"),
            ],
        ))
        return findings  # 9 checks: 4.1–4.9

    # ── Stage 5: Performance Validation ──────────────────────────────────────

    def check_performance(self) -> list:
        findings = []
        ij = self.intake_json

        # Checks with stated-metric cross-reference
        metric_checks = [
            ("5.1", "ROC-AUC ≥ 0.70",   "SS1/23",  "P4.1", "HIGH",
             "stated_auc",    0.70,  "ROC-AUC ≥ 0.70 on independent test set"),
            ("5.2", "Recall ≥ 0.60",     "SS1/23",  "P4.4", "HIGH",
             "stated_recall", 0.60,  "Recall ≥ 0.60"),
            ("5.3", "Gini ≥ 0.40",       "SS11/13", "§10.3","HIGH",
             "stated_gini",   0.40,  "Gini ≥ 0.40"),
            ("5.4", "Brier Score < 0.25","SS11/13", "§10.5","HIGH",
             "stated_brier",  None,  "Brier Score < 0.25"),
        ]
        for (check_id, title, source, principle, severity,
             key, threshold_val, threshold_str) in metric_checks:
            value = ij.get(key)
            if value is None:
                status   = "WARN"
                observed = f"{title} not stated in MDD — cannot verify"
            else:
                if key == "stated_brier":
                    status = "PASS" if value < 0.25 else "FAIL"
                else:
                    status = "PASS" if value >= threshold_val else "FAIL"
                observed = f"Developer stated: {value}"

            findings.append({
                "check_id":     check_id,
                "stage":        "Stage 5: Performance Validation",
                "title":        title,
                "source":       source,
                "principle":    principle,
                "severity":     severity,
                "status":       status,
                "observed":     observed,
                "threshold":    threshold_str,
                "detail":       ("Cross-reference against independently "
                                 "replicated metric in Stage 4"),
                "mdd_reference": self._mdd_quote(
                    title.split("≥")[0].strip().lower()
                ),
                "check_type":   "cross_reference",
            })

        # Remaining performance checks as stubs
        findings.extend(self._stub_stage(
            "Stage 5: Performance Validation",
            [
                ("5.5",  "Train/Test AUC Gap ≤ 0.10",
                 "SS1/23",  "P4.4",  "MEDIUM"),
                ("5.6",  "PR-AUC ≥ 0.30",
                 "SS1/23",  "P3.3",  "MEDIUM"),
                ("5.7",  "KS Statistic Computed",
                 "SS11/13", "§10.3", "MEDIUM"),
                ("5.8",  "Calibration Test Performed",
                 "SS11/13", "§10.5", "HIGH"),
                ("5.9",  "SHAP Explainability Outputs Reviewed",
                 "SS1/23",  "P4.2",  "HIGH"),
                ("5.10", "Decision Threshold Justified",
                 "SS1/23",  "P5.1",  "MEDIUM"),
                ("5.11", "Sensitivity Analysis Performed",
                 "SS1/23",  "P4.3",  "MEDIUM"),
            ],
        ))

        return findings  # 11 checks: 5.1–5.11

    # ── Stage 6: Stress & Backtesting (stubs) ────────────────────────────────

    def check_stress_backtesting(self) -> list:
        return self._stub_stage(
            "Stage 6: Stress & Backtesting",
            [
                ("6.1", "Sensitivity Analysis on Top-3 Features",
                 "SS1/23",  "P4.3",   "HIGH"),
                ("6.2", "Stress Test Under Adverse Macro Scenario",
                 "SS3/18",  "§2.1",   "HIGH"),
                ("6.3", "Out-of-Time Validation Performed",
                 "SS11/13", "§10.6",  "HIGH"),
                ("6.4", "PSI < 0.25",
                 "SS11/13", "§10.6",  "MEDIUM"),
                ("6.5", "Champion vs Challenger Benchmarking",
                 "SS1/23",  "P3.3",   "MEDIUM"),
                ("6.6", "Downturn LGD Applied",
                 "SS11/13", "§6.2",   "HIGH"),
                ("6.7", "ECL Macro Scenario Weighting (≥ 3 Scenarios)",
                 "IFRS 9",  "B5.5.49","MEDIUM"),
                ("6.8", "PD Backtesting vs Actual Defaults",
                 "SS11/13", "§10.3",  "HIGH"),
            ],
        )  # 8 checks: 6.1–6.8

    # ── Stage 7: Regulatory Compliance ───────────────────────────────────────

    def check_regulatory_compliance(self) -> list:
        findings = []
        ij = self.intake_json

        # 7.1 Calibrated PD output
        calibration = ij.get("calibration_method", "Not specified")
        cal_found   = (
            calibration not in ("Not specified", "", None)
            or self._mdd_contains("calibrat", "platt", "isotonic",
                                  "through the cycle", "ttc")
        )
        findings.append({
            "check_id": "7.1",
            "stage": "Stage 7: Regulatory Compliance",
            "title": "Calibrated PD Output Produced",
            "source": "IFRS 9", "principle": "5.5", "severity": "HIGH",
            "status": "PASS" if cal_found else "FAIL",
            "observed": f"Calibration method: {calibration}",
            "threshold": ("Calibrated PD ∈ [0,1] per borrower; "
                          "Platt scaling or isotonic regression applied"),
            "detail": "IFRS 9 §5.5 requires calibrated PD output for ECL",
            "mdd_reference": self._mdd_quote("calibrat", "platt",
                                              "through the cycle"),
            "check_type": "doc",
        })

        # 7.2 IFRS 9 staging implemented
        staging_found = self._mdd_contains(
            "stage 1", "stage 2", "stage 3", "staging",
            "sicr", "lifetime ecl", "12 month"
        )
        findings.append({
            "check_id": "7.2",
            "stage": "Stage 7: Regulatory Compliance",
            "title": "IFRS 9 Stage 1/2/3 Classification Implemented",
            "source": "IFRS 9", "principle": "5.5.3", "severity": "HIGH",
            "status": "PASS" if staging_found else "FAIL",
            "observed": ("IFRS 9 staging logic found in MDD"
                         if staging_found
                         else "No IFRS 9 staging documented in MDD"),
            "threshold": ("Every borrower assigned a stage; "
                          "thresholds documented"),
            "detail": ("IFRS 9 §5.5.3 — staging drives ECL horizon "
                       "(12-month vs lifetime)"),
            "mdd_reference": self._mdd_quote("stage 1", "stage 2",
                                              "staging", "sicr"),
            "check_type": "doc",
        })

        # 7.3 SICR detection criteria
        sicr_found = self._mdd_contains(
            "sicr", "significant increase", "credit risk",
            "30 dpd", "pd doubling", "backstop"
        )
        findings.append({
            "check_id": "7.3",
            "stage": "Stage 7: Regulatory Compliance",
            "title": "SICR Detection Criteria Documented",
            "source": "IFRS 9", "principle": "5.5.7", "severity": "HIGH",
            "status": "PASS" if sicr_found else "FAIL",
            "observed": ("SICR criteria found in MDD"
                         if sicr_found
                         else "No SICR detection criteria in MDD"),
            "threshold": "PD doubling trigger AND 30+ DPD backstop documented",
            "detail": ("IFRS 9 §5.5.7 requires both quantitative and "
                       "qualitative SICR triggers"),
            "mdd_reference": self._mdd_quote("sicr", "significant increase",
                                              "backstop"),
            "check_type": "doc",
        })

        # 7.4 Lifetime ECL for Stage 2/3
        ecl_found = self._mdd_contains(
            "lifetime ecl", "lifetime pd", "pd term structure",
            "expected credit loss", "ecl"
        )
        findings.append({
            "check_id": "7.4",
            "stage": "Stage 7: Regulatory Compliance",
            "title": "Lifetime ECL for Stage 2/3 Computed",
            "source": "IFRS 9", "principle": "5.5.3", "severity": "HIGH",
            "status": "PASS" if ecl_found else "FAIL",
            "observed": ("Lifetime ECL methodology found in MDD"
                         if ecl_found
                         else "No lifetime ECL methodology in MDD"),
            "threshold": "ECL = PD(lifetime) × LGD × EAD for Stage 2/3",
            "detail": ("IFRS 9 §5.5.3 — simple scaled 12m PD is insufficient "
                       "for Stage 2/3"),
            "mdd_reference": self._mdd_quote("lifetime", "ecl",
                                              "term structure"),
            "check_type": "doc",
        })

        # Remaining regulatory checks as stubs
        findings.extend(self._stub_stage(
            "Stage 7: Regulatory Compliance",
            [
                ("7.5",  "Credit Risk Concentration Analysis",
                 "IFRS 7",  "7.34",  "MEDIUM"),
                ("7.6",  "EAD Reported by Segment",
                 "IFRS 7",  "7.36",  "MEDIUM"),
                ("7.7",  "Past-Due Ageing Schedule Produced",
                 "IFRS 7",  "7.37",  "MEDIUM"),
                ("7.8",  "Annual SMF Attestation Submitted",
                 "SS11/13", "§4.40", "HIGH"),
                ("7.9",  "Governance Controls Match Risk Tier",
                 "SS1/23",  "P1.3",  "HIGH"),
                ("7.10", "Post-Model Overlays Governed",
                 "SS1/23",  "P5.1",  "MEDIUM"),
            ],
        ))

        return findings  # 10 checks: 7.1–7.10

    # ── Stage 8: Findings & Report (stubs) ───────────────────────────────────

    def check_findings_report(self) -> list:
        return self._stub_stage(
            "Stage 8: Findings & Report",
            [
                ("8.1", "HIGH Findings Listed with Remediation Owners",
                 "SS1/23",  "P4.1", "HIGH"),
                ("8.2", "MEDIUM Findings with Recommended Actions",
                 "SS1/23",  "P4.1", "MEDIUM"),
                ("8.3", "Overall Validation Verdict Issued",
                 "SS1/23",  "P4.1", "HIGH"),
                ("8.4", "Model Risk Tier Confirmed or Revised",
                 "SS1/23",  "P1.3", "HIGH"),
                ("8.5", "Revalidation Frequency Recommended",
                 "SS1/23",  "P4.5", "MEDIUM"),
                ("8.6", "Conditions for Conditional Approval Documented",
                 "SS1/23",  "P4.2", "HIGH"),
                ("8.7", "Validation Report Signed Off by Independent Validator",
                 "SS1/23",  "P4.1", "HIGH"),
                ("8.8", "Findings Tracker / Action Log Created",
                 "SS11/13", "§5.1", "MEDIUM"),
            ],
        )  # 8 checks: 8.1–8.8
