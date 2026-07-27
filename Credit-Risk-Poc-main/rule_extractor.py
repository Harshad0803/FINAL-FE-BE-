"""
rule_extractor.py
Extracts MDD documentation compliance rules from the hosted IFRS 9 / SS1-23
agent and maps each rule to the val_mdd_rules.json schema consumed by the
model-validation pipeline.

Focus: what SS1/23 and IFRS 9 say MUST BE PRESENT in a Model Development
Document (MDD) — business objective, data description, feature engineering,
model selection rationale, assumptions, SICR criteria, governance, etc.

Rules split into two validation stages:
  data_validation     — what the MDD must document about data preparation
  conceptual_soundness — what the MDD must document about model design/methodology

NOTE: verify=False is for internal testing only.
"""
from __future__ import annotations
import os
import re
import sys
from typing import Any

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Agent config ──────────────────────────────────────────────────────────────
AGENT_URL = os.environ.get(
    "IFRS_AGENT_URL",
    "https://agenticai.inuatintgenw.deloitte.com/api/v1/prediction/9e0b9c8a-1633-45e6-8bf0-9e294420eb02",
)
AGENT_TIMEOUT = int(os.environ.get("IFRS_AGENT_TIMEOUT", "300"))
AGENT_OVERRIDE_CONFIG: dict = {}

# ── Stage taxonomy ────────────────────────────────────────────────────────────
# Only two stages — both belong to model validation.
# Keywords are used for keyword-search cross-checking against an uploaded MDD.
ALLOWED_STAGES: tuple[str, ...] = ("data_validation", "conceptual_soundness")

_STAGE_DESC: dict[str, str] = {
    "data_validation": (
        "What the MDD must document about data: data sources, observation window, "
        "population definition, sampling methodology, exclusion criteria, default "
        "definition, target variable construction, handling of missing data, "
        "treatment of outliers, forward-looking macroeconomic variables, and "
        "steps taken to prevent data leakage or survivorship bias."
    ),
    "conceptual_soundness": (
        "What the MDD must document about model design and methodology: business "
        "objective and intended use, model selection rationale and alternatives "
        "considered, feature engineering decisions, SICR criteria and staging logic, "
        "model assumptions and known limitations, benchmarking against simpler models, "
        "explainability approach, sensitivity analysis, model governance and version "
        "control, and any post-model adjustments."
    ),
}

# Keywords used by the app to keyword-search an uploaded MDD.
# Each stage maps to the terms the validator expects to find in that section.
STAGE_KEYWORDS: dict[str, list[str]] = {
    "data_validation": [
        "data source", "observation window", "performance window", "training data",
        "population", "exclusion criteria", "sampling", "default definition",
        "days past due", "dpd", "target variable", "missing data", "outlier",
        "macroeconomic", "forward-looking", "scenario", "leakage", "survivorship bias",
        "data dictionary", "data quality",
    ],
    "conceptual_soundness": [
        "business objective", "intended use", "scope", "model selection", "rationale",
        "justification", "challenger", "benchmark", "alternative model",
        "feature engineering", "variable selection", "sicr", "significant increase",
        "credit risk", "staging", "assumption", "limitation", "sensitivity",
        "explainability", "shap", "feature importance", "governance", "version control",
        "model owner", "review cycle",
    ],
}

# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_stage_prompt(stage: str) -> str:
    """Focused prompt for one validation stage — MDD documentation requirements only."""
    return f"""
You are a regulatory model-risk analyst.

From the knowledge base, identify ALL regulatory requirements related to what a
Model Development Document (MDD) must contain or demonstrate for the
'{stage}' aspect:

{_STAGE_DESC[stage]}

Do not assume any specific regulatory framework.
Extract requirements from whichever regulatory documents are available in the
knowledge base and preserve the original regulatory reference.

For each requirement output exactly one block:

RULE_START
RULE_ID: <original regulatory reference exactly as it appears in the document>
DESCRIPTION: <one or two sentence plain-English statement of what the MDD must include or demonstrate>
MDD_SECTION: <the MDD section where this is typically documented>
KEYWORDS: <comma-separated list of 3-6 words/phrases>
SEVERITY: <high | medium | low>
RULE_END

Only output rules relevant to '{stage}'.
Focus on documentation requirements only.
Do not invent references.
Keep the original document name and principle/reference unchanged.
No text before the first RULE_START or after the last RULE_END.
""".strip()


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_agent_rules(raw: str) -> list[dict]:
    blocks = re.findall(r"RULE_START(.*?)RULE_END", raw, re.DOTALL)
    parsed: list[dict] = []
    for block in blocks:
        def field(name: str) -> str:
            m = re.search(rf"{name}:\s*(.+?)(?=\n[A-Z_]+:|$)", block, re.DOTALL)
            return m.group(1).strip() if m else ""
        parsed.append({
            "rule_id":     field("RULE_ID"),
            "description": field("DESCRIPTION"),
            "mdd_section": field("MDD_SECTION"),
            "keywords":    field("KEYWORDS"),
            "severity":    field("SEVERITY"),
        })
    return parsed


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")

def _infer_source(rule_id: str) -> str:
    t = (rule_id or "").lower()
    if "ss1" in t or "ss 1" in t:
        return "SS1/23"
    if "ss11" in t or "ss 11" in t:
        return "SS11/13"
    if "ifrs 7" in t or "ifrs7" in t:
        return "IFRS 7"
    return "IFRS 9"

def _normalise_severity(raw: str) -> str:
    s = (raw or "").lower().strip()
    return s if s in ("high", "medium", "low") else "medium"

def _parse_keywords(raw: str) -> list[str]:
    """Split comma-separated keyword string into a clean list."""
    if not raw:
        return []
    return [k.strip().lower() for k in raw.split(",") if k.strip()]


# ── RuleExtractor ─────────────────────────────────────────────────────────────

class RuleExtractor:
    """
    Extracts MDD documentation requirements from the hosted IFRS 9 / SS1-23 agent.
    Produces rules in the val_mdd_rules.json schema for keyword-search validation.
    """

    def __init__(self):
        pass

    def _call_agent(self, prompt: str, timeout: int | None = None) -> str:
        payload: dict = {"question": prompt}
        if AGENT_OVERRIDE_CONFIG:
            payload["overrideConfig"] = AGENT_OVERRIDE_CONFIG
        resp = requests.post(
            AGENT_URL, json=payload, verify=False,
            timeout=timeout or AGENT_TIMEOUT,
        )
        resp.raise_for_status()
        try:
            body = resp.json()
        except ValueError:
            return resp.text.strip()
        return self._find_text(body)

    @staticmethod
    def _find_text(obj) -> str:
        if isinstance(obj, str):
            return obj.strip()
        if isinstance(obj, dict):
            for key in ("text", "output", "answer", "response", "result", "content", "message"):
                if key in obj:
                    found = RuleExtractor._find_text(obj[key])
                    if found:
                        return found
        if isinstance(obj, list):
            for item in obj:
                found = RuleExtractor._find_text(item)
                if found:
                    return found
        return ""

    def _normalise(self, raw: dict, stage: str) -> dict:
        rule_id     = (raw.get("rule_id") or "").strip()
        description = (raw.get("description") or "").strip()
        mdd_section = (raw.get("mdd_section") or "").strip()
        keywords    = _parse_keywords(raw.get("keywords", ""))
        severity    = _normalise_severity(raw.get("severity", ""))
        source      = _infer_source(rule_id)
        check       = _slugify(rule_id) or _slugify(description[:48])

        # Supplement with stage default keywords if agent returned none
        if not keywords:
            keywords = STAGE_KEYWORDS.get(stage, [])[:6]

        return {
            "id":               None,            # assigned by val_build_rules
            "stage":            stage,
            "check_id_hint":    "2.x" if stage == "data_validation" else "3.x",
            "check":            check,
            "severity":         severity,
            "source":           source,
            "principle":        rule_id or "?",
            "rule":             description,
            "flag":             f"MDD missing required documentation: {description[:80]}" if description else "MDD documentation requirement not satisfied.",
            "suggestion":       f"Add a '{mdd_section}' section to the MDD." if mdd_section else "Update the MDD to satisfy this requirement.",
            "mdd_section_hint": mdd_section,
            "keywords":         keywords,
            "min_keyword_hits": max(1, min(2, len(keywords) // 2)),
        }

    def extract_from_agent(
        self,
        verbose: bool = False,
        per_call_timeout: int = 180,
    ) -> list[dict]:
        """
        Query the agent once per validation stage and return normalised rules.
        Deduplicates by `check` key across both stages.
        """
        merged: dict[str, dict] = {}

        for stage in ALLOWED_STAGES:
            prompt = _build_stage_prompt(stage)
            try:
                raw_text = self._call_agent(prompt, timeout=per_call_timeout)
            except requests.exceptions.RequestException as e:
                print(f"[RuleExtractor] stage '{stage}' request failed ({e}) — skipping.", file=sys.stderr)
                continue

            parsed = _parse_agent_rules(raw_text)
            added = 0
            for raw in parsed:
                norm = self._normalise(raw, stage)
                c = norm["check"]
                if c and c not in merged:
                    merged[c] = norm
                    added += 1

            if verbose:
                print(f"[RuleExtractor] stage '{stage}': {len(parsed)} block(s), +{added} new, {len(merged)} total", file=sys.stderr)

        if not merged:
            print("[RuleExtractor] No rules extracted.", file=sys.stderr)
        return list(merged.values())
