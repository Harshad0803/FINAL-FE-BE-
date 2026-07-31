"""
agent2.py
Model-validation compliance checker.

Loads MDD documentation rules from rag_store/val_mdd_rules.json and provides:
  - check_for_validation()  : quantitative rules cross-checked against dataset metrics
  - check_mdd_keywords()    : keyword-search cross-check of rules against uploaded MDD text
  - get_attestation_checklist() : descriptive rules for manual reviewer sign-off

The model-development pipeline (data/feature/training/evaluation stages) no
longer uses this class — it has been removed. All compliance logic is now
focused on model validation only.
"""

import json
import re
from pathlib import Path
from typing import Any

from llm_providers import LLMProviderError, complete_with_fallback

# Prompt template for the LLM-based conceptual-validation check. Kept out of
# application code — see check_documents_with_llm() below.
_LLM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "conceptual_validation_prompt.txt"

_FRAMEWORK_ALIASES = {
    "IFRS9": ["ifrs 9", "ifrs9"],
    "SS1/23": ["ss1/23"],
    "RBI": ["rbi", "chapter iv", "regulatory principles for model risk management"],
}


def _rule_matches_frameworks(rule: dict, frameworks: list[str] | None) -> bool:
    """Return True if a rule's source/principle matches any selected framework.
    Checks both `source` and `principle` fields (not just `source`) because
    _infer_source() currently mislabels some RBI rules as IFRS9 — matching on
    principle text (e.g. "Chapter IV Section 26(1)" vs "B5.5.28") avoids that
    bug without touching the classifier itself, per explicit instruction not
    to fix _infer_source().
    """
    if not frameworks:
        return True
    source = str(rule.get("source", "")).lower()
    principle = str(rule.get("principle", "")).lower()
    for fw in frameworks:
        aliases = _FRAMEWORK_ALIASES.get(fw, [fw.lower()])
        if any(alias in source or alias in principle for alias in aliases):
            return True
    return False


class Agent2:
    def __init__(self, rules_path: str = "rag_store/val_mdd_rules.json"):
        path = Path(rules_path)
        if not path.exists():
            # Graceful degradation — validation UI still renders without rules
            self.rules: list[dict] = []
        else:
            with path.open(encoding="utf-8") as f:
                self.rules = json.load(f)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_field(self, field_hint: str, context_dict: dict) -> Any:
        """Dot-notation field resolver: 'missing_rate' → context_dict['missing_rate']."""
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
        """Return True if the check PASSES (no flag needed)."""
        try:
            if operator == ">=":   return float(value) >= float(threshold)
            if operator == "<=":   return float(value) <= float(threshold)
            if operator == ">":    return float(value) >  float(threshold)
            if operator == "<":    return float(value) <  float(threshold)
            if operator == "==":   return value == threshold
            if operator == "!=":   return value != threshold
            if operator == "is_true":    return bool(value)
            if operator == "is_false":   return not bool(value)
            if operator == "is_present": return value is not None
        except (TypeError, ValueError):
            pass
        return True  # unknown operator → don't flag

    # ── Quantitative validation checks ───────────────────────────────────────

    def check_for_validation(self, stage: str, context_dict: dict, frameworks: list[str] | None = None) -> list[dict]:
        """
        Evaluate machine-testable rules (checkable_against_data=True) for a
        model-validation stage against supplied dataset metrics.

        Returns findings shaped for the app.py render loop:
          {check_id, title, severity, status, source, principle, observed, threshold, detail}
        """
        findings: list[dict] = []
        normalized_stage = self._normalize_stage(stage)
        for rule in self.rules:
            if not self._stages_match(rule.get("stage"), normalized_stage):
                continue
            if not _rule_matches_frameworks(rule, frameworks):
                continue
            if not rule.get("checkable_against_data", False):
                continue
            logic = rule.get("logic")
            if not logic or not isinstance(logic, dict):
                continue
            field_hint = logic.get("field_hint", "")
            if not field_hint:
                continue
            value = self._resolve_field(field_hint, context_dict)
            if value is None:
                continue
            passed = self._apply_operator(value, logic.get("operator", ""), logic.get("threshold"))
            findings.append({
                "check_id":  rule.get("id", rule.get("check", "?")),
                "title":     rule.get("rule", rule.get("flag", "")),
                "severity":  rule.get("severity", "medium"),
                "status":    "PASS" if passed else "FAIL",
                "source":    rule.get("source", ""),
                "principle": rule.get("principle", ""),
                "observed":  str(value),
                "threshold": str(logic.get("threshold", "")),
                "detail":    rule.get("suggestion", ""),
            })
        return findings

    @staticmethod
    def _normalize_stage(stage: str | None) -> str:
        """Normalize stage names so legacy and current names are treated as equivalent."""
        if not stage:
            return ""
        normalized = str(stage).strip().lower()
        return {
            "data": "data_validation",
            "data_validation": "data_validation",
            "conceptual": "conceptual_soundness",
            "conceptual_soundness": "conceptual_soundness",
        }.get(normalized, normalized)

    @classmethod
    def _stages_match(cls, rule_stage: Any, requested_stage: str) -> bool:
        """Return True when a rule stage matches the requested stage, including legacy aliases."""
        if not requested_stage:
            return True
        normalized_rule_stage = cls._normalize_stage(rule_stage)
        return normalized_rule_stage == requested_stage

    @staticmethod
    def _map_stage(stage: str) -> str:
        """Map legacy stage names to current ones for callers that still use the old helper."""
        return Agent2._normalize_stage(stage)

    # ── MDD keyword-search cross-check ────────────────────────────────────────

    def check_mdd_keywords(
        self,
        mdd_text: str,
        stage: str | None = None,
        frameworks: list[str] | None = None,
    ) -> list[dict]:
        """
        Keyword-search the uploaded MDD text against each rule's `keywords` list.

        For each rule, counts how many of its keywords appear in the MDD.
        Returns a result dict per rule shaped for the app.py render loop:
          {check_id, title, severity, status, source, principle, observed, threshold, detail}

        status = PASS  if keyword hits >= min_keyword_hits
               = WARN  if some keywords hit but below threshold, or no keywords defined
               = FAIL  if zero keyword hits for a rule with keywords
        """
        if not mdd_text:
            return []

        mdd_lower = mdd_text.lower()
        results: list[dict] = []

        for rule in self.rules:
            if stage is not None:
                rule_stage = rule.get("stage", "")
                if rule_stage not in (stage, self._map_stage(stage)):
                    continue

            if not _rule_matches_frameworks(rule, frameworks):
                continue

            # Skip machine-testable rules — those go through check_for_validation
            if rule.get("checkable_against_data", False):
                continue

            keywords: list[str] = rule.get("keywords", [])
            min_hits: int = rule.get("min_keyword_hits", 2)

            if not keywords:
                status   = "WARN"
                observed = "No keywords defined — manual review required"
            else:
                hits = [kw for kw in keywords if kw.lower() in mdd_lower]
                n_hits = len(hits)
                if n_hits >= min_hits:
                    status = "PASS"
                elif n_hits > 0:
                    status = "WARN"
                else:
                    status = "FAIL"
                observed = (
                    f"{n_hits}/{len(keywords)} keywords found: {', '.join(hits)}"
                    if hits else f"0/{len(keywords)} keywords found"
                )

            results.append({
                "check_id":  rule.get("id", rule.get("check", "?")),
                "title":     rule.get("rule", rule.get("flag", "")),
                "severity":  rule.get("severity", "medium"),
                "status":    status,
                "source":    rule.get("source", ""),
                "principle": rule.get("principle", ""),
                "observed":  observed,
                "threshold": f"≥ {min_hits} keyword(s)" if keywords else "N/A",
                "detail":    rule.get("suggestion", ""),
            })

        return results

    # ── LLM-based conceptual-validation cross-check ───────────────────────────
    # Replaces keyword-search (check_mdd_keywords, above) as the conceptual /
    # qualitative-rule check: instead of counting keyword hits, an LLM reads
    # the document(s) and judges whether each rule's intent is actually
    # satisfied. check_mdd_keywords is left in place, unchanged, for any
    # other caller that still relies on it.

    def check_documents_with_llm(
        self,
        docs: dict[str, str],
        stage: str | None = None,
        only_rule_ids: list[str] | None = None,
        frameworks: list[str] | None = None,
    ) -> list[dict]:
        """
        LLM-based conceptual/qualitative rule check.

        docs: {"MDD": mdd_text, ...} — document name -> full text.
        stage: optional stage filter, same semantics as check_mdd_keywords.
        only_rule_ids: optional allow-list of rule ids to check.

        Provider: Deloitte Agent (primary), automatically falling back to
        Ollama on any connection error, timeout, or HTTP failure — see
        llm_providers.py. Prompt template lives at
        prompts/conceptual_validation_prompt.txt.

        Returns findings in the same shape as check_for_validation /
        check_mdd_keywords (shaped for the app.py render loop), plus two
        extra fields the UI already understands: `check_source`
        ("llm" normally, "llm_error" if both providers failed) and
        `reasoning`.
        """
        print("[TRACE] Entered check_documents_with_llm")
        candidate_rules: list[dict] = []
        normalized_stage = self._normalize_stage(stage)
        print("[TRACE] About to start processing rules")
        for rule in self.rules:
            # Machine-testable rules stay on check_for_validation — never
            # routed to the LLM.
            if rule.get("checkable_against_data", False):
                continue
            if stage is not None:
                if not self._stages_match(rule.get("stage"), normalized_stage):
                    continue
            if not _rule_matches_frameworks(rule, frameworks):
                continue
            rule_id = rule.get("id", rule.get("check", "?"))
            if only_rule_ids and str(rule_id) not in only_rule_ids:
                continue
            candidate_rules.append(rule)

        if not candidate_rules:
            return []

        doc_text = "\n\n".join(f"### {name}\n{text}" for name, text in docs.items() if text)

        early_return = not doc_text

        if early_return:
            # No document uploaded yet — surface every candidate rule as
            # PENDING so the "RAG Agent Rules" column still lists them
            # (mirrors check_mdd_keywords' behaviour with empty mdd_text).
            return [
                self._llm_finding(rule, status="PENDING", observed="No document uploaded yet.",
                                   reasoning="", check_source="llm")
                for rule in candidate_rules
            ]

        # Batch rules into smaller groups instead of one prompt for every
        # rule in the stage. A stage like Conceptual Soundness can have more
        # qualitative rules than Data Validation; cramming them all into one
        # prompt makes the response slower and more likely to be truncated
        # (which surfaces as an empty/invalid JSON response). Smaller
        # batches mean smaller, faster completions, and a failure in one
        # batch doesn't wipe out the rules in the others.
        results: list[dict] = []
        try:
            prompt = self._build_llm_prompt(candidate_rules, doc_text)
            print("[TRACE] About to call complete_with_fallback")
            completion, provider_used = complete_with_fallback(prompt)
            print("[TRACE] Returned from complete_with_fallback")

            verdicts_by_id, verdicts_by_index, verdicts_in_order = self._parse_llm_verdicts(completion)

        except (LLMProviderError, ValueError) as exc:
            # Providers failed, response empty, or invalid JSON — return WARN for all rules
            for rule in candidate_rules:
                results.append(self._llm_finding(
                    rule, status="WARN", observed=f"LLM check unavailable: {exc}",
                    reasoning="", check_source="llm_error",
                ))
            return results

        # Sanity check: the number of verdicts should match the number of rules
        if len(verdicts_in_order) != len(candidate_rules):
            diag = (
                f"LLM ({provider_used}) returned {len(verdicts_in_order)} verdict(s) for a "
                f"{len(candidate_rules)}-rule request — count mismatch suggests the response isn't "
                f"tracking this prompt (stale/cached/unrelated response from the provider), "
                f"not a dropped item. Raw response: {completion[:400]!r}"
            )
            for rule in candidate_rules:
                results.append(self._llm_finding(
                    rule, status="WARN", observed=diag,
                    reasoning="", check_source="llm_error",
                ))
            return results

        # Map each verdict back to its rule using rule_id, index, or response order
        for idx, rule in enumerate(candidate_rules):
            rule_id = str(rule.get("id", rule.get("check", "?")))
            verdict = verdicts_by_id.get(rule_id)
            if verdict is None:
                verdict = verdicts_by_index.get(idx)
            if verdict is None:
                verdict = verdicts_in_order[idx]
            if verdict is None:
                got_ids = [str(v.get("rule_id", v.get("id", "?"))) for v in verdicts_in_order]
                diag = (
                    f"LLM ({provider_used}) returned {len(verdicts_in_order)} verdict(s) for this "
                    f"{len(candidate_rules)}-rule batch, none matching rule_id '{rule_id}' or index {idx}. "
                    f"IDs seen: {got_ids[:8]}"
                )
                results.append(self._llm_finding(
                    rule, status="WARN", observed=diag,
                    reasoning="", check_source="llm_error",
                ))
                continue

            status = str(verdict.get("status", "WARN")).upper()
            if status not in ("PASS", "WARN", "FAIL"):
                status = "WARN"
            evidence = str(verdict.get("evidence") or "").strip()
            if not evidence:
                evidence = "Not found" if status == "FAIL" else "LLM gave a verdict but no supporting evidence text."
            results.append(self._llm_finding(
                rule, status=status, observed=evidence,
                reasoning=verdict.get("reasoning", ""), check_source="llm",
            ))
        return results

    # Rules per LLM call. Smaller batches -> faster, more reliable
    # completions (less likely for the model to drop or garble an item) at
    # the cost of more round-trips; tune if you have very large rule sets
    # or a slow provider.
    _LLM_BATCH_SIZE = 1

    @staticmethod
    def _llm_finding(rule: dict, status: str, observed: str, reasoning: str, check_source: str) -> dict:
        return {
            "check_id":       rule.get("id", rule.get("check", "?")),
            "title":          rule.get("rule", rule.get("flag", "")),
            "severity":       rule.get("severity", "medium"),
            "status":         status,
            "source":         rule.get("source", ""),
            "principle":      rule.get("principle", ""),
            "observed":       observed,
            "observed_value": observed,
            "threshold":      "LLM conceptual review",
            "detail":         rule.get("suggestion", ""),
            "check_source":   check_source,
            "reasoning":      reasoning,
        }

    @staticmethod
    def _build_llm_prompt(rules: list[dict], doc_text: str) -> str:
        template = _LLM_PROMPT_PATH.read_text(encoding="utf-8")
        rules_payload = [
            {
                "index":            i,
                "rule_id":          str(r.get("id", r.get("check", "?"))),
                "statement":        r.get("rule", r.get("flag", "")),
                "principle":        r.get("principle", ""),
                "mdd_section_hint": r.get("mdd_section_hint", ""),
            }
            for i, r in enumerate(rules)
        ]
        return (
            template
            .replace("{{RULE_COUNT}}", str(len(rules)))
            .replace("{{RULE_INDEXES}}", ", ".join(str(i) for i in range(len(rules))))
            .replace("{{RULES_JSON}}", json.dumps(rules_payload, indent=2))
            .replace("{{DOCUMENT_TEXT}}", doc_text)
        )

    @staticmethod
    def _parse_llm_verdicts(completion: str) -> tuple[dict, dict, list]:
        """Parse the LLM's JSON response into ({rule_id: verdict}, {index: verdict}, [verdict, ...]).

        The `index` map is keyed by the 0-based "index" field the prompt
        asks each item to echo back (see _build_llm_prompt). It's the most
        reliable way to line a verdict back up with its rule: unlike
        `rule_id` (an opaque string models sometimes drop, renumber, or
        reformat) or plain response order (which breaks the moment the
        model skips one item mid-batch), an explicit index survives both a
        renamed/missing rule_id AND a dropped item elsewhere in the batch.

        The ordered list is a last-resort fallback for callers when neither
        rule_id nor index matching finds anything and the counts line up.

        Raises ValueError with a message that distinguishes an empty
        response (provider answered but produced nothing — usually means
        the request was cut off, so raising the relevant timeout env var
        is the fix) from a malformed/non-JSON response (a real parsing
        problem, not a timing one).

        Tolerant of several common deviations from the requested shape,
        since models don't always follow instructions exactly:
          - a JSON object wrapping the array under a key, e.g.
            {"results": [...]} / {"verdicts": [...]} instead of a bare array.
          - items using "id" instead of "rule_id".
          - a flat map straight from rule_id to a status string, e.g.
            {"V01": "PASS", "V02": "FAIL"} instead of {"V01": {"status": ...}}.
          - a bare list of status strings in rule order, e.g. ["PASS", "FAIL"].
        """
        text = completion.strip()
        if not text:
            raise ValueError(
                "LLM returned an empty response — this usually means the "
                "request was cut off before the model finished (raise "
                "DELOITTE_TIMEOUT_SECONDS / OLLAMA_TIMEOUT_SECONDS), rather "
                "than a genuine rule-evaluation failure."
            )
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM response was not valid JSON ({exc}); got: {text[:200]!r}") from exc

        def as_verdict(value: Any) -> dict | None:
            """Normalize one response entry into a {status, evidence, reasoning}-shaped dict."""
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                # Bare status string, e.g. "PASS" — no evidence/reasoning given.
                return {"status": value}
            return None

        if isinstance(data, dict) and not any(isinstance(v, (dict, str)) for v in data.values()):
            # Looks like {"results": [...]} rather than {rule_id: verdict} —
            # unwrap the first list-valued key we find.
            for value in data.values():
                if isinstance(value, list):
                    data = value
                    break

        if isinstance(data, dict) and "status" in data:
            # A single verdict object returned bare instead of wrapped in a
            # list — e.g. the model only answered one rule of a multi-rule
            # batch and skipped the "[ ]" wrapper. Without this check the
            # branch below would misread THIS object's own fields
            # (index/rule_id/status/evidence/reasoning) as if each were a
            # separate rule_id -> verdict entry in a flat map, producing
            # several bogus verdicts with no real rule_id. Detected by
            # "status" being a top-level key: a genuine rule_id -> verdict
            # map would use rule_ids as keys, not "status" itself.
            data = [data]

        def index_of(item: Any) -> int | None:
            if not isinstance(item, dict):
                return None
            idx = item.get("index")
            try:
                return int(idx) if idx is not None else None
            except (TypeError, ValueError):
                return None

        if isinstance(data, list):
            items = []
            by_id = {}
            by_index = {}
            for item in data:
                verdict = as_verdict(item)
                if verdict is None:
                    continue
                items.append(verdict)
                if isinstance(item, dict):
                    rid = item.get("rule_id", item.get("id", item.get("ruleId")))
                    if rid is not None:
                        by_id[str(rid)] = verdict
                    idx = index_of(item)
                    if idx is not None:
                        by_index[idx] = verdict
            return by_id, by_index, items
        if isinstance(data, dict):
            # Flat map: rule_id -> verdict (object) or rule_id -> "STATUS" string.
            by_id = {}
            by_index = {}
            items = []
            for k, v in data.items():
                verdict = as_verdict(v)
                if verdict is None:
                    continue
                by_id[str(k)] = verdict
                idx = index_of(v)
                if idx is not None:
                    by_index[idx] = verdict
                items.append(verdict)
            return by_id, by_index, items
        return {}, {}, []

    # ── Attestation checklist (manual reviewer sign-off) ─────────────────────

    def get_attestation_checklist(self, stage: str | None = None) -> list[dict]:
        """
        Return all descriptive (non-machine-testable) rules for reviewer sign-off,
        optionally filtered by stage.
        """
        items: list[dict] = []
        for r in self.rules:
            if r.get("checkable_against_data", False):
                continue
            if stage is not None and r.get("stage") not in (stage, self._map_stage(stage)):
                continue
            items.append({
                "rule_id":     r.get("id", "?"),
                "source":      r.get("source", ""),
                "principle":   r.get("principle", ""),
                "stage":       r.get("stage", ""),
                "severity":    r.get("severity", "medium"),
                "statement":   r.get("rule", r.get("flag", "")),
                "mdd_section": r.get("mdd_section_hint", ""),
                "keywords":    r.get("keywords", []),
                "suggestion":  r.get("suggestion", ""),
            })
        return items

    # ── Summary helpers ───────────────────────────────────────────────────────

    def rules_for_stage(self, stage: str) -> list[dict]:
        """Return all rules for a given validation stage."""
        return [r for r in self.rules if r.get("stage") in (stage, self._map_stage(stage))]

    @property
    def total_rules(self) -> int:
        return len(self.rules)
