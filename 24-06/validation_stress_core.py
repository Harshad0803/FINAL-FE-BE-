"""
validation_stress_core.py — Stage 6 Stress Testing & Backtesting engine.

Ported from the Streamlit app's render_val_stress_backtesting() (app.py), which
relied on st.session_state to hold a pipeline + X_train/X_test/y_train/y_test/
y_proba produced once by Stage 4 Model Replication and reused across tabs.

This FastAPI backend is stateless between requests (see /validation/replication
in main.py — X_train/X_test/y_train/y_test/y_proba/y_pred are popped from the
result before it's returned to the frontend and nothing is cached server-side).
So instead of reading a replicated model out of session state, every function
here operates on a `rep_result` dict — the *raw* (un-stripped) return value of
`run_replication()` from val_replication_core.py, produced fresh within the
same request. main.py's /validation/stress/* endpoints call run_replication()
once per request and pass the result into the functions below.

Explicitly NOT ported: the "House Prices → LGD" directional test from the
Streamlit version. That check depended on a separately replicated LGD model
bundle (`lgd_model_bundle` / `lgd_features` in session state). LGD/EAD
pipelines have been removed from this codebase and are not being
reintroduced, so that check is dropped rather than stubbed.

Nothing in this module hardcodes pass/fail verdicts about a specific dataset —
every result is computed from whatever pipeline/data run_replication() hands
back for the caller's chosen dataset + algorithm.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

# ── Shared thresholds (ported 1:1 from app.py's inline constants) ───────────

SENSITIVITY_AUC_DROP_THRESHOLD = 0.05      # check 6.1
BACKTEST_GAP_PASS = 0.03                   # check 6.8
BACKTEST_GAP_WARN = 0.06
PSI_WARN = 0.10                            # check 6.4
PSI_FAIL = 0.25
DIRECTIONAL_NOISE_FLOOR = 1e-4             # checks 6.9a-c

FREQ_MAP = {
    "monthly": "ME",
    "quarterly": "QE",
    "half_yearly": "6ME",
    "yearly": "YE",
}

MACRO_SCENARIOS = [
    {
        "id": "6.2a", "name": "Adverse Scenario",
        "desc": "GDP \u22122%, unemployment +3pp, house prices \u221210%",
        "gdp_pct": -0.02, "unemployment_pp": 0.03, "house_price_pct": -0.10,
        "income_pct": 0.0, "pd_multiplier": 1.0,
    },
    {
        "id": "6.2b", "name": "Severe Scenario",
        "desc": "GDP \u22125%, unemployment +6pp, house prices \u221225%",
        "gdp_pct": -0.05, "unemployment_pp": 0.06, "house_price_pct": -0.25,
        "income_pct": 0.0, "pd_multiplier": 1.0,
    },
    {
        "id": "6.2c", "name": "COVID-style Shock",
        "desc": "Income \u221225%, forbearance spike, PD uplift \u00d72.5",
        "gdp_pct": 0.0, "unemployment_pp": 0.0, "house_price_pct": 0.0,
        "income_pct": -0.25, "pd_multiplier": 2.5,
    },
]

DIRECTIONAL_TESTS = [
    {
        "id": "6.9a", "driver": "Unemployment", "expected": "PD \u2191",
        "keywords": ["unemploy", "jobless"],
        "shock_desc": "+10% (relative)",
        "shock_fn": (lambda s: s * 1.10),
        "expect_increase": True,
    },
    {
        "id": "6.9b", "driver": "GDP", "expected": "PD \u2191",
        "keywords": ["gdp"],
        "shock_desc": "\u221210% (relative)",
        "shock_fn": (lambda s: s * 0.90),
        "expect_increase": True,
    },
    {
        "id": "6.9c", "driver": "Credit score", "expected": "PD \u2191",
        "keywords": ["credit_score", "creditscore", "fico", "bureau_score"],
        "shock_desc": "\u221210% (relative)",
        "shock_fn": (lambda s: s * 0.90),
        "expect_increase": True,
    },
]


# ── Small shared helpers ─────────────────────────────────────────────────────

def _find_driver_col(num_cols: list[str], keywords: list[str]) -> Optional[str]:
    for c in num_cols:
        cl = c.lower()
        if any(kw in cl for kw in keywords):
            return c
    return None


def detect_driver_columns(X: pd.DataFrame) -> dict[str, Optional[str]]:
    """Best-effort detection of macro/credit driver columns by name, mirroring
    the Streamlit version's keyword heuristics."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return {
        "gdp": _find_driver_col(num_cols, ["gdp"]),
        "unemployment": _find_driver_col(num_cols, ["unemploy", "jobless"]),
        "house_price": _find_driver_col(num_cols, [
            "house_price", "hpi", "home_price", "home_value",
            "property_price", "collateral_value",
        ]),
        "income": _find_driver_col(num_cols, ["income", "salary", "wage", "earnings"]),
    }


def _status_from_thresholds(value: float, warn_at: float, fail_at: float, higher_is_worse: bool = True) -> str:
    if higher_is_worse:
        if value >= fail_at:
            return "FAIL"
        if value >= warn_at:
            return "WARN"
        return "PASS"
    if value <= fail_at:
        return "FAIL"
    if value <= warn_at:
        return "WARN"
    return "PASS"


# ── 6.1 — Sensitivity analysis (ablation-based + manual shock) ──────────────

def get_ablation_sensitivity(ablation: Optional[dict]) -> dict:
    """Summarize the per-feature AUC-drop-on-removal results already computed
    by run_replication()'s ablation step. Returns the ranked table plus the
    6.1 check verdict — does NOT recompute ablation itself."""
    ablation = ablation or {}
    rows = [
        {"feature": k, "auc_drop": round(float(v), 4)}
        for k, v in ablation.items()
        if isinstance(v, (int, float)) and not np.isnan(v)
    ]
    rows.sort(key=lambda r: r["auc_drop"], reverse=True)
    rows = rows[:10]

    if not rows:
        return {
            "available": False,
            "rows": [],
            "check": {
                "id": "6.1", "title": "Sensitivity Analysis on Top Features",
                "status": "PENDING",
                "observed": "Ablation results not available.",
                "threshold": f"Max AUC drop per feature < {SENSITIVITY_AUC_DROP_THRESHOLD}",
                "source": "SS1/23 P4.3",
            },
        }

    top = rows[0]
    status = "PASS" if top["auc_drop"] < SENSITIVITY_AUC_DROP_THRESHOLD else "FAIL"
    return {
        "available": True,
        "rows": rows,
        "check": {
            "id": "6.1", "title": "Sensitivity Analysis on Top Features",
            "status": status,
            "observed": f"Largest AUC drop: {top['auc_drop']:.4f} when removing '{top['feature']}'",
            "threshold": f"Max AUC drop per feature < {SENSITIVITY_AUC_DROP_THRESHOLD}",
            "source": "SS1/23 P4.3",
        },
    }


def apply_manual_shock(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_proba: Any,
    feature: str,
    direction: str,
    magnitude_pct: float,
) -> dict:
    """Shock a single numeric feature by +/- magnitude_pct% across the whole
    test set and re-score. `direction` is "increase" or "decrease"."""
    if feature not in X_test.columns:
        raise ValueError(f"Unknown feature '{feature}'.")

    X_shocked = X_test.copy()
    multiplier = 1 + magnitude_pct / 100 if direction == "increase" else 1 - magnitude_pct / 100
    X_shocked[feature] = X_shocked[feature] * multiplier

    shocked_proba = pipeline.predict_proba(X_shocked)[:, 1]
    base_pd = float(np.array(y_proba).mean())
    shock_pd = float(shocked_proba.mean())
    pd_change = shock_pd - base_pd
    pd_change_pct = (pd_change / base_pd * 100) if base_pd > 0 else 0.0

    return {
        "feature": feature,
        "direction": direction,
        "magnitude_pct": magnitude_pct,
        "base_pd": base_pd,
        "shock_pd": shock_pd,
        "pd_change": pd_change,
        "pd_change_pct": pd_change_pct,
    }


# ── 6.2 — Macro stress scenarios ─────────────────────────────────────────────

def apply_macro_scenario(pipeline: Any, X_test: pd.DataFrame, y_proba: Any, scenario: dict) -> dict:
    driver_cols = detect_driver_columns(X_test)
    X_scn = X_test.copy()
    applied: list[str] = []

    gdp_col = driver_cols["gdp"]
    if gdp_col and scenario["gdp_pct"] != 0:
        X_scn[gdp_col] = X_scn[gdp_col] * (1 + scenario["gdp_pct"])
        applied.append(f"{gdp_col} x {1 + scenario['gdp_pct']:.2f}")

    unemp_col = driver_cols["unemployment"]
    if unemp_col and scenario["unemployment_pp"] != 0:
        is_fraction = X_scn[unemp_col].abs().max() <= 1.5
        delta = scenario["unemployment_pp"] / 100 if is_fraction else scenario["unemployment_pp"] * 100
        X_scn[unemp_col] = X_scn[unemp_col] + delta
        applied.append(f"{unemp_col} +{scenario['unemployment_pp']*100:.0f}pp")

    hp_col = driver_cols["house_price"]
    if hp_col and scenario["house_price_pct"] != 0:
        X_scn[hp_col] = X_scn[hp_col] * (1 + scenario["house_price_pct"])
        applied.append(f"{hp_col} x {1 + scenario['house_price_pct']:.2f}")

    inc_col = driver_cols["income"]
    if inc_col and scenario["income_pct"] != 0:
        X_scn[inc_col] = X_scn[inc_col] * (1 + scenario["income_pct"])
        applied.append(f"{inc_col} x {1 + scenario['income_pct']:.2f}")

    scn_proba = pipeline.predict_proba(X_scn)[:, 1]
    base_pd = float(np.array(y_proba).mean())
    scn_pd = float(np.clip(scn_proba.mean() * scenario["pd_multiplier"], 0, 1))
    if scenario["pd_multiplier"] != 1.0:
        applied.append(f"flat PD uplift x {scenario['pd_multiplier']}")

    pd_change = scn_pd - base_pd
    return {
        "id": scenario["id"],
        "name": scenario["name"],
        "desc": scenario["desc"],
        "applied": applied,
        "base_pd": base_pd,
        "scn_pd": scn_pd,
        "pd_change": pd_change,
        "pd_change_pct": (pd_change / base_pd * 100) if base_pd > 0 else 0.0,
    }


def run_all_macro_scenarios(pipeline: Any, X_test: pd.DataFrame, y_proba: Any) -> dict:
    driver_cols = detect_driver_columns(X_test)
    results = [apply_macro_scenario(pipeline, X_test, y_proba, scn) for scn in MACRO_SCENARIOS]
    return {
        "detected_drivers": {k: v for k, v in driver_cols.items() if v},
        "scenarios": results,
    }


# ── 6.4 — PSI stability (train vs test score distribution) ──────────────────

def compute_psi_stability(pipeline: Any, X_train: pd.DataFrame, y_proba: Any) -> dict:
    train_proba = pipeline.predict_proba(X_train)[:, 1]
    test_proba = np.array(y_proba)

    psi_min = min(train_proba.min(), test_proba.min())
    psi_max = max(train_proba.max(), test_proba.max())
    bins = np.linspace(psi_min, psi_max, 11)
    bin_labels = [f"{bins[i]:.3f}\u2013{bins[i+1]:.3f}" for i in range(len(bins) - 1)]

    tr_counts, _ = np.histogram(train_proba, bins=bins)
    te_counts, _ = np.histogram(test_proba, bins=bins)

    tr_pct = np.where(tr_counts == 0, 0.0001, tr_counts / len(train_proba))
    te_pct = np.where(te_counts == 0, 0.0001, te_counts / len(test_proba))

    psi_per_bin = (te_pct - tr_pct) * np.log(te_pct / tr_pct)
    psi_total = float(psi_per_bin.sum())

    status = _status_from_thresholds(psi_total, PSI_WARN, PSI_FAIL)

    bins_out = [
        {
            "bin": bin_labels[i],
            "train_pct": round(float(tr_pct[i]) * 100, 2),
            "test_pct": round(float(te_pct[i]) * 100, 2),
            "psi": round(float(psi_per_bin[i]), 5),
        }
        for i in range(len(bin_labels))
    ]

    return {
        "psi_total": round(psi_total, 4),
        "bins": bins_out,
        "check": {
            "id": "6.4", "title": "Score Distribution Stability (PSI)",
            "status": status,
            "observed": f"PSI = {psi_total:.4f}",
            "threshold": "PSI < 0.10 PASS, 0.10-0.25 WARN, > 0.25 FAIL",
            "source": "SS11/13 \u00a710.6",
        },
    }


# ── 6.8 — Backtesting (predicted PD vs realised default rate over time) ─────

def _detect_date_column(df: pd.DataFrame, hint: Optional[str] = None) -> Optional[str]:
    if hint and hint in df.columns:
        return hint
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        cl = c.lower()
        if "date" in cl or "time" in cl or "period" in cl:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() > 0.7:
                    return c
            except Exception:
                continue
    return None


def run_backtesting(
    val_df: Optional[pd.DataFrame],
    y_test: Any,
    y_proba: Any,
    freq_key: str = "quarterly",
    date_col: Optional[str] = None,
) -> dict:
    freq_code = FREQ_MAP.get(freq_key, "QE")

    if val_df is None:
        return {"available": False, "reason": "No dataset available for date lookup.", "date_col": None, "periods": [], "check": None}

    resolved_date_col = _detect_date_column(val_df, hint=date_col)
    if resolved_date_col is None:
        return {
            "available": False,
            "reason": "No date column detected in the uploaded validation dataset. Backtesting requires a datetime column.",
            "date_col": None,
            "periods": [],
            "check": None,
        }

    try:
        df_dates = val_df.copy()
        df_dates[resolved_date_col] = pd.to_datetime(df_dates[resolved_date_col], errors="coerce")

        y_arr = np.array(y_test)
        yp_arr = np.array(y_proba)
        # y_test keeps the original dataframe's row index (a random test
        # split, not the tail of the dataset), so dates must be looked up
        # by that index — .iloc[-len(y_arr):] previously grabbed whichever
        # rows happened to be last in val_df, misaligning every date with
        # the wrong prediction/actual pair and skewing the whole chart
        # toward whatever period those unrelated rows fell in.
        dates = df_dates[resolved_date_col].reindex(y_test.index)

        bt_df = pd.DataFrame({
            "date": dates.values,
            "actual": y_arr,
            "pred_pd": yp_arr,
        }).dropna(subset=["date"])
        bt_df = bt_df.set_index("date").sort_index()

        period = bt_df.resample(freq_code).agg(actual_dr=("actual", "mean"), avg_pred_pd=("pred_pd", "mean"))
        period = period.dropna()
        period["gap"] = (period["actual_dr"] - period["avg_pred_pd"]).abs()
        flagged = int((period["gap"] > 0.05).sum())

        periods_out = [
            {
                "period": idx.strftime("%Y-%m-%d"),
                "actual_dr": round(float(row["actual_dr"]), 4),
                "avg_pred_pd": round(float(row["avg_pred_pd"]), 4),
                "gap": round(float(row["gap"]), 4),
                "flagged": bool(row["gap"] > 0.05),
            }
            for idx, row in period.iterrows()
        ]

        mean_gap = float(period["gap"].mean()) if len(period) else 0.0
        max_gap = float(period["gap"].max()) if len(period) else 0.0
        status = _status_from_thresholds(mean_gap, BACKTEST_GAP_PASS, BACKTEST_GAP_WARN)

        return {
            "available": True,
            "date_col": resolved_date_col,
            "freq": freq_key,
            "periods": periods_out,
            "mean_gap": round(mean_gap, 4),
            "max_gap": round(max_gap, 4),
            "flagged_periods": flagged,
            "check": {
                "id": "6.8", "title": "Backtesting \u2014 PD vs Realised Default Rate",
                "status": status,
                "observed": f"Mean gap: {mean_gap:.4f} | Max gap: {max_gap:.4f} | Flagged periods: {flagged}",
                "threshold": "Mean gap < 3% PASS, < 6% WARN, >= 6% FAIL",
                "source": "SS11/13 \u00a710.3",
            },
        }
    except Exception as exc:
        return {"available": False, "reason": f"Backtesting error: {exc}", "date_col": resolved_date_col, "periods": [], "check": None}


# ── 6.9 — Directional testing ────────────────────────────────────────────────

def run_directional_tests(pipeline: Any, X_test: pd.DataFrame, y_proba: Any) -> list[dict]:
    num_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
    base_pd = float(np.array(y_proba).mean())
    results: list[dict] = []

    for t in DIRECTIONAL_TESTS:
        col = _find_driver_col(num_cols, t["keywords"])
        if col is None:
            results.append({
                "id": t["id"], "driver": t["driver"], "expected": t["expected"],
                "status": "SKIP",
                "note": f"No {t['driver'].lower()}-type column detected in the test set \u2014 this directional check cannot be run.",
                "source": "SS1/23 P4.3",
            })
            continue
        try:
            X_dir = X_test.copy()
            X_dir[col] = t["shock_fn"](X_dir[col])
            shocked_proba = pipeline.predict_proba(X_dir)[:, 1]
            new_pd = float(shocked_proba.mean())
            delta = new_pd - base_pd

            if abs(delta) < DIRECTIONAL_NOISE_FLOOR:
                status = "WARN"
            elif (delta > 0) == t["expect_increase"]:
                status = "PASS"
            else:
                status = "FAIL"

            results.append({
                "id": t["id"], "driver": t["driver"], "expected": t["expected"],
                "status": status, "column": col, "shock_desc": t["shock_desc"],
                "base_pd": base_pd, "new_pd": new_pd, "delta": delta,
                "source": "SS1/23 P4.3 \u00b7 SS3/18 \u00a72.1",
            })
        except Exception as exc:
            results.append({
                "id": t["id"], "driver": t["driver"], "expected": t["expected"],
                "status": "ERROR", "error": str(exc),
            })

    return results


# ── Orchestration ────────────────────────────────────────────────────────────

def run_stress_suite(rep_result: dict, val_df: Optional[pd.DataFrame], freq_key: str = "quarterly", date_col: Optional[str] = None) -> dict:
    """Runs every Stage 6 check that doesn't require interactive input
    (everything except the manual single-feature shock, which is a separate
    on-demand call — see run_manual_shock below)."""
    pipeline = rep_result.get("pipeline")
    X_train = rep_result.get("X_train")
    X_test = rep_result.get("X_test")
    y_test = rep_result.get("y_test")
    y_proba = rep_result.get("y_proba")
    ablation = rep_result.get("ablation")

    if pipeline is None or X_test is None or y_proba is None:
        return {
            "available": False,
            "reason": "Model replication did not return a usable pipeline/test set.",
        }

    sensitivity = get_ablation_sensitivity(ablation)
    macro = run_all_macro_scenarios(pipeline, X_test, y_proba)
    backtest = run_backtesting(val_df, y_test, y_proba, freq_key=freq_key, date_col=date_col)
    psi = compute_psi_stability(pipeline, X_train, y_proba) if X_train is not None else {
        "psi_total": None, "bins": [],
        "check": {"id": "6.4", "title": "Score Distribution Stability (PSI)", "status": "PENDING",
                  "observed": "Training data required (X_train) but not available.",
                  "threshold": "PSI < 0.10 PASS, 0.10-0.25 WARN, > 0.25 FAIL", "source": "SS11/13 \u00a710.6"},
    }
    directional = run_directional_tests(pipeline, X_test, y_proba)

    numeric_features = X_test.select_dtypes(include=[np.number]).columns.tolist()

    checks = [c for c in [sensitivity.get("check"), backtest.get("check"), psi.get("check")] if c]
    dir_fail = sum(1 for r in directional if r["status"] == "FAIL")
    dir_warn = sum(1 for r in directional if r["status"] == "WARN")
    # SKIP (no driver column detected) and ERROR (exception) results are
    # directional-test outcomes that can't be scored PASS/WARN/FAIL — they
    # get their own "na" bucket rather than being dropped from every counter,
    # so the summary tile's total always equals pass+warn+fail+pending+na.
    dir_na = sum(1 for r in directional if r["status"] in ("SKIP", "ERROR"))

    return {
        "available": True,
        "numeric_features": numeric_features,
        "sensitivity": sensitivity,
        "macro_scenarios": macro,
        "backtest": backtest,
        "psi": psi,
        "directional": directional,
        "summary": {
            "checks": checks,
            "pass": sum(1 for c in checks if c["status"] == "PASS") + sum(1 for r in directional if r["status"] == "PASS"),
            "warn": sum(1 for c in checks if c["status"] == "WARN") + dir_warn,
            "fail": sum(1 for c in checks if c["status"] == "FAIL") + dir_fail,
            "pending": sum(1 for c in checks if c["status"] == "PENDING"),
            "na": dir_na,
        },
    }


def run_manual_shock(rep_result: dict, feature: str, direction: str, magnitude_pct: float) -> dict:
    pipeline = rep_result.get("pipeline")
    X_test = rep_result.get("X_test")
    y_proba = rep_result.get("y_proba")
    if pipeline is None or X_test is None or y_proba is None:
        raise ValueError("Model replication did not return a usable pipeline/test set.")
    return apply_manual_shock(pipeline, X_test, y_proba, feature, direction, magnitude_pct)
