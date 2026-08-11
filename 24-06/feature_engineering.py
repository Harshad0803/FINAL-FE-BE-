"""
feature_engineering.py - Adaptive Feature Engineering Engine
Analyzes the dataset and auto-applies relevant transformations.
"""

import re
import itertools
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Single source of truth for skew-driven transform recommendations — see
# preprocessing_new.recommend_transform() docstring. Previously this module used
# its own inline |skew| > 1.5 threshold, inconsistent with preprocessing_new.py's
# separate |skew| > 1.0 threshold; both now go through the same function.
from preprocessing_new import recommend_transform


MAX_DIAGNOSTIC_ROWS = 20000
MAX_IV_FEATURES = 40
MAX_WOE_FEATURES = 8
EPS = 0.5

# —— Interaction engine configuration (Design Requirement #3: all thresholds
# are constants here, not buried in logic, so they're easy to tune). See
# build_interaction_features() below. Phase numbers in comments throughout
# this module refer to the 8-phase interaction discovery workflow:
#   1. Rank all candidate features (IV, MI fallback)
#   2. Generate candidate interactions (numeric×numeric, numeric×WOE, WOE×WOE,
#      plus domain-driven banking interactions)
#   4. Evaluate every interaction (IV, Gini)
#   5. Keep only useful interactions (IV threshold, capped count)
#   6. Remove redundant interactions (correlation vs existing/other interactions)
#   7. Store as plan-ready metadata
#   8. Apply only the approved interactions, unchanged, to train/val/test
TOP_FEATURES_FOR_INTERACTIONS = 8   # candidates are drawn ONLY from the top-N ranked
                                     # features — never a full cross of every feature
MAX_INTERACTIONS = 10               # hard cap on how many interactions survive into the plan
MIN_INTERACTION_IV = 0.02           # minimum IV for an interaction to be considered useful
MAX_CATEGORY_CARDINALITY = 8        # WOE x WOE only generated for low-cardinality pairs
CORRELATION_THRESHOLD = 0.90        # redundancy filter, vs existing features AND vs other interactions

# —— EAD calculation — lives here directly ————————————————————————————————
# Previously this logic was moved out to ecl_engine.py and re-exported here.
# Now that the ECL engine has been removed from the POC entirely, these
# functions are inlined directly in feature_engineering.py (their original
# home) instead of importing them from a module that no longer exists.
# Nothing about the logic itself has changed.

_OUTSTANDING_BALANCE_SYNONYMS = [
    "outstanding_balance", "outstanding_principal", "current_balance",
    "book_balance", "loan_balance", "principal_outstanding",
    "outstanding_amount", "balance_outstanding", "current_principal", "outstanding",
]
_LOAN_AMOUNT_SYNONYMS = [
    "loan_amount", "total_loan_amount", "original_loan_amount", "sanctioned_amount",
    "disbursed_amount", "loan_principal", "original_principal", "principal", "loan_amt",
]
_INTEREST_RATE_SYNONYMS = [
    "interest_rate", "int_rate", "interest", "apr", "coupon",
    "annual_rate", "nominal_rate", "rate",
]
_YEARS_ELAPSED_SYNONYMS = [
    "years_elapsed", "years_on_book", "loan_age_years", "age_years",
    "seasoning_years", "elapsed_years", "time_on_book_years",
]
_MONTHS_ELAPSED_SYNONYMS = [
    "months_on_book", "loan_age_months", "age_months", "seasoning_months",
    "months_elapsed", "mob", "time_on_book",
]
_TERM_SYNONYMS = [
    "term_years", "loan_term_years", "tenure_years", "maturity_years",
    "original_term", "loan_term", "tenure", "term",
]
_PAYMENT_FREQUENCY_SYNONYMS = [
    "payment_frequency", "repayment_frequency", "installment_frequency",
    "pay_frequency", "freq", "frequency",
]

# Number of payment periods per year, by frequency label.
PAYMENT_FREQUENCIES: Dict[str, int] = {
    "monthly": 12,
    "quarterly": 4,
    "semi-annual": 2,
    "annual": 1,
}
DEFAULT_PAYMENT_FREQUENCY = "monthly"


def _match_column(columns, synonyms):
    norm = {_norm_name(c): c for c in columns}
    for syn in synonyms:                       # exact (normalized) match first
        if _norm_name(syn) in norm:
            return norm[_norm_name(syn)]
    for syn in synonyms:                       # then substring match
        ns = _norm_name(syn)
        for nc, orig in norm.items():
            if ns and ns in nc:
                return orig
    return None


def detect_outstanding_balance_col(df):
    return _match_column(df.columns, _OUTSTANDING_BALANCE_SYNONYMS)


def detect_loan_amount_col(df):
    return _match_column(df.columns, _LOAN_AMOUNT_SYNONYMS)


def detect_interest_rate_col(df):
    return _match_column(df.columns, _INTEREST_RATE_SYNONYMS)


def detect_years_elapsed_col(df):
    """Return (column, is_in_months)."""
    col = _match_column(df.columns, _YEARS_ELAPSED_SYNONYMS)
    if col:
        return col, False
    col = _match_column(df.columns, _MONTHS_ELAPSED_SYNONYMS)
    if col:
        return col, True
    return None, False


def detect_term_col(df):
    return _match_column(df.columns, _TERM_SYNONYMS)


def detect_payment_frequency_col(df):
    return _match_column(df.columns, _PAYMENT_FREQUENCY_SYNONYMS)


def compute_outstanding_balance(loan_amount, interest_rate, years_elapsed,
                                term_years=None, rate_is_percent=None):
    """
    Outstanding balance per loan from loan amount, annual interest rate and years
    elapsed. With a loan term, an amortizing (declining) balance is used:

        B = P * ((1+r)^N - (1+r)^m) / ((1+r)^N - 1)     (m = elapsed, N = term)

    Without a term, an interest-accrual balance is used: B = P * (1+r)^m.
    The interest rate is auto-normalised from percent to a fraction when needed.
    All inputs are pandas Series sharing one index.

    NOTE: This is the legacy continuous-time balance function, retained for
    backward compatibility with any code that still calls it directly. For EAD
    estimation, use compute_ead() below, which models a real discrete payment
    schedule (monthly/quarterly/semi-annual/annual) and avoids double-counting
    between the amortizing balance and the accrued-interest add-on.
    """
    P = pd.to_numeric(loan_amount, errors="coerce").astype(float)
    r = pd.to_numeric(interest_rate, errors="coerce").astype(float)
    t = pd.to_numeric(years_elapsed, errors="coerce").astype(float)
    idx = P.index

    P = P.fillna(0.0).clip(lower=0)
    nonzero = r.replace(0, np.nan).dropna()
    auto_percent = bool(len(nonzero)) and float(nonzero.median()) > 1.0
    if rate_is_percent is True or (rate_is_percent is None and auto_percent):
        r = r / 100.0
    r = r.fillna(0.0).clip(lower=0)
    t = t.fillna(0.0).clip(lower=0)

    if term_years is not None:
        N = pd.to_numeric(term_years, errors="coerce").reindex(idx).astype(float)
        N = N.where(N > 0).fillna(t.clip(lower=1.0))
        m = np.minimum(t, N)
        one_plus_r = 1.0 + r
        factor_N = one_plus_r ** N
        factor_m = one_plus_r ** m
        denom = (factor_N - 1.0).replace(0, np.nan)
        amort = P * (factor_N - factor_m) / denom
        straight = P * (1.0 - (m / N.replace(0, np.nan)).clip(0, 1))
        bal = amort.where(r > 1e-9, straight).fillna(straight)
    else:
        bal = P * (1.0 + r) ** t

    return bal.clip(lower=0).rename("outstanding_balance")


def _periods_per_year(payment_frequency: str) -> int:
    """Map a payment frequency label to the number of payment periods per year."""
    key = str(payment_frequency).strip().lower()
    if key not in PAYMENT_FREQUENCIES:
        raise ValueError(
            f"Unknown payment_frequency '{payment_frequency}'. "
            f"Must be one of {list(PAYMENT_FREQUENCIES.keys())}."
        )
    return PAYMENT_FREQUENCIES[key]


def _amortizing_balance_after_k_payments(P, i_period, n_periods, k):
    """
    Standard level-payment amortization formula (vectorised, pandas/numpy).

    Returns the outstanding principal immediately after the k-th payment, using:

        B(k) = P * [(1+i)^n - (1+i)^k] / [(1+i)^n - 1]      (i > 0)
        B(k) = P * (1 - k/n)                                 (i == 0, straight-line)
    """
    one_i = 1.0 + i_period
    factor_n = one_i ** n_periods
    factor_k = one_i ** k
    denom = (factor_n - 1.0).replace(0, np.nan)
    amort = P * (factor_n - factor_k) / denom
    n_safe = n_periods.replace(0, np.nan)
    straight = P * (1.0 - (k / n_safe).clip(0, 1))
    return amort.where(i_period > 1e-12, straight).fillna(straight)


def compute_ead_schedule(
    loan_amount,
    interest_rate,
    years_elapsed,
    term_years,
    payment_frequency="monthly",
    rate_is_percent=None,
) -> Dict[str, pd.Series]:
    """
    Payment-schedule-based EAD engine.

        EAD = Outstanding Principal (as of the last completed payment)
              + Accrued Interest (since that last payment, for the elapsed
                fraction of the CURRENT payment period only)

    Returns a dict of aligned pandas Series: payment_frequency, periods_per_year,
    n_periods, periods_elapsed, completed_periods, period_fraction_elapsed,
    periodic_rate, annual_rate, outstanding_principal, accrued_interest, ead.
    """
    P = pd.to_numeric(loan_amount, errors="coerce").astype(float)
    idx = P.index
    P = P.fillna(0.0).clip(lower=0)

    r_annual = pd.to_numeric(interest_rate, errors="coerce").astype(float).reindex(idx)
    nonzero = r_annual.replace(0, np.nan).dropna()
    auto_percent = bool(len(nonzero)) and float(nonzero.median()) > 1.0
    if rate_is_percent is True or (rate_is_percent is None and auto_percent):
        r_annual = r_annual / 100.0
    r_annual = r_annual.fillna(0.0).clip(lower=0)

    t = pd.to_numeric(years_elapsed, errors="coerce").astype(float).reindex(idx).fillna(0.0).clip(lower=0)
    N_years = pd.to_numeric(term_years, errors="coerce").astype(float).reindex(idx)
    N_years = N_years.where(N_years > 0).fillna(t.clip(lower=1.0))

    # —— Payment frequency: scalar string or a per-row Series of labels ——————
    if isinstance(payment_frequency, pd.Series):
        freq_labels = payment_frequency.reindex(idx).fillna(DEFAULT_PAYMENT_FREQUENCY)
        freq_labels = freq_labels.astype(str).str.strip().str.lower()
        unknown = ~freq_labels.isin(PAYMENT_FREQUENCIES.keys())
        if unknown.any():
            freq_labels = freq_labels.where(~unknown, DEFAULT_PAYMENT_FREQUENCY)
        periods_per_year = freq_labels.map(PAYMENT_FREQUENCIES).astype(float)
    else:
        f = _periods_per_year(payment_frequency)
        freq_labels = pd.Series([str(payment_frequency).strip().lower()] * len(idx), index=idx)
        periods_per_year = pd.Series(float(f), index=idx)

    n_periods = (N_years * periods_per_year).round().clip(lower=1)
    periodic_rate = r_annual / periods_per_year

    periods_elapsed = (t * periods_per_year).clip(lower=0)
    periods_elapsed = np.minimum(periods_elapsed, n_periods)  # cap at maturity
    k = np.floor(periods_elapsed)
    frac = (periods_elapsed - k).clip(0, 1)

    outstanding_principal = _amortizing_balance_after_k_payments(
        P, periodic_rate, n_periods, k
    ).clip(lower=0)
    accrued_interest = (outstanding_principal * periodic_rate * frac).clip(lower=0)
    ead = (outstanding_principal + accrued_interest).clip(lower=0)

    return {
        "payment_frequency": freq_labels.rename("payment_frequency"),
        "periods_per_year": periods_per_year.rename("periods_per_year"),
        "n_periods": n_periods.rename("n_periods"),
        "periods_elapsed": periods_elapsed.rename("periods_elapsed"),
        "completed_periods": k.rename("completed_periods"),
        "period_fraction_elapsed": frac.rename("period_fraction_elapsed"),
        "periodic_rate": periodic_rate.rename("periodic_rate"),
        "annual_rate": r_annual.rename("annual_rate"),
        "outstanding_principal": outstanding_principal.rename("outstanding_principal"),
        "accrued_interest": accrued_interest.rename("accrued_interest"),
        "ead": ead.rename("ead"),
    }


def compute_ead(
    loan_amount,
    interest_rate,
    years_elapsed,
    term_years,
    payment_frequency="monthly",
    rate_is_percent=None,
) -> pd.Series:
    """Convenience wrapper around compute_ead_schedule() that returns just the
    final EAD series (outstanding_principal + accrued_interest)."""
    return compute_ead_schedule(
        loan_amount, interest_rate, years_elapsed, term_years,
        payment_frequency=payment_frequency, rate_is_percent=rate_is_percent,
    )["ead"]


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# —— Origination-PD columns must be hidden from model development (Change 3) ————
_ORIG_PD_SIGNATURES = [
    "originationpd", "origpd", "pdorig", "pdatorigination",
    "pdorigination", "initialpd", "basepd", "originalpd", "pdinitial",
]

# —— DPD columns are definitional leakage (dpd≥90 ≡ default=1) ——————————————
_DPD_SIGNATURES = [
    "dpd", "dayspastdue",
]

# —— Post-default / outcome-only columns (recovery, LGD, EAD-at-default, CCF,
# default date) are unobservable for a loan that hasn't defaulted yet, so they
# must never be PD model features — same reasoning as origination-PD/DPD above.
# preprocessing_new.finalize_xy/prepare_data already strip these before X reaches
# this step; this is a second, independent guard in case apply_feature_engineering
# is ever called on a frame that skipped that path.
_POST_DEFAULT_SIGNATURES = [
    "recoveryrate", "recoveryamount", "recoveryamt", "recoveredamount", "recovery",
    "lgd", "lossgivendefault", "lossseverity", "realizedlgd",
    "eadatdefault", "exposureatdefault", "balanceatdefault", "defaultbalance",
    "ccf", "creditconversionfactor", "realizedccf",
    "defaultdate", "dateofdefault", "defaultdt", "chargeoffdate", "writeoffdate",
    "delinquencydate",
]


def find_origination_pd_cols(columns):
    """Columns that look like an origination PD (excluded from the PD model)."""
    out = []
    for c in columns:
        nc = _norm_name(c)
        if any(sig in nc for sig in _ORIG_PD_SIGNATURES):
            out.append(c)
    return out


def find_dpd_cols(columns):
    """Columns that look like Days-Past-Due (excluded from the PD model — DPD>=90
    is definitionally equivalent to default=1, so leaving it in is direct leakage)."""
    out = []
    for c in columns:
        nc = _norm_name(c)
        if any(sig in nc for sig in _DPD_SIGNATURES):
            out.append(c)
    return out


def find_post_default_cols(columns):
    """Columns realized only at/after default (recovery, LGD, EAD-at-default,
    CCF, default date) — excluded from the PD model."""
    out = []
    for c in columns:
        nc = _norm_name(c)
        if any(sig in nc for sig in _POST_DEFAULT_SIGNATURES):
            out.append(c)
    return out


def _sample_xy(X: pd.DataFrame, y: pd.Series, max_rows: int = MAX_DIAGNOSTIC_ROWS):
    if len(X) <= max_rows:
        return X, y
    sample_idx = X.sample(n=max_rows, random_state=42).index
    return X.loc[sample_idx], y.loc[sample_idx]


def _binary_target(y: pd.Series) -> pd.Series:
    vals = sorted(y.dropna().unique())
    if len(vals) != 2:
        raise ValueError("WOE/IV requires a binary target")
    return y.map({vals[0]: 0, vals[1]: 1}).astype(int)


# —— LEAKAGE FIX ————————————————————————————————————————————————————————————
# Bucketing is split into a FIT step (learns boundaries) and an APPLY step
# (uses learned boundaries). Boundaries / medians are learned on TRAIN ONLY and
# stored in the FE plan, then applied unchanged to validation and test. This
# replaces the old _bucket_feature(), which re-derived quantile edges and the
# fill median from whatever frame it was handed — leaking val/test statistics
# whenever it was used to transform val/test.

def _fit_bucketer(s: pd.Series, max_bins: int = 5) -> Dict[str, Any]:
    """
    Learn bucket boundaries on TRAIN data only. Returns a spec dict that is later
    applied (unchanged) to train, validation and test via _apply_bucketer().
    Buckets are emitted as integer codes so the WOE/bin map keys never depend on
    interval-string formatting that could differ between splits.

    MNAR fix: bin edges are fitted on NON-NULL values only.  NaN rows are not
    filled before binning and instead map to dedicated code -2 in _apply_bucketer,
    so the missing population gets its own WOE bin rather than blending into the
    median bin.  Code -1 is still reserved for out-of-range / unseen values.
    """
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        median = float(vals.median()) if vals.notna().any() else 0.0
        vals_nonnull = vals.dropna()          # fit on observed values only
        edges = None
        value_categories = None
        if vals_nonnull.nunique() > max_bins:
            try:
                _, raw_edges = pd.qcut(vals_nonnull, q=max_bins, duplicates="drop", retbins=True)
                edges = [float(e) for e in raw_edges]
                if len(edges) >= 2:
                    # Open the outer edges so val/test values outside the train
                    # range still fall into the first/last train bin (no NaNs).
                    edges[0] = -np.inf
                    edges[-1] = np.inf
                else:
                    edges = None
            except Exception:
                edges = None
        if edges is None:
            # Low-cardinality numeric: each distinct TRAIN value is its own bucket.
            value_categories = [str(v) for v in sorted(vals_nonnull.round(6).unique())]
        return {"kind": "numeric", "edges": edges, "median": median,
                "value_categories": value_categories}

    # Categorical: bucket = the category seen in TRAIN (missing -> __MISSING__).
    cats = s.astype("object").fillna("__MISSING__").astype(str)
    return {"kind": "categorical", "categories": list(pd.Index(cats.unique()))}


def _apply_bucketer(s: pd.Series, spec: Dict[str, Any]) -> pd.Series:
    """
    Apply a TRAIN-learned bucketer spec to ANY split.

    Code semantics:
      -2  dedicated missing bin (NaN in the original series)
      -1  out-of-range / unseen category (not seen in TRAIN)
      0…N normal train bins
    """
    if spec.get("kind") == "numeric":
        vals = pd.to_numeric(s, errors="coerce")
        was_missing = vals.isna()
        if spec.get("edges"):
            codes = pd.cut(vals, bins=spec["edges"], labels=False, include_lowest=True)
            result = pd.Series(codes, index=s.index).fillna(-1).astype(int)
        else:
            cat_index = {c: i for i, c in enumerate(spec.get("value_categories") or [])}
            result = vals.round(6).astype(str).map(cat_index).fillna(-1).astype(int)
        # Override: rows that were originally NaN get the dedicated missing bin code.
        result[was_missing] = -2
        return result

    cat_index = {c: i for i, c in enumerate(spec.get("categories") or [])}
    cats = s.astype("object").fillna("__MISSING__").astype(str)
    return cats.map(cat_index).fillna(-1).astype(int)


def _woe_map_for_series(s: pd.Series, y_bin: pd.Series) -> Tuple[Dict[Any, float], float, Dict[str, Any]]:
    """
    Learn a WOE map on TRAIN data only. Returns (woe_map, iv, bucketer_spec).
    The bucketer spec is returned so the SAME boundaries can be re-applied to
    validation/test at transform time without re-deriving anything.
    """
    spec = _fit_bucketer(s)
    codes = _apply_bucketer(s, spec)
    df = pd.DataFrame({"bucket": codes, "target": y_bin}).dropna()
    if df.empty or df["target"].nunique() != 2:
        return {}, 0.0, spec

    grouped = df.groupby("bucket", observed=False)["target"].agg(["sum", "count"])
    grouped["bad"] = grouped["sum"]
    grouped["good"] = grouped["count"] - grouped["sum"]
    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()
    n_buckets = max(len(grouped), 1)

    bad_dist = (grouped["bad"] + EPS) / (total_bad + EPS * n_buckets)
    good_dist = (grouped["good"] + EPS) / (total_good + EPS * n_buckets)
    woe = np.log(good_dist / bad_dist)
    iv = ((good_dist - bad_dist) * woe).sum()
    woe_map = woe.replace([np.inf, -np.inf], 0).fillna(0).to_dict()  # keys = int bucket codes
    return woe_map, float(max(iv, 0.0)), spec


def compute_information_value(
    X: pd.DataFrame,
    y: pd.Series,
    candidate_cols: List[str],
) -> Tuple[Dict[str, float], Dict[str, Dict[Any, float]], Dict[str, Dict[str, Any]]]:
    """
    Compute IV, WOE maps AND the bucketer specs, learned on the supplied frame
    only (the caller now passes X_train / y_train). The bucketer specs are
    returned so the exact TRAIN boundaries can be re-applied to val/test.
    """
    X_s, y_s = _sample_xy(X, y)
    y_bin = _binary_target(y_s)
    iv_scores, woe_maps, woe_specs = {}, {}, {}

    for col in candidate_cols[:MAX_IV_FEATURES]:
        if col not in X_s.columns:
            continue
        try:
            woe_map, iv, spec = _woe_map_for_series(X_s[col], y_bin)
            if woe_map:
                iv_scores[col] = round(iv, 5)
                woe_maps[col] = woe_map
                woe_specs[col] = spec
        except Exception:
            continue

    return iv_scores, woe_maps, woe_specs


def compute_multicollinearity_report(
    X: pd.DataFrame,
    numeric_cols: List[str],
    corr_threshold: float = 0.90,
    max_cols: int = 30,
) -> Dict[str, Any]:
    """Fast correlation/VIF-style check on a capped numeric subset."""
    numeric_cols = [c for c in numeric_cols if c in X.columns][:max_cols]
    if len(numeric_cols) < 2:
        return {"high_corr_pairs": [], "vif": {}}

    X_s, _ = _sample_xy(X[numeric_cols], pd.Series(np.zeros(len(X)), index=X.index))
    X_num = X_s.apply(pd.to_numeric, errors="coerce").fillna(X_s.median(numeric_only=True)).fillna(0)
    corr = X_num.corr().replace([np.inf, -np.inf], 0).fillna(0)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr_pairs = [
        {"feature_1": row, "feature_2": col, "correlation": round(float(upper.loc[row, col]), 4)}
        for col in upper.columns
        for row in upper.index
        if abs(float(upper.loc[row, col])) >= corr_threshold
    ]

    vif = {}
    try:
        corr_for_inv = corr.values + np.eye(len(corr)) * 1e-6
        inv_corr = np.linalg.pinv(corr_for_inv)
        vif = {
            col: round(float(max(inv_corr[i, i], 0)), 3)
            for i, col in enumerate(corr.columns)
        }
    except Exception:
        vif = {}

    return {"high_corr_pairs": high_corr_pairs, "vif": vif}


# ———————————————————————————————————————————————
# Interaction feature engineering (production-quality discovery engine)
# ———————————————————————————————————————————————
# Replaces the old "top-4-numeric-by-MI, multiply" approach. Every candidate
# feature is ranked by IV first (MI as a numeric-only fallback); categorical
# features are never label-multiplied — only their already-learned WOE
# transform is used. Every generated interaction is independently evaluated
# (never assumed useful) and redundancy-filtered before being kept. Everything
# here runs on TRAIN data only — see build_interaction_features()'s docstring.

def _woe_transform_column(X: pd.DataFrame, col: str, woe_map: Dict[Any, float],
                          woe_spec: Dict[str, Any]) -> pd.Series:
    """Apply an ALREADY-LEARNED (train-fit) WOE bucketer + map to a column.
    Never re-derives bucket edges or the WOE map itself — safe to call on any
    split, including validation/test, since spec/map are both train-learned."""
    codes = _apply_bucketer(X[col], woe_spec)
    return codes.map(woe_map).fillna(0.0).astype(float)


# —— Domain-driven banking interactions ——————————————————————————————————
# A small set of well-known credit-risk interactions, proposed as ADDITIONAL
# candidates regardless of whether their source features made the top-N
# ranking cut — domain knowledge earns a chance to be tested, not a free pass
# into the plan. Every one of these still goes through Phase 4-6 evaluation
# and redundancy filtering exactly like a ranked candidate, and is dropped if
# it isn't actually predictive on this dataset.
_DOMAIN_INTERACTION_SIGNATURES = [
    (("ltv", "loantovalue"), ("creditscore", "ficoscore", "bureauscore", "cibil", "riskscore")),
    (("income", "annualincome", "grossincome", "monthlyincome"), ("dti", "debttoincome", "debttoincomeratio")),
    (("age",), ("employmentlength", "employmentyears", "yearsemployed", "tenure")),
    (("loanamount", "principal", "creditlimit", "sanctioned"), ("income", "annualincome", "grossincome")),
    (("creditscore", "ficoscore", "bureauscore"), ("numlatepayments", "delinquencycount",
                                                     "numcreditinquiries", "numinquiries")),
]


def _find_domain_interaction_pairs(columns: List[str]) -> List[Tuple[str, str]]:
    """Name-signature matches for well-known banking interactions (LTV x credit
    score, income x DTI, age x employment length, ...)."""
    norm_map = {_norm_name(c): c for c in columns}
    pairs = []
    for sigs_a, sigs_b in _DOMAIN_INTERACTION_SIGNATURES:
        col_a = next((norm_map[n] for n in norm_map if any(s in n for s in sigs_a)), None)
        col_b = next((norm_map[n] for n in norm_map if any(s in n for s in sigs_b)), None)
        if col_a and col_b and col_a != col_b:
            pairs.append((col_a, col_b))
    return pairs


def _gen_domain_interactions(X, top_numeric, top_categorical, woe_maps, woe_specs, max_cardinality):
    """Phase 2, additional class: known banking interactions, checked against
    ALL columns in X (not just the top-N ranked ones — domain knowledge gets
    a chance to be evaluated even if its features didn't rank highly on their
    own). Numeric x Numeric multiplies raw values; anything touching a
    categorical only proceeds if that column already has a learned WOE map —
    same "always prefer WOE" rule as the ranked generators."""
    out = []
    numeric_set = set(pd.Index(X.columns).intersection(
        [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    ))
    for col_a, col_b in _find_domain_interaction_pairs(list(X.columns)):
        if col_a in numeric_set and col_b in numeric_set:
            va = pd.to_numeric(X[col_a], errors="coerce").fillna(0)
            vb = pd.to_numeric(X[col_b], errors="coerce").fillna(0)
            out.append({
                "name": f"{col_a}_x_{col_b}", "feature_a": col_a, "feature_b": col_b,
                "interaction_type": "numeric_numeric", "values": va * vb, "source": "domain",
            })
        elif col_a in numeric_set and col_b in woe_maps and col_b in woe_specs:
            va = pd.to_numeric(X[col_a], errors="coerce").fillna(0)
            vb = _woe_transform_column(X, col_b, woe_maps[col_b], woe_specs[col_b])
            out.append({
                "name": f"{col_a}_x_{col_b}_WOE", "feature_a": col_a, "feature_b": col_b,
                "interaction_type": "numeric_woe", "values": va * vb, "source": "domain",
            })
        elif col_b in numeric_set and col_a in woe_maps and col_a in woe_specs:
            va = pd.to_numeric(X[col_b], errors="coerce").fillna(0)
            vb = _woe_transform_column(X, col_a, woe_maps[col_a], woe_specs[col_a])
            out.append({
                "name": f"{col_b}_x_{col_a}_WOE", "feature_a": col_b, "feature_b": col_a,
                "interaction_type": "numeric_woe", "values": va * vb, "source": "domain",
            })
        # A categorical x categorical domain pair with no WOE on either side is
        # simply skipped — never falls back to raw label codes.
    return out


def _gen_numeric_numeric(X, top_numeric, top_categorical, woe_maps, woe_specs, max_cardinality):
    """Phase 2, class 1: A * B for every pair of top-ranked numeric features."""
    out = []
    for a, b in itertools.combinations(top_numeric, 2):
        va = pd.to_numeric(X[a], errors="coerce").fillna(0)
        vb = pd.to_numeric(X[b], errors="coerce").fillna(0)
        out.append({
            "name": f"{a}_x_{b}", "feature_a": a, "feature_b": b,
            "interaction_type": "numeric_numeric", "values": va * vb, "source": "ranked",
        })
    return out


def _gen_numeric_woe(X, top_numeric, top_categorical, woe_maps, woe_specs, max_cardinality):
    """Phase 2, class 2: Numeric x WOE(Category). Only categoricals that already
    have a learned WOE map are used — never raw label-encoded integers."""
    out = []
    for num_col in top_numeric:
        for cat_col in top_categorical:
            if cat_col not in woe_maps or cat_col not in woe_specs:
                continue
            va = pd.to_numeric(X[num_col], errors="coerce").fillna(0)
            vb = _woe_transform_column(X, cat_col, woe_maps[cat_col], woe_specs[cat_col])
            out.append({
                "name": f"{num_col}_x_{cat_col}_WOE", "feature_a": num_col, "feature_b": cat_col,
                "interaction_type": "numeric_woe", "values": va * vb, "source": "ranked",
            })
    return out


def _gen_woe_woe(X, top_numeric, top_categorical, woe_maps, woe_specs, max_cardinality):
    """Phase 2, class 3: WOE(Cat1) x WOE(Cat2) — only for pairs where BOTH
    columns are low-cardinality AND already have a learned WOE map."""
    out = []
    low_card = [
        c for c in top_categorical
        if c in woe_maps and c in woe_specs and X[c].nunique(dropna=True) <= max_cardinality
    ]
    for a, b in itertools.combinations(low_card, 2):
        va = _woe_transform_column(X, a, woe_maps[a], woe_specs[a])
        vb = _woe_transform_column(X, b, woe_maps[b], woe_specs[b])
        out.append({
            "name": f"{a}_WOE_x_{b}_WOE", "feature_a": a, "feature_b": b,
            "interaction_type": "woe_woe", "values": va * vb, "source": "ranked",
        })
    return out


# Pluggable generator registry (Design Requirement 6 — Future Extensibility).
# A future generator (SHAP interaction discovery, XGBoost interaction
# constraints, PDP-based search, ...) is added here, matching the same
# signature (X, top_numeric, top_categorical, woe_maps, woe_specs,
# max_cardinality) -> list[dict with a "values" pd.Series] — the orchestration
# in build_interaction_features() below needs no changes to pick it up.
INTERACTION_GENERATORS = [_gen_numeric_numeric, _gen_numeric_woe, _gen_woe_woe, _gen_domain_interactions]


def _evaluate_interaction(values: pd.Series, y_bin: pd.Series) -> Dict[str, Any]:
    """Phase 4: score ONE interaction independently — never assume it's useful.
    Reuses the same WOE/IV machinery as any other feature (_woe_map_for_series)
    for IV, plus an AUC-based Gini as a second, preferred-when-available metric."""
    iv = 0.0
    try:
        _, iv, _ = _woe_map_for_series(values, y_bin)
    except Exception:
        iv = 0.0

    gini = None
    try:
        from sklearn.metrics import roc_auc_score
        filled = values.fillna(values.median() if values.notna().any() else 0.0)
        auc = float(roc_auc_score(y_bin, filled))
        if auc < 0.5:
            auc = 1.0 - auc
        gini = round(2.0 * auc - 1.0, 4)
    except Exception:
        gini = None

    return {"IV": round(float(iv), 5), "Gini": gini}


def _redundancy_filter(
    X: pd.DataFrame,
    ranked_candidates: List[Dict[str, Any]],
    existing_numeric_cols: List[str],
    corr_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Phase 6: drop an interaction if it's highly correlated (|corr| > threshold)
    with EITHER an existing numeric feature OR another, already-kept (i.e.
    higher-ranked) interaction. `ranked_candidates` must already be sorted
    best-first — greedy, so ties are broken in favor of the better-scored one.
    """
    existing_cols = [c for c in existing_numeric_cols if c in X.columns]
    existing = X[existing_cols].apply(pd.to_numeric, errors="coerce") if existing_cols else None

    kept: List[Dict[str, Any]] = []
    kept_series: List[pd.Series] = []

    for cand in ranked_candidates:
        v = cand["values"]
        redundant = False

        if existing is not None:
            for col in existing.columns:
                try:
                    corr = v.corr(existing[col])
                    if pd.notna(corr) and abs(corr) > corr_threshold:
                        redundant = True
                        break
                except Exception:
                    continue

        if not redundant:
            for kv in kept_series:
                try:
                    corr = v.corr(kv)
                    if pd.notna(corr) and abs(corr) > corr_threshold:
                        redundant = True
                        break
                except Exception:
                    continue

        if not redundant:
            kept.append(cand)
            kept_series.append(v)

    return kept


_INTERACTION_TYPE_LABELS = {
    "numeric_numeric": "Numeric × Numeric",
    "numeric_woe":      "Numeric × WOE",
    "woe_woe":           "WOE × WOE",
}


def build_interaction_features(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: List[str],
    cat_cols: List[str],
    iv_scores: Dict[str, float],
    mi_scores: Dict[str, float],
    woe_maps: Dict[str, Dict[Any, float]],
    woe_specs: Dict[str, Dict[str, Any]],
    top_n: int = TOP_FEATURES_FOR_INTERACTIONS,
    max_interactions: int = MAX_INTERACTIONS,
    min_iv: float = MIN_INTERACTION_IV,
    max_cardinality: int = MAX_CATEGORY_CARDINALITY,
    corr_threshold: float = CORRELATION_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Full interaction-discovery pipeline, Phases 1-6 — run on X_train/y_train
    ONLY. Callers must never invoke this on validation/test: every threshold
    decision (which features rank highest, which interactions clear the IV
    bar, which are redundant) is a supervised, train-only decision baked into
    the returned `interaction_features` list. apply_feature_engineering()
    replays that exact list unchanged on every split (Phase 8) — it never
    calls this function itself.

    Returns:
      interaction_features: plan-ready list of dicts (Phase 7 shape), e.g.
        {"name", "feature_a", "feature_b", "type", "interaction_type",
         "score", "metric", "gini", "selection_metric"}
      interaction_scores: {name: {"IV": ..., "Gini": ...}} for every candidate
        that was evaluated (Phase 4), including ones later filtered out —
        useful for showing the reviewer what was tried, not just what was kept.
      feature_scores: {feature: {"score": ..., "metric": "IV"|"MI"}} (Phase 1)
    """
    # Phase 1 — rank ALL candidate features. IV first; MI is a numeric-only
    # fallback for columns IV couldn't be computed for (categoricals have no
    # MI fallback — an un-scoreable categorical is simply not a candidate).
    feature_scores: Dict[str, Dict[str, Any]] = {}
    for col in numeric_cols:
        if col in iv_scores:
            feature_scores[col] = {"score": iv_scores[col], "metric": "IV"}
        elif col in mi_scores:
            feature_scores[col] = {"score": mi_scores[col], "metric": "MI"}
    for col in cat_cols:
        if col in iv_scores:
            feature_scores[col] = {"score": iv_scores[col], "metric": "IV"}

    if not feature_scores:
        return [], {}, feature_scores

    ranked = sorted(feature_scores.items(), key=lambda kv: kv[1]["score"], reverse=True)
    top_features = [c for c, _ in ranked[:top_n]]
    top_numeric = [c for c in top_features if c in numeric_cols]
    top_categorical = [c for c in top_features if c in cat_cols]

    # Phase 2 — generate candidates via the pluggable generator registry.
    # Ranking first (above), generating only from the selected top features
    # (never every feature) is what keeps this from being combinatorial.
    candidates: List[Dict[str, Any]] = []
    for generator in INTERACTION_GENERATORS:
        try:
            candidates.extend(generator(X, top_numeric, top_categorical, woe_maps, woe_specs, max_cardinality))
        except Exception:
            continue

    if not candidates:
        return [], {}, feature_scores

    # Phase 4 — evaluate every interaction independently.
    y_bin = _binary_target(y)
    interaction_scores: Dict[str, Dict[str, Any]] = {}
    for cand in candidates:
        scores = _evaluate_interaction(cand["values"], y_bin)
        cand["IV"] = scores["IV"]
        cand["Gini"] = scores["Gini"]
        interaction_scores[cand["name"]] = scores

    # Phase 5 — keep only useful interactions: IV threshold gate, then cap at
    # max_interactions ranked by Gini (preferred when available) with IV as
    # the tiebreaker/fallback ranking metric.
    survivors = [c for c in candidates if c["IV"] >= min_iv]
    survivors.sort(key=lambda c: (c["Gini"] if c["Gini"] is not None else c["IV"]), reverse=True)
    survivors = survivors[:max_interactions]

    # Phase 6 — redundancy filtering vs existing features AND vs each other.
    survivors = _redundancy_filter(X, survivors, numeric_cols, corr_threshold)

    # Phase 7 — plan-ready metadata. Raw values are dropped now that scoring
    # and filtering are done; only lightweight, explainable metadata survives
    # into the plan (apply_feature_engineering regenerates the actual values).
    interaction_features = [
        {
            "name": c["name"],
            "feature_a": c["feature_a"],
            "feature_b": c["feature_b"],
            "type": c["interaction_type"],
            "interaction_type": _INTERACTION_TYPE_LABELS.get(c["interaction_type"], c["interaction_type"]),
            "score": c["IV"],
            "metric": "IV",
            "gini": c["Gini"],
            "selection_metric": "IV",
            "source": c.get("source", "ranked"),
        }
        for c in survivors
    ]

    return interaction_features, interaction_scores, feature_scores



def analyze_for_feature_engineering(
    X: pd.DataFrame,
    y: pd.Series,
    col_types: Dict[str, List[str]],
    task_type: str = "binary",
    model_family: str = "linear",
    transform_choices: Dict[str, str] = None,
) -> Dict[str, Any]:
    """
    Analyze dataset and decide which feature engineering steps to apply.
    Returns a plan dict explaining what will be done and why.

    Log-transform columns work the same "propose, don't apply" way as
    preprocessing_new.py's pipeline routing: every numeric column's
    recommend_transform() output is stored in plan["transform_recommendations"],
    but plan["log_transform_cols"] — what apply_feature_engineering() actually
    creates a `{col}_log` column for — is driven ONLY by `transform_choices`
    (a {column: "none"|"log1p"|"yeo_johnson"} dict of the reviewer's CONFIRMED
    selections). Pass None / omit it and no `_log` columns are created at all.

    model_family: "tree" or "linear" — passed through to recommend_transform()
    (see preprocessing_new.py) so recommendations are annotated appropriately for
    the model family actually being trained.
    """
    transform_choices = transform_choices or {}
    plan = {
        "log_transform_cols": [],
        "transform_recommendations": {},  # {col: recommend_transform() output} — every numeric column, for the UI
        "interaction_pairs": [],       # back-compat: (feature_a, feature_b) tuples, derived below
        "interaction_features": [],    # canonical: plan-ready metadata, see build_interaction_features()
        "interaction_scores": {},      # {name: {"IV":, "Gini":}} for every candidate evaluated, not just kept
        "feature_scores": {},          # {feature: {"score":, "metric": "IV"|"MI"}} — Phase 1 ranking
        "binning_cols": [],
        "freq_encoding_cols": [],
        "datetime_cols": col_types.get("datetime", []),
        "drop_high_corr_pairs": [],
        "low_variance_cols": [],
        "mi_scores": {},
        "iv_scores": {},
        "woe_maps": {},
        "woe_cols": [],
        "low_iv_cols": [],
        "multicollinearity": {"high_corr_pairs": [], "vif": {}},
        "excluded_orig_pd": [],
        "applied_steps": [],
        # —— LEAKAGE FIX: learned application state (TRAIN-only), applied verbatim
        #    to val/test by apply_feature_engineering(). ——
        "freq_maps": {},   # {col: {raw_value: frequency}}      learned on X_train
        "bin_specs": {},   # {col: bucketer_spec}               learned on X_train
        "woe_specs": {},   # {col: bucketer_spec for WOE}       learned on X_train
        "learned_on": "train",
    }

    numeric_cols = [c for c in col_types.get("numeric", []) if c in X.columns]
    cat_cols = [c for c in col_types.get("categorical", []) if c in X.columns]

    # Change 3: hide origination PD and DPD from model development (leakage / definitional)
    _orig_pd_cols = find_origination_pd_cols(X.columns)
    _dpd_cols     = find_dpd_cols(X.columns)
    _hidden_cols  = list(dict.fromkeys(_orig_pd_cols + _dpd_cols))
    if _hidden_cols:
        numeric_cols = [c for c in numeric_cols if c not in _hidden_cols]
        cat_cols     = [c for c in cat_cols     if c not in _hidden_cols]
        plan["excluded_orig_pd"] = _hidden_cols

    for col in numeric_cols:
        plan["transform_recommendations"][col] = recommend_transform(X[col], model_family=model_family)

    # log_transform_cols is driven ONLY by transform_choices (confirmed by the
    # reviewer in the UI) — never by the recommendation directly. An empty/
    # missing transform_choices means no `_log` column is created for anything,
    # regardless of how skewed a column is.
    plan["log_transform_cols"] = [
        c for c in numeric_cols if transform_choices.get(c) == "log1p"
    ]
    if plan["log_transform_cols"]:
        plan["applied_steps"].append({
            "step": "Log Transform",
            "columns": plan["log_transform_cols"],
            "reason": "Reviewer-confirmed log1p transform (see plan['transform_recommendations'] "
                      "for the skew evidence behind each recommendation).",
        })

    # —— Mutual Information (numeric-only fallback ranking metric — see Phase 1
    #    in build_interaction_features(): used only for a column IV couldn't be
    #    computed for. No longer drives interaction generation directly. ——
    if len(numeric_cols) >= 2 and task_type == "binary":
        try:
            X_num = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
            X_num = X_num.fillna(X_num.median()).fillna(0)
            from sklearn.feature_selection import mutual_info_classif
            mi = mutual_info_classif(X_num, y.fillna(y.mode()[0]), random_state=42, discrete_features=False)
            mi_series = pd.Series(mi, index=numeric_cols).sort_values(ascending=False)
            plan["mi_scores"] = mi_series.round(5).to_dict()
        except Exception:
            pass

    # —— Information Value / WOE — MOVED UP from the end of this function so the
    #    interaction engine below can rank categorical features by IV and reuse
    #    already-learned WOE maps, instead of re-deriving anything. ——
    if task_type == "binary":
        try:
            iv_candidates = numeric_cols + cat_cols
            iv_scores, woe_maps, woe_specs = compute_information_value(X, y, iv_candidates)
            plan["iv_scores"] = dict(sorted(iv_scores.items(), key=lambda item: item[1], reverse=True))
            plan["woe_maps"] = woe_maps
            plan["woe_specs"] = woe_specs   # LEAKAGE FIX: store TRAIN bucketer specs for WOE
            plan["low_iv_cols"] = [
                c for c, iv in plan["iv_scores"].items() if iv < 0.02
            ]
            plan["woe_cols"] = [
                c for c in list(plan["iv_scores"].keys())[:MAX_WOE_FEATURES]
                if c not in plan["low_iv_cols"]
            ]
            if plan["woe_cols"]:
                plan["applied_steps"].append({
                    "step": "WOE Transformation",
                    "columns": plan["woe_cols"],
                    "reason": "Top IV features receive lightweight Weight of Evidence encoded copies.",
                })
            if plan["low_iv_cols"]:
                plan["applied_steps"].append({
                    "step": "Information Value Selection",
                    "columns": plan["low_iv_cols"][:20],
                    "reason": "Very low-IV features are removed using a simple IV < 0.02 rule.",
                })
        except Exception:
            pass

    # —— Interaction features — production-quality discovery engine ——————————
    # Replaces the old "top-4-numeric-by-MI, multiply" approach. See
    # build_interaction_features() for the full Phase 1/2/4/5/6/7 workflow:
    # rank by IV (MI fallback) -> generate numeric×numeric / numeric×WOE /
    # WOE×WOE candidates from the top-ranked features only -> evaluate every
    # one independently (IV, Gini) -> keep only IV >= MIN_INTERACTION_IV,
    # capped at MAX_INTERACTIONS -> drop anything redundant with an existing
    # feature or another interaction. Everything here is TRAIN-only; Phase 8
    # (apply_feature_engineering) replays this list unchanged on every split.
    if task_type == "binary" and (numeric_cols or cat_cols):
        try:
            interaction_features, interaction_scores, feature_scores = build_interaction_features(
                X, y, numeric_cols, cat_cols,
                plan["iv_scores"], plan["mi_scores"], plan["woe_maps"], plan["woe_specs"],
            )
            plan["interaction_features"] = interaction_features
            plan["interaction_scores"] = interaction_scores
            plan["feature_scores"] = feature_scores
            # Back-compat: a simple (feature_a, feature_b) tuple list, derived
            # from interaction_features — display-only, NOT used by
            # apply_feature_engineering (which reads interaction_features).
            plan["interaction_pairs"] = [
                (f["feature_a"], f["feature_b"]) for f in interaction_features
            ]
            if interaction_features:
                plan["applied_steps"].append({
                    "step": "Interaction Features",
                    "columns": [f["name"] for f in interaction_features],
                    "reason": (
                        f"Generated Numeric×Numeric, Numeric×WOE, and WOE×WOE candidates from the "
                        f"top {TOP_FEATURES_FOR_INTERACTIONS} features by IV (MI fallback), plus known "
                        f"banking interactions (LTV×credit score, income×DTI, etc.) where present — "
                        f"evaluated each independently (IV, Gini), kept only IV ≥ {MIN_INTERACTION_IV} "
                        f"capped at {MAX_INTERACTIONS}, and removed anything redundant "
                        f"(|corr| > {CORRELATION_THRESHOLD}) with an existing feature or another interaction."
                    ),
                })
        except Exception:
            pass

    for col in numeric_cols:
        if X[col].nunique() > 20:
            plan["binning_cols"].append(col)
    if plan["binning_cols"]:
        # LEAKAGE FIX: learn the quantile edges + fill median on TRAIN now and
        # store them, so val/test get binned with the SAME boundaries later.
        for col in plan["binning_cols"][:5]:
            plan["bin_specs"][col] = _fit_bucketer(X[col], max_bins=5)
        plan["applied_steps"].append({
            "step": "Quantile Binning",
            "columns": plan["binning_cols"][:5],
            "reason": "High-cardinality numeric columns binned into 5 quantile buckets (edges learned on training data only).",
        })

    for col in cat_cols:
        if X[col].nunique() > 8:
            plan["freq_encoding_cols"].append(col)
    if plan["freq_encoding_cols"]:
        # LEAKAGE FIX: learn the category->frequency map on TRAIN now and store
        # it; unseen val/test categories will map to 0 at apply time.
        for col in plan["freq_encoding_cols"]:
            plan["freq_maps"][col] = X[col].value_counts(normalize=True).to_dict()
        plan["applied_steps"].append({
            "step": "Frequency Encoding",
            "columns": plan["freq_encoding_cols"],
            "reason": "High-cardinality categoricals encoded by frequency (frequencies learned on training data only).",
        })

    plan["multicollinearity"] = compute_multicollinearity_report(X, numeric_cols)
    high_corr_pairs = [
        (p["feature_2"], p["feature_1"], p["correlation"])
        for p in plan["multicollinearity"].get("high_corr_pairs", [])
    ]
    plan["drop_high_corr_pairs"] = high_corr_pairs
    if high_corr_pairs:
        plan["applied_steps"].append({
            "step": "Multicollinearity Check",
            "columns": list({p[1] for p in high_corr_pairs}),
            "reason": f"{len(high_corr_pairs)} highly correlated pair(s) found; one feature removed from each pair.",
        })

    try:
        # Scale-free near-constant check: a column is flagged only if
        # (a) it has ≤ 1 unique non-null value — completely constant, or
        # (b) its mode covers > 99 % of non-null rows — effectively constant.
        # VarianceThreshold(0.01) on raw data was scale-dependent and wrongly
        # flagged 0-to-1 ratio variables (e.g. ltv) as near-zero variance.
        X_num = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
        _near_const = []
        for col in numeric_cols:
            series = X_num[col].dropna()
            if series.empty or series.nunique() <= 1:
                _near_const.append(col)
            elif series.value_counts(normalize=True).iloc[0] > 0.99:
                _near_const.append(col)
        plan["low_variance_cols"] = _near_const
        if plan["low_variance_cols"]:
            plan["applied_steps"].append({
                "step": "Near-Constant Removal",
                "columns": plan["low_variance_cols"],
                "reason": "Column is near-constant (top value covers > 99% of rows, or ≤ 1 unique value).",
            })
    except Exception:
        pass

    return plan


def apply_feature_engineering(
    X: pd.DataFrame,
    plan_or_y: Any,
    plan: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply a feature engineering plan to a feature matrix.

    This accepts both the current call form ``apply_feature_engineering(X, plan)``
    and the legacy reference-style call ``apply_feature_engineering(X, y, plan)``.
    The extra target argument is ignored because supervised statistics must never
    be (re)learned at apply time.

    Returns (X_engineered, summary).
    """
    plan = plan if plan is not None else plan_or_y
    X = X.copy()
    summary = {"added": [], "removed": [], "transformed": []}
    original_shape = X.shape

    # Change 3: drop origination PD and DPD — they must never be model features
    _orig_pd_cols = find_origination_pd_cols(X.columns)
    _dpd_cols     = find_dpd_cols(X.columns)
    _post_default_cols = find_post_default_cols(X.columns)
    _hidden_cols  = list(dict.fromkeys(_orig_pd_cols + _dpd_cols + _post_default_cols))
    if _hidden_cols:
        X = X.drop(columns=_hidden_cols, errors="ignore")
    summary["excluded_orig_pd"] = _hidden_cols
    summary["excluded_post_default"] = _post_default_cols

    # —— Log transform (stateless) ——
    for col in plan.get("log_transform_cols", []):
        if col in X.columns:
            new_col = f"{col}_log"
            X[new_col] = np.log1p(pd.to_numeric(X[col], errors="coerce").clip(lower=0).fillna(0))
            summary["added"].append(new_col)
            summary["transformed"].append(f"log1p({col}) -> {new_col}")

    # —— Interaction features (Phase 8 — apply ONLY what was approved in the
    #    plan; never regenerate or re-score candidates here). Each entry's
    #    "type" says which of the three classes it is, so the numeric×WOE and
    #    WOE×WOE cases can reuse the exact TRAIN-learned bucketer + WOE map
    #    from plan["woe_maps"]/plan["woe_specs"] — the same one WOE encoding
    #    below uses — rather than deriving anything new from this split. ——
    for feat in plan.get("interaction_features", []):
        a, b, itype, name = feat.get("feature_a"), feat.get("feature_b"), feat.get("type"), feat.get("name")
        if not a or not b or not name or a not in X.columns or b not in X.columns:
            continue
        try:
            if itype == "numeric_numeric":
                va = pd.to_numeric(X[a], errors="coerce").fillna(0)
                vb = pd.to_numeric(X[b], errors="coerce").fillna(0)
            elif itype == "numeric_woe":
                # a = numeric, b = categorical (WOE-transformed)
                wmap, spec = plan.get("woe_maps", {}).get(b), plan.get("woe_specs", {}).get(b)
                if wmap is None or spec is None:
                    continue
                va = pd.to_numeric(X[a], errors="coerce").fillna(0)
                vb = _apply_bucketer(X[b], spec).map(wmap).fillna(0.0).astype(float)
            elif itype == "woe_woe":
                wmap_a, spec_a = plan.get("woe_maps", {}).get(a), plan.get("woe_specs", {}).get(a)
                wmap_b, spec_b = plan.get("woe_maps", {}).get(b), plan.get("woe_specs", {}).get(b)
                if None in (wmap_a, spec_a, wmap_b, spec_b):
                    continue
                va = _apply_bucketer(X[a], spec_a).map(wmap_a).fillna(0.0).astype(float)
                vb = _apply_bucketer(X[b], spec_b).map(wmap_b).fillna(0.0).astype(float)
            else:
                continue
            X[name] = va * vb
            summary["added"].append(name)
            summary["transformed"].append(f"{feat.get('interaction_type', itype)}: {a} x {b} -> {name}")
        except Exception:
            continue

    # —— Quantile binning — apply TRAIN-learned edges (no qcut here) ——
    for col in plan.get("binning_cols", [])[:5]:
        spec = plan.get("bin_specs", {}).get(col)
        if col in X.columns and spec is not None:
            new_col = f"{col}_bin"
            X[new_col] = _apply_bucketer(X[col], spec).values
            summary["added"].append(new_col)
            summary["transformed"].append(f"bin({col}) -> {new_col} [train edges]")

    # —— Frequency encoding — apply TRAIN-learned frequency map (no value_counts) ——
    for col in plan.get("freq_encoding_cols", []):
        freq_map = plan.get("freq_maps", {}).get(col)
        if col in X.columns and freq_map is not None:
            new_col = f"{col}_freq"
            X[new_col] = X[col].map(freq_map).fillna(0.0)  # unseen categories -> 0
            summary["added"].append(new_col)
            summary["transformed"].append(f"freq_encode({col}) -> {new_col} [train freqs]")

    # —— WOE encoding — apply TRAIN-learned buckets + WOE map (no re-bucketing) ——
    for col in plan.get("woe_cols", []):
        wmap = plan.get("woe_maps", {}).get(col)
        spec = plan.get("woe_specs", {}).get(col)
        if col in X.columns and wmap and spec is not None:
            new_col = f"{col}_woe"
            codes = _apply_bucketer(X[col], spec)
            X[new_col] = codes.map(wmap).fillna(0.0).astype(float).values  # unseen bucket -> 0
            summary["added"].append(new_col)
            summary["transformed"].append(f"woe_encode({col}) -> {new_col} [train WOE]")

    # —— WoE-pending raw column removal ——
    # Columns the reviewer chose "Keep via WoE encoding" in Step 3 have now been
    # encoded above as <col>_woe.  Drop the raw column so the model never sees the
    # NaN-carrying original — making these columns safe for all model families.
    for col in plan.get("woe_pending_drop", []):
        if col in X.columns:
            X = X.drop(columns=[col])
            summary["removed"].append(col)

    # —— Feature removal ——
    # Use the reviewer-confirmed list if the UI has run (propose-confirm flow),
    # otherwise fall back to the full proposal list so inference / non-UI callers
    # still work without changes.
    if "confirmed_remove_cols" in plan:
        cols_to_drop = [c for c in plan["confirmed_remove_cols"] if c in X.columns]
    else:
        cols_to_drop_corr = list({p[1] for p in plan.get("drop_high_corr_pairs", [])})
        low_var_cols = plan.get("low_variance_cols", [])
        low_iv_cols = plan.get("low_iv_cols", [])
        cols_to_drop = [c for c in dict.fromkeys(cols_to_drop_corr + low_var_cols + low_iv_cols) if c in X.columns]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
        summary["removed"].extend(cols_to_drop)

    summary["original_shape"] = original_shape
    summary["final_shape"] = X.shape
    summary["features_added"] = len(summary["added"])
    summary["features_removed"] = len(summary["removed"])

    return X, summary


# ———————————————————————————————————————————————
# EAD configuration resolver
# ———————————————————————————————————————————————
# Ported unchanged from the previous backend feature_engineering.py so
# main.py's `from feature_engineering import resolve_ead_configuration`
# keeps working — this module no longer defines the EAD detection/compute
# primitives itself (those come from ecl_engine, imported above), but the
# resolution/summary logic that decides WHICH source to use still lives here.

def _format_ead_summary(series: Optional[pd.Series]) -> Dict[str, Any]:
    if series is None:
        return {}
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return {
        "mean": round(float(values.mean()), 2) if values.notna().any() else None,
        "median": round(float(values.median()), 2) if values.notna().any() else None,
        "min": round(float(values.min()), 2) if values.notna().any() else None,
        "max": round(float(values.max()), 2) if values.notna().any() else None,
    }


def resolve_ead_configuration(
    df: pd.DataFrame,
    mode: str = "auto",
    ob_col: Optional[str] = None,
    la_col: Optional[str] = None,
    ir_col: Optional[str] = None,
    ye_col: Optional[str] = None,
    tm_col: Optional[str] = None,
    ye_months: bool = False,
    tm_months: bool = False,
) -> Dict[str, Any]:
    """Resolve the same EAD source configuration the Streamlit app exposes."""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    resolved_mode = mode if mode in {"outstanding_balance", "estimate"} else "outstanding_balance"

    if resolved_mode == "estimate":
        la_col = la_col or detect_loan_amount_col(df)
        ir_col = ir_col or detect_interest_rate_col(df)
        ye_col, ye_is_months = ye_col or detect_years_elapsed_col(df), ye_months or detect_years_elapsed_col(df)[1]
        tm_col = tm_col or detect_term_col(df)
        missing = [label for label, value in [
            ("loan amount", la_col),
            ("interest rate", ir_col),
            ("elapsed time", ye_col),
            ("total loan term", tm_col),
        ] if not value]
        if missing:
            return {
                "mode": "estimate",
                "source_col": "outstanding_balance (estimated)",
                "method": "Estimated amortizing outstanding balance",
                "series": None,
                "available": False,
                "missing_columns": missing,
                "selected": {
                    "loan_amount": la_col,
                    "interest_rate": ir_col,
                    "years_elapsed": ye_col,
                    "years_elapsed_is_months": bool(ye_is_months),
                    "term": tm_col,
                    "term_is_months": bool(tm_months),
                },
                "summary": {},
            }
        la_s = pd.to_numeric(df[la_col], errors="coerce")
        ir_s = pd.to_numeric(df[ir_col], errors="coerce")
        ye_s = pd.to_numeric(df[ye_col], errors="coerce")
        if ye_months or ye_is_months:
            ye_s = ye_s / 12.0
        tm_s = pd.to_numeric(df[tm_col], errors="coerce")
        if tm_months:
            tm_s = tm_s / 12.0
        series = compute_outstanding_balance(la_s, ir_s, ye_s, term_years=tm_s)
        return {
            "mode": "estimate",
            "source_col": "outstanding_balance (estimated)",
            "method": f"Estimated amortizing outstanding balance from '{la_col}', '{ir_col}', '{ye_col}', term '{tm_col}'",
            "series": series.astype(float),
            "available": True,
            "missing_columns": [],
            "selected": {
                "loan_amount": la_col,
                "interest_rate": ir_col,
                "years_elapsed": ye_col,
                "years_elapsed_is_months": bool(ye_months or ye_is_months),
                "term": tm_col,
                "term_is_months": bool(tm_months),
            },
            "summary": _format_ead_summary(series),
        }

    ob_col = ob_col or detect_outstanding_balance_col(df)
    if ob_col and ob_col in num_cols:
        series = pd.to_numeric(df[ob_col], errors="coerce").clip(lower=0)
        return {
            "mode": "outstanding_balance",
            "source_col": ob_col,
            "method": f"Outstanding balance column '{ob_col}'",
            "series": series.astype(float),
            "available": True,
            "selected": {"outstanding_balance_col": ob_col},
            "summary": _format_ead_summary(series),
        }

    la_col = la_col or detect_loan_amount_col(df)
    ir_col = ir_col or detect_interest_rate_col(df)
    ye_col, ye_is_months = detect_years_elapsed_col(df)
    tm_col = tm_col or detect_term_col(df)
    return {
        "mode": "estimate",
        "source_col": "outstanding_balance (estimated)",
        "method": "Estimated amortizing outstanding balance",
        "series": None,
        "available": bool(la_col and ir_col and ye_col and tm_col),
        "selected": {
            "loan_amount": la_col,
            "interest_rate": ir_col,
            "years_elapsed": ye_col,
            "years_elapsed_is_months": bool(ye_is_months),
            "term": tm_col,
            "term_is_months": False,
        },
        "summary": {},
    }


def compute_univariate_gini(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: List[str],
) -> Dict[str, float]:
    """
    Compute per-feature Gini coefficient (2*AUC - 1) against the binary target.
    Handles AUC < 0.5 by flipping (uses absolute discriminative power).
    Returns {col: gini} sorted descending, capped at MAX_IV_FEATURES columns.
    """
    try:
        y_bin = _binary_target(y)
        result: Dict[str, float] = {}
        for col in numeric_cols[:MAX_IV_FEATURES]:
            if col not in X.columns:
                continue
            try:
                vals = pd.to_numeric(X[col], errors="coerce")
                vals = vals.fillna(vals.median()).fillna(0)
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(y_bin, vals))
                if auc < 0.5:
                    auc = 1.0 - auc
                result[col] = round(2.0 * auc - 1.0, 4)
            except Exception:
                continue
        return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))
    except Exception:
        return {}


def get_feature_importance_summary(importance_dict: Dict[str, float], top_n: int = 15) -> pd.DataFrame:
    """Return top-N feature importances as a sorted DataFrame."""
    df = pd.DataFrame(list(importance_dict.items()), columns=["Feature", "Importance"])
    df = df.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)
    df["Importance"] = df["Importance"].round(4)
    return df
