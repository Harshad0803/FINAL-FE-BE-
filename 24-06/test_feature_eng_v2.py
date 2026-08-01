"""
Simple test to compare feature engineering outputs between original and current
"""
import sys
import pandas as pd
from pathlib import Path

# Test data path
test_csv = Path(__file__).resolve().parent / "demo_data" / "clean_portfolio.csv"
target_col = "default"

print(f"Loading {test_csv}...")
df = pd.read_csv(test_csv)
print(f"Raw shape: {df.shape}\n")

print("="*80)
print("TESTING ORIGINAL IMPLEMENTATION")
print("="*80)

# Import original modules
sys.path.insert(0, "Credit-Risk-Poc-main")
from utils import detect_column_types, detect_task_type
from preprocessing import prepare_data as prepare_data_orig
from train import split_data
from feature_engineering import analyze_for_feature_engineering, apply_feature_engineering

# Original flow
col_types_orig = detect_column_types(df)
task_type_orig = detect_task_type(df[target_col])
print(f"Task type: {task_type_orig}")

X_orig, y_orig, preproc_orig, prep_report_orig, feature_names_orig = prepare_data_orig(df, col_types_orig, target_col)
print(f"After prepare_data: X={X_orig.shape}, y={y_orig.shape}")

X_train_orig, X_val_orig, X_test_orig, y_train_orig, y_val_orig, y_test_orig = split_data(X_orig, y_orig)
print(f"After split: X_train={X_train_orig.shape}")

plan_orig = analyze_for_feature_engineering(X_train_orig, y_train_orig, col_types_orig, task_type_orig)
print(f"\nFE Plan Analysis:")
print(f"  log_transform_cols: {len(plan_orig.get('log_transform_cols', []))}")
print(f"  interaction_pairs: {len(plan_orig.get('interaction_pairs', []))}")
print(f"  binning_cols: {len(plan_orig.get('binning_cols', []))}")
print(f"  freq_encoding_cols: {len(plan_orig.get('freq_encoding_cols', []))}")
print(f"  woe_cols: {len(plan_orig.get('woe_cols', []))}")
print(f"  drop_high_corr_pairs: {len(plan_orig.get('drop_high_corr_pairs', []))}")
print(f"  low_variance_cols: {len(plan_orig.get('low_variance_cols', []))}")
print(f"  low_iv_cols: {len(plan_orig.get('low_iv_cols', []))}")

try:
    X_train_fe_orig, summary_orig = apply_feature_engineering(X_train_orig, plan_orig)
except TypeError:
    # Try with keyword argument
    X_train_fe_orig, summary_orig = apply_feature_engineering(X_train_orig, plan=plan_orig)
print(f"\nAfter FE:")
print(f"  Shape: {X_train_fe_orig.shape}")
print(f"  Features added: {summary_orig.get('features_added', 0)}")
print(f"  Features removed: {summary_orig.get('features_removed', 0)}")

orig_cols = sorted(X_train_fe_orig.columns.tolist())
print(f"\nOriginal engineered features ({len(orig_cols)}):")
for col in orig_cols:
    print(f"  {col}")

# Store for comparison
orig_feature_count = len(orig_cols)
orig_col_set = set(orig_cols)

print("\n" + "="*80)
print("TESTING CURRENT IMPLEMENTATION")
print("="*80)

# Reset sys.path and import current modules
sys.path = [p for p in sys.path if "Credit-Risk-Poc-main" not in p]
sys.path.insert(0, ".")

# Force reload of modules
for mod in list(sys.modules.keys()):
    if mod.startswith(('utils', 'preprocessing', 'train', 'feature_engineering')):
        del sys.modules[mod]

from utils import detect_column_types as detect_column_types_curr
from utils import detect_task_type as detect_task_type_curr
from preprocessing import prepare_data as prepare_data_curr
from train import split_data as split_data_curr
from feature_engineering import analyze_for_feature_engineering as analyze_fe_curr
from feature_engineering import apply_feature_engineering as apply_fe_curr

# Current flow
col_types_curr = detect_column_types_curr(df)
task_type_curr = detect_task_type_curr(df[target_col])
print(f"Task type: {task_type_curr}")

X_curr, y_curr, preproc_curr, prep_report_curr, feature_names_curr = prepare_data_curr(df, col_types_curr, target_col)
print(f"After prepare_data: X={X_curr.shape}, y={y_curr.shape}")

X_train_curr, X_val_curr, X_test_curr, y_train_curr, y_val_curr, y_test_curr = split_data_curr(X_curr, y_curr)
print(f"After split: X_train={X_train_curr.shape}")

plan_curr = analyze_fe_curr(X_train_curr, y_train_curr, col_types_curr, task_type_curr)
print(f"\nFE Plan Analysis:")
print(f"  log_transform_cols: {len(plan_curr.get('log_transform_cols', []))}")
print(f"  interaction_pairs: {len(plan_curr.get('interaction_pairs', []))}")
print(f"  binning_cols: {len(plan_curr.get('binning_cols', []))}")
print(f"  freq_encoding_cols: {len(plan_curr.get('freq_encoding_cols', []))}")
print(f"  woe_cols: {len(plan_curr.get('woe_cols', []))}")
print(f"  drop_high_corr_pairs: {len(plan_curr.get('drop_high_corr_pairs', []))}")
print(f"  low_variance_cols: {len(plan_curr.get('low_variance_cols', []))}")
print(f"  low_iv_cols: {len(plan_curr.get('low_iv_cols', []))}")

X_train_fe_curr, summary_curr = apply_fe_curr(X_train_curr, plan_curr)
print(f"\nAfter FE:")
print(f"  Shape: {X_train_fe_curr.shape}")
print(f"  Features added: {summary_curr.get('features_added', 0)}")
print(f"  Features removed: {summary_curr.get('features_removed', 0)}")

curr_cols = sorted(X_train_fe_curr.columns.tolist())
print(f"\nCurrent engineered features ({len(curr_cols)}):")
for col in curr_cols:
    print(f"  {col}")

curr_feature_count = len(curr_cols)
curr_col_set = set(curr_cols)

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nOriginal feature count: {orig_feature_count}")
print(f"Current feature count: {curr_feature_count}")
print(f"Difference: {curr_feature_count - orig_feature_count}")

only_in_curr = curr_col_set - orig_col_set
only_in_orig = orig_col_set - curr_col_set

if only_in_curr:
    print(f"\nFeatures ONLY in current ({len(only_in_curr)}):")
    for col in sorted(only_in_curr):
        print(f"  + {col}")

if only_in_orig:
    print(f"\nFeatures ONLY in original ({len(only_in_orig)}):")
    for col in sorted(only_in_orig):
        print(f"  - {col}")

# Compare plan details
print("\n--- FEATURE REMOVAL PLANS ---")

def compare_lists(name, orig_val, curr_val):
    if orig_val != curr_val:
        print(f"\n{name}:")
        print(f"  Original ({len(orig_val)}): {orig_val}")
        print(f"  Current ({len(curr_val)}): {curr_val}")
    else:
        print(f"\n{name}: IDENTICAL ({len(orig_val)} items)")

compare_lists("drop_high_corr_pairs", plan_orig.get('drop_high_corr_pairs', []), plan_curr.get('drop_high_corr_pairs', []))
compare_lists("low_variance_cols", plan_orig.get('low_variance_cols', []), plan_curr.get('low_variance_cols', []))
compare_lists("low_iv_cols", plan_orig.get('low_iv_cols', [])[:10], plan_curr.get('low_iv_cols', [])[:10])  # Show first 10
