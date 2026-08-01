"""
Direct comparison of engineered features between original and current
"""
import sys
from pathlib import Path
import pandas as pd

test_csv = Path(__file__).resolve().parent / "demo_data" / "clean_portfolio.csv"
target_col = "default"

print(f"Loading {test_csv}...")
df = pd.read_csv(test_csv)
print(f"Raw shape: {df.shape}\n")

print("="*80)
print("ORIGINAL PIPELINE")
print("="*80)

sys.path.insert(0, "../Credit-Risk-Poc-main")
from utils import detect_column_types, detect_task_type
from preprocessing import prepare_data as prepare_data_orig
from train import split_data
from feature_engineering import analyze_for_feature_engineering, apply_feature_engineering

col_types_orig = detect_column_types(df)
task_type_orig = detect_task_type(df[target_col])

X_orig, y_orig, _, _, _ = prepare_data_orig(df, col_types_orig, target_col)
X_train_orig, _, _, y_train_orig, _, _ = split_data(X_orig, y_orig)

plan_orig = analyze_for_feature_engineering(X_train_orig, y_train_orig, col_types_orig, task_type_orig)
X_train_fe_orig, summary_orig = apply_feature_engineering(X_train_orig, plan_orig)

orig_cols = sorted(X_train_fe_orig.columns.tolist())
print(f"\nOriginal engineered features ({len(orig_cols)}):")
for i, col in enumerate(orig_cols, 1):
    print(f"  {i:2d}. {col}")

print(f"\nRemoval stats from original plan:")
print(f"  drop_high_corr_pairs: {len(plan_orig.get('drop_high_corr_pairs', []))}")
print(f"  low_variance_cols: {len(plan_orig.get('low_variance_cols', []))}")
print(f"  low_iv_cols: {len(plan_orig.get('low_iv_cols', []))}")

print("\n" + "="*80)
print("CURRENT PIPELINE")
print("="*80)

# Reset and reimport
sys.path = [p for p in sys.path if "../Credit-Risk-Poc-main" not in p and "Credit-Risk-Poc-main" not in p]
sys.path.insert(0, ".")

for mod in list(sys.modules.keys()):
    if any(x in mod for x in ['utils', 'preprocessing', 'train', 'feature_engineering']):
        del sys.modules[mod]

from utils import detect_column_types as detect_column_types_curr
from utils import detect_task_type as detect_task_type_curr
from preprocessing import prepare_data as prepare_data_curr
from train import split_data as split_data_curr
from feature_engineering import analyze_for_feature_engineering as analyze_fe_curr
from feature_engineering import apply_feature_engineering as apply_fe_curr

col_types_curr = detect_column_types_curr(df)
task_type_curr = detect_task_type_curr(df[target_col])

X_curr, y_curr, _, _, _ = prepare_data_curr(df, col_types_curr, target_col)
X_train_curr, _, _, y_train_curr, _, _ = split_data_curr(X_curr, y_curr)

plan_curr = analyze_fe_curr(X_train_curr, y_train_curr, col_types_curr, task_type_curr)
X_train_fe_curr, summary_curr = apply_fe_curr(X_train_curr, plan_curr)

curr_cols = sorted(X_train_fe_curr.columns.tolist())
print(f"\nCurrent engineered features ({len(curr_cols)}):")
for i, col in enumerate(curr_cols, 1):
    print(f"  {i:2d}. {col}")

print(f"\nRemoval stats from current plan:")
print(f"  drop_high_corr_pairs: {len(plan_curr.get('drop_high_corr_pairs', []))}")
print(f"  low_variance_cols: {len(plan_curr.get('low_variance_cols', []))}")
print(f"  low_iv_cols: {len(plan_curr.get('low_iv_cols', []))}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

orig_set = set(orig_cols)
curr_set = set(curr_cols)

only_in_curr = curr_set - orig_set
only_in_orig = orig_set - curr_set

print(f"\nOriginal feature count: {len(orig_cols)}")
print(f"Current feature count: {len(curr_cols)}")
print(f"Difference: {len(curr_cols) - len(orig_cols)}")

if only_in_curr:
    print(f"\nFeatures ONLY in current ({len(only_in_curr)}):")
    for col in sorted(only_in_curr):
        print(f"  + {col}")

if only_in_orig:
    print(f"\nFeatures ONLY in original ({len(only_in_orig)}):")
    for col in sorted(only_in_orig):
        print(f"  - {col}")
