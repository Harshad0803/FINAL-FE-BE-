"""
Debug script to compare feature engineering between original and current implementations
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add both paths to system path
sys.path.insert(0, str(Path("Credit-Risk-Poc-main")))
sys.path.insert(1, str(Path(".")))

# Import original and current modules
import importlib.util

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load original implementations
orig_feature_eng = load_module("orig_feature_eng", "Credit-Risk-Poc-main/feature_engineering.py")
orig_preprocessing = load_module("orig_preprocessing", "Credit-Risk-Poc-main/preprocessing.py")
orig_train = load_module("orig_train", "Credit-Risk-Poc-main/train.py")
orig_utils = load_module("orig_utils", "Credit-Risk-Poc-main/utils.py")

# Load current implementations
curr_feature_eng = load_module("curr_feature_eng", "feature_engineering.py")
curr_preprocessing = load_module("curr_preprocessing", "preprocessing.py")
curr_train = load_module("curr_train", "train.py")
curr_utils = load_module("curr_utils", "utils.py")

# Load and prepare data
print("Loading demo data...")
df = pd.read_csv(Path(__file__).resolve().parent / "demo_data" / "clean_portfolio.csv")
print(f"Raw shape: {df.shape}")

print("\n" + "="*80)
print("ORIGINAL STREAMLIT PIPELINE")
print("="*80)

# Original pipeline
X_orig, y_orig = orig_preprocessing.prepare_data(df)
print(f"After prepare_data: {X_orig.shape}")

col_types_orig = orig_utils.detect_column_types(X_orig, y_orig)
task_type_orig = orig_utils.detect_task_type(y_orig)
print(f"Task type: {task_type_orig}")
print(f"Column types: {[(k, len(v)) for k, v in col_types_orig.items()]}")

# Split
X_train_orig, X_val_orig, X_test_orig, y_train_orig, y_val_orig, y_test_orig = orig_train.split_data(X_orig, y_orig)
print(f"After split - Train: {X_train_orig.shape}, Val: {X_val_orig.shape}, Test: {X_test_orig.shape}")

# Analyze FE
plan_orig = orig_feature_eng.analyze_for_feature_engineering(X_train_orig, y_train_orig, col_types_orig, task_type_orig)
print(f"\nFE Plan from original:")
print(f"  - log_transform_cols: {len(plan_orig.get('log_transform_cols', []))}")
print(f"  - interaction_pairs: {len(plan_orig.get('interaction_pairs', []))}")
print(f"  - binning_cols: {len(plan_orig.get('binning_cols', []))}")
print(f"  - freq_encoding_cols: {len(plan_orig.get('freq_encoding_cols', []))}")
print(f"  - woe_cols: {len(plan_orig.get('woe_cols', []))}")
print(f"  - drop_high_corr_pairs: {len(plan_orig.get('drop_high_corr_pairs', []))}")
print(f"  - low_variance_cols: {len(plan_orig.get('low_variance_cols', []))}")
print(f"  - low_iv_cols: {len(plan_orig.get('low_iv_cols', []))}")

# Apply FE
X_train_fe_orig, summary_orig = orig_feature_eng.apply_feature_engineering(X_train_orig, plan_orig)
print(f"\nAfter apply_feature_engineering (original):")
print(f"  - Shape: {X_train_fe_orig.shape}")
print(f"  - Features added: {summary_orig.get('features_added', 0)}")
print(f"  - Features removed: {summary_orig.get('features_removed', 0)}")

orig_cols = set(X_train_fe_orig.columns)
print(f"\nOriginal final columns ({len(orig_cols)}):")
for col in sorted(orig_cols):
    print(f"  - {col}")

print("\n" + "="*80)
print("CURRENT FASTAPI PIPELINE")
print("="*80)

# Current pipeline
X_curr, y_curr = curr_preprocessing.prepare_data(df)
print(f"After prepare_data: {X_curr.shape}")

col_types_curr = curr_utils.detect_column_types(X_curr, y_curr)
task_type_curr = curr_utils.detect_task_type(y_curr)
print(f"Task type: {task_type_curr}")
print(f"Column types: {[(k, len(v)) for k, v in col_types_curr.items()]}")

# Split
X_train_curr, X_val_curr, X_test_curr, y_train_curr, y_val_curr, y_test_curr = curr_train.split_data(X_curr, y_curr)
print(f"After split - Train: {X_train_curr.shape}, Val: {X_val_curr.shape}, Test: {X_test_curr.shape}")

# Analyze FE
plan_curr = curr_feature_eng.analyze_for_feature_engineering(X_train_curr, y_train_curr, col_types_curr, task_type_curr)
print(f"\nFE Plan from current:")
print(f"  - log_transform_cols: {len(plan_curr.get('log_transform_cols', []))}")
print(f"  - interaction_pairs: {len(plan_curr.get('interaction_pairs', []))}")
print(f"  - binning_cols: {len(plan_curr.get('binning_cols', []))}")
print(f"  - freq_encoding_cols: {len(plan_curr.get('freq_encoding_cols', []))}")
print(f"  - woe_cols: {len(plan_curr.get('woe_cols', []))}")
print(f"  - drop_high_corr_pairs: {len(plan_curr.get('drop_high_corr_pairs', []))}")
print(f"  - low_variance_cols: {len(plan_curr.get('low_variance_cols', []))}")
print(f"  - low_iv_cols: {len(plan_curr.get('low_iv_cols', []))}")

# Apply FE
X_train_fe_curr, summary_curr = curr_feature_eng.apply_feature_engineering(X_train_curr, plan_curr)
print(f"\nAfter apply_feature_engineering (current):")
print(f"  - Shape: {X_train_fe_curr.shape}")
print(f"  - Features added: {summary_curr.get('features_added', 0)}")
print(f"  - Features removed: {summary_curr.get('features_removed', 0)}")

curr_cols = set(X_train_fe_curr.columns)
print(f"\nCurrent final columns ({len(curr_cols)}):")
for col in sorted(curr_cols):
    print(f"  - {col}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

only_in_curr = curr_cols - orig_cols
only_in_orig = orig_cols - curr_cols

print(f"\nOriginal feature count: {len(orig_cols)}")
print(f"Current feature count: {len(curr_cols)}")
print(f"Difference: {len(curr_cols) - len(orig_cols)}")

if only_in_curr:
    print(f"\nColumns ONLY in current ({len(only_in_curr)}):")
    for col in sorted(only_in_curr):
        print(f"  - {col}")

if only_in_orig:
    print(f"\nColumns ONLY in original ({len(only_in_orig)}):")
    for col in sorted(only_in_orig):
        print(f"  - {col}")

# Print removal details
print(f"\n--- DROP LISTS COMPARISON ---")
print(f"Original drop_high_corr_pairs: {plan_orig.get('drop_high_corr_pairs', [])}")
print(f"Current drop_high_corr_pairs: {plan_curr.get('drop_high_corr_pairs', [])}")

print(f"\nOriginal low_variance_cols: {plan_orig.get('low_variance_cols', [])}")
print(f"Current low_variance_cols: {plan_curr.get('low_variance_cols', [])}")

print(f"\nOriginal low_iv_cols: {plan_orig.get('low_iv_cols', [])}")
print(f"Current low_iv_cols: {plan_curr.get('low_iv_cols', [])}")
