import importlib.util
import sys
import os
import json
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
CRP_ROOT = os.path.abspath(os.path.join(ROOT, '..', 'Credit-Risk-Poc-main'))
CURRENT_ROOT = ROOT

def load_module_from_path(name, path):
	spec = importlib.util.spec_from_file_location(name, path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod

# Load original modules under unique names
orig_model_selector = load_module_from_path('orig_model_selector', os.path.join(CRP_ROOT, 'model_selector.py'))
orig_preproc = load_module_from_path('orig_preproc', os.path.join(CRP_ROOT, 'preprocessing.py'))
orig_fe = load_module_from_path('orig_fe', os.path.join(CRP_ROOT, 'feature_engineering.py'))
orig_utils = load_module_from_path('orig_utils', os.path.join(CRP_ROOT, 'utils.py'))

# Load current modules
cur_model_selector = load_module_from_path('cur_model_selector', os.path.join(CURRENT_ROOT, 'model_selector.py'))
cur_preproc = load_module_from_path('cur_preproc', os.path.join(CURRENT_ROOT, 'preprocessing.py'))
cur_fe = load_module_from_path('cur_fe', os.path.join(CURRENT_ROOT, 'feature_engineering.py'))
cur_utils = load_module_from_path('cur_utils', os.path.join(CURRENT_ROOT, 'utils.py'))

csv_path = os.path.join(os.path.dirname(__file__), 'demo_data', 'clean_portfolio.csv')
df = pd.read_csv(csv_path)

target_col = 'default'

# Original flow
col_types = orig_utils.detect_column_types(df)
X, y, preproc, prep_report, feature_names = orig_preproc.prepare_data(df, col_types, target_col)
from train import split_data as split_data_func
X_train, X_val, X_test, y_train, y_val, y_test = split_data_func(X, y, test_size=0.15, val_size=0.15, task_type=orig_utils.detect_task_type(df[target_col]), random_state=42)
plan = orig_fe.analyze_for_feature_engineering(X_train, y_train, col_types, orig_utils.detect_task_type(df[target_col]))
X_train_engineered, fe_summary = orig_fe.apply_feature_engineering(X_train, plan)
_raw_orig = orig_model_selector.recommend_models(n_samples=X_train_engineered.shape[0], n_features=X_train_engineered.shape[1], class_imbalance_ratio=float(y_train.value_counts().max()/y_train.value_counts().min()) if y_train.nunique()>1 else 1.0, task_type=orig_utils.detect_task_type(df[target_col]))
orig_recs = []
for r in _raw_orig:
	# strip non-serializable items (like class references)
	rec = {k: v for k, v in r.items() if not isinstance(v, type)}
	orig_recs.append(rec)

# Current flow
col_types_c = cur_utils.detect_column_types(df)
Xc, yc, preproc_c, prep_report_c, feature_names_c = cur_preproc.prepare_data(df, col_types_c, target_col)
X_train_c, X_val_c, X_test_c, y_train_c, y_val_c, y_test_c = split_data_func(Xc, yc, test_size=0.15, val_size=0.15, task_type=cur_utils.detect_task_type(df[target_col]), random_state=42)
plan_c = cur_fe.analyze_for_feature_engineering(X_train_c, y_train_c, col_types_c, cur_utils.detect_task_type(df[target_col]))
X_train_engineered_c, fe_summary_c = cur_fe.apply_feature_engineering(X_train_c, plan_c)
_raw_cur = cur_model_selector.recommend_models(n_samples=X_train_engineered_c.shape[0], n_features=X_train_engineered_c.shape[1], class_imbalance_ratio=float(y_train_c.value_counts().max()/y_train_c.value_counts().min()) if y_train_c.nunique()>1 else 1.0, task_type=cur_utils.detect_task_type(df[target_col]))
cur_recs = []
for r in _raw_cur:
	rec = {k: v for k, v in r.items() if not isinstance(v, type)}
	cur_recs.append(rec)

print('Original TRAIN shape:', X_train_engineered.shape)
print('Current TRAIN shape: ', X_train_engineered_c.shape)
print('\nOriginal recommendations:')
print(json.dumps(orig_recs, indent=2))
print('\nCurrent recommendations:')
print(json.dumps(cur_recs, indent=2))
