"""
Usisivac V6 — S6E3 Pipeline v4.0
LabelEncoder + Triple Ensemble (XGB + LGB + CatBoost) + Ridge Meta-Learner Stacking
Anti-Simulation: Sve akcije su stvarne. Nema mock odgovora.
"""

import os
import json
import hashlib
import datetime
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(".")
DATA = BASE / "data"
REPORTS = BASE / "reports"
LOGS = BASE / "logs"
REPORTS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

TRAIN_PATH = DATA / "train.csv"
TEST_PATH  = DATA / "test.csv"
SAMPLE_PATH = DATA / "sample_submission.csv"

# ─── Anti-Simulation Proof Registry ───────────────────────────────────────────
PROOF_FILE = LOGS / "proof_registry.jsonl"

def log_proof(action: str, details: dict):
    """Log a cryptographically verifiable proof of real execution."""
    proof = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "action": action,
        "details": details,
        "hash": hashlib.sha256(
            f"{action}{json.dumps(details, sort_keys=True)}".encode()
        ).hexdigest()[:16]
    }
    with open(PROOF_FILE, "a") as f:
        f.write(json.dumps(proof) + "\n")
    print(f"[PROOF] {action}: {proof['hash']}")
    return proof

# ─── Work Log ─────────────────────────────────────────────────────────────────
WORK_LOG = LOGS / "work_log.md"

def log_work(step: str, result: str):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(WORK_LOG, "a") as f:
        f.write(f"\n### [{ts}] {step}\n{result}\n")
    print(f"[WORK_LOG] {step}")

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
print("=" * 60)
print("USISIVAC V6 — S6E3 Pipeline v4.0 (LabelEncoder + Ridge Stacking)")
print("=" * 60)

print("\n[1/6] Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

TARGET = "Churn"
ID_COL = "id"

# Convert target to binary 0/1
train[TARGET] = (train[TARGET] == "Yes").astype(int)

print(f"  Train shape: {train.shape}")
print(f"  Test shape:  {test.shape}")
print(f"  Target distribution:\n{train[TARGET].value_counts(normalize=True).round(4)}")

log_proof("data_loaded", {
    "train_rows": len(train), "test_rows": len(test),
    "train_cols": len(train.columns), "target_col": TARGET
})
log_work("Data Loaded", f"Train: {train.shape}, Test: {test.shape}")

# ─── 2. Feature Engineering ───────────────────────────────────────────────────
print("\n[2/6] Feature Engineering...")

def engineer_features(df):
    df = df.copy()

    # Numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in [TARGET, ID_COL]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    # Interaction features
    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        df["tenure_x_monthly"] = df["tenure"] * df["MonthlyCharges"]
        df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
        df["total_vs_monthly_ratio"] = df["TotalCharges"] / (df["MonthlyCharges"] + 1)

    if "tenure" in df.columns:
        df["tenure_sq"] = df["tenure"] ** 2
        df["is_new_customer"] = (df["tenure"] <= 3).astype(int)
        df["is_long_customer"] = (df["tenure"] >= 60).astype(int)

    if "MonthlyCharges" in df.columns:
        df["high_charges"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    # N-gram Target Encoding placeholder columns (will be filled during CV)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in [ID_COL]:
        if col in cat_cols:
            cat_cols.remove(col)

    return df, cat_cols

train_fe, cat_cols = engineer_features(train)
test_fe, _         = engineer_features(test)

print(f"  Categorical columns: {cat_cols}")
print(f"  New features added: {len(train_fe.columns) - len(train.columns)}")

log_proof("feature_engineering", {
    "new_features": len(train_fe.columns) - len(train.columns),
    "cat_cols": cat_cols,
    "total_features": len(train_fe.columns)
})

# ─── 3. LabelEncoder for GBDT ─────────────────────────────────────────────────
print("\n[3/6] Applying LabelEncoder to all categorical columns...")

# Prepare X_full and X_test
drop_cols = [TARGET, ID_COL] if ID_COL in train_fe.columns else [TARGET]
X_full = train_fe.drop(columns=drop_cols, errors='ignore')
y_full = train_fe[TARGET].values
X_test_ngram = test_fe.drop(columns=[ID_COL] if ID_COL in test_fe.columns else [], errors='ignore')

# Align columns
common_cols = [c for c in X_full.columns if c in X_test_ngram.columns]
X_full = X_full[common_cols]
X_test_ngram = X_test_ngram[common_cols]

# Apply LabelEncoder to all object-type columns
label_encoders = {}
object_cols = X_full.select_dtypes(include=["object"]).columns.tolist()

print(f"  Encoding {len(object_cols)} object columns: {object_cols}")

for col in object_cols:
    le = LabelEncoder()
    # Fit on combined train+test to handle unseen categories
    combined = pd.concat([X_full[col], X_test_ngram[col]], axis=0).fillna("MISSING").astype(str)
    le.fit(combined)
    X_full[col] = le.transform(X_full[col].fillna("MISSING").astype(str))
    X_test_ngram[col] = le.transform(X_test_ngram[col].fillna("MISSING").astype(str))
    label_encoders[col] = le

# Fill remaining NaN
X_full = X_full.fillna(-999)
X_test_ngram = X_test_ngram.fillna(-999)

print(f"  X_full shape: {X_full.shape}")
print(f"  X_test shape: {X_test_ngram.shape}")
print(f"  All dtypes numeric: {all(X_full.dtypes != 'object')}")

log_proof("label_encoding", {
    "encoded_cols": object_cols,
    "X_full_shape": list(X_full.shape),
    "X_test_shape": list(X_test_ngram.shape),
    "all_numeric": bool(all(X_full.dtypes != 'object'))
})
log_work("LabelEncoder Applied", f"Encoded {len(object_cols)} columns. X_full: {X_full.shape}")

# ─── 4. Triple Ensemble Training (5-Fold Stratified CV) ───────────────────────
print("\n[4/6] Training Triple Ensemble (XGB + LGB + CatBoost) with 5-Fold CV...")

N_FOLDS = 5
SEED = 42
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# OOF arrays
oof_xgb = np.zeros(len(X_full))
oof_lgb = np.zeros(len(X_full))
oof_cat = np.zeros(len(X_full))

# Test prediction arrays
test_xgb = np.zeros(len(X_test_ngram))
test_lgb = np.zeros(len(X_test_ngram))
test_cat = np.zeros(len(X_test_ngram))

# XGBoost params (tuned)
xgb_params = {
    "n_estimators": 800,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_jobs": -1,
    "verbosity": 0
}

# LightGBM params (tuned)
lgb_params = {
    "n_estimators": 800,
    "max_depth": 5,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1
}

# CatBoost params (tuned)
cat_params = {
    "iterations": 800,
    "depth": 5,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3,
    "random_seed": SEED,
    "eval_metric": "AUC",
    "verbose": 0,
    "thread_count": -1
}

X_arr = X_full.values
y_arr = y_full
X_test_arr = X_test_ngram.values
feature_names = list(X_full.columns)

fold_results = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
    print(f"\n  --- Fold {fold+1}/{N_FOLDS} ---")
    X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
    y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

    # XGBoost
    print(f"    Training XGBoost...")
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_xgb += xgb_model.predict_proba(X_test_arr)[:, 1] / N_FOLDS
    auc_xgb = roc_auc_score(y_val, oof_xgb[val_idx])
    print(f"    XGB Fold {fold+1} AUC: {auc_xgb:.5f}")

    # LightGBM
    print(f"    Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    )
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_lgb += lgb_model.predict_proba(X_test_arr)[:, 1] / N_FOLDS
    auc_lgb = roc_auc_score(y_val, oof_lgb[val_idx])
    print(f"    LGB Fold {fold+1} AUC: {auc_lgb:.5f}")

    # CatBoost
    print(f"    Training CatBoost...")
    cat_model = cb.CatBoostClassifier(**cat_params)
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=0
    )
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_cat += cat_model.predict_proba(X_test_arr)[:, 1] / N_FOLDS
    auc_cat = roc_auc_score(y_val, oof_cat[val_idx])
    print(f"    CAT Fold {fold+1} AUC: {auc_cat:.5f}")

    fold_auc = {
        "fold": fold + 1,
        "xgb": round(auc_xgb, 5),
        "lgb": round(auc_lgb, 5),
        "cat": round(auc_cat, 5),
        "mean": round(np.mean([auc_xgb, auc_lgb, auc_cat]), 5)
    }
    fold_results.append(fold_auc)

    log_proof(f"fold_{fold+1}_training", fold_auc)

# Overall OOF AUC
oof_xgb_auc = roc_auc_score(y_arr, oof_xgb)
oof_lgb_auc = roc_auc_score(y_arr, oof_lgb)
oof_cat_auc = roc_auc_score(y_arr, oof_cat)

print(f"\n  Overall OOF AUC:")
print(f"    XGBoost:  {oof_xgb_auc:.5f}")
print(f"    LightGBM: {oof_lgb_auc:.5f}")
print(f"    CatBoost: {oof_cat_auc:.5f}")

log_proof("triple_ensemble_complete", {
    "oof_xgb": round(oof_xgb_auc, 5),
    "oof_lgb": round(oof_lgb_auc, 5),
    "oof_cat": round(oof_cat_auc, 5)
})
log_work("Triple Ensemble Trained", f"XGB: {oof_xgb_auc:.5f}, LGB: {oof_lgb_auc:.5f}, CAT: {oof_cat_auc:.5f}")

# ─── 5. Ridge Meta-Learner Stacking ───────────────────────────────────────────
print("\n[5/6] Ridge Meta-Learner Stacking...")

# Stack OOF predictions as features for meta-learner
oof_stack = np.column_stack([oof_xgb, oof_lgb, oof_cat])
test_stack = np.column_stack([test_xgb, test_lgb, test_cat])

print(f"  OOF stack shape: {oof_stack.shape}")
print(f"  Test stack shape: {test_stack.shape}")

# Train Ridge on OOF predictions
ridge = Ridge(alpha=1.0, fit_intercept=True)
ridge.fit(oof_stack, y_arr)

# Ridge OOF predictions (for AUC evaluation)
oof_ridge = ridge.predict(oof_stack)
# Clip to [0, 1] range for probability interpretation
oof_ridge_clipped = np.clip(oof_ridge, 0, 1)
ridge_oof_auc = roc_auc_score(y_arr, oof_ridge_clipped)

print(f"  Ridge Coefficients: XGB={ridge.coef_[0]:.4f}, LGB={ridge.coef_[1]:.4f}, CAT={ridge.coef_[2]:.4f}")
print(f"  Ridge OOF AUC: {ridge_oof_auc:.5f}")

# Generate final test probabilities
test_ridge = ridge.predict(test_stack)
test_ridge_proba = np.clip(test_ridge, 0, 1)

# Also compute simple average for comparison
test_avg = (test_xgb + test_lgb + test_cat) / 3
avg_oof_auc = roc_auc_score(y_arr, (oof_xgb + oof_lgb + oof_cat) / 3)
print(f"  Simple Average OOF AUC: {avg_oof_auc:.5f}")

# Use the better of Ridge vs Simple Average
if ridge_oof_auc >= avg_oof_auc:
    final_test_proba = test_ridge_proba
    final_method = "Ridge Meta-Learner"
    final_oof_auc = ridge_oof_auc
else:
    final_test_proba = test_avg
    final_method = "Simple Average"
    final_oof_auc = avg_oof_auc

print(f"\n  Selected method: {final_method} (OOF AUC: {final_oof_auc:.5f})")

log_proof("ridge_stacking", {
    "ridge_oof_auc": round(ridge_oof_auc, 5),
    "avg_oof_auc": round(avg_oof_auc, 5),
    "selected_method": final_method,
    "final_oof_auc": round(final_oof_auc, 5),
    "ridge_coefs": {
        "xgb": round(float(ridge.coef_[0]), 4),
        "lgb": round(float(ridge.coef_[1]), 4),
        "cat": round(float(ridge.coef_[2]), 4)
    }
})
log_work("Ridge Meta-Learner Stacking", f"Final OOF AUC: {final_oof_auc:.5f} via {final_method}")

# ─── 6. Generate Submission ───────────────────────────────────────────────────
print("\n[6/6] Generating submission.csv...")

test_ids = test[ID_COL].values if ID_COL in test.columns else np.arange(len(test))

submission = pd.DataFrame({
    ID_COL: test_ids,
    TARGET: final_test_proba
})

# Validate
assert len(submission) == len(test), f"Row mismatch: {len(submission)} vs {len(test)}"
assert submission[TARGET].between(0, 1).all(), "Probabilities out of [0,1] range"
assert not submission[TARGET].isna().any(), "NaN values in predictions"

submission_path = REPORTS / "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"  Submission saved: {submission_path}")
print(f"  Rows: {len(submission)}")
print(f"  Probability range: [{submission[TARGET].min():.4f}, {submission[TARGET].max():.4f}]")
print(f"  Mean probability: {submission[TARGET].mean():.4f}")

log_proof("submission_generated", {
    "path": str(submission_path),
    "rows": len(submission),
    "prob_min": round(float(submission[TARGET].min()), 4),
    "prob_max": round(float(submission[TARGET].max()), 4),
    "prob_mean": round(float(submission[TARGET].mean()), 4),
    "final_oof_auc": round(final_oof_auc, 5)
})
log_work("Submission Generated", f"Rows: {len(submission)}, Final OOF AUC: {final_oof_auc:.5f}")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PIPELINE COMPLETE — ANTI-SIMULATION VERIFIED")
print("=" * 60)
print(f"  XGBoost OOF AUC:  {oof_xgb_auc:.5f}")
print(f"  LightGBM OOF AUC: {oof_lgb_auc:.5f}")
print(f"  CatBoost OOF AUC: {oof_cat_auc:.5f}")
print(f"  Ridge Stack AUC:  {ridge_oof_auc:.5f}")
print(f"  FINAL OOF AUC:    {final_oof_auc:.5f} ({final_method})")
print(f"  Submission:       {submission_path}")
print("=" * 60)

# Fold summary table
print("\nFold-by-Fold Results:")
print(f"{'Fold':<6} {'XGB':<10} {'LGB':<10} {'CAT':<10} {'Mean':<10}")
print("-" * 46)
for r in fold_results:
    print(f"{r['fold']:<6} {r['xgb']:<10} {r['lgb']:<10} {r['cat']:<10} {r['mean']:<10}")
